"""Torch-free worker proxy and task wrapper for batched evaluation/collection."""

from contextvars import ContextVar

from civ5rl.core.observation import ordered_actions
from civ5rl.core.selection import ActionSelection

_client = ContextVar("evaluation_client", default=None)
_proxies = ContextVar("evaluation_proxies", default=None)


class ProxyAgent(ActionSelection):
    """Player-only Agent RPC. No model or CUDA state is allocated in a worker."""

    def __init__(
        self, connection, mode, temperature=1.0, collect_selection_metrics=False
    ):
        self.connection, self.mode = connection, mode
        self.temperature, self.collect_selection_metrics = (
            temperature,
            collect_selection_metrics,
        )
        self.events = []
        self.shared = None
        proxies = _proxies.get()
        if proxies is not None:
            proxies.append(self)

    def call(self, command, *args):
        self.connection.send((command, args))
        if not self.connection.poll(300):
            raise TimeoutError("Evaluation inference did not reply within 300 seconds")
        status, value = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"Evaluation inference failed: {value}")
        return value

    def reset(self, context):
        from civ5rl.engine.runtime.shared import EncodedSlots

        self.close()
        info = self.call(
            "reset", self.mode, self.temperature, self.collect_selection_metrics
        )
        self.shared = EncodedSlots(descriptor=info["shared"])
        self.requires_attack_preview = info["attack_preview"]
        self.prefix_skip = info.get("prefix_skip", 0)
        self.prefix_restored = self.prefix_skip > 0
        self.capture_prefixes = info.get("capture_prefixes", False)
        self.reset_public_state(context, info["variant"])

    def act(self, observation, actions):
        if self.prefix_skip:
            raise ValueError("Restored public prefix is incomplete")
        self.prefix_restored = False
        actions = ordered_actions(actions)
        encoded = self.encoder.encode(observation, actions)
        result = self.call("act", self.shared.write(0, encoded))
        if "index" in result:
            action = self.action_from_index(result["index"], observation, actions)
        else:
            import torch

            action = self.action_from_logits(
                torch.from_numpy(result["logits"])[None], observation, actions
            )
        self.events.extend(result.get("events", ()))
        return action

    def set_selection_mode(self, mode):
        """Change selection only; preserve encoder, GRU and private RNG."""
        self.call("selection_mode", mode)
        self.mode = mode

    def observe_step(self, observation, actions, forced_action):
        actions = ordered_actions(actions)
        if forced_action not in actions:
            raise ValueError("Forced action is not legal")
        encoded = self.encoder.encode(observation, actions)
        if getattr(self, "prefix_restored", False) and not self.prefix_skip:
            raise ValueError("Restored prefix exhausted before branch boundary")
        if self.prefix_skip:
            self.prefix_skip -= 1
        else:
            self.call("observe", self.shared.write(0, encoded))
        self.encoder.remember_action(forced_action, observation)

    def finish_source(self):
        if self.capture_prefixes:
            self.call("save_prefixes")

    def drain_diagnostics(self):
        events, self.events = self.events, []
        return events

    def close(self):
        if self.shared is not None:
            self.shared.close()
            self.shared = None


def evaluation_agent(
    checkpoint, mode, *, temperature=1.0, collect_selection_metrics=False
):
    connection = _client.get()
    if connection is not None:
        return ProxyAgent(connection, mode, temperature, collect_selection_metrics)
    from civ5rl.core.agent import RLAgent

    return RLAgent.from_checkpoint(
        checkpoint,
        "cpu",
        mode,
        temperature=temperature,
        collect_selection_metrics=collect_selection_metrics,
    )


def _remote_task(function, task, connection):
    token = _client.set(connection)
    agents = []
    proxies_token = _proxies.set(agents)
    try:
        return function(task)
    finally:
        for agent in agents:
            agent.close()
        _proxies.reset(proxies_token)
        _client.reset(token)
        try:
            connection.send(("close", ()))
        except (OSError, EOFError):
            pass
        connection.close()
