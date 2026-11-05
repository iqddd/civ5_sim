"""Bounded acceptance checks for silent CP/PPO experiment corruption."""

from copy import deepcopy
from dataclasses import asdict
from time import perf_counter

import numpy as np
import torch

from civ5rl.core.env import DecisionEnv
from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.ppo_fast import BatchedBackend
from civ5rl.experiments.ppo_after_imitation import create_policy, optimizer_for
from civ5rl.protocols.imitation import load_episode, search_config
from civ5rl.provenance.common import ROOT, configure, write_json
from civ5sim import MatchConfig
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest
from civ5sim.greedy_agent import GreedyAgent
from civ5sim.match import run_match


def public_episodes(policy, device):
    """Existing public trajectories as numerical fixtures, never research rollouts."""
    episodes = []
    for i in (0, 1, 2):
        source = load_episode(
            ROOT
            / f"artifacts/rl/cp-imitation/data/training/match-{i:06}/demonstration.json.gz"
        )
        encoder = ObservationEncoder(source["context"], "C")
        encoded = []
        actions = []
        observations = []
        chosen = []
        for row in source["decisions"][:20]:
            from civ5sim import Action

            legal = tuple(Action(**a) for a in row["actions"])
            encoded.append(encoder.encode(row["observation"], legal))
            actions.append(legal)
            observations.append(row["observation"])
            chosen.append(row["chosen"])
            encoder.remember_action(legal[row["chosen"]], row["observation"])
        with torch.no_grad():
            if policy.spatial_cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            logits, values, _ = policy.forward_sequences(
                batch_inputs(encoded, device), [len(encoded)]
            )
            probs = logits.log_softmax(-1).cpu().numpy()
            values = values.cpu().numpy()
        episodes.append(
            {
                "encoded": encoded,
                "encoded_variant": "C",
                "observations": observations,
                "actions": actions,
                "chosen": chosen,
                "old_logp": [float(p[k]) for p, k in zip(probs, chosen)],
                "old_log_probs": [p[: len(a)].copy() for p, a in zip(probs, actions)],
                "old_value": values.tolist(),
                "result": {"reward": int(i == 0)},
            }
        )
    return episodes


def warmup(policy, device):
    episodes = public_episodes(policy, device)
    for ep in episodes:
        with torch.no_grad():
            if policy.spatial_cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            batch = batch_inputs(ep["encoded"], device)
            policy.forward_sequences(batch, [len(ep["chosen"])])
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def collector_smoke(config, plan, out, device="cpu"):
    """One CPU wave confirms the real Windows file-entrypoint worker path."""
    from civ5rl.engine.collector import Collector
    from civ5rl.protocols.ppo_after_imitation import TrainingFactory

    configure(17, device)
    policy = create_policy(config, device)
    # Freeze the exact research configuration; stop the check after one complete wave.
    with Collector(
        config, TrainingFactory(config, plan["base"]), device, out / "smoke-errors"
    ) as collector:
        empty, _, _ = collector.collect(policy, None, 0, deadline=perf_counter() - 1)
        assert not empty
        deadline = perf_counter() + 0.1
        episodes, next_episode, timings = collector.collect(
            policy, None, 0, max_decisions=1, deadline=deadline
        )
        assert perf_counter() >= deadline
    assert len(episodes) == 128 and next_episode == 128
    assert all(
        e["result"]["reward"] == int(e["result"]["winner"] == e["spec"]["learner_side"])
        for e in episodes
    )
    assert all(e["old_hidden"][0] is None for e in episodes)
    assert all(e["spec"]["opponent"] == "cp" for e in episodes)
    replay = None
    if device.startswith("cuda"):
        # Validate true online behavior probabilities before any research update.
        backend = BatchedBackend(policy, episodes, config, device)
        replay = backend.precheck()
    write_json(
        out / f"collector-smoke-{device}.json",
        {
            "passed": True,
            "episodes": len(episodes),
            "decisions": sum(len(e["chosen"]) for e in episodes),
            "next_episode": next_episode,
            "timings": timings,
            "episode_zero_memory": True,
            "reward_win_probability": True,
            "research_budget_charged": False,
            "online_behavior_replay": replay,
        },
    )


def cp_equivalence(config, plan, out):
    results = []
    for side in (0, 1):
        case = next(
            c
            for c in plan["development"]
            if c["learner_side"] == side and c["placement"] == "contact"
        )

        class Recorder(GreedyAgent):
            def reset(self, context):
                super().reset(context)
                self.commands = []

            def act(self, observation, actions):
                action = super().act(observation, actions)
                self.commands.append(action)
                return action

        class CheckedCP(CPAgent):
            def reset(self, context):
                super().reset(context)
                self.callbacks = 0

            def on_action_result(self, action, result, observation):
                self.callbacks += 1
                super().on_action_result(action, result, observation)

        learner = Recorder()
        enemy = CheckedCP(search_config(config))
        agents = [None, None]
        agents[side], agents[1 - side] = learner, enemy
        try:
            match = run_match(
                MatchConfig(
                    case["scenario"]["map"], case["armies"], 30, max_calls_per_turn=2000
                ),
                agents,
                scenario=case["scenario"],
                agent_seeds=case["agent_seeds"],
            )
            assert match.result["status"] == "completed", match.result["error"]
            callbacks = enemy.callbacks
        finally:
            enemy.close()
        spec = {
            **case,
            "opponent": "cp",
            "opponent_search": config["teacher_search"],
            "reward_profile": "win_probability",
            "max_calls_per_turn": 2000,
        }
        with DecisionEnv(spec) as env:
            # Count actual callbacks including the last physical CP command.
            count = [0]
            callback = env.opponent.on_action_result

            def checked(*args):
                count[0] += 1
                return callback(*args)

            env.opponent.on_action_result = checked
            state = env.request()
            for action in learner.commands:
                assert not state["done"] and action in state["actions"]
                env.prepare(action)
                result = env.battle.step(action)
                env.applied(action, result)
                state = env.request()
            assert state["done"]
            assert env.history == [s["action"] for s in match.replay["steps"]]
            assert env.battle.snapshot() == match.final
            assert count[0] == callbacks
            assert state["result"]["reward"] == int(match.result["winner"] == side)
            assert state["result"]["winner"] == match.result["winner"]
            with __import__("civ5sim", fromlist=["Battle"]).Battle(
                case["scenario"]
            ) as replay:
                for actual, reference in zip(env.history, match.replay["steps"]):
                    assert asdict(replay.step(actual)) == reference["result"]
                    assert digest(replay.snapshot()) == reference["state_sha256"]
        results.append(
            {
                "side": side,
                "commands": len(match.replay["steps"]),
                "callbacks": callbacks,
                "outcome": match.result["outcome"],
                "all_state_hashes_equal": True,
            }
        )
    write_json(out / "cp-equivalence.json", results)
    return results


def ppo_parity(config, device, out):
    reference_config = deepcopy(config)
    reference_config["performance"]["ppo_optimized_inputs"] = False
    reference = create_policy(config, device)
    # Same BF16/compiled forward shapes, but ordinary gradient accumulation and cuDNN GRU.
    reference.coalesced_spatial = False
    reference.coalesced_accumulation = False
    reference._memory_runner = None
    candidate = create_policy(config, device)
    episodes = public_episodes(reference, device)
    spans = [(i, 1, len(e["chosen"])) for _ in range(4) for i, e in enumerate(episodes)]
    count = sum(b - a for _, a, b in spans)
    records = []
    expected = None
    for name, policy, cfg in (
        ("reference", reference, reference_config),
        ("optimized", candidate, config),
    ):
        print(f"PPO parity {device}/{name}", flush=True)
        optimizer = optimizer_for(policy, cfg, device)
        backend = BatchedBackend(policy, episodes, cfg, device)
        pre = backend.precheck()
        optimizer.zero_grad(set_to_none=True)
        backend.backward(spans, count, 0)
        grads = {k: p.grad.detach().cpu().clone() for k, p in policy.named_parameters()}
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        backend.record_gradient(norm)
        loss = backend.statistics(count)[0]
        optimizer.step()
        weights = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        if expected is None:
            expected = grads, weights, loss
        else:
            for field, actual, target in (
                ("gradients", grads, expected[0]),
                ("weights", weights, expected[1]),
            ):
                for k in actual:
                    torch.testing.assert_close(
                        actual[k],
                        target[k],
                        rtol=3e-5,
                        atol=2e-7,
                        msg=lambda msg: k + ": " + msg,
                    )
            for k in loss:
                np.testing.assert_allclose(
                    loss[k], expected[2][k], rtol=3e-5, atol=2e-7
                )
        records.append(
            {
                "path": name,
                "precheck": pre,
                "loss": loss,
                "gradient_norm": float(norm),
                "Adam_steps": sorted(
                    set(float(s["step"]) for s in optimizer.state.values())
                ),
            }
        )
    # The critic reset must not alter policy logits or recurrent outputs on public histories.
    parent = __import__(
        "civ5rl.experiments.imitation_helpers", fromlist=["load_policy"]
    ).load_policy(config, ROOT / config["experiment"]["parent"], device)
    fresh = create_policy(config, device)
    with torch.no_grad():
        batch = batch_inputs(episodes[0]["encoded"], device)
        a, _, ha = parent.forward_sequences(batch, [len(episodes[0]["chosen"])])
        b, v, hb = fresh.forward_sequences(batch, [len(episodes[0]["chosen"])])
        assert torch.equal(a, b) and torch.equal(ha, hb) and torch.count_nonzero(v) == 0
    result = {
        "passed": True,
        "device": device,
        "rtol": 3e-5,
        "atol": 2e-7,
        "records": records,
        "policy_GRU_unchanged": True,
        "critic_initially_zero": True,
    }
    write_json(out / f"ppo-parity-{device}.json", result)
    del reference, candidate, parent, fresh
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def preflight(config, plan, out):
    configure(17, "cpu")
    cp_equivalence(config, plan, out)
    ppo_parity(config, "cpu", out)
    configure(17, "cuda")
    ppo_parity(config, "cuda", out)
    collector_smoke(config, plan, out)
    collector_smoke(config, plan, out, "cuda")
    write_json(
        out / "preflight.json",
        {
            "passed": True,
            "CP_equivalence_both_sides": True,
            "PPO_CPU_CUDA_parity": True,
        },
    )


if __name__ == "__main__":
    from civ5rl.provenance.common import read_json

    out = ROOT / "artifacts/rl/ppo-after-imitation"
    preflight(
        read_json(ROOT / "configs/rl-ppo-after-imitation.json"),
        read_json(out / "manifest.json"),
        out,
    )
