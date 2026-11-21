"""Verified command-sequence branches of saved policy-vs-CP matches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation.replay import apply_replay_step, verify_replay
from civ5rl.protocols.imitation import search_config
from civ5rl.provenance.common import configure, file_sha256, read_json, write_json
from civ5sim import Action, Battle, MatchConfig, run_match
from civ5sim.cp_agent import CPAgent


@dataclass(frozen=True)
class Branch:
    source: Path
    replay: dict[str, Any]
    case: dict[str, Any]
    command: int
    replacements: list[Action]
    forbidden_move_targets: tuple[int, ...]
    continue_to_round: int | None

    @property
    def boundary(self) -> int:
        return self.command - 1


@dataclass
class BranchProgress:
    index: int = 0
    original_argmax: dict[str, Any] | None = None
    masked_decisions: int = 0


class BranchCP:
    def __init__(self, branch: Branch, progress: BranchProgress):
        self.branch = branch
        self.progress = progress
        self.inner = CPAgent(search_config({"teacher_search": branch.case["opponent_search"]}))

    def reset(self, context):
        self.inner.reset({**context, "max_rounds": self.branch.case["max_rounds"]})
        self.planner = self.inner.planner

    def act_with_planning(self, observation, actions, services):
        action = self.inner.act_with_planning(observation, actions, services)
        index = self.progress.index
        if index < self.branch.boundary and asdict(action) != self.branch.replay["steps"][index]["action"]:
            raise ValueError(f"CP decision differs at command {index + 1}")
        self.progress.index += 1
        return action

    def on_action_result(self, *args):
        return self.inner.on_action_result(*args)

    def drain_diagnostics(self):
        return self.inner.drain_diagnostics()


class BranchModel:
    def __init__(self, branch: Branch, progress: BranchProgress, restriction_turn, checkpoint, device):
        from civ5rl.core.agent import RLAgent

        self.branch = branch
        self.progress = progress
        self.restriction_turn = restriction_turn
        self.inner = RLAgent.from_checkpoint(checkpoint, "cpu", "argmax")
        self.inner.policy.to(device)
        self.inner.device = device
        self.requires_attack_preview = self.inner.requires_attack_preview

    def reset(self, context):
        self.inner.reset({**context, "max_rounds": self.branch.case["max_rounds"]})

    def act(self, observation, actions):
        import torch

        index = self.progress.index
        self.progress.index += 1
        boundary = self.branch.boundary
        replacements = self.branch.replacements
        if index < boundary + len(replacements):
            actions = ordered_actions(actions)
            # Advance recurrent state on the public history, then remember the actual command.
            with torch.inference_mode():
                logits = self.inner._infer(observation, actions)
            if index < boundary:
                action = Action(**self.branch.replay["steps"][index]["action"])
            else:
                action = replacements[index - boundary]
            if action not in actions:
                raise ValueError(f"Recorded model command is illegal at {index + 1}")
            self.inner.encoder.remember_action(action, observation)
            if index == boundary:
                self.progress.original_argmax = asdict(actions[int(logits[0].argmax())])
            return action

        forbidden = self.branch.forbidden_move_targets
        if forbidden and observation["turn"] == self.restriction_turn:
            actions = ordered_actions(actions)
            with torch.inference_mode():
                logits = self.inner._infer(observation, actions).clone()
                for action_index, action in enumerate(actions):
                    if action.kind in ("move", "swap") and action.target in forbidden:
                        logits[0, action_index] = -torch.inf
                if not torch.isfinite(logits).any():
                    raise ValueError("Movement restriction leaves no action")
                self.progress.masked_decisions += 1
                return self.inner.action_from_logits(logits, observation, actions)
        return self.inner.act(observation, actions)

    def drain_diagnostics(self):
        return self.inner.drain_diagnostics()


def _inspect_branch(branch: Branch) -> dict[str, Any]:
    steps = branch.replay["steps"]
    learner_side = branch.case["learner_side"]
    with Battle(branch.replay["initial"]) as battle:
        for number, step in enumerate(steps[: branch.boundary], 1):
            apply_replay_step(battle, step, mismatch=f"Source prefix differs at command {number}")
        snapshot = battle.snapshot()
        if branch.continue_to_round is None and snapshot["state"]["side"] != learner_side:
            raise ValueError("Replacement must be on the model's side")
        restriction_turn = snapshot["state"]["turn"]
        if any(
            type(tile) is not int or not 0 <= tile < len(snapshot["map"]["tiles"])
            for tile in branch.forbidden_move_targets
        ):
            raise ValueError("Forbidden move target is outside the map")
        checks = []
        for offset, action in enumerate(branch.replacements):
            active_side = battle.snapshot()["state"]["side"]
            available = [
                asdict(candidate) for candidate in battle.available_actions() if candidate.unit == action.unit
            ]
            result = battle.step(action) if active_side == learner_side else None
            legal = result is not None and result.ok
            checks.append(
                {
                    "command": branch.command + offset,
                    "action": asdict(action),
                    "legal": legal,
                    "error": result.error if result is not None else "not_model_turn",
                    "available_for_unit": available,
                }
            )
            if not legal:
                break
    replacements = [asdict(action) for action in branch.replacements]
    return {
        "command": branch.command,
        "original": steps[branch.boundary]["action"] if branch.boundary < len(steps) else None,
        "replacement": replacements[0] if replacements else None,
        "forbidden_move_targets": list(branch.forbidden_move_targets),
        "restriction_turn": restriction_turn,
        "replacements": replacements,
        "checks": checks,
        "legal": len(checks) == len(replacements) and all(check["legal"] for check in checks),
        "error": next((check["error"] for check in checks if not check["legal"]), None),
        "prefix_verified": branch.boundary,
        "source": str(branch.source),
        "source_sha256": file_sha256(branch.source),
    }


def _run_branch_match(branch: Branch, restriction_turn, checkpoint, device):
    from civ5rl.engine.evaluation.executor import evaluation_precision

    configure(17, device)
    progress = BranchProgress()
    opponent = BranchCP(branch, progress)
    try:
        with evaluation_precision():
            model = BranchModel(branch, progress, restriction_turn, checkpoint, device)
            case = branch.case
            scenario = case["scenario"]
            config = MatchConfig(
                scenario["map"],
                case["armies"],
                branch.continue_to_round or case["max_rounds"],
                seed=case["streams"]["battle"],
                fog=scenario["fog"],
                random_damage=scenario["random_damage"],
                max_calls_per_turn=case["max_calls_per_turn"],
                turn_reset_mode=branch.replay["initial"]["state"].get("turn_reset_mode", "legacy"),
            )
            agents = [opponent, model] if case["learner_side"] == 1 else [model, opponent]
            match = run_match(config, agents, scenario=scenario, agent_seeds=case["agent_seeds"])
    finally:
        opponent.inner.close()
    return match, progress


def _save_branch(
    branch: Branch, inspection, match, progress: BranchProgress, out, checkpoint, checkpoint_sha256
):
    if match.result["status"] != "completed":
        raise RuntimeError(f"Branch failed: {match.result.get('error')}")
    boundary = branch.boundary
    if match.replay["steps"][:boundary] != branch.replay["steps"][:boundary]:
        raise ValueError("Reconstructed model/CP prefix differs physically or in RNG")
    replacements = inspection["replacements"]
    if [
        step["action"] for step in match.replay["steps"][boundary : boundary + len(replacements)]
    ] != replacements:
        raise ValueError("Replacement was not executed")
    if (
        file_sha256(checkpoint) != checkpoint_sha256
        or file_sha256(branch.source) != inspection["source_sha256"]
    ):
        raise ValueError("An input changed during branching")
    restriction_turn = inspection["restriction_turn"]
    horizon = branch.case["max_rounds"]
    restriction = inspection["forbidden_move_targets"]
    experiment = {
        "kind": "counterfactual_action",
        "description": (
            f"С команды {branch.command:03d} заменены действия {replacements}. "
            f"До конца хода {restriction_turn} запрещены перемещения на клетки {restriction}."
        ),
        "prefix_commands": boundary,
        "source": str(branch.source),
        "source_sha256": inspection["source_sha256"],
        "replacement": inspection["replacement"],
        "replacements": replacements,
        "forbidden_move_targets": restriction,
        "restriction_turn": restriction_turn,
        "masked_decisions": progress.masked_decisions,
        "continue_to_round": branch.continue_to_round,
        "model_time_horizon": horizon,
        "original_action": inspection["original"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "original_model_argmax_at_branch": progress.original_argmax,
        "cp_prefix_decisions_verified": True,
    }
    if branch.continue_to_round is not None:
        experiment.update(
            {
                "kind": "continuation",
                "resume_after": boundary,
                "description": (
                    f"Продолжение до {branch.continue_to_round} раундов. Все {boundary} исходных команд сохранены. "
                    f"Временные признаки модели ограничены {horizon} раундами."
                ),
            }
        )
    match.replay["experiment"] = experiment
    match.save(out)
    write_json(out / "case.json", branch.case)
    verify_replay(out, project_relative=False)
    write_json(
        out / "checked.json",
        {
            "replay_verified": True,
            "prefix_verified": boundary,
            "replay_sha256": file_sha256(out / "replay.json"),
        },
    )
    return {
        "directory": str(out),
        **{
            key: match.result[key]
            for key in ("status", "winner", "completed_rounds", "alive", "remaining_hp", "commands")
        },
    }


def branch_replay(
    source,
    command,
    action,
    *,
    checkpoint=None,
    out=None,
    device="cuda",
    check_only=False,
    forbidden_move_targets=(),
    continue_to_round=None,
):
    source = Path(source).resolve()
    if source.is_dir():
        source = source / "replay.json"
    replay = read_json(source)
    case = read_json(source.with_name("case.json"))
    if continue_to_round is not None:
        if type(continue_to_round) is not int or continue_to_round <= case["max_rounds"]:
            raise ValueError("Continuation must extend the original round limit")
        command = len(replay["steps"]) + 1
    if not 1 <= command <= len(replay["steps"]) + int(continue_to_round is not None):
        raise ValueError("command is a one-based existing command number")
    supplied = action if isinstance(action, list) else [action]
    if not supplied and not forbidden_move_targets and continue_to_round is None:
        raise ValueError("Provide replacements or forbidden_move_targets")
    replacements = [value if isinstance(value, Action) else Action(**value) for value in supplied]
    for replacement in replacements:
        replacement.native()
    branch = Branch(
        source, replay, case, command, replacements, tuple(forbidden_move_targets), continue_to_round
    )
    inspection = _inspect_branch(branch)
    if check_only:
        return inspection
    if not inspection["legal"]:
        raise ValueError(
            f"Replacement at {command:03d} is illegal: {inspection['error']}; use --check-only for legal alternatives"
        )
    if out is None or checkpoint is None:
        raise ValueError("checkpoint and out are required for execution")
    out, checkpoint = Path(out).resolve(), Path(checkpoint).resolve()
    if out.exists():
        raise ValueError("Use a new output directory")
    if out == source.parent or source.is_relative_to(out):
        raise ValueError("Output must not contain the source")
    checkpoint_sha256 = file_sha256(checkpoint)
    match, progress = _run_branch_match(branch, inspection["restriction_turn"], checkpoint, device)
    return _save_branch(branch, inspection, match, progress, out, checkpoint, checkpoint_sha256)
