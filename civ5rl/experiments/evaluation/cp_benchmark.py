"""Evaluate a CP agent on frozen placements, publishing aggregate results only."""

from collections import Counter, defaultdict
from pathlib import Path
from time import get_clock_info, perf_counter, thread_time_ns

from civ5rl.engine.evaluation.proxy import evaluation_agent
from civ5rl.protocols.common import case_content_hash
from civ5rl.provenance.common import read_json, write_json
from civ5sim import MatchConfig, run_match
from civ5sim.agents import load_agent


class DecisionCPU:
    """Disjoint top-level phases on the CP worker thread, including native calls."""

    clock_ns = staticmethod(thread_time_ns)
    phases = ("prepare_actions", "prepare_observation", "prepare_planning", "act", "callback")

    def __init__(self, side):
        self.side = side
        self.previous_turn = None
        self.counts = Counter(matches=1)

    def __call__(self, sample):
        kind = sample["kind"]
        if sample["side"] != self.side or kind not in self.phases:
            return
        self.counts[kind + "_ns"] += sample["ns"]
        if kind == "act":
            self.counts["decisions"] += 1
            if self.previous_turn != sample["turn"]:
                self.counts["turns"] += 1
                self.previous_turn = sample["turn"]


def _match(task):
    case, checkpoint, contender, timing = task
    policy_side = case["learner_side"]
    cp_side = 1 - policy_side
    meter = DecisionCPU(cp_side) if timing else None
    cp = load_agent(contender)
    policy = evaluation_agent(checkpoint, "argmax", cpu_seed=17)
    agents = [None, None]
    agents[policy_side], agents[cp_side] = policy, cp
    config = MatchConfig(
        case["scenario"]["map"],
        case["armies"],
        case["max_rounds"],
        max_calls_per_turn=case["max_calls_per_turn"],
        turn_reset_mode=case["scenario"].get("turn_reset_mode", "legacy"),
    )
    try:
        match = run_match(
            config, agents, scenario=case["scenario"], agent_seeds=case["agent_seeds"], metrics=meter
        )
        if match.result["status"] != "completed":
            raise RuntimeError(match.result["error"])
        winner = match.result["winner"]
        result = "D" if winner is None else "W" if winner == cp_side else "L"
        if meter and meter.counts["decisions"] != match.result["agent_calls"][cp_side]:
            raise RuntimeError("Incomplete CP decision timing")
        return case["placement"], cp_side, result, meter.counts if meter else None
    finally:
        cp.close()


def benchmark_policy(args):
    import hashlib

    from civ5rl.engine.evaluation.executor import EvaluationExecutor, EvaluationRuntime, evaluation_precision
    from civ5sim.combat_ai import LIBRARY_PATH

    manifest = read_json(args.manifest)
    cases = manifest["development_512"]
    strata = Counter((c["placement"], c["learner_side"]) for c in cases)
    expected = {(placement, side): 128 for placement in ("contact", "separated") for side in (0, 1)}
    if (
        strata != expected
        or len({(c["case_id"], c["learner_side"]) for c in cases}) != 512
        or any(case_content_hash(c) != c["case_hash"] for c in cases)
    ):
        raise ValueError("Expected 512 intact, paired development cases")
    totals = defaultdict(Counter)
    groups = defaultdict(Counter)
    cpu = defaultdict(Counter)
    started = perf_counter()
    runtime = EvaluationRuntime(args.device, args.workers)
    with evaluation_precision():
        executor = EvaluationExecutor(str(args.checkpoint), runtime)
        tasks = ((case, str(args.checkpoint), args.agent, args.cpu_timing) for case in cases)
        for count, (placement, side, result, measured) in enumerate(executor.run(_match, tasks), 1):
            if measured:
                cpu[placement].update(measured)
            totals[placement][result] += 1
            groups[(placement, side)][result] += 1
            if count % 64 == 0:
                print(f"{count}/{len(cases)} matches", flush=True)
        identity = executor.runtime.identity()

    def counts(counter):
        return {key: counter[key] for key in "WDL"}

    result = {
        "agent": args.agent,
        "checkpoint": str(args.checkpoint),
        "split": "development_512",
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "ai_library_sha256": hashlib.sha256(LIBRARY_PATH.read_bytes()).hexdigest(),
        "runtime": identity,
        "seconds": perf_counter() - started,
        "by_placement": {p: counts(c) for p, c in totals.items()},
        "by_placement_and_side": {p: {str(s): counts(groups[p, s]) for s in (0, 1)} for p in totals},
    }
    if args.cpu_timing:
        result["cpu_timing"] = {
            "clock": get_clock_info("thread_time").implementation,
            "all": dict(sum(cpu.values(), Counter())),
            "by_placement": dict(cpu),
        }
    write_json(args.out, result)
    print(result["by_placement"], flush=True)
