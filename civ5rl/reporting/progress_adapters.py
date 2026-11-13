"""Artifact semantics for run families, with explicit experiment overrides."""

from statistics import median

from civ5rl.reporting.progress import Observation, Sample, Stage


def _status(source):
    return source.read("status.json", {}).get("status", "unknown")


def _phase(source, fallback):
    status = _status(source)
    if status in {"complete", "paused", "technical_error", "closed_by_user"}:
        return status
    events = [source.last(p) for p in ("telemetry/phases.jsonl", "training-telemetry/phases.jsonl",
                                       "training-telemetry/resources.jsonl")]
    recent = [e for e in events if e.get("timestamp_utc", "") >= source.session.get("started_at", "")]
    return max(recent, key=lambda e: e.get("timestamp_utc", "")).get("phase", fallback) if recent else fallback


def _evaluations(source, purpose):
    baseline = set(source.session.get("evaluation_keys", []))
    rows = source.read("evaluations.json", {})
    return [r for key, r in rows.items() if key not in baseline and r.get("purpose") == purpose]


def _evaluation_seconds(source, purpose):
    # Short records are cache hits, not measurements of this workload. Use only
    # this process's measurements when a recorder session is available.
    rows = [r for r in _evaluations(source, purpose) if r.get("seconds", 0) > 10]
    latest_runtime = rows[-1].get("runtime") if rows else None
    values = [r["seconds"] for r in rows if r.get("runtime") == latest_runtime]
    return median(values[-3:]) if values else None


def _warnings(source):
    result = ["ETA excludes future failures and downtime; current unfinished work is counted in full."]
    if not source.session:
        result.append("No process session: historical measurements may mix machines and restarts.")
    return result


def _phase_seconds(source, phase):
    rows = source.rows("training-telemetry/phases.jsonl")
    values = [b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(rows, rows[1:])
              if a.get("phase") == phase and a.get("session_id") == b.get("session_id")
              and a.get("timestamp_utc", "") >= source.session.get("started_at", "")
              and b["elapsed_seconds"] > a["elapsed_seconds"]]
    return median(values[-3:]) if values else None


class PPOAdapter:
    def observe(self, source):
        cfg = source.config
        state = source.read("state.json", {})
        rows = source.rows("updates.jsonl")
        previous_work, previous_active = 0, 0.0
        samples = []
        start = source.session.get("ppo_rollout", 0)
        for row in rows:
            work = row.get("total_decisions", previous_work + row.get("decisions", 0))
            active = row.get("active_seconds")
            if active is not None and row.get("rollout", row.get("update", 0)) > start:
                samples.append(Sample(work - previous_work, active - previous_active,
                                      row.get("ppo", {}).get("runtime_compilation_seconds", 0) or 0))
            previous_work = work
            if active is not None:
                previous_active = active
        experiment = cfg.get("experiment", {})
        target = experiment.get("decisions", cfg.get("decisions"))
        complete = _status(source) == "complete"
        limit = experiment.get("active_seconds")
        budgets = {}
        if limit is not None:
            limit += source.read("active-budget.json", {}).get("additional_active_seconds", 0)
            budgets["active"] = dict(limit_seconds=limit, spent_seconds=state.get("active_seconds", previous_active))
        grouped = state.get("Adam_counters")
        return Observation(_status(source), _phase(source, state.get("phase", "unknown")), "decisions", previous_work, target,
                           samples=samples, budgets=budgets, work_budget="active",
                           stages=[] if complete or type(self) is not PPOAdapter else [Stage("unmodeled_remaining_phases", 1, None)],
                           work_stopped=complete or bool(state.get("stop_reason")),
                           details=dict(collected=state.get("decisions"), updates=state.get("updates"),
                                        adam_inherited=state.get("origin_Adam_counters", state.get("origin_Adam_counter", experiment.get("parent_optimizer_steps", 0))),
                                        adam_new_accepted=state.get("optimizer_steps", 0),
                                        **({"adam_counters": grouped,
                                            "critic_preparation_steps": state.get("critic_progress", {}).get("steps", 0)}
                                           if grouped is not None else {"adam_total": state.get("origin_Adam_counter", experiment.get("parent_optimizer_steps", 0)) + state.get("optimizer_steps", 0)}),
                                        rate_scope="collection_and_optimization_active_time",
                                        session_id=source.session.get("id")),
                           warnings=_warnings(source))


class TargetBCPPOAdapter(PPOAdapter):
    def observe(self, source):
        observation = super().observe(source)
        cfg, state = source.config, source.read("state.json", {})
        experiment = cfg["experiment"]
        purpose = cfg.get("ppo", {}).get("evaluation_purpose", "development")
        seconds = _evaluation_seconds(source, purpose)
        points = state.get("development_points", {})
        stopped = bool(state.get("stop_reason"))
        # After an early training stop, future decision milestones are abandoned;
        # the runner evaluates the final changed policy once instead.
        pending = [p for p in experiment["milestones"] if str(p) not in points
                   and p <= (state.get("decisions", 0) if stopped else experiment["decisions"])]
        evaluated = state.get("evaluated", {})
        if stopped:
            latest = state.get("latest")
            pending = [] if any(r.get("checkpoint") == latest for r in evaluated.values()) else ["final"]
        elif not evaluated:
            pending = ["initial", *pending]
        if observation.status == "complete":
            pending = []
        observation.stages.append(Stage(purpose, len(pending), seconds))
        confirmation = source.read("confirmation.json")
        if confirmation is None and observation.status != "complete":
            # Two policies on Additional-256. Scale by the actual dev workload.
            evaluations = source.read("evaluations.json", {}).values()
            counts = [r.get("runs", {}).get("argmax", {}).get("cp", {}).get("summary", {}).get("matches")
                      for r in evaluations if r.get("purpose") == purpose]
            matches = next((n for n in counts if n), None)
            additional = seconds * 512 / matches if seconds is not None and matches else None
            observation.stages.append(Stage("additional_256_two_policies", 1, additional, conditional=True))
        observation.details["pending_development_points"] = pending
        observation.details["early_gain_required_wins"] = state.get("early_gain_required_wins")
        observation.details["evaluation_runtime"] = source.read("evaluation-runtime.json", {}).get("runtime")
        observation.warnings.append("Additional-256 depends on selection and the experiment's development gate.")
        return observation


class BCAdapter:
    def observe(self, source):
        cfg = source.config
        state = source.read("state.json", {})
        training = source.read("training.json", {})
        collection = source.read("collection.json", {})
        rows = source.rows("steps.jsonl")
        samples = []
        completed, previous_active = 0, 0.0
        start = source.session.get("bc_step", 0)
        current_rows = [r for r in rows if r.get("step", 0) > start]
        has_active = bool(current_rows) and all("active_seconds" in r for r in current_rows)
        for row in rows:
            completed += row.get("labels", 0)
            seconds = row.get("seconds", 0)
            if "active_seconds" in row:
                if row.get("step", 0) > start and start and not previous_active:
                    previous_active = source.session.get("active_seconds", 0)
                seconds = row["active_seconds"] - previous_active
                previous_active = row["active_seconds"]
            if row.get("step", 0) > start:
                samples.append(Sample(row.get("labels", 0), seconds, row.get("compile_seconds", 0) or 0))
        labels = collection.get("labels")
        if cfg.get("kind") in {"rl01.cp_imitation_target_more_data", "rl01.cp_imitation_target_c2"}:
            labels = collection.get("new_labels", 0) + collection.get("old_labels", 0) or labels
        target = labels * cfg["epochs"] if labels is not None and "epochs" in cfg else None
        if cfg.get("max_adam_steps") is not None:
            # A step cap can terminate on a short minibatch: no fabricated exact
            # label target. Model the scheduled optimizer work in steps instead.
            target = cfg["max_adam_steps"]
            completed = max((r.get("step", 0) for r in rows), default=0)
            samples = [Sample(1, s.seconds, s.compilation_seconds) for s in samples]
        active = max(state.get("active_seconds", 0), training.get("active_seconds", 0), previous_active)
        budgets = {}
        if cfg.get("training_seconds") is not None:
            budgets["active"] = dict(limit_seconds=cfg["training_seconds"] + source.read("active-budget.json", {}).get("additional_active_seconds", 0), spent_seconds=active,
                                     work_estimate_complete=has_active or training.get("status") == "complete")
        warnings = _warnings(source)
        if not has_active:
            warnings.append("BC rate uses step timers without complete active-time measurements; while training, budget fit is unknown.")
        observation = Observation(_status(source), _phase(source, _status(source)),
                                  "steps" if cfg.get("max_adam_steps") is not None else "labels",
                                  completed, target, samples=samples, budgets=budgets,
                                  work_budget="active", work_stopped=training.get("status") == "complete" or _status(source) == "complete",
                                  stages=[] if _status(source) == "complete" or type(self) is not BCAdapter else [Stage("unmodeled_remaining_phases", 1, None)],
                                  details=dict(epoch=state.get("epoch"), steps=len(rows),
                                               rate_scope="training_active_time" if has_active else "optimizer_steps_only",
                                               session_id=source.session.get("id")), warnings=warnings)
        return observation


class TargetScaleBCAdapter(BCAdapter):
    def observe(self, source):
        observation = super().observe(source)
        cfg = source.config
        evaluated = source.read("evaluations.json", {}).values()
        epochs = {r.get("epoch") for r in evaluated if r.get("purpose") == "development"}
        pending = [e for e in cfg.get("evaluate_epochs", []) if e not in epochs]
        training = source.read("training.json", {})
        if training.get("status") == "complete":
            completed_epochs = {r["epoch"] for r in training.get("epochs", [])}
            pending = [e for e in pending if e in completed_epochs]
            latest = training.get("latest")
            if latest and not any(r.get("checkpoint") == latest for r in source.read("evaluations.json", {}).values()):
                pending = ["final"]
        if cfg.get("skip_milestone_evaluations"):
            pending = []
        if observation.status == "complete":
            pending = []
        observation.stages.append(Stage("development", len(pending), _evaluation_seconds(source, "development")))
        if source.read("confirmation.json") is None and observation.status != "complete":
            observation.stages.append(Stage("confirmation", 1, None, conditional=True))
        observation.warnings.append("Development timing excludes separate fixed-training diagnostics and final report work.")
        return observation


class C2BCAdapter(BCAdapter):
    def observe(self, source):
        observation = super().observe(source)
        cfg = source.config
        evaluated = source.read("evaluations.json", {}).values()
        seen = {r.get("checkpoint") for r in evaluated if r.get("purpose") == "development_512"}
        # Four references plus saved milestone candidates, then latest if distinct.
        paths = {"cp", "checkpoints/reference-bc-epoch-10.pt", "checkpoints/reference-update-0037.pt",
                 "checkpoints/reference-bc-more-data-epoch-10.pt"}
        paths.update(f"checkpoints/epoch-{int(e):02d}.pt" for e in cfg.get("evaluate_epochs", []))
        training = source.read("training.json", {})
        if training.get("status") == "complete":
            paths = {p for p in paths if p == "cp" or (source.directory / p).is_file()}
            if training.get("latest"):
                paths.add(training["latest"])
        pending = paths - seen if observation.status != "complete" else set()
        observation.stages.append(Stage("development_512", len(pending), _evaluation_seconds(source, "development_512")))
        epochs_done = len([r for r in training.get("epochs", source.read("state.json", {}).get("epochs_history", [])) if r.get("completed")])
        offline_pending = 0 if observation.work_stopped else max(0, cfg.get("epochs", 0) - epochs_done)
        observation.stages.append(Stage("offline_metrics", offline_pending, _phase_seconds(source, "offline_metrics")))
        if source.read("confirmation.json") is None and observation.status != "complete":
            observation.stages.append(Stage("additional_256", 1, None, conditional=True))
        return observation


class UnknownAdapter:
    def observe(self, source):
        return Observation(_status(source), "unknown", "work", 0, None,
                           stages=[Stage("unmodeled_work", 1, None)],
                           warnings=[f"No progress adapter for {source.config.get('kind')!r}; ETA is unknown."])


# Exact experiment adapters take precedence over family adapters. New run types
# register an adapter; the estimator, recorder, transport and CLI stay unchanged.
ADAPTERS = {
    "rl01.ppo_after_target_bc": TargetBCPPOAdapter,
    "rl01.cp_imitation_target_scale": TargetScaleBCAdapter,
    "rl01.cp_imitation_target_more_data": TargetScaleBCAdapter,
    "rl01.cp_imitation_target_c2": C2BCAdapter,
}
FAMILIES = [("rl01.ppo_", PPOAdapter), ("rl01.cp_imitation", BCAdapter)]


def adapter_for(kind):
    if kind in ADAPTERS:
        return ADAPTERS[kind]()
    return next((cls() for prefix, cls in FAMILIES if kind.startswith(prefix)), UnknownAdapter())
