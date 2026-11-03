"""Opt-in decision evidence. Full-information artifacts never enter agent observations."""

import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import asdict, is_dataclass
from functools import wraps
from pathlib import Path

from .engine import digest

ROOT = Path(__file__).resolve().parents[1]
MODES = ("off", "compact", "full")


def plain(value):
    return json.loads(json.dumps(asdict(value) if is_dataclass(value) else value))


def provenance():
    files = {}
    for folder in ("native", "civ5sim", "tools"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.suffix in (".py", ".cpp", ".h", ".inc"):
                files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    libraries = {}
    for pattern in ("*.dll", "*.so", "combat_rules.h", "cp_geometry.inc"):
        for path in sorted((ROOT / "build").glob(pattern)):
            libraries[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git": commit, "sources_sha256": digest(files), "sources": files, "libraries": libraries}


def observational(method):
    """Recording failure never changes the command or planner result."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- diagnostic isolation
            self.errors.append(f"{method.__name__}: {type(exc).__name__}: {exc}")
            return None
    return wrapped


class DecisionRecorder:
    def __init__(self, mode):
        if mode not in MODES or mode == "off":
            raise ValueError("Recorder requires compact or full")
        self.mode = mode
        self.errors = []
        self.states = {}
        self.calls = []
        self.queries = []
        self.assignments = {}  # Hold objects: Python IDs must not be recycled.
        self.tasks = {}
        self.current = None
        self.trace_directory = None
        try:
            self.metadata = provenance()
        except Exception as exc:  # noqa: BLE001 -- fingerprints cannot abort gameplay
            self.metadata = {"unavailable": str(exc)}
            self.errors.append(f"provenance: {exc}")

    def close(self):
        if self.trace_directory is not None:
            self.trace_directory.cleanup()
            self.trace_directory = None

    def __del__(self):
        self.close()

    def store(self, value):
        value = plain(value)
        key = digest(value)
        self.states.setdefault(key, value)
        return key

    @observational
    def begin(self, command, side, observation, agent):
        self.current = {"id": len(self.calls), "command": command, "side": side,
                        "observation": self.store(observation), "events": [], "queries": [],
                        "agent": f"{type(agent).__module__}:{type(agent).__qualname__}"}
        self.calls.append(self.current)
        if hasattr(agent, "diagnostic_state"):
            self.current["coordinator_before"] = plain(agent.diagnostic_state())
        # CP owns this explicit optional hook. Other agents need not support it.
        if hasattr(agent, "planner"):
            agent._decision_recorder = self
            if agent.planner is not None:
                agent.planner._decision_recorder = self

    @observational
    def event(self, event, task):
        task_id = None
        if task is not None:
            identity = id(task)
            if identity not in self.tasks:
                self.tasks[identity] = (task, len(self.tasks))
            task_id = self.tasks[identity][1]
        self.current["events"].append({**plain(event), "task_id": task_id})

    @observational
    def action(self, action, assignment=None):
        self.current["action"] = plain(action)
        origin = self.assignments.get(id(assignment))
        if origin is not None and origin[0] is assignment:
            self.current["origin"] = {"query": origin[1], "assignment": origin[2]}

    @observational
    def result(self, request, config, mode, result, trace, version):
        query_id = len(self.queries)
        packed = plain(request)
        packed["state"] = self.store(packed["state"])
        query = {"id": query_id, "call": self.current["id"], "request": packed,
                 "config": plain(config), "mode": mode, "result": plain(result),
                 "trace": trace, "version": version}
        self.queries.append(query)
        self.current["queries"].append(query_id)
        for index, assignment in enumerate(getattr(result, "assignments", ())):
            self.assignments[id(assignment)] = (assignment, query_id, index)
        if trace is not None:
            # Do not retain an entire match's expanded search trees in RAM.
            # Preserve the response/provenance even if spooling the trace fails.
            query["trace"] = {"available": False, "error": "trace storage incomplete"}
            if self.trace_directory is None:
                self.trace_directory = tempfile.TemporaryDirectory(prefix="civ5sim-diagnostics-")
            path = Path(self.trace_directory.name) / f"{query_id}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(trace, stream)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            query["trace"] = {"artifact": f"{query_id}.json.gz", "sha256": checksum,
                              "truncated": trace.get("truncated", False)}

    @observational
    def end(self, action):
        self.current["action"] = plain(action)

    @observational
    def failure(self, phase, exc):
        message = f"{phase}: {type(exc).__name__}: {exc}"
        self.errors.append(message)
        if self.current is not None:
            self.current["failure"] = message

    def save(self, directory, replay):
        payload = {"schema": 1, "kind": "civ5sim.decision-diagnostics", "mode": self.mode,
                   "replay_sha256": digest(replay), "provenance": self.metadata,
                   "complete": not self.errors, "errors": self.errors,
                   "states": self.states, "calls": self.calls, "queries": self.queries}
        path = Path(directory) / "diagnostics.json.gz"
        try:
            if self.trace_directory is not None:
                destination = Path(directory) / "diagnostic-traces"
                destination.mkdir(exist_ok=True)
                for source in Path(self.trace_directory.name).glob("*.json.gz"):
                    temporary_trace = destination / (source.name + ".tmp")
                    shutil.copyfile(source, temporary_trace)
                    temporary_trace.replace(destination / source.name)
            temporary = path.with_suffix(".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False)
            temporary.replace(path)
        except (OSError, ValueError) as exc:
            warnings.warn(f"Incomplete diagnostic package: {exc}", stacklevel=2)
            (Path(directory) / "diagnostics-error.txt").write_text(str(exc), encoding="utf-8")


def recorder(mode):
    if mode not in MODES:
        raise ValueError(f"Unknown diagnostics mode: {mode}")
    return None if mode == "off" else DecisionRecorder(mode)


def load_package(path, replay=None):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema") != 1 or value.get("kind") != "civ5sim.decision-diagnostics":
        raise ValueError("Unsupported diagnostic package")
    if replay is not None and value["replay_sha256"] != digest(replay):
        raise ValueError("Diagnostic package belongs to a different replay")
    for key, state in value["states"].items():
        if digest(state) != key:
            raise ValueError("Corrupt diagnostic state")
    value["_directory"] = str(Path(path).resolve().parent)
    return value


def query_with_trace(package, query):
    """Resolve only the requested trace, verifying its bytes and safe relative name."""
    trace = query.get("trace")
    if not trace or "artifact" not in trace:
        return query
    name = trace["artifact"]
    if name != f"{query['id']}.json.gz":
        raise ValueError("Invalid diagnostic trace name")
    path = Path(package["_directory"]) / "diagnostic-traces" / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != trace["sha256"]:
        raise ValueError("Corrupt diagnostic trace")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return {**query, "trace": json.load(stream)}


def unpack_request(package, query):
    from .combat_ai import (
        Aggression,
        Contact,
        ContactMemory,
        PlanningState,
        PlanningTile,
        PlanningUnit,
        SearchConfig,
        SearchRequest,
    )
    raw = dict(query["request"])
    state = dict(package["states"][raw.pop("state")])
    state["tiles"] = tuple(PlanningTile(**tile) for tile in state["tiles"])
    state["units"] = tuple(PlanningUnit(**unit) for unit in state["units"])
    state["visibility"] = tuple(state["visibility"])
    state["visibility_history"] = tuple(tuple(row) for row in state["visibility_history"])
    contact = dict(raw.pop("contacts"))
    contact["known_enemies"] = tuple(Contact(**c) for c in contact["known_enemies"])
    raw["selected_units"] = tuple(raw["selected_units"])
    if raw["aggression"] is not None:
        raw["aggression"] = Aggression(raw["aggression"])
    config = dict(query["config"])
    config["aggression"] = Aggression(config["aggression"])
    return SearchConfig(**config), SearchRequest(
        state=PlanningState(**state), contacts=ContactMemory(**contact), **raw)


def repeat_query(package, query):
    from .combat_ai import Planner, SupportMode
    config, request = unpack_request(package, query)
    with Planner(config) as planner:
        planner.set_trace(True)
        if query["mode"] == "candidates":
            result = planner.attack_candidates(request)
        elif query["mode"] == "search":
            result = planner.search(request)
        else:
            result = planner.support(request, SupportMode(query["mode"]))
        result = plain(result)
        current_provenance = provenance()
        recorded_provenance = package.get("provenance", {})
        same_environment = all(current_provenance.get(key) == recorded_provenance.get(key)
                               for key in ("sources_sha256", "libraries"))
        return {"kind": "current_code_experiment", "matches_recorded_environment": same_environment, "query": query["id"],
                "version": planner.version, "result": result, "trace": planner.read_trace(),
                "matches_recorded_result": result == query["result"],
                "provenance": current_provenance}
