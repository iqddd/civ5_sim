"""Serve a local, verified step-by-step viewer for saved combat replays."""

import argparse
import gzip
import hashlib
import json
import sys
import threading
from collections import OrderedDict
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from civ5sim import Battle
from civ5sim.engine import digest

TEMPLATE = Path(__file__).with_name("combat_viewer.html")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_replay(path):
    """Execute recorded commands through the kernel; expose verified display frames.

    A frame is the initial position or the position AFTER its numbered command.
    Map/roster are shared; mutable unit rows follow the explicitly named fields.
    """
    path = Path(path)
    replay = read_json(path)
    if replay.get("schema") != 2 or replay.get("kind") != "civ5sim.replay":
        raise ValueError("Ожидался реплей civ5sim.replay, schema 2")
    if not isinstance(replay.get("steps"), list):
        raise TypeError("В реплее отсутствует список команд")
    current_rules = read_json(ROOT / "build/combat_rules.json")
    if replay["initial"]["rules"] != current_rules:
        # Historical matches were executed under another kernel. Keep frames
        # verified by that kernel, bound to these exact replay/result bytes.
        cache = path.with_name("viewer-cache.json.gz")
        if cache.exists():
            with gzip.open(cache, "rt", encoding="utf-8") as stream:
                archived = json.load(stream)
            result = path.with_name("result.json")
            result_hash = hashlib.sha256(result.read_bytes()).hexdigest() if result.exists() else None
            if (archived.get("kind") == "civ5sim.viewer_cache" and archived.get("schema") == 1
                    and archived.get("replay_sha256") == hashlib.sha256(path.read_bytes()).hexdigest()
                    and archived.get("result_sha256") == result_hash
                    and archived["payload"].get("verified")
                    and len(archived["payload"]["frames"]) == len(replay["steps"]) + 1):
                return archived["payload"] | {"visibility_mode": "legacy"}
        raise ValueError("Для прежних правил нужен архив кадров, проверенный прежним ядром")
    fields = ("tile", "hp", "moves", "attacked", "acted", "fortified", "waiting")
    frames = []
    turns = []
    with Battle(replay["initial"]) as battle:
        initial = battle.snapshot()

        def append_frame(snapshot):
            frames.append({
                "turn": snapshot["state"]["turn"], "side": snapshot["state"]["side"],
                "units": [[u[key] for key in fields] for u in snapshot["units"]],
                "visibility": ["".join("1" if v else "0" for v in battle.observe(side)["visibility"])
                               for side in (0, 1)],
            })
            if not turns or turns[-1]["turn"] != snapshot["state"]["turn"]:
                turns.append({"turn": snapshot["state"]["turn"], "side": snapshot["state"]["side"],
                              "frame": len(frames)-1})

        append_frame(initial)
        snapshot = initial
        for index, step in enumerate(replay["steps"]):
            result = battle.step(step["action"])
            snapshot = battle.snapshot()
            if asdict(result) != step["result"] or digest(snapshot) != step["state_sha256"]:
                raise ValueError(f"Реплей расходится на команде {index+1}")
            append_frame(snapshot)
    outcome_path = path.with_name("result.json")
    result = read_json(outcome_path) if outcome_path.exists() else None
    # A result file must describe this replay, rather than an unrelated sibling.
    if result is not None and (not isinstance(result, dict) or
                               result.get("initial_state_sha256") != digest(initial) or
                               result.get("final_state_sha256") != digest(snapshot)):
        result = None
    return {
        "schema": 1, "kind": "civ5sim.viewer", "name": path.parent.name,
        "map": initial["map"], "roster": [{key: u[key] for key in ("id", "type", "side")}
                                           for u in initial["units"]],
        "unit_fields": fields, "frames": frames, "turns": turns,
        "steps": replay["steps"], "ai_events": replay.get("ai_events", []),
        "settings": {key: initial["state"][key] for key in ("fog", "random_damage")},
        "result": None if result is None else {key: result.get(key) for key in (
            "agents", "outcome", "winner", "reason", "status", "seeds")},
        "verified": True,
        "visibility_mode": "delayed",
        "experiment": replay.get("experiment"),
    }


class ReplayStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise ValueError(f"Каталог или реплей не найден: {self.root}")
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def refresh_target(self):
        try:
            value = read_json(self.root / "replay-refresh.json")
            if value.get("schema") == 1 and isinstance(value.get("cases"), list):
                return value
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return None

    def provenance(self, key, path, target):
        def optional(candidate):
            try:
                value = read_json(candidate)
                return value if isinstance(value, dict) else {}
            except (OSError, ValueError, TypeError):
                return {}
        case = path.parent.parent if path.parent.name in ("0", "1") else path.parent
        passport = optional(case / "passport.json")
        if not passport:
            passport = optional(self.root / "passport.json")
        checked = optional(path.with_name("checked.json"))
        status = "untracked"
        if target:
            status = "old"
            if key in target["cases"]:
                status = "updated" if (checked.get("replay_verified") is True
                    and checked.get("run_signature") == target.get("run_signature")
                    and passport.get("run_signature") == target.get("run_signature")) else "pending"
        if checked.get("continuation") is True:
            status = "continuation"
        stamp = path.stat()
        result = path.with_name("result.json")
        revision = digest({"replay": [stamp.st_mtime_ns, stamp.st_size],
            "result": result.stat().st_mtime_ns if result.exists() else None,
            "run": checked.get("run_signature"), "port": passport.get("port")})
        return {"port": passport.get("port"), "run_signature": checked.get("run_signature"),
                "refresh_status": status, "revision": revision,
                "resume_after": checked.get("prefix_verified") if status == "continuation" else None}

    def paths(self):
        paths = [self.root] if self.root.is_file() else self.root.rglob("replay.json")
        base = self.root.parent if self.root.is_file() else self.root
        result = {}
        for path in paths:
            resolved = path.resolve()
            if not resolved.is_relative_to(base):
                continue
            # The evaluation corpus stores two identical repeats. Show one.
            if path.parent.name == "1" and (path.parent.parent / "0/replay.json").exists():
                continue
            result[resolved.relative_to(base).as_posix()] = resolved
        return result

    def catalog(self):
        result = []
        target = self.refresh_target()
        for key, path in self.paths().items():
            metadata = path.with_name("result.json")
            if not metadata.exists() and not self.root.is_file():
                continue  # MatchOutcome writes its completion record last.
            try:
                value = read_json(metadata) if metadata.exists() else {}
                config = value.get("config", {})
                parent = path.parent.parent.name if path.parent.name == "0" else path.parent.name
                result.append({
                    "id": key, "name": parent, "agents": value.get("agents", []),
                    "seed": config.get("seed"), "fog": config.get("fog"),
                    "random_damage": config.get("random_damage"),
                    "commands": value.get("commands"), "outcome": value.get("outcome"),
                    "winner": value.get("winner"), "status": value.get("status"),
                    "armies": [len(a) for a in config.get("armies", [])],
                    **self.provenance(key, path, target),
                })
            except (OSError, ValueError, TypeError, AttributeError):
                continue  # A just-written result may not yet be complete.
        return sorted(result, key=lambda row: row["id"])

    def status(self):
        try:
            value = read_json(self.root / "summary.json")
            summary = value["summary"]
            replacement = value.get("greedy_replacement")
            target = self.refresh_target()
            if target and (not replacement or replacement.get("run_signature") != target["run_signature"]):
                replacement = {"cases": 0, "expected": len(target["cases"]),
                               "run_signature": target["run_signature"], "outcomes": {}}
            return {"complete": value.get("schema") == 1,
                    "acceptance": bool(value.get("acceptance")),
                    "checked": summary["matches"], "failures": len(value.get("failures", [])),
                    "greedy": (replacement["outcomes"] if replacement else
                               summary.get("outcomes", {}).get("opponent/greedy", {}).get("outcomes", {})),
                    "replacement": replacement, "target_port": target.get("port") if target else None,
                    "mixed_versions": bool(value.get("mixed_versions"))}
        except (OSError, KeyError, ValueError, TypeError):
            return None  # A live summary may be between writes.

    def load(self, key):
        with self.lock:
            path = self.paths().get(key)
            if path is None:
                raise FileNotFoundError("Матч не найден")
            stamp = path.stat()
            result_path = path.with_name("result.json")
            result_stamp = result_path.stat().st_mtime_ns if result_path.exists() else None
            provenance = self.provenance(key, path, self.refresh_target())
            signature = (stamp.st_mtime_ns, stamp.st_size, result_stamp,
                         provenance["revision"], provenance["refresh_status"])
            cached = self.cache.get(key)
            if cached and cached[0] == signature:
                self.cache.move_to_end(key)
                return cached[1]
            payload = prepare_replay(path)
            payload = payload | {"provenance": provenance}
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.cache[key] = (signature, encoded)
            self.cache.move_to_end(key)
            while len(self.cache) > 3:
                self.cache.popitem(last=False)
            return encoded


def create_server(root, port=8765):
    store = ReplayStore(root)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            try:
                if request.path in ("/", "/index.html"):
                    self.respond(200, TEMPLATE.read_bytes(), "text/html; charset=utf-8")
                elif request.path == "/api/matches":
                    self.respond(200, json.dumps(store.catalog(), ensure_ascii=False).encode("utf-8"))
                elif request.path == "/api/status":
                    self.respond(200, json.dumps(store.status(), ensure_ascii=False).encode("utf-8"))
                elif request.path == "/api/replay":
                    key = parse_qs(request.query).get("id", [""])[0]
                    self.respond(200, store.load(key))
                elif request.path == "/favicon.ico":
                    self.respond(204, b"", "image/x-icon")
                else:
                    self.respond(404, b'{"error":"Not found"}')
            except FileNotFoundError as exc:
                self.problem(HTTPStatus.NOT_FOUND, str(exc))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                self.problem(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))

        def problem(self, status, message):
            self.respond(status, json.dumps({"error": message}, ensure_ascii=False).encode("utf-8"))

        def respond(self, status, body, content_type="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Match directory, evaluation directory, or replay.json")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with create_server(args.root, args.port) as server:
        print(f"Combat viewer: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
