"""Incremental artifact reader used by progress adapters and process recorders."""

import json
from pathlib import Path


class ArtifactSource:
    def __init__(self, directory, config=None):
        self.directory = Path(directory)
        self.config_override = config
        self._json = {}
        self._logs = {}

    def read(self, name, default=None):
        path = self.directory / name
        try:
            stat = path.stat()
        except FileNotFoundError:
            return default
        signature = (stat.st_mtime_ns, stat.st_size)
        old = self._json.get(name)
        if old is None or old[0] != signature:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            self._json[name] = signature, value
        return self._json[name][1]

    def rows(self, name):
        path = self.directory / name
        if not path.exists():
            return []
        stat = path.stat()
        old = self._logs.get(name)
        # Also reset on replacement or a same-size rewrite. Incomplete trailing
        # records are retried next time; malformed committed records are errors.
        if old is None or old[0] != stat.st_ino or stat.st_size < old[1] or (stat.st_size == old[1] and stat.st_mtime_ns != old[2]):
            old = (stat.st_ino, 0, stat.st_mtime_ns, [])
        inode, offset, _, rows = old
        with path.open("rb") as stream:
            stream.seek(offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break
                if line.strip():
                    rows.append(json.loads(line))
                offset = stream.tell()
                self._logs[name] = inode, offset, stat.st_mtime_ns, rows
        self._logs[name] = inode, offset, stat.st_mtime_ns, rows
        return rows

    @property
    def config(self):
        return self.config_override or self.read("resolved-config.json") or self.read("run.json", {}).get("config", {})

    @property
    def session(self):
        return self.read("derived/progress/session.json", {})

    def last(self, name):
        path = self.directory / name
        if not path.exists():
            return {}
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            lines = stream.read().split(b"\n")
        # Writers terminate every committed row with a newline. The final
        # unterminated fragment is not a phase event yet.
        return json.loads(lines[-2]) if len(lines) >= 2 and lines[-2].strip() else {}
