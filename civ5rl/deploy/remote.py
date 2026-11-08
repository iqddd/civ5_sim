"""Package and verify bounded remote experiment artifacts; no job supervision."""

import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _relative_file(root, name):
    root = Path(root).resolve()
    path = Path(name)
    path = path if path.is_absolute() else root / path
    if not path.is_relative_to(root):
        raise ValueError(f"Input outside project root: {name}")
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        if part in ("..", "", "."):
            raise ValueError(f"Unsafe input path: {name}")
        current /= part
        if (
            current.is_symlink()
            or getattr(current.lstat(), "st_file_attributes", 0) & 0x400
        ):
            raise ValueError(f"Symlink/reparse input: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Missing or unsafe input: {name}")
    return relative.as_posix()


def _open_maps(node):
    if isinstance(node, dict):
        if "path" in node and "split" in node:
            split = str(node["split"]).lower()
            if split in ("test", "closed_test", "closed-test"):
                return
            if split in (
                "train",
                "training",
                "validation",
                "holdout",
                "dev",
                "development",
            ):
                yield node["path"]
            else:
                raise ValueError(f"Unknown map split: {split}")
        for value in node.values():
            yield from _open_maps(value)
    elif isinstance(node, list):
        for value in node:
            yield from _open_maps(value)


def _resolve_config(root, name, stack=()):
    name = _relative_file(root, name)
    if name in stack:
        raise ValueError("Config inheritance cycle")
    value = _read(root / name)
    base = value.get("base_config")
    if not base:
        return value, [name]
    inherited, files = _resolve_config(root, base, (*stack, name))

    def merge(dst, src):
        for key, item in src.items():
            if key == "base_config":
                continue
            if isinstance(item, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], item)
            else:
                dst[key] = item

    merge(inherited, value)
    return inherited, [*files, name]


def package_capsule(
    config, out_dir, *, root=ROOT, manifest=None, parent=None, extra_inputs=()
):
    root = Path(root).resolve()
    config_name = _relative_file(root, config)
    raw, config_files = _resolve_config(root, config_name)
    cfg = raw.get("config", raw)
    experiment = cfg.get("experiment", {})
    parent = parent or experiment.get("parent")
    manifest = manifest or experiment.get("parent_manifest")
    if not parent or not manifest:
        raise ValueError(
            "Specify parent and manifest, or set experiment.parent and parent_manifest"
        )
    names = [
        *config_files,
        parent,
        manifest,
        *extra_inputs,
        "build/combat_rules.h",
        "build/combat_rules.json",
        "build/cp_geometry.inc",
    ]
    if experiment.get("parent_manifest"):
        names.append(experiment["parent_manifest"])
    corpus = experiment.get("corpus")
    if corpus:
        names.append(corpus)
    names.extend(_open_maps(_read(root / _relative_file(root, manifest))))
    inputs = sorted({_relative_file(root, name) for name in names})
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    )
    sources = sorted(
        {
            _relative_file(root, os.fsdecode(name))
            for name in tracked.split(b"\0")
            if name and (root / os.fsdecode(name)).is_file()
        }
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    archives = {}
    for filename, paths in (("source.zip", sources), ("inputs.zip", inputs)):
        dest = out / filename
        members = {}
        with zipfile.ZipFile(
            dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name in paths:
                members[name] = _sha(root / name)
                archive.write(root / name, name)
        archives[filename] = {
            "bytes": dest.stat().st_size,
            "sha256": _sha(dest),
            "members": members,
        }
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    transfer = {
        "schema": 1,
        "commit": commit,
        "archives": archives,
        "replays": [
            n
            for n in inputs
            if Path(n).name.startswith("replay-") and n.endswith(".json")
        ],
    }
    _write(out / "transfer.json", transfer)
    return {
        "commit": commit,
        "archives": {
            k: {"bytes": v["bytes"], "sha256": v["sha256"], "files": len(v["members"])}
            for k, v in archives.items()
        },
    }


def _members(archive):
    seen = set()
    folded = set()
    components = {}
    for info in archive.infolist():
        name = info.filename
        p = PurePosixPath(name)
        mode = info.external_attr >> 16
        if (
            not name
            or "\\" in name
            or p.is_absolute()
            or re.match(r"^[A-Za-z]:", name)
            or any(part in ("..", ".") for part in name.split("/"))
            or name in seen
            or name.casefold().rstrip("/") in folded
            or stat.S_ISLNK(mode)
        ):
            raise ValueError(f"Unsafe or duplicate ZIP member: {name!r}")
        for part_index in range(1, len(p.parts) + 1):
            prefix = "/".join(p.parts[:part_index])
            previous = components.setdefault(prefix.casefold(), prefix)
            if previous != prefix:
                raise ValueError(f"Case-colliding ZIP paths: {previous!r}, {prefix!r}")
        seen.add(name)
        folded.add(name.casefold().rstrip("/"))
        if not info.is_dir():
            yield info


def _validate_zip(path, expected=None):
    files = {}
    with zipfile.ZipFile(path) as archive:
        for info in _members(archive):
            h = hashlib.sha256()
            with archive.open(info) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            files[info.filename] = h.hexdigest()
    if expected is not None and files != expected:
        raise ValueError(f"ZIP member hashes differ: {path}")
    return files


def safe_extract(archive_path, destination, *, expected=None, overwrite=False):
    archive_path, dest_path = Path(archive_path), Path(destination).absolute()
    for ancestor in (dest_path, *dest_path.parents):
        if ancestor.is_symlink() or (
            ancestor.exists()
            and getattr(ancestor.lstat(), "st_file_attributes", 0) & 0x400
        ):
            raise ValueError(f"Symlink/reparse destination: {ancestor}")
    dest = dest_path.resolve()
    files = _validate_zip(archive_path, expected)
    # Reject portable file/directory collisions before any output is created.
    folded_files = {name.casefold() for name in files}
    for name in files:
        if any(
            str(parent).casefold() in folded_files
            for parent in PurePosixPath(name).parents
            if str(parent) != "."
        ):
            raise ValueError(f"ZIP file is also a parent directory: {name}")
    # All paths, symlinks and conflicts are checked before any output is created.
    for name, digest in files.items():
        path = dest.joinpath(*PurePosixPath(name).parts)
        for ancestor in (path, *path.parents):
            if ancestor == dest.parent:
                break
            if ancestor.is_symlink() or (
                ancestor.exists()
                and getattr(ancestor.lstat(), "st_file_attributes", 0) & 0x400
            ):
                raise ValueError(f"Symlink/reparse destination: {ancestor}")
        if not path.resolve().is_relative_to(dest):
            raise ValueError(f"Destination outside root: {name}")
        if path.exists() and (
            not path.is_file() or (_sha(path) != digest and not overwrite)
        ):
            raise ValueError(f"Conflicting destination: {path}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for name, digest in files.items():
            path = dest.joinpath(*PurePosixPath(name).parts)
            if path.is_file() and _sha(path) == digest:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, path.open("wb") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
            if _sha(path) != digest:
                raise ValueError(f"Extracted file hash mismatch: {path}")
    return files


def bootstrap_linux_node(capsule_dir, project_dir):
    if sys.platform != "linux":
        raise RuntimeError("Bootstrap compilation requires Linux")
    capsule, project = Path(capsule_dir), Path(project_dir)
    transfer = _read(capsule / "transfer.json")
    for name in ("source.zip", "inputs.zip"):
        path, meta = capsule / name, transfer["archives"][name]
        if path.stat().st_size != meta["bytes"] or _sha(path) != meta["sha256"]:
            raise ValueError(f"Archive checksum/size differs: {name}")
        _validate_zip(path, meta["members"])
    # Build and replay in an isolated staging project before changing the destination.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "project"
        for name in ("source.zip", "inputs.zip"):
            safe_extract(
                capsule / name, stage, expected=transfer["archives"][name]["members"]
            )
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O3",
                "-fPIC",
                "-shared",
                "-I",
                "build",
                "native/combat.cpp",
                "native/combat_state.cpp",
                "native/geometry.cpp",
                "-o",
                "build/libcombat.so",
            ],
            cwd=stage,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [sys.executable, "tools/build_combat_ai.py", "--dll-only"],
            cwd=stage,
            check=True,
            capture_output=True,
        )
        replay_code = r"""
import json, sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from civ5sim import Battle
from civ5sim.engine import digest
for name in json.load(sys.stdin):
    replay = json.loads(Path(name).read_text())
    with Battle(replay['initial']) as battle:
        for step in replay['steps']:
            if asdict(battle.step(step['action'])) != step['result'] or digest(battle.snapshot()) != step['state_sha256']:
                raise ValueError('Replay diverged: ' + name)
"""
        subprocess.run(
            [sys.executable, "-c", replay_code],
            cwd=stage,
            input=json.dumps(transfer.get("replays", [])),
            text=True,
            check=True,
            capture_output=True,
        )
        merged = Path(tmp) / "merged.zip"
        with zipfile.ZipFile(merged, "w") as z:
            for path in sorted(stage.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    z.write(path, path.relative_to(stage).as_posix())
        safe_extract(merged, project)
    from tools.rl_collection_hardware import hardware

    return {
        "hardware": hardware(),
        "replays": transfer.get("replays", []),
        "determinism_verified": bool(transfer.get("replays")),
    }


_INSPECT_CODE = r"""
import json, pathlib, sys
p=json.load(sys.stdin); root=pathlib.Path(p['directory']); limit=p['tail']
def read(name):
 q=root/name
 if not q.is_file(): return None
 if q.stat().st_size > 4*1024*1024: return {'error':'file exceeds 4 MiB'}
 try: return json.loads(q.read_text())
 except (ValueError, OSError): return {'error':'unreadable JSON'}
def tail(name):
 q=root/name
 if not q.is_file(): return []
 with q.open('rb') as f:
  size=q.stat().st_size; f.seek(max(0,size-1024*1024)); data=f.read()
 lines=data.splitlines()[-limit:]; result=[]
 for line in lines:
  try: value=json.loads(line)
  except ValueError: result.append({'error':'partial, oversized or invalid record'}); continue
  if name == 'updates.jsonl':
   keys=('rollout','update','decisions','total_decisions','active_seconds','W','D','L','collection')
   value={k:value.get(k) for k in keys if k in value} | {'ppo':{k:value.get('ppo',{}).get(k) for k in ('optimizer_steps','attempted_optimizer_steps','rejected_optimizer_steps','final_accepted_kl','stop_reason','ppo_seconds','backward_seconds','kl_seconds')}}
  result.append(value)
 return result
state=read('state.json')
if isinstance(state,dict):
 state={k:state.get(k) for k in ('rollouts','updates','decisions','optimizer_steps','attempted_steps','rejected_steps','active_seconds','evaluation_seconds','phase','stop_reason','baseline_wins','latest','best','development_points','rollout_summary','critic') if k in state}
print(json.dumps({'status':read('status.json'),'state':state,'updates_tail':tail('updates.jsonl'),'resources_tail':tail('telemetry/resources.jsonl'),'selected':read('selected.json'),'evaluations':read('evaluations.json'),'development':read('development.json'),'confirmation':read('confirmation.json'),'performance':read('performance.json')}))
"""
_FINGERPRINT_CODE = r"""
import hashlib,json,pathlib,sys
p=pathlib.Path(json.load(sys.stdin)['path']); h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''): h.update(b)
print(json.dumps({'bytes':p.stat().st_size,'sha256':h.hexdigest()}))
"""


def _connection(instance_json):
    instance = instance_json if isinstance(instance_json, dict) else _read(instance_json)
    raw = os.environ.get("ASSISTANT_REMOTE_CONN", "").strip()
    if raw.startswith("{"):
        connection = json.loads(raw)
    elif raw:
        parts = raw.split("|", 4)
        if len(parts) < 4:
            raise ValueError("Invalid ASSISTANT_REMOTE_CONN")
        connection = dict(
            zip(("host", "port", "username", "private_key"), parts[:4], strict=True)
        )
    else:
        connection = {}
    host = (
        instance.get("host")
        or instance.get("SSH_host")
        or connection.get("host")
        or os.environ.get("ASSISTANT_REMOTE_HOST")
    )
    port = (
        instance.get("port")
        or instance.get("SSH_port")
        or connection.get("port")
        or os.environ.get("ASSISTANT_REMOTE_PORT")
        or 22
    )
    user = connection.get("username") or os.environ.get("ASSISTANT_REMOTE_USERNAME")
    key = connection.get("private_key") or os.environ.get(
        "ASSISTANT_REMOTE_PRIVATE_KEY"
    )
    if (
        not host
        or not user
        or not key
        or not re.fullmatch(r"[A-Za-z0-9_.:-]+", str(host))
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(user))
    ):
        raise ValueError(
            "Missing or invalid SSH host, username or private key configuration"
        )
    port = int(port)
    if not 1 <= port <= 65535 or not Path(key).expanduser().is_file():
        raise ValueError("Invalid SSH port or private key")
    return str(host), port, str(user), str(Path(key).expanduser())


def _remote_path(path):
    if (
        not path.startswith("/")
        or "\x00" in path
        or "\n" in path
        or "\r" in path
        or ".." in PurePosixPath(path).parts
    ):
        raise ValueError("Remote path must be an absolute, normalized Linux path")
    return path


def _ssh_json(connection, code, payload):
    host, port, user, key = connection
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-p",
        str(port),
        "-i",
        key,
        f"{user}@{host}",
        "python3 -c " + shlex.quote(code),
    ]
    result = subprocess.run(
        command,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"SSH operation failed (exit {result.returncode})")
    return json.loads(result.stdout)


def inspect_remote(instance_json, remote_run_dir, *, tail=3):
    if not 1 <= tail <= 20:
        raise ValueError("tail must be between 1 and 20")
    return _ssh_json(
        _connection(instance_json),
        _INSPECT_CODE,
        {"directory": _remote_path(remote_run_dir), "tail": tail},
    )


def finalize_remote(
    instance_json, remote_archive, output, *, overwrite=False, adam_offset=None
):
    connection = _connection(instance_json)
    remote_archive = _remote_path(remote_archive)
    before = _ssh_json(connection, _FINGERPRINT_CODE, {"path": remote_archive})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    partial = output.parent / (output.name + ".download.partial")
    host, port, user, key = connection
    command = [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-P",
        str(port),
        "-i",
        key,
        f"{user}@{host}:{shlex.quote(remote_archive)}",
        str(partial),
    ]
    result = subprocess.run(command, capture_output=True, timeout=3600, check=False)
    if result.returncode:
        raise RuntimeError(
            f"SCP failed (exit {result.returncode}); partial download retained"
        )
    after = _ssh_json(connection, _FINGERPRINT_CODE, {"path": remote_archive})
    if (
        before != after
        or partial.stat().st_size != before["bytes"]
        or _sha(partial) != before["sha256"]
    ):
        raise ValueError(
            "Archive changed during transfer or local checksum/size differs"
        )
    members = _validate_zip(partial)
    if "transfer-receipt.json" in members:
        raise ValueError("Archive contains reserved transfer receipt")
    receipt_path = output / "transfer-receipt.json"
    if receipt_path.exists() and not overwrite:
        previous = _read(receipt_path)
        if previous.get("fingerprint") != before or previous.get("members") != members:
            raise ValueError("Conflicting transfer receipt")
    safe_extract(partial, output, expected=members, overwrite=overwrite)
    audit = None
    if adam_offset is not None:
        from civ5rl.provenance.audit import verify_checkpoint_chain

        audit = verify_checkpoint_chain(output, adam_offset=adam_offset)
    receipt = {
        "archive": remote_archive,
        "fingerprint": before,
        "members": members,
        "checkpoint_audit": audit,
    }
    _write(receipt_path, receipt)
    partial.unlink()
    return {
        "fingerprint": before,
        "files": len(members),
        "checkpoint_audit": audit,
        "receipt": str(receipt_path),
    }
