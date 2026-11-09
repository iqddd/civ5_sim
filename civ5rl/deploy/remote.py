"""Stage gitignored experiment inputs and inspect live remote runs over SSH."""

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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


def _more_data_bc_inputs(root, cfg):
    names = []
    eval_ref = cfg.get("evaluation_reference")
    if eval_ref:
        names.append(eval_ref)
    old_run = cfg.get("old_corpus_run")
    if old_run:
        old_root = root / old_run
        for meta in (
            "collection.json",
            "manifest.json",
            "run.json",
            "resolved-config.json",
            "evaluations.json",
            "selection.json",
            "confirmation.json",
        ):
            if (old_root / meta).is_file():
                names.append(f"{old_run}/{meta}")
        collection = _read(old_root / "collection.json")
        for row in collection.get("matches", []):
            match_id = int(row["match_id"])
            prefix = f"{old_run}/data/match-{match_id:06d}"
            for fname in (
                "case.json",
                "collection-metrics.json",
                "demonstration-side-0.json.gz",
                "demonstration-side-1.json.gz",
            ):
                names.append(f"{prefix}/{fname}")
    ref_run = cfg.get("reference_run")
    if ref_run:
        ref_root = root / ref_run
        for meta in (
            "run.json",
            "manifest.json",
            "evaluations.json",
            "selected.json",
            "confirmation.json",
        ):
            if (ref_root / meta).is_file():
                names.append(f"{ref_run}/{meta}")
        bc_prefix = (cfg.get("parent_policy_hash") or "")[:16]
        if bc_prefix and (ref_root / "evaluation/confirmation" / bc_prefix).is_dir():
            for path in sorted(
                (ref_root / "evaluation/confirmation" / bc_prefix).rglob("*")
            ):
                if path.is_file() and path.name in ("case.json", "metrics.json"):
                    names.append(path.relative_to(root).as_posix())
    return names


def _c2_bc_inputs(root, cfg):
    names = _more_data_bc_inputs(root, cfg)
    bc10 = cfg.get("bc10_checkpoint")
    if bc10:
        names.append(bc10)
    more_run = cfg.get("more_data_corpus_run")
    if more_run:
        more_root = root / more_run
        for meta in (
            "collection.json",
            "combined-corpus-manifest.json",
            "manifest.json",
            "run.json",
            "resolved-config.json",
            "evaluations.json",
            "selection.json",
            "confirmation.json",
        ):
            if (more_root / meta).is_file():
                names.append(f"{more_run}/{meta}")
        collection = _read(more_root / "collection.json")
        for row in collection.get("new_matches", []):
            match_id = int(row["match_id"])
            prefix = f"{more_run}/data/match-{match_id:06d}"
            for fname in (
                "case.json",
                "collection-metrics.json",
                "demonstration-side-0.json.gz",
                "demonstration-side-1.json.gz",
            ):
                names.append(f"{prefix}/{fname}")
    return names


def input_group(rel_posix, output_posix):
    """Minimal off-spine directory prefix: pairwise disjoint and outside output."""
    parts = PurePosixPath(rel_posix).parts
    out_parts = PurePosixPath(output_posix).parts
    if len(parts) < 2:
        raise ValueError(f"Gitignored input must reside in a subdirectory: {rel_posix}")
    if parts[: len(out_parts)] == out_parts:
        raise ValueError(
            f"Input overlaps experiment output {output_posix}: {rel_posix}"
        )
    for depth in range(1, len(parts)):
        if parts[:depth] != out_parts[:depth]:
            return "/".join(parts[:depth])
    raise ValueError(
        f"Input sits directly in an ancestor directory of {output_posix}: {rel_posix}"
    )


def _git_tracked_files(root):
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return {
        PurePosixPath(os.fsdecode(item).replace("\\", "/")).as_posix()
        for item in raw.split(b"\0")
        if item
    }


def stage_inputs(
    config,
    inputs_dir,
    *,
    root=ROOT,
    manifest=None,
    parent=None,
    extra_inputs=(),
):
    """Stage gitignored read-only inputs into disjoint trees for prepare.json sync."""
    root = Path(root).resolve()
    inputs_dir = Path(inputs_dir).resolve()
    config_name = _relative_file(root, config)
    raw, config_files = _resolve_config(root, config_name)
    cfg = raw.get("config", raw)
    output_posix = PurePosixPath(str(cfg["output"]).replace("\\", "/")).as_posix()
    experiment = cfg.get("experiment", {})
    parent = parent or cfg.get("parent") or experiment.get("parent")
    manifest = (
        manifest or cfg.get("target_manifest") or experiment.get("parent_manifest")
    )
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
    if cfg.get("kind") == "rl01.cp_imitation_target_more_data":
        names.extend(_more_data_bc_inputs(root, cfg))
    if cfg.get("kind") == "rl01.cp_imitation_target_c2":
        names.extend(_c2_bc_inputs(root, cfg))
    if (
        cfg.get("kind")
        in (
            "rl01.ppo_target_scale",
            "rl01.cp_imitation_target_scale",
            "rl01.cp_imitation_target_c2",
        )
        and (root / "artifacts/rl/corpus/manifest.json").is_file()
    ):
        names.append("artifacts/rl/corpus/manifest.json")
        names.extend(_open_maps(_read(root / "artifacts/rl/corpus/manifest.json")))
    names.extend(_open_maps(_read(root / _relative_file(root, manifest))))

    resolved_inputs = sorted({_relative_file(root, name) for name in names})
    tracked = _git_tracked_files(root)
    tracked_inputs = {}
    staged_files = {}
    groups = set()
    for name in resolved_inputs:
        if name in tracked:
            tracked_inputs[name] = _sha(root / name)
            continue
        group = input_group(name, output_posix)
        if any(t == group or t.startswith(group + "/") for t in tracked):
            raise ValueError(
                f"Input sync group {group!r} overlaps git-tracked files; "
                "commit the file or place gitignored inputs in an ignored directory"
            )
        groups.add(group)
        src = root / name
        dst = inputs_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = _sha(src)
        if _sha(dst) != digest:
            raise ValueError(f"Staged input hash mismatch: {name}")
        staged_files[name] = digest

    return {
        "schema": 1,
        "output": output_posix,
        "groups": sorted(groups),
        "staged_files": staged_files,
        "tracked_inputs": tracked_inputs,
    }


_INSPECT_CODE = r"""
import json, pathlib, sys
p=json.load(sys.stdin); root=pathlib.Path(p['directory']); limit=p['tail']
def read(name):
 q=root/name
 if not q.is_file(): return None
 if q.stat().st_size > 256*1024*1024: return {'error':'file exceeds 256 MiB'}
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
 state={k:state.get(k) for k in ('rollouts','updates','decisions','optimizer_steps','attempted_steps','rejected_steps','active_seconds','evaluation_seconds','phase','stop_reason','baseline_wins','latest','best','development_points','rollout_summary','critic','epoch','epoch_completed','batch_cursor','Adam_steps','free_labels_processed','epochs_history') if k in state}
collection=read('collection.json')
if isinstance(collection,dict):
 collection={k:collection.get(k) for k in ('status','stop_reason','labels','new_labels','old_labels','positions','new_positions','old_positions','seconds','waves','completed_waves','eta_seconds','physical_repeats','new_matches_count','old_matches_count') if k in collection}
training=read('training.json')
if isinstance(training,dict):
 training={k:training.get(k) for k in ('status','stop_reason','epochs','Adam_steps','free_labels_processed','active_seconds','latest','microbatch','value_head_unchanged','GPU_utilization_review') if k in training}
resources=tail('telemetry/resources.jsonl') or tail('training-telemetry/resources.jsonl')
print(json.dumps({'status':read('status.json'),'state':state,'collection':collection,'training':training,'updates_tail':tail('updates.jsonl'),'steps_tail':tail('steps.jsonl'),'resources_tail':resources,'selected':read('selected.json') or read('selection.json'),'evaluations':read('evaluations.json'),'development':read('development.json'),'confirmation':read('confirmation.json'),'performance':read('performance.json')}))
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
