"""Remote execution entrypoint inside the prepared repository worktree."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from civ5rl.deploy.remote import _read, _resolve_config, _write
from tools.rl_collection_hardware import hardware

BC_KINDS = {
    "rl01.cp_imitation_target_more_data",
    "rl01.cp_imitation_target_c2",
    "rl01.cp_imitation_target_scale",
}


def ensure_native_libraries(root=ROOT):
    if sys.platform != "linux":
        return
    if not (root / "build/libcombat.so").is_file():
        subprocess.run(
            [sys.executable, "tools/build_combat.py", "--dll-only"],
            cwd=root,
            check=True,
        )
    if not (root / "build/libcombat_ai.so").is_file():
        subprocess.run(
            [sys.executable, "tools/build_combat_ai.py", "--dll-only"],
            cwd=root,
            check=True,
        )


def main(argv=None, *, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tune", "run", "resume"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tuning-rollout")
    parser.add_argument("--tuning-dir")
    parser.add_argument("--runtime-config")
    parser.add_argument("--transfer-dir")
    parser.add_argument("--transfer-token")
    parser.add_argument("--transfer-destination")
    parser.add_argument("--input-transfer", action="append", default=[])
    parser.add_argument("--evaluation-device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--evaluation-workers", type=int)
    parser.add_argument("--evaluation-max-batch", type=int)
    parser.add_argument("--evaluation-wait-seconds", type=float)
    args = parser.parse_args(argv)

    root = Path(root).resolve()
    if args.transfer_dir:
        from civ5rl.deploy.archive import seven_zip, unpack

        # Fail before training if the machine cannot package the eventual JSON output.
        seven_zip()
        for raw_transfer in args.input_transfer:
            item = json.loads(raw_transfer)
            unpack(item["packet"], item["destination"], token=item["token"], direction="upload")
        unpack(args.transfer_dir, args.transfer_destination,
               token=args.transfer_token, direction="upload")
        _write(Path(args.transfer_dir) / "applied.json", {"token": args.transfer_token})
    ensure_native_libraries(root)
    raw, _ = _resolve_config(root, args.config)
    cfg = raw.get("config", raw)
    kind = cfg["kind"]
    output = root / cfg["output"]
    has_progress = any(
        (output / name).exists() for name in ("state.json", "collection.json", "run.json")
    )

    if args.mode == "tune":
        if has_progress:
            raise ValueError("Cannot retune an existing scientific run")
        if not args.tuning_rollout or not args.tuning_dir:
            raise ValueError("--tuning-rollout and --tuning-dir are required for tune")
        tuning_dir = Path(args.tuning_dir)
        parent = cfg.get("parent") or cfg.get("experiment", {}).get("parent")
        command = [
            sys.executable,
            "tools/tune_rl_collection.py",
            "--config",
            args.config,
            "--checkpoint",
            parent,
            "--rollout",
            args.tuning_rollout,
            "--out",
            str(tuning_dir),
            "--budget-seconds",
            "600",
        ]
        code = subprocess.run(command, cwd=root, check=False).returncode
        tuning_dir.mkdir(parents=True, exist_ok=True)
        if not (tuning_dir / "machine.json").exists():
            _write(tuning_dir / "machine.json", {"hardware": hardware()})
        return code
    else:
        if args.mode == "run" and has_progress:
            raise ValueError("Run already exists; use resume")
        if args.mode == "resume":
            if kind in BC_KINDS and not has_progress:
                raise ValueError("Resume requires existing BC run artifacts")
            if kind not in BC_KINDS and not (output / "state.json").exists():
                raise ValueError("Resume requires existing state.json")
        output.mkdir(parents=True, exist_ok=True)
        if not (output / "machine.json").exists():
            _write(output / "machine.json", {"hardware": hardware()})
        runner_script = (
            "tools/run_cp_imitation.py" if kind in BC_KINDS else "tools/run_ppo.py"
        )
        command = [
            sys.executable,
            runner_script,
            args.mode,
            "--config",
            args.config,
        ]
        for name in ("device", "workers", "max_batch", "wait_seconds"):
            value = getattr(args, "evaluation_" + name)
            if value is not None:
                command += ["--evaluation-" + name.replace("_", "-"), str(value)]
        if kind == "rl01.ppo_after_target_bc":
            if not args.runtime_config:
                raise ValueError("--runtime-config is required for ppo_after_target_bc")
            summary_path = Path(args.runtime_config).parent / "summary.json"
            summary = _read(summary_path)
            if (
                summary.get("preliminary") is not False
                or summary.get("protected_inputs_unchanged") is not True
            ):
                raise ValueError("No confirmed comparable tuning for this experiment")
            command += ["--runtime-config", args.runtime_config]

    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
