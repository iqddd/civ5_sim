"""Remote execution entrypoint inside the prepared repository worktree."""

# ruff: noqa: E402
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from civ5rl.deploy.remote import _read, _resolve_config, _write
from civ5rl.engine.runtime.hardware import hardware

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
    parser.add_argument("--mode", choices=("tune", "run", "resume", "profile"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tuning-rollout")
    parser.add_argument("--tuning-source")
    parser.add_argument("--tuning-dir")
    parser.add_argument("--runtime-config")
    parser.add_argument("--transfer-dir")
    parser.add_argument("--transfer-token")
    parser.add_argument("--transfer-destination")
    parser.add_argument("--final-run-bundle", action="store_true")
    parser.add_argument("--input-transfer", action="append", default=[])
    parser.add_argument("--evaluation-device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--evaluation-workers", type=int)
    parser.add_argument("--evaluation-max-batch", type=int)
    parser.add_argument("--evaluation-wait-seconds", type=float)
    parser.add_argument("--host-resume-id")
    parser.add_argument("--host-resume-reason")
    parser.add_argument("--continue-after-dual-target", action="store_true")
    parser.add_argument("--diagnostic-cpv2", action="store_true")
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
    if args.continue_after_dual_target:
        from civ5rl.provenance.common import atomic_output, file_sha256

        for name, checksum in _read(output / "execution.json")["binaries"].items():
            if name not in {"build/libcombat.so", "build/libcombat_ai.so", "build/combat_rules.json"}:
                raise ValueError(f"Incompatible saved binary for Linux continuation: {name}")
            source = output / "execution-binaries" / Path(name).name
            if file_sha256(source) != checksum:
                raise ValueError(f"Saved binary differs from execution.json: {name}")
            with atomic_output(root / name) as temporary:
                shutil.copyfile(source, temporary)
    has_progress = any(
        (output / name).exists() for name in ("state.json", "collection.json", "run.json")
    )
    if args.host_resume_reason:
        if args.mode != "resume" or kind != "rl01.ppo_after_target_bc":
            raise ValueError("Host resume revision is supported only for paused after-target-BC PPO")
        from civ5rl.reporting.host_resume import record_host_resume
        revision = record_host_resume(output, args.host_resume_id, args.host_resume_reason)
        print(json.dumps({"host_resume": revision}), flush=True)

    if args.mode == "profile":
        if kind != "rl01.ppo_after_target_bc" or not args.tuning_dir or not args.tuning_rollout:
            raise ValueError("Self-play profiling requires the PPO config, output directory and case carrier")
        command = [
            sys.executable, "tools/bench_rl.py", "selfplay-profile", "--config", args.config,
            "--checkpoint", cfg["experiment"]["parent"], "--rollout", args.tuning_rollout,
            "--out", args.tuning_dir, "--episode-limit", "150", "--repeats", "2",
        ]
        if cfg["experiment"].get("three_opponent_evaluation"):
            command[command.index("--episode-limit") + 1] = "200"
            if args.runtime_config:
                runtime = _read(args.runtime_config)
                command += ["--workers", str(runtime["cpu_workers"]), "--envs", str(runtime["num_envs"]),
                            "--reply-size", str(runtime["worker_reply_size"])]
    elif args.mode == "tune":
        if has_progress:
            raise ValueError("Cannot retune an existing scientific run")
        if not (args.tuning_rollout or args.tuning_source) or not args.tuning_dir:
            raise ValueError("--tuning-rollout and --tuning-dir are required for tune")
        tuning_dir = Path(args.tuning_dir)
        if kind == "rl01.rollout_comparison_collection":
            command = [sys.executable, "tools/bench_rl.py", "comparison-collection",
                       "--config", args.config, "--run", args.tuning_source,
                       "--out", str(tuning_dir), "--worker-counts", "8,16,32,48",
                       "--forks", "64", "--seconds", "300", "--barrier-workers", "0"]
            return subprocess.run(command, cwd=root, check=False).returncode
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
        comparison_kinds = {
            "rl01.rollout_comparison",
            "rl01.rollout_comparison_collection",
            "rl01.turn_comparison_fit",
        }
        if args.mode == "resume":
            if kind in comparison_kinds and not has_progress:
                raise ValueError("Resume requires existing comparison artifacts")
            if kind in BC_KINDS and not has_progress:
                raise ValueError("Resume requires existing BC run artifacts")
            if kind not in BC_KINDS | comparison_kinds and not (output / "state.json").exists():
                raise ValueError("Resume requires existing state.json")
        output.mkdir(parents=True, exist_ok=True)
        if not (output / "machine.json").exists():
            _write(output / "machine.json", {"hardware": hardware()})
        runner_script = (
            "tools/run_comparison.py" if kind in comparison_kinds else
            "tools/run_cp_imitation.py" if kind in BC_KINDS else "tools/run_ppo.py"
        )
        command = [
            sys.executable,
            runner_script,
            "resume" if kind == "rl01.rollout_comparison" else
            "collect" if kind == "rl01.rollout_comparison_collection" and args.mode == "run" else
            "train" if kind == "rl01.turn_comparison_fit" and args.mode == "run" else args.mode,
            "--config",
            args.config,
        ]
        for name in ("device", "workers", "max_batch", "wait_seconds"):
            value = getattr(args, "evaluation_" + name)
            if value is not None:
                command += ["--evaluation-" + name.replace("_", "-"), str(value)]
        if kind in {"rl01.ppo_after_target_bc", "rl01.rollout_comparison_collection"}:
            if not args.runtime_config:
                raise ValueError("A measured --runtime-config is required")
            summary_path = Path(args.runtime_config).parent / "summary.json"
            summary = _read(summary_path)
            if (
                summary.get("preliminary") is not False
                or summary.get("protected_inputs_unchanged") is not True
            ):
                raise ValueError("No confirmed comparable tuning for this experiment")
            command += ["--runtime-config", args.runtime_config]
        if args.continue_after_dual_target:
            command.append("--continue-after-dual-target")
        if args.diagnostic_cpv2:
            command += ["--diagnostic-cpv2", "--max-updates", "3"]

    code = subprocess.run(command, cwd=root, check=False).returncode
    if code == 0 and args.final_run_bundle and args.mode in ("run", "resume"):
        from civ5rl.deploy.final_run import publish_final_run

        publish_final_run(output, args.transfer_dir, args.transfer_token)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
