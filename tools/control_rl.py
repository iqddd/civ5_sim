"""Pause, inspect or resume a background RL training run on Windows/Linux."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve_directory(directory):
    directory = Path(directory).resolve()
    legacy = ROOT / "artifacts/rl/performance-20261104"
    if not directory.exists() and directory in {
        legacy / "pilot",
        legacy / "pilot-v14",
        legacy / "pilot-v15-stable",
    }:
        directory = ROOT / "artifacts/rl/training-result"
    visited = set()
    while (directory / "continuation.json").exists():
        if directory in visited or len(visited) >= 8:
            raise ValueError("Training continuation cycle or excessive chain")
        visited.add(directory)
        value = json.loads(
            (directory / "continuation.json").read_text(encoding="utf-8")
        )
        if value.get("kind") != "rl01.continuation":
            raise ValueError("Invalid training continuation record")
        directory = Path(value["directory"]).resolve()
    return directory


def main():
    from civ5rl.common import read_json
    from civ5rl.control import is_active

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pause", "resume", "status"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--foreground", action="store_true")
    args = parser.parse_args()
    directory = resolve_directory(args.out)
    if not (directory / "passport.json").exists():
        raise ValueError(f"No training run at {directory}")
    if args.action == "status":
        print(
            json.dumps(
                {
                    "directory": str(directory),
                    "active": is_active(directory),
                    "stop_requested": (directory / "stop.request").exists(),
                    "status": read_json(directory / "run-status.json")
                    if (directory / "run-status.json").exists()
                    else None,
                    "latest": read_json(directory / "latest.json")
                    if (directory / "latest.json").exists()
                    else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.action == "pause":
        (directory / "stop.request").touch()
        print(
            "Stop requested; the active PPO update/checkpoint will finish before exit."
        )
    else:
        if is_active(directory):
            raise RuntimeError(
                "Run is still active; wait for paused status before resuming"
            )
        if not args.foreground:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            with (
                (directory / "background.stdout.log").open(
                    "a", encoding="utf-8"
                ) as out,
                (directory / "background.stderr.log").open(
                    "a", encoding="utf-8"
                ) as err,
            ):
                child = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "resume",
                        "--out",
                        str(directory),
                        "--foreground",
                    ],
                    cwd=ROOT,
                    stdout=out,
                    stderr=err,
                    stdin=subprocess.DEVNULL,
                    creationflags=flags,
                    start_new_session=os.name != "nt",
                )
            print(json.dumps({"pid": child.pid, "directory": str(directory)}))
            return
        execution = read_json(directory / "execution.json")
        (directory / "stop.request").unlink(missing_ok=True)
        from tools.train_rl import run

        run(
            read_json(directory / "resolved-config.json"),
            read_json(directory / "corpus-manifest.json"),
            directory,
            device=execution["device"],
            budget=execution["budget"],
            resume="latest",
            variant=execution["variant"],
            evaluate_every=execution["evaluate"],
            replays=execution.get("replays", True),
            stage_cycle=execution["stage_cycle"],
            max_updates=None,
        )


if __name__ == "__main__":
    main()
