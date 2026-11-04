"""Authorized sequence: smoke -> representative benchmark -> C/17 pilot -> dev."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from civ5rl.common import read_json, write_json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="New experiment directory")
    args = p.parse_args()
    directory = Path(args.out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    def execute(phase, argv):
        write_json(directory / "status.json", {"phase": phase, "status": "running"})
        with (
            (directory / f"{phase}.stdout.log").open("a", encoding="utf-8") as stdout,
            (directory / f"{phase}.stderr.log").open("a", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                [python, *argv], cwd=ROOT, stdout=stdout, stderr=stderr
            )
            write_json(
                directory / "status.json",
                {"phase": phase, "status": "running", "pid": process.pid},
            )
            code = process.wait()
        if code:
            write_json(
                directory / "status.json",
                {"phase": phase, "status": "failed", "exit_code": code},
            )
            raise RuntimeError(
                f"{phase} failed: see {directory / (phase + '.stderr.log')}"
            )

    execute(
        "smoke",
        ["tools/train_rl.py", "--phase", "smoke", "--out", str(directory / "smoke")],
    )
    execute("smoke-report", ["tools/report_rl.py", str(directory / "smoke")])
    execute(
        "benchmark", ["tools/benchmark_rl.py", "--out", str(directory / "benchmark")]
    )
    execute(
        "pilot",
        ["tools/train_rl.py", "--phase", "pilot", "--out", str(directory / "pilot")],
    )
    execute("pilot-report", ["tools/report_rl.py", str(directory / "pilot")])
    best = read_json(directory / "pilot/run-result.json")["best_checkpoint"]
    execute(
        "memory-dev",
        [
            "tools/evaluate_rl.py",
            str(directory / "pilot/checkpoints" / best),
            "--split",
            "memory_dev",
            "--out",
            str(directory / "memory-dev"),
        ],
    )
    write_json(
        directory / "status.json",
        {"phase": "complete", "status": "completed", "test_opened": False},
    )


if __name__ == "__main__":
    main()
