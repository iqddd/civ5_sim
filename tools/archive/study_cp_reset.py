"""Independent CPU experiment; no edits to frozen imitation runs."""

import argparse
import multiprocessing as mp
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from civ5rl.provenance.common import read_json, write_json
from civ5sim.cp_reset_study import check_identity, prepare, report, run_task


def run(config,output):
    manifest = check_identity(config,output)
    started = perf_counter()
    checks = []
    with ProcessPoolExecutor(max_workers=config["workers"],mp_context=mp.get_context("spawn")) as pool:
        for cohort in ("corpus-10","large-16"):
            tasks = [(config,c,s,str(output),True) for c in manifest["cases"] if c["cohort"]==cohort for s in (0,1)]
            completed = []
            phase_started = perf_counter()
            for first in range(0,len(tasks),config["workers"]):
                wave = tasks[first:first+config["workers"]]
                # Reuse completed match files only under the frozen identity; never retrain anything.
                results = []
                pending = []
                for task in wave:
                    _,case,side,_,_ = task
                    path = output/cohort/f"layout-{case['layout_id']:03d}"/f"stateless-{side}"/"metrics.json"
                    if path.exists():
                        results.append(read_json(path))
                    else:
                        pending.append(task)
                results.extend(pool.map(run_task,pending))
                completed.extend(results)
                elapsed = perf_counter()-phase_started
                remaining = (len(tasks)-len(completed))*elapsed/len(completed)
                counts = {o:sum(r["outcome"]==o for r in completed) for o in ("W","D","L")}
                write_json(output/"progress.json",{"cohort":cohort,"completed":len(completed),"total":len(tasks),
                           "seconds":elapsed,"eta_seconds":remaining,"WDL_stateless":counts})
                print(f"{cohort}: {len(completed)}/{len(tasks)}, WDL={counts}, elapsed={elapsed:.1f}s ETA={remaining:.1f}s",flush=True)
                if first == 0:
                    # A whole real match without shadow must reproduce its instrumented trajectory.
                    for task in wave[:2]:
                        control = list(task)
                        control[-1] = False
                        reference = next(r for r in results if r["layout_id"]==task[1]["layout_id"] and r["stateless_side"]==task[2])
                        repeated = pool.submit(run_task,tuple(control)).result()
                        if repeated["trajectory_sha256"] != reference["trajectory_sha256"]:
                            raise AssertionError("Shadow advice changed real commands, RNG or states")
                        checks.append({"cohort":cohort,"layout":task[1]["layout_id"],"side":task[2],
                                       "trajectory_sha256":reference["trajectory_sha256"],"shadow_noninterference":True})
                    write_json(output/"checks.json",checks)
            report(config,output)
            write_json(output/f"{cohort}-execution.json",{"seconds":perf_counter()-phase_started,"matches":len(completed)})
    report(config,output)
    hashes = {str(p.relative_to(output)).replace("\\","/"):__import__("hashlib").sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.rglob("*")) if p.is_file() and p.name not in ("artifact-hashes.json","status.json")}
    write_json(output/"artifact-hashes.json",hashes)
    write_json(output/"status.json",{"status":"complete","matches":len(manifest["cases"])*2,
                                     "seconds":perf_counter()-started,"shadow_checks":len(checks)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("prepare","run","report"))
    parser.add_argument("--config",default="configs/cp-reset-study.json")
    args = parser.parse_args()
    config = read_json(ROOT/args.config)
    output = ROOT/config["output"]
    try:
        if args.phase == "prepare":
            manifest = prepare(config,output)
            print(f"Prepared {len(manifest['cases'])} layouts, {len(manifest['cases'])*2} matches",flush=True)
        elif args.phase == "run":
            run(config,output)
        else:
            report(config,output)
    except BaseException:
        output.mkdir(parents=True,exist_ok=True)
        (output/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        write_json(output/"status.json",{"status":"technical_error"})
        raise


if __name__ == "__main__":
    main()
