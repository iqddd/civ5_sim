"""Run the bounded alternative-action comparison protocol."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','collect','train','evaluate','resume','precheck'])
    parser.add_argument('--config',type=Path,default=Path('configs/rl-c2p-rollout-comparison.json'))
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    from time import perf_counter
    preparation_started=perf_counter()
    from civ5rl.experiments.rollout_comparison import load_config,run
    cfg=load_config(args.config)
    if args.phase=='precheck':
        if args.out is None:parser.error('precheck requires a separate --out')
        if args.out.resolve()==(ROOT/cfg['output']).resolve():parser.error('precheck must not use scientific output')
        from civ5rl.reporting.comparison_precheck import precheck
        return precheck(cfg,args.out)
    from civ5rl.reporting.progress_process import ProcessProgress
    out=args.out or ROOT/cfg['output']
    with ProcessProgress(out,cfg):run(args.phase,cfg,out,preparation_seconds=perf_counter()-preparation_started)

if __name__=='__main__':main()
