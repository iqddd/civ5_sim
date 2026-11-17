"""Run the bounded alternative-action comparison protocol or its collection-only successor."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','collect','train','evaluate','resume','precheck'])
    parser.add_argument('--config',type=Path,default=Path('configs/rl-c2p-rollout-comparison.json'))
    parser.add_argument('--out',type=Path)
    group=parser.add_argument_group('collection-only runtime (rl01.rollout_comparison_collection)')
    group.add_argument('--runtime-config',type=Path,help='recommended-runtime.json from bench_rl.py comparison-collection')
    group.add_argument('--workers',type=int)
    group.add_argument('--source-share',type=float)
    group.add_argument('--buffer-limit',type=int)
    group.add_argument('--pack-mib-per-second',type=float)
    group.add_argument('--prefix-states',action=argparse.BooleanOptionalAction,default=None)
    args=parser.parse_args()
    from time import perf_counter
    preparation_started=perf_counter()
    kind=json.loads(args.config.read_text(encoding='utf-8')).get('kind')
    if kind=='rl01.action_value':
        if args.phase=='collect':parser.error('action-value fitting uses saved outcomes only')
        if any(getattr(args,n) is not None for n in ('runtime_config','workers','source_share','buffer_limit','pack_mib_per_second','prefix_states')):
            parser.error('collection runtime options do not apply to action-value fitting')
        from civ5rl.experiments.action_value import load_config,run
        cfg=load_config(args.config)
        if args.phase=='precheck' and (args.out is None or args.out.resolve()==(ROOT/cfg['output']).resolve()):
            parser.error('precheck requires a separate --out')
        from civ5rl.reporting.progress_process import ProcessProgress
        out=args.out or ROOT/cfg['output']
        with ProcessProgress(out,cfg):run(args.phase,cfg,out)
        return
    if kind=='rl01.rollout_comparison_collection':
        from civ5rl.experiments.rollout_comparison_collection import load_config,run,CollectionRuntime
        if args.phase not in ('prepare','collect','resume'):
            parser.error('collection-only runs support prepare, collect and resume')
        values=json.loads(args.runtime_config.read_text(encoding='utf-8'))['runtime'] if args.runtime_config else {}
        for name in ('workers','source_share','buffer_limit','pack_mib_per_second','prefix_states'):
            if getattr(args,name) is not None:values[name]=getattr(args,name)
        runtime=CollectionRuntime(**values).resolved()
        cfg=load_config(args.config)
        out=args.out or ROOT/cfg['output']
        from civ5rl.reporting.progress_process import ProcessProgress
        with ProcessProgress(out,cfg):
            run(args.phase,cfg,out,runtime,preparation_seconds=perf_counter()-preparation_started)
        return
    if any(getattr(args,n) is not None for n in ('runtime_config','workers','source_share','buffer_limit','pack_mib_per_second','prefix_states')):
        parser.error('collection runtime options apply only to rl01.rollout_comparison_collection')
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
