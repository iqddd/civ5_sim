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
    parser.add_argument('--history-microbatch',type=int,help='Whole-turn fit: scored histories per GPU microbatch')
    parser.add_argument('--compile',action=argparse.BooleanOptionalAction,default=None,help='Whole-turn fit: compile CUDA operators')
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
    from civ5rl.provenance.execution import execution_session
    if kind=='rl01.turn_comparison_fit':
        if args.phase=='collect':parser.error('whole-turn fitting uses existing continuations only')
        if any(getattr(args,n) is not None for n in ('runtime_config','workers','source_share','buffer_limit','pack_mib_per_second','prefix_states')):
            parser.error('collection options do not apply to whole-turn fitting')
        if args.history_microbatch is not None and args.history_microbatch<1:parser.error('history microbatch must be positive')
        from civ5rl.experiments.turn_fit import load_config,run
        cfg=load_config(args.config)
        if args.phase=='precheck' and (args.out is None or args.out.resolve()==(ROOT/cfg['output']).resolve()):
            parser.error('precheck requires a separate --out')
        from civ5rl.reporting.progress_process import ProcessProgress
        out=args.out or ROOT/cfg['output']
        execution_session(out,args.phase)
        with ProcessProgress(out,cfg):
            run(args.phase,cfg,out,microbatch=args.history_microbatch or 2,
                compile_model=args.compile is not False,preparation_seconds=perf_counter()-preparation_started)
        return
    if args.history_microbatch is not None or args.compile is not None:
        parser.error('history microbatch and compile options apply only to whole-turn fitting')
    if kind=='rl01.turn_comparison_collection':
        if args.phase not in ('prepare','precheck','collect','resume'):parser.error('whole-turn protocol is collection-only')
        if any(getattr(args,n) is not None for n in ('runtime_config','source_share','buffer_limit','pack_mib_per_second','prefix_states')):
            parser.error('whole-turn runtime accepts --workers only')
        from civ5rl.experiments.turn_comparison import load_config,run
        cfg=load_config(args.config)
        if args.phase=='precheck' and (args.out is None or args.out.resolve()==(ROOT/cfg['output']).resolve()):
            parser.error('precheck requires a separate --out')
        from civ5rl.reporting.progress_process import ProcessProgress
        out=args.out or ROOT/cfg['output']
        execution_session(out,args.phase)
        with ProcessProgress(out,cfg):run(args.phase,cfg,out,args.workers or 8)
        return
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
        execution_session(out,args.phase)
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
        execution_session(out,args.phase)
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
    execution_session(out,args.phase)
    with ProcessProgress(out,cfg):run(args.phase,cfg,out,preparation_seconds=perf_counter()-preparation_started)

if __name__=='__main__':main()
