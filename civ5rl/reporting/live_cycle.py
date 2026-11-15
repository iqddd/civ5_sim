"""Disposable, real TargetFactory collection and complete PPO-cycle measurements."""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=5)
    parser.add_argument('--envs', type=int, default=80)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error('cycles must be positive')
    args.source_root = args.source_root.resolve()
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Choose an empty diagnostic output directory')
    sys.path.insert(0, str(args.source_root))
    os.chdir(args.source_root)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import gc
    import random
    import statistics
    import subprocess
    from functools import partial
    from time import perf_counter

    import torch
    from civ5rl.engine.collector import Collector
    from civ5rl.engine.ppo import update
    from civ5rl.engine.ppo_adam import grouped_optimizer
    from civ5rl.engine.ppo_fast import BatchedBackend
    from civ5rl.engine.resources import ResourceMonitor
    from civ5rl.engine.spans import Span
    from civ5rl.experiments.imitation_helpers import CompilationClock, load_policy
    from civ5rl.experiments.ppo_artifacts import load_rollout, save_rollout
    from civ5rl.protocols.ppo_target_scale import TargetFactory
    from civ5rl.provenance import checkpoint
    from civ5rl.provenance.common import configure, file_sha256, write_json
    from civ5rl.provenance.rng import restore_rng

    inputs = [args.checkpoint.resolve(), args.settings.resolve(), args.base.resolve()]
    if any(p == args.out or p.is_relative_to(args.out) for p in inputs):
        raise ValueError('Output contains a protected input')
    hashes = {str(p): file_sha256(p) for p in inputs}
    cfg = json.loads(args.settings.read_text(encoding='utf-8'))
    cfg = cfg.get('config', cfg)
    base = json.loads(args.base.read_text(encoding='utf-8'))
    configure(17, 'cuda')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    policy = load_policy(cfg, args.checkpoint, 'cuda')
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    optimizer = grouped_optimizer(policy, cfg, 'cuda')
    optimizer.load_state_dict(saved['optimizer'])
    shuffle = random.Random(17)
    restore_rng(saved, shuffle)
    state = dict(saved['state'])
    next_episode = int(state['next_episode'])
    decisions = int(state['decisions'])
    args.out.mkdir(parents=True)
    identity = dict(diagnostic=True, source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=args.source_root, text=True).strip())
    rows = []
    runtime = dict(num_envs=args.envs, cpu_workers=args.workers, reply_size=1,
                   collection_scheduler='streaming', inference_min_batch=0,
                   inference_wait_seconds=0)
    write_json(args.out / 'conditions.json', dict(
        **identity, runtime=runtime, protected_sha256=hashes,
        turn_reset_mode=cfg['environment'].get('turn_reset_mode', 'legacy'),
        starting_next_episode=next_episode, starting_decisions=decisions,
        cycles=args.cycles, warmup_cycles=1,
        scientific_training_extended=False, fresh_scenarios=True))
    factory = TargetFactory(cfg, base)

    def save_state(phase, cycle):
        state.update(phase=phase, next_episode=next_episode, decisions=decisions,
                     diagnostic_cycle=cycle)
        checkpoint.save(args.out / 'checkpoints' / ('pending.pt' if phase == 'ppo' else 'latest.pt'),
                        policy, optimizer, state, shuffle, cfg, identity)

    with ResourceMonitor(args.out / 'telemetry', interval=.25) as monitor, Collector(
        cfg, factory, 'cuda', args.out / 'errors', **runtime
    ) as collector, monitor.capture_cuda_oom(policy):
        for cycle in range(args.cycles + 1):
            warmup = cycle == 0
            monitor.mark('warmup_collection' if warmup else 'collection', rollout=cycle)
            monitor.record_cuda_memory(policy, 'before_collection')
            torch.cuda.synchronize()
            start = perf_counter()
            with CompilationClock() as collection_compiler:
                episodes, next_episode, timings = collector.collect(
                    policy, None, next_episode, monitor,
                    episode_length_hint=(rows[-1]['decisions'] / rows[-1]['episodes']) if rows else None)
                torch.cuda.synchronize()
            collected = perf_counter()
            n = sum(len(ep['chosen']) for ep in episodes)
            decisions += n
            monitor.mark('rollout_save', rollout=cycle)
            artifact = save_rollout(episodes, args.out / 'rollouts', cycle + 1)
            save_state('ppo', cycle)
            prepared = perf_counter()
            monitor.mark('warmup_ppo' if warmup else 'ppo', rollout=cycle)
            with CompilationClock() as compiler:
                compiler.monitor = monitor
                metrics = update(policy, optimizer, episodes, cfg, decisions, 'cuda', shuffle,
                                 monitor=monitor, _backend=partial(BatchedBackend, pinned_replay=True))
                torch.cuda.synchronize()
            updated = perf_counter()
            monitor.mark('checkpoint_save', rollout=cycle)
            save_state('collect', cycle)
            # Match the runner's release and collection of Python objects.
            del episodes
            gc.collect()
            stopped = perf_counter()
            row = dict(cycle=cycle, warmup=warmup, decisions=n,
                       episodes=artifact['episodes'], collection_seconds=collected-start,
                       collection_compilation_seconds=collection_compiler.seconds(start,collected),
                       save_pending_seconds=prepared-collected, ppo_seconds=updated-prepared,
                       ppo_compilation_seconds=compiler.seconds(prepared,updated),
                       save_release_seconds=stopped-updated, full_seconds=stopped-start,
                       accepted_steps=metrics['optimizer_steps'],
                       rejected_steps=metrics['rejected_optimizer_steps'],
                       collection=timings, ppo=metrics, rollout=artifact,
                       Adam_counters={g['name']: sorted({int(optimizer.state[p]['step']) for p in g['params']})
                                      for g in optimizer.param_groups})
            rows.append(row)
            write_json(args.out / 'cycles.json', rows)
            monitor.record_cuda_memory(policy, 'after_update_release')
            print('CYCLE ' + json.dumps({k:v for k,v in row.items() if k not in ('collection','ppo','rollout')}), flush=True)

        # Separate short real-data trace; its wall time never enters cycle rates.
        monitor.mark('trace_preparation')
        final_episodes = load_rollout(args.out / 'rollouts' / artifact['path'])
        backend = BatchedBackend(policy, final_episodes, cfg, 'cuda', pinned_replay=True)
        spans = []
        for i, episode in enumerate(final_episodes):
            if episode['chosen']:
                spans.append(Span(i, 0, min(128,len(episode['chosen']))))
            if len(spans) == cfg['ppo']['sequences_per_minibatch']:
                break
        tokens = sum(span.stop-span.start for span in spans)
        def trace_step():
            optimizer.zero_grad(set_to_none=True)
            backend.backward(spans, tokens, cfg['ppo']['entropy_coefficient'])
            backend.measure_step_replay()
            torch.cuda.synchronize()
        for _ in range(3):
            trace_step()
        monitor.mark('trace')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            trace_step()
        profiler.export_chrome_trace(str(args.out/'real-data-trace.json'))
        optimizer.zero_grad(set_to_none=True)
        monitor.mark('finished')

    if {str(p): file_sha256(p) for p in inputs} != hashes:
        raise AssertionError('Protected input changed')
    measured = rows[1:]
    result = dict(conditions=json.loads((args.out/'conditions.json').read_text()),
                  cycles=rows, scientific_training_extended=False,
                  decisions_per_second=sum(r['decisions'] for r in measured)/sum(r['full_seconds'] for r in measured),
                  median_collection_seconds=statistics.median(r['collection_seconds'] for r in measured),
                  median_ppo_seconds=statistics.median(r['ppo_seconds'] for r in measured))
    write_json(args.out/'benchmark.json', result)
    print('COMPLETE '+json.dumps({k:v for k,v in result.items() if k not in ('cycles','conditions')}),flush=True)


if __name__ == '__main__':
    main()
