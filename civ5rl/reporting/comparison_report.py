"""Compact artifact-only summary for the rollout comparison protocol."""
from pathlib import Path
from civ5rl.provenance.common import read_json


def report(out):
    out=Path(out)
    sections=[]
    for name in ('collection.json','dataset.json','training.json','result.json'):
        if not (out/name).exists():continue
        value=read_json(out/name)
        keys=('status','phase','seconds','overshoot_seconds','selected_forks','completed_forks','accepted_pairs','rejections','training_steps','stop_reason','changed','selected','evaluations_skipped')
        compact={k:value[k] for k in keys if k in value}
        if name=='collection.json':compact.update(selected_forks=len(value['forks']),completed_forks=sum(f['status']=='completed' for f in value['forks']))
        sections.extend([f'## {name}', '', str(compact), ''])
    sections.extend(['Historical Development comparison uses the recorded 361 wins of update171; GPU/runtime differences limit attribution. Additional, when admitted, runs both policies through one local evaluator.',''])
    (out/'report.md').write_text('# Rollout comparison\n\n'+'\n'.join(sections),encoding='utf-8')
