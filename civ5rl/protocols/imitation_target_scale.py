"""Ordinary stateful CP on both sides; no policy or private training inputs."""
import traceback
from collections import Counter
from pathlib import Path
from time import perf_counter

from civ5rl.protocols.imitation import (RecordingCP, dump_episode, run_imitation_match,
                                      search_config, summarize_match_timings)
from civ5rl.provenance.common import write_json
from civ5sim.engine import digest

KIND = 'rl01.cp_imitation_target_scale'


def match_task(task):
    config, case, destination = task
    directory = Path(destination)
    directory.mkdir(parents=True,exist_ok=True)
    if case['source'] != KIND or case['split'] != 'train' or case['case_hash'] != digest(
            {k:v for k,v in case.items() if k!='case_hash'}):
        raise ValueError('Invalid CP imitation training case')
    write_json(directory/'case.json',case)
    agents = [RecordingCP(search_config(config)) for _ in (0,1)]
    try:
        outcome, timings, started = run_imitation_match(config,case,agents,directory)
        if outcome.result['status'] != 'completed':
            raise RuntimeError(str(outcome.result['error']))
        entries=[]; kinds=Counter(); units=Counter()
        for side, agent in enumerate(agents):
            episode={'schema':1,'match_id':case['episode_id'],'side':side,'split':'train',
                     'parent_world':case['parent_world'],'context':agent.public_context,
                     'decisions':agent.decisions,'winner':outcome.result['winner']}
            masks=[r['supervised'] for r in agent.decisions]
            path=directory/f'demonstration-side-{side}.json.gz'
            dump_episode(path,episode)
            for row in agent.decisions:
                if row['supervised']:
                    action=row['actions'][row['chosen']];kinds[action['kind']]+=1
                    own={u['id']:u['type'] for u in row['observation']['units']}
                    units[own.get(action.get('unit'),'no_unit')]+=1
            entries.append({'path':str(path),'side':side,'match_id':case['episode_id'],
                            'labels':sum(masks),'positions':len(masks),'supervised':masks})
        result={'match_id':case['episode_id'],'status':'completed','entries':entries,
                'labels':sum(e['labels'] for e in entries),'positions':sum(e['positions'] for e in entries),
                'winner':outcome.result['winner'],'reason':outcome.result['reason'],
                'rounds':outcome.result['completed_rounds'],'seconds':perf_counter()-started,
                'map_size':case['map_size'],'army_bin':case['army_bin'],'army_size':case['army_size'],
                'parent_world':case['parent_world'],'placement':case['placement'],
                'commands':dict(kinds),'unit_types':dict(units),
                'timings':summarize_match_timings(timings,by_side=True)}
        write_json(directory/'collection-metrics.json',result)
        return result
    except BaseException:
        (directory/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
