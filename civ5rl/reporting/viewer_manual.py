"""Manual public-action previews and subprocess branches for the local viewer."""
import json
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from civ5sim import Action, Battle
from civ5sim.engine import digest
from civ5rl.provenance.common import file_sha256, write_json

FIELDS = ("tile", "hp", "moves", "attacked", "acted", "fortified", "waiting")


class ManualViewer:
    def __init__(self, store, policies, project_root):
        self.store, self.policies, self.root = store, policies, Path(project_root)
        import threading
        self.writer = threading.Lock()

    def source(self, payload):
        path = self.store.paths().get(payload['id'])
        if path is None:
            raise ValueError('Матч не найден')
        replay = json.loads(path.read_text(encoding='utf-8'))
        frame = payload['frame']
        if type(frame) is not int or not 0 <= frame < len(replay['steps']):
            raise ValueError('Выберите позицию перед существующей командой')
        if payload.get('source_sha256') and payload['source_sha256'] != file_sha256(path):
            raise ValueError('Исходный реплей изменился; откройте его заново')
        return path, replay, frame

    def checkpoints(self, path, replay):
        candidates = {m['id']: dict(id=m['id'], label=m.get('label',m['id']), checkpoint=str(m['checkpoint'])) for m in self.policies.models.values()}
        checkpoint = replay.get('experiment', {}).get('checkpoint')
        passport = self.store.root / 'passport.json'
        if not checkpoint and passport.exists():
            checkpoint = json.loads(passport.read_text(encoding='utf-8')).get('checkpoint')
        if checkpoint:
            candidates['source'] = dict(id='source',label='Модель исходной серии',checkpoint=checkpoint)
        return {k: v for k,v in candidates.items() if Path(v['checkpoint']).is_file()}

    def preview(self, payload):
        path, replay, frame = self.source(payload)
        case = json.loads(path.with_name('case.json').read_text(encoding='utf-8'))
        actions = payload.get('actions', [])
        if not isinstance(actions,list) or len(actions)>256:
            raise ValueError('Слишком длинная последовательность')
        frames, steps = [], []
        with Battle(replay['initial']) as battle:
            for i, step in enumerate(replay['steps'][:frame]):
                result = battle.step(step['action'])
                if asdict(result)!=step['result'] or digest(battle.snapshot())!=step['state_sha256']:
                    raise ValueError(f'Реплей расходится на {i+1}')
            if battle.snapshot()['state']['side'] != case['learner_side']:
                raise ValueError('Выберите ход модели, а не CP')
            for value in actions:
                if battle.snapshot()['state']['side'] != case['learner_side']:
                    raise ValueError('Ход завершён: создайте вариацию для продолжения')
                a=Action(**value)
                if a not in battle.available_actions():
                    raise ValueError('Команда недопустима в текущей позиции')
                result=battle.step(a);snapshot=battle.snapshot()
                steps.append(dict(action=asdict(a),result=asdict(result),state_sha256=digest(snapshot)))
                frames.append(dict(turn=snapshot['state']['turn'],side=snapshot['state']['side'],
                    units=[[u[k] for k in FIELDS] for u in snapshot['units']],
                    visibility=[''.join('1' if v else '0' for v in battle.observe(s)['visibility']) for s in (0,1)]))
            done = battle.snapshot()['state']['side'] != case['learner_side'] or any(battle.army_status()['destroyed'])
            legal=[] if done else [asdict(a) for a in battle.available_actions()]
        models=self.checkpoints(path,replay)
        return dict(frames=frames,steps=steps,legal=legal,done=done,source_sha256=file_sha256(path),
                    models=[dict(id=v['id'],label=v['label']) for v in models.values()])

    def create(self, payload):
        if not self.writer.acquire(blocking=False):
            raise ValueError('Другая вариация уже рассчитывается')
        try:
            preview=self.preview(payload)
            if not payload.get('actions'):
                raise ValueError('Добавьте хотя бы одну команду')
            path,replay,frame=self.source(payload)
            models=self.checkpoints(path,replay)
            if payload.get('model') not in models:
                raise ValueError('Не выбран доступный checkpoint продолжения')
            identifier='manual-'+uuid.uuid4().hex[:12]
            transport=self.store.root/'.manual-requests'
            transport.mkdir(exist_ok=True)
            request=transport/(identifier+'.json')
            write_json(request,payload['actions'])
            out=self.store.root/identifier
            command=[sys.executable,str(self.root/'tools/ctl_rl.py'),'branch-replay',
                '--source',str(path),'--command',str(frame+1),'--actions',str(request),
                '--checkpoint',models[payload['model']]['checkpoint'],'--out',str(out)]
            try:
                try:
                    result=subprocess.run(command,cwd=self.root,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
                except subprocess.TimeoutExpired as exc:
                    raise ValueError('Продолжение превысило лимит10 минут; процесс остановлен') from exc
                if result.returncode:
                    raise ValueError('Продолжение не создано: '+result.stderr[-2000:])
            finally:
                request.unlink(missing_ok=True)
            return dict(id=(out/'replay.json').relative_to(self.store.root).as_posix(),frame=frame)
        finally:
            self.writer.release()
