"""Resume uses a committed checkout; artifacts remain the state of the run.

Preparation does not bind code. The first collection/training invocation records
its Git commit and compiled binaries. Later invocations require that checkout,
not a reconstruction of every ancestor's sources, dependencies or environment.
"""
import subprocess
import shutil
from pathlib import Path

from civ5rl.provenance.common import ROOT,read_json,write_json,file_sha256
from civ5rl.engine.runtime.control import RunLock

CODE_PATHS=('civ5rl','civ5sim','civ5map','native','tools','requirements-rl*.txt')


def git_code():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    changed=subprocess.check_output(['git','diff','--name-only','-z','HEAD','--',*CODE_PATHS],cwd=ROOT)
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z','--',*CODE_PATHS],cwd=ROOT)
    paths=sorted(set(p.decode('utf-8') for p in (changed+untracked).split(b'\0') if p))
    return commit,paths


def binaries():
    return {name:file_sha256(ROOT/name) for name in (
        'build/combat.dll','build/combat_ai.dll','build/libcombat.so','build/libcombat_ai.so',
        'build/combat_rules.json') if (ROOT/name).is_file()}


def execution_session(directory,phase):
    """Bind once at execution, reject a different/dirty checkout on resume.

    This never checks out Git or edits checkpoints. A completed result may be
    inspected with current tools. An unfinished pre-policy run must be resumed
    with its original checkout/tools or used as a parent of a new run.
    """
    if phase not in ('collect','train','run','resume'):return
    directory=Path(directory)
    with RunLock(directory):
        status=read_json(directory/'status.json') if (directory/'status.json').exists() else {}
        if status.get('status') in ('complete','closed_by_user'):return
        commit,dirty=git_code()
        if dirty:
            raise ValueError('Commit code changes before execution; resume requires a clean checkout. '
                             f'Uncommitted code: {dirty[:8]}')
        path=directory/'execution.json'
        current=dict(git_commit=commit,binaries=binaries())
        if path.exists():
            saved=read_json(path)
            if saved!=current:
                raise ValueError(f'Resume requires Git commit {saved["git_commit"]} and its compiled binaries. '
                    f'Restore the saved binaries from {directory / "execution-binaries"}, '
                    'use that checkout, or start a new run from the checkpoint.')
        else:
            if (directory/'passport.json').exists() and status.get('status') not in (None,'prepared'):
                raise ValueError('Unfinished historical run has no execution.json. Use its original checkout/tools '
                                 'to resume, or start a new run from the saved checkpoint.')
            # Native builds are ignored by Git and rebuilds need not be byte-
            # reproducible. Keep the few actual runtime binaries with the run.
            for name in current['binaries']:
                target=directory/'execution-binaries'/Path(name).name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/name,target)
            write_json(path,current)
