"""Verified inactive comparison data storage, outside execution passports."""
from pathlib import Path
import shutil
import tempfile
from civ5rl.deploy.archive import pack,unpack,_extract,_inventory
from civ5rl.provenance.common import read_json


def archive_tree(out,name):
    if name not in ('sources','continuations'):raise ValueError('Unknown comparison tree')
    out=Path(out).resolve();source=out/name;packet=out/(name+'.storage')
    if not source.exists():return
    manifest=pack(source,packet,token='comparison-'+name,direction='download',force=True)
    with tempfile.TemporaryDirectory(prefix='verify-comparison-') as temporary:
        _extract(packet,Path(temporary),manifest)
    if _inventory(source)!=(manifest['files'],manifest['directories']):
        raise RuntimeError('Comparison data changed during verification')
    if source.resolve()!=out/name or source.is_symlink():raise ValueError('Unsafe comparison storage path')
    shutil.rmtree(source)


def restore_tree(out,name):
    out=Path(out);source=out/name;packet=out/(name+'.storage')
    if not source.exists() and packet.exists():
        unpack(packet,source,token='comparison-'+name,direction='download')
