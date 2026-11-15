"""Persistent phase time excludes downtime; a heartbeat bounds crash accounting loss."""
from contextlib import contextmanager
from pathlib import Path
from threading import Event,Thread,Lock
from time import perf_counter
from civ5rl.provenance.common import read_json,write_json


class PhaseBudget:
    def __init__(self,out,phase,limit):
        self.path=Path(out)/'phase-budgets'/f'{phase}.json'
        self.base=read_json(self.path)['seconds'] if self.path.exists() else 0.
        self.started=perf_counter();self.limit=limit
        self.lock=Lock();self.stop=Event();self.thread=None
    def elapsed(self):return self.base+perf_counter()-self.started
    def expired(self):return self.elapsed()>=self.limit
    def publish(self):
        with self.lock:
            write_json(self.path,dict(seconds=self.elapsed(),limit=self.limit,crash_accounting_resolution_seconds=1))
    def __enter__(self):
        self.publish()
        def record():
            while not self.stop.wait(1):self.publish()
        self.thread=Thread(target=record,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.publish()
