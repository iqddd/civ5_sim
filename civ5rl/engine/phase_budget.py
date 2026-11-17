"""Persistent phase time excludes downtime; a heartbeat bounds crash accounting loss."""
from contextlib import contextmanager
from pathlib import Path
from threading import Event,Thread,Lock
from time import perf_counter,sleep
import warnings
from civ5rl.provenance.common import read_json,write_json


class PhaseBudget:
    def __init__(self,out,phase,limit,*,initial_seconds=0.):
        self.path=Path(out)/'phase-budgets'/f'{phase}.json'
        self.base=read_json(self.path)['seconds'] if self.path.exists() else initial_seconds
        self.started=perf_counter();self.limit=limit
        self.lock=Lock();self.stop=Event();self.thread=None
    def elapsed(self):return self.base+perf_counter()-self.started
    def expired(self):return self.elapsed()>=self.limit
    def publish(self):
        """Retry short Windows reader/rename conflicts without resetting clocks."""
        with self.lock:
            for attempt in range(6):
                try:
                    write_json(self.path,dict(seconds=self.elapsed(),limit=self.limit,crash_accounting_resolution_seconds=1))
                    return
                except PermissionError:
                    if attempt==5:raise
                    sleep(.01*2**attempt)
    def __enter__(self):
        self.publish()
        def record():
            while not self.stop.wait(1):
                try:self.publish()
                except PermissionError:
                    warnings.warn('Budget heartbeat publication denied; retrying on the next tick',RuntimeWarning)
        self.thread=Thread(target=record,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.publish()
