"""Readiness waits for evaluation request pipes on POSIX and Windows."""

from __future__ import annotations

import os
import selectors
from multiprocessing.connection import wait as wait_connections


class ConnectionWaiter:
    """Keep POSIX pipe registrations across batches; retain Windows pipe waits.

    Only the serving thread owns the selector. Synchronize removals before
    additions using saved integer descriptors: a closed Connection cannot expose
    its old fd, and that number can already belong to a newly submitted match.
    Pending workers cannot write another request until their batch reply, so
    gathering can leave them registered and filter them out of readiness.
    """

    def __init__(self):
        self.selector = selectors.DefaultSelector() if os.name == "posix" else None
        self.registered = {}

    def wait(self, connections, timeout):
        if self.selector is None:
            return wait_connections(connections, timeout=timeout)
        for connection, fd in list(self.registered.items()):
            if connection.closed:
                self.selector.unregister(fd)
                del self.registered[connection]
        for connection in connections:
            if connection not in self.registered:
                fd = connection.fileno()
                self.selector.register(fd, selectors.EVENT_READ, connection)
                self.registered[connection] = fd
        ready = {key.data for key, _ in self.selector.select(timeout)}
        # A pending peer closing its pipe wakes select too; end this gather
        # rather than spinning on its EOF until the batching timeout.
        return [connection for connection in connections if connection in ready]

    def close(self):
        if self.selector is not None:
            self.selector.close()
        self.registered.clear()
