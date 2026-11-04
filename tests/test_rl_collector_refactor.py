"""Failures during worker setup must release every acquired resource."""

import unittest
from pathlib import Path
from unittest.mock import patch

from civ5rl.collector import Collector


class Connection:
    def __init__(self):
        self.closed = False
        self.messages = []

    def close(self):
        self.closed = True

    def send(self, message):
        self.messages.append(message)


class Process:
    def __init__(self, fail=False):
        self.fail = fail
        self.pid = None
        self.alive = False

    def start(self):
        if self.fail:
            raise RuntimeError("start failed")
        self.pid = 123
        self.alive = True

    def join(self, timeout):
        self.alive = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False


class Slots:
    def __init__(self):
        self.descriptor = {}
        self.closed = False

    def close(self):
        self.closed = True


class Context:
    def __init__(self, failure):
        self.failure = failure
        self.pipes = []
        self.processes = []

    def Pipe(self):
        if self.failure == "pipe" and len(self.pipes) == 1:
            raise RuntimeError("pipe failed")
        pair = (Connection(), Connection())
        self.pipes.append(pair)
        return pair

    def Process(self, **kwargs):
        process = Process(self.failure == "start" and len(self.processes) == 1)
        self.processes.append(process)
        return process


class CollectorResourceTests(unittest.TestCase):
    def test_failed_setup_releases_previous_and_current_resources(self):
        for failure in ("pipe", "slots", "profile", "start"):
            with self.subTest(failure=failure):
                context = Context(failure)
                slots = []

                def make_slots(*args, failure=failure, slots=slots):
                    if failure == "slots" and len(slots) == 1:
                        raise RuntimeError("slots failed")
                    item = Slots()
                    slots.append(item)
                    return item

                def mkdir(*args, failure=failure, slots=slots, **kwargs):
                    if failure == "profile" and len(slots) == 2:
                        raise RuntimeError("profile failed")

                config = {
                    "collection": {"envs_per_worker": 1, "cpu_workers": 2},
                    "performance": {
                        "worker_encoding": True,
                        "worker_profile_dir": "profile",
                    },
                }
                with (
                    patch("civ5rl.validation.validate_config"),
                    patch("civ5rl.collector.mp.get_context", return_value=context),
                    patch("civ5rl.shared.EncodedSlots", side_effect=make_slots),
                    patch.object(Path, "mkdir", mkdir),
                    self.assertRaisesRegex(RuntimeError, "failed"),
                ):
                    Collector(config, None, "cpu", Path("unused"))
                self.assertTrue(all(c.closed for pair in context.pipes for c in pair))
                self.assertTrue(all(slot.closed for slot in slots))
                self.assertTrue(all(not process.alive for process in context.processes))
                self.assertEqual(context.pipes[0][0].messages, [("close", None)])

    def test_close_is_idempotent(self):
        context = Context(None)
        config = {"collection": {"envs_per_worker": 1, "cpu_workers": 1}}
        with (
            patch("civ5rl.validation.validate_config"),
            patch("civ5rl.collector.mp.get_context", return_value=context),
        ):
            collector = Collector(config, None, "cpu", Path("unused"))
            collector.close()
            collector.close()
        self.assertEqual(context.pipes[0][0].messages, [("close", None)])
        self.assertTrue(all(c.closed for c in context.pipes[0]))


if __name__ == "__main__":
    unittest.main()
