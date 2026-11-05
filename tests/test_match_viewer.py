"""Exact replay frames, visibility, catalog selection and local HTTP boundary."""

import gzip
import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from test_maps import field

from civ5sim import Action, Battle
from civ5sim.engine import digest
from tools.view_combat import ReplayStore, create_server, prepare_replay

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "build" / ("combat.dll" if os.name == "nt" else "libcombat.so")


@unittest.skipUnless(LIBRARY.exists(), "Build combat library first")
class MatchViewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_replay(self, directory, *, fog=False):
        directory.mkdir(parents=True, exist_ok=True)
        scenario = {"schema": 1, "kind": "civ5sim.scenario", "map": field(8, 5),
                    "fog": fog, "random_damage": True, "seed": 17,
                    "units": [{"type": "warrior", "side": 0, "tile": 17},
                              {"type": "warrior", "side": 1, "tile": 19 if not fog else 39}]}
        snapshots = []
        observations = []
        with Battle(scenario, log=True) as battle:
            snapshots.append(battle.snapshot())
            observations.append([battle.observe(s)["visibility"] for s in (0, 1)])
            commands = [Action("move", 0, 18)]
            if not fog:
                commands.append(Action("melee", 0, 19))
            commands.append(Action("end_turn"))
            for action in commands:
                self.assertTrue(battle.step(action).ok)
                snapshots.append(battle.snapshot())
                observations.append([battle.observe(s)["visibility"] for s in (0, 1)])
            replay = battle.replay()
        path = directory / "replay.json"
        path.write_text(json.dumps(replay), encoding="utf-8")
        metadata = {"schema": 1, "kind": "civ5sim.match_result", "status": "completed",
                    "config": {"seed": 17, "fog": fog, "random_damage": True,
                               "armies": [["warrior"], ["warrior"]]},
                    "initial_state_sha256": digest(snapshots[0]),
                    "final_state_sha256": digest(snapshots[-1]),
                    "agents": ["cp", "greedy"], "commands": len(commands)}
        (directory / "result.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path, snapshots, observations

    def test_every_frame_matches_kernel_and_final_turn(self):
        path, snapshots, _ = self.write_replay(self.root / "match")
        result = prepare_replay(path)
        self.assertTrue(result["verified"])
        self.assertEqual(len(result["frames"]), len(result["steps"])+1)
        for frame, snapshot in zip(result["frames"], snapshots):
            self.assertEqual(frame["turn"], snapshot["state"]["turn"])
            self.assertEqual(frame["side"], snapshot["state"]["side"])
            for row, unit in zip(frame["units"], snapshot["units"]):
                self.assertEqual(row, [unit[key] for key in result["unit_fields"]])
        self.assertEqual(result["turns"][-1], {"turn": 1, "side": 1, "frame": 3})
        self.assertEqual(result["result"]["agents"], ["cp", "greedy"])

    def test_fog_masks_are_real_observations_for_each_frame(self):
        path, _, observations = self.write_replay(self.root / "fog", fog=True)
        result = prepare_replay(path)
        for frame, views in zip(result["frames"], observations):
            self.assertEqual(frame["visibility"], ["".join("1" if v else "0" for v in view)
                                                   for view in views])
        self.assertEqual(result["frames"][0]["visibility"][0][39], "0")

    def test_corrupted_replay_rejected_and_cache_invalidated(self):
        path, _, _ = self.write_replay(self.root / "match")
        store = ReplayStore(self.root)
        self.assertTrue(json.loads(store.load("match/replay.json"))["verified"])
        replay = json.loads(path.read_text())
        replay["steps"][0]["state_sha256"] = "invalid"
        path.write_text(json.dumps(replay), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "команде 1"):
            store.load("match/replay.json")

    def test_historical_frames_require_matching_replay_and_result_hashes(self):
        path, _, _ = self.write_replay(self.root / "legacy")
        prepared = prepare_replay(path)
        replay = json.loads(path.read_text())
        replay['initial']['rules']['fingerprint'] = 'old-kernel'
        path.write_text(json.dumps(replay), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'архив кадров'):
            prepare_replay(path)
        result = path.with_name('result.json')
        with gzip.open(path.with_name('viewer-cache.json.gz'), 'wt', encoding='utf-8') as stream:
            json.dump({'schema': 1, 'kind': 'civ5sim.viewer_cache', 'payload': prepared,
                'replay_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'result_sha256': hashlib.sha256(result.read_bytes()).hexdigest()}, stream)
        self.assertEqual(prepare_replay(path)['visibility_mode'], 'legacy')
        replay['steps'][0]['state_sha256'] = 'changed'
        path.write_text(json.dumps(replay), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'архив кадров'):
            prepare_replay(path)

    def test_catalog_deduplicates_repeats_and_rejects_path_escape(self):
        self.write_replay(self.root / "case/0")
        self.write_replay(self.root / "case/1")
        incomplete, _, _ = self.write_replay(self.root / "incomplete")
        incomplete.with_name("result.json").unlink()
        store = ReplayStore(self.root)
        self.assertEqual([m["id"] for m in store.catalog()], ["case/0/replay.json"])
        with self.assertRaises(FileNotFoundError):
            store.load("../replay.json")
        single = ReplayStore(incomplete)
        self.assertEqual(len(single.catalog()), 1)

    def test_rerun_colors_require_matching_checked_passport_and_target(self):
        path, _, _ = self.write_replay(self.root / "case/0")
        self.write_replay(self.root / "other/0")
        target = {'schema': 1, 'port': 'AI/0.9.3', 'run_signature': 'new',
                  'cases': ['case/0/replay.json']}
        (self.root/'replay-refresh.json').write_text(json.dumps(target))
        passport = path.parent.parent/'passport.json'
        checked = path.with_name('checked.json')
        passport.write_text(json.dumps({'port': 'AI/0.9.2', 'run_signature': 'old'}))
        checked.write_text(json.dumps({'replay_verified': True, 'run_signature': 'old'}))
        store = ReplayStore(self.root)
        before = {r['id']: r for r in store.catalog()}
        self.assertEqual(before['case/0/replay.json']['refresh_status'], 'pending')
        self.assertEqual(before['other/0/replay.json']['refresh_status'], 'old')
        loaded = json.loads(store.load('case/0/replay.json'))
        self.assertEqual(loaded['provenance']['port'], 'AI/0.9.2')
        checked.write_text(json.dumps({'replay_verified': True, 'run_signature': 'new'}))
        self.assertEqual(store.catalog()[0]['refresh_status'], 'pending')
        passport.write_text(json.dumps({'port': 'AI/0.9.3', 'run_signature': 'new'}))
        after = store.catalog()[0]
        self.assertEqual(after['refresh_status'], 'updated')
        self.assertNotEqual(after['revision'], before['case/0/replay.json']['revision'])
        # Even byte-identical replays need a fresh payload after a new run.
        loaded = json.loads(store.load('case/0/replay.json'))
        self.assertEqual(loaded['provenance']['port'], 'AI/0.9.3')
        self.assertEqual(loaded['provenance']['refresh_status'], 'updated')
        target['run_signature'] = 'next'
        (self.root/'replay-refresh.json').write_text(json.dumps(target))
        self.assertEqual(json.loads(store.load('case/0/replay.json'))['provenance']['refresh_status'],
                         'pending')

    def test_atomic_publication_retries_transient_windows_reader(self):
        from unittest.mock import patch

        from tools.archive.publish_greedy_replays import atomic_json
        target = self.root/'summary.json'
        target.write_text(json.dumps({'old': True}))
        original = Path.replace
        attempts = []

        def replace(path, destination):
            attempts.append(path)
            if len(attempts) < 3:
                self.assertEqual(json.loads(target.read_text()), {'old': True})
                raise PermissionError('Temporary Windows reader')
            return original(path, destination)

        with patch.object(Path, 'replace', new=replace):
            atomic_json(target, {'updated': True})
        self.assertEqual(json.loads(target.read_text()), {'updated': True})
        self.assertFalse(target.with_suffix('.json.tmp').exists())
        self.assertEqual(len(attempts), 3)

    def test_unrelated_result_is_not_displayed(self):
        path, _, _ = self.write_replay(self.root / "match")
        result_path = path.with_name("result.json")
        result = json.loads(result_path.read_text())
        result["final_state_sha256"] = "another battle"
        result_path.write_text(json.dumps(result))
        self.assertIsNone(prepare_replay(path)["result"])

    def test_empty_replay_with_result_preserves_initial_position(self):
        path, _, _ = self.write_replay(self.root / "match")
        replay = json.loads(path.read_text())
        replay["steps"] = []
        path.write_text(json.dumps(replay))
        result_path = path.with_name("result.json")
        result = json.loads(result_path.read_text())
        result["final_state_sha256"] = result["initial_state_sha256"]
        result["status"] = "error"
        result_path.write_text(json.dumps(result))
        prepared = prepare_replay(path)
        self.assertEqual(len(prepared["frames"]), 1)
        self.assertEqual(prepared["result"]["status"], "error")

    def test_continuation_has_separate_catalog_status_and_boundary(self):
        path, _, _ = self.write_replay(self.root / "continuation" / "0")
        replay = json.loads(path.read_text())
        replay["experiment"] = {"kind": "continuation", "resume_after": 1,
                                "description": "New coordinator after command 1"}
        path.write_text(json.dumps(replay))
        path.with_name("checked.json").write_text(json.dumps(
            {"replay_verified": True, "continuation": True, "prefix_verified": 1}))
        row = ReplayStore(self.root).catalog()[0]
        self.assertEqual(row["refresh_status"], "continuation")
        self.assertEqual(row["resume_after"], 1)
        self.assertEqual(prepare_replay(path)["experiment"], replay["experiment"])

    def test_http_catalog_verified_frames_and_not_found(self):
        self.write_replay(self.root / "match")
        server = create_server(self.root, 0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(base + "/") as response:
                self.assertIn(b'id="timeline"', response.read())
            with urlopen(base + "/api/matches") as response:
                self.assertEqual(len(json.load(response)), 1)
            with urlopen(base + "/api/status") as response:
                self.assertIsNone(json.load(response))
            (self.root / "summary.json").write_text(json.dumps({"acceptance": False,
                "summary": {"matches": 7, "outcomes": {"opponent/greedy": {"outcomes": {"win": 4, "loss": 3}}}}}))
            with urlopen(base + "/api/status") as response:
                status = json.load(response)
                self.assertFalse(status["complete"])
                self.assertEqual(status["checked"], 7)
                self.assertEqual(status["greedy"], {"win": 4, "loss": 3})
            with urlopen(base + "/api/replay?id=match/replay.json") as response:
                self.assertTrue(json.load(response)["verified"])
            with self.assertRaises(HTTPError) as error:
                urlopen(base + "/api/replay?id=../secret.json")
            self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown()
            worker.join(timeout=5)
            server.server_close()


if __name__ == "__main__":
    unittest.main()
