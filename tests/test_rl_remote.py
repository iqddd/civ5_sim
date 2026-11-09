"""Focused checks for prepare.json staging, split mirrors, and external scratch."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from civ5rl.deploy.operations import package, verify_package
from civ5rl.deploy.remote import _sha, input_group, stage_inputs
from civ5rl.engine.imitation_stream import PreparedDemonstrations, prepare_inputs
from civ5rl.experiments.imitation_target_scale import load_config
from civ5rl.protocols.imitation import dump_episode
from civ5rl.protocols.ppo_target_scale import TargetFactory
from civ5rl.provenance.common import ROOT, configure, read_json
from civ5sim import Battle


class RemoteTests(unittest.TestCase):
    def test_package_builds_prepare_json_with_disjoint_gitignored_sync_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            (root / ".gitignore").write_text("build/\n/artifacts/\n", encoding="utf-8")

            def put(name, data):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data, encoding="utf-8")

            put("requirements-rl.txt", "numpy\n")
            put(
                "configs/base.json",
                json.dumps(
                    {
                        "kind": "rl01.ppo_target_scale",
                        "output": "artifacts/rl/new-run",
                        "experiment": {
                            "parent": "artifacts/rl/parent-run/parent.pt",
                            "parent_manifest": "artifacts/rl/corpus/manifest.json",
                        },
                    }
                ),
            )
            put(
                "configs/run.json",
                json.dumps({"base_config": "configs/base.json"}),
            )
            for name in (
                "build/combat_rules.h",
                "build/combat_rules.json",
                "build/cp_geometry.inc",
                "artifacts/rl/parent-run/parent.pt",
                "artifacts/rl/corpus/open.json",
                "artifacts/rl/corpus/closed.json",
            ):
                put(name, name)
            put(
                "artifacts/rl/corpus/manifest.json",
                json.dumps(
                    {
                        "maps": [
                            {"path": "artifacts/rl/corpus/open.json", "split": "train"},
                            {"path": "artifacts/rl/corpus/closed.json", "split": "test"},
                        ]
                    }
                ),
            )

            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.test",
                    "commit",
                    "-qm",
                    "initial",
                ],
                cwd=root,
                check=True,
            )

            deploy_dir = Path(tmp) / "deployment"
            dep = package(
                "configs/run.json",
                deploy_dir,
                root=root,
                repo="iqddd/civ5_sim",
                verify_origin=False,
            )
            verify_package(deploy_dir, dep)

            prep = json.loads((deploy_dir / "prepare.json").read_text(encoding="utf-8"))
            self.assertEqual(prep["execution_subdir"], "civ5_sim")
            self.assertEqual(prep["repositories"], [{"repo": "iqddd/civ5_sim", "commit": dep["commit"]}])
            self.assertEqual(prep["requirements"], ["civ5_sim/requirements-rl.txt"])

            remotes = [item["remote"] for item in prep["sync"]]
            self.assertEqual(
                remotes,
                [
                    "civ5_sim/artifacts/rl/corpus",
                    "civ5_sim/artifacts/rl/parent-run",
                    "civ5_sim/build",
                ],
            )
            # Closed test split and git-tracked configs must never enter staged sync inputs.
            manifest = json.loads(
                (deploy_dir / "inputs-manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn("artifacts/rl/corpus/open.json", manifest["staged_files"])
            self.assertNotIn("artifacts/rl/corpus/closed.json", manifest["staged_files"])
            self.assertNotIn("configs/run.json", manifest["staged_files"])
            self.assertIn("configs/run.json", manifest["tracked_inputs"])
            self.assertEqual(
                _sha(deploy_dir / "inputs/artifacts/rl/corpus/open.json"),
                manifest["staged_files"]["artifacts/rl/corpus/open.json"],
            )

            # Reject deployment directories inside the experiment output mirror.
            with self.assertRaises(ValueError):
                package(
                    "configs/run.json",
                    root / "artifacts/rl/new-run/deployment",
                    root=root,
                    repo="iqddd/civ5_sim",
                    verify_origin=False,
                )

    def test_input_group_rejects_output_overlap_and_forms_disjoint_prefixes(self):
        out = "artifacts/rl/cp-imitation-target-more-data"
        self.assertEqual(input_group("build/combat_rules.json", out), "build")
        self.assertEqual(
            input_group("artifacts/rl/corpus/worlds/w.json", out),
            "artifacts/rl/corpus",
        )
        with self.assertRaises(ValueError):
            input_group("artifacts/rl/cp-imitation-target-more-data/x.json", out)
        with self.assertRaises(ValueError):
            input_group("artifacts/top.json", out)
        self.assertIsNotNone(stage_inputs)

    def test_scratch_dir_keeps_prepared_npz_outside_synced_run_directory(self):
        configure(17, "cpu")
        plan = read_json(ROOT / "artifacts/rl/cp-imitation-target-scale/manifest.json")
        base_cfg = load_config(ROOT / "configs/rl-cp-imitation-target-more-data.json")
        factory = TargetFactory(base_cfg, plan["base"])
        case = factory.case(
            0, plan["base"]["worlds"]["training"][0], 10, 0, True, 0, "scratch_ns", n=5
        )
        with Battle(case["scenario"]) as battle:
            obs = battle.observe(0)
            legal = tuple(battle.available_actions())
        episode = {
            "match_id": 0,
            "side": 0,
            "context": {
                "map": case["scenario"]["map"],
                "side": 0,
                "armies": case["armies"],
                "max_rounds": 30,
                "fog": True,
                "random_damage": True,
            },
            "decisions": [
                {
                    "observation": obs,
                    "actions": [a.__dict__ for a in legal],
                    "chosen": 0,
                    "supervised": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-mirror"
            scratch_dir = Path(tmp) / "scratch"
            ep_path = run_dir / "data/match-000000/demonstration-side-0.json.gz"
            dump_episode(ep_path, episode)
            entry = {
                "entry_id": "new:000000:0",
                "corpus_source": "new",
                "match_id": 0,
                "side": 0,
                "path": "data/match-000000/demonstration-side-0.json.gz",
                "labels": 1,
                "positions": 1,
                "supervised": [True],
            }
            with patch.dict(os.environ, {"CIV5RL_SCRATCH_DIR": str(scratch_dir)}):
                prepared = prepare_inputs(run_dir, [entry], workers=1)
                self.assertFalse((run_dir / "prepared-inputs").exists())
                self.assertTrue(
                    (scratch_dir / "run-mirror/prepared-inputs/index.json").is_file()
                )
                ds = PreparedDemonstrations(run_dir, prepared["entries"], "cpu")
                _, labels, lengths = ds.batch([(0, 0, 1)])
                self.assertEqual(lengths, [1])
                self.assertEqual(int(labels["mask"].sum().item()), 1)


if __name__ == "__main__":
    unittest.main()
