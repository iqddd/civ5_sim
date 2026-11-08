"""Checks for silent pairing, accounting and RNG contamination errors."""

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from civ5rl.experiments.evaluation import clustered_paired_bootstrap
from civ5rl.provenance.audit import (
    verify_checkpoint_chain,
    verify_ppo_ledger,
    write_artifact_hashes,
)


class AuditTests(unittest.TestCase):
    def test_cluster_pairing_and_world_resampling(self):
        before, after = {}, {}
        for world in range(2):
            for case_id, placement in enumerate(("contact", "separated")):
                for side in (0, 1):
                    key = (world, case_id, side)
                    case = {"placement": placement}
                    before[key] = ({"side": side, "winner": 1 - side}, case, None)
                    after[key] = (
                        {"side": side, "winner": side if world else 1 - side},
                        case,
                        None,
                    )
        result = clustered_paired_bootstrap(before, after, clusters=2, samples=10000)
        assert result["delta"] == 0.5
        assert result["CI95"] == [0.0, 1.0]
        assert result["transitions"]["L"]["W"] == 4
        after[(0, 0, 0)] = (after[(0, 0, 0)][0], {"placement": "separated"}, None)
        with self.assertRaisesRegex(ValueError, "metadata"):
            clustered_paired_bootstrap(before, after, clusters=2)

    def test_ledger_rejects_wrong_counts_and_invalid_kl(self):
        with tempfile.TemporaryDirectory() as temp:
            self.check_ledger(Path(temp))

    def check_ledger(self, tmp_path):
        def write(name, obj):
            (tmp_path / name).write_text(json.dumps(obj), encoding="utf-8")

        write("run.json", {"config": {"ppo": {"stop_kl": 0.015}}})
        write(
            "state.json",
            {
                "updates": 1,
                "decisions": 4,
                "optimizer_steps": 1,
                "attempted_steps": 1,
                "rejected_steps": 0,
            },
        )
        row = {
            "decisions": 4,
            "ppo": {
                "optimizer_steps": 1,
                "optimizer_step_records": [{"accepted": True, "attempted_kl": 0.01}],
            },
        }
        write("updates.jsonl", row)
        assert verify_ppo_ledger(tmp_path)["status"] == "verified"
        row["ppo"]["optimizer_step_records"][0]["attempted_kl"] = float("nan")
        write("updates.jsonl", row)
        with self.assertRaisesRegex(ValueError, "KL"):
            verify_ppo_ledger(tmp_path)
        row["ppo"]["optimizer_step_records"][0]["attempted_kl"] = 0.01
        row["ppo"]["optimizer_steps"] = 2
        write("updates.jsonl", row)
        with self.assertRaisesRegex(ValueError, "optimizer_steps"):
            verify_ppo_ledger(tmp_path)
        row["ppo"]["optimizer_steps"] = 1
        row["rollout"] = 1
        skipped = {
            "rollout": 2,
            "decisions": 3,
            "ppo": {
                "PPO_skipped": True,
                "optimizer_steps": 0,
                "attempted_optimizer_steps": 0,
                "rejected_optimizer_steps": 0,
            },
        }
        (tmp_path / "updates.jsonl").write_text(
            json.dumps(row) + "\n" + json.dumps(skipped) + "\n", encoding="utf-8"
        )
        write(
            "state.json",
            {
                "updates": 1,
                "decisions": 7,
                "optimizer_steps": 1,
                "attempted_steps": 1,
                "rejected_steps": 0,
            },
        )
        assert verify_ppo_ledger(tmp_path)["totals"]["updates"] == 1
        row["rollout"], skipped["rollout"] = 5, 6
        (tmp_path / "updates.jsonl").write_text(
            json.dumps(row) + "\n" + json.dumps(skipped), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Out-of-sequence"):
            verify_ppo_ledger(tmp_path)
        row["rollout"] = 1
        skipped["rollout"] = 1
        (tmp_path / "updates.jsonl").write_text(
            json.dumps(row) + "\n" + json.dumps(skipped), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Out-of-sequence"):
            verify_ppo_ledger(tmp_path)
        skipped["rollout"] = 2
        skipped["ppo"]["attempted_optimizer_steps"] = 1
        (tmp_path / "updates.jsonl").write_text(
            json.dumps(row) + "\n" + json.dumps(skipped), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "records unavailable"):
            verify_ppo_ledger(tmp_path)

    def test_checkpoint_audit_preserves_callers_random_streams(self):
        with tempfile.TemporaryDirectory() as temp:
            self.check_checkpoint(Path(temp))

    def check_checkpoint(self, tmp_path):
        from civ5rl.core.model import Policy
        from civ5rl.provenance.rng import capture_rng

        policy = Policy()
        optimizer = torch.optim.Adam(policy.parameters())
        checkpoint = {
            "variant": policy.variant,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": {"optimizer_steps": 0},
            **capture_rng(random.Random(99)),
        }
        (tmp_path / "checkpoints").mkdir()
        torch.save(checkpoint, tmp_path / "checkpoints" / "initial.pt")
        py, numpy, cpu = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state().clone(),
        )
        verify_checkpoint_chain(tmp_path)
        assert py == random.getstate()
        assert (
            numpy[0] == np.random.get_state()[0]
            and np.array_equal(numpy[1], np.random.get_state()[1])
            and numpy[2:] == np.random.get_state()[2:]
        )
        assert torch.equal(cpu, torch.get_rng_state())
        checkpoint["state"]["optimizer_steps"] = 1
        torch.save(checkpoint, tmp_path / "checkpoints" / "initial.pt")
        with self.assertRaisesRegex(ValueError, "counter"):
            verify_checkpoint_chain(tmp_path)
        parameter = next(policy.parameters())
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        checkpoint["optimizer"] = optimizer.state_dict()
        torch.save(checkpoint, tmp_path / "checkpoints" / "initial.pt")
        verify_checkpoint_chain(tmp_path)
        first = next(iter(checkpoint["optimizer"]["state"].values()))
        first["step"] = torch.tensor(1.5)
        torch.save(checkpoint, tmp_path / "checkpoints" / "initial.pt")
        with self.assertRaisesRegex(ValueError, "Invalid Adam step"):
            verify_checkpoint_chain(tmp_path)
        first["step"] = torch.tensor(1.0)
        first["exp_avg"] = torch.zeros(1)
        torch.save(checkpoint, tmp_path / "checkpoints" / "initial.pt")
        with self.assertRaisesRegex(ValueError, "exp_avg shape"):
            verify_checkpoint_chain(tmp_path)

    def test_hashing_skips_live_lock_and_derived_transients(self):
        from civ5rl.engine.control import RunLock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "data.bin",
                "output.log",
                "temp.tmp",
                "artifact-hashes.json",
                "__pycache__/cache.pyc",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"data")
            with RunLock(root):
                hashes = write_artifact_hashes(root, root / "derived-hashes.json")
            self.assertEqual(set(hashes), {"data.bin"})
