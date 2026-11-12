"""Catch silent cross-battle GRU contamination and forced-action omissions."""

from copy import deepcopy
import tempfile
import unittest
from pathlib import Path

import torch

from civ5rl.core.agent import RLAgent
from civ5rl.config import load_config
from civ5rl.core.model import Policy, batch_inputs
from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation import BatchedInference, EvaluationExecutor, EvaluationRuntime, evaluation_options, runtime_for
from civ5rl.experiments.ppo_target_scale import evaluation_task, evaluate
from civ5rl.provenance.common import configure, read_json, write_json
from civ5sim import Action, Battle
from civ5sim.engine import digest
from test_rl import context, scenario


class BatchedEvaluationTests(unittest.TestCase):
    def test_public_histories_forced_actions_and_separate_hidden_states(self):
        configure(17, "cuda" if torch.cuda.is_available() else "cpu")
        policy = Policy("C2").eval()
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        initial = scenario()
        with Battle(initial) as battle:
            observation, actions = battle.observe(0), ordered_actions(battle.available_actions())
        for device in devices:
            with self.subTest(device=device), torch.inference_mode():
                other = deepcopy(policy).to(device)
                reference = [RLAgent(policy, "cpu", "argmax") for _ in range(2)]
                shared = [RLAgent(other, device, "argmax") for _ in range(2)]
                for agent in [*reference, *shared]:
                    agent.reset(context(initial))
                engine = BatchedInference(other, device)
                for step in range(3):
                    indices = [0, 1] if step != 1 else [1]
                    command = "observe" if step == 1 else "act"
                    requests = []
                    for index in indices:
                        a, b = reference[index], shared[index]
                        encoded_a = a.encoder.encode(observation, actions)
                        encoded_b = b.encoder.encode(observation, actions)
                        cpu_logits, _, cpu_hidden = policy(batch_inputs([encoded_a], "cpu"), a.hidden)
                        gpu_logits, _, gpu_hidden = other(batch_inputs([encoded_b], device), b.hidden)
                        torch.testing.assert_close(cpu_logits, gpu_logits.cpu(), rtol=1e-4, atol=1e-5)
                        torch.testing.assert_close(cpu_hidden, gpu_hidden.cpu(), rtol=1e-4, atol=1e-5)
                        requests.append((b, command, observation, actions, Action("end_turn")))
                    outputs = engine.infer(requests)
                    for index, output in zip(indices, outputs):
                        if command == "observe":
                            reference[index].observe_step(observation, actions, Action("end_turn"))
                        else:
                            self.assertEqual(reference[index].act(observation, actions), output[0])
                        torch.testing.assert_close(reference[index].hidden, shared[index].hidden.cpu(), rtol=1e-4, atol=1e-5)
                shared[0].reset(context(initial))
                self.assertIsNone(shared[0].hidden)
                self.assertIsNotNone(shared[1].hidden)

    def test_shared_service_preserves_ordinary_match_replays(self):
        configure(17)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "policy.pt"
            policy = Policy("C2")
            torch.save(dict(variant="C2", model=policy.state_dict(), config={}), checkpoint)
            initial = scenario()
            case = dict(scenario=initial, armies=context(initial)["armies"], learner_side=0,
                        max_rounds=3, split="validation", agent_seeds=[42, 43], case_id=0,
                        parent_world=0, placement="test", map_size=10)
            case["case_hash"] = digest(case)
            cfg = load_config("configs/rl-ppo-after-target-bc.json")
            historical, fresh = root / "historical", root / "fresh"
            write_json(historical / "run.json", dict(kind=cfg["kind"]))
            write_json(fresh / "run.json", dict(evaluation_runtime_default="auto"))
            self.assertEqual(runtime_for(historical, cfg).device, "cpu")
            self.assertEqual(runtime_for(fresh, cfg).device, "cuda" if torch.cuda.is_available() else "cpu")
            cpu = root / "cpu"
            expected = evaluation_task((cfg, case, str(checkpoint), str(cpu)))
            tasks = [(cfg, case, str(checkpoint), str(root / f"shared-{i}")) for i in range(2)]
            service = EvaluationExecutor(checkpoint, EvaluationRuntime("cpu", workers=2))
            actual = list(service.run(evaluation_task, tasks))
            self.assertEqual(len(actual), 2)
            for index in range(2):
                replay = read_json(root / f"shared-{index}/replay.json")
                original = read_json(cpu / "replay.json")
                self.assertEqual(replay, original)
                self.assertEqual(actual[index]["winner"], expected["winner"])
            self.assertEqual(len(service.connections), 0)
            # A service OOM after a completed case must preserve it and release
            # blocked workers. Retry uses the same backend, without CPU fallback.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            runtime = EvaluationRuntime(device, workers=1)
            failed = EvaluationExecutor(checkpoint, runtime)
            infer = failed.inference.infer
            first, second = root / "partial-first", root / "partial-second"

            def fail_after_first(requests):
                if (first / "metrics.json").exists():
                    raise torch.cuda.OutOfMemoryError("injected evaluation OOM")
                return infer(requests)

            failed.inference.infer = fail_after_first
            with self.assertRaises(RuntimeError):
                list(failed.run(evaluation_task, [(cfg, case, str(checkpoint), str(first)),
                                                  (cfg, case, str(checkpoint), str(second))]))
            self.assertTrue((first / "metrics.json").is_file())
            self.assertFalse((second / "metrics.json").exists())
            retry = EvaluationExecutor(checkpoint, runtime)
            list(retry.run(evaluation_task, [(cfg, case, str(checkpoint), str(second))]))
            self.assertEqual(read_json(second / "metrics.json")["winner"], expected["winner"])
            # The public evaluator must not serve a GPU request from CPU cache.
            plan = dict(development=[case])
            with evaluation_options(device="cpu", workers=1):
                cpu_point = evaluate(cfg, plan, root, "policy.pt", "development")
            if torch.cuda.is_available():
                with evaluation_options(device="cuda", workers=1):
                    cuda_point = evaluate(cfg, plan, root, "policy.pt", "development")
                    cached = evaluate(cfg, plan, root, "policy.pt", "development")
                self.assertEqual(cuda_point, cached)
                self.assertNotEqual(cpu_point["runs"]["argmax"]["cp"]["directory"],
                                    cuda_point["runs"]["argmax"]["cp"]["directory"])
                self.assertEqual(cuda_point["runtime"]["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
