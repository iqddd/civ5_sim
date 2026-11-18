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
from civ5rl.engine.evaluation.executor import BatchedInference, EvaluationExecutor, EvaluationRuntime, evaluation_options, runtime_for
from civ5rl.experiments.ppo_target_scale import evaluation_task, evaluate
from civ5rl.provenance.common import configure, read_json, write_json
from civ5sim import Action, Battle
from civ5sim.engine import digest
from tests.test_rl import context, scenario


def public_sequence(task):
    checkpoint,initial,mode,metrics=task
    from civ5rl.engine.evaluation.proxy import evaluation_agent
    agent=evaluation_agent(checkpoint,mode,temperature=.5,collect_selection_metrics=metrics)
    agent.reset(context(initial))
    with Battle(initial) as battle:
        observation=battle.observe(0);actions=ordered_actions(battle.available_actions())
    selected=[]
    for i in range(3):
        selected.append(agent.act(observation,actions))
        if i==0:agent.observe_step(observation,actions,Action('end_turn'))
    return (mode,metrics),(selected,agent.selection_metrics,agent.rng.getstate(),agent.encoder.previous.tolist(),agent.encoder.contacts)


class BatchedEvaluationTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'Resident CUDA hidden states')
    def test_resident_states_preserve_history_across_reuse_and_prefill(self):
        # A recycled slot must not import another battle's memory; a removed
        # agent must return its storage even outside the RPC executor lifecycle.
        from types import SimpleNamespace
        from civ5rl.core.observation import ObservationEncoder
        configure(17, 'cuda')
        policy = Policy('C2').to('cuda').eval()
        engine = BatchedInference(policy, 'cuda', hidden_capacity=1)
        initial = scenario()
        with Battle(initial) as battle:
            encoded = ObservationEncoder(context(initial), 'C2').encode(
                battle.observe(0), ordered_actions(battle.available_actions()))

        def fresh(hidden=None):
            return SimpleNamespace(members=[SimpleNamespace(hidden=hidden)], mode='argmax',
                                   collect_selection_metrics=False)

        a, b = fresh(), fresh()
        expected = {id(a): None, id(b): None}

        @torch.inference_mode()
        def step(agents, command='act'):
            h0 = torch.cat([expected[id(agent)] if expected[id(agent)] is not None else
                            torch.zeros(1, 1, policy.hidden_size, device='cuda') for agent in agents], 1)
            logits, _, hidden = policy(batch_inputs([encoded] * len(agents), 'cuda'), h0)
            result = engine.infer_encoded([(agent, command, encoded) for agent in agents])
            for i, agent in enumerate(agents):
                expected[id(agent)] = hidden[:, i:i+1].clone()
                torch.testing.assert_close(agent.members[0].hidden, expected[id(agent)], rtol=0, atol=0)
                if command == 'act':
                    self.assertEqual(result[i]['index'], int(logits[i].argmax()))
                else:
                    self.assertIsNone(result[i])

        step([a, b]); step([b, a])
        untouched = b.members[0].hidden.clone()
        step([a], 'observe')
        torch.testing.assert_close(b.members[0].hidden, untouched, rtol=0, atol=0)
        prefill = a.members[0].hidden.clone()
        engine.release_agent(a)
        c = fresh(); expected[id(c)] = None
        step([c, b])
        engine.release_agent(c)
        a.members[0].hidden = prefill
        expected[id(a)] = prefill.clone()
        step([a, b])
        # Reset on an existing member also discards the previous recurrent state.
        a.members[0].hidden = None; expected[id(a)] = None
        step([a])

        with torch.inference_mode():
            def transient():
                agent = fresh()
                engine.infer_encoded([(agent, 'observe', encoded)])
            transient()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            for _ in range(20):
                transient()
            torch.cuda.synchronize()
            self.assertEqual(torch.cuda.memory_allocated(), allocated)

    def test_worker_encoded_sampling_forced_and_session_isolation(self):
        # Moving the encoder/RNG to a worker must preserve executed history,
        # sampling and metrics across concurrent matches and pool reuse.
        configure(17,'cpu')
        with tempfile.TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'policy.pt';policy=Policy('C2')
            torch.save(dict(variant='C2',model=policy.state_dict(),config={}),checkpoint)
            tasks=[(str(checkpoint),scenario(),mode,metrics)
                   for mode,metrics in (('argmax',False),('sample',True),('argmax',True))]
            expected=dict(public_sequence(t) for t in tasks)
            with EvaluationExecutor(checkpoint,EvaluationRuntime('cpu',workers=2)) as service:
                for _ in range(2):
                    actual=list(service.run(public_sequence,tasks))
                    # EvaluationExecutor yields completion order.
                    for key,row in actual:
                        reference=expected[key]
                        self.assertEqual(row[0],reference[0]);self.assertEqual(row[2:],reference[2:])
                        for a,b in zip(row[1],reference[1]):
                            for name in a:self.assertAlmostEqual(a[name],b[name],places=6)
                    self.assertEqual(len(actual),len(expected))
                    self.assertEqual(service.shared,{})
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
            infer = failed.inference.infer_encoded
            first, second = root / "partial-first", root / "partial-second"

            def fail_after_first(requests):
                if (first / "metrics.json").exists():
                    raise torch.cuda.OutOfMemoryError("injected evaluation OOM")
                return infer(requests)

            failed.inference.infer_encoded = fail_after_first
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
