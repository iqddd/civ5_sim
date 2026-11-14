"""Read-only probabilities on an exact recorded public history (no CP calls)."""

import json
import math
import os
import threading
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path


class ViewerPolicies:
    def __init__(self, config_path, project_root):
        self.root = Path(project_root)
        self.models = {}
        if Path(config_path).exists():
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
            for model in config["models"]:
                checkpoint = Path(model["checkpoint"])
                if not checkpoint.is_absolute():
                    checkpoint = self.root / checkpoint
                if model["id"] in self.models:
                    raise ValueError("Duplicate viewer policy id")
                self.models[model["id"]] = {**model, "checkpoint": checkpoint.resolve()}
        self.lock = threading.Lock()
        self.policies = {}
        self.cache = OrderedDict()

    def close(self):
        with self.lock:
            self.cache.clear()
            self.policies.clear()
            # Avoid importing Torch for viewers that never loaded a model.
            import gc
            import sys
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    def catalog(self):
        return [{"id": m["id"], "label": m.get("label", m["id"]),
                 "side": m.get("side", 1), "available": m["checkpoint"].is_file()}
                for m in self.models.values()]

    def probabilities(self, path, model_id, frame, temperature):
        if not math.isfinite(temperature) or not 0 < temperature <= 10:
            raise ValueError("Температура должна быть больше 0 и не больше 10")
        if model_id not in self.models:
            raise ValueError("Модель не настроена для этой панели")
        model = self.models[model_id]
        checkpoint = model["checkpoint"]
        stamp = checkpoint.stat()
        policy_key = (model_id, stamp.st_mtime_ns, stamp.st_size)
        path = Path(path)
        replay_stamp = path.stat()
        key = (policy_key, str(path), replay_stamp.st_mtime_ns, replay_stamp.st_size, frame)
        with self.lock:
            if key not in self.cache:
                self.cache[key] = self._infer(path, model, policy_key, frame)
                while len(self.cache) > 64:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            result = self.cache[key]
            if not result["available"]:
                return result
            import torch

            probabilities = (torch.tensor(result["logits"], dtype=torch.float32) / temperature).softmax(0).tolist()
            order = sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)
            return {k: v for k, v in result.items() if k not in ("logits", "actions")} | {
                "temperature": temperature,
                "actions": [{**result["actions"][i], "probability": probabilities[i],
                             "argmax": i == order[0]} for i in order],
            }

    def _infer(self, path, model, policy_key, frame):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from civ5rl.core.agent import RLAgent
        from civ5rl.core.observation import ordered_actions
        from civ5sim.engine import Action, Battle, digest

        replay = json.loads(path.read_text(encoding="utf-8"))
        if not 0 <= frame <= len(replay["steps"]):
            raise ValueError("Позиция за пределами реплея")
        if frame == len(replay["steps"]):
            return {"available": False, "message": "Бой завершён — следующего действия нет."}
        side = int(model.get("side", 1))
        if side not in (0, 1):
            raise ValueError("Policy side must be 0 or 1")
        turn = replay["initial"]["state"]["turn"] + sum(
            s["action"]["kind"] == "end_turn" for s in replay["steps"][:frame]
        )
        active_side = (replay["initial"]["state"]["side"] + turn - replay["initial"]["state"]["turn"]) % 2
        if active_side != side:
            return {"available": False, "message": f"Сейчас ход стороны {'AB'[active_side]}. Модель настроена за {'AB'[side]}."}
        if policy_key not in self.policies:
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = RLAgent.from_checkpoint(model["checkpoint"], device=device, mode="argmax")
            agent.policy.enable_amp(False)
            # Retain only the current file version of each registered model.
            self.policies = {k: v for k, v in self.policies.items() if k[0] != policy_key[0]}
            self.policies[policy_key] = agent
        agent = self.policies[policy_key]
        result_path = path.with_name("result.json")
        if not result_path.exists():
            raise ValueError("Для восстановления контекста нужен result.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        config = result["config"]
        from civ5rl.core.observation import public_map

        agent.reset({"map": public_map(replay["initial"]["map"]), "side": side,
                     "armies": config["armies"], "max_rounds": config["max_rounds"],
                     "fog": config["fog"], "random_damage": config["random_damage"],
                     "seed": result["seeds"][f"agent{side}"]})

        def inputs(battle):
            observation = battle.observe(side)
            if agent.requires_attack_preview:
                actions, previews = battle.actions_with_previews()
                actions = tuple(actions)
                order = ordered_actions(actions)
                lookup = {action: i for i, action in enumerate(actions)}
                observation["attack_previews"] = [previews[lookup[action]].tolist() for action in order]
                actions = order
            else:
                actions = battle.available_actions()
            return observation, ordered_actions(actions)

        with Battle(replay["initial"]) as battle, torch.inference_mode():
            for number, step in enumerate(replay["steps"][:frame], 1):
                if battle.observe()["active_side"] == side:
                    observation, actions = inputs(battle)
                    agent.observe_step(observation, actions, Action(**step["action"]))
                actual = battle.step(step["action"])
                if asdict(actual) != step["result"] or digest(battle.snapshot()) != step["state_sha256"]:
                    raise ValueError(f"История расходится на команде {number}")
            observation, actions = inputs(battle)
            logits = agent._infer(observation, actions)[0].float().cpu().tolist()
            recorded = replay["steps"][frame]["action"]
            return {"available": True, "frame": frame, "next_command": frame + 1,
                    "model": model["id"], "side": side, "runtime": f"{agent.device.upper()} FP32",
                    "logits": logits,
                    "actions": [{"action": asdict(a), "recorded": asdict(a) == recorded} for a in actions]}
