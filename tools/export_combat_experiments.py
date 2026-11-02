"""Export the three combos regressions as kernel-verified viewer replays."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from civ5sim import Battle
from civ5sim.engine import digest


def main():
    source = ROOT / "artifacts/combat-ai/experiments-v1"
    inputs = json.loads((source / "inputs.json").read_text())
    variants = json.loads((source / "results.json").read_text())["summary"]["variants"]
    for name in ("differential14", "differential33", "differential37"):
        case = variants["base"]["cases"][name]
        initial = inputs[name]["snapshot"]
        steps = []
        with Battle(initial) as battle:
            start = battle.snapshot()
            for action in case["validation"]["commands"]:
                result = battle.step(action)
                steps.append({"action": action, "result": asdict(result),
                              "state_sha256": digest(battle.snapshot())})
            final = battle.snapshot()
        assert digest(final) == case["validation"]["final_sha256"]
        experiment = {
            "title": name,
            "description": "Синтетический тест 3×3. Показан найденный план base, а не полный матч. "
                "При тех же лимитах combos вернул пустой план: 180 позиций, 0 завершённых планов. "
                "Противник здесь не выполняет ответный ход. Индекс клетки = y × 8 + x; координаты с нуля.",
            "target": inputs[name]["request"]["target"],
            "base_stats": case["result"]["stats"],
            "damage": case["validation"]["damage"],
            "own_damage": case["validation"]["own_damage"],
        }
        path = ROOT / "build/combat-ai-v1-final/experiments" / name
        path.mkdir(parents=True, exist_ok=True)
        replay = {"schema": 1, "kind": "civ5sim.replay", "initial": start,
                  "steps": steps, "experiment": experiment}
        metadata = {"initial_state_sha256": digest(start), "final_state_sha256": digest(final),
                    "agents": ["base: найденный план", "Без ответного хода"],
                    "commands": len(steps), "status": "experiment", "outcome": "experiment",
                    "config": {"seed": int(name[-2:]), "fog": start["state"]["fog"],
                               "random_damage": False, "armies": [[0, 1, 2], [3, 4, 5]]}}
        for filename, value in (("replay.json", replay), ("result.json", metadata)):
            (path / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, len(steps), "commands")


if __name__ == "__main__":
    main()
