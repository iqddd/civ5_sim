"""Publish a verified, single-run CP/greedy batch under the existing viewer URLs."""

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from civ5sim.evaluation import write_json
from tools.benchmark_combat_ai import aggregate
from tools.summarize_coordinator_acceptance import engagement_audit, verify_waits


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compressed(path, value):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def publish_available(batch, viewer, archive, summary):
    """Publish only workers with a completed replay-verification record."""
    before = {row["case"]: row for row in read(archive / "before-results.json")}
    passport = read(batch / "passport.json")
    manifest_path = archive / "published.json"
    published = read(manifest_path)["replays"] if manifest_path.exists() else {}
    backup = (ROOT / "build/cp07-greedy-replaced").resolve()
    scratch = (ROOT / "build/cp07-greedy-publish-temp").resolve()
    backup.mkdir(exist_ok=True)
    scratch.mkdir(exist_ok=True)
    added = 0
    for record in summary["records"]:
        name = record["case"]["id"]
        if name in published:
            continue
        if name not in before or set(record["case"]["agents"]) != {"cp", "greedy"}:
            raise ValueError("Unexpected replacement case")
        source = (batch / "matches" / name).resolve()
        target = (viewer / "matches" / name).resolve()
        saved = (backup / name).resolve()
        temporary = (scratch / name).resolve()
        if not (source.is_relative_to(batch / "matches") and target.is_relative_to(viewer / "matches")
                and saved.is_relative_to(backup) and temporary.is_relative_to(scratch)):
            raise ValueError("Replacement escaped its bounded directory")
        result = read(source / "0/result.json")
        checked = read(source / "0/checked.json")
        if result["status"] != "completed" or not checked.get("replay_verified"):
            raise ValueError(f"Unverified replay: {name}")
        if checked["run_signature"] != passport["run_signature"]:
            raise ValueError(f"Wrong replay source version: {name}")
        if result["initial_state_sha256"] != before[name]["result"]["initial_state_sha256"]:
            raise ValueError(f"Initial placement changed: {name}")
        if saved.exists() or temporary.exists():
            raise ValueError(f"Interrupted or duplicate publication: {name}")
        # Copy first, so the benchmark's staging data and later audits remain
        # complete. Only the two final directory renames change visible data.
        shutil.copytree(source, temporary)
        shutil.copyfile(batch / "passport.json", temporary / "passport.json")
        target.rename(saved)
        temporary.rename(target)
        published[name] = hashlib.sha256((target / "0/replay.json").read_bytes()).hexdigest()
        atomic_json(manifest_path, {"replays": published, "viewer_root": str(viewer.relative_to(ROOT)),
                                   "old_folders": str(backup.relative_to(ROOT))})
        added += 1
    if added:
        current = read(viewer / "summary.json")
        replacement = {r["case"]["id"]: r for r in summary["records"] if r["case"]["id"] in published}
        merged = [replacement.get(r["case"]["id"], r) for r in current["records"]]
        new_results = aggregate(list(replacement.values()))
        current.setdefault("mechanical_acceptance_before_recruitment", bool(current.get("acceptance")))
        current.update(acceptance=False, mixed_versions=True, records=merged, summary=aggregate(merged),
            greedy_replacement={"cases": len(published), "expected": 64, "repeats": 1,
                               "run_signature": passport["run_signature"],
                               "outcomes": new_results["outcomes"]["opponent/greedy"]["outcomes"],
                               "report": str(archive.relative_to(ROOT) / "report.md")})
        atomic_json(viewer / "summary.json", current)
        print(f"Published {len(published)}/64 checked matches", flush=True)
    return len(published)


def publish(batch, viewer, archive):
    summary = read(batch / "summary.json")
    records = summary["records"]
    if not summary.get("scope_complete") or summary["failures"] or summary.get("repeats") != 1:
        raise ValueError("The replacement batch has not completed cleanly exactly once")
    if len(records) != 64 or any(set(r["case"]["agents"]) != {"cp", "greedy"} for r in records):
        raise ValueError("Expected exactly the 64 CP/greedy cases")
    before = read(archive / "before-results.json")
    by_name = {r["case"]: r for r in before}
    if set(by_name) != {r["case"]["id"] for r in records}:
        raise ValueError("Case set changed")
    for record in records:
        name = record["case"]["id"]
        old = by_name[name]
        config = record["case"]["config"]
        side = record["case"]["agents"].index("cp")
        if (config["seed"], config["fog"], config["random_damage"], side) != (
                old["seed"], old["fog"], old["random_damage"], old["cp_side"]):
            raise ValueError(f"Scenario changed: {name}")
        path = batch / "matches" / name / "0"
        result = read(path / "result.json")
        if result["initial_state_sha256"] != old["result"]["initial_state_sha256"]:
            raise ValueError(f"Initial placement or RNG changed: {name}")
        if read(path / "checked.json")["run_signature"] != summary["run_signature"]:
            raise ValueError(f"Unchecked replacement: {name}")
    waits = verify_waits(batch, records)
    engagements = engagement_audit(batch, records)
    compressed(archive / "after-records.json.gz", records)
    compressed(archive / "after-matches.json.gz", {
        row["case"]["id"]: {filename: read(batch / "matches" / row["case"]["id"] / "0" / filename)
                             for filename in ("replay.json", "result.json", "checked.json")}
        for row in records})
    compressed(archive / "after-engagements.json.gz", engagements)
    write_json(archive / "after-waiting-audit.json", {"checked": len(waits), "rows": waits})
    shutil.copyfile(batch / "passport.json", archive / "after-passport.json")
    shutil.copyfile(batch / "tactical.json", archive / "after-tactical.json")
    write_json(archive / "after-summary.json", {k: v for k, v in summary.items() if k != "records"})
    after = []
    changes = {"loss_to_win": [], "win_to_loss": [], "other": []}
    for record in records:
        name = record["case"]["id"]
        side = record["case"]["agents"].index("cp")
        result = record["result"]
        outcome = "draw" if result["winner"] is None else "win" if result["winner"] == side else "loss"
        after.append({"case": name, "cp_side": side, "outcome": outcome,
                      "remaining_hp": result["remaining_hp"][side], "remaining_units": result["alive"][side],
                      "result": result})
        old = by_name[name]["outcome"]
        if old != outcome:
            key = "loss_to_win" if (old, outcome) == ("loss", "win") else "win_to_loss" if (old, outcome) == ("win", "loss") else "other"
            changes[key].append(name)
    write_json(archive / "after-results.json", after)
    write_json(archive / "changed-outcomes.json", changes)
    wins = sum(r["outcome"] == "win" for r in after)
    losses = sum(r["outcome"] == "loss" for r in after)
    lines = ["# CP против greedy: отбор атакующей группы", "",
             f"До правки: побед CP — 42, поражений — 22. После: побед — {wins}, поражений — {losses}, ничьих — {64-wins-losses}.",
             "64 разных случая, каждый переигран один раз с прежней расстановкой и seed. Все новые реплеи исполнены и проверены.",
             "Обычные эвристики поиска, бюджеты, туман, урон и greedy сохранены; изменён допуск участников атакующего запроса.",
             "Это сравнение одного и того же набора сценариев; оно не является доказательством оптимальности или общей силы AI.",
             "Остальные 84 матча панели относятся к предыдущей версии; полная приёмка новой версии CP-07 этим поднабором не закрывается.", "",
             "| Матч | Сторона CP | До | После | HP после | Юнитов после |", "|---|---|---|---|---:|---:|"]
    for row in after:
        lines.append(f"| {row['case']} | {'A' if row['cp_side']==0 else 'B'} | {by_name[row['case']]['outcome']} | {row['outcome']} | {row['remaining_hp']} | {row['remaining_units']} |")
    (archive / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    publish_available(batch, viewer, archive, summary)
    print(f"Published64: CP wins={wins}, losses={losses}; no repeated reruns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--batch", type=Path, default=ROOT / "build/cp07-greedy-striking")
    parser.add_argument("--viewer", type=Path, default=ROOT / "build/cp07-coordinator-final")
    parser.add_argument("--archive", type=Path, default=ROOT / "artifacts/combat-ai/greedy-striking-20261102")
    parser.add_argument("--watch", action="store_true", help="Publish checked workers immediately, then archive the final report")
    args = parser.parse_args()
    if args.watch:
        while True:
            try:
                value = read(args.batch / "summary.json")
            except (OSError, ValueError):
                time.sleep(1)
                continue
            publish_available(args.batch.resolve(), args.viewer.resolve(), args.archive.resolve(), value)
            if value.get("schema") == 1:
                break
            time.sleep(1)
    publish(args.batch.resolve(), args.viewer.resolve(), args.archive.resolve())
