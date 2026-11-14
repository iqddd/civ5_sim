"""Project CPU admission around the stock Vast skill's rental lifecycle."""

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path


def normalize_cpu(name):
    name = name.lower().replace("™", "").replace("®", "")
    name = re.sub(r"\b(?:amd|intel|processor)\b|\b\d+(?:st|nd|rd|th) gen\b|\b\d+[- ]cores?\b", "", name)
    name = name.replace("ryzen threadripper", "threadripper")
    return " ".join(name.split())


def cpu_ratings(document):
    ratings = {}
    data = json.loads(Path(document).read_text(encoding="utf-8"))
    for item in data["processors"]:
        name, score = normalize_cpu(item["name"]), item["single_thread"]
        if name in ratings or (score is not None and (type(score) is not int or score <= 0)):
            raise ValueError(f"Invalid/duplicate CPU rating: {item['name']}")
        ratings[name] = score
    return ratings


def admit_offers(offers, ratings, minimum):
    unknown = sorted({offer.get("cpu_name", "") for offer in offers
                      if normalize_cpu(offer.get("cpu_name", "")) not in ratings})
    if unknown:
        raise ValueError("PassMark Single Thread lookup required before selection: " + "; ".join(unknown))
    result = []
    for offer in offers:
        if ratings[normalize_cpu(offer["cpu_name"])] is None:
            continue  # Explicitly researched but unverified model; never borrow another CPU's score.
        score = float(offer.get("cpu_cores_effective") or 0) * 100 * ratings[normalize_cpu(offer["cpu_name"])] / 2067
        if math.isfinite(score) and score >= minimum:
            result.append(offer)
    return result


def rent(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu-score-min", type=float)
    parser.add_argument("--manager-script", type=Path,
                        default=Path.home() / ".codex/skills/vastai-instance-manager/scripts/vastai_instance.py")
    args, stock_args = parser.parse_known_args(argv)
    if args.cpu_score_min is not None and (not math.isfinite(args.cpu_score_min) or args.cpu_score_min <= 0):
        parser.error("--cpu-score-min must be finite and positive")
    spec = importlib.util.spec_from_file_location("civ5rl_vast_manager", args.manager_script)
    manager = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = manager
    spec.loader.exec_module(manager)
    if args.cpu_score_min is not None:
        config = manager.config_from_args(manager.parser().parse_args(["create", *stock_args]))
        select = manager.select_candidates
        document = Path(__file__).resolve().parents[2] / "configs/vast-cpu-ratings.json"

        def filtered(offers, banned):
            affordable = [o for o in select(offers, banned) if manager.price_allowed(o.get("dph_total"), config)]
            try:
                return admit_offers(affordable, cpu_ratings(document), args.cpu_score_min)
            except ValueError as error:
                raise manager.MarketError(str(error)) from error

        manager.select_candidates = filtered
    return manager.main(["create", *stock_args])
