"""Paired CP reset ablation; independent shadow advice never acts on the battle."""

import json
import random
import statistics
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from civ5map.maps import components, load
from civ5rl.protocols.corpus import ScenarioFactory, validate_corpus
from civ5rl.protocols.imitation import dump_episode, file_hash, make_case, search_config
from civ5rl.protocols.imitation_own_states import ReplanningCPAdvisor
from civ5rl.provenance.common import ROOT, read_json, seed_for, write_json

from . import MatchConfig
from .combat_ai import AssignmentKind as K
from .combat_ai import CallPlanningServices
from .cp_agent import CPAgent, hex_distance
from .engine import Action, Battle, digest
from .evaluation import verify_replay
from .match import run_match


def task_view(agent):
    task = agent._task
    if task is None:
        return None
    return {"kind": task["kind"], "target": task["target"],
            "selected": list(task["selected"])}


def public_memory(agent):
    return asdict(agent.memory), agent._last_seen, agent._seen_at


class StatelessCP(ReplanningCPAdvisor):
    """Exactly the own-state advisor, now executing its own recommendations."""

    def _issue(self, action):
        self.issued_task = task_view(self)
        self.issued_searches = self._searches
        return super()._issue(action)

    def act_with_planning(self, observation, actions, planning):
        return self.advise(observation, actions, planning)

    def on_action_result(self, action, result, observation):
        self.observe_actual_result(action, result, observation)


def queued_action(agent, observation):
    own = {u["id"]: u for u in observation["units"] if u["side"] == agent.side}
    for entry in agent._queue:
        if entry.kind in (K.INITIAL, K.BLOCKED, K.RESTART, K.FINISH_TEMP, K.SWAP_REVERSE):
            continue
        if entry.kind == K.FINISH:
            unit = own.get(entry.unit)
            if unit is None or unit["moves"] <= 0 or unit["waiting"]:
                continue
            return Action("skip", entry.unit)
        return Action({K.MOVE: "move", K.MELEE: "melee", K.RANGED: "ranged", K.SWAP: "swap"}[entry.kind],
                      entry.unit, entry.command_target)
    return None


class ObservedCP(CPAgent):
    def __init__(self, config, shadow=True):
        super().__init__(config)
        self.shadow = StatelessCP(config) if shadow else None

    def reset(self, context):
        super().reset(context)
        if self.shadow:
            self.shadow.reset(deepcopy(context))
        self.rows, self.callbacks = [], []
        self.last_reason = None

    def close(self):
        super().close()
        if getattr(self, "shadow", None):
            self.shadow.close()

    def act_with_planning(self, observation, actions, planning):
        before = task_view(self)
        expected = queued_action(self, observation)
        candidate = (expected is not None and self._turn == observation["turn"]
                     and not self._replan and self.last_reason == "matched"
                     and self._last_observation == observation)
        # _events is drained by the ordinary match organizer after each call.
        event_start = len(self._events)
        started = perf_counter()
        action = super().act_with_planning(observation, actions, planning)
        real_seconds = perf_counter()-started
        events = self._events[event_start:]
        continued = (candidate and action == expected and not any(
            e["event"] in ("plan_cancelled", "task_finished", "task_assigned", "search", "search_reused", "support")
            for e in events))
        actual_task = task_view(self)
        shadow_action = shadow_task = None
        shadow_seconds = 0.0
        if self.shadow:
            service = CallPlanningServices(self.shadow.planner, planning._state, self.side, observation["turn"])
            started = perf_counter()
            try:
                shadow_action = self.shadow.act_with_planning(deepcopy(observation), actions, service)
            finally:
                service.expire()
            shadow_seconds = perf_counter()-started
            shadow_task = self.shadow.issued_task
            if public_memory(self) != public_memory(self.shadow):
                raise AssertionError("Shadow/public CP memory diverged")
            if shadow_action not in actions:
                raise AssertionError("Illegal shadow advice")
        roster = set(actual_task["selected"]) if actual_task else set()
        fresh = set(shadow_task["selected"]) if shadow_task else set()
        row = {
            "decision": len(self.rows), "turn": observation["turn"],
            "legal_count": len(actions), "previous_result": self.last_reason,
            "continued_queue": bool(continued), "stateful_action": asdict(action),
            "queued_action": asdict(expected) if expected else None,
            "stateless_action": asdict(shadow_action) if shadow_action else None,
            "stateful_task": actual_task, "previous_task": before, "stateless_task": shadow_task,
            "group_jaccard": len(roster & fresh)/len(roster | fresh) if roster | fresh else None,
            "same_group": roster == fresh, "same_task": actual_task == shadow_task,
            "stateful_seconds": real_seconds, "shadow_seconds": shadow_seconds,
            "observation_sha256": digest(observation), "public_memory_equal": bool(self.shadow),
        }
        self.rows.append(row)
        return action

    def on_action_result(self, action, result, observation):
        before = task_view(self)
        start = len(self._events)
        super().on_action_result(action, result, observation)
        events = self._events[start:]
        result_event = next(e for e in events if e["event"] == "action_result")
        self.last_reason = result_event["reason"]
        self.callbacks.append({"decision": len(self.rows)-1, "reason": self.last_reason,
                               "before": before, "after": task_view(self)})
        if self.shadow:
            self.shadow.on_action_result(action, result, observation)
            if public_memory(self) != public_memory(self.shadow):
                raise AssertionError("Post-command public memories diverged")


class TrackedStatelessCP(StatelessCP):
    def reset(self, context):
        super().reset(context)
        self.action_seconds = []

    def act_with_planning(self, observation, actions, planning):
        start = perf_counter()
        action = super().act_with_planning(observation, actions, planning)
        self.action_seconds.append(perf_counter()-start)
        return action


def large_case(config, maps, world, index, n, contact):
    """Same roster sampler; expand dense contact placement radius if necessary."""
    namespace = "cp_reset_study_16"
    streams = {name: seed_for(config["seed"], index, name, namespace=namespace)
               for name in ("roster", "placement", "battle")}
    roster = ScenarioFactory._sample_roster(random.Random(streams["roster"]), (n, n))
    options = sorted((m for m in maps if m["parent_seed"] == world), key=lambda e:e["path"])
    entry = options[(index//57) % len(options)]
    m = load(ROOT/entry["path"])
    group = sorted(components(m)[0])
    rng = random.Random(streams["placement"])
    rejects = Counter()
    for attempt in range(10000):
        radius = 2
        if contact:
            a = rng.choice(group)
            anchors = [t for t in group if hex_distance(m, a, t) == 2]
            if not anchors:
                rejects["no_anchor"] += 1
                continue
            b = rng.choice(anchors)
            radius = 2 + min(attempt//500, 3)
            pools = [[t for t in group if t not in (a,b) and hex_distance(m, anchor, t) <= radius]
                     for anchor in (a,b)]
            if min(map(len,pools)) < n-1:
                rejects["contact_capacity"] += 1
                continue
            first = [a, *rng.sample(pools[0],n-1)]
            second_pool = [t for t in pools[1] if t not in first]
            if len(second_pool) < n-1:
                rejects["contact_overlap"] += 1
                continue
            positions = [first,[b,*rng.sample(second_pool,n-1)]]
            for row in positions:
                rng.shuffle(row)
        else:
            pools = [[y*16+x for x,y in zone] for zone in m["spawn_candidates"]]
            # Keep zones disjoint; extend too-small zones toward their centre.
            if min(map(len,pools)) < n:
                centres = [min(pool) for pool in pools]
                assigned = [[],[]]
                for tile in group:
                    side = min((0,1),key=lambda s:(hex_distance(m,centres[s],tile),s))
                    assigned[side].append(tile)
                pools = [sorted(pool,key=lambda t:hex_distance(m,centres[s],t))[:max(n,len(pools[s]))]
                         for s,pool in enumerate(assigned)]
            positions = [rng.sample(pool,n) for pool in pools]
        scenario = {"schema":1,"kind":"civ5sim.scenario","map":m,"fog":True,"random_damage":True,
                    "active_side":0,"seed":streams["battle"],
                    "units":[{"type":k,"side":s,"tile":t,"hp":100} for s in (0,1)
                             for k,t in zip(roster,positions[s])]}
        with Battle(scenario) as battle:
            seen = [any(u["side"] != s for u in battle.observe(s)["units"]) for s in (0,1)]
        if (contact and all(seen)) or (not contact and not any(seen)):
            return {"scenario":scenario,"armies":[roster,roster.copy()],"parent_world":world,
                    "placement":"contact" if contact else "separated","map_entry":entry,
                    "agent_seeds":[seed_for(17,index,"agent",s,namespace) for s in (0,1)],
                    "placement_attempts":attempt+1,"placement_radius":radius,
                    "rejections":dict(rejects),"streams":streams}
        rejects["visibility"] += 1
    raise RuntimeError(f"Cannot place world={world} army={n} contact={contact}: {dict(rejects)}")


def prepare(config, output):
    output = Path(output)
    if (output/"manifest.json").exists():
        return check_identity(config,output)
    parent = read_json(ROOT/config["imitation_manifest"])
    corpus = read_json(ROOT/config["corpus"])
    if file_hash(ROOT/config["corpus"]) != parent["source_corpus_sha256"]:
        raise ValueError("Source corpus manifest changed")
    worlds = parent["worlds"]["training"]
    maps = [e for e in corpus["maps"] if e["split"] == "train" and e["parent_seed"] in worlds
            and e["size"] in (10,16)]
    validate_corpus({"maps":maps})
    cases = []
    if config["small_layouts_per_world"] != 2:
        raise ValueError("Study requires one contact and one separated layout per world")
    for world_index in range(len(worlds)):
        used = set()
        for contact in (True,False):
            # Select old deterministic case IDs, without changing their seeds or generation.
            index = next(world_index+visit*len(worlds) for visit in range(4)
                         if ((world_index+visit*len(worlds))//2)%2 == (0 if contact else 1)
                         and world_index+visit*len(worlds) not in used)
            used.add(index)
            case = make_case(config,parent,"training",index)
            case.update(cohort="corpus-10",layout_id=len(cases),army_size=2)
            cases.append(case)
    counts = []
    for n in config["large_sizes"]:
        counts.extend((n,j%2 == 0) for j in range(config["large_layouts_per_size"]))
    random.Random(config["seed"]).shuffle(counts)
    for i,(n,contact) in enumerate(counts):
        case = large_case(config,[e for e in maps if e["size"]==16],worlds[i%len(worlds)],i,n,contact)
        case.update(cohort="large-16",layout_id=i,army_size=n)
        cases.append(case)
    for case in cases:
        case["case_hash"] = digest({k:v for k,v in case.items() if k != "case_hash"})
    sources = sorted({*ROOT.glob("civ5sim/*.py"), ROOT/"civ5map/maps.py",
                      ROOT/"civ5rl/protocols/corpus.py", ROOT/"civ5rl/protocols/imitation.py",
                      ROOT/"civ5rl/protocols/imitation_own_states.py", ROOT/"civ5rl/provenance/common.py",
                      ROOT/"civ5rl/core/observation.py", ROOT/"tools/study_cp_reset.py",
                      ROOT/"configs/cp-reset-study.json", *ROOT.glob("build/combat*.dll")})
    files = {str(p.relative_to(ROOT)).replace("\\","/"):file_hash(p) for p in sources}
    for path in (config["corpus"],config["imitation_manifest"],*[e["path"] for e in maps]):
        files[path] = file_hash(ROOT/path)
    import platform
    manifest = {"schema":1,"config":config,"cases":cases,"worlds":worlds,"files":files,
                "environment":{"python":platform.python_version(),"platform":platform.platform()},
                "protocol_sha256":file_hash(ROOT/"docs/cp-reset-study.md")}
    manifest["hash"] = digest(manifest)
    write_json(output/"manifest.json",manifest)
    write_json(output/"status.json",{"status":"prepared","cases":len(cases),"matches":len(cases)*2})
    return manifest


def check_identity(config,output):
    manifest = read_json(Path(output)/"manifest.json")
    if digest({k:v for k,v in manifest.items() if k != "hash"}) != manifest["hash"]:
        raise ValueError("Study manifest hash changed")
    if manifest["config"] != config:
        raise ValueError("Study config changed")
    import platform
    if manifest["environment"] != {"python":platform.python_version(),"platform":platform.platform()}:
        raise ValueError("Study environment changed")
    for path,expected in manifest["files"].items():
        if file_hash(ROOT/path) != expected:
            raise ValueError(f"Study execution identity changed: {path}")
    return manifest


def run_task(task):
    config,case,side,output,shadow = task
    directory = Path(output)/case["cohort"]/f"layout-{case['layout_id']:03d}"/f"stateless-{side}"
    if not shadow:
        directory = directory/"without-shadow-check"
    directory.mkdir(parents=True,exist_ok=True)
    normal = ObservedCP(search_config(config),shadow=shadow)
    reset = TrackedStatelessCP(search_config(config))
    agents = [normal,normal]
    agents[side],agents[1-side] = reset,normal
    write_json(directory/"case.json",case)
    started = perf_counter()
    try:
        outcome = run_match(MatchConfig(case["scenario"]["map"],case["armies"],config["max_rounds"],
                                       max_calls_per_turn=config["max_calls_per_turn"]),
                            agents,scenario=case["scenario"],agent_seeds=case["agent_seeds"])
        seconds = perf_counter()-started
        outcome.save(directory)
        dump_episode(directory/"comparisons.json.gz",{"rows":normal.rows,"callbacks":normal.callbacks})
        if outcome.result["status"] != "completed":
            raise RuntimeError(f"Match error {directory}: {outcome.result['error']}")
        verify_replay(outcome)
        result = {"cohort":case["cohort"],"layout_id":case["layout_id"],"world":case["parent_world"],
                  "army_size":case["army_size"],"placement":case["placement"],"stateless_side":side,
                  "winner":outcome.result["winner"],"outcome":"D" if outcome.result["winner"] is None
                  else "W" if outcome.result["winner"] == side else "L",
                  "reason":outcome.result["reason"],"rounds":outcome.result["completed_rounds"],
                  "commands":outcome.result["commands"],"remaining_hp":outcome.result["remaining_hp"],
                  "seconds":seconds,"stateful_seconds":sum(r["stateful_seconds"] for r in normal.rows),
                  "shadow_seconds":sum(r["shadow_seconds"] for r in normal.rows),
                  "stateless_seconds":sum(reset.action_seconds),"stateful_calls":len(normal.rows),
                  "stateless_calls":len(reset.action_seconds),"replay_verified":True,
                  "trajectory_sha256":digest({k:v for k,v in outcome.replay.items() if k != "ai_events"}),
                  "directory":str(directory.relative_to(ROOT)).replace("\\","/")}
        write_json(directory/"metrics.json",result)
        return result
    except BaseException:
        import traceback
        (directory/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        normal.close()
        reset.close()


def fraction(a,b):
    return a/b if b else None


def interval(world_values,config):
    """Ratio of sums, resampling whole parent worlds with all their matches."""
    values = np.asarray(list(world_values.values()),dtype=float)
    if not len(values):
        return None
    indices = np.random.default_rng(config["bootstrap_seed"]).integers(0,len(values),
                                               size=(config["bootstrap_samples"],len(values)))
    totals = values[indices].sum(1)
    ratios = np.divide(totals[:,0],totals[:,1],out=np.full(len(totals),np.nan),where=totals[:,1]!=0)
    return np.nanquantile(ratios,[0.025,0.975]).tolist()


def battle_summary(rows,config):
    counts = Counter(r["outcome"] for r in rows)
    worlds = defaultdict(lambda:[0,0])
    score_worlds = defaultdict(lambda:[0,0])
    diff_worlds = defaultdict(lambda:[0,0])
    for r in rows:
        w = r["world"]
        worlds[w][0] += r["outcome"] == "W"
        worlds[w][1] += 1
        score_worlds[w][0] += {"W":1,"D":0.5,"L":0}[r["outcome"]]
        score_worlds[w][1] += 1
        diff_worlds[w][0] += {"W":1,"D":0,"L":-1}[r["outcome"]]
        diff_worlds[w][1] += 1
    return {"matches":len(rows),"worlds":len(worlds),"W":counts["W"],"D":counts["D"],"L":counts["L"],
            "win_rate":fraction(counts["W"],len(rows)),"win_rate_ci":interval(worlds,config),
            "score":fraction(counts["W"]+0.5*counts["D"],len(rows)),"score_ci":interval(score_worlds,config),
            "win_minus_loss":fraction(counts["W"]-counts["L"],len(rows)),"win_minus_loss_ci":interval(diff_worlds,config),
            "mean_rounds":statistics.mean(r["rounds"] for r in rows),
            "mean_hp_stateless":statistics.mean(r["remaining_hp"][r["stateless_side"]] for r in rows),
            "mean_hp_stateful":statistics.mean(r["remaining_hp"][1-r["stateless_side"]] for r in rows)}


def command_summary(selected,config):
    worlds = defaultdict(lambda:[0,0])
    match_fractions = []
    kinds = Counter()
    groups, targets = [], []
    different = type_diff = unit_diff = 0
    for match,rows in selected:
        count = sum(r["stateful_action"] != r["stateless_action"] for r in rows)
        different += count
        if rows:
            match_fractions.append(count/len(rows))
            worlds[match["world"]][0] += count
            worlds[match["world"]][1] += len(rows)
        for r in rows:
            a,b = r["stateful_action"],r["stateless_action"]
            type_diff += a["kind"] != b["kind"]
            unit_diff += a["unit"] != b["unit"]
            kinds[f"{a['kind']} -> {b['kind']}"] += 1
            if r["stateful_task"] is not None:
                groups.append((r["same_group"],r["group_jaccard"]))
                targets.append(r["stateless_task"] is not None and
                               r["stateless_task"]["target"] == r["stateful_task"]["target"])
    n = sum(len(rows) for _,rows in selected)
    per_world = [v[0]/v[1] for v in worlds.values()]
    return {"decisions":n,"different":different,"pooled_rate":fraction(different,n),
            "clustered_pooled_ci":interval(worlds,config),"mean_per_match":statistics.mean(match_fractions) if match_fractions else None,
            "mean_per_world":statistics.mean(per_world) if per_world else None,
            "matches_with_denominator":len(match_fractions),"matches_without_denominator":len(selected)-len(match_fractions),
            "type_different":type_diff,"unit_different":unit_diff,"type_rate":fraction(type_diff,n),
            "unit_rate":fraction(unit_diff,n),"same_group_rate":fraction(sum(s for s,_ in groups),len(groups)),
            "mean_group_jaccard":statistics.mean(j for _,j in groups) if groups else None,
            "same_target_rate":fraction(sum(targets),len(targets)),"group_comparisons":len(groups),
            "action_transitions":dict(kinds)}


def report(config,output):
    from civ5rl.protocols.imitation import load_episode
    output = Path(output)
    manifest = check_identity(config,output)
    rows = [read_json(p) for p in sorted(output.glob("*/layout-*/stateless-*/metrics.json"))]
    if not rows:
        return
    result = {"manifest_hash":manifest["hash"],"matches":len(rows),"cohorts":{}}
    lines = ["# Stateful CP против CP со сбросом тактики", "",
             "Все W/D/L ниже — с точки зрения stateless CP. Ничьи не считаются победами.", "",
             "SearchConfig, привилегии CP, правила, HP=100, fog/random damage и 30 раундов одинаковы.",
             "Каждая расстановка играется обеими сторонами. 95% CI: 10 000 bootstrap по исходным мирам.", ""]
    for cohort in ("corpus-10","large-16"):
        matches = [r for r in rows if r["cohort"]==cohort]
        if not matches:
            continue
        comparisons = [(r,load_episode(ROOT/r["directory"]/"comparisons.json.gz")) for r in matches]
        by_reason = defaultdict(Counter)
        selection = {}
        for label,predicate in (
            ("all_free",lambda r:r["legal_count"]>=2),
            ("valid_queue_free",lambda r:r["continued_queue"] and r["legal_count"]>=2),
            ("valid_combat_queue_free",lambda r:r["continued_queue"] and r["legal_count"]>=2
             and r["stateful_task"] is not None and r["stateful_task"]["kind"]=="combat"),
            ("valid_other_queue_free",lambda r:r["continued_queue"] and r["legal_count"]>=2
             and (r["stateful_task"] is None or r["stateful_task"]["kind"]!="combat")),
        ):
            selection[label] = command_summary([(m,[r for r in c["rows"] if predicate(r)]) for m,c in comparisons],config)
        for _,comparison in comparisons:
            for cb in comparison["callbacks"]:
                if cb["reason"] in ("matched","turn_ended") or cb["before"] is None or cb["before"]["kind"]!="combat":
                    continue
                counter = by_reason[cb["reason"]]
                counter["events"] += 1
                counter["task_retained"] += cb["after"] is not None and cb["after"]["kind"] == "combat"
                counter["roster_retained"] += cb["after"] is not None and cb["before"]["selected"] == cb["after"]["selected"]
        battle = battle_summary(matches,config)
        slices = {field:{str(value):battle_summary([r for r in matches if r[field]==value],config)
                         for value in sorted({r[field] for r in matches})}
                  for field in ("placement","stateless_side","army_size")}
        result["cohorts"][cohort] = {"battle":battle,"slices":slices,"commands":selection,
                                     "combat_replanning":{k:dict(v) for k,v in by_reason.items()},
                                     "stateful_seconds":sum(r["stateful_seconds"] for r in matches),
                                     "stateless_seconds":sum(r["stateless_seconds"] for r in matches),
                                     "shadow_seconds":sum(r["shadow_seconds"] for r in matches),
                                     "sum_match_seconds":sum(r["seconds"] for r in matches)}
        lines += [f"## {cohort}","",f"Матчи: {len(matches)}; миров: {battle['worlds']}.","",
                  f"**Stateless W/D/L: {battle['W']}/{battle['D']}/{battle['L']}**; победы {battle['win_rate']:.1%}; score {battle['score']:.1%}.",
                  f"Score 95% CI: {battle['score_ci']}; (W−L)/N: {battle['win_minus_loss']:.1%}, CI {battle['win_minus_loss_ci']}.","",
                  "| Решения | N | Другая команда | Доля | Среднее по матчам | Среднее по мирам |", "|---|---:|---:|---:|---:|---:|"]
        for label,s in selection.items():
            if s["decisions"]:
                lines += [f"| {label} | {s['decisions']} | {s['different']} | {s['pooled_rate']:.1%} | {s['mean_per_match']:.1%} | {s['mean_per_world']:.1%} |"]
        q = selection["valid_queue_free"]
        lines += ["",f"Основной CI расхождения: {q['clustered_pooled_ci']}; матчи без подходящих решений: {q['matches_without_denominator']}.",
                  f"На действующей очереди: другая группа {None if q['same_group_rate'] is None else 1-q['same_group_rate']}; средний Jaccard {q['mean_group_jaccard']}.","",
                  "### Срезы боевой силы", "", "| Срез | N | W/D/L | Win rate | Score |", "|---|---:|---:|---:|---:|"]
        for field,values in slices.items():
            for value,s in values.items():
                lines += [f"| {field}={value} | {s['matches']} | {s['W']}/{s['D']}/{s['L']} | {s['win_rate']:.1%} | {s['score']:.1%} |"]
        lines += ["", "### Состав после отмены боевого плана", "", "| Причина | Случаи | Задача сохранена | Состав сохранён |", "|---|---:|---:|---:|"]
        for reason,c in by_reason.items():
            lines += [f"| {reason} | {c['events']} | {c['task_retained']} | {c['roster_retained']} |"]
        lines += [""]
    lines += ["## Границы вывода", "",
              "Расхождение измерено на траекториях stateful CP против stateless CP. Оно не доказывает ошибочность каждой новой команды.",
              "Сравнивается полный сброс тактического состояния, включая группы, цели, лимиты и попытки; вклад одной только очереди не изолирован.",
              "CI учитывают зависимость решений и обмен сторон по исходным мирам. Малые срезы размеров армии описательные.",
              "Новые training-worlds, обучение сети и закрытые наборы не использовались. Все полные реплеи сохранены и проверены ядром."]
    write_json(output/"summary.json",result)
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result
