// CP-06: neutral-land adaptations of shouldHeal (CvUnit.cpp:28716),
// ExecuteHeals, FindClosestSafePlotForHealing, FindSafestPlotInReach and
// MoveToEmptySpaceNearTarget. Physics and privileged queries stay in model.cpp.
#include "planner.h"
#include "../combat_state.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>

namespace combat_ai {
namespace {
constexpr int unreachable = 1000000;
bool passable(const CaiBattleState& s, int tile) {
  return s.tiles[tile].elevation < 2 && s.tiles[tile].feature != 5;
}
std::vector<int> distances(const RulesAdapter& a, const Branch& b,
                           const TacticalView& v, int target, int side, int source) {
  const auto& s = b.state();
  std::vector<int> result(s.tile_count, unreachable);
  // Match CP's approximate target: destination, ring 1, then ring 2.
  for (int ring = 0; ring <= 2; ++ring) {
    std::fill(result.begin(),result.end(),unreachable);
    std::queue<int> open;
    for (int t = 0; t < s.tile_count; ++t)
      if (passable(s,t) && a.distance(target,t) == ring &&
          (v.occupied[t] < 0 || t == source)) {
        result[t] = 0; open.push(t);
      }
    while (!open.empty()) {
      int t = open.front(); open.pop();
      for (int n : a.neighbors(t))
        if (passable(s,n) && result[n] == unreachable &&
            (v.occupied[n] < 0 || v.units[v.occupied[n]].side == side)) {
          result[n] = result[t] + 1; open.push(n);
        }
    }
    if (result[source] < unreachable) break;
  }
  return result;
}
bool occupied_contact(const RulesAdapter& a, const Branch& b, int tile) {
  for (auto c : b.contacts().known_enemies) {
    CaiUnit enemy{};
    if (a.known_enemy(b,c.id,enemy) && enemy.tile == tile) return true;
  }
  return false;
}
int friends_at(const TacticalView& v, const RulesAdapter& a, int tile, int side, int id) {
  int count = 0;
  for (int n : a.neighbors(tile)) {
    int other = v.occupied[n];
    if (other >= 0 && other != id && v.units[other].side == side) ++count;
  }
  return count;
}
}
Plan support_assignments(const CaiSearchRequest& r, int32_t mode, DecisionTrace* trace) {
  auto adapter = make_rules_adapter(*r.state,r.side);
  ContactMemory memory{r.contacts.turn,r.contacts.side,{}};
  for (int i = 0; i < r.contacts.known_enemy_count; ++i)
    memory.known_enemies.push_back({r.contacts.known_enemies[i].id,
                                   r.contacts.known_enemies[i].last_seen_own_turn});
  auto branch = adapter->root(*r.state,memory);
  auto danger = make_danger_model(*adapter, trace);
  Plan plan{{},{CAI_NO_PLAN,0,{}}};
  std::vector<int> selected(r.selected_units,r.selected_units+r.selected_count);
  if (mode == CAI_APPROACH) {
    TacticalView view(*branch,r.side);
    auto d = distances(*adapter,*branch,view,r.target,r.side,view.units[selected.front()].tile);
    std::stable_sort(selected.begin(),selected.end(),[&](int a,int b) {
      return std::make_pair(d[view.units[a].tile],a) < std::make_pair(d[view.units[b].tile],b);
    });
  }
  for (int id : selected) {
    TacticalView view(*branch,r.side);
    auto u = view.units[id];
    if (u.hp <= 0 || u.moves <= 0 || u.waiting) continue;
    if (mode == CAI_APPROACH && u.tile == r.target) continue;
    int current_danger = danger->incoming(*branch,id,u.tile);
    bool adjacent = false, attack = false;
    for (int e : view.enemies)
      if (adapter->distance(u.tile,view.units[e].tile) == 1) adjacent = true;
    if (mode == CAI_HEAL_BEFORE && u.hp >= 13 && u.hp < 33)
      for (auto c : adapter->actions(*branch))
        if (c.unit == id && (c.kind == combat_rules::MELEE || c.kind == combat_rules::RANGED))
          attack = true;
    bool heal = u.hp < 100 && (mode == CAI_HEAL_BEFORE ?
                  (u.hp < 13 || (u.hp < 33 && !attack)) :
                  (adjacent ? current_danger > u.hp : u.hp < 80));
    if (trace) trace->event("support_heal_eligibility", {{"unit",id},{"mode",mode},{"hp",u.hp},
        {"heal",heal},{"adjacent",adjacent},{"attack",attack},{"danger",current_danger}});
    if (mode != CAI_APPROACH && !heal) continue;
    std::vector<int> route;
    if (mode == CAI_APPROACH) route = distances(*adapter,*branch,view,r.target,r.side,u.tile);
    int best_tile = u.tile;
    auto best = std::make_tuple(unreachable,unreachable,unreachable,unreachable);
    for (auto reachable : adapter->reachable(*branch,id)) {
      int t = reachable.tile;
      if (occupied_contact(*adapter,*branch,t)) continue;
      ++plan.result.stats.checked;
      int incoming = danger->incoming(*branch,id,t);
      int friends = friends_at(view,*adapter,t,r.side,id);
      std::tuple<int,int,int,int> score;
      if (mode == CAI_APPROACH) {
        if (route[t] >= route[u.tile] || incoming >= u.hp) {
          if (trace) trace->event("approach_rejected", {{"unit",id},{"tile",t},{"distance",route[t]},
              {"original_distance",route[u.tile]},{"danger",incoming},{"hp",u.hp}});
          continue;
        }
        score = {route[t],incoming,-adapter->defense(t),t};
      } else {
        // Healing happens only without acting. Moving never earns this heal.
        int healing = t == u.tile && !u.acted ? NEUTRAL_HEAL_RATE : 0;
        int safety = (u.hp + healing - incoming/(friends+1))*10 -
                     adapter->distance(u.tile,t)*2;
        if (current_danger == 0 && t != u.tile) continue;
        // The enemy responds before natural healing at our next turn start.
        if (safety > 0 && incoming < u.hp)
          score = {0,-safety,incoming,t};
        else {
          // CP safest-plot fallback: compressed danger, movement/fortify loss,
          // enemy adjacency and friendly support; no territory/city terms.
          int compressed = incoming > 144 ? 12*int(std::sqrt(double(incoming))) : incoming;
          if (t != u.tile && !u.acted)
            compressed += NEUTRAL_HEAL_RATE + ((u.type!=2 && u.type!=4)?3:0);
          int enemies = 0;
          bool exposed = false;
          for (int e : view.enemies)
            if (adapter->distance(t,view.units[e].tile)==1) ++enemies;
          for (int e : view.enemies)
            if (adapter->can_see(*branch,view.units[e].tile,t)) exposed = true;
          if (exposed) compressed += compressed/4;
          if (enemies) compressed += compressed/10;
          score = {1,compressed+(enemies-friends)*13,incoming,t};
        }
      }
      if (trace) trace->event("support_destination", {{"unit",id},{"tile",t},{"danger",incoming},
          {"rank0",std::get<0>(score)},{"rank1",std::get<1>(score)},{"rank2",std::get<2>(score)}});
      if (score < best) { best = score; best_tile = t; }
    }
    if (trace) trace->event("support_selected_tile", {{"unit",id},{"tile",best_tile}});
    if (mode == CAI_APPROACH && best_tile == u.tile) continue;
    if (best_tile != u.tile) {
      auto transition = adapter->apply(*branch,{combat_rules::MOVE,id,best_tile});
      if (!transition.accepted) continue;
      TacticalView after(*branch,r.side);
      auto moved = after.units[id];
      plan.assignments.push_back({CAI_MOVE,id,u.tile,moved.tile,moved.moves,
                                  CAI_ROLE_NONE,0,0,0,CAI_NO_ID,best_tile});
      bool new_enemy = false;
      for (int e : after.enemies)
        if (std::find(view.enemies.begin(),view.enemies.end(),e)==view.enemies.end()) new_enemy=true;
      if (transition.contact_interrupted || new_enemy) break;
      u = moved;
    }
    plan.assignments.push_back({CAI_FINISH,id,u.tile,u.tile,u.moves,
                                CAI_ROLE_NONE,0,0,0,CAI_NO_ID,CAI_NO_ID});
  }
  plan.result.reason = plan.assignments.empty() ? CAI_NO_PLAN : CAI_COMPLETED;
  plan.result.assignment_count = int32_t(plan.assignments.size());
  return plan;
}

std::vector<int32_t> attack_candidates(const CaiSearchRequest& request, DecisionTrace* trace) {
  // Pinned CvTacticalAI.cpp:4529-4612 FindUnitsWithinStrikingDistance, called
  // by ExecuteAttack (:4143) BEFORE ExecuteAttackWithUnits (:3251-3275).
  // This is a recruitment query, not a tactical plan or an attack guarantee.
  auto adapter = make_rules_adapter(*request.state,request.side);
  ContactMemory memory{request.contacts.turn,request.contacts.side,{}};
  for (int i=0;i<request.contacts.known_enemy_count;++i)
    memory.known_enemies.push_back({request.contacts.known_enemies[i].id,
                                   request.contacts.known_enemies[i].last_seen_own_turn});
  auto root = adapter->root(*request.state,memory);
  TacticalView view(*root,request.side);
  std::vector<int32_t> selected;
  int target_unit = view.occupied[request.target];
  if (target_unit < 0 || view.units[target_unit].side == request.side) return selected;
  constexpr int move_denominator = 60;
  constexpr int nearby_maneuver_radius = 2;
  constexpr int ranged_attack_range = 2; // Current five-type profile.
  for (int i=0;i<request.selected_count;++i) {
    int id = request.selected_units[i];
    const auto& unit = view.units[id];
    if (unit.hp <= 0 || unit.moves <= 0 || unit.waiting) {
      if (trace) trace->event("recruit_unavailable", {{"unit",id},{"hp",unit.hp},{"moves",unit.moves},{"waiting",unit.waiting}});
      continue;
    }
    int distance = adapter->distance(unit.tile,request.target);
    // CP's cheap baseMoves*4 bound precedes pathfinding, not a fixed radius.
    if (distance > (combat_rules::moves[unit.type]/move_denominator)*4) {
      if (trace) trace->event("recruit_distance_bound", {{"unit",id},{"distance",distance}});
      continue;
    }
    bool ranged = combat_rules::ranged[unit.type] > 0;
    bool can_reach = ranged && adapter->can_execute(*root,{combat_rules::RANGED,id,request.target});
    if (!can_reach) {
      for (auto tile : adapter->move_intents(*root,id)) {
        int occupant = view.occupied[tile.tile];
        // Reach intents also contain enemy endpoints for melee candidates.
        // These are not standing/firing plots for this recruitment query.
        if (occupant >= 0 && view.units[occupant].side != request.side) continue;
        int target_distance = adapter->distance(tile.tile,request.target);
        if ((!ranged && target_distance == 1) ||
            (ranged && target_distance >= 1 && target_distance <= ranged_attack_range &&
             adapter->can_target(tile.tile,request.target))) {
          // CP explicitly admits reaching a firing/adjacent plot with no
          // movement left; attacking may only be possible next turn.
          can_reach = true;
          break;
        }
      }
    }
    if (!can_reach && distance > nearby_maneuver_radius) {
      if (trace) trace->event("recruit_unreachable", {{"unit",id},{"distance",distance}});
      continue;
    }
    // CP additionally checks expected ranged damage (target HP/20 capped at3).
    // No city/air/domain branches exist in the supported simulator profile.
    if (ranged) {
      auto forecast = adapter->forecast(*root,id,unit.tile,target_unit,request.target);
      if (forecast.code || forecast.damage < std::min(view.units[target_unit].hp/20,3)) {
        if (trace) trace->event("recruit_insufficient_damage", {{"unit",id},{"damage",forecast.damage},{"error",forecast.code}});
        continue;
      }
    }
    if (trace) trace->event("recruit_admitted", {{"unit",id},{"can_reach",can_reach},{"distance",distance}});
    selected.push_back(id);
  }
  return selected;
}

// CP Only 029e598: PerformOpportunityAttack (5525), stationary ranged
// opportunity (5638), FindSafestPlotInReach (5715), ExecuteMovesToSafestPlot
// (3566). Movement-enabled ranged searches live in CPAgent and share its
// ordinary search budget. Return after a retreat so those searches see real
// damage/contact, rather than a speculative continuation.
Plan recovery_assignments(const CaiSearchRequest& request, CaiSearchDiagnostics& diagnostics, DecisionTrace* trace) {
  Plan plan{{},{CAI_NO_PLAN,0,{}}};
  if (request.selected_count != 1) {
    plan.result.reason = CAI_INVALID_REQUEST;
    return plan;
  }
  auto adapter = make_rules_adapter(*request.state,request.side);
  ContactMemory memory{request.contacts.turn,request.contacts.side,{}};
  for (int i=0;i<request.contacts.known_enemy_count;++i)
    memory.known_enemies.push_back({request.contacts.known_enemies[i].id,
                                   request.contacts.known_enemies[i].last_seen_own_turn});
  auto root = adapter->root(*request.state,memory);
  auto danger = make_danger_model(*adapter, trace);
  int id = request.selected_units[0];
  TacticalView initial(*root,request.side);
  auto original = initial.units[id];
  if (original.hp <= 0 || original.moves <= 0 || original.waiting) return plan;
  auto record = [&](const CaiUnit& before, const Branch& after, Command command,
                    const Transition& transition) {
    const auto now = after.state().units[id];
    auto kind = command.kind == combat_rules::MOVE ? CAI_MOVE :
                command.kind == combat_rules::MELEE ? CAI_MELEE : CAI_RANGED;
    auto assignment = stationary_assignment(kind,before,CAI_ROLE_NONE);
    assignment.to_tile = kind == CAI_MOVE ? now.tile : command.target;
    assignment.command_target = command.target;
    assignment.remaining_moves = now.moves;
    assignment.damage = transition.result.damage;
    assignment.self_damage = transition.result.retaliation;
    if (kind != CAI_MOVE) {
      int enemy = initial.occupied[command.target];
      if (enemy >= 0 && after.state().units[enemy].hp <= 0) assignment.killed_unit = enemy;
    }
    return assignment;
  };
  struct Opportunity {
    int score;
    std::vector<CaiAssignment> assignments;
  };
  std::map<int,Opportunity> melee_targets;
  auto melee = [&](const Branch& source, const std::vector<CaiAssignment>& prefix) {
    TacticalView view(source,request.side);
    const auto before = view.units[id];
    for (auto command : adapter->actions(source)) {
      if (command.unit != id || command.kind != combat_rules::MELEE) continue;
      int victim = initial.occupied[command.target];
      if (victim < 0 || initial.units[victim].side == request.side) continue;
      // CP: no suicide and no single-attacker raid into an enemy cluster.
      auto forecast = adapter->preview(source,command);
      if (forecast.code || forecast.retaliation >= before.hp) {
        if (trace) trace->event("recovery_attack_rejected", {{"unit",id},{"tile",command.target},
            {"error",forecast.code},{"hp",before.hp},{"self_damage",forecast.retaliation}});
        continue;
      }
      bool cluster = false;
      for (int enemy : view.enemies)
        if (adapter->distance(command.target,view.units[enemy].tile) == 1) cluster = true;
      if (cluster) {
        if (trace) trace->event("recovery_enemy_cluster", {{"unit",id},{"tile",command.target}});
        continue;
      }
      int dealt = forecast.damage;
      int received = forecast.retaliation;
      if (dealt >= view.units[victim].hp) dealt += 30;
      // Neutral-land healing; no kill-heal promotions in this roster.
      if (100-before.hp < NEUTRAL_HEAL_RATE/2 && received < 50)
        received = std::max(0,received-NEUTRAL_HEAL_RATE);
      int score = 1000*dealt/(received+10);
      if (trace) trace->event("recovery_exchange", {{"unit",id},{"tile",command.target},
          {"damage",forecast.damage},{"self_damage",forecast.retaliation},{"adjusted_dealt",dealt},
          {"adjusted_received",received},{"score",score}});
      auto branch = source.clone();
      auto transition = adapter->apply(*branch,command);
      ++plan.result.stats.checked;
      if (!transition.accepted) continue;
      auto assignment = record(before,*branch,command,transition);
      std::vector<CaiAssignment> moves = prefix;
      moves.push_back(assignment);
      TacticalView after(*branch,request.side);
      bool contact = false;
      for (int enemy : after.enemies)
        if (std::find(initial.enemies.begin(),initial.enemies.end(),enemy)==initial.enemies.end()) contact = true;
      // After advancing on a kill, CP tries to return to the original plot.
      Command back{combat_rules::MOVE,id,original.tile};
      auto now = after.units[id];
      if (!contact && now.tile != original.tile && adapter->can_execute(*branch,back)) {
        auto returned = adapter->apply(*branch,back);
        ++plan.result.stats.checked;
        if (returned.accepted) moves.push_back(record(now,*branch,back,returned));
      }
      auto found = melee_targets.find(victim);
      if (found == melee_targets.end() || score > found->second.score)
        melee_targets.insert_or_assign(victim,Opportunity{score,std::move(moves)});
    }
  };
  if (!combat_rules::ranged[original.type]) {
    melee(*root,{});
    for (auto command : adapter->actions(*root)) {
      if (command.unit != id || command.kind != combat_rules::MOVE) continue;
      auto branch = root->clone();
      auto transition = adapter->apply(*branch,command);
      ++plan.result.stats.checked;
      if (!transition.accepted || transition.contact_interrupted) continue;
      TacticalView next(*branch,request.side);
      if (next.enemies != initial.enemies) continue;
      auto movement = record(original,*branch,command,transition);
      if (movement.remaining_moves <= 0) continue;
      melee(*branch,{movement});
    }
  }
  const Opportunity* best_attack = nullptr;
  for (const auto& target : melee_targets)
    if (!best_attack || target.second.score > best_attack->score) best_attack = &target.second;
  // CP increments the required exchange score for each eligible enemy.
  if (best_attack && best_attack->score >= 1000*int(melee_targets.size())) {
    plan.assignments = best_attack->assignments;
    diagnostics.recovery_kind = CAI_RECOVERY_ATTACK;
  } else {
    // CP's zero-danger list precedes its danger list. Without owned cities,
    // city-distance ties prefer staying, then tile ID. Cities, embarkation, civilian cover and
    // pushing allied blockers are outside this simulator's supported profile.
    int best_tile = original.tile;
    auto safety_rank = std::make_tuple(2,std::numeric_limits<int>::max(),1,0);
    auto destinations = adapter->reachable(*root,id);
    destinations.push_back({original.tile,original.moves});
    for (auto reachable : destinations) {
      int tile = reachable.tile;
      if (tile != original.tile && occupied_contact(*adapter,*root,tile)) continue;
      int incoming = danger->incoming(*root,id,tile);
      ++plan.result.stats.checked;
      int compressed = incoming > 144 ? 12*int(std::sqrt(double(incoming))) : incoming;
      if (tile != original.tile && !original.acted)
        compressed += NEUTRAL_HEAL_RATE + ((original.type!=2 && original.type!=4)?3:0);
      compressed += 12; // All supported plots are neutral territory.
      int enemies = 0;
      bool exposed = false;
      for (int enemy : initial.enemies) {
        enemies += adapter->distance(tile,initial.units[enemy].tile)==1;
        exposed = exposed || adapter->can_see(*root,initial.units[enemy].tile,tile);
      }
      if (exposed) compressed += compressed/4;
      if (enemies) compressed += compressed/10;
      int friends = friends_at(initial,*adapter,tile,request.side,id);
      auto rank = std::make_tuple(incoming == 0 ? 0 : 1,
          incoming == 0 ? 0 : compressed+(enemies-friends)*13,
          tile == original.tile ? 0 : 1,tile);
      if (trace) trace->event("recovery_destination", {{"unit",id},{"tile",tile},{"hp",original.hp},
          {"incoming",incoming},{"compressed",compressed},{"enemies",enemies},{"exposed",exposed},
          {"friends",friends},
          {"zero_danger",incoming == 0},{"score",std::get<1>(rank)},{"remaining",reachable.remaining_moves}});
      if (rank < safety_rank) { safety_rank = rank; best_tile = tile; }
    }
    if (trace) trace->event("recovery_selected_tile", {{"unit",id},{"tile",best_tile}});
    if (best_tile != original.tile) {
      Command command{combat_rules::MOVE,id,best_tile};
      auto transition = adapter->apply(*root,command);
      if (transition.accepted) {
        plan.assignments.push_back(record(original,*root,command,transition));
        diagnostics.recovery_kind = CAI_RECOVERY_RETREAT;
      }
    }
    if (plan.assignments.empty()) {
      // CP ExecuteMovesToSafestPlot: if staying, shoot only above half HP.
      // Pick damage + kill bonus, rather than treating every kill as superior.
      Command shot{};
      int best_damage = 0;
      if (combat_rules::ranged[original.type] && original.hp > 50) {
        for (auto command : adapter->actions(*root)) {
          if (command.unit != id || command.kind != combat_rules::RANGED) continue;
          auto forecast = adapter->preview(*root,command);
          int enemy = initial.occupied[command.target];
          if (forecast.code || enemy < 0) continue;
          int score = forecast.damage + (forecast.damage >= initial.units[enemy].hp ? 30 : 0);
          if (score > best_damage) { best_damage = score; shot = command; }
        }
      }
      if (best_damage > 0) {
        auto transition = adapter->apply(*root,shot);
        ++plan.result.stats.checked;
        if (transition.accepted) {
          plan.assignments.push_back(record(original,*root,shot,transition));
          diagnostics.recovery_kind = CAI_RECOVERY_ATTACK;
        }
      }
      if (plan.assignments.empty()) {
        plan.assignments.push_back(stationary_assignment(CAI_FINISH,original,CAI_ROLE_NONE));
        diagnostics.recovery_kind = CAI_RECOVERY_WAIT;
      }
    }
  }
  plan.result.reason = CAI_COMPLETED;
  plan.result.assignment_count = int32_t(plan.assignments.size());
  return plan;
}
} // namespace combat_ai
