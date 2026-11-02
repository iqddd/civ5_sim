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
Plan support_assignments(const CaiSearchRequest& r, int32_t mode) {
  auto adapter = make_rules_adapter(*r.state,r.side);
  ContactMemory memory{r.contacts.turn,r.contacts.side,{}};
  for (int i = 0; i < r.contacts.known_enemy_count; ++i)
    memory.known_enemies.push_back({r.contacts.known_enemies[i].id,
                                   r.contacts.known_enemies[i].last_seen_own_turn});
  auto branch = adapter->root(*r.state,memory);
  auto danger = make_danger_model(*adapter);
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
        if (route[t] >= route[u.tile] || incoming >= u.hp) continue;
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
      if (score < best) { best = score; best_tile = t; }
    }
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

std::vector<int32_t> attack_candidates(const CaiSearchRequest& request) {
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
    if (unit.hp <= 0 || unit.moves <= 0 || unit.waiting) continue;
    int distance = adapter->distance(unit.tile,request.target);
    // CP's cheap baseMoves*4 bound precedes pathfinding, not a fixed radius.
    if (distance > (combat_rules::moves[unit.type]/move_denominator)*4) continue;
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
    if (!can_reach && distance > nearby_maneuver_radius) continue;
    // CP additionally checks expected ranged damage (target HP/20 capped at3).
    // No city/air/domain branches exist in the supported simulator profile.
    if (ranged) {
      auto forecast = adapter->forecast(*root,id,unit.tile,target_unit,request.target);
      if (forecast.code || forecast.damage < std::min(view.units[target_unit].hp/20,3)) continue;
    }
    selected.push_back(id);
  }
  return selected;
}

// Agreed port policy for a unit returned by bounded normal search. This is
// deliberately separate from CP's ordinary risk/formation acceptance rules.
Plan recovery_assignments(const CaiSearchRequest& request, CaiSearchDiagnostics& diagnostics) {
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
  auto danger = make_danger_model(*adapter);
  int id = request.selected_units[0];
  TacticalView initial(*root,request.side);
  auto original = initial.units[id];
  if (original.hp <= 0 || original.moves <= 0 || original.waiting) return plan;
  using Rank = std::tuple<int,int,int,int,int,int>;
  Rank best{1,0,0,0,0,0};
  std::vector<CaiAssignment> best_attack;
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
  auto attacks = [&](const Branch& source, const std::vector<CaiAssignment>& prefix) {
    bool found = false;
    const auto before = source.state().units[id];
    for (auto command : adapter->actions(source)) {
      if (command.unit != id || (command.kind != combat_rules::MELEE && command.kind != combat_rules::RANGED))
        continue;
      // Recovery never plans an attack on an enemy discovered only by its
      // hypothetical approach. Real contact returns control to dispatch.
      int victim = initial.occupied[command.target];
      if (victim < 0 || initial.units[victim].side == request.side) continue;
      auto branch = source.clone();
      auto transition = adapter->apply(*branch,command);
      ++plan.result.stats.checked;
      if (!transition.accepted) continue;
      found = true;
      auto assignment = record(before,*branch,command,transition);
      auto now = branch->state().units[id];
      int incoming = now.hp > 0 ? danger->incoming(*branch,id,now.tile) : 0;
      Rank rank{-int(assignment.killed_unit >= 0),-assignment.damage,assignment.self_damage,
                incoming,command.target,command.kind};
      if (best_attack.empty() || rank < best) {
        best = rank;
        best_attack = prefix;
        best_attack.push_back(assignment);
      }
    }
    return found;
  };
  // An immediate legal attack has priority over every hypothetical approach.
  if (!attacks(*root,{})) {
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
      attacks(*branch,{movement});
    }
  }
  if (!best_attack.empty()) {
    plan.assignments = std::move(best_attack);
    diagnostics.recovery_kind = CAI_RECOVERY_ATTACK;
  } else {
    // CP FindSafestPlotInReach compression/support terms, with the extra
    // conservative requirement that the raw incoming danger strictly falls.
    int current = danger->incoming(*root,id,original.tile);
    int best_tile = original.tile;
    auto safety_rank = std::make_tuple(std::numeric_limits<int>::max(),0,0);
    for (auto reachable : adapter->reachable(*root,id)) {
      int tile = reachable.tile;
      if (tile == original.tile || occupied_contact(*adapter,*root,tile)) continue;
      int incoming = danger->incoming(*root,id,tile);
      ++plan.result.stats.checked;
      if (incoming >= current) continue;
      int compressed = incoming > 144 ? 12*int(std::sqrt(double(incoming))) : incoming;
      if (!original.acted)
        compressed += NEUTRAL_HEAL_RATE + ((original.type!=2 && original.type!=4)?3:0);
      int enemies = 0;
      bool exposed = false;
      for (int enemy : initial.enemies) {
        enemies += adapter->distance(tile,initial.units[enemy].tile)==1;
        exposed = exposed || adapter->can_see(*root,initial.units[enemy].tile,tile);
      }
      if (exposed) compressed += compressed/4;
      if (enemies) compressed += compressed/10;
      auto rank = std::make_tuple(compressed+(enemies-friends_at(initial,*adapter,tile,request.side,id))*13,
                                  incoming,tile);
      if (rank < safety_rank) { safety_rank = rank; best_tile = tile; }
    }
    if (best_tile != original.tile) {
      Command command{combat_rules::MOVE,id,best_tile};
      auto transition = adapter->apply(*root,command);
      if (transition.accepted) {
        plan.assignments.push_back(record(original,*root,command,transition));
        diagnostics.recovery_kind = CAI_RECOVERY_RETREAT;
      }
    }
    if (plan.assignments.empty()) {
      plan.assignments.push_back(stationary_assignment(CAI_FINISH,original,CAI_ROLE_NONE));
      diagnostics.recovery_kind = CAI_RECOVERY_WAIT;
    }
  }
  plan.result.reason = CAI_COMPLETED;
  plan.result.assignment_count = int32_t(plan.assignments.size());
  return plan;
}
} // namespace combat_ai
