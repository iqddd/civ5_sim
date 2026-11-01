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
} // namespace combat_ai
