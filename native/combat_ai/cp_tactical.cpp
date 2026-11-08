#include "cp_tactical.h"
#include "storage_metrics.h"
#include "profile.h"
#include "../combat_state.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#ifndef CAI_CP_AREA
#define CAI_CP_AREA 1
#endif
// Experimental ablation only: disable both blocker combos and finish pairing.
#ifndef CAI_CP_COMBOS
#define CAI_CP_COMBOS 1
#endif

namespace combat_ai {
namespace {
// IDs follow civ5sim/engine.py:UNIT_TYPES/FEATURES; these are ABI facts.
enum class UnitType { Warrior, Archer, ChariotArcher, Spearman, Horseman };
enum class Feature { None, Forest, Jungle, Marsh, Oasis, Ice, FloodPlains, Fallout, Atoll };
constexpr int kMountainElevation = 2;
constexpr int kMoveDenominator = 60;
constexpr int kMaxHitPoints = 100;
constexpr int kRangedAttackRange = 2;
constexpr int kMeleeAttackRange = 1;
constexpr int kMaxTacticalPlots = 255;
constexpr int kMaxDestinationDistance = 6;
constexpr int kTargetAreaRadius = 5;
constexpr int kEnemyAreaRadius = 2;
constexpr int kFogRadius = 2;
// CvDangerPlots.cpp:24,913 converts hypothetical hidden attackers to danger units.
constexpr int kFogDefaultDanger = 1;
// CvTacticalAI.cpp:10359-10361. No personality input in ABI 6: use the
// agreed midpoint of CP's 0..10 offense scale, rather than its lower bound.
constexpr int kDefaultOffenseFlavor = 5;
constexpr int kMinHpForTacticalSearch = 50 - 2 * kDefaultOffenseFlavor;
// Contract: docs/combat-ai/combat-ai-contract.md#early-finish.
// CvTacticalAI.cpp:addFinishMovesIfAcceptable: reward an untouched unit left
// available after early completion. This is a CP score, not a movement cost.
constexpr int kUntouchedEarlyFinishBonus = 123;
// Pinned CvTacticalAI.cpp:8519: examine five times the requested branch count.
constexpr int kCandidateOversampling = 5;
// Pinned CvTacticalAI.cpp:8559: allow mildly negative frontline rotations.
constexpr int kMinBlockerComboScore = -10;
bool passable(const CaiBattleState &state, int tile) {
  return state.tiles[tile].elevation < kMountainElevation &&
         static_cast<Feature>(state.tiles[tile].feature) != Feature::Ice;
}
bool ranged(int type) {
  return combat_rules::ranged[type] != 0;
}
int attack_range(int type) {
  return ranged(type) ? kRangedAttackRange : kMeleeAttackRange;
}
bool can_move_after_attacking(int type) {
  return static_cast<UnitType>(type) == UnitType::Horseman;
}
bool can_fortify(int type) {
  return static_cast<UnitType>(type) != UnitType::ChariotArcher &&
         static_cast<UnitType>(type) != UnitType::Horseman;
}
bool better(const CpMove &left, const CpMove &right) {
  return left.score > right.score;
}
} // namespace
CpTactics::CpTactics(const RulesAdapter &rules, const Branch &branch, const CaiConfig &settings,
                     const CaiSearchRequest &request)
    : adapter(rules), root(branch), config(settings), root_state(branch.state()), side(request.side),
      target(request.target), selected(request.selected_units, request.selected_count, root_state.unit_count),
      admitted_at_setup(root_state.unit_count, 0), initial(branch, request.side) {
  root_visible = initial.visibility;
  visibility_score.assign(root_state.tile_count, 0);
  for (int slot_index = 0; slot_index < int(selected.size()); ++slot_index) {
    SlotIdx slot{slot_index};
    const auto &unit = root_unit(slot);
    admitted_at_setup[selected.unit_at(slot)] = unit.hp > 0 && unit.moves > 0 && !unit.waiting;
  }
  // CvDangerPlots::UpdateDangerSingleUnit: fixed possible-attacker lists.
  // Use branch-owned data, rather than reading a second request snapshot.
  attackers.resize(root_state.tile_count);
  fog_danger.assign(root_state.tile_count, 0);
  std::vector<uint8_t> known(initial.units.size(), 0);
  // Adapter validation owns the invariant; at() also checks the local boundary.
  for (auto contact : root.contacts().known_enemies)
    known.at(contact.id) = 1;
  for (const auto &enemy : initial.units) {
    if (enemy.hp <= 0 || enemy.side == side || !known[enemy.id])
      continue;
    for (int tile : adapter.threatened_tiles(root, enemy.id))
      attackers[tile].push_back(enemy.id);
    // CP-only AddFogDanger(origin, enemy, 2, false). Hidden origins have a
    // fixed-radius bound; these full-map scans cost O(enemies * tiles).
    if (!root_state.fog || !root_visible[enemy.tile])
      continue;
    for (int hidden = 0; hidden < root_state.tile_count; ++hidden) {
      if (root_visible[hidden] || !passable(root_state, hidden) ||
          adapter.distance(enemy.tile, hidden) > kFogRadius)
        continue;
      for (int tile = 0; tile < root_state.tile_count; ++tile)
        if (adapter.distance(hidden, tile) <= kFogRadius)
          fog_danger[tile] += kFogDefaultDanger;
    }
  }
}
const CaiUnit &CpTactics::root_unit(UnitId id) const {
  return initial.units.at(id.value);
}
const CaiUnit &CpTactics::root_unit(SlotIdx slot) const {
  return root_unit(selected.unit_at(slot));
}
const CaiUnit &CpTactics::current_unit(const CpBoard &board, SlotIdx slot) const {
  return board.view.units.at(selected.unit_at(slot).value);
}
bool CpTactics::enemy_can_see(int tile) const {
  if (!root_state.fog)
    return true;
  // TacticalView::enemies contains only living enemies visible at the root.
  // CP's known-enemy visibility estimate stays fixed for this search.
  return std::any_of(initial.enemies.begin(), initial.enemies.end(),
                     [&](int enemy) { return adapter.can_see(root, root_unit(UnitId{enemy}).tile, tile); });
}
Damage CpTactics::damage(int attacker, int from, int defender, int to, int extra, bool unknown_origin) {
  std::vector<int> key{attacker, from, defender, to, extra, int(unknown_origin)};
  auto cached = attack_cache.find(key);
  if (cached != attack_cache.end())
    return cached->second;
  auto value = adapter.forecast(root, attacker, from, defender, to, extra, true, unknown_origin);
  attack_cache.emplace(std::move(key), value);
  return value;
}
int CpTactics::danger(int unit, int tile, int self_damage, const std::vector<int> &killed) {
  // The ignored IDs form a set, including a hypothetically appended near-kill.
  auto ignored = killed;
  std::sort(ignored.begin(), ignored.end());
  ignored.erase(std::unique(ignored.begin(), ignored.end()), ignored.end());
  std::vector<int> key{unit, tile, self_damage};
  key.insert(key.end(), ignored.begin(), ignored.end());
  if (trace) {
    trace->event("danger_query", {{"unit",unit},{"tile",tile},{"self_damage",self_damage}});
    for (int id : ignored) trace->event("danger_ignored", {{"unit",unit},{"tile",tile},{"enemy",id}});
  }
  auto cached = danger_cache.find(key);
  if (cached != danger_cache.end()) {
    if (trace) trace->event("danger_cached", {{"unit",unit},{"tile",tile},{"self_damage",self_damage},{"total",cached->second}});
    return cached->second;
  }
  if (trace) {
    trace->event("danger_begin", {{"unit",unit},{"tile",tile},{"self_damage",self_damage},{"fog",fog_danger[tile]}});
  }
  int total = fog_danger[tile];
  bool visible = enemy_can_see(tile);
  for (int enemy : attackers[tile]) {
    const auto &enemy_unit = root_unit(UnitId{enemy});
    if (enemy_unit.tile == tile || std::binary_search(ignored.begin(), ignored.end(), enemy))
      continue;
    bool unknown = adapter.distance(enemy_unit.tile, tile) > attack_range(enemy_unit.type);
    auto forecast = damage(enemy, enemy_unit.tile, unit, tile, self_damage, unknown);
    int contribution = forecast.damage;
    if (!visible)
      contribution = contribution * 80 / 100;
    if (forecast.retaliation >= enemy_unit.hp)
      contribution /= 2;
    if (trace) trace->event("danger_attacker", {{"unit",unit},{"tile",tile},{"enemy",enemy},
        {"damage",forecast.damage},{"retaliation",forecast.retaliation},{"enemy_hp",enemy_unit.hp},
        {"visible",visible},{"unknown_origin",unknown},{"contribution",contribution}});
    total += contribution;
  }
  if (trace) trace->event("danger_total", {{"unit",unit},{"tile",tile},{"total",total}});
  danger_cache.emplace(std::move(key), total);
  return total;
}
CpBoard CpTactics::rebuild_board(const TacticalPosition &position) const {
  CAI_STORAGE_COUNT(BoardRebuild);
  CpBoard board(TacticalView(*position.branch, side));
  board.plots.resize(board.view.visibility.size());
  for (int tile = 0; tile < int(board.plots.size()); ++tile)
    if (position.plots[tile]) {
      auto &plot = board.plots.add(tile);
      plot.valid = true;
      int occupant = board.view.occupied[tile];
      if (occupant >= 0) {
        const auto &unit = board.view.units[occupant];
        if (unit.side != side) {
          plot.enemy = static_cast<int16_t>(occupant);
          board.enemies.push_back(occupant);
        } else {
          UnitId unit_id{occupant};
          plot.blocked = !admitted_at_setup[unit_id];
          // Root pruning emits BLOCKED, not a completed-turn defender. Admission
          // remains valid; the coordinator may still move the excluded unit.
          plot.end = admitted_at_setup[unit_id]
                         ? position.slot_states[selected.slot_of(unit_id)] == SlotState::Finished
                         : bool(root_unit(unit_id).waiting);
        }
      }
    }
  std::vector<int> distance_enemies = board.enemies;
  // refreshVolatilePlotProperties/checkEdgePlotsForSurprises: visible
  // enemies just outside the tactical area affect distances, not targets.
  for (int tile = 0; tile < int(board.plots.size()); ++tile)
    if (board.plots[tile].valid) {
      for (int neighbor : adapter.neighbors(tile)) {
        if (adapter.distance(neighbor, target) < adapter.distance(tile, target) ||
            board.plots[neighbor].valid)
          continue;
        if (!board.view.visibility[neighbor]) {
          if (passable(root_state, neighbor))
            board.plots.at(tile).edge = true;
        } else {
          int occupant = board.view.occupied[neighbor];
          if (occupant >= 0 && board.view.units[occupant].side != side &&
              std::find(distance_enemies.begin(), distance_enemies.end(), occupant) == distance_enemies.end())
            distance_enemies.push_back(occupant);
        }
      }
    }
  for (int tile = 0; tile < int(board.plots.size()); ++tile)
    if (board.plots[tile].valid) {
      auto &plot = board.plots.at(tile);
      for (int occupant : distance_enemies) {
        int distance = adapter.distance(tile, board.view.units[occupant].tile);
        plot.distance = static_cast<uint8_t>(std::min(int(plot.distance), distance));
        plot.enemies += distance == 1;
      }
      for (int neighbor : adapter.neighbors(tile))
        if (board.plots[neighbor].valid) {
          int occupant = board.view.occupied[neighbor];
          if (occupant >= 0 && board.view.units[occupant].side == side) {
            ++plot.friends;
            plot.friends_end += board.plots[neighbor].end;
          }
        }
    }
  return board;
}
const CpBoard &CpTactics::board(const TacticalPosition &position) const {
  CAI_PROFILE_SCOPE(BoardRefresh);
  if (!position.tactical_board)
    position.tactical_board = rebuild_board(position);
  auto &current = *position.tactical_board;
  TacticalView next(*position.branch, side);
  bool refresh_volatile = current.view.visibility != next.visibility ||
                          current.view.enemies != next.enemies;
  for (int tile = 0; !refresh_volatile && tile < int(position.plots.size()); ++tile)
    refresh_volatile = current.plots[tile].valid != bool(position.plots[tile]);
  for (int enemy : next.enemies)
    refresh_volatile = refresh_volatile || current.view.units[enemy].tile != next.units[enemy].tile;
  // CP:8995 refreshVolatilePlotProperties; kills and discoveries invalidate
  // enemy distance/adjacency and unknown-edge fields throughout the area.
  if (refresh_volatile) {
    position.tactical_board = rebuild_board(position);
    return *position.tactical_board;
  }
  CAI_STORAGE_COUNT(BoardUpdate);
  // CP:CvTacticalPlot::friendlyUnitMovingIn/Out and friendlyUnitEndTurn:
  // ordinary movement and FINISH affect only occupancy and adjacent support.
  std::vector<uint8_t> dirty(position.plots.size(), 0);
  for (int tile = 0; tile < int(position.plots.size()); ++tile) {
    if (!current.plots[tile].valid)
      continue;
    auto &plot = current.plots.at(tile);
    int occupant = next.occupied[tile];
    bool finished = false;
    if (occupant >= 0 && next.units[occupant].side == side) {
      UnitId unit_id{occupant};
      finished = admitted_at_setup[unit_id]
                     ? position.slot_states[selected.slot_of(unit_id)] == SlotState::Finished
                     : bool(root_unit(unit_id).waiting);
    }
    if (current.view.occupied[tile] == occupant && plot.end == finished)
      continue;
    plot.end = finished;
    plot.blocked = occupant >= 0 && next.units[occupant].side == side &&
                   !admitted_at_setup[UnitId{occupant}];
    dirty[tile] = 1;
    for (int neighbor : adapter.neighbors(tile))
      dirty[neighbor] = 1;
  }
  current.view = std::move(next);
  for (int tile = 0; tile < int(dirty.size()); ++tile) {
    if (!dirty[tile] || !current.plots[tile].valid)
      continue;
    auto &plot = current.plots.at(tile);
    plot.friends = plot.friends_end = 0;
    for (int neighbor : adapter.neighbors(tile)) {
      int occupant = current.view.occupied[neighbor];
      if (current.plots[neighbor].valid && occupant >= 0 && current.view.units[occupant].side == side) {
        ++plot.friends;
        plot.friends_end += current.plots[neighbor].end;
      }
    }
  }
  return current;
}
bool CpTactics::cover(const CpBoard &board, int tile) const {
  const auto &plot = board.plots[tile];
  if (plot.distance < 2)
    return false;
  if (plot.distance > 3)
    return true;
  for (int neighbor : adapter.neighbors(tile))
    if (board.plots[neighbor].valid && board.plots[neighbor].distance < plot.distance &&
        !board.plots[neighbor].end)
      return false;
  return true;
}
EndScore CpTactics::turn_end(const CpBoard &board, SlotIdx slot, int tile, EndQuery query) const {
  const auto &original = root_unit(slot);
  const auto &now = current_unit(board, slot);
  const auto &plot = board.plots[tile];
  int friends = (query.friend_count == FriendCount::Finished ? plot.friends_end : plot.friends) +
                int(query.assume_extra_friend);
  bool acceptable = !(plot.enemies > 3 || (plot.enemies == 3 && bias < 1.f));
  int incoming_damage = query.incoming_damage;
  bool covered = cover(board, tile);
  if (covered)
    incoming_damage /= 2;
  if (plot.edge)
    incoming_damage = std::max(50, incoming_damage);
  int adjusted_danger = incoming_damage;
  int threshold_factor = 0;
  int score = 0;
  int remaining_hp = std::max(1, now.hp - query.extra_self_damage);
  if (incoming_damage > 0) {
    // CvTacticalAI.cpp:7136-7186, ScoreCombatUnitTurnEnd.
    if (config.aggression != CAI_BRAVEHEART) {
      if (ranged(original.type) && incoming_damage > remaining_hp && (plot.enemies > 0 || friends == 0))
        acceptable = false;
      int minimum_hp_factor = query.killed_enemy || safe[slot] == 0 ? 23 : 42;
      if (board.enemies.size() == 1 && query.killed_enemy)
        minimum_hp_factor = 12;
      threshold_factor = minimum_hp_factor;
      if (remaining_hp * (friends + 1) < minimum_hp_factor * incoming_damage / remaining_hp)
        acceptable = false;
    }
    incoming_damage /= friends + 1;
    // CP real units retain root HP during the sim. Preserve that denominator
    // and healing predicate; remaining_hp above includes simulated self-damage.
    int scaled_danger = incoming_damage * kMaxHitPoints / std::max(1, original.hp);
    if (scaled_danger > 225)
      scaled_danger = 15 * int(std::sqrt(double(scaled_danger)));
    score -= scaled_danger;
  }
  // Pinned ScorePlotForCombatUnitMove passes remaining==maxMoves directly
  // as bHasMoved. Preserve that call convention, including its inverted name.
  if (!query.has_full_moves) {
    if (original.hp < kMaxHitPoints)
      ++score;
    if (plot.distance < 3 && can_fortify(original.type))
      ++score;
  }
  if (friends > 0)
    score += 7;
  if (!enemy_can_see(tile))
    ++score;
  score += adapter.defense(tile) / 5;
  if (trace) trace->event("turn_end", {{"unit",now.id},{"tile",tile},{"hp",remaining_hp},
      {"danger",query.incoming_damage},{"adjusted_danger",adjusted_danger},{"friends",friends},
      {"finished_friends",query.friend_count == FriendCount::Finished},{"enemies",plot.enemies},
      {"edge",plot.edge},{"covered",covered},{"assume_extra_friend",query.assume_extra_friend},
      {"root_hp",original.hp},{"safe_plots",safe[slot]},{"killed_enemy",query.killed_enemy},
      {"threshold_factor",threshold_factor},{"hp_support",remaining_hp*(friends+1)},
      {"threshold",threshold_factor*adjusted_danger/remaining_hp},{"acceptable",acceptable},{"score",score}});
  return {score, acceptable};
}
MoveScore CpTactics::move_score(const TacticalPosition &position, const CpBoard &board, SlotIdx slot,
                                int tile, int remaining_moves, MoveEvaluation mode) {
  if (!visibility_scores_ready)
    throw std::logic_error("move scoring before visibility setup");
  const auto &now = current_unit(board, slot);
  MoveScore result;
  result.move.unit = now.id;
  result.move.tile = tile;
  result.move.remaining = remaining_moves;
  if (!board.plots[tile].valid)
    return result;
  int distance = board.enemies.empty() ? adapter.distance(tile, target) : board.plots[tile].distance;
  int base_score = formation_score(roles[slot], distance, !board.enemies.empty());
  if (!board.enemies.empty() && position.last_kill[slot])
    base_score = std::max(60, base_score);
  int self_damage = root_unit(slot).hp - now.hp;
  int incoming_damage = danger(now.id, tile, self_damage, position.killed);
  int situational_score = 0;
  if (mode != MoveEvaluation::Intermediate) {
    EndQuery query;
    query.incoming_damage = incoming_damage;
    query.killed_enemy = position.last_kill[slot] != 0;
    query.friend_count = mode == MoveEvaluation::Initial ? FriendCount::Current : FriendCount::Finished;
    query.has_full_moves = remaining_moves == combat_rules::moves[now.type];
    auto end = turn_end(board, slot, tile, query);
    if (!end.acceptable)
      return result;
    situational_score = end.score - self_damage;
  } else {
    int overkill_percent = 100 * incoming_damage / std::max(1, now.hp);
    situational_score = -overkill_percent / 20;
    if (overkill_percent > 300 && remaining_moves < 3 * kMoveDenominator && now.moves > 3 * kMoveDenominator)
      return result;
  }
  if (ranged(now.type))
    situational_score += visibility_score[tile];
  result.move.score = base_score + situational_score + remaining_moves / kMoveDenominator;
  result.acceptable = true;
  if (trace) trace->event("move_score", {{"unit",now.id},{"tile",tile},{"mode",int(mode)},
      {"formation",base_score},{"situational",situational_score},{"remaining",remaining_moves},{"score",result.move.score}});
  return result;
}
CpMove CpTactics::attack(const TacticalPosition &position, const CpBoard &board, SlotIdx slot, int from,
                         int enemy) {
  const auto &now = current_unit(board, slot);
  const auto &enemy_unit = board.view.units[enemy];
  CpMove result;
  result.unit = now.id;
  result.tile = enemy_unit.tile;
  result.kind = ranged(now.type) ? CP_RANGED : CP_MELEE;
  auto forecast = damage(now.id, from, enemy, enemy_unit.tile, root_unit(UnitId{enemy}).hp - enemy_unit.hp);
  if (forecast.code) {
    if (trace) trace->event("attack_forecast_invalid", {{"unit",now.id},{"from",from},{"enemy",enemy},{"error",forecast.code}});
    return result;
  }
  int dealt = forecast.damage, received = forecast.retaliation;
  if (!ranged(now.type)) {
    // Contract: docs/combat-ai/combat-ai-contract.md#hypothetical-flanking.
    // Unlike CP:6743/8006, exclude the attacker only if board() counted it.
    // The hypothetical origin `from` has not changed the board's occupancy.
    bool attacker_counted = board.plots[now.tile].valid && adapter.distance(now.tile, enemy_unit.tile) == 1;
    int friends = board.plots[enemy_unit.tile].friends - int(attacker_counted);
    int flanking_delta = friends - board.plots[from].enemies;
    dealt += flanking_delta * dealt / 10;
    received -= flanking_delta * received / 10;
  }
  result.near_kill = dealt >= enemy_unit.hp - 10 ? enemy : -1;
  // CP:6776-6780 deliberately tolerates one HP in the kill prediction.
  // Actual kills are recorded by the executor from the resulting physical state.
  result.kill = dealt >= enemy_unit.hp - 1;
  if (result.kill)
    dealt = std::min(dealt, enemy_unit.hp + 10);
  if (received > 0) {
    // CvTacticalAI.cpp:6832-6886, ScoreAttackDamage.
    constexpr float factors[5] = {0.f, .7f, 1.1f, 2.3f, 4.2f};
    // This is the actual pinned CP initializer/indexing, not a shifted table.
    constexpr int hp_limit[5] = {70, 40, 20, -5, 0};
    auto ignored = position.killed;
    if (result.near_kill >= 0)
      ignored.push_back(result.near_kill);
    // CP:6864 advances only after a melee kill. Ranged forecasts always have
    // zero retaliation; express the destination rule even for future changes.
    int after_attack = result.kind == CP_MELEE && result.kill ? enemy_unit.tile : from;
    int incoming = danger(now.id, after_attack, received, ignored);
    int damage_delta = int(dealt * bias * factors[config.aggression] + .5f) - received;
    if (config.aggression == CAI_NONE ||
        ((incoming > 100 || root_unit(UnitId{now.id}).hp - received < hp_limit[config.aggression]) &&
         damage_delta < 0 && !result.kill) ||
        (root_unit(UnitId{now.id}).hp - received < 3 && config.aggression != CAI_BRAVEHEART)) {
      if (trace) trace->event("attack_exchange_rejected", {{"unit",now.id},{"from",from},{"enemy",enemy},
          {"incoming",incoming},{"damage_delta",damage_delta},{"hp",root_unit(UnitId{now.id}).hp},
          {"received",received},{"hp_limit",hp_limit[config.aggression]},{"aggression",config.aggression}});
      return result;
    }
  }
  if (trace) trace->event("attack_score", {{"unit",now.id},{"from",from},{"enemy",enemy},
      {"forecast_damage",forecast.damage},{"forecast_received",forecast.retaliation},
      {"score",dealt},{"received",received},{"kill",result.kill},{"near_kill",result.near_kill}});
  result.score = dealt;
  result.self_damage = received;
  return result;
}
CpTactics::PotentialAttackScore CpTactics::potential(const TacticalPosition &position,
    const CpBoard &board, SlotIdx slot, int tile, int num_attacks) {
  const auto &now = current_unit(board, slot);
  PotentialAttackScore result;
  int best_score = 0;
  for (int enemy_id : board.enemies) {
    int destination = board.view.units[enemy_id].tile, distance = adapter.distance(tile, destination);
    if (distance < 1 || distance > attack_range(now.type))
      continue;
    if (ranged(now.type)) {
      const auto &targets = ranged_targets(now.id, tile);
      if (!std::binary_search(targets.begin(), targets.end(), destination))
        continue;
    }
    auto value = attack(position, board, slot, tile, enemy_id);
    if (value.kill && num_attacks > 0 && value.score > 0) {
      value.score += 100;
      // CP:7052-7057 records the last qualifying victim, not necessarily
      // the highest-scoring attack. A move can anticipate one kill this turn.
      result.assumed_victim = value.near_kill;
    }
    best_score = std::max(best_score, value.score);
  }
  result.score = num_attacks > 0 ? num_attacks * best_score / 2 : best_score / 4;
  return result;
}
bool CpTactics::can_end(const TacticalPosition &position, const CpBoard &board, SlotIdx slot,
                        const CpMove &move, int tile) {
  if (!board.plots[tile].valid) {
    if (trace) trace->event("end_precheck_invalid_plot", {{"unit",move.unit},{"tile",tile}});
    return false;
  }
  if (safe[slot] == 0) {
    if (trace) trace->event("end_precheck_no_safe_plot", {{"unit",move.unit},{"tile",tile}});
    return true;
  }
  const auto &now = current_unit(board, slot);
  EndQuery query;
  // CP:10121 passes the assumed victim to ScoreCombatUnitTurnEnd, but that
  // function only fills an unused gEnemiesToIgnore (7092-7094). Recompute
  // danger with the same near-kill assumption used by ScoreAttackDamage;
  // otherwise the proposed victim can veto the attack that would remove it.
  auto ignored = position.killed;
  if (move.near_kill >= 0)
    ignored.push_back(move.near_kill);
  query.incoming_damage = danger(move.unit, tile, root_unit(slot).hp - now.hp, ignored);
  query.killed_enemy = move.kill;
  query.extra_self_damage = move.self_damage;
  query.friend_count = FriendCount::Finished;
  query.assume_extra_friend =
      std::count(position.slot_states.begin(), position.slot_states.end(), SlotState::Pending) > 1;
  query.has_full_moves = move.remaining == combat_rules::moves[now.type];
  return turn_end(board, slot, tile, query).acceptable;
}
const std::vector<ReachableTile> &CpTactics::move_plots(
    const TacticalPosition &position, const CpBoard &board, SlotIdx slot) {
  const auto &unit = current_unit(board, slot);
  // Pinned CP:8140,9247: unit/start/moves/freed enemy plots. Friendly stacking
  // does not affect move_intents. Unlike GameCore's root-backed pathfinder,
  // this adapter uses branch visibility/ZOC, so newly observed enemies must
  // participate in the key too. Enemy units cannot relocate during our search.
  std::vector<int> key{unit.id, unit.tile, unit.moves};
  key.insert(key.end(), position.killed.begin(), position.killed.end());
  key.push_back(-1);
  key.insert(key.end(), board.view.enemies.begin(), board.view.enemies.end());
  auto found = reach_cache.find(key);
  if (found != reach_cache.end()) {
    CAI_STORAGE_COUNT(ReachHit);
    return found->second;
  }
  CAI_STORAGE_COUNT(ReachMiss);
  // Store geometric intentions; tactical validity/blocking is filtered at use.
  // This preserves CP's lazy reuse without freezing the pre-discovery area.
  return reach_cache.emplace(std::move(key), adapter.move_intents(*position.branch, unit.id)).first->second;
}
const std::vector<int> &CpTactics::ranged_targets(int unit, int from) {
  auto key = std::make_pair(unit, from);
  auto found = ranged_target_cache.find(key);
  if (found != ranged_target_cache.end()) {
    CAI_STORAGE_COUNT(RangeHit);
    return found->second;
  }
  CAI_STORAGE_COUNT(RangeMiss);
  // CP:9332-9341 ignores visibility here; target validity is checked at use.
  // Store geometric tiles, not a branch's enemy list. Terrain never changes.
  std::vector<int> targets;
  for (int tile = 0; tile < root_state.tile_count; ++tile)
    if (tile != from && adapter.distance(from, tile) <= kRangedAttackRange && adapter.can_target(from, tile))
      targets.push_back(tile);
  return ranged_target_cache.emplace(key, std::move(targets)).first->second;
}
std::vector<CpMove> CpTactics::preferred(const TacticalPosition &position, const CpBoard &board, SlotIdx slot,
                                         int limit) {
  std::vector<CpMove> moves, shots;
  const auto &now = current_unit(board, slot);
  if (position.slot_states[slot] != SlotState::Pending || !board.plots[now.tile].valid)
    return moves;
  for (auto reach : move_plots(position, board, slot)) {
    int tile = reach.tile;
    if (trace) trace->event("consider_destination", {{"unit",now.id},{"tile",tile},{"remaining",reach.remaining_moves}});
    if ((CAI_CP_AREA && adapter.distance(target, tile) > kMaxDestinationDistance) ||
        !board.plots[tile].valid || board.plots[tile].blocked) {
      if (trace) trace->event("destination_area_rejected", {{"unit",now.id},{"tile",tile},
          {"valid",board.plots[tile].valid},{"blocked",board.plots[tile].blocked}});
      continue;
    }
    if (board.plots[tile].enemy >= 0) {
      if (config.aggression == CAI_NONE || ranged(now.type) || now.attacked ||
          adapter.distance(now.tile, tile) > 1) {
        if (trace) trace->event("melee_intent_rejected", {{"unit",now.id},{"tile",tile},
            {"aggression",config.aggression},{"ranged",ranged(now.type)},{"attacked",now.attacked}});
        continue;
      }
      auto attack_move = attack(position, board, slot, now.tile, board.plots[tile].enemy);
      if (attack_move.score < 0)
        continue;
      attack_move.remaining = adapter.attack_moves(*position.branch, {combat_rules::MELEE, now.id, tile});
      int after = attack_move.kill ? tile : now.tile;
      if (attack_move.kill && board.plots[tile].enemies > 3) {
        if (trace) trace->event("kill_destination_surrounded", {{"unit",now.id},{"tile",tile},{"enemies",board.plots[tile].enemies}});
        continue;
      }
      if (attack_move.remaining == 0 && !can_end(position, board, slot, attack_move, after))
        continue;
      attack_move.score =
          (attack_move.score + 8) * 10 + (attack_move.kill ? 1000 : 0) + (tile == target ? 3 : 0);
      // CP:8230-8231 adds even the finite impossible score. This remains a
      // candidate penalty, rather than an additional hard rejection.
      attack_move.score +=
          move_score(position, board, slot, after, attack_move.remaining, MoveEvaluation::Intermediate)
              .move.score;
      moves.push_back(attack_move);
      continue;
    }
    auto move =
        move_score(position, board, slot, tile, reach.remaining_moves, MoveEvaluation::Intermediate).move;
    if (tile == now.tile) {
      if (config.aggression != CAI_NONE && ranged(now.type) && !now.attacked && now.moves > 0) {
        for (int destination : ranged_targets(now.id, tile)) {
          int enemy = board.plots[destination].enemy;
          if (enemy < 0)
            continue;
          auto attack_move = attack(position, board, slot, tile, enemy);
          if (attack_move.score < 0)
            continue;
          attack_move.remaining = 0; // All supported ranged types end their turn on attack.
          if (!can_end(position, board, slot, attack_move, tile))
            continue;
          attack_move.score *= 10;
          if (now.moves > kMoveDenominator && board.plots[tile].distance == 1 &&
              roles[slot] != CAI_FIRST_LINE)
            attack_move.score /= 2;
          attack_move.score += (attack_move.kill ? 1000 : 0) + (destination == target ? 2 : 0) + move.score;
          shots.push_back(attack_move);
        }
      }
      if (shots.empty() && can_end(position, board, slot, move, tile)) {
        if (reach.remaining_moves == combat_rules::moves[now.type]) {
          if (root_unit(UnitId{now.id}).hp < 95)
            move.score += std::min(23, 100 - root_unit(UnitId{now.id}).hp);
          if (can_fortify(now.type))
            move.score += 23;
        }
        // CP's stay/FINISH branch intentionally discards the potential victim.
        move.score += potential(position, board, slot, tile, 1).score * 10;
        move.kind = CP_FINISH_TEMP;
        moves.push_back(move);
      }
    } else {
      // Contract: docs/combat-ai/combat-ai-contract.md#continuation.
      if (position.move_flags[slot] != MoveFlag::None || board.plots[tile].end) {
        if (trace) trace->event("continuation_rejected", {{"unit",now.id},{"tile",tile},
            {"move_flag",int(position.move_flags[slot])},{"plot_end",board.plots[tile].end}});
        continue;
      }
      int num_attacks =
          now.attacked ? 0 : std::min(1, (reach.remaining_moves + kMoveDenominator - 1) / kMoveDenominator);
      // Preserve CP's arithmetic with kCpImpossibleScore and attack bonuses.
      auto potential_attack = potential(position, board, slot, tile, num_attacks);
      move.score += potential_attack.score * 10;
      move.near_kill = potential_attack.assumed_victim;
      if ((move.remaining > kMoveDenominator && can_move_after_attacking(now.type)) ||
          can_end(position, board, slot, move, tile)) {
        moves.push_back(move);
      }
    }
  }
  if (trace) for (const auto& move : shots) trace->event("shot_before_limit", {{"unit",move.unit},{"tile",move.tile},{"score",move.score}});
  if (int(shots.size()) >= limit && limit > 1) {
    std::stable_sort(shots.begin(), shots.end(), better);
    shots.resize(limit - 1);
  }
  moves.insert(moves.end(), shots.begin(), shots.end());
  std::stable_sort(moves.begin(), moves.end(), better);
  if (trace) for (const auto& move : moves) trace->event("preferred_before_limit", {{"unit",move.unit},{"tile",move.tile},{"kind",move.kind},{"score",move.score}});
  if (int(moves.size()) > limit) {
    int count = limit;
    while (count < int(moves.size()) && moves[count].score == moves[count - 1].score)
      ++count;
    moves.resize(count);
  }
  CpMove blocked;
  blocked.unit = now.id;
  blocked.tile = now.tile;
  blocked.kind = CP_BLOCKED;
  blocked.score = -10;
  blocked.remaining = now.moves;
  moves.push_back(blocked);
  if (trace) for (const auto& move : moves) trace->event("preferred", {{"unit",move.unit},{"tile",move.tile},
      {"kind",move.kind},{"score",move.score},{"remaining",move.remaining},{"assumed_victim",move.near_kill}});
  return moves;
}
void CpTactics::initialize(TacticalPosition &position) {
  if (initialization_started)
    throw std::logic_error("tactical context initialized twice");
  initialization_started = true;
  const auto &state = root_state;
  position.plots.assign(state.tile_count, 0);
  int plot_count = 0;
  auto add_plot = [&](int tile) {
    if (!position.plots[tile] && plot_count < kMaxTacticalPlots) {
      position.plots[tile] = 1;
      ++plot_count;
    }
  };
  std::vector<int> first_enemies;
  for (int tile = 0; tile < state.tile_count; ++tile)
    if (root_visible[tile] && adapter.distance(tile, target) <= kTargetAreaRadius) {
      add_plot(tile);
      int id = initial.occupied[tile];
      if (id >= 0 && initial.units[id].side != side)
        first_enemies.push_back(id);
    }
  for (int id : first_enemies)
    for (int tile = 0; tile < state.tile_count; ++tile)
      if (root_visible[tile] && adapter.distance(root_unit(UnitId{id}).tile, tile) <= kEnemyAreaRadius)
        add_plot(tile);
  // FindBestUnitAssignments counts target-area enemies BEFORE addAvailableUnit.
  // Only admitted participants may subsequently add their neighborhoods.
  bool have_target_enemies = !board(position).enemies.empty();
  roles.assign(selected.size(), CAI_FIRST_LINE);
  safe.assign(selected.size(), 0);
  initial_distance.assign(selected.size(), 0);
  for (int slot_index = 0; slot_index < int(selected.size()); ++slot_index) {
    SlotIdx slot{slot_index};
    const auto &original = root_unit(slot);
    roles[slot] = movement_role(original, original.damage_taken_last_turn);
    bool projected_to_die = original.damage_taken_last_turn > original.hp;
    if (!admitted_at_setup[UnitId{original.id}] || (!have_target_enemies && projected_to_die)) {
      if (trace) trace->event("root_admission_rejected", {{"unit",original.id},
          {"admitted",admitted_at_setup[UnitId{original.id}]},{"projected_to_die",projected_to_die},
          {"target_enemies",have_target_enemies},{"moves",original.moves},{"waiting",original.waiting}});
      position.slot_states[slot] = SlotState::ExcludedAtRoot;
      admitted_at_setup[UnitId{original.id}] = 0;
    }
  }
  // Admission changed the non-sim blocking flags in the provisional board.
  position.tactical_board.reset();
  for (int slot_index = 0; slot_index < int(selected.size()); ++slot_index) {
    SlotIdx slot{slot_index};
    if (!admitted_at_setup[selected.unit_at(slot)])
      continue;
    int center = root_unit(slot).tile;
    // iterateRingPlots starts with ring zero. At the 255-plot boundary a
    // neighbor must not consume the last slot before the participant itself.
    add_plot(center);
    for (int neighbor : adapter.neighbors(center))
      add_plot(neighbor);
  }
  if (!CAI_CP_AREA)
    std::fill(position.plots.begin(), position.plots.end(), uint8_t{1}); // Benchmark ablation only.
  const auto &initial_board = board(position);
  initial_enemies = initial_board.enemies;
  int our_units = 0;
  for (int slot_index = 0; slot_index < int(selected.size()); ++slot_index)
    our_units += admitted_at_setup[selected.unit_at(SlotIdx{slot_index})];
  for (int tile = 0; tile < state.tile_count; ++tile)
    if (initial_board.plots[tile].blocked && initial_board.plots[tile].distance < 3)
      ++our_units;
  bias = std::max(.9f, std::sqrt(float(our_units) / std::max(1, int(initial_enemies.size()))));
  // O(tactical plots * map tiles), with at most 255 tactical plots.
  for (int tile = 0; tile < state.tile_count; ++tile)
    if (position.plots[tile])
      for (int neighbor = 0; neighbor < state.tile_count; ++neighbor)
        if (position.plots[neighbor] && initial_board.plots[neighbor].distance <= 3 &&
            adapter.distance(tile, neighbor) == 2 && adapter.can_see(root, tile, neighbor))
          ++visibility_score[tile];
  visibility_scores_ready = true;
  for (int slot_index = 0; slot_index < int(selected.size()); ++slot_index) {
    SlotIdx slot{slot_index};
    const auto &original = root_unit(slot);
    initial_distance[slot] = initial_board.plots[original.tile].distance;
    if (!admitted_at_setup[UnitId{original.id}]) {
      position.slot_states[slot] = SlotState::ExcludedAtRoot;
      continue;
    }
    // Root-only safe-plot count, before the tactical/non-sim plot filter.
    for (auto reach : move_plots(position, initial_board, slot))
      if ((!CAI_CP_AREA || adapter.distance(reach.tile, target) <= kMaxDestinationDistance) &&
          (initial.occupied[reach.tile] < 0 || initial.occupied[reach.tile] == original.id) &&
          danger(original.id, reach.tile, 0, {}) < original.hp)
        ++safe[slot];
    active_order.push_back(slot);
  }
  // dropSuperfluousUnits: importance is the first preferred action, including
  // BLOCKED (-10). Retain stable input order on equal scores.
  SlotValues<int32_t> importance(selected.size(), 0);
  int keep = std::min(int(CAI_MAX_SELECTED), int(active_order.size()));
  for (SlotIdx slot : active_order) {
    if (keep <= 0)
      break;
    auto choices = preferred(position, initial_board, slot, 1);
    importance[slot] = choices.empty() ? kCpImpossibleScore : choices.front().score;
    const auto &original = root_unit(slot);
    if (original.hp < kMinHpForTacticalSearch) {
      if (!choices.empty() && choices.front().kind == CP_MOVE && original.moves < 3 * kMoveDenominator) {
        importance[slot] = kCpImpossibleScore;
        if (trace) trace->event("drop_wounded_mover", {{"unit",original.id},{"hp",original.hp},{"moves",original.moves}});
        --keep;
      } else
        importance[slot] /= 2;
    }
  }
  std::stable_sort(active_order.begin(), active_order.end(),
                   [&](SlotIdx left, SlotIdx right) { return importance[left] > importance[right]; });
  keep = std::max(0, keep);
  for (int j = keep; j < int(active_order.size()); ++j)
    position.slot_states[active_order[j]] = SlotState::ExcludedAtRoot;
  active_order.resize(keep);
  if (trace) for (int i = 0; i < int(selected.size()); ++i) {
    SlotIdx slot{i};
    trace->event("root_participant", {{"unit",selected.unit_at(slot).value},{"state",int(position.slot_states[slot])},
        {"role",roles[slot]},{"safe",safe[slot]},{"importance",importance[slot]}});
  }
}
void CpTactics::extend_plots(TacticalPosition &position, const std::vector<int32_t> &added) const {
  int count = std::accumulate(position.plots.begin(), position.plots.end(), 0);
  for (int tile : added)
    if (!position.plots[tile] && count < kMaxTacticalPlots) {
      position.plots[tile] = 1;
      ++count;
    }
}
std::vector<CpCombo> CpTactics::candidates(const TacticalPosition &position, bool depth_first) {
  const auto &current = board(position);
  std::vector<CpMove> overall;
  for (SlotIdx slot : active_order)
    if (position.slot_states[slot] == SlotState::Pending) {
      auto choices = preferred(position, current, slot, depth_first ? 2 : config.choices_per_unit);
      overall.insert(overall.end(), choices.begin(), choices.end());
    }
  std::stable_sort(overall.begin(), overall.end(), better);
  std::vector<CpCombo> result;
  auto add = [&](CpMove first, std::optional<CpMove> second = std::nullopt) {
    // Contract: docs/combat-ai/combat-ai-contract.md#combo-invariants.
    // A supplied second action must survive intact, never silently disappear.
    CpCombo combo{{first}, first.score};
    if (second) {
      if (second->unit == first.unit)
        throw std::logic_error("combo uses the same unit twice");
      combo.moves.push_back(*second);
      combo.score += second->score;
    }
    result.push_back(std::move(combo));
  };
  size_t max_candidates =
      std::max(size_t((depth_first ? 1 : config.max_branches) * kCandidateOversampling), overall.size() / 2);
  if (trace) trace->event("candidate_shortlist", {{"available",int64_t(overall.size())},{"limit",int64_t(max_candidates)}});
  for (size_t candidate_index = 0; candidate_index < std::min(overall.size(), max_candidates);
       ++candidate_index) {
    auto primary = overall[candidate_index];
    if (primary.kind != CP_MOVE) {
      // CP:8530-8537 consumes this FINISH/BLOCKED as part of the pair.
      bool paired = false;
      if constexpr (CAI_CP_COMBOS != 0) {
        if (depth_first && candidate_index + 1 < overall.size() &&
            overall[candidate_index + 1].unit != primary.unit &&
            (overall[candidate_index + 1].kind == CP_FINISH_TEMP ||
             overall[candidate_index + 1].kind == CP_BLOCKED)) {
          add(primary, overall[++candidate_index]);
          paired = true;
        }
      }
      if (!paired)
        add(primary);
      continue;
    }
    int blocker = current.view.occupied[primary.tile];
    if (blocker < 0) {
      add(primary);
      continue;
    }
    // In the ablation these occupied-friendly intents cannot be executed.
    // They are filtered here, before materializing a child, not counted as pruned.
    if constexpr (CAI_CP_COMBOS == 0)
      continue;
    auto blocker_slot = selected.find_slot(UnitId{blocker});
    if (!blocker_slot || position.slot_states[*blocker_slot] != SlotState::Pending) {
      if (trace) trace->event("combo_blocker_unavailable", {{"unit",primary.unit},{"tile",primary.tile},{"blocker",blocker}});
      continue;
    }
    for (auto freeing : overall)
      if (freeing.unit == blocker && freeing.kind == CP_MOVE) {
        if (primary.score + freeing.score < kMinBlockerComboScore) {
          if (trace) trace->event("combo_score_rejected", {{"unit",primary.unit},{"blocker",blocker},
              {"score",primary.score + freeing.score},{"minimum",kMinBlockerComboScore}});
          continue;
        }
        if (freeing.tile == current.view.units[primary.unit].tile) {
          // Pinned CvTacticalAI.cpp:8318,8566. preferred() already filters the
          // primary (and blocker) by move_flags; retain CP's explicit blocker guard.
          if (position.move_flags[*blocker_slot] == MoveFlag::Swapped)
            continue;
          // Simulator swaps have explicit physical restrictions in addition to
          // CP's reciprocal intentions (notably adjacency and available moves).
          if (!adapter.can_execute(*position.branch, {combat_rules::SWAP, primary.unit, primary.tile}))
            continue;
          primary.kind = CP_SWAP;
          freeing.kind = CP_SWAP_REVERSE;
          add(primary, freeing);
          break;
        }
        if (current.view.occupied[freeing.tile] < 0) {
          freeing.kind = CP_FORCED;
          add(freeing, primary);
          break;
        }
      }
  }
  // Contract: docs/combat-ai/combat-ai-contract.md#candidate-order; score orders candidates,
  // while the executor's caller accumulates rank excluding finish/block records.
  std::stable_sort(result.begin(), result.end(),
                   [](const CpCombo &left, const CpCombo &right) { return left.score > right.score; });
  if (trace) {
    int index = 0;
    for (const auto& combo : result) {
      for (const auto& move : combo.moves) trace->event("combo", {{"index",index},{"unit",move.unit},
          {"tile",move.tile},{"kind",move.kind},{"score",move.score},{"combo_score",combo.score}});
      ++index;
    }
  }
  return result;
}
FinalEvaluation CpTactics::evaluate_final_position(const TacticalPosition &position) {
  FinalEvaluation evaluation;
  evaluation.score = position.rank;
  const auto &current = board(position);
  int unsafe_positions = 0;
  int positive_changes = 0;
  int negative_changes = 0;
  int target_distance_before = 0;
  int target_distance_after = 0;
  int loss_limit = config.risk_budget_base + config.risk_budget_per_kill * int(position.killed.size());

  for (int slot_index = 0; slot_index < int(selected.size()); ++slot_index) {
    SlotIdx slot{slot_index};
    UnitId unit_id = selected.unit_at(slot);
    const auto &unit = current_unit(current, slot);
    const auto &original = root_unit(slot);
    // Contract: docs/combat-ai/combat-ai-contract.md#slot-states.
    if (position.slot_states[slot] == SlotState::ExcludedAtRoot || original.moves <= 0 || original.waiting) {
      if (trace) trace->event("end_skip_root", {{"unit",unit.id}});
      continue;
    }
    // Pinned CP 029e5988, CvTacticalAI.cpp:8761-8763, addFinishMovesIfAcceptable:
    // A_BLOCKED with movement left is handed back without checking its end
    // position, even after acting. Safe continuation is assumed, not verified.
    // Keep that CP policy separate from root exclusion; neither emits FINISH.
    if (position.slot_states[slot] == SlotState::Blocked) {
      if (trace) trace->event("end_skip_blocked", {{"unit",unit.id},{"hp",unit.hp},{"tile",unit.tile}});
      continue;
    }
    if (unit.hp <= 0) {
      evaluation.score -= original.hp;
      continue;
    }
    if (position.slot_states[slot] == SlotState::Pending) {
      if (trace) trace->event("end_skip_pending", {{"unit",unit.id},{"touched",position.touched[slot]}});
      // Contract: docs/combat-ai/combat-ai-contract.md#early-finish. Acted Pending units
      // are also handed back without end-position evaluation, but earn no bonus.
      if (!position.touched[slot])
        evaluation.score += kUntouchedEarlyFinishBonus;
      continue;
    }

    auto end = move_score(position, current, slot, unit.tile, 0, MoveEvaluation::Final);
    int end_score = end.move.score;
    if (!end.acceptable) {
      ++unsafe_positions;
      // CP:addFinishMovesIfAcceptable returns the first offender after the
      // available loss allowance is consumed. Do not change evaluation order.
      if (unsafe_positions > loss_limit && evaluation.bad_unit < 0)
        evaluation.bad_unit = unit_id.value;
    }
    auto last_move = latest_move_score(position, unit_id);
    evaluation.score += last_move ? std::max(0, end_score - *last_move) : end_score;

    int distance = current.plots[unit.tile].distance;
    if (roles[slot] == CAI_THIRD_LINE) {
      positive_changes += initial_distance[slot] == 1 && distance > 1;
      negative_changes += initial_distance[slot] > 1 && distance == 1;
    } else {
      int ideal_distance = roles[slot] == CAI_FIRST_LINE ? 1 : 2;
      positive_changes += initial_distance[slot] != ideal_distance && distance == ideal_distance;
      negative_changes += initial_distance[slot] == ideal_distance && distance != ideal_distance;
    }
    target_distance_before += adapter.distance(original.tile, target);
    target_distance_after += adapter.distance(unit.tile, target);
    evaluation.finishes.push_back(stationary_assignment(CAI_FINISH, unit, roles[slot], end_score));
  }

  bool attacked = has_attack_assignment(position);
  // Contract: docs/combat-ai/combat-ai-contract.md#plan-acceptance.
  bool improved = !position.killed.empty() || position.discovered ||
                  (!current.enemies.empty() ? (positive_changes > negative_changes ||
                                               (positive_changes == negative_changes && attacked))
                                            : target_distance_after < target_distance_before);
  evaluation.accepted = unsafe_positions <= loss_limit && improved;
  evaluation.risk_rejected = unsafe_positions > loss_limit;
  evaluation.improvement_rejected = !improved;
  if (trace) trace->event("final_evaluation", {{"accepted",evaluation.accepted},{"score",evaluation.score},
      {"unsafe",unsafe_positions},{"loss_limit",loss_limit},{"positive",positive_changes},{"negative",negative_changes},
      {"attacked",attacked},{"discovered",position.discovered},{"kills",int(position.killed.size())},
      {"distance_before",target_distance_before},{"distance_after",target_distance_after},
      {"risk_rejected",evaluation.risk_rejected},{"improvement_rejected",evaluation.improvement_rejected}});
  return evaluation;
}
} // namespace combat_ai
