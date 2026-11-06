// Adapted from CvTacticalAI.cpp ScorePlotForCombatUnitMove and
// ScoreCombatUnitTurnEnd, pinned CP Only commit in combat-ai-sources.json.
#include "tactical.h"
#include "profile.h"
#include <algorithm>
#include <cmath>

namespace combat_ai {
TacticalView::TacticalView(const Branch& branch, int32_t side) {
  CAI_PROFILE_SCOPE(TacticalViewBuild);
  const auto& s = branch.state();
  units.assign(s.units, s.units + s.unit_count);
  visibility.assign(s.visibility, s.visibility + s.tile_count);
  occupied.assign(s.tile_count, -1);
  for (const auto& u : units) {
    if (u.hp <= 0 || (u.side != side && !visibility[u.tile])) continue;
    occupied[u.tile] = u.id;
    if (u.side != side) enemies.push_back(u.id);
  }
}
int32_t movement_role(const CaiUnit& u, int32_t damage_taken_last_turn) {
  if (u.type == 1 || u.type == 2) return CAI_SECOND_LINE;
  // CvUnit::isProjectedToDieNextTurn is historical damage, NOT GetDanger.
  return damage_taken_last_turn > u.hp ? CAI_THIRD_LINE : CAI_FIRST_LINE;
}
int32_t enemy_distance(const TacticalView& v, const RulesAdapter& a, int32_t tile) {
  int result = 4;
  for (int id : v.enemies) result = std::min(result, a.distance(tile, v.units[id].tile));
  return result;
}
int32_t formation_score(int32_t role, int32_t distance, bool enemies) {
  constexpr int combat[4][5] = {{-1,-1,-1,-1,-1}, {12,12,6,1,-1},
                               {-1,6,12,2,-1}, {1,1,8,12,-1}};
  constexpr int target[4][5] = {{-1,-1,-1,-1,-1}, {6,6,5,3,1},
                               {8,8,3,2,1}, {10,10,2,1,1}};
  // Keep index evaluation separate from array selection for portable codegen.
  const int column = std::clamp(distance, 0, 4);
  if (enemies) return 10 * combat[role][column];
  return 10 * target[role][column];
}
EndScore score_turn_end(const CaiConfig& config, const RulesAdapter& adapter,
                       DangerModel& danger, const Branch& branch,
                       const TacticalView& v, int32_t id, bool moved,
                       bool last_kill, bool can_flee, float bias) {
  const auto& u = v.units[id];
  int incoming = danger.incoming(branch, id, u.tile);
  int friends = 0, enemies = 0;
  bool cover = !v.enemies.empty(), toward_enemy = false;
  const int distance = enemy_distance(v, adapter, u.tile);
  for (int tile : adapter.neighbors(u.tile)) {
    int other = v.occupied[tile];
    if (other >= 0 && v.units[other].side == u.side) ++friends;
    if (other >= 0 && v.units[other].side != u.side) ++enemies;
    // CP hasCoverFromOtherUnits: every adjacent plot closer to an enemy is
    // occupied by a friendly. An open route toward the enemy breaks cover.
    if (enemy_distance(v, adapter, tile) < distance) {
      toward_enemy = true;
      if (other < 0 || v.units[other].side != u.side) cover = false;
    }
  }
  bool acceptable = !(enemies > 3 || (enemies == 3 && bias < 1.f));
  if (cover && toward_enemy) incoming /= 2; // CP Only CORE_TWO_PASS_DANGER=0.
  // CP's edge-of-known-world caution; only geometry, no hidden contents.
  bool edge = false;
  for (int tile : adapter.neighbors(u.tile)) if (!v.visibility[tile]) edge = true;
  if (edge) incoming = std::max(50, incoming);
  if (incoming > 0 && config.aggression != CAI_BRAVEHEART) {
    if ((u.type == 1 || u.type == 2) && incoming > u.hp &&
        (enemies > 0 || friends == 0)) acceptable = false;
    int magic = last_kill || !can_flee ? 23 : 42;
    if (last_kill && v.enemies.size() == 1) magic = 12;
    int threshold = magic * incoming / std::max(1, u.hp);
    if (u.hp * (friends + 1) < threshold) acceptable = false;
  }
  incoming /= friends + 1;
  int scaled = incoming * 100 / std::max(1, u.hp);
  if (scaled > 225) scaled = 15 * int(std::sqrt(double(scaled)));
  int score = -scaled;
  if (!moved) {
    if (u.hp < 100) ++score;
    if (distance < 3 && u.type != 2 && u.type != 4) ++score;
  }
  if (friends > 0) score += 7;
  bool known_to_enemy = !branch.state().fog;
  for (int other : v.enemies)
    if (adapter.can_see(branch, v.units[other].tile, u.tile)) known_to_enemy = true;
  if (!known_to_enemy) ++score;
  score += adapter.defense(u.tile) / 5;
  return {score, acceptable};
}
}  // namespace combat_ai
