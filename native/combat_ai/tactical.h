#pragma once
#include "model.h"

namespace combat_ai {
// Numeric score is always finite, even when the ordinary end-position check
// rejects the plot. A risk-budget exception must not erase the danger penalty.
struct EndScore { int32_t score; bool acceptable; };
struct TacticalView {
  std::vector<CaiUnit> units;
  std::vector<int32_t> visibility, enemies, occupied;
  TacticalView(const Branch&, int32_t side);
};
int32_t movement_role(const CaiUnit&, int32_t damage_taken_last_turn);
int32_t enemy_distance(const TacticalView&, const RulesAdapter&, int32_t tile);
int32_t formation_score(int32_t role, int32_t distance, bool have_enemies);
EndScore score_turn_end(const CaiConfig&, const RulesAdapter&, DangerModel&,
                       const Branch&, const TacticalView&, int32_t id,
                       bool moved, bool last_kill, bool can_flee,
                       float aggression_bias);
}  // namespace combat_ai
