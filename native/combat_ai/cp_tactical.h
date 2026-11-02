#pragma once
// Supported land-combat branches of the pinned CP Only CvTacticalPosition.
// Source mapping and engine-query adaptations: docs/combat-ai-cp-transfer.md.
#include "planner.h"
#include <map>

namespace combat_ai {
// Pinned CP's finite invalid score also participates in candidate arithmetic.
// Keep its soft-penalty behavior; this is not a hard admissibility sentinel.
inline constexpr int kCpImpossibleScore = -1000;
enum CpMoveKind : int32_t {
  CP_MOVE,
  CP_MELEE,
  CP_RANGED,
  CP_FINISH_TEMP,
  CP_BLOCKED,
  CP_FORCED,
  CP_SWAP,
  CP_SWAP_REVERSE
};
struct CpMove {
  int unit = -1, tile = -1;
  CpMoveKind kind = CP_MOVE;
  int score = kCpImpossibleScore;
  int remaining = 0, self_damage = 0, near_kill = -1;
  bool kill = false;
};
struct CpCombo {
  std::vector<CpMove> moves;
  int score = 0;
};
enum class MoveEvaluation { Intermediate, Initial, Final };
struct MoveScore {
  CpMove move;
  bool acceptable = false;
};
struct CpPlot {
  bool valid = false, blocked = false, end = false, edge = false;
  int enemy = -1, distance = 4, enemies = 0, friends = 0, friends_end = 0;
};
struct CpBoard {
  TacticalView view;
  std::vector<CpPlot> plots;
  std::vector<int> enemies;
  explicit CpBoard(const Branch &b, int side) : view(b, side) {}
};
// A search owns one context, including the root danger index and caches. The
// root branch must outlive it. These are not caches shared across searches.
class CpTactics {
  const RulesAdapter &adapter;
  const Branch &root;
  const CaiConfig &config;
  const CaiBattleState root_state;
  int side, target;
  const SelectedUnits selected;
  // Admission at setup is independent of later exclusion from active_order.
  UnitValues<uint8_t> admitted_at_setup;
  std::vector<int> root_visible, visibility_score;
  bool initialization_started = false;
  bool visibility_scores_ready = false;
  std::vector<std::vector<int>> attackers;
  std::vector<int> fog_danger;
  std::map<std::vector<int>, int> danger_cache;
  std::map<std::vector<int>, Damage> attack_cache;
  const CaiUnit &root_unit(UnitId) const;
  const CaiUnit &root_unit(SlotIdx) const;
  const CaiUnit &current_unit(const CpBoard &, SlotIdx) const;
  bool enemy_can_see(int tile) const;
  Damage damage(int attacker, int from, int defender, int to, int extra, bool unknown_origin = false);
  bool cover(const CpBoard &, int tile) const;
  enum class FriendCount { Current, Finished };
  struct EndQuery {
    int incoming_damage = 0;
    bool killed_enemy = false;
    int extra_self_damage = 0;
    FriendCount friend_count = FriendCount::Current;
    bool assume_extra_friend = false;
    bool has_full_moves = false;
  };
  EndScore turn_end(const CpBoard &, SlotIdx slot, int tile, EndQuery) const;
  CpMove attack(const TacticalPosition &, const CpBoard &, SlotIdx slot, int from, int enemy);
  int potential(const TacticalPosition &, const CpBoard &, SlotIdx slot, int tile, int num_attacks);
  bool can_end(const TacticalPosition &, const CpBoard &, SlotIdx slot, const CpMove &, int tile);

public:
  const TacticalView initial;
  SlotValues<int32_t> roles, safe, initial_distance;
  std::vector<int> initial_enemies;
  std::vector<SlotIdx> active_order;
  float bias = 1.f;
  // Root units own the last-turn history. It is fixed throughout this search.
  CpTactics(const RulesAdapter &, const Branch &, const CaiConfig &, const CaiSearchRequest &);
  // One-shot setup. Reinitializing the same search context is an invariant error.
  void initialize(TacticalPosition &);
  void extend_plots(TacticalPosition &, const std::vector<int32_t> &) const;
  CpBoard board(const TacticalPosition &) const;
  int danger(int unit, int tile, int self_damage, const std::vector<int> &killed);
  MoveScore move_score(const TacticalPosition &, const CpBoard &, SlotIdx slot, int tile, int remaining_moves,
                       MoveEvaluation mode);
  std::vector<CpMove> preferred(const TacticalPosition &, const CpBoard &, SlotIdx slot, int limit);
  std::vector<CpCombo> candidates(const TacticalPosition &, bool depth_first);
  // Pure with respect to the supplied position; only context query caches may
  // change. No FINISH records are installed until the caller accepts it.
  FinalEvaluation evaluate_final_position(const TacticalPosition &);
};
} // namespace combat_ai
