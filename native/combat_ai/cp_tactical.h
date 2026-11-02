#pragma once
// Supported land-combat branches of the pinned CP Only CvTacticalPosition.
// Source mapping and engine-query adaptations: docs/combat-ai-cp-transfer.md.
#include "planner.h"
#include <map>

namespace combat_ai {
enum CpMoveKind : int32_t { CP_MOVE, CP_MELEE, CP_RANGED, CP_FINISH_TEMP, CP_BLOCKED,
                  CP_FORCED, CP_SWAP, CP_SWAP_REVERSE };
struct CpMove {
  int unit = -1, tile = -1;
  CpMoveKind kind = CP_MOVE;
  int score = -1000;
  int remaining = 0, self_damage = 0, near_kill = -1;
  bool kill = false;
};
struct CpCombo { std::vector<CpMove> moves; int score = 0; };
struct CpPlot {
  bool valid = false, blocked = false, end = false, edge = false;
  int enemy = -1, distance = 4, enemies = 0, friends = 0, friends_end = 0;
};
struct CpBoard {
  TacticalView view;
  std::vector<CpPlot> plots;
  std::vector<int> enemies;
  explicit CpBoard(const Branch& b, int side) : view(b,side) {}
};
// A search owns one context, including the root danger index and caches. The
// root branch must outlive it. These are not caches shared across searches.
class CpTactics {
  const RulesAdapter& adapter;
  const Branch& root;
  const CaiConfig& config;
  CaiBattleState map;
  int side, target;
  std::vector<int> selected, index, admitted, root_visible, visibility_score;
  std::vector<int32_t> damage_history;
  std::vector<std::vector<int>> attackers;
  std::vector<int> fog_danger;
  std::map<std::vector<int>,int> danger_cache;
  std::map<std::vector<int>,Damage> attack_cache;
  Damage damage(int attacker, int from, int defender, int to, int extra,
                bool unknown_origin = false);
  bool cover(const CpBoard&, int tile) const;
  EndScore turn_end(const TacticalPosition&, const CpBoard&, SlotIdx slot, int tile,
                    int incoming, bool kill, int extra_self, bool final,
                    bool relaxed, bool full_moves) const;
  CpMove attack(const TacticalPosition&, const CpBoard&, SlotIdx slot, int from, int enemy);
  int potential(const TacticalPosition&, const CpBoard&, SlotIdx slot, int tile,
                int num_attacks, int& near_kill);
  bool can_end(const TacticalPosition&, const CpBoard&, SlotIdx slot, const CpMove&, int tile);
 public:
  TacticalView initial;
  SlotValues<int32_t> roles, safe, initial_distance;
  std::vector<int> initial_enemies;
  std::vector<SlotIdx> active_order;
  float bias = 1.f;
  // ABI 5 has no historical damage input: the empty profile explicitly means
  // zero. A C++ caller with real history can provide one value per roster ID.
  CpTactics(const RulesAdapter&, const Branch&, const CaiConfig&, const CaiSearchRequest&,
            const std::vector<int32_t>& previous_damage = {});
  void initialize(TacticalPosition&);
  void extend_plots(TacticalPosition&, const std::vector<int32_t>&) const;
  CpBoard board(const TacticalPosition&) const;
  int danger(int unit, int tile, int self_damage, const std::vector<int>& killed);
  CpMove move_score(const TacticalPosition&, const CpBoard&, SlotIdx slot, int tile,
                    int remaining, bool final, bool* acceptable = nullptr,
                    bool initial_eval = false);
  std::vector<CpMove> preferred(const TacticalPosition&, const CpBoard&, SlotIdx slot, int limit);
  std::vector<CpCombo> candidates(const TacticalPosition&, bool depth_first);
  // Pure with respect to the supplied position; only context query caches may
  // change. No FINISH records are installed until the caller accepts it.
  FinalEvaluation evaluate_final_position(const TacticalPosition&);
};
} // namespace combat_ai
