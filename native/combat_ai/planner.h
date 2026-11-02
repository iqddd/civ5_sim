#pragma once
#include "tactical.h"
#include "planner_types.h"
#include "cp_position.h"
#include <map>
#include <optional>

namespace combat_ai {
struct TacticalPosition {
  std::unique_ptr<Branch> branch;
  std::vector<CaiAssignment> assignments;
  SlotValues<SlotState> slot_states;
  SlotValues<MoveFlag> move_flags;
  SlotValues<int32_t> last_kill, touched;
  std::vector<int32_t> killed;
  // Tactical plots grow only when a simulated move reveals cells.
  std::vector<uint8_t> plots;
  // Copied with the CP position. board() synchronizes changed occupancy/end
  // fields lazily; enemy/visibility changes refresh volatile properties.
  mutable std::optional<CpBoard> tactical_board;
  int32_t generation = 0, attack_score = 0, rank = 0;
  bool discovered = false;
  // Transient: first actual contact source in this atomic candidate. Flushed
  // into RESTART after both moves; never inherited by a queued child.
  std::optional<UnitId> restart_source;
};
struct FinalEvaluation {
  bool accepted = false;
  int32_t score = 0;
  std::vector<CaiAssignment> finishes;
  int32_t bad_unit = CAI_NO_ID;
  bool risk_rejected = false, improvement_rejected = false;
};
bool terminal_position(const TacticalPosition&, const std::vector<int32_t>& initial_enemy_ids);
void finalize_assignments(TacticalPosition&, const FinalEvaluation&);
std::optional<int32_t> latest_move_score(const TacticalPosition&, UnitId);
bool has_attack_assignment(const TacticalPosition&);
// Includes history that affects final scoring, not only the physical board.
std::vector<int32_t> search_position_key(const TacticalPosition&, const TacticalView&,
                                       int32_t side, const SelectedUnits&);

class CpTactics;
struct CpCombo;
struct CpMove;
// Mutates only a disposable child. A rejected physical transition causes the
// caller to discard that child; malformed internal candidates throw.
class CandidateExecutor {
  const RulesAdapter& adapter;
  CpTactics& tactics;
  const SelectedUnits& selected;
  int32_t side;
  bool apply_command(TacticalPosition&, Command, SlotIdx, int32_t score);
 public:
  CandidateExecutor(const RulesAdapter& rules, CpTactics& context,
      const SelectedUnits& selection, int32_t observer)
      : adapter(rules), tactics(context), selected(selection), side(observer) {}
  bool apply(TacticalPosition&, const CpCombo&);
};
struct SearchWorkspace {
  CaiSearchDiagnostics diagnostics{};
  // CvTactPosStorage: fixed addresses, peek at the next slot and consume only
  // after accepting a child. Rejected children reuse the unconsumed slot.
  std::vector<TacticalPosition> positions;
  int32_t used_positions = 0;
  int32_t previous_side = CAI_NO_ID;
  std::vector<int32_t> previous_map;
  std::vector<int32_t> open;
  std::map<std::vector<int32_t>, std::pair<int32_t, int32_t>> equivalent;
  void clear(bool hard = false);
  void release_unused_branches();
};
struct Plan { std::vector<CaiAssignment> assignments; CaiSearchResult result; };
Plan find_assignments(SearchWorkspace&, const CaiConfig&, const CaiSearchRequest&);
Plan support_assignments(const CaiSearchRequest&, int32_t mode);
Plan recovery_assignments(const CaiSearchRequest&, CaiSearchDiagnostics&);
std::vector<int32_t> attack_candidates(const CaiSearchRequest&);
}  // namespace combat_ai
