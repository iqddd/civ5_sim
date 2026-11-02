#pragma once
#include "tactical.h"
#include <map>

namespace combat_ai {
struct TacticalPosition {
  std::unique_ptr<Branch> branch;
  std::vector<CaiAssignment> assignments;
  std::vector<int32_t> done, moved, last_kill, touched, killed;
  // Tactical plots grow only when a simulated move reveals cells. done==2
  // means BLOCKED: the unit remains available to the outer coordinator.
  std::vector<int32_t> plots;
  int32_t generation = 0, attack_score = 0, rank = 0;
  int32_t final_score = 0;
  bool discovered = false, finalized = false, pending_restart = false;
};
struct SearchWorkspace {
  std::vector<std::unique_ptr<TacticalPosition>> positions;
  std::vector<int32_t> open;
  std::map<std::vector<int32_t>, std::pair<int32_t, int32_t>> equivalent;
  void clear();
};
struct Plan { std::vector<CaiAssignment> assignments; CaiSearchResult result; };
Plan find_assignments(SearchWorkspace&, const CaiConfig&, const CaiSearchRequest&);
Plan support_assignments(const CaiSearchRequest&, int32_t mode);
}  // namespace combat_ai
