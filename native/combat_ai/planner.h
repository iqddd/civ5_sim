#pragma once
#include "tactical.h"
#include <map>

namespace combat_ai {
struct TacticalPosition {
  std::unique_ptr<Branch> branch;
  std::vector<CaiAssignment> assignments;
  std::vector<int32_t> done, moved, last_kill, touched, killed;
  int32_t generation = 0, attack_score = 0, rank = 0;
  int32_t final_score = 0;
  bool discovered = false, finalized = false;
};
struct SearchWorkspace {
  std::vector<std::unique_ptr<TacticalPosition>> positions;
  std::vector<int32_t> open;
  std::map<std::vector<int32_t>, std::pair<int32_t, int32_t>> equivalent;
  void clear();
};
struct Plan { std::vector<CaiAssignment> assignments; CaiSearchResult result; };
Plan find_assignments(SearchWorkspace&, const CaiConfig&, const CaiSearchRequest&);
}  // namespace combat_ai
