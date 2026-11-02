#pragma once
#include "tactical.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifndef CAI_CP_AREA
#define CAI_CP_AREA 1
#endif

namespace combat_ai {
// Supported properties of pinned CP's CvTacticalPlot / CvTacticalPosition.
struct CpPlot {
  bool valid = false, blocked = false, end = false, edge = false;
  int16_t enemy = -1;  // ABI roster <=512; -1 is the absence sentinel.
  uint8_t distance = 4, enemies = 0, friends = 0, friends_end = 0;
};
static_assert(sizeof(CpPlot) <= 12, "tactical properties must remain compact");

// CP stores at most 255 tactical plots. Global map tiles remain distinct from
// their byte-sized local indices; 255 means absent, never global tile 255.
// The historical area-off ablation has no 255-plot limit and uses wider indices.
class CpPlots {
  using LocalIndex = std::conditional_t<CAI_CP_AREA != 0, uint8_t, uint16_t>;
  static constexpr LocalIndex absent = std::numeric_limits<LocalIndex>::max();
  std::vector<LocalIndex> indices;
  std::vector<CpPlot> records;
  CpPlot missing{};

 public:
  void resize(size_t map_size) {
    indices.assign(map_size, absent);
    records.clear();
    records.reserve(std::min(map_size, size_t(absent)));
  }
  size_t size() const { return indices.size(); }
  size_t local_size() const { return records.size(); }
  const CpPlot &operator[](size_t tile) const {
    auto index = indices.at(tile);
    return index == absent ? missing : records[index];
  }
  CpPlot &add(size_t tile) {
    auto &index = indices.at(tile);
    if (index == absent) {
      if (records.size() >= size_t(absent))
        throw std::logic_error("tactical plot index overflow");
      index = static_cast<LocalIndex>(records.size());
      records.emplace_back();
    }
    return records[index];
  }
  CpPlot &at(size_t tile) {
    auto index = indices.at(tile);
    if (index == absent)
      throw std::logic_error("write to absent tactical plot");
    return records[index];
  }
};
struct CpBoard {
  TacticalView view;
  CpPlots plots;
  std::vector<int> enemies;
  explicit CpBoard(TacticalView value) : view(std::move(value)) {}
};
}  // namespace combat_ai
