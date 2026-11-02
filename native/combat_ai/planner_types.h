#pragma once
#include "combat_ai.h"
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

namespace combat_ai {
// IDs address the complete physical roster; slots address only selected units.
// Neither type implicitly converts to an integer or to the other index space.
struct UnitId {
  int32_t value = CAI_NO_ID;
  explicit constexpr UnitId(int32_t id = CAI_NO_ID) : value(id) {}
  bool operator==(UnitId other) const { return value == other.value; }
  bool operator!=(UnitId other) const { return !(*this == other); }
};
struct SlotIdx {
  int32_t value = -1;
  explicit constexpr SlotIdx(int32_t slot = -1) : value(slot) {}
  bool valid() const { return value >= 0; }
  bool operator==(SlotIdx other) const { return value == other.value; }
  bool operator!=(SlotIdx other) const { return !(*this == other); }
};

template<class Index, class Value> class IndexedValues {
  std::vector<Value> values;
 public:
  IndexedValues() = default;
  IndexedValues(size_t count, Value value) : values(count,value) {}
  explicit IndexedValues(std::vector<Value> data) : values(std::move(data)) {}
  Value& operator[](Index index) {
    assert(index.value >= 0 && size_t(index.value) < values.size());
    return values.at(size_t(index.value));
  }
  const Value& operator[](Index index) const {
    assert(index.value >= 0 && size_t(index.value) < values.size());
    return values.at(size_t(index.value));
  }
  void assign(size_t count, Value value) { values.assign(count,value); }
  void push_back(Value value) { values.push_back(value); }
  size_t size() const { return values.size(); }
  auto begin() { return values.begin(); }
  auto end() { return values.end(); }
  auto begin() const { return values.begin(); }
  auto end() const { return values.end(); }
  bool operator==(const IndexedValues& other) const { return values == other.values; }
};
template<class T> using SlotValues = IndexedValues<SlotIdx,T>;
template<class T> using UnitValues = IndexedValues<UnitId,T>;

enum class SlotState : uint8_t {
  Pending,
  Finished,
  // BLOCKED never consumes a physical turn; the coordinator may reuse a
  // still-available unit. Also used for root-ineligible selected units.
  Blocked
};
enum class MoveFlag : uint8_t { None, Moved, Swapped };

class SelectedUnits {
  SlotValues<UnitId> units;
  UnitValues<SlotIdx> slots;
 public:
  SelectedUnits(const int32_t* selected, int32_t count, int32_t roster_size);
  UnitId unit_at(SlotIdx slot) const { return units[slot]; }
  SlotIdx slot_of(UnitId unit) const;
  size_t size() const { return units.size(); }
};

// Named C++17 construction preserves the C ABI without eleven positional ints.
inline CaiAssignment stationary_assignment(CaiAssignmentKind kind,
    const CaiUnit& unit, int32_t role, int32_t score = 0) {
  CaiAssignment record{};
  record.kind = kind;
  record.unit = unit.id;
  record.from_tile = unit.tile;
  record.to_tile = unit.tile;
  record.remaining_moves = unit.moves;
  record.movement_role = role;
  record.score = score;
  record.damage = 0;
  record.self_damage = 0;
  record.killed_unit = CAI_NO_ID;
  record.command_target = CAI_NO_ID;
  return record;
}
} // namespace combat_ai
