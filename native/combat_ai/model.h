#pragma once
// C++ extension boundary for CP-02/03; declarations only in CP-01.
#include "combat_ai.h"

#include <memory>
#include <vector>

namespace combat_ai {
struct ReachableTile { int32_t tile, remaining_moves; };
struct Command { int32_t kind, unit, target; };  // simulator Action kinds
struct Damage { int32_t damage, retaliation; };
struct Transition {
  bool accepted, contact_interrupted;
  std::vector<int32_t> newly_visible;
};
// CP-03 remembers observed enemy IDs, not arbitrary never-seen enemies.
// Owned counterpart of CaiSearchRequest::contacts, copied for every root.
struct ContactMemory { int32_t turn = 0; std::vector<int32_t> known_enemies; };

class Branch {
 public:
  virtual ~Branch() = default;
  virtual std::unique_ptr<Branch> clone() const = 0;
  // Own mutable unit/visibility state; immutable map tables may be shared.
  virtual const CaiBattleState& state() const = 0;
  virtual const ContactMemory& contacts() const = 0;
};

class RulesAdapter {
 public:
  virtual ~RulesAdapter() = default;
  virtual std::unique_ptr<Branch> root(const CaiBattleState& state,
                                       const ContactMemory& contacts) const = 0;
  virtual std::vector<ReachableTile> reachable(const Branch&, int32_t unit) const = 0;
  virtual std::vector<Command> actions(const Branch&) const = 0;
  virtual Damage preview(const Branch&, Command) const = 0; // fixed roll 600
  virtual Transition apply(Branch&, Command) const = 0;     // fixed roll 600
  virtual int32_t distance(int32_t from, int32_t to) const = 0;
  virtual std::vector<int32_t> neighbors(int32_t tile) const = 0;
  virtual bool can_see(const Branch&, int32_t from, int32_t to) const = 0;
  virtual bool can_shoot(const Branch&, int32_t unit, int32_t to) const = 0;
  // Privileged v1 queries only at the CP source call sites. No RNG access.
  // prospective_units reads only cells newly visible after an assumed move.
  virtual std::vector<CaiUnit> prospective_units(
      const Branch&, const std::vector<int32_t>& newly_visible) const = 0;
  // known_enemy requires an ID from ContactMemory; returns false when dead.
  virtual bool known_enemy(const Branch&, int32_t id, CaiUnit& out) const = 0;
};

class DangerModel {
 public:
  virtual ~DangerModel() = default;
  // CP-03: defender-dependent incoming damage; branch-local cache, no real RNG.
  virtual int32_t incoming(const Branch&, int32_t defender, int32_t tile) = 0;
};
}  // namespace combat_ai
