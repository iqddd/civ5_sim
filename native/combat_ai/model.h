#pragma once
// C++ extension boundary for combat AI rules and branch state.
#include "combat_ai.h"

#include <memory>
#include <vector>

namespace combat_ai {
struct ReachableTile { int32_t tile, remaining_moves; };
struct Command { int32_t kind, unit, target; };  // simulator Action kinds
struct CommandResult { int32_t code, moved, damage, retaliation, attacker_strength, defender_strength; };
struct Damage { int32_t damage, retaliation, code, attacker_strength, defender_strength; };
struct Transition {
  bool accepted, contact_interrupted;
  std::vector<int32_t> newly_visible;
  CommandResult result;
};
struct Contact { int32_t id, last_seen_own_turn; };
// Owned counterpart of CaiSearchRequest::contacts, copied for every root.
struct ContactMemory {
  int32_t turn = 0, side = -1;
  std::vector<Contact> known_enemies;
};
// Counts observer-turn starts, up to a harmless initial offset. No int32 overflow.
int32_t own_turn(int32_t turn, int32_t active_side, int32_t observer);

class Branch {
 public:
  virtual ~Branch() = default;
  virtual std::unique_ptr<Branch> clone() const = 0;
  // Own mutable unit/visibility state; immutable map tables may be shared.
  // The view and its arrays are owned by this branch and remain valid until
  // its next state() call, mutation, or destruction. Map tiles are shared.
  virtual const CaiBattleState& state() const = 0;
  virtual const ContactMemory& contacts() const = 0;
};

class RulesAdapter {
 public:
  virtual ~RulesAdapter() = default;
  virtual std::unique_ptr<Branch> root(const CaiBattleState& state,
                                       const ContactMemory& contacts) const = 0;
  // Planned reach uses known occupancy and ignores future contacts; those can
  // stop a move earlier or change its cost. Adjacent moves use the direct edge.
  virtual std::vector<ReachableTile> reachable(const Branch&, int32_t unit) const = 0;
  // CP MOVEFLAG_IGNORE_STACKING_SELF: occupied friendly destinations remain
  // intentions, not executable commands. Enemy cells are endpoints only.
  virtual std::vector<ReachableTile> move_intents(const Branch&, int32_t unit) const = 0;
  // CP danger-map reach: maximum moves, ignore occupancy and ZOC.
  virtual std::vector<int32_t> threatened_tiles(const Branch&, int32_t unit) const = 0;
  // ScoreAttackDamage/GetDanger query at assumed plots, without mutation,
  // movement validation or revealing contacts. Damage uses simulator rules.
  virtual Damage forecast(const Branch&, int32_t attacker, int32_t from,
      int32_t defender, int32_t to, int32_t extra_defender_damage = 0,
      bool ignore_adjacency = true, bool unknown_origin = false) const = 0;
  virtual int32_t attack_moves(const Branch&, Command) const = 0;
  virtual std::vector<Command> actions(const Branch&) const = 0;
  virtual bool can_execute(const Branch&, Command) const = 0;
  // next_turn refreshes only the attacker's moves/attack/wait flags on a
  // private copy, preserving branch visibility, contact history and turn.
  virtual Damage preview(const Branch&, Command, bool next_turn = false) const = 0; // roll 600
  virtual Transition apply(Branch&, Command) const = 0;     // fixed roll 600
  virtual int32_t distance(int32_t from, int32_t to) const = 0;
  virtual std::vector<int32_t> neighbors(int32_t tile) const = 0;
  virtual int32_t defense(int32_t tile) const = 0;
  // Geometric sight only; independent of branch visibility.
  virtual bool can_see(const Branch&, int32_t from, int32_t to) const = 0;
  // Geometric targeting from a living ranged unit; ignores movement and target contents.
  virtual bool can_shoot(const Branch&, int32_t unit, int32_t to) const = 0;
  virtual bool can_target(int32_t from, int32_t to) const = 0;
  // Privileged v1 queries only at the CP source call sites. No RNG access.
  // prospective_units reads only cells newly visible after an assumed move.
  virtual std::vector<CaiUnit> prospective_units(
      const Branch&, const std::vector<int32_t>& newly_visible) const = 0;
  // known_enemy requires an ID from ContactMemory; returns false when dead.
  virtual bool known_enemy(const Branch&, int32_t id, CaiUnit& out) const = 0;
};

// Invalid snapshots, foreign branches, query IDs and tile indices throw
// std::invalid_argument. Invalid commands return the core result code.
std::unique_ptr<RulesAdapter> make_rules_adapter(const CaiBattleState&, int32_t observer);

class DangerModel {
 public:
  virtual ~DangerModel() = default;
  // Next enemy response, including CP probability/fog heuristics. The defender
  // must be living and ours; tile must be passable and free (or its own tile).
  // Queries never discover contacts. Cache belongs to the branch, no real RNG.
  virtual int32_t incoming(const Branch&, int32_t defender, int32_t tile) = 0;
};
// Adapter must outlive the model. Foreign branches/unsupported adapters throw.
std::unique_ptr<DangerModel> make_danger_model(const RulesAdapter&);
}  // namespace combat_ai
