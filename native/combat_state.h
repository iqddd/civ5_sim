#pragma once
#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <queue>
#include <vector>
#include "combat.h"
#include "combat_rules.h"
#include "geometry.h"

namespace combat_rules {
constexpr int strength[5] = {8, 5, 6, 11, 12}, ranged[5] = {0, 7, 10, 0, 0},
              moves[5] = {120, 120, 240, 120, 240};
enum { MOVE, MELEE, RANGED, FORTIFY, SKIP, END_TURN, SWAP };
enum {
  OK,
  BAD_ACTION,
  BAD_UNIT,
  NOT_ACTIVE,
  EXHAUSTED,
  BAD_TARGET,
  BLOCKED,
  NO_ATTACK,
  NO_FORTIFY
};
struct StepEvents { bool contact_interrupted = false; };
int damage(int a, int d, int roll);
bool valid_units(const Geometry& g, const Unit* units, int n);
struct State {
  std::shared_ptr<const Geometry> g;
  std::vector<Unit> units;
  std::vector<int> occupied;
  std::array<std::vector<int>, 2> visible;
  int side = 0, turn = 0, fog = 1;
  explicit State(std::shared_ptr<const Geometry> geo);
  void sight(const Unit& u, int delta);
  bool known(int observer, int tile) const;
  int known_unit(int tile, int observer) const;
  void rebuild();
  bool river(int from, int to) const;
  int movement(const Unit& u, int from, int to, int remaining,
               int observer) const;
  // One search per unit; friendly occupied cells are transit nodes only.
  void paths(int id, std::vector<int>& remaining,
             std::vector<int>& previous) const;
  int supports(int center, int supporting_side, int excluded, int observer,
               bool preview) const;
  int class_bonus(const Unit& u, const Unit& other) const;
  std::pair<int, int> strengths(int attacker, int defender, bool shot,
                                bool preview) const;
  Result combat(int a, int d, bool shot, bool preview,
                const std::function<int()>& roll = {}) const;
  int valid_attack(Action a) const;
  int validate(Action a) const;
  void relocate(Unit& u, int target);
  void kill(Unit& u);
  Result step(Action a, const std::function<int()>& roll = {},
              StepEvents* events = nullptr);
  int actions(Action* out, int capacity) const;
  std::vector<Action> actions() const;

 private:
  template <class Emit> void emit_actions(Emit&& emit) const;
};
}  // namespace combat_rules
