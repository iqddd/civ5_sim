// Standalone CP Only land-combat subset. See docs/combat.md for rule
// provenance.
#include "combat.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

#include "combat_rules.h"
#include "geometry.h"

namespace {
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
struct State {
  std::shared_ptr<const Geometry> g;
  std::vector<Unit> units;
  std::vector<int> occupied;
  std::array<std::vector<int>, 2> visible;
  int side = 0, turn = 0, fog = 1, random = 1;
  uint64_t rng = 0;
  explicit State(std::shared_ptr<const Geometry> geo)
      : g(std::move(geo)), occupied(g->tiles.size(), -1) {
    for (auto& v : visible) v.resize(g->tiles.size());
  }
  void sight(const Unit& u, int delta) {
    if (u.hp > 0)
      for (int tile : g->sight[u.tile]) visible[u.side][tile] += delta;
  }
  bool known(int observer, int tile) const {
    return !fog || visible[observer][tile] > 0;
  }
  int known_unit(int tile, int observer) const {
    int id = occupied[tile];
    return id < 0 || units[id].side == observer || known(observer, tile) ? id
                                                                         : -1;
  }
  void rebuild() {
    std::fill(occupied.begin(), occupied.end(), -1);
    for (auto& v : visible) std::fill(v.begin(), v.end(), 0);
    for (int i = 0; i < int(units.size()); ++i)
      if (units[i].hp > 0) {
        occupied[units[i].tile] = i;
        sight(units[i], 1);
      }
  }
  int roll() {
    // SplitMix64: all bits and state serialized, independent of host libraries.
    rng += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t x = rng;
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return int(x % ATTACK_SAME_STRENGTH_POSSIBLE_EXTRA_DAMAGE);
  }
  bool river(int from, int to) const {
    for (int d = 0; d < 6; ++d)
      if (g->neighbors[from][d] == to)
        return (g->tiles[from].rivers & (1 << d)) != 0;
    return false;
  }
  int movement(const Unit& u, int from, int to, int remaining,
               int observer) const {
    int cost = g->cost(to);
    if (river(from, to) || (u.type == 2 && g->rough(to))) return remaining;
    // Crossing between two adjacent cells of the SAME enemy's ZOC.
    for (int n : g->neighbors[from])
      if (n >= 0) {
        int id = known_unit(n, observer);
        if (id >= 0 && units[id].side != u.side && g->distance(n, to) == 1)
          return remaining;
      }
    return std::min(cost, remaining);
  }
  // One search per unit; friendly occupied cells are transit nodes only.
  void paths(int id, std::vector<int>& remaining,
             std::vector<int>& previous) const {
    const Unit& u = units[id];
    remaining.assign(g->tiles.size(), -1);
    previous.assign(g->tiles.size(), -1);
    std::priority_queue<std::pair<int, int>> queue;
    remaining[u.tile] = u.moves;
    queue.push({u.moves, u.tile});
    while (!queue.empty()) {
      auto [left, from] = queue.top();
      queue.pop();
      if (left != remaining[from] || left <= 0) continue;
      for (int to : g->neighbors[from])
        if (g->passable(to)) {
          int occ = known_unit(to, u.side);
          if (occ >= 0 && units[occ].side != u.side) continue;
          int after = left - movement(u, from, to, left, u.side);
          if (occ >= 0 && after == 0) continue;
          if (after > remaining[to]) {
            remaining[to] = after;
            previous[to] = from;
            queue.push({after, to});
          }
        }
    }
  }
  int supports(int center, int supporting_side, int excluded, int observer,
               bool preview) const {
    int count = 0;
    for (int tile : g->neighbors[center])
      if (tile >= 0) {
        int id = occupied[tile];
        if (id < 0 || id == excluded || units[id].side != supporting_side)
          continue;
        // CP counts only supporters seen by the other side. Preview cannot
        // consult that side's hidden visibility sources; it uses known units.
        if (preview) {
          if (units[id].side == observer || known(observer, tile)) ++count;
        } else if (visible[1 - supporting_side][tile] > 0)
          ++count;
      }
    return count;
  }
  int class_bonus(const Unit& u, const Unit& other) const {
    // CP-only IsMounted defaults to false for the chariot archer. Its combat
    // class is ARCHER, so only HORSEMAN receives the anti-mounted modifier.
    return u.type == 3 && other.type == 4 ? 50 : 0;
  }
  std::pair<int, int> strengths(int attacker, int defender, bool shot,
                                bool preview) const {
    const Unit& a = units[attacker];
    const Unit& d = units[defender];
    int am =
        -((100 - a.hp) * WOUNDED_DAMAGE_MULTIPLIER / 100) + class_bonus(a, d);
    int dm =
        -((100 - d.hp) * WOUNDED_DAMAGE_MULTIPLIER / 100) + class_bonus(d, a);
    bool no_terrain = d.type == 2 || d.type == 4;
    int terrain = g->defense(d.tile);
    dm += no_terrain ? std::min(terrain, 0) : terrain;
    if (d.fortified && !no_terrain) dm += FORTIFY_MODIFIER_PER_TURN;
    if (!shot) {
      if (river(a.tile, d.tile)) am += RIVER_ATTACK_MODIFIER;
      int ours = supports(d.tile, a.side, attacker, a.side, preview);
      int theirs = supports(a.tile, d.side, defender, a.side, preview);
      am += std::max(0, ours - theirs) * BONUS_PER_ADJACENT_FRIEND;
      dm += std::max(0, theirs - ours) * BONUS_PER_ADJACENT_FRIEND;
    }
    int abase = shot ? ranged[a.type] : strength[a.type];
    int dbase = shot && ranged[d.type] ? ranged[d.type] : strength[d.type];
    return {std::max(1, abase * std::max(10, 100 + am)),
            std::max(1, dbase * std::max(10, 100 + dm))};
  }
  Result combat(int a, int d, bool shot, bool preview) {
    auto [as, ds] = strengths(a, d, shot, preview);
    int dealt = sim_damage(
        as, ds,
        random && !preview ? roll()
                           : ATTACK_SAME_STRENGTH_POSSIBLE_EXTRA_DAMAGE / 2);
    int taken =
        shot ? 0
             : sim_damage(ds, as,
                          random && !preview
                              ? roll()
                              : ATTACK_SAME_STRENGTH_POSSIBLE_EXTRA_DAMAGE / 2);
    const Unit& au = units[a];
    const Unit& du = units[d];
    // CP's overkill correction precedes mutual-death correction.
    if (!shot && dealt > du.hp) taken = taken * du.hp / dealt;
    if (!shot && dealt >= du.hp && taken >= au.hp) {
      if (dealt + (100 - du.hp) > taken + (100 - au.hp))
        taken = au.hp - 1;
      else
        dealt = du.hp - 1;
    }
    return {OK, 0, std::min(dealt, du.hp), std::min(taken, au.hp), as, ds};
  }
  int valid_attack(Action a) const {
    const Unit& u = units[a.unit];
    if (a.target < 0 || a.target >= int(occupied.size())) return BAD_TARGET;
    int d = known_unit(a.target, u.side);
    if (d < 0 || units[d].side == u.side) return BAD_TARGET;
    if (u.attacked) return NO_ATTACK;
    if (a.kind == MELEE) {
      if (ranged[u.type]) return NO_ATTACK;
      if (g->distance(u.tile, a.target) != 1) return BAD_TARGET;
    } else {
      if (!ranged[u.type]) return NO_ATTACK;
      const auto& targets = g->targets[u.tile];
      if (!std::binary_search(targets.begin(), targets.end(), a.target))
        return BLOCKED;
    }
    return OK;
  }
  int validate(Action a) const {
    if (a.kind < 0 || a.kind > SWAP) return BAD_ACTION;
    if (a.kind == END_TURN) return OK;
    if (a.unit < 0 || a.unit >= int(units.size()) || units[a.unit].hp <= 0)
      return BAD_UNIT;
    const Unit& u = units[a.unit];
    if (u.side != side) return NOT_ACTIVE;
    if (u.moves <= 0 || u.waiting) return EXHAUSTED;
    if (a.kind == MELEE || a.kind == RANGED) return valid_attack(a);
    if (a.kind == FORTIFY && (u.type == 2 || u.type == 4)) return NO_FORTIFY;
    if (a.kind == SWAP) {
      if (!g->passable(a.target) || g->distance(u.tile, a.target) != 1)
        return BAD_TARGET;
      int partner = known_unit(a.target, u.side);
      if (partner < 0 || units[partner].side != u.side) return BAD_TARGET;
      // CP ReadyToSwap requires movement, not an unused attack. A processed
      // skip/fortify order does not block the partner; the swap wakes it.
      if (units[partner].moves <= 0) return EXHAUSTED;
    }
    if (a.kind == MOVE) {
      if (!g->passable(a.target) || a.target >= int(occupied.size()) ||
          a.target == u.tile)
        return BAD_TARGET;
      if (known_unit(a.target, u.side) >= 0) return BLOCKED;
    }
    return OK;
  }
  void relocate(Unit& u, int target) {
    sight(u, -1);
    occupied[u.tile] = -1;
    u.tile = target;
    occupied[target] = int(&u - units.data());
    sight(u, 1);
  }
  void kill(Unit& u) {
    sight(u, -1);
    occupied[u.tile] = -1;
    u.hp = 0;
    u.moves = 0;
    u.fortified = 0;
    u.waiting = 1;
  }
  Result step(Action a) {
    int error = validate(a);
    if (error) return {error, 0, 0, 0, 0, 0};
    if (a.kind == END_TURN) {
      side = 1 - side;
      ++turn;
      for (Unit& u : units)
        if (u.side == side && u.hp > 0) {
          if (!u.acted) {
            u.hp = std::min(100, u.hp + NEUTRAL_HEAL_RATE);
            if (u.type != 2 && u.type != 4) u.fortified = 1;
          }
          u.moves = moves[u.type];
          u.attacked = 0;
          u.acted = 0;
          u.waiting = 0;
        }
      return {};
    }
    Unit& u = units[a.unit];
    if (a.kind == SWAP) {
      int partner = occupied[a.target];
      Unit& other = units[partner];
      // Compute both costs against the original occupancy and visibility.
      int cost = movement(u, u.tile, other.tile, u.moves, u.side);
      int other_cost =
          movement(other, other.tile, u.tile, other.moves, other.side);
      sight(u, -1);
      sight(other, -1);
      std::swap(u.tile, other.tile);
      occupied[u.tile] = a.unit;
      occupied[other.tile] = partner;
      u.moves -= cost;
      other.moves -= other_cost;
      u.acted = other.acted = 1;
      u.fortified = other.fortified = 0;
      u.waiting = other.waiting = 0;
      sight(u, 1);
      sight(other, 1);
      return {OK, 2, 0, 0, 0, 0};
    }
    if (a.kind == MOVE) {
      std::vector<int> remaining, previous;
      std::vector<int> path;
      if (g->distance(u.tile, a.target) == 1)
        path.push_back(a.target);
      else {
        paths(a.unit, remaining, previous);
        if (remaining[a.target] < 0) return {BLOCKED, 0, 0, 0, 0, 0};
        for (int tile = a.target; tile != u.tile; tile = previous[tile])
          path.push_back(tile);
        std::reverse(path.begin(), path.end());
      }
      Result result{};
      // Friendly transit is committed atomically between unoccupied stops.
      // A newly visible enemy/ZOC can stop the order before its destination.
      std::vector<int> seen;
      if (fog)
        for (int i = 0; i < int(units.size()); ++i)
          if (units[i].hp > 0 && units[i].side != u.side &&
              known(u.side, units[i].tile))
            seen.push_back(i);
      int from = u.tile, left = u.moves, traversed = 0;
      for (int tile : path) {
        int occ = occupied[tile];
        if (occ >= 0 && units[occ].side != u.side) break;
        int after = left - movement(u, from, tile, left, u.side);
        if (occ >= 0 && after == 0) break;
        from = tile;
        left = after;
        ++traversed;
        if (occ < 0) {
          relocate(u, tile);
          u.moves = left;
          u.acted = 1;
          u.fortified = 0;
          result.moved = traversed;
          bool contact = false;
          if (fog)
            for (int i = 0; i < int(units.size()); ++i)
              if (units[i].hp > 0 && units[i].side != u.side &&
                  known(u.side, units[i].tile) &&
                  std::find(seen.begin(), seen.end(), i) == seen.end()) {
                contact = true;
                break;
              }
          if (contact)
            break;  // expose a new contact before continuing a long order
        }
        if (left == 0) break;
      }
      return result;
    }
    if (a.kind == FORTIFY || a.kind == SKIP) {
      // A skip consumes no movement and therefore permits natural healing.
      // CP fortification becomes effective at the next owner-turn reset.
      u.waiting = 1;
      return {};
    }
    int defender = occupied[a.target];
    Result result = combat(a.unit, defender, a.kind == RANGED, false);
    Unit& d = units[defender];
    int spent = movement(u, u.tile, d.tile, u.moves, u.side);
    if (a.kind == RANGED) spent = 60;
    u.moves = std::max(0, u.moves - spent);
    u.attacked = 1;
    u.acted = 1;
    u.fortified = 0;
    if (d.hp <= result.damage)
      kill(d);
    else
      d.hp -= result.damage;
    if (u.hp <= result.retaliation)
      kill(u);
    else
      u.hp -= result.retaliation;
    if (u.hp > 0 && d.hp == 0 && a.kind == MELEE) {
      relocate(u, a.target);
      result.moved = 1;
    }
    if (u.type != 4) u.moves = 0;
    return result;
  }
};
bool valid_units(const Geometry& g, const Unit* units, int n) {
  std::vector<int> occupied(g.tiles.size(), 0);
  for (int i = 0; i < n; ++i) {
    const auto& u = units[i];
    if (u.type < 0 || u.type > 4 || u.side < 0 || u.side > 1 || u.hp < 0 ||
        u.hp > 100 || u.tile < 0 || u.tile >= int(g.tiles.size()) ||
        !g.passable(u.tile) || u.moves < 0 || u.moves > moves[u.type] ||
        u.attacked < 0 || u.attacked > 1 || u.acted < 0 || u.acted > 1 ||
        u.fortified < 0 || u.fortified > 1 || u.waiting < 0 || u.waiting > 1 ||
        (u.fortified && (u.type == 2 || u.type == 4)))
      return false;
    if (u.hp > 0 && occupied[u.tile]++) return false;
  }
  return true;
}
}  // namespace

int sim_abi() { return 1; }
const char* sim_rules() { return RULES_FINGERPRINT; }
uint64_t sim_memory(void* p, int include_geometry) {
  if (!p) return 0;
  const auto& s = *static_cast<State*>(p);
  uint64_t bytes = sizeof(State) + s.units.capacity() * sizeof(Unit) +
                   s.occupied.capacity() * sizeof(int);
  for (const auto& v : s.visible) bytes += v.capacity() * sizeof(int);
  if (include_geometry) {
    const auto& g = *s.g;
    bytes += sizeof(Geometry) + g.tiles.capacity() * sizeof(Tile) +
             g.neighbors.capacity() * sizeof(std::array<int, 6>);
    for (const auto* list : {&g.sight, &g.targets}) {
      bytes += list->capacity() * sizeof(std::vector<int>);
      for (const auto& v : *list) bytes += v.capacity() * sizeof(int);
    }
  }
  return bytes;
}
int sim_damage(int a, int d, int roll) {
  if (a < 1 || d < 1 || roll < 0 ||
      roll >= ATTACK_SAME_STRENGTH_POSSIBLE_EXTRA_DAMAGE)
    return -1;
  double ratio = double(std::max(a, d)) / std::min(a, d);
  ratio = std::min(1000.0, (std::pow((ratio + 3) / 4, 4.0) + 1) / 2);
  if (d > a) ratio = 1 / ratio;
  return std::max(0,
                  int((ATTACK_SAME_STRENGTH_MIN_DAMAGE + roll) * ratio) / 100);
}
void* sim_new(int w, int h, int parity, int wrap, const Tile* tiles,
              const Unit* units, int n, int fog, int random, uint64_t seed) {
  if (w < 1 || h < 1 || w > 128 || h > 128 || w * h > 4096 || n < 0 ||
      n > 512 || !tiles || (!units && n) || parity < 0 || parity > 1 ||
      wrap < 0 || wrap > 1 || fog < 0 || fog > 1 || random < 0 || random > 1)
    return nullptr;
  for (int i = 0; i < w * h; ++i)
    if (tiles[i].terrain < 0 || tiles[i].terrain > 6 ||
        tiles[i].elevation < 0 || tiles[i].elevation > 3 ||
        tiles[i].feature < 0 || tiles[i].feature > 8 || tiles[i].rivers < 0 ||
        tiles[i].rivers > 63)
      return nullptr;
  try {
    auto geo = std::make_shared<Geometry>(w, h, parity, wrap, tiles);
    if (!valid_units(*geo, units, n)) return nullptr;
    auto state = std::make_unique<State>(geo);
    state->units.assign(units, units + n);
    state->fog = fog;
    state->random = random;
    state->rng = seed;
    state->rebuild();
    return state.release();
  } catch (...) {
    return nullptr;
  }
}
void sim_free(void* p) { delete static_cast<State*>(p); }
void* sim_clone(void* p) {
  if (!p) return nullptr;
  try {
    return new State(*static_cast<State*>(p));
  } catch (...) {
    return nullptr;
  }
}
int sim_reset(void* p, const Unit* units, int n, const Info* info) {
  if (!p || !info || (!units && n) || n < 0 || n > 512) return BAD_UNIT;
  auto& s = *static_cast<State*>(p);
  if (info->side < 0 || info->side > 1 || info->turn < 0 || info->fog < 0 ||
      info->fog > 1 || info->random < 0 || info->random > 1 ||
      !valid_units(*s.g, units, n))
    return BAD_UNIT;
  s.units.assign(units, units + n);
  s.side = info->side;
  s.turn = info->turn;
  s.fog = info->fog;
  s.random = info->random;
  s.rng = info->rng;
  s.rebuild();
  return 0;
}
int sim_units(void* p, int observer, Unit* out) {
  if (!p || observer < -1 || observer > 1) return -1;
  auto& s = *static_cast<State*>(p);
  if (out)
    for (int i = 0; i < int(s.units.size()); ++i) {
      const auto& u = s.units[i];
      if (observer < 0 ||
          (u.hp > 0 && (u.side == observer || s.known(observer, u.tile)))) {
        out[i] = u;
        if (observer >= 0 && u.side != observer) {
          out[i].moves = -1;
          out[i].attacked = -1;
          out[i].acted = -1;
          out[i].waiting = -1;
        }
      } else
        out[i] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    }
  return int(s.units.size());
}
int sim_visibility(void* p, int side, int* out) {
  if (!p || side < 0 || side > 1) return -1;
  auto& s = *static_cast<State*>(p);
  if (out)
    for (int i = 0; i < int(s.occupied.size()); ++i)
      out[i] = s.known(side, i) ? 1 : 0;
  return int(s.occupied.size());
}
int sim_info(void* p, Info* out) {
  if (!p || !out) return -1;
  auto& s = *static_cast<State*>(p);
  *out = {s.side, s.turn, s.fog, s.random, 0, 0, s.rng};
  for (const auto& u : s.units)
    if (u.hp > 0) {
      if (u.side)
        ++out->alive1;
      else
        ++out->alive0;
    }
  return 0;
}
int sim_actions(void* p, Action* out, int cap) {
  if (!p || cap < 0) return -1;
  const auto& s = *static_cast<State*>(p);
  int count = 0;
  auto emit = [&](Action a) {
    if (out && count < cap) out[count] = a;
    ++count;
  };
  emit({END_TURN, -1, -1});
  std::vector<int> remaining, previous;
  for (int id = 0; id < int(s.units.size()); ++id) {
    const auto& u = s.units[id];
    if (u.side != s.side || u.hp <= 0 || u.moves <= 0 || u.waiting) continue;
    emit({SKIP, id, -1});
    if (u.type != 2 && u.type != 4) emit({FORTIFY, id, -1});
    for (int tile : s.g->neighbors[u.tile])
      if (tile >= 0 && s.validate({SWAP, id, tile}) == OK)
        emit({SWAP, id, tile});
    s.paths(id, remaining, previous);
    for (int tile = 0; tile < int(remaining.size()); ++tile)
      if (remaining[tile] >= 0 && tile != u.tile &&
          s.known_unit(tile, u.side) < 0)
        emit({MOVE, id, tile});
    if (u.attacked) continue;
    if (ranged[u.type]) {
      for (int tile : s.g->targets[u.tile])
        if (s.valid_attack({RANGED, id, tile}) == 0) emit({RANGED, id, tile});
    } else {
      for (int tile : s.g->neighbors[u.tile])
        if (tile >= 0 && s.valid_attack({MELEE, id, tile}) == 0)
          emit({MELEE, id, tile});
    }
  }
  return count;
}
Result sim_step(void* p, Action a) {
  if (!p) return {BAD_ACTION, 0, 0, 0, 0, 0};
  try {
    return static_cast<State*>(p)->step(a);
  } catch (...) {
    return {BAD_ACTION, 0, 0, 0, 0, 0};
  }
}
Result sim_preview(void* p, Action a) {
  if (!p) return {BAD_ACTION, 0, 0, 0, 0, 0};
  auto& s = *static_cast<State*>(p);
  int error = s.validate(a);
  if (error) return {error, 0, 0, 0, 0, 0};
  if (a.kind != MELEE && a.kind != RANGED) return {BAD_ACTION, 0, 0, 0, 0, 0};
  return s.combat(a.unit, s.occupied[a.target], a.kind == RANGED, true);
}
void sim_batch(void** states, const Action* actions, Result* out, int n) {
  if (!states || !actions || !out || n < 0) return;
  for (int i = 0; i < n; ++i) out[i] = sim_step(states[i], actions[i]);
}
