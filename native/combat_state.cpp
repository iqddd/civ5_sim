#include "combat_state.h"
#include <cmath>
#include <limits>

namespace combat_rules {
State::State(std::shared_ptr<const Geometry> geo)
      : g(std::move(geo)), occupied(g->tiles.size(), -1) {
    for (auto& v : visible) v.resize(g->tiles.size());
    for (auto& v : visible_this_turn) v.resize(g->tiles.size());
  }

void State::sight(const Unit& u, int delta) {
    if (u.hp > 0)
      for (int tile : g->sight[u.tile]) {
        visible[u.side][tile] += delta;
        if (visible[u.side][tile] > 0) visible_this_turn[u.side][tile] = 1;
      }
  }

bool State::known(int observer, int tile) const {
    return !fog || visible_this_turn[observer][tile] != 0;
  }

int State::known_unit(int tile, int observer) const {
    int id = occupied[tile];
    return id < 0 || units[id].side == observer || known(observer, tile) ? id
                                                                         : -1;
  }

void State::rebuild() {
    std::fill(occupied.begin(), occupied.end(), -1);
    for (auto& v : visible) std::fill(v.begin(), v.end(), 0);
    for (auto& v : visible_this_turn) std::fill(v.begin(), v.end(), 0);
    for (int i = 0; i < int(units.size()); ++i)
      if (units[i].hp > 0) {
        occupied[units[i].tile] = i;
        sight(units[i], 1);
      }
  }

void State::flip_visibility(int observer) {
    // CvPlot::flipVisibility, CvPlayer::setTurnActive: expire at turn end/start.
    for (int tile = 0; tile < int(occupied.size()); ++tile)
      visible_this_turn[observer][tile] = visible[observer][tile] > 0;
  }

bool State::restore_visibility(int observer, const int* mask, int count) {
    if (observer < 0 || observer > 1 || !mask || count != int(occupied.size())) return false;
    for (int tile = 0; tile < count; ++tile)
      if ((mask[tile] != 0 && mask[tile] != 1) ||
          (mask[tile] == 0 && visible[observer][tile] > 0)) return false;
    for (int tile = 0; tile < count; ++tile) visible_this_turn[observer][tile] = uint8_t(mask[tile]);
    return true;
  }

bool State::river(int from, int to) const {
    for (int d = 0; d < 6; ++d)
      if (g->neighbors[from][d] == to)
        return (g->tiles[from].rivers & (1 << d)) != 0;
    return false;
  }

int State::movement(const Unit& u, int from, int to, int remaining,
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

void State::paths(int id, std::vector<int>& remaining,
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

int State::supports(int center, int supporting_side, int excluded, int observer,
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
        } else if (visible_this_turn[1 - supporting_side][tile])
          ++count;
      }
    return count;
  }

int State::class_bonus(const Unit& u, const Unit& other) const {
    // CP-only IsMounted defaults to false for the chariot archer. Its combat
    // class is ARCHER, so only HORSEMAN receives the anti-mounted modifier.
    return u.type == 3 && other.type == 4 ? 50 : 0;
  }

std::pair<int, int> State::strengths(int attacker, int defender, bool shot,
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

Result State::combat(int a, int d, bool shot, bool preview,
                const std::function<int()>& roll) const {
    auto [as, ds] = strengths(a, d, shot, preview);
    int dealt = damage(as, ds, roll ? roll() : 600);
    int taken = shot ? 0 : damage(ds, as, roll ? roll() : 600);
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

int State::valid_attack(Action a) const {
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

int State::validate(Action a) const {
    if (a.kind < 0 || a.kind > SWAP) return BAD_ACTION;
    if (a.kind == END_TURN)
      return turn == std::numeric_limits<int>::max() ? BAD_ACTION : OK;
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

void State::relocate(Unit& u, int target) {
    sight(u, -1);
    occupied[u.tile] = -1;
    u.tile = target;
    occupied[target] = int(&u - units.data());
    sight(u, 1);
  }

void State::kill(Unit& u) {
    sight(u, -1);
    occupied[u.tile] = -1;
    u.hp = 0;
    u.moves = 0;
    u.fortified = 0;
    u.waiting = 1;
  }

Result State::step(Action a, const std::function<int()>& roll,
              StepEvents* events) {
    if (events) events->contact_interrupted = false;
    int error = validate(a);
    if (error) return {error, 0, 0, 0, 0, 0};
    if (a.kind == END_TURN) {
      const int reset_side = turn_reset_mode ? side : 1 - side;
      flip_visibility(side);
      side = 1 - side;
      flip_visibility(side);
      ++turn;
      // CvUnit::flipDamageReceivedPerTurn / CvPlayer::SetAllUnitsUnprocessed.
      // Rotate before healing; healing does not subtract from this history.
      for (Unit& unit : units) if (unit.side == side) {
        unit.damage_taken_last_turn = unit.damage_taken_this_turn;
        unit.damage_taken_this_turn = 0;
      }
      for (Unit& u : units)
        if (u.side == reset_side && u.hp > 0) {
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
          if (contact) {
            if (events && tile != a.target) events->contact_interrupted = true;
            break;
          }  // expose a new contact before continuing a long order
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
    Result result = combat(a.unit, defender, a.kind == RANGED, false, roll);
    Unit& d = units[defender];
    // CvUnitCombat.cpp:1895-1901 explicitly excludes the attacker's retaliation
    // damage. Actual defender damage is capped by HP in combat(). Saturation
    // only protects externally supplied counters near int32's limit.
    d.damage_taken_this_turn = int(std::min(int64_t(std::numeric_limits<int>::max()),
        int64_t(d.damage_taken_this_turn) + result.damage));
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

int damage(int a, int d, int roll) {
  if (a < 1 || d < 1 || roll < 0 ||
      roll >= ATTACK_SAME_STRENGTH_POSSIBLE_EXTRA_DAMAGE)
    return -1;
  double ratio = double(std::max(a, d)) / std::min(a, d);
  ratio = std::min(1000.0, (std::pow((ratio + 3) / 4, 4.0) + 1) / 2);
  if (d > a) ratio = 1 / ratio;
  return std::max(0,
                  int((ATTACK_SAME_STRENGTH_MIN_DAMAGE + roll) * ratio) / 100);
}
bool valid_units(const Geometry& g, const Unit* units, int n) {
  std::vector<int> occupied(g.tiles.size(), 0);
  for (int i = 0; i < n; ++i) {
    const auto& u = units[i];
    if (u.type < 0 || u.type > 4 || u.side < 0 || u.side > 1 || u.hp < 0 ||
        u.hp > 100 || u.tile < 0 || u.tile >= int(g.tiles.size()) ||
        !g.passable(u.tile) || u.moves < 0 || u.moves > moves[u.type] ||
        u.attacked < 0 || u.attacked > 1 || u.acted < 0 || u.acted > 1 ||
        u.fortified < 0 || u.fortified > 1 || u.waiting < 0 || u.waiting > 1 ||
        (u.fortified && (u.type == 2 || u.type == 4)) ||
        u.damage_taken_this_turn < 0 || u.damage_taken_last_turn < 0)
      return false;
    if (u.hp > 0 && occupied[u.tile]++) return false;
  }
  return true;
}
template <class Emit>
void State::emit_actions(Emit&& emit) const {
  if (turn < std::numeric_limits<int>::max()) emit({END_TURN, -1, -1});
  std::vector<int> remaining, previous;
  for (int id = 0; id < int(units.size()); ++id) {
    const auto& u = units[id];
    if (u.side != side || u.hp <= 0 || u.moves <= 0 || u.waiting) continue;
    emit({SKIP, id, -1});
    if (u.type != 2 && u.type != 4) emit({FORTIFY, id, -1});
    for (int tile : g->neighbors[u.tile])
      if (tile >= 0 && validate({SWAP, id, tile}) == OK)
        emit({SWAP, id, tile});
    paths(id, remaining, previous);
    for (int tile = 0; tile < int(remaining.size()); ++tile)
      if (remaining[tile] >= 0 && tile != u.tile &&
          known_unit(tile, u.side) < 0)
        emit({MOVE, id, tile});
    if (u.attacked) continue;
    if (ranged[u.type]) {
      for (int tile : g->targets[u.tile])
        if (valid_attack({RANGED, id, tile}) == 0) emit({RANGED, id, tile});
    } else {
      for (int tile : g->neighbors[u.tile])
        if (tile >= 0 && valid_attack({MELEE, id, tile}) == 0)
          emit({MELEE, id, tile});
    }
  }
}
int State::actions(Action* out, int capacity) const {
  int count = 0;
  emit_actions([&](Action a) {
    if (out && count < capacity) out[count] = a;
    ++count;
  });
  return count;
}
std::vector<Action> State::actions() const {
  std::vector<Action> result;
  emit_actions([&](Action a) { result.push_back(a); });
  return result;
}

int State::actions_with_previews(Action* out, float* features, int capacity) const {
  int count = 0;
  emit_actions([&](Action action) {
    if (count < capacity) out[count] = action;
    ++count;
  });
  const int stored = std::min(count, capacity);
  std::sort(out, out + stored, [](const Action& a, const Action& b) {
    if (a.kind != b.kind) return a.kind < b.kind;
    if (a.unit != b.unit) return a.unit < b.unit;
    return a.target < b.target;
  });
  for (int i = 0; i < stored; ++i) {
      const Action action = out[i];
      float* row = features + 5 * i;
      std::fill(row, row + 5, 0.f);
      if (action.kind == MELEE || action.kind == RANGED) {
        // emit_actions already validated the command against public information.
        const int defender = occupied[action.target];
        const auto result = combat(action.unit, defender, action.kind == RANGED, true);
        row[0] = 1.f;
        row[1] = result.damage / 100.f;
        row[2] = result.retaliation / 100.f;
        row[3] = result.damage >= units[defender].hp ? 1.f : 0.f;
        row[4] = result.retaliation >= units[action.unit].hp ? 1.f : 0.f;
      }
  }
  return count;
}
}  // namespace combat_rules
