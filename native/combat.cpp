#include "combat_state.h"

using namespace combat_rules;
namespace {
struct Battle : State {
  using State::State;
  int random = 1;
  uint64_t rng = 0;
  int roll() {
    rng += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t x = rng;
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return int(x % ATTACK_SAME_STRENGTH_POSSIBLE_EXTRA_DAMAGE);
  }
};
}  // namespace

int sim_abi() { return 2; }
const char* sim_rules() { return RULES_FINGERPRINT; }
uint64_t sim_memory(void* p, int include_geometry) {
  if (!p) return 0;
  const auto& s = *static_cast<Battle*>(p);
  uint64_t bytes = sizeof(Battle) + s.units.capacity() * sizeof(Unit) +
                   s.occupied.capacity() * sizeof(int);
  for (const auto& v : s.visible) bytes += v.capacity() * sizeof(int);
  for (const auto& v : s.visible_this_turn) bytes += v.capacity() * sizeof(uint8_t);
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
int sim_damage(int a, int d, int roll) { return damage(a, d, roll); }

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
    auto state = std::make_unique<Battle>(geo);
    if (n) state->units.assign(units, units + n);
    state->fog = fog;
    state->random = random;
    state->rng = seed;
    state->rebuild();
    return state.release();
  } catch (...) {
    return nullptr;
  }
}
void sim_free(void* p) { delete static_cast<Battle*>(p); }
void* sim_clone(void* p) {
  if (!p) return nullptr;
  try {
    return new Battle(*static_cast<Battle*>(p));
  } catch (...) {
    return nullptr;
  }
}
int sim_reset(void* p, const Unit* units, int n, const Info* info) {
  if (!p || !info || (!units && n) || n < 0 || n > 512) return BAD_UNIT;
  auto& s = *static_cast<Battle*>(p);
  if (info->side < 0 || info->side > 1 || info->turn < 0 || info->fog < 0 ||
      info->fog > 1 || info->random < 0 || info->random > 1 ||
      !valid_units(*s.g, units, n))
    return BAD_UNIT;
  if (n)
    s.units.assign(units, units + n);
  else
    s.units.clear();
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
  auto& s = *static_cast<Battle*>(p);
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
          out[i].damage_taken_this_turn = -1;
          out[i].damage_taken_last_turn = -1;
        }
      } else
        out[i] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    }
  return int(s.units.size());
}
int sim_visibility(void* p, int side, int* out) {
  if (!p || side < 0 || side > 1) return -1;
  auto& s = *static_cast<Battle*>(p);
  if (out)
    for (int i = 0; i < int(s.occupied.size()); ++i)
      out[i] = s.known(side, i) ? 1 : 0;
  return int(s.occupied.size());
}
int sim_visibility_history(void* p, int side, int* out) {
  if (!p || side < 0 || side > 1) return -1;
  const auto& s = *static_cast<Battle*>(p);
  if (out) std::copy(s.visible_this_turn[side].begin(), s.visible_this_turn[side].end(), out);
  return int(s.occupied.size());
}
int sim_restore_visibility(void* p, const int* masks, int count) {
  if (!p || !masks) return BAD_TARGET;
  auto& s = *static_cast<Battle*>(p);
  const int tiles = int(s.occupied.size());
  if (count != 2 * tiles) return BAD_TARGET;
  // Validate both rows before committing either: an invalid restore is atomic.
  for (int side = 0; side < 2; ++side)
    for (int tile = 0; tile < tiles; ++tile) {
      int value = masks[side * tiles + tile];
      if ((value != 0 && value != 1) ||
          (!value && s.visible[side][tile] > 0)) return BAD_TARGET;
    }
  for (int side = 0; side < 2; ++side)
    s.restore_visibility(side, masks + side * tiles, tiles);
  return OK;
}
int sim_info(void* p, Info* out) {
  if (!p || !out) return -1;
  auto& s = *static_cast<Battle*>(p);
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
  return static_cast<Battle*>(p)->actions(out, cap);
}
int sim_actions_with_previews(void* p, Action* out, float* features, int cap) {
  if (!p || !out || !features || cap < 0) return -1;
  return static_cast<Battle*>(p)->actions_with_previews(out, features, cap);
}

Result sim_step(void* p, Action a) {
  if (!p) return {BAD_ACTION, 0, 0, 0, 0, 0};
  try {
    auto& s = *static_cast<Battle*>(p);
    return s.step(a, s.random ? std::function<int()>([&s] { return s.roll(); })
                              : std::function<int()>());
  } catch (...) {
    return {BAD_ACTION, 0, 0, 0, 0, 0};
  }
}
Result sim_preview(void* p, Action a) {
  if (!p) return {BAD_ACTION, 0, 0, 0, 0, 0};
  auto& s = *static_cast<Battle*>(p);
  int error = s.validate(a);
  if (error) return {error, 0, 0, 0, 0, 0};
  if (a.kind != MELEE && a.kind != RANGED) return {BAD_ACTION, 0, 0, 0, 0, 0};
  return s.combat(a.unit, s.occupied[a.target], a.kind == RANGED, true);
}
void sim_batch(void** states, const Action* actions, Result* out, int n) {
  if (!states || !actions || !out || n < 0) return;
  for (int i = 0; i < n; ++i) out[i] = sim_step(states[i], actions[i]);
}

int sim_turn_reset_mode(void* p) {
  return p ? static_cast<Battle*>(p)->turn_reset_mode : -1;
}
int sim_set_turn_reset_mode(void* p, int mode) {
  if (!p || mode < 0 || mode > 1) return BAD_ACTION;
  static_cast<Battle*>(p)->turn_reset_mode = mode;
  return OK;
}
