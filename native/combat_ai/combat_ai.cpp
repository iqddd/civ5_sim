#include "combat_ai.h"
#include "model.h"
#include "planner.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

// No game objects, battle handles, global scratch buffers or RNG live here.
struct CaiContext {
  CaiConfig config;
  combat_ai::SearchWorkspace workspace;
  explicit CaiContext(const CaiConfig& input) : config(input) {}
};

namespace {
bool bit(int32_t value) { return value == 0 || value == 1; }
bool valid_config(const CaiConfig& c) {
  return c.position_capacity >= 1 && c.position_capacity <= 1000000 &&
         c.max_completed >= 1 && c.max_completed <= 4000 &&
         c.max_branches >= 2 && c.max_branches <= 9 &&
         c.choices_per_unit >= 2 && c.choices_per_unit <= 9 &&
         c.aggression >= CAI_NONE && c.aggression <= CAI_BRAVEHEART &&
         c.privileged_cp_queries == 1 &&
         c.risk_budget_base >= 0 && c.risk_budget_base <= 13 &&
         c.risk_budget_per_kill >= 0 && c.risk_budget_per_kill <= 13;
}
bool valid_request(const CaiSearchRequest& r) {
  if (!r.state || r.side < 0 || r.side > 1 || r.expected_turn < 0 ||
      r.selected_count < 0 || r.selected_count > CAI_MAX_SELECTED ||
      (r.selected_count && !r.selected_units)) return false;
  const auto& s = *r.state;
  if (s.width < 1 || s.width > 128 || s.height < 1 || s.height > 128 ||
      s.tile_count != s.width * s.height || s.tile_count > 4096 ||
      !s.tiles || !bit(s.row_parity) || !bit(s.wrap_x) || !bit(s.fog) ||
      s.active_side < 0 || s.active_side > 1 || s.turn < 0 ||
      s.unit_count < 0 || s.unit_count > 512 || (s.unit_count && !s.units) ||
      !s.visibility || s.visibility_count != s.tile_count ||
      r.target < 0 || r.target >= s.tile_count ||
      r.contacts.turn < 0 || r.contacts.turn > s.turn ||
      (r.contacts.side != r.side && !(r.contacts.side == -1 &&
       r.contacts.turn == 0 && r.contacts.known_enemy_count == 0)) ||
      r.contacts.known_enemy_count < 0 || r.contacts.known_enemy_count > s.unit_count ||
      (r.contacts.known_enemy_count && !r.contacts.known_enemies)) return false;
  for (int32_t i = 0; i < s.tile_count; ++i) {
    const auto& t = s.tiles[i];
    if (t.terrain < 0 || t.terrain > 6 || t.elevation < 0 || t.elevation > 3 ||
        t.feature < 0 || t.feature > 8 || t.rivers < 0 || t.rivers > 63 ||
        !bit(s.visibility[i])) return false;
  }
  std::vector<int32_t> occupied(s.tile_count, -1);
  constexpr std::array<int32_t, 5> max_moves = {120, 120, 240, 120, 240};
  for (int32_t i = 0; i < s.unit_count; ++i) {
    const auto& u = s.units[i];
    // IDs are exactly the simulator's stable roster indices, never remapped.
    if (u.id != i || u.type < 0 || u.type > 4 || u.side < 0 || u.side > 1 ||
        u.tile < 0 || u.tile >= s.tile_count || u.hp < 0 || u.hp > 100 ||
        u.moves < 0 || u.moves > max_moves[u.type] || !bit(u.attacked) ||
        !bit(u.acted) || !bit(u.fortified) || !bit(u.waiting) ||
        (u.fortified && (u.type == 2 || u.type == 4)))
      return false;
    const auto& t = s.tiles[u.tile];
    if (t.elevation >= 2 || t.feature == 5) return false;
    if (u.hp > 0) {
      if (occupied[u.tile] != -1) return false;
      occupied[u.tile] = u.id;
    }
  }
  std::array<bool, 512> selected{};
  for (int32_t i = 0; i < r.selected_count; ++i) {
    int32_t id = r.selected_units[i];
    if (id < 0 || id >= s.unit_count || selected[id] ||
        s.units[id].side != r.side || s.units[id].hp <= 0) return false;
    selected[id] = true;
  }
  std::array<bool, 512> known{};
  int previous_side = s.active_side ^ ((s.turn - r.contacts.turn) & 1);
  int observed_turn = combat_ai::own_turn(r.contacts.turn,previous_side,r.side);
  for (int32_t i = 0; i < r.contacts.known_enemy_count; ++i) {
    const auto& contact = r.contacts.known_enemies[i];
    int32_t id = contact.id;
    if (id < 0 || id >= s.unit_count || known[id] || s.units[id].side == r.side)
      return false;
    if (contact.last_seen_own_turn < 0 || contact.last_seen_own_turn > observed_turn)
      return false;
    // Dead contacts are allowed in historical memory; CP-03 adds no threat.
    known[id] = true;
  }
  return true;
}
}  // namespace

int32_t cai_abi(void) { return CAI_ABI; }
const char* cai_version(void) { return "civ5sim.combat_ai/0.4.0 CP-04"; }
CaiConfig cai_default_config(void) { return {6000, 2560, 5, 5, CAI_MEDIUM, 1, 0, 1}; }
CaiContext* cai_new(const CaiConfig* config) {
  if (!config || !valid_config(*config)) return nullptr;
  try { return new CaiContext(*config); } catch (...) { return nullptr; }
}
void cai_free(CaiContext* context) { delete context; }
int32_t cai_search(CaiContext* context, const CaiSearchRequest* request,
                   CaiAssignment* assignments, int32_t capacity,
                   CaiSearchResult* result) {
  if (result) *result = {CAI_INVALID_REQUEST, 0, {0, 0, 0, 0, 0}};
  if (!context || !request || !result || capacity < 0 ||
      (capacity && !assignments)) return CAI_BAD_ARGUMENT;
  try {
    if (!valid_request(*request)) return CAI_OK;
    if (request->expected_turn != request->state->turn ||
        request->side != request->state->active_side) {
      result->reason = CAI_TURN_CHANGED;
    } else if (request->selected_count == 0) {
      result->reason = CAI_NO_PLAN;
    } else {
      auto plan = combat_ai::find_assignments(context->workspace, context->config, *request);
      *result = plan.result;
      if (capacity < result->assignment_count) return CAI_BUFFER_TOO_SMALL;
      std::copy(plan.assignments.begin(), plan.assignments.end(), assignments);
    }
    return CAI_OK;
  } catch (const std::invalid_argument&) {
    *result = {CAI_INVALID_REQUEST, 0, {0, 0, 0, 0, 0}};
    return CAI_OK;
  } catch (...) { return CAI_INTERNAL_ERROR; }
}
