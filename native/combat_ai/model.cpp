#include "model.h"
#include "trace.h"
#include "storage_metrics.h"
#include "profile.h"
#include "../combat_state.h"
#include <algorithm>
#include <iterator>
#include <map>
#include <queue>
#include <stdexcept>
#include <utility>

namespace combat_ai {
int32_t own_turn(int32_t turn, int32_t active_side, int32_t observer) {
  if (turn < 0 || (active_side != 0 && active_side != 1) || (observer != 0 && observer != 1))
    throw std::invalid_argument("invalid turn/side");
  return int32_t((int64_t(turn) + (active_side == observer)) / 2);
}
namespace {
using combat_rules::State;
// Five-type simulator ABI; review these assumptions when adding unit mechanics.
enum class UnitType { Warrior, Archer, ChariotArcher, Spearman, Horseman, Count };
constexpr int kMoveDenominator = 60;
constexpr int kMaxHitPoints = 100;
constexpr int kPercent = 100;
constexpr int kMinStrengthPercent = 10;
constexpr int kRangedAttackRange = 2;
constexpr int kMeleeAttackRange = 1;
constexpr int kFogRadius = 2;
// Pinned CvDangerPlots.cpp:24,171-201,882-913 (CP Only).
constexpr int kFogDefaultDanger = 1;
constexpr int kUnseenTargetDamagePercent = 80;
constexpr int kSuicideDamageDivisor = 2;
static_assert(std::size(combat_rules::moves) == int(UnitType::Count) &&
                  std::size(combat_rules::strength) == int(UnitType::Count) &&
                  std::size(combat_rules::ranged) == int(UnitType::Count),
              "Review adapter traits when extending the five-type rules");
constexpr bool full_moves_use_whole_points() {
  for (int amount : combat_rules::moves)
    if (amount <= 0 || amount % kMoveDenominator != 0)
      return false;
  return true;
}
static_assert(full_moves_use_whole_points(),
              "Danger distance bound requires whole movement points");
bool is_type(const Unit& unit, UnitType type) { return unit.type == static_cast<int>(type); }
bool is_ranged(const Unit& unit) { return combat_rules::ranged[unit.type] != 0; }
int attack_range(const Unit& unit) {
  return is_ranged(unit) ? kRangedAttackRange : kMeleeAttackRange;
}
// Both adapter searches maximize remaining movement. Their expansion and edge
// policies intentionally differ: move intentions respect known enemies/ZOC,
// while CP danger reach ignores occupancy and ZOC. State::paths additionally
// rejects occupied endpoints and remains the physical kernel's own query.
template <class CanExpand, class MovementCost>
std::vector<int> movement_remainders(const Geometry& geometry, int origin, int initial_moves,
                                     CanExpand can_expand, MovementCost movement_cost) {
  std::vector<int> remaining(geometry.tiles.size(), -1);
  std::priority_queue<std::pair<int, int>> pending;
  remaining[origin] = initial_moves;
  pending.push({initial_moves, origin});
  while (!pending.empty()) {
    auto [moves_left, from] = pending.top();
    pending.pop();
    if (moves_left != remaining[from] || moves_left <= 0 || !can_expand(from))
      continue;
    for (int to : geometry.neighbors[from]) {
      // passable() also rejects the -1 sentinel at non-wrapped map edges.
      if (!geometry.passable(to))
        continue;
      int after = moves_left - movement_cost(from, to, moves_left);
      if (after > remaining[to]) {
        remaining[to] = after;
        pending.push({after, to});
      }
    }
  }
  return remaining;
}
int destination_moves(const State& state, const Unit& unit, int target, int searched_moves) {
  // The kernel executes an adjacent command directly even when pathfinding
  // found a cheaper detour. Apply that same correction to both reach queries.
  if (target != unit.tile && state.g->distance(unit.tile, target) == kMeleeAttackRange)
    return unit.moves - state.movement(unit, unit.tile, target, unit.moves, unit.side);
  return searched_moves;
}
void validate_model_geometry(const Geometry& geometry) {
  // Guard the conservative incoming() distance bound against future changes
  // to kernel terrain costs or geometric firing range.
  for (int from = 0; from < int(geometry.tiles.size()); ++from) {
    if (geometry.passable(from) && geometry.cost(from) < kMoveDenominator)
      throw std::logic_error("Danger distance bound requires terrain cost >= 60");
    for (int target : geometry.targets[from])
      if (geometry.distance(from, target) > kRangedAttackRange)
        throw std::logic_error("Danger distance bound requires firing range <= 2");
  }
}
bool bit(int n) { return n == 0 || n == 1; }
void require(bool condition, const char* message) {
  if (!condition)
    throw std::invalid_argument(message);
}
CaiUnit export_unit(int unit_id, const Unit& unit) {
  return {unit_id,
          unit.type,
          unit.side,
          unit.tile,
          unit.hp,
          unit.moves,
          unit.attacked,
          unit.acted,
          unit.fortified,
          unit.waiting,
          unit.damage_taken_last_turn};
}
CommandResult export_result(const Result& result) {
  return {result.code,
          result.moved,
          result.damage,
          result.retaliation,
          result.attacker_strength,
          result.defender_strength};
}
void validate_header(const CaiBattleState& state) {
  require(state.width >= 1 && state.width <= 128 && state.height >= 1 && state.height <= 128 &&
              state.tile_count == state.width * state.height && state.tile_count <= 4096 &&
              state.tiles && bit(state.row_parity) && bit(state.wrap_x) && bit(state.fog) &&
              bit(state.active_side) && state.turn >= 0 && state.unit_count >= 0 &&
              state.unit_count <= 512 && (!state.unit_count || state.units) && state.visibility &&
              state.visibility_count == state.tile_count,
          "invalid battle snapshot");
  require(state.visibility_history_count == 0 ||
              (state.visibility_history && state.visibility_history_count == 2 * state.tile_count),
          "invalid visibility history");
  for (int tile_index = 0; tile_index < state.tile_count; ++tile_index) {
    const auto& map_tile = state.tiles[tile_index];
    require(map_tile.terrain >= 0 && map_tile.terrain <= 6 && map_tile.elevation >= 0 &&
                map_tile.elevation <= 3 && map_tile.feature >= 0 && map_tile.feature <= 8 &&
                map_tile.rivers >= 0 && map_tile.rivers <= 63 && bit(state.visibility[tile_index]),
            "invalid tile or visibility");
  }
}
class OwnedBranch final : public Branch {
 public:
  State rules;
  int observer;
  ContactMemory memory;
  std::vector<int32_t> last_accepted_visibility;
  std::shared_ptr<const std::vector<CaiTile>> tiles;
  mutable std::vector<CaiUnit> units;
  mutable std::vector<int32_t> visibility;
  mutable std::vector<int32_t> visibility_history;
  mutable CaiBattleState snapshot{};
  mutable bool snapshot_valid = false;
  mutable std::map<std::pair<int32_t, int32_t>, int32_t> danger_cache;
  OwnedBranch(std::shared_ptr<const Geometry> geometry,
              std::shared_ptr<const std::vector<CaiTile>> map_tiles, int side)
      : rules(std::move(geometry)), observer(side), tiles(std::move(map_tiles)) {}
  std::unique_ptr<Branch> clone() const override {
    CAI_STORAGE_COUNT(BranchAllocation);
    auto copy = std::make_unique<OwnedBranch>(*this);
    // Export buffers are branch-owned. Never retain the parent's raw view.
    copy->snapshot = {};
    copy->snapshot_valid = false;
    copy->danger_cache.clear();
    return copy;
  }
  std::unique_ptr<Branch> clone_into(std::unique_ptr<Branch> storage) const override {
    auto* copy = dynamic_cast<OwnedBranch*>(storage.get());
    if (!copy)
      return clone();
    require(copy != this, "cannot overwrite the source branch");
    CAI_STORAGE_COUNT(BranchReuse);
    // Vector assignment retains capacity. No cached export pointers or danger
    // results survive; immutable geometry belongs to the source search now.
    copy->rules = rules;
    copy->observer = observer;
    copy->memory = memory;
    copy->last_accepted_visibility = last_accepted_visibility;
    copy->tiles = tiles;
    copy->units.clear();
    copy->visibility.clear();
    copy->visibility_history.clear();
    copy->snapshot = {};
    copy->snapshot_valid = false;
    copy->danger_cache.clear();
    return storage;
  }
  const CaiBattleState& state() const override {
    if (snapshot_valid) return snapshot;
    CAI_PROFILE_SCOPE(StateExport);
    units.clear();
    for (int index = 0; index < int(rules.units.size()); ++index)
      units.push_back(export_unit(index, rules.units[index]));
    visibility.resize(tiles->size());
    for (int index = 0; index < int(tiles->size()); ++index)
      visibility[index] = rules.known(observer, index) ? 1 : 0;
    visibility_history.clear();
    for (const auto& row : rules.visible_this_turn)
      visibility_history.insert(visibility_history.end(), row.begin(), row.end());
    snapshot = {rules.g->w,        rules.g->h,
                rules.g->parity,   rules.g->wrap,
                tiles->data(),     int32_t(tiles->size()),
                units.data(),      int32_t(units.size()),
                visibility.data(), int32_t(visibility.size()),
                rules.side,        rules.turn,
                rules.fog, visibility_history.data(), int32_t(visibility_history.size())};
    snapshot_valid = true;
    return snapshot;
  }
  const ContactMemory& contacts() const override { return memory; }
  void observe() {
    snapshot_valid = false;
    danger_cache.clear();
    memory.side = observer;
    // Branch-local observation time; the caller's real memory is a copy.
    // Keep contacts() valid as input to root(state(), contacts()).
    memory.turn = rules.turn;
    int current_own_turn = own_turn(rules.turn, rules.side, observer);
    auto& contacts = memory.known_enemies;
    contacts.erase(std::remove_if(contacts.begin(), contacts.end(),
                                  [&](const Contact& contact) {
                                    return current_own_turn - contact.last_seen_own_turn > 1;
                                  }),
                   contacts.end());
    for (int unit_id = 0; unit_id < int(rules.units.size()); ++unit_id) {
      const auto& unit = rules.units[unit_id];
      if (unit.hp <= 0 || unit.side == observer || !rules.known(observer, unit.tile))
        continue;
      auto found =
          std::find_if(contacts.begin(), contacts.end(),
                       [unit_id](const Contact& contact) { return contact.id == unit_id; });
      if (found == contacts.end())
        contacts.push_back({unit_id, current_own_turn});
      else
        found->last_seen_own_turn = current_own_turn;
    }
    std::sort(contacts.begin(), contacts.end(),
              [](const Contact& left, const Contact& right) { return left.id < right.id; });
  }
};
class Adapter final : public RulesAdapter {
  std::shared_ptr<const Geometry> geo;
  std::shared_ptr<const std::vector<CaiTile>> map_tiles;
  int observer;
  const OwnedBranch& branch(const Branch& base) const {
    auto* owned = dynamic_cast<const OwnedBranch*>(&base);
    require(owned && owned->rules.g == geo && owned->observer == observer, "foreign branch");
    return *owned;
  }
  OwnedBranch& branch(Branch& value) const {
    auto* owned = dynamic_cast<OwnedBranch*>(&value);
    require(owned && owned->rules.g == geo && owned->observer == observer, "foreign branch");
    return *owned;
  }
  void tile(int index) const {
    require(index >= 0 && index < int(geo->tiles.size()), "invalid tile");
  }
  void unit(const OwnedBranch& position, int index) const {
    require(index >= 0 && index < int(position.rules.units.size()), "invalid unit");
  }

 public:
  Adapter(const CaiBattleState& state, int side, MapCache* cache) : observer(side) {
    require(bit(side), "invalid observer");
    validate_header(state);
    if (cache && cache->geometry && cache->tiles) {
      const auto& geometry = *cache->geometry;
      if (state.width == geometry.w && state.height == geometry.h &&
          state.row_parity == geometry.parity && state.wrap_x == geometry.wrap &&
          std::equal(cache->tiles->begin(), cache->tiles->end(), state.tiles,
                     [](const CaiTile& a, const CaiTile& b) {
                       return a.terrain == b.terrain && a.elevation == b.elevation &&
                              a.feature == b.feature && a.rivers == b.rivers;
                     })) {
        geo = cache->geometry;
        map_tiles = cache->tiles;
        return;
      }
    }
    CAI_PROFILE_SCOPE(GeometryBuild);
    std::vector<Tile> map_input;
    map_input.reserve(state.tile_count);
    for (int tile_index = 0; tile_index < state.tile_count; ++tile_index) {
      const auto& map_tile = state.tiles[tile_index];
      map_input.push_back(
          {map_tile.terrain, map_tile.elevation, map_tile.feature, map_tile.rivers});
    }
    geo = std::make_shared<Geometry>(state.width, state.height, state.row_parity, state.wrap_x,
                                     map_input.data());
    validate_model_geometry(*geo);
    map_tiles =
        std::make_shared<const std::vector<CaiTile>>(state.tiles, state.tiles + state.tile_count);
    if (cache) {
      cache->geometry = geo;
      cache->tiles = map_tiles;
    }
  }
  std::unique_ptr<Branch> root(const CaiBattleState& state,
                               const ContactMemory& contacts) const override {
    CAI_PROFILE_SCOPE(RootImport);
    validate_header(state);
    require(state.width == geo->w && state.height == geo->h && state.row_parity == geo->parity &&
                state.wrap_x == geo->wrap,
            "map mismatch");
    for (int index = 0; index < state.tile_count; ++index) {
      const auto& input_tile = state.tiles[index];
      const auto& map_tile = geo->tiles[index];
      require(input_tile.terrain == map_tile.terrain &&
                  input_tile.elevation == map_tile.elevation &&
                  input_tile.feature == map_tile.feature && input_tile.rivers == map_tile.rivers,
              "map mismatch");
    }
    auto position = std::make_unique<OwnedBranch>(geo, map_tiles, observer);
    for (int index = 0; index < state.unit_count; ++index) {
      const auto& unit_state = state.units[index];
      require(unit_state.id == index, "invalid unit id");
      position->rules.units.push_back({unit_state.type, unit_state.side, unit_state.tile,
                                       unit_state.hp, unit_state.moves, unit_state.attacked,
                                       unit_state.acted, unit_state.fortified, unit_state.waiting,
                                       0, unit_state.damage_taken_last_turn});
    }
    require(combat_rules::valid_units(*geo, position->rules.units.data(), state.unit_count),
            "invalid units");
    position->rules.side = state.active_side;
    position->rules.turn = state.turn;
    position->rules.fog = state.fog;
    position->rules.rebuild();
    // A mid-turn root also owns sight left by departed units. The request's
    // public mask is authoritative history, and must include current sources.
    if (state.visibility_history_count) {
      for (int side = 0; side < 2; ++side)
        require(position->rules.restore_visibility(side,
                    state.visibility_history + side * state.tile_count, state.tile_count),
                "invalid visibility history");
    } else if (state.fog) {
      // Standalone fixtures may supply only the observer's public mask.
      require(position->rules.restore_visibility(observer, state.visibility, state.visibility_count),
              "visibility mismatch");
    }
    for (int index = 0; index < state.tile_count; ++index)
      require(state.visibility[index] == int(position->rules.known(observer, index)),
              "visibility mismatch");
    require(contacts.turn >= 0 && contacts.turn <= state.turn, "invalid contact turn");
    require(contacts.side == observer ||
                (contacts.side == -1 && contacts.turn == 0 && contacts.known_enemies.empty()),
            "invalid contact side");
    int previous_side = state.active_side ^ ((state.turn - contacts.turn) & 1);
    int observed_turn = own_turn(contacts.turn, previous_side, observer);
    std::vector<bool> seen(state.unit_count, false);
    for (const auto& contact : contacts.known_enemies) {
      int contact_id = contact.id;
      require(contact_id >= 0 && contact_id < state.unit_count && !seen[contact_id] &&
                  position->rules.units[contact_id].side != observer,
              "invalid contact");
      require(contact.last_seen_own_turn >= 0 && contact.last_seen_own_turn <= observed_turn,
              "invalid last seen turn");
      seen[contact_id] = true;
    }
    position->memory = contacts;
    position->observe();
    return position;
  }
  std::vector<ReachableTile> reachable(const Branch& base, int32_t id) const override {
    const auto& position = branch(base);
    unit(position, id);
    const auto& moving_unit = position.rules.units[id];
    if (moving_unit.hp <= 0 || moving_unit.moves <= 0 || moving_unit.waiting ||
        moving_unit.side != position.rules.side)
      return {};
    std::vector<int> remaining, previous;
    position.rules.paths(id, remaining, previous);
    std::vector<ReachableTile> destinations;
    for (int target_tile = 0; target_tile < int(remaining.size()); ++target_tile)
      if (remaining[target_tile] >= 0 &&
          (target_tile == moving_unit.tile ||
           position.rules.known_unit(target_tile, moving_unit.side) < 0)) {
        destinations.push_back(
            {target_tile,
             destination_moves(position.rules, moving_unit, target_tile, remaining[target_tile])});
      }
    return destinations;
  }
  std::vector<Command> actions(const Branch& base) const override {
    std::vector<Command> commands;
    for (const auto& action : branch(base).rules.actions())
      commands.push_back({action.kind, action.unit, action.target});
    return commands;
  }
  bool can_execute(const Branch& base, Command command) const override {
    return branch(base).rules.validate({command.kind, command.unit, command.target}) ==
           combat_rules::OK;
  }
  std::vector<ReachableTile> move_intents(const Branch& base, int32_t id) const override {
    const auto& position = branch(base);
    unit(position, id);
    const auto& moving_unit = position.rules.units[id];
    if (moving_unit.hp <= 0 || moving_unit.moves <= 0 || moving_unit.waiting ||
        moving_unit.side != position.rules.side)
      return {};
    auto remaining = movement_remainders(
        *geo, moving_unit.tile, moving_unit.moves,
        [&](int from) {
          int occupant = position.rules.known_unit(from, moving_unit.side);
          return from == moving_unit.tile || occupant < 0 ||
                 position.rules.units[occupant].side == moving_unit.side;
        },
        [&](int from, int to, int moves_left) {
          return position.rules.movement(moving_unit, from, to, moves_left, moving_unit.side);
        });
    std::vector<ReachableTile> destinations;
    for (int target_tile = 0; target_tile < int(remaining.size()); ++target_tile)
      if (remaining[target_tile] >= 0) {
        destinations.push_back(
            {target_tile,
             destination_moves(position.rules, moving_unit, target_tile, remaining[target_tile])});
      }
    return destinations;
  }
  std::vector<int32_t> threatened_tiles(const Branch& base, int32_t id) const override {
    const auto& position = branch(base);
    unit(position, id);
    const auto& enemy_unit = position.rules.units[id];
    if (enemy_unit.hp <= 0)
      return {};
    // CvDangerPlots::UpdateDangerSingleUnit, CP Only CORE_TWO_PASS_DANGER=0.
    // No branch copying, assumed enemy movement, occupancy or ZOC queries.
    auto remaining = movement_remainders(
        *geo, enemy_unit.tile, combat_rules::moves[enemy_unit.type], [](int) { return true; },
        [&](int from, int to, int moves_left) {
          bool consumes_all_moves =
              position.rules.river(from, to) ||
              (is_type(enemy_unit, UnitType::ChariotArcher) && geo->rough(to));
          return consumes_all_moves ? moves_left : std::min(moves_left, geo->cost(to));
        });
    std::vector<bool> threatened(remaining.size(), false);
    for (int tile_index = 0; tile_index < int(remaining.size()); ++tile_index) {
      if (is_ranged(enemy_unit)) {
        if (remaining[tile_index] > 0)
          for (int target : geo->targets[tile_index])
            threatened[target] = true;
      } else if (remaining[tile_index] >= 0)
        threatened[tile_index] = true;
    }
    std::vector<int32_t> threats;
    for (int tile_index = 0; tile_index < int(threatened.size()); ++tile_index)
      if (threatened[tile_index])
        threats.push_back(tile_index);
    return threats;
  }
  Damage forecast(const Branch& base, int32_t attacker, int32_t from, int32_t defender, int32_t to,
                  int32_t extra_damage, bool ignore_adjacency, bool unknown_origin) const override {
    const auto& position = branch(base);
    unit(position, attacker);
    unit(position, defender);
    tile(from);
    tile(to);
    const auto& attacker_unit = position.rules.units[attacker];
    const auto& defender_unit = position.rules.units[defender];
    if (attacker_unit.hp <= 0 || defender_unit.hp <= 0 ||
        attacker_unit.side == defender_unit.side || extra_damage < 0 || !geo->passable(to))
      return {0, 0, combat_rules::BAD_TARGET, 0, 0};
    bool ranged_attack = is_ranged(attacker_unit);
    // Fix the original CP argument mix-up (CvTacticalAI.cpp:6228/6230):
    // iExtraDefenderDamage belongs to the defender in both attack modes.
    // Contract: docs/combat-ai-contract.md#extra-defender-damage.
    // Same five-type strength service/constants as State::strengths, with
    // explicit assumed coordinates and CP's preview damage convention. Keep
    // this query inside the AI: it does not change the kernel rules passport.
    int attacker_hp = attacker_unit.hp;
    int defender_hp = defender_unit.hp - extra_damage;
    int attacker_modifier =
        -((kMaxHitPoints - attacker_hp) * WOUNDED_DAMAGE_MULTIPLIER / kPercent) +
        position.rules.class_bonus(attacker_unit, defender_unit);
    int defender_modifier =
        -((kMaxHitPoints - defender_hp) * WOUNDED_DAMAGE_MULTIPLIER / kPercent) +
        position.rules.class_bonus(defender_unit, attacker_unit);
    bool ignores_defense_bonus = is_type(defender_unit, UnitType::ChariotArcher) ||
                                 is_type(defender_unit, UnitType::Horseman);
    int terrain_defense = geo->defense(to);
    defender_modifier += ignores_defense_bonus ? std::min(terrain_defense, 0) : terrain_defense;
    if (defender_unit.fortified && !ignores_defense_bonus)
      defender_modifier += FORTIFY_MODIFIER_PER_TURN;
    if (!ranged_attack) {
      if (!unknown_origin && position.rules.river(from, to))
        attacker_modifier += RIVER_ATTACK_MODIFIER;
      int attacker_support =
          ignore_adjacency
              ? 0
              : position.rules.supports(to, attacker_unit.side, attacker, attacker_unit.side, true);
      int defender_support = ignore_adjacency || unknown_origin
                                 ? 0
                                 : position.rules.supports(from, defender_unit.side, defender,
                                                           attacker_unit.side, true);
      attacker_modifier +=
          std::max(0, attacker_support - defender_support) * BONUS_PER_ADJACENT_FRIEND;
      defender_modifier +=
          std::max(0, defender_support - attacker_support) * BONUS_PER_ADJACENT_FRIEND;
    }
    int attacker_base = ranged_attack ? combat_rules::ranged[attacker_unit.type]
                                      : combat_rules::strength[attacker_unit.type];
    int defender_base = ranged_attack && is_ranged(defender_unit)
                            ? combat_rules::ranged[defender_unit.type]
                            : combat_rules::strength[defender_unit.type];
    int attacker_strength =
        std::max(1, attacker_base * std::max(kMinStrengthPercent, kPercent + attacker_modifier));
    int defender_strength =
        std::max(1, defender_base * std::max(kMinStrengthPercent, kPercent + defender_modifier));
    // CP's scoring preview returns uncapped deterministic damage. Real
    // transitions separately apply overkill/mutual-death correction.
    return {combat_rules::damage(attacker_strength, defender_strength, CAI_FIXED_ROLL),
            ranged_attack
                ? 0
                : combat_rules::damage(defender_strength, attacker_strength, CAI_FIXED_ROLL),
            0, attacker_strength, defender_strength};
  }
  int32_t attack_moves(const Branch& base, Command command) const override {
    const auto& position = branch(base);
    unit(position, command.unit);
    tile(command.target);
    const auto& attacker = position.rules.units[command.unit];
    require(attacker.hp > 0 && command.kind == combat_rules::MELEE,
            "attack_moves requires a living melee attacker");
    if (!is_type(attacker, UnitType::Horseman))
      return 0;
    return std::max(
        0, attacker.moves - position.rules.movement(attacker, attacker.tile, command.target,
                                                    attacker.moves, attacker.side));
  }
  Damage preview(const Branch& base, Command command, bool next_turn = false) const override {
    const auto& current = branch(base).rules;
    auto evaluate = [&](const State& position) -> Damage {
      Action action{command.kind, command.unit, command.target};
      int code = position.validate(action);
      if (!code && action.kind != combat_rules::MELEE && action.kind != combat_rules::RANGED)
        code = combat_rules::BAD_ACTION;
      if (code)
        return {0, 0, code, 0, 0};
      Result result = position.combat(action.unit, position.occupied[action.target],
                                      action.kind == combat_rules::RANGED, false);
      return {result.damage, result.retaliation, result.code, result.attacker_strength,
              result.defender_strength};
    };
    if (!next_turn)
      return evaluate(current);
    if (command.unit < 0 || command.unit >= int(current.units.size()))
      return {0, 0, combat_rules::BAD_UNIT, 0, 0};
    State projected = current;
    auto& attacker = projected.units[command.unit];
    attacker.moves = combat_rules::moves[attacker.type];
    attacker.attacked = 0;
    attacker.waiting = 0;
    return evaluate(projected);
  }
  Transition apply(Branch& base, Command command) const override {
    auto& position = branch(base);
    std::vector<bool> previous_visibility(geo->tiles.size());
    for (int tile_index = 0; tile_index < int(previous_visibility.size()); ++tile_index)
      previous_visibility[tile_index] = position.rules.known(observer, tile_index);
    combat_rules::StepEvents events;
    Result result = position.rules.step({command.kind, command.unit, command.target}, {}, &events);
    Transition transition{
        result.code == combat_rules::OK, events.contact_interrupted, {}, export_result(result)};
    if (transition.accepted) {
      for (int tile_index = 0; tile_index < int(previous_visibility.size()); ++tile_index)
        if (!previous_visibility[tile_index] && position.rules.known(observer, tile_index))
          transition.newly_visible.push_back(tile_index);
      position.last_accepted_visibility = transition.newly_visible;
      position.observe();
    }
    // A rejected command changes neither the branch nor the visibility from
    // its last accepted transition. Callers must check transition.accepted first.
    return transition;
  }
  int32_t distance(int32_t from, int32_t to) const override {
    tile(from);
    tile(to);
    return geo->distance(from, to);
  }
  int32_t defense(int32_t tile_index) const override {
    tile(tile_index);
    return geo->defense(tile_index);
  }
  std::vector<int32_t> neighbors(int32_t tile_index) const override {
    tile(tile_index);
    std::vector<int32_t> adjacent_tiles;
    for (int neighbor : geo->neighbors[tile_index])
      if (neighbor >= 0)
        adjacent_tiles.push_back(neighbor);
    return adjacent_tiles;
  }
  bool can_see(const Branch& base, int32_t from, int32_t to) const override {
    branch(base);
    tile(from);
    tile(to);
    const auto& visible_tiles = geo->sight[from];
    return std::binary_search(visible_tiles.begin(), visible_tiles.end(), to);
  }
  bool can_shoot(const Branch& base, int32_t id, int32_t to) const override {
    const auto& position = branch(base);
    unit(position, id);
    tile(to);
    const auto& attacker = position.rules.units[id];
    const auto& targets = geo->targets[attacker.tile];
    return attacker.hp > 0 && is_ranged(attacker) &&
           std::binary_search(targets.begin(), targets.end(), to);
  }
  bool can_target(int32_t from, int32_t to) const override {
    tile(from);
    tile(to);
    const auto& targets = geo->targets[from];
    return std::binary_search(targets.begin(), targets.end(), to);
  }
  std::vector<CaiUnit> prospective_units(const Branch& base,
                                         const std::vector<int32_t>& newly_visible) const override {
    const auto& position = branch(base);
    std::vector<CaiUnit> revealed_units;
    std::vector<bool> seen(geo->tiles.size(), false);
    for (int tile_index : newly_visible) {
      tile(tile_index);
      require(std::find(position.last_accepted_visibility.begin(),
                        position.last_accepted_visibility.end(),
                        tile_index) != position.last_accepted_visibility.end(),
              "tile not newly visible");
      if (seen[tile_index])
        continue;
      seen[tile_index] = true;
      int id = position.rules.occupied[tile_index];
      if (id >= 0)
        revealed_units.push_back(export_unit(id, position.rules.units[id]));
    }
    return revealed_units;
  }
  bool known_enemy(const Branch& base, int32_t id, CaiUnit& enemy_out) const override {
    const auto& position = branch(base);
    unit(position, id);
    const auto& enemy = position.rules.units[id];
    if (enemy.side == observer || enemy.hp <= 0 ||
        std::none_of(position.memory.known_enemies.begin(), position.memory.known_enemies.end(),
                     [id](const Contact& contact) { return contact.id == id; }))
      return false;
    enemy_out = export_unit(id, enemy);
    return true;
  }
  int32_t incoming(const Branch& base, int32_t defender, int32_t target, DecisionTrace* trace) const {
    const auto& position = branch(base);
    unit(position, defender);
    tile(target);
    require(
        position.rules.units[defender].hp > 0 && position.rules.units[defender].side == observer,
        "invalid defender");
    require(geo->passable(target), "impassable danger tile");
    auto key = std::make_pair(defender, target);
    auto cached = position.danger_cache.find(key);
    if (cached != position.danger_cache.end()) {
      if (trace) trace->event("physical_danger_cached", {{"unit",defender},{"tile",target},{"total",cached->second}});
      return cached->second;
    }

    // Only admitted contacts may affect prediction. Never-seen units are
    // removed even from occupancy, support and visibility calculations.
    State projected = position.rules;
    std::vector<bool> admitted(projected.units.size(), false);
    for (const auto& contact : position.memory.known_enemies)
      admitted[contact.id] = true;
    std::vector<int> admitted_enemies, visible_enemies;
    for (int unit_id = 0; unit_id < int(projected.units.size()); ++unit_id) {
      auto& enemy = projected.units[unit_id];
      if (enemy.side == observer || enemy.hp <= 0)
        continue;
      if (!admitted[unit_id]) {
        enemy.hp = 0;
        continue;
      }
      admitted_enemies.push_back(unit_id);
      if (position.rules.known(observer, enemy.tile))
        visible_enemies.push_back(unit_id);
    }
    projected.rebuild();
    require(projected.occupied[target] < 0 || projected.occupied[target] == defender,
            "occupied danger tile");

    // CP IsKnownVisibleToTeam: only sight sources currently visible to us.
    // Do this before any assumed relocation, including the defender's.
    bool enemy_can_see_target = !position.rules.fog;
    for (int id : visible_enemies) {
      const auto& sight = geo->sight[projected.units[id].tile];
      if (std::binary_search(sight.begin(), sight.end(), target))
        enemy_can_see_target = true;
    }
    int total = 0;
    // AddFogDanger(origin, enemy, 2, false): each hidden passable origin
    // candidate contributes to a disk of radius two, not the whole map.
    if (position.rules.fog)
      for (int id : visible_enemies)
        for (int hidden_origin = 0; hidden_origin < int(geo->tiles.size()); ++hidden_origin)
          if (geo->passable(hidden_origin) && !position.rules.known(observer, hidden_origin) &&
              geo->distance(projected.units[id].tile, hidden_origin) <= kFogRadius &&
              geo->distance(hidden_origin, target) <= kFogRadius)
            total += kFogDefaultDanger;

    if (trace) trace->event("physical_danger_fog", {{"unit",defender},{"tile",target},{"fog",total}});
    if (projected.units[defender].tile != target) {
      projected.relocate(projected.units[defender], target);
      projected.units[defender].fortified = 0;
    }
    projected.side = 1 - observer;
    // Reachability assumes complete knowledge of the admitted physical
    // position. The probability discount above uses a separate knowledge map.
    projected.fog = 0;
    for (int id : admitted_enemies) {
      const auto& enemy_unit = projected.units[id];
      const bool ranged_attack = is_ranged(enemy_unit);
      // Any edge leaving enough movement to attack costs at least one point.
      // Full move budgets and terrain/range assumptions are checked above.
      // This bound only skips impossible attackers before copying/pathfinding.
      if (geo->distance(enemy_unit.tile, target) >
          combat_rules::moves[enemy_unit.type] / kMoveDenominator + attack_range(enemy_unit))
        continue;
      State response = projected;
      auto& attacker = response.units[id];
      attacker.moves = combat_rules::moves[attacker.type];
      attacker.attacked = 0;
      attacker.waiting = 0;
      std::vector<int> remaining, previous;
      response.paths(id, remaining, previous);
      int max_damage = 0;
      int starting_tile = attacker.tile;
      for (int from = 0; from < int(remaining.size()); ++from) {
        if (remaining[from] <= 0 || (response.occupied[from] >= 0 && from != starting_tile))
          continue;
        if (ranged_attack) {
          const auto& targets = geo->targets[from];
          if (!std::binary_search(targets.begin(), targets.end(), target))
            continue;
        } else if (geo->distance(from, target) != kMeleeAttackRange)
          continue;
        // A private copy keeps the per-position support/visibility exact and
        // cannot mutate sibling candidate attacks or the real branch.
        State attack = response;
        auto& attacker_at_origin = attack.units[id];
        if (from != starting_tile)
          attack.relocate(attacker_at_origin, from);
        attacker_at_origin.moves = remaining[from];
        if (from != starting_tile)
          attacker_at_origin.fortified = 0;
        if (attack.validate({ranged_attack ? combat_rules::RANGED : combat_rules::MELEE, id,
                             target}) != combat_rules::OK)
          continue;
        auto attack_result = attack.combat(id, defender, ranged_attack, false);
        int contribution = attack_result.damage;
        if (!enemy_can_see_target)
          contribution = contribution * kUnseenTargetDamagePercent / kPercent;
        if (attack_result.retaliation >= attacker_at_origin.hp)
          contribution /= kSuicideDamageDivisor;
        if (trace) trace->event("physical_danger_attack", {{"unit",defender},{"tile",target},{"enemy",id},
            {"from",from},{"damage",attack_result.damage},{"retaliation",attack_result.retaliation},
            {"enemy_hp",attacker_at_origin.hp},{"visible",enemy_can_see_target},{"contribution",contribution}});
        max_damage = std::max(max_damage, contribution);
      }
      if (trace) trace->event("physical_danger_attacker", {{"unit",defender},{"tile",target},{"enemy",id},{"contribution",max_damage}});
      total += max_damage;
    }
    if (trace) trace->event("physical_danger_total", {{"unit",defender},{"tile",target},{"total",total}});
    position.danger_cache.emplace(key, total);
    return total;
  }
};
class Danger final : public DangerModel {
  // Non-owning by contract: make_danger_model requires the adapter to outlive us.
  const Adapter& adapter;
  DecisionTrace* trace;

 public:
  explicit Danger(const Adapter& value, DecisionTrace* sink) : adapter(value), trace(sink) {}
  int32_t incoming(const Branch& position, int32_t defender, int32_t tile) override {
    return adapter.incoming(position, defender, tile, trace);
  }
};
}  // namespace
std::unique_ptr<RulesAdapter> make_rules_adapter(const CaiBattleState& state, int32_t observer,
                                                MapCache* cache) {
  return std::make_unique<Adapter>(state, observer, cache);
}
std::unique_ptr<DangerModel> make_danger_model(const RulesAdapter& adapter, DecisionTrace* trace) {
  auto* concrete = dynamic_cast<const Adapter*>(&adapter);
  require(concrete != nullptr, "unsupported danger adapter");
  return std::make_unique<Danger>(*concrete, trace);
}
}  // namespace combat_ai
