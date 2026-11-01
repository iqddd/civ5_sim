#include "model.h"
#include "../combat_state.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace combat_ai {
namespace {
using combat_rules::State;
bool bit(int n) { return n == 0 || n == 1; }
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
CaiUnit export_unit(int id, const Unit& u) {
  return {id, u.type, u.side, u.tile, u.hp, u.moves, u.attacked, u.acted,
          u.fortified, u.waiting};
}
CommandResult export_result(const Result& r) {
  return {r.code, r.moved, r.damage, r.retaliation, r.attacker_strength,
          r.defender_strength};
}
void validate_header(const CaiBattleState& s) {
  require(s.width >= 1 && s.width <= 128 && s.height >= 1 && s.height <= 128 &&
          s.tile_count == s.width * s.height && s.tile_count <= 4096 && s.tiles &&
          bit(s.row_parity) && bit(s.wrap_x) && bit(s.fog) &&
          bit(s.active_side) && s.turn >= 0 && s.unit_count >= 0 &&
          s.unit_count <= 512 && (!s.unit_count || s.units) && s.visibility &&
          s.visibility_count == s.tile_count, "invalid battle snapshot");
  for (int i = 0; i < s.tile_count; ++i) {
    const auto& t = s.tiles[i];
    require(t.terrain >= 0 && t.terrain <= 6 && t.elevation >= 0 && t.elevation <= 3 &&
            t.feature >= 0 && t.feature <= 8 && t.rivers >= 0 && t.rivers <= 63 &&
            bit(s.visibility[i]), "invalid tile or visibility");
  }
}
class OwnedBranch final : public Branch {
 public:
  State rules;
  int observer;
  ContactMemory memory;
  std::vector<int32_t> last_visible;
  std::shared_ptr<const std::vector<CaiTile>> tiles;
  mutable std::vector<CaiUnit> units;
  mutable std::vector<int32_t> visibility;
  mutable CaiBattleState snapshot{};
  OwnedBranch(std::shared_ptr<const Geometry> geometry,
              std::shared_ptr<const std::vector<CaiTile>> map_tiles, int side)
      : rules(std::move(geometry)), observer(side), tiles(std::move(map_tiles)) {}
  std::unique_ptr<Branch> clone() const override {
    return std::make_unique<OwnedBranch>(*this);
  }
  const CaiBattleState& state() const override {
    units.clear();
    for (int i = 0; i < int(rules.units.size()); ++i)
      units.push_back(export_unit(i, rules.units[i]));
    visibility.resize(tiles->size());
    for (int i = 0; i < int(tiles->size()); ++i)
      visibility[i] = rules.known(observer, i) ? 1 : 0;
    snapshot = {rules.g->w, rules.g->h, rules.g->parity, rules.g->wrap,
                tiles->data(), int32_t(tiles->size()), units.data(), int32_t(units.size()),
                visibility.data(), int32_t(visibility.size()), rules.side, rules.turn, rules.fog};
    return snapshot;
  }
  const ContactMemory& contacts() const override { return memory; }
  void observe() {
    for (int i = 0; i < int(rules.units.size()); ++i) {
      const auto& u = rules.units[i];
      if (u.hp > 0 && u.side != observer && rules.known(observer, u.tile) &&
          std::find(memory.known_enemies.begin(), memory.known_enemies.end(), i) == memory.known_enemies.end())
        memory.known_enemies.push_back(i);
    }
  }
};
class Adapter final : public RulesAdapter {
  std::shared_ptr<const Geometry> geo;
  std::shared_ptr<const std::vector<CaiTile>> map_tiles;
  int observer;
  const OwnedBranch& branch(const Branch& b) const {
    auto* p = dynamic_cast<const OwnedBranch*>(&b);
    require(p && p->rules.g == geo && p->observer == observer, "foreign branch");
    return *p;
  }
  void tile(int n) const { require(n >= 0 && n < int(geo->tiles.size()), "invalid tile"); }
  void unit(const OwnedBranch& b, int n) const {
    require(n >= 0 && n < int(b.rules.units.size()), "invalid unit");
  }
 public:
  Adapter(const CaiBattleState& s, int side) : observer(side) {
    require(bit(side), "invalid observer");
    validate_header(s);
    std::vector<Tile> ts;
    ts.reserve(s.tile_count);
    for (int i = 0; i < s.tile_count; ++i) {
      const auto& t = s.tiles[i];
      ts.push_back({t.terrain, t.elevation, t.feature, t.rivers});
    }
    geo = std::make_shared<Geometry>(s.width, s.height, s.row_parity, s.wrap_x, ts.data());
    map_tiles = std::make_shared<const std::vector<CaiTile>>(s.tiles, s.tiles + s.tile_count);
  }
  std::unique_ptr<Branch> root(const CaiBattleState& s, const ContactMemory& contacts) const override {
    validate_header(s);
    require(s.width == geo->w && s.height == geo->h && s.row_parity == geo->parity &&
            s.wrap_x == geo->wrap, "map mismatch");
    for (int i = 0; i < s.tile_count; ++i) {
      const auto& a = s.tiles[i]; const auto& b = geo->tiles[i];
      require(a.terrain == b.terrain && a.elevation == b.elevation &&
              a.feature == b.feature && a.rivers == b.rivers, "map mismatch");
    }
    auto b = std::make_unique<OwnedBranch>(geo, map_tiles, observer);
    for (int i = 0; i < s.unit_count; ++i) {
      const auto& u = s.units[i];
      require(u.id == i, "invalid unit id");
      b->rules.units.push_back({u.type,u.side,u.tile,u.hp,u.moves,u.attacked,u.acted,u.fortified,u.waiting});
    }
    require(combat_rules::valid_units(*geo, b->rules.units.data(), s.unit_count), "invalid units");
    b->rules.side = s.active_side; b->rules.turn = s.turn; b->rules.fog = s.fog;
    b->rules.rebuild();
    for (int i = 0; i < s.tile_count; ++i)
      require(s.visibility[i] == int(b->rules.known(observer, i)), "visibility mismatch");
    require(contacts.turn >= 0 && contacts.turn <= s.turn, "invalid contact turn");
    std::vector<bool> seen(s.unit_count, false);
    for (int id : contacts.known_enemies) {
      require(id >= 0 && id < s.unit_count && !seen[id] && b->rules.units[id].side != observer,
              "invalid contact");
      seen[id] = true;
    }
    b->memory = contacts;
    b->observe();
    return b;
  }
  std::vector<ReachableTile> reachable(const Branch& base, int32_t id) const override {
    const auto& b = branch(base); unit(b, id);
    const auto& u = b.rules.units[id];
    if (u.hp <= 0 || u.moves <= 0 || u.waiting || u.side != b.rules.side) return {};
    std::vector<int> remaining, previous;
    b.rules.paths(id, remaining, previous);
    std::vector<ReachableTile> out;
    for (int i = 0; i < int(remaining.size()); ++i)
      if (remaining[i] >= 0 && (i == u.tile || b.rules.known_unit(i, u.side) < 0)) {
        int left = remaining[i];
        // The kernel executes adjacent moves directly, even if paths found a
        // cheaper detour to the same destination.
        if (i != u.tile && geo->distance(u.tile, i) == 1)
          left = u.moves - b.rules.movement(u, u.tile, i, u.moves, u.side);
        out.push_back({i, left});
      }
    return out;
  }
  std::vector<Command> actions(const Branch& base) const override {
    std::vector<Command> out;
    for (const auto& a : branch(base).rules.actions()) out.push_back({a.kind,a.unit,a.target});
    return out;
  }
  Damage preview(const Branch& base, Command c) const override {
    const auto& s = branch(base).rules;
    Action a{c.kind,c.unit,c.target};
    int code = s.validate(a);
    if (!code && a.kind != combat_rules::MELEE && a.kind != combat_rules::RANGED)
      code = combat_rules::BAD_ACTION;
    if (code) return {0,0,code,0,0};
    Result r = s.combat(a.unit, s.occupied[a.target], a.kind == combat_rules::RANGED, false);
    return {r.damage,r.retaliation,r.code,r.attacker_strength,r.defender_strength};
  }
  Transition apply(Branch& base, Command c) const override {
    auto& b = const_cast<OwnedBranch&>(branch(base));
    std::vector<bool> before(geo->tiles.size());
    for (int i = 0; i < int(before.size()); ++i) before[i] = b.rules.known(observer, i);
    combat_rules::StepEvents events;
    Result r = b.rules.step({c.kind,c.unit,c.target}, {}, &events);
    Transition t{r.code == combat_rules::OK, events.contact_interrupted, {}, export_result(r)};
    if (t.accepted) {
      for (int i = 0; i < int(before.size()); ++i)
        if (!before[i] && b.rules.known(observer, i)) t.newly_visible.push_back(i);
      b.last_visible = t.newly_visible;
      b.observe();
    }
    return t;
  }
  int32_t distance(int32_t a, int32_t b) const override { tile(a); tile(b); return geo->distance(a,b); }
  std::vector<int32_t> neighbors(int32_t n) const override {
    tile(n); std::vector<int32_t> out;
    for (int x : geo->neighbors[n]) if (x >= 0) out.push_back(x);
    return out;
  }
  bool can_see(const Branch& base, int32_t from, int32_t to) const override {
    branch(base); tile(from); tile(to);
    const auto& v = geo->sight[from]; return std::binary_search(v.begin(), v.end(), to);
  }
  bool can_shoot(const Branch& base, int32_t id, int32_t to) const override {
    const auto& b = branch(base); unit(b,id); tile(to);
    const auto& u = b.rules.units[id];
    const auto& v = geo->targets[u.tile];
    return u.hp > 0 && combat_rules::ranged[u.type] && std::binary_search(v.begin(),v.end(),to);
  }
  std::vector<CaiUnit> prospective_units(const Branch& base, const std::vector<int32_t>& tiles) const override {
    const auto& b = branch(base); std::vector<CaiUnit> out;
    std::vector<bool> seen(geo->tiles.size(), false);
    for (int n : tiles) {
      tile(n);
      require(std::find(b.last_visible.begin(),b.last_visible.end(),n) != b.last_visible.end(), "tile not newly visible");
      if (seen[n]) continue;
      seen[n] = true;
      int id = b.rules.occupied[n];
      if (id >= 0) out.push_back(export_unit(id,b.rules.units[id]));
    }
    return out;
  }
  bool known_enemy(const Branch& base, int32_t id, CaiUnit& out) const override {
    const auto& b = branch(base); unit(b,id);
    const auto& u = b.rules.units[id];
    if (u.side == observer || u.hp <= 0 || std::find(b.memory.known_enemies.begin(),b.memory.known_enemies.end(),id) == b.memory.known_enemies.end()) return false;
    out = export_unit(id,u); return true;
  }
};
} // namespace
std::unique_ptr<RulesAdapter> make_rules_adapter(const CaiBattleState& state, int32_t observer) {
  return std::make_unique<Adapter>(state, observer);
}
} // namespace combat_ai
