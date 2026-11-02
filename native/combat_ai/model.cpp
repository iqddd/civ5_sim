#include "model.h"
#include "../combat_state.h"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

namespace combat_ai {
int32_t own_turn(int32_t turn, int32_t active_side, int32_t observer) {
  if (turn < 0 || (active_side != 0 && active_side != 1) ||
      (observer != 0 && observer != 1)) throw std::invalid_argument("invalid turn/side");
  return int32_t((int64_t(turn) + (active_side == observer)) / 2);
}
namespace {
using combat_rules::State;
bool bit(int n) { return n == 0 || n == 1; }
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
CaiUnit export_unit(int id, const Unit& u) {
  return {id, u.type, u.side, u.tile, u.hp, u.moves, u.attacked, u.acted,
          u.fortified, u.waiting, u.damage_taken_last_turn};
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
  mutable std::map<std::pair<int32_t,int32_t>, int32_t> danger_cache;
  OwnedBranch(std::shared_ptr<const Geometry> geometry,
              std::shared_ptr<const std::vector<CaiTile>> map_tiles, int side)
      : rules(std::move(geometry)), observer(side), tiles(std::move(map_tiles)) {}
  std::unique_ptr<Branch> clone() const override {
    auto copy = std::make_unique<OwnedBranch>(*this);
    copy->danger_cache.clear();
    return copy;
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
    danger_cache.clear();
    memory.side = observer;
    int now = own_turn(rules.turn, rules.side, observer);
    auto& contacts = memory.known_enemies;
    contacts.erase(std::remove_if(contacts.begin(), contacts.end(), [&](const Contact& c) {
      return now - c.last_seen_own_turn > 1;
    }), contacts.end());
    for (int i = 0; i < int(rules.units.size()); ++i) {
      const auto& u = rules.units[i];
      if (u.hp <= 0 || u.side == observer || !rules.known(observer, u.tile)) continue;
      auto found = std::find_if(contacts.begin(), contacts.end(),
                                [i](const Contact& c) { return c.id == i; });
      if (found == contacts.end()) contacts.push_back({i, now});
      else found->last_seen_own_turn = now;
    }
    std::sort(contacts.begin(), contacts.end(), [](const Contact& a, const Contact& b) { return a.id < b.id; });
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
      b->rules.units.push_back({u.type,u.side,u.tile,u.hp,u.moves,u.attacked,u.acted,u.fortified,u.waiting,
                                0,u.damage_taken_last_turn});
    }
    require(combat_rules::valid_units(*geo, b->rules.units.data(), s.unit_count), "invalid units");
    b->rules.side = s.active_side; b->rules.turn = s.turn; b->rules.fog = s.fog;
    b->rules.rebuild();
    for (int i = 0; i < s.tile_count; ++i)
      require(s.visibility[i] == int(b->rules.known(observer, i)), "visibility mismatch");
    require(contacts.turn >= 0 && contacts.turn <= s.turn, "invalid contact turn");
    require(contacts.side == observer || (contacts.side == -1 && contacts.turn == 0 &&
            contacts.known_enemies.empty()), "invalid contact side");
    int previous_side = s.active_side ^ ((s.turn - contacts.turn) & 1);
    int observed_turn = own_turn(contacts.turn, previous_side, observer);
    std::vector<bool> seen(s.unit_count, false);
    for (const auto& contact : contacts.known_enemies) {
      int id = contact.id;
      require(id >= 0 && id < s.unit_count && !seen[id] && b->rules.units[id].side != observer,
              "invalid contact");
      require(contact.last_seen_own_turn >= 0 && contact.last_seen_own_turn <= observed_turn,
              "invalid last seen turn");
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
  bool can_execute(const Branch& base, Command c) const override {
    return branch(base).rules.validate({c.kind,c.unit,c.target})==combat_rules::OK;
  }
  std::vector<ReachableTile> move_intents(const Branch& base, int32_t id) const override {
    const auto& b = branch(base); unit(b,id);
    const auto& u = b.rules.units[id];
    if (u.hp <= 0 || u.moves <= 0 || u.waiting || u.side != b.rules.side) return {};
    std::vector<int> remaining(geo->tiles.size(),-1);
    std::priority_queue<std::pair<int,int>> queue;
    remaining[u.tile] = u.moves; queue.push({u.moves,u.tile});
    while (!queue.empty()) {
      auto [left,from] = queue.top(); queue.pop();
      if (left != remaining[from] || left <= 0) continue;
      int occupant = b.rules.known_unit(from,u.side);
      if (from != u.tile && occupant >= 0 && b.rules.units[occupant].side != u.side) continue;
      for (int to : geo->neighbors[from]) if (geo->passable(to)) {
        int after = left - b.rules.movement(u,from,to,left,u.side);
        if (after > remaining[to]) {
          remaining[to] = after; queue.push({after,to});
        }
      }
    }
    std::vector<ReachableTile> out;
    for (int t = 0; t < int(remaining.size()); ++t) if (remaining[t] >= 0) {
      int left = remaining[t];
      if (t != u.tile && geo->distance(u.tile,t) == 1)
        left = u.moves - b.rules.movement(u,u.tile,t,u.moves,u.side);
      out.push_back({t,left});
    }
    return out;
  }
  std::vector<int32_t> threatened_tiles(const Branch& base, int32_t id) const override {
    const auto& b = branch(base); unit(b,id);
    const auto& u = b.rules.units[id];
    if (u.hp <= 0) return {};
    // CvDangerPlots::UpdateDangerSingleUnit, CP Only CORE_TWO_PASS_DANGER=0.
    // No branch copying, assumed enemy movement, occupancy or ZOC queries.
    std::vector<int> left(geo->tiles.size(),-1);
    std::priority_queue<std::pair<int,int>> queue;
    left[u.tile] = combat_rules::moves[u.type]; queue.push({left[u.tile],u.tile});
    while (!queue.empty()) {
      auto [remaining,from] = queue.top(); queue.pop();
      if (remaining != left[from] || remaining <= 0) continue;
      for (int to : geo->neighbors[from]) if (geo->passable(to)) {
        int cost = b.rules.river(from,to) || (u.type == 2 && geo->rough(to)) ?
                   remaining : std::min(remaining,geo->cost(to));
        int after = remaining-cost;
        if (after > left[to]) { left[to] = after; queue.push({after,to}); }
      }
    }
    std::vector<bool> attacked(left.size(),false);
    for (int t = 0; t < int(left.size()); ++t) {
      if (combat_rules::ranged[u.type]) {
        if (left[t] > 0) for (int target : geo->targets[t]) attacked[target] = true;
      } else if (left[t] >= 0) attacked[t] = true;
    }
    std::vector<int32_t> out;
    for (int t = 0; t < int(attacked.size()); ++t) if (attacked[t]) out.push_back(t);
    return out;
  }
  Damage forecast(const Branch& base, int32_t attacker, int32_t from,
      int32_t defender, int32_t to, int32_t extra_damage, bool ignore_adjacency,
      bool unknown_origin) const override {
    const auto& b = branch(base); unit(b,attacker); unit(b,defender); tile(from); tile(to);
    const auto& a = b.rules.units[attacker]; const auto& d = b.rules.units[defender];
    if (a.hp <= 0 || d.hp <= 0 || a.side == d.side || extra_damage < 0 || !geo->passable(to))
      return {0,0,combat_rules::BAD_TARGET,0,0};
    bool shot = combat_rules::ranged[a.type] != 0;
    // Preserve the pinned GetSimulatedDamageFromAttackOnUnit call convention:
    // for melee, iExtraDefenderDamage goes to GetMaxAttackStrength's extra
    // attacker damage argument; ranged applies it to the defender instead.
    // Same five-type strength service/constants as State::strengths, with
    // explicit assumed coordinates and CP's preview damage convention. Keep
    // this query inside the AI: it does not change the kernel rules passport.
    int ahp=a.hp-(shot?0:extra_damage), dhp=d.hp-(shot?extra_damage:0);
    int am=-((100-ahp)*WOUNDED_DAMAGE_MULTIPLIER/100)+b.rules.class_bonus(a,d);
    int dm=-((100-dhp)*WOUNDED_DAMAGE_MULTIPLIER/100)+b.rules.class_bonus(d,a);
    bool no_terrain=d.type==2 || d.type==4;
    int terrain=geo->defense(to);
    dm+=no_terrain?std::min(terrain,0):terrain;
    if (d.fortified && !no_terrain) dm+=FORTIFY_MODIFIER_PER_TURN;
    if (!shot) {
      if (!unknown_origin && b.rules.river(from,to)) am+=RIVER_ATTACK_MODIFIER;
      int ours=ignore_adjacency?0:b.rules.supports(to,a.side,attacker,a.side,true);
      int theirs=ignore_adjacency || unknown_origin?0:b.rules.supports(from,d.side,defender,a.side,true);
      am+=std::max(0,ours-theirs)*BONUS_PER_ADJACENT_FRIEND;
      dm+=std::max(0,theirs-ours)*BONUS_PER_ADJACENT_FRIEND;
    }
    int as=std::max(1,(shot?combat_rules::ranged[a.type]:combat_rules::strength[a.type])*std::max(10,100+am));
    int ds=std::max(1,(shot && combat_rules::ranged[d.type]?combat_rules::ranged[d.type]:combat_rules::strength[d.type])*std::max(10,100+dm));
    // CP's scoring preview returns uncapped deterministic damage. Real
    // transitions separately apply overkill/mutual-death correction.
    return {combat_rules::damage(as,ds,CAI_FIXED_ROLL),
            shot ? 0 : combat_rules::damage(ds,as,CAI_FIXED_ROLL),0,as,ds};
  }
  int32_t attack_moves(const Branch& base, Command c) const override {
    const auto& b = branch(base); unit(b,c.unit); tile(c.target);
    const auto& u = b.rules.units[c.unit];
    if (u.type != 4) return 0;
    return std::max(0,u.moves-b.rules.movement(u,u.tile,c.target,u.moves,u.side));
  }
  Damage preview(const Branch& base, Command c, bool next_turn = false) const override {
    const auto& current = branch(base).rules;
    auto evaluate = [&](const State& s) -> Damage {
      Action a{c.kind,c.unit,c.target};
      int code = s.validate(a);
      if (!code && a.kind != combat_rules::MELEE && a.kind != combat_rules::RANGED)
        code = combat_rules::BAD_ACTION;
      if (code) return {0,0,code,0,0};
      Result r = s.combat(a.unit, s.occupied[a.target], a.kind == combat_rules::RANGED, false);
      return {r.damage,r.retaliation,r.code,r.attacker_strength,r.defender_strength};
    };
    if (!next_turn) return evaluate(current);
    if (c.unit < 0 || c.unit >= int(current.units.size()))
      return {0,0,combat_rules::BAD_UNIT,0,0};
    State projected = current;
    auto& u = projected.units[c.unit];
    u.moves = combat_rules::moves[u.type];
    u.attacked = u.waiting = 0;
    return evaluate(projected);
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
  int32_t defense(int32_t n) const override { tile(n); return geo->defense(n); }
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
  bool can_target(int32_t from, int32_t to) const override {
    tile(from); tile(to);
    const auto& targets=geo->targets[from];
    return std::binary_search(targets.begin(),targets.end(),to);
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
    if (u.side == observer || u.hp <= 0 ||
        std::none_of(b.memory.known_enemies.begin(),b.memory.known_enemies.end(),
                     [id](const Contact& c) { return c.id == id; })) return false;
    out = export_unit(id,u); return true;
  }
  int32_t incoming(const Branch& base, int32_t defender, int32_t target) const {
    const auto& b = branch(base); unit(b,defender); tile(target);
    require(b.rules.units[defender].hp > 0 && b.rules.units[defender].side == observer,
            "invalid defender");
    require(geo->passable(target), "impassable danger tile");
    auto key = std::make_pair(defender,target);
    auto cached = b.danger_cache.find(key);
    if (cached != b.danger_cache.end()) return cached->second;

    // Only admitted contacts may affect prediction. Never-seen units are
    // removed even from occupancy, support and visibility calculations.
    State projected = b.rules;
    std::vector<bool> admitted(projected.units.size(), false);
    for (const auto& c : b.memory.known_enemies) admitted[c.id] = true;
    std::vector<int> enemies, visible_enemies;
    for (int i = 0; i < int(projected.units.size()); ++i) {
      auto& u = projected.units[i];
      if (u.side == observer || u.hp <= 0) continue;
      if (!admitted[i]) { u.hp = 0; continue; }
      enemies.push_back(i);
      if (b.rules.known(observer,u.tile)) visible_enemies.push_back(i);
    }
    projected.rebuild();
    require(projected.occupied[target] < 0 || projected.occupied[target] == defender,
            "occupied danger tile");

    // CP IsKnownVisibleToTeam: only sight sources currently visible to us.
    // Do this before any assumed relocation, including the defender's.
    bool known_visible = !b.rules.fog;
    for (int id : visible_enemies) {
      const auto& sight = geo->sight[projected.units[id].tile];
      if (std::binary_search(sight.begin(),sight.end(),target)) known_visible = true;
    }
    int total = 0;
    // AddFogDanger(origin, enemy, 2, false): each hidden passable origin
    // candidate contributes to a disk of radius two, not the whole map.
    if (b.rules.fog)
      for (int id : visible_enemies)
        for (int hidden = 0; hidden < int(geo->tiles.size()); ++hidden)
          if (geo->passable(hidden) && !b.rules.known(observer,hidden) &&
              geo->distance(projected.units[id].tile,hidden) <= 2 &&
              geo->distance(hidden,target) <= 2) ++total;

    if (projected.units[defender].tile != target) {
      projected.relocate(projected.units[defender], target);
      projected.units[defender].fortified = 0;
    }
    projected.side = 1 - observer;
    // Reachability assumes complete knowledge of the admitted physical
    // position. The probability discount above uses a separate knowledge map.
    projected.fog = 0;
    for (int id : enemies) {
      const auto& original = projected.units[id];
      const bool shot = combat_rules::ranged[original.type] != 0;
      // Every traversed edge costs at least 60; a positive remainder is
      // required to attack. This conservative bound only skips impossible
      // attackers, before allocating a response and running pathfinding.
      if (geo->distance(original.tile, target) >
          combat_rules::moves[original.type] / 60 + (shot ? 2 : 1)) continue;
      State response = projected;
      auto& attacker = response.units[id];
      attacker.moves = combat_rules::moves[attacker.type];
      attacker.attacked = attacker.waiting = 0;
      std::vector<int> remaining, previous;
      response.paths(id,remaining,previous);
      int best = 0;
      int origin = attacker.tile;
      for (int from = 0; from < int(remaining.size()); ++from) {
        if (remaining[from] <= 0 ||
            (response.occupied[from] >= 0 && from != origin)) continue;
        if (shot) {
          const auto& targets = geo->targets[from];
          if (!std::binary_search(targets.begin(),targets.end(),target)) continue;
        } else if (geo->distance(from,target) != 1) continue;
        // A private copy keeps the per-position support/visibility exact and
        // cannot mutate sibling candidate attacks or the real branch.
        State attack = response;
        auto& a = attack.units[id];
        if (from != origin) attack.relocate(a, from);
        a.moves = remaining[from];
        if (from != origin) a.fortified = 0;
        if (attack.validate({shot ? combat_rules::RANGED : combat_rules::MELEE,id,target}) != combat_rules::OK)
          continue;
        auto damage = attack.combat(id,defender,shot,false);
        int contribution = damage.damage;
        if (!known_visible) contribution = contribution * 80 / 100;
        if (damage.retaliation >= a.hp) contribution /= 2;
        best = std::max(best,contribution);
      }
      total += best;
    }
    b.danger_cache.emplace(key,total);
    return total;
  }
};
class Danger final : public DangerModel {
  const Adapter& adapter;
 public:
  explicit Danger(const Adapter& value) : adapter(value) {}
  int32_t incoming(const Branch& b, int32_t defender, int32_t tile) override {
    return adapter.incoming(b,defender,tile);
  }
};
} // namespace
std::unique_ptr<RulesAdapter> make_rules_adapter(const CaiBattleState& state, int32_t observer) {
  return std::make_unique<Adapter>(state, observer);
}
std::unique_ptr<DangerModel> make_danger_model(const RulesAdapter& adapter) {
  auto* concrete = dynamic_cast<const Adapter*>(&adapter);
  require(concrete != nullptr, "unsupported danger adapter");
  return std::make_unique<Danger>(*concrete);
}
} // namespace combat_ai
