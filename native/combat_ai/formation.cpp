// Group deployment and exchange search for the CPv2 coordinator.
#include "../combat_state.h"
#include "planner.h"
#include <cmath>

namespace combat_ai {
namespace {
using namespace combat_rules;

// A reachable firing origin for each enemy and threatened destination.
// Hypothesize a fresh enemy turn, using only the visible roster.
using Threats = std::vector<std::vector<std::pair<int, int>>>;
Threats threat_map(State &state) {
  Threats threats(state.occupied.size());
  for (int id = 0; id < int(state.units.size()); ++id) {
    const Unit &u = state.units[id];
    if (!u.hp || u.side == state.side)
      continue;
    std::vector<int> left, prev;
    state.paths(id, left, prev);
    std::vector<int> origin(left.size(), -1);
    for (int t = 0; t < int(left.size()); ++t) {
      if (left[t] <= 0 || (t != u.tile && state.occupied[t] >= 0))
        continue;
      auto mark = [&](int dest) {
        if (dest >= 0 && (origin[dest] < 0 || t == u.tile))
          origin[dest] = t;
      };
      if (ranged[u.type])
        for (int dest : state.g->targets[t])
          mark(dest);
      else
        for (int dest : state.g->neighbors[t])
          mark(dest);
    }
    for (int t = 0; t < int(left.size()); ++t)
      if (origin[t] >= 0)
        threats[t].emplace_back(id, origin[t]);
  }
  return threats;
}

double danger(State &s, const Threats &threats, int id, int tile,
              int killed = -1) {
  Unit &u = s.units[id];
  int old = u.tile;
  u.tile = tile;
  double total = 0, worst = 0;
  for (auto [enemy, origin] : threats[tile]) {
    if (enemy == killed)
      continue;
    Unit &e = s.units[enemy];
    int home = e.tile;
    e.tile = origin;
    auto hit = s.combat(enemy, id, ranged[e.type] != 0, true);
    e.tile = home;
    double loss = hit.damage - 0.35 * hit.retaliation;
    total += loss;
    worst = std::max(worst, loss);
  }
  u.tile = old;
  double incoming = .55 * total + .45 * worst;
  return .64 * std::min(incoming, double(u.hp)) +
         .85 * std::clamp(incoming - u.hp + 20, 0., 60.);
}

double position(const State &s, int id, int tile, int target) {
  const Unit &u = s.units[id];
  double score = 0;
  int allies = 0, nearest = 100;
  for (int i = 0; i < int(s.units.size()); ++i) {
    const Unit &v = s.units[i];
    if (i == id || !v.hp)
      continue;
    int d = s.g->distance(tile, v.tile);
    if (v.side == u.side) {
      allies += d <= 2;
      score -= .6 * std::min(10, d);
    } else
      nearest = std::min(nearest, d);
  }
  score += 2 * std::min(4, allies);
  if (u.type != 2 && u.type != 4)
    score += .23 * s.g->defense(tile);
  if (ranged[u.type])
    score += .45 * std::min(18, int(s.g->targets[tile].size()));
  score -= 1.2 * s.g->distance(tile, target);
  if (nearest < 100)
    score -= 2 * std::max(0, nearest - 2);
  return score;
}

struct Choice {
  double score = 0;
  Action action{END_TURN, -1, -1};
  void offer(double value, Action candidate) {
    if (value > score) {
      score = value;
      action = candidate;
    }
  }
};

Choice search(State &s, const CaiSearchRequest &r) {
  auto threats = threat_map(s);
  Choice best;
  for (int slot = 0; slot < r.selected_count; ++slot) {
    int id = r.selected_units[slot];
    Unit &u = s.units[id];
    if (!u.hp || !u.moves || u.waiting)
      continue;
    int home = u.tile;
    double base =
        position(s, id, home, r.target) - danger(s, threats, id, home);
    std::vector<int> left, prev;
    s.paths(id, left, prev);
    for (int t = 0; t < int(left.size()); ++t) {
      if (left[t] < 0 || (t != home && s.occupied[t] >= 0))
        continue;
      double terrain = position(s, id, t, r.target);
      double exposure = danger(s, threats, id, t);
      double delta = terrain - exposure - base;
      double rest_cost =
          u.acted ? 0 : std::min(10, 100 - u.hp) * 1.2 + (u.fortified ? 4 : 0);
      Action move{MOVE, id, t};
      if (t != home)
        best.offer(delta - rest_cost - 2., move);
      if (!left[t] || u.attacked)
        continue;
      bool shot = ranged[u.type] != 0;
      auto attack = [&](int dest) {
        if (dest < 0)
          return;
        int victim = s.occupied[dest];
        if (victim < 0 || s.units[victim].side == u.side)
          return;
        u.tile = t;
        auto hit = s.combat(id, victim, shot, true);
        u.tile = home;
        const Unit &v = s.units[victim];
        bool kill = hit.damage >= v.hp;
        int final = kill && !shot ? dest : t;
        double incoming =
            kill ? danger(s, threats, id, final, victim) : exposure;
        double focus = (100 - v.hp) * .18;
        double value = hit.damage * (shot ? 1.24 : 1.) -
                       hit.retaliation * 1.04 + focus + (kill ? 84 : 0) -
                       (hit.retaliation >= u.hp ? 90 : 0) +
                       .65 * (terrain - base - incoming) - rest_cost * .55 +
                       (dest == r.target ? 3 : 0) - (t != home ? 1 : 0);
        best.offer(value,
                   t == home ? Action{shot ? RANGED : MELEE, id, dest} : move);
      };
      if (shot)
        for (int dest : s.g->targets[t])
          attack(dest);
      else
        for (int dest : s.g->neighbors[t])
          attack(dest);
    }
  }
  return best;
}
struct Order {
  Action first;
  Action strike;
  double score;
};

// Enumerate both direct strikes and reachable firing positions. A move and its
// intended strike form one alternative even though the real executor checks
// the observation again between the commands.
std::vector<Order> orders(State &s, int objective, bool advance,
                          const std::vector<char> *eligible = nullptr) {
  std::vector<Order> choices;
  const Action none{END_TURN, -1, -1};
  for (int id = 0; id < int(s.units.size()); ++id) {
    Unit &u = s.units[id];
    if (!u.hp || u.side != s.side || !u.moves || u.waiting ||
        (eligible && !(*eligible)[id]))
      continue;
    int home = u.tile;
    auto distance_to_enemy = [&](int tile) {
      int d = 1000;
      for (const auto &v : s.units)
        if (v.hp && v.side != s.side)
          d = std::min(d, s.g->distance(tile, v.tile));
      return d;
    };
    int old_dist = distance_to_enemy(home);
    std::vector<int> left, prev;
    s.paths(id, left, prev);
    for (int t = 0; t < int(left.size()); ++t) {
      if (left[t] < 0 || (t != home && s.occupied[t] >= 0))
        continue;
      const Action move{MOVE, id, t};
      bool shot = ranged[u.type] != 0;
      if (!u.attacked && left[t] > 0) {
        auto offer = [&](int dest) {
          if (dest < 0)
            return;
          int victim = s.occupied[dest];
          if (victim < 0 || s.units[victim].side == s.side)
            return;
          u.tile = t;
          auto hit = s.combat(id, victim, shot, true);
          u.tile = home;
          const auto &v = s.units[victim];
          double value = hit.damage - 0.72 * hit.retaliation +
                         .32 * (100 - v.hp) + (v.hp <= hit.damage ? 82 : 0) -
                         (u.hp <= hit.retaliation ? 72 : 0) + (shot ? 8 : 0) +
                         (dest == objective ? 7 : 0) + .08 * s.g->defense(t) -
                         (t == home ? 0 : 1.5);
          Action attack{shot ? RANGED : MELEE, id, dest};
          if (value > 0)
            choices.push_back(
                {t == home ? attack : move, t == home ? none : attack, value});
        };
        if (shot)
          for (int dest : s.g->targets[t])
            offer(dest);
        else
          for (int dest : s.g->neighbors[t])
            offer(dest);
      }
      if (t == home || !advance || u.hp < 55)
        continue;
      int d = distance_to_enemy(t);
      if (d >= old_dist || old_dist <= 1)
        continue;
      double value = 2. * (old_dist - d) + .1 * s.g->defense(t) -
                     (shot && d == 1 ? 10 : 0);
      choices.push_back({move, none, value});
    }
  }
  return choices;
}

void execute(State &s, const Order &order) {
  s.step(order.first);
  if (order.strike.kind != END_TURN && s.validate(order.strike) == OK)
    s.step(order.strike);
}
void play_turn(State &s, int target, bool advance) {
  for (int n = 0; n < int(s.units.size()) * 4; ++n) {
    auto choices = orders(s, target, advance);
    if (choices.empty())
      break;
    auto best = std::max_element(
        choices.begin(), choices.end(),
        [](const Order &a, const Order &b) { return a.score < b.score; });
    execute(s, *best);
  }
  s.step({END_TURN, -1, -1});
}
double material(const State &s, int side) {
  double score = 0;
  for (const Unit &u : s.units)
    if (u.hp) {
      const double weight =
          1. + .045 * (std::max(strength[u.type], ranged[u.type]) - 8);
      score += (u.side == side ? 1 : -1) * weight *
               (60. + u.hp + .05 * s.g->defense(u.tile));
    }
  return score;
}
Choice exchange_search(State &root, const CaiSearchRequest &request,
                       Choice formation) {
  std::vector<char> eligible(root.units.size());
  for (int i = 0; i < request.selected_count; ++i)
    eligible[request.selected_units[i]] = 1;
  auto options = orders(root, request.target, true, &eligible);
  int enemy_count = 0;
  for (auto &u : root.units)
    if (u.hp && u.side != root.side)
      ++enemy_count;
  if (!enemy_count)
    return formation;
  std::stable_sort(
      options.begin(), options.end(),
      [](const Order &a, const Order &b) { return a.score > b.score; });
  std::vector<Order> starts;
  int attack_count = 0, move_count = 0;
  const Action none{END_TURN, -1, -1};
  for (const auto &order : options) {
    const auto strike =
        order.strike.kind == END_TURN ? order.first : order.strike;
    bool attack = strike.kind != MOVE;
    if (attack ? attack_count >= 7 : move_count >= 5)
      continue;
    bool duplicate = false;
    for (const auto &other : starts) {
      auto hit = other.strike.kind == END_TURN ? other.first : other.strike;
      if (hit.unit == strike.unit &&
          (attack ? hit.target == strike.target : hit.kind == MOVE))
        duplicate = true;
    }
    if (!duplicate) {
      starts.push_back(order);
      if (attack)
        ++attack_count;
      else
        ++move_count;
    }
  }
  starts.push_back({formation.action, none, 0});
  if (formation.action.kind != END_TURN)
    starts.push_back({none, none, 0});
  Choice best;
  best.score = -1e20;
  for (const auto &order : starts) {
    State s = root;
    if (order.first.kind != END_TURN) {
      execute(s, order);
      play_turn(s, request.target, false);
    } else
      s.step(order.first);
    double immediate = material(s, root.side);
    play_turn(s, -1, true);
    double reply = material(s, root.side);
    play_turn(s, request.target, true);
    double followup = material(s, root.side);
    play_turn(s, -1, false);
    double balance = material(s, root.side);
    const bool is_formation = order.first.kind == formation.action.kind &&
                              order.first.unit == formation.action.unit &&
                              order.first.target == formation.action.target;
    best.offer(.12 * immediate + .33 * reply + .16 * followup + .55 * balance +
                   .09 * order.score + (is_formation ? 5. : 0.),
               order.first);
  }
  return best;
}
} // namespace

Plan formation_plan(const CaiSearchRequest &r, MapCache &cache) {
  make_rules_adapter(*r.state, r.side, &cache);
  State state(cache.geometry);
  state.fog = 0;
  state.side = r.side;
  state.turn_reset_mode = r.state->turn_reset_mode;
  const auto &board = *r.state;
  for (int i = 0; i < board.unit_count; ++i) {
    const auto &u = board.units[i];
    if (!u.hp || (u.side != r.side && !board.visibility[u.tile])) {
      state.units.push_back({});
    } else if (u.side == r.side) {
      state.units.push_back({u.type, u.side, u.tile, u.hp, u.moves, u.attacked,
                             u.acted, u.fortified, u.waiting});
    } else {
      state.units.push_back({u.type, u.side, u.tile, u.hp,
                             combat_rules::moves[u.type], 0, 0, u.fortified,
                             0});
    }
  }
  state.rebuild();
  auto best = exchange_search(state, r, search(state, r));
  Plan plan{{}, {CAI_COMPLETED, 0, {}}};
  auto cmd = best.action;
  if (cmd.kind == END_TURN) {
    for (int i = 0; i < r.selected_count; ++i) {
      const auto &u = board.units[r.selected_units[i]];
      plan.assignments.push_back(
          {CAI_FINISH, u.id, u.tile, u.tile, u.moves, 0, 0, 0, 0, -1, -1});
    }
  } else {
    int tile = state.units[cmd.unit].tile;
    int victim = state.occupied[cmd.target];
    auto hit = state.step(cmd);
    if (hit.code)
      throw std::logic_error("Invalid formation command");
    const auto &u = state.units[cmd.unit];
    int kind = cmd.kind == MOVE    ? CAI_MOVE
               : cmd.kind == MELEE ? CAI_MELEE
                                   : CAI_RANGED;
    plan.assignments.push_back(
        {kind, cmd.unit, tile, cmd.target, u.moves, 0, int(best.score),
         hit.damage, hit.retaliation,
         victim >= 0 && !state.units[victim].hp ? victim : -1, cmd.target});
  }
  plan.result.assignment_count = int(plan.assignments.size());
  return plan;
}
} // namespace combat_ai
