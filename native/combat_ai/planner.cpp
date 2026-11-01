// Local CP tactical search, adapted from CvTacticalAI.cpp:6634-10594.
// See docs/combat-ai-contract.md for deliberate changes from the pinned CP.
#include "planner.h"
#include "../combat_state.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

namespace combat_ai {
void SearchWorkspace::clear() { positions.clear(); open.clear(); equivalent.clear(); }
namespace {
using namespace combat_rules;
struct Candidate {
  std::vector<Command> commands;
  int32_t score, selected_order, kind, tile;
};
class Search {
  SearchWorkspace& work;
  const CaiConfig& config;
  const CaiSearchRequest& request;
  std::unique_ptr<RulesAdapter> adapter;
  std::unique_ptr<DangerModel> danger;
  std::vector<int32_t> selected, selected_index, roles, initial_distance;
  std::vector<CaiUnit> initial_units;
  std::vector<int32_t> initial_enemies;
  CaiSearchStats stats{};
  bool depth_first = false, exhausted = false;
  int32_t best = -1, best_score = std::numeric_limits<int32_t>::min();
  float bias = 1.f;

  bool in_area(const TacticalView& v, int tile) const {
    if (adapter->distance(request.target, tile) <= 5) return true;
    for (int id : v.enemies)
      if (adapter->distance(v.units[id].tile, tile) <= 2) return true;
    for (int id : selected)
      if (initial_units[id].moves > 0 && !initial_units[id].waiting &&
          adapter->distance(initial_units[id].tile, tile) <= 1) return true;
    return false;
  }
  std::vector<int32_t> local_enemies(const TacticalView& v) const {
    std::vector<int32_t> result;
    for (int id : v.enemies) {
      bool local = adapter->distance(request.target, v.units[id].tile) <= 5;
      for (int ours : selected)
        if (request.state->units[ours].moves > 0 && !request.state->units[ours].waiting &&
            adapter->distance(v.units[ours].tile, v.units[id].tile) <= 4) local = true;
      if (local) result.push_back(id);
    }
    return result;
  }
  TacticalView view(const Branch& branch) const {
    TacticalView result(branch, request.side);
    result.enemies = local_enemies(result);
    return result;
  }
  int plot_score(const TacticalPosition& p, const TacticalView& v, int i,
                 bool final, bool* acceptable = nullptr) {
    const auto& u = v.units[selected[i]];
    if (u.hp <= 0) { if (acceptable) *acceptable = true; return 0; }
    int distance = v.enemies.empty() ? adapter->distance(u.tile, request.target)
                                    : enemy_distance(v, *adapter, u.tile);
    int formation = formation_score(roles[i], distance, !v.enemies.empty());
    if (p.last_kill[i]) formation = std::max(60, formation);
    int score = formation;
    if (final) {
      bool flee = false;
      // Safe-plot availability belongs to this branch, never a global CP cache.
      if (u.moves > 0 && !u.waiting)
        for (auto reachable : adapter->reachable(*p.branch, u.id)) {
          bool occupied_contact = false;
          for (auto contact : p.branch->contacts().known_enemies) {
            CaiUnit enemy{};
            if (adapter->known_enemy(*p.branch, contact.id, enemy) && enemy.tile == reachable.tile)
              occupied_contact = true;
          }
          if (occupied_contact) continue;
          if (danger->incoming(*p.branch, u.id, reachable.tile) == 0) { flee = true; break; }
        }
      auto end = score_turn_end(config, *adapter, *danger, *p.branch, v, u.id,
                                p.touched[i] != 0, p.last_kill[i] != 0, flee, bias);
      if (acceptable) *acceptable = end.acceptable;
      score += end.score;
      // HP in the branch already contains retaliation. Penalize it exactly once.
      score -= initial_units[u.id].hp - u.hp;
    } else {
      score -= (100 * danger->incoming(*p.branch, u.id, u.tile) / std::max(1, u.hp)) / 20;
    }
    // CP visibility score for ranged units: geometric range-2 cells near enemies.
    if (roles[i] == CAI_SECOND_LINE)
      for (int tile = 0; tile < int(v.visibility.size()); ++tile)
        if (adapter->distance(u.tile, tile) == 2 &&
            enemy_distance(v, *adapter, tile) < 3 &&
            adapter->can_see(*p.branch, u.tile, tile)) ++score;
    return score + u.moves / 60;
  }
  int attack_score(const TacticalPosition& before, const TacticalView& old,
                   const TacticalPosition& after, const TacticalView& next,
                   Command command, const Transition& transition) {
    const auto& u = old.units[command.unit];
    const auto& now = next.units[command.unit];
    int defender = old.occupied[command.target];
    if (defender < 0) return -1;
    bool kill = next.units[defender].hp == 0;
    int damage = transition.result.damage;
    int self = transition.result.retaliation;
    // Unlike CP's hp-1 tolerance, exact fixed-roll transitions decide kills.
    int scored_damage = std::min(damage, old.units[defender].hp + 10);
    if (self > 0) {
      constexpr float factor[5] = {0.f, .7f, 1.1f, 2.3f, 4.2f};
      constexpr int hp_limit[5] = {0, 70, 40, 20, -5};
      int incoming = now.hp > 0 ? danger->incoming(*after.branch, u.id, now.tile) : 0;
      int delta = int(scored_damage * bias * factor[config.aggression] + .5f) - self;
      if (((incoming > 100 || now.hp < hp_limit[config.aggression]) && delta < 0 && !kill) ||
          (now.hp < 3 && config.aggression != CAI_BRAVEHEART)) return -1;
    }
    int result = (scored_damage + (command.kind == MELEE ? 8 : 0)) * 10;
    if (kill) result += 1000;
    if (command.target == request.target) result += command.kind == MELEE ? 3 : 2;
    if (command.kind == RANGED && u.moves > 60 && enemy_distance(old, *adapter, u.tile) == 1)
      result = (result - (kill ? 1000 : 0)) / 2 + (kill ? 1000 : 0);
    (void)before;
    return result;
  }
  int potential(const TacticalPosition& p, int id, bool future = false) {
    const Branch* branch = p.branch.get();
    auto v = view(*branch);
    if (v.units[id].hp <= 0 || config.aggression == CAI_NONE) return 0;
    const bool can_attack_now = v.units[id].moves > 0 && !v.units[id].attacked && !v.units[id].waiting;
    if (!can_attack_now && !future) return 0;
    int best_attack = 0;
    for (int other : v.enemies) {
      Command c{(v.units[id].type == 1 || v.units[id].type == 2) ? RANGED : MELEE,
                id, v.units[other].tile};
      auto damage = adapter->preview(*branch, c, !can_attack_now);
      if (damage.code) continue;
      int score = std::min(damage.damage, v.units[other].hp + 10);
      if (can_attack_now && damage.damage >= v.units[other].hp) score += 100;
      best_attack = std::max(best_attack, score);
    }
    return best_attack * 10 / (can_attack_now ? 2 : 4);
  }
  int rank(TacticalPosition& p, const TacticalView& v) {
    int score = p.attack_score;
    for (int i = 0; i < int(selected.size()); ++i)
      if (v.units[selected[i]].hp > 0 && initial_units[selected[i]].moves > 0 &&
          !initial_units[selected[i]].waiting) score += plot_score(p, v, i, false);
    return score;
  }
  std::unique_ptr<TacticalPosition> copy(const TacticalPosition& parent) {
    auto child = std::make_unique<TacticalPosition>();
    child->branch = parent.branch->clone();
    child->assignments = parent.assignments;
    child->done = parent.done; child->moved = parent.moved;
    child->last_kill = parent.last_kill; child->touched = parent.touched;
    child->killed = parent.killed; child->attack_score = parent.attack_score;
    child->discovered = parent.discovered;
    child->generation = parent.generation + 1;
    return child;
  }
  bool apply(TacticalPosition& child, Command command, int i) {
    auto old = view(*child.branch);
    const auto u = old.units[command.unit];
    int partner = command.kind == SWAP ? old.occupied[command.target] : -1;
    auto transition = adapter->apply(*child.branch, command);
    if (!transition.accepted) return false;
    auto next = view(*child.branch);
    const auto now = next.units[command.unit];
    if (command.kind == MOVE && now.tile == u.tile && now.moves == u.moves &&
        transition.newly_visible.empty()) return false;
    int kind = command.kind == MOVE ? CAI_MOVE : command.kind == SWAP ? CAI_SWAP :
               command.kind == MELEE ? CAI_MELEE : CAI_RANGED;
    int killed = -1, score = 0;
    if (command.kind == MELEE || command.kind == RANGED) {
      int target = old.occupied[command.target];
      if (target >= 0 && next.units[target].hp == 0) {
        killed = target;
        child.killed.push_back(target);
        std::sort(child.killed.begin(), child.killed.end());
      }
      score = attack_score(child, old, child, next, command, transition);
      if (score < 0) return false;
      child.attack_score += score;
    }
    // A move with new sight is forced in CP and may be followed by another move.
    child.moved[i] = (command.kind == MOVE || command.kind == SWAP) && transition.newly_visible.empty();
    child.last_kill[i] = killed >= 0;
    child.touched[i] = 1;
    child.done[i] = now.hp <= 0 || now.moves <= 0 || now.waiting;
    int to = command.kind == MELEE || command.kind == RANGED ? command.target : now.tile;
    child.assignments.push_back({kind, u.id, u.tile, to, now.moves, roles[i], score,
                                 transition.result.damage, transition.result.retaliation, killed});
    if (partner >= 0) {
      int j = selected_index[partner];
      if (j < 0) return false;
      child.moved[j] = transition.newly_visible.empty();
      child.touched[j] = 1; child.last_kill[j] = 0;
      child.done[j] = next.units[partner].moves <= 0;
      child.assignments.push_back({CAI_SWAP_REVERSE, partner, old.units[partner].tile,
        next.units[partner].tile, next.units[partner].moves, roles[j], 0, 0, 0, -1});
    }
    if (!transition.newly_visible.empty()) {
      auto revealed = adapter->prospective_units(*child.branch, transition.newly_visible);
      bool enemy = std::any_of(revealed.begin(), revealed.end(), [&](const CaiUnit& x) {
        return x.side != request.side && x.hp > 0;
      });
      child.discovered = child.discovered || enemy;
      child.assignments.push_back({CAI_RESTART, u.id, -1, -1, now.moves, roles[i], 0, 0, 0, -1});
    }
    return true;
  }
  std::vector<Candidate> candidates(const TacticalPosition& parent) {
    auto old = view(*parent.branch);
    std::vector<std::vector<Command>> per_unit(selected.size());
    for (Command c : adapter->actions(*parent.branch)) {
      if (c.kind == END_TURN || c.unit < 0) continue;
      int i = selected_index[c.unit];
      if (i < 0 || parent.done[i] || c.kind == FORTIFY || c.kind == SKIP) continue;
      if ((c.kind == MELEE || c.kind == RANGED) && config.aggression == CAI_NONE) continue;
      if (c.kind == MOVE && (parent.moved[i] || !in_area(old, c.target))) continue;
      if (c.kind == SWAP) {
        int partner = old.occupied[c.target];
        int j = partner < 0 ? -1 : selected_index[partner];
        if (j < 0 || parent.done[j] || parent.moved[i] || parent.moved[j]) continue;
      }
      per_unit[i].push_back(c);
    }
    std::vector<Candidate> result;
    for (int i = 0; i < int(selected.size()); ++i) {
      if (parent.done[i]) continue;
      std::vector<Candidate> choices;
      int initial_score = plot_score(parent, old, i, false);
      for (Command command : per_unit[i]) {
        auto child = copy(parent);
        if (!apply(*child, command, i)) { ++stats.pruned; continue; }
        auto next = view(*child->branch);
        int gain = plot_score(*child, next, i, false) - initial_score + child->attack_score - parent.attack_score;
        if (command.kind == SWAP) {
          int j = selected_index[old.occupied[command.target]];
          gain += plot_score(*child, next, j, false) - plot_score(parent, old, j, false);
        }
        if (command.kind == MOVE || command.kind == SWAP) gain += potential(*child, selected[i]);
        child->rank = parent.rank + gain;
        auto& last = child->assignments[parent.assignments.size()];
        if (last.kind == CAI_MOVE || last.kind == CAI_SWAP) last.score = gain;
        choices.push_back({{command}, gain, i, command.kind, command.target});
      }
      auto compare = [](const Candidate& a, const Candidate& b) {
        return std::make_tuple(-a.score, a.selected_order, a.kind, a.tile) <
               std::make_tuple(-b.score, b.selected_order, b.kind, b.tile);
      };
      std::stable_sort(choices.begin(), choices.end(), compare);
      int limit = depth_first ? 2 : config.choices_per_unit;
      // Keep a movement alternative when many attacks compete (CP ranged rule).
      if (int(choices.size()) > limit) {
        auto move = std::find_if(choices.begin() + limit, choices.end(), [](const Candidate& c) {
          return c.kind == MOVE || c.kind == SWAP;
        });
        bool have_move = std::any_of(choices.begin(), choices.begin() + limit, [](const Candidate& c) {
          return c.kind == MOVE || c.kind == SWAP;
        });
        if (!have_move && move != choices.end()) std::swap(choices[limit - 1], *move);
        choices.resize(limit);
      }
      // CP SComboMove: first move a blocker out, then occupy its freed tile.
      // Construct at most one follow-up per other selected unit for each of
      // this unit's bounded preferred moves. Commands are still applied in order.
      for (const auto& c : choices) {
        if (c.kind != MOVE) continue;
        auto first = copy(parent);
        if (!apply(*first, c.commands.front(), i)) continue;
        int freed = old.units[selected[i]].tile;
        for (int j = 0; j < int(selected.size()); ++j) {
          if (j == i || parent.done[j] || parent.moved[j]) continue;
          auto reachable = adapter->reachable(*first->branch, selected[j]);
          if (std::none_of(reachable.begin(), reachable.end(), [&](auto tile) { return tile.tile == freed; })) continue;
          auto combo = copy(*first);
          Command second{MOVE, selected[j], freed};
          if (!apply(*combo, second, j)) continue;
          auto next = view(*combo->branch);
          int gain = plot_score(*combo,next,i,false) + plot_score(*combo,next,j,false) -
                     initial_score - plot_score(parent,old,j,false) +
                     potential(*combo,selected[i]) + potential(*combo,selected[j]);
          result.push_back({{c.commands.front(), second},gain,i,MOVE,freed});
        }
      }
      for (auto& c : choices) result.push_back(std::move(c));
      // Deferred finish: no SKIP mutation, since it must not fabricate fortification.
      bool acceptable;
      int finish_gain = plot_score(parent, old, i, true, &acceptable) - initial_score;
      if (!parent.touched[i]) {
        finish_gain += std::min(23, 100 - old.units[selected[i]].hp);
        if (old.units[selected[i]].type != 2 && old.units[selected[i]].type != 4) finish_gain += 23;
      }
      finish_gain += potential(parent, selected[i], true);
      // Keep unsafe finishes for the final joint check and risk-budget exception.
      result.push_back({{}, finish_gain, i, SKIP, old.units[selected[i]].tile});
    }
    std::stable_sort(result.begin(), result.end(), [](const Candidate& a, const Candidate& b) {
      return std::make_tuple(-a.score, a.selected_order, a.kind, a.tile) <
             std::make_tuple(-b.score, b.selected_order, b.kind, b.tile);
    });
    return result;
  }
  std::vector<int32_t> key(const TacticalPosition& p, const TacticalView& v) {
    std::vector<int32_t> out;
    for (const auto& u : v.units) {
      // Never use an unobserved enemy's true position for tactical decisions.
      if (u.side != request.side && !v.visibility[u.tile]) continue;
      out.insert(out.end(), {u.id,u.tile,u.hp,u.moves,u.attacked,u.acted,u.fortified,u.waiting});
    }
    out.push_back(-2);
    out.insert(out.end(), v.visibility.begin(), v.visibility.end());
    for (int i = 0; i < int(selected.size()); ++i)
      out.insert(out.end(), {p.done[i], p.moved[i], p.last_kill[i], p.touched[i]});
    out.push_back(int(p.discovered));
    out.push_back(int(p.killed.size()));
    out.insert(out.end(), p.killed.begin(), p.killed.end());
    const auto& memory = p.branch->contacts();
    out.insert(out.end(), {memory.turn,memory.side});
    for (auto contact : memory.known_enemies)
      out.insert(out.end(), {contact.id,contact.last_seen_own_turn});
    return out;
  }
  bool terminal(const TacticalPosition& p, const TacticalView& v) const {
    bool all_done = std::all_of(p.done.begin(), p.done.end(), [](int x) { return x != 0; });
    return all_done || (!initial_enemies.empty() && local_enemies(v).empty());
  }
  bool complete(TacticalPosition& p, const TacticalView& v, int& score) {
    if (p.finalized) { score = p.final_score; return true; }
    int losses = 0, positive = 0, negative = 0, target_before = 0, target_after = 0;
    score = p.attack_score;
    for (int i = 0; i < int(selected.size()); ++i) {
      int id = selected[i];
      const auto& u = v.units[id];
      // CP addAvailableUnit excludes root-unavailable units. They remain in
      // the physical state as occupancy and support, without a new finish order.
      if (initial_units[id].moves <= 0 || initial_units[id].waiting) continue;
      if (u.hp <= 0) { score -= initial_units[id].hp; continue; }
      // Early finish leaves untouched, still-available units to later tasks.
      if (!p.done[i] && !p.touched[i]) { score += 123; continue; }
      bool acceptable = false;
      int end = plot_score(p, v, i, true, &acceptable);
      if (!acceptable) ++losses;
      score += end;
      int distance = enemy_distance(v, *adapter, u.tile);
      int ideal = roles[i] == CAI_FIRST_LINE ? 1 : 2;
      if (roles[i] == CAI_THIRD_LINE) {
        positive += initial_distance[i] == 1 && distance > 1;
        negative += initial_distance[i] > 1 && distance == 1;
      } else {
        positive += initial_distance[i] != ideal && distance == ideal;
        negative += initial_distance[i] == ideal && distance != ideal;
      }
      target_before += adapter->distance(initial_units[id].tile, request.target);
      target_after += adapter->distance(u.tile, request.target);
      // Replace deferred bookkeeping records, then append a single final record.
      p.assignments.erase(std::remove_if(p.assignments.begin(), p.assignments.end(), [&](const CaiAssignment& a) {
        return a.unit == id && a.kind == CAI_FINISH_TEMP;
      }), p.assignments.end());
      p.assignments.push_back({CAI_FINISH, id, u.tile, u.tile, u.moves, roles[i], end, 0, 0, -1});
    }
    if (losses > config.risk_budget_base + config.risk_budget_per_kill * int(p.killed.size())) return false;
    bool improved = !p.killed.empty() || p.attack_score > 0 || p.discovered ||
                    (!v.enemies.empty() ? positive > negative : target_after < target_before);
    if (improved) { p.finalized = true; p.final_score = score; }
    return improved;
  }
  bool heap_compare(int a, int b) const {
    const auto& x = *work.positions[a]; const auto& y = *work.positions[b];
    if (x.generation != y.generation)
      return depth_first ? x.generation < y.generation : x.generation > y.generation;
    if (x.rank != y.rank) return x.rank < y.rank;
    return a > b;
  }
 public:
  Search(SearchWorkspace& w, const CaiConfig& c, const CaiSearchRequest& r)
      : work(w), config(c), request(r), adapter(make_rules_adapter(*r.state, r.side)),
        danger(make_danger_model(*adapter)), selected(r.selected_units, r.selected_units + r.selected_count),
        selected_index(r.state->unit_count, -1) {}
  Plan run() {
    work.clear();
    ContactMemory memory{request.contacts.turn, request.contacts.side, {}};
    for (int i = 0; i < request.contacts.known_enemy_count; ++i)
      memory.known_enemies.push_back({request.contacts.known_enemies[i].id,
                                    request.contacts.known_enemies[i].last_seen_own_turn});
    auto root = std::make_unique<TacticalPosition>();
    root->branch = adapter->root(*request.state, memory);
    auto initial = view(*root->branch);
    initial_units = initial.units;
    initial_enemies = local_enemies(initial);
    int active_count = 0;
    for (int id : selected)
      if (initial.units[id].moves > 0 && !initial.units[id].waiting) ++active_count;
    bias = std::max(.9f, std::sqrt(float(active_count) / std::max(1, int(initial_enemies.size()))));
    for (int i = 0; i < int(selected.size()); ++i) {
      int id = selected[i]; selected_index[id] = i;
      roles.push_back(movement_role(initial.units[id], danger->incoming(*root->branch, id, initial.units[id].tile)));
      initial_distance.push_back(enemy_distance(initial, *adapter, initial.units[id].tile));
      root->done.push_back(initial.units[id].moves <= 0 || initial.units[id].waiting);
      root->moved.push_back(0); root->last_kill.push_back(0); root->touched.push_back(0);
      root->assignments.push_back({CAI_INITIAL,id,initial.units[id].tile,initial.units[id].tile,
                                  initial.units[id].moves,roles.back(),0,0,0,-1});
    }
    root->rank = rank(*root, initial);
    work.equivalent.emplace(key(*root, initial), std::make_pair(0, 0));
    work.positions.push_back(std::move(root)); work.open.push_back(0);
    stats.generated = stats.peak_positions = 1;
    auto compare = [&](int a, int b) { return heap_compare(a,b); };
    while (!work.open.empty() && !exhausted) {
      if (!depth_first && work.positions[work.open.front()]->generation > 3) {
        depth_first = true;
        std::make_heap(work.open.begin(), work.open.end(), compare);
      }
      std::pop_heap(work.open.begin(), work.open.end(), compare);
      int current = work.open.back(); work.open.pop_back();
      auto& parent = *work.positions[current];
      auto current_view = view(*parent.branch);
      auto equivalent = work.equivalent.find(key(parent, current_view));
      if (equivalent != work.equivalent.end() && equivalent->second.second != current) {
        ++stats.pruned; continue;
      }
      ++stats.checked;
      if (terminal(parent, current_view)) {
        int score;
        if (complete(parent, current_view, score)) {
          ++stats.completed;
          if (best < 0 || score > best_score) { best = current; best_score = score; }
          if (stats.completed >= config.max_completed && !work.open.empty()) { exhausted = true; break; }
        } else ++stats.pruned;
        continue;
      }
      if (stats.generated >= config.position_capacity) { exhausted = true; break; }
      parent.rank = rank(parent, current_view);
      auto choices = candidates(parent);
      int children = 0;
      for (auto& candidate : choices) {
        if (children >= (depth_first ? 1 : config.max_branches)) break;
        if (stats.generated >= config.position_capacity) { exhausted = true; break; }
        auto child = copy(parent);
        if (candidate.commands.empty()) {
          int i = candidate.selected_order, id = selected[i];
          child->done[i] = 1;
          child->assignments.push_back({CAI_FINISH_TEMP, id, current_view.units[id].tile,
            current_view.units[id].tile, current_view.units[id].moves, roles[i], 0, 0, 0, -1});
        } else {
          bool accepted = true;
          for (auto command : candidate.commands)
            if (!apply(*child, command, selected_index[command.unit])) { accepted = false; break; }
          if (!accepted) { ++stats.pruned; continue; }
          // The blocker made a forced move, so CP permits it to move again.
          if (candidate.commands.size() > 1) child->moved[candidate.selected_order] = 0;
          auto& first = child->assignments[parent.assignments.size()];
          if (first.kind == CAI_MOVE || first.kind == CAI_SWAP) first.score = candidate.score;
        }
        child->rank = parent.rank + candidate.score;
        auto next = view(*child->branch);
        auto identity = key(*child, next);
        auto found = work.equivalent.find(identity);
        if (found != work.equivalent.end() && found->second.first >= child->attack_score) {
          ++stats.pruned; continue;
        }
        bool finished = terminal(*child, next);
        int final_score = 0;
        if (finished && !complete(*child, next, final_score)) { ++stats.pruned; continue; }
        int id = int(work.positions.size());
        work.equivalent[std::move(identity)] = {child->attack_score, id};
        work.positions.push_back(std::move(child));
        ++stats.generated; ++children;
        stats.peak_positions = stats.generated;
        if (finished) {
          ++stats.completed;
          if (best < 0 || final_score > best_score) { best = id; best_score = final_score; }
          if (stats.completed >= config.max_completed) { exhausted = true; break; }
        } else {
          work.open.push_back(id); std::push_heap(work.open.begin(), work.open.end(), compare);
        }
      }
      // Accepted terminal nodes have already updated best; pool exhaustion
      // therefore always preserves the best completed plan seen so far.
    }
    Plan plan{{}, {exhausted ? CAI_BUDGET_EXHAUSTED : best < 0 ? CAI_NO_PLAN : CAI_COMPLETED, 0, stats}};
    if (best >= 0) plan.assignments = work.positions[best]->assignments;
    plan.result.assignment_count = int32_t(plan.assignments.size());
    return plan;
  }
};
}  // namespace
Plan find_assignments(SearchWorkspace& workspace, const CaiConfig& config, const CaiSearchRequest& request) {
  return Search(workspace, config, request).run();
}
}  // namespace combat_ai
