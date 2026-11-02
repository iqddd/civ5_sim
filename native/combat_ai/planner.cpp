// Local CP tactical search, adapted from CvTacticalAI.cpp:6634-10594.
// See docs/combat-ai-contract.md for deliberate changes from the pinned CP.
#include "planner.h"
#include "cp_tactical.h"
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
class Search {
  SearchWorkspace& work;
  const CaiConfig& config;
  const CaiSearchRequest& request;
  std::unique_ptr<RulesAdapter> adapter;
  std::unique_ptr<CpTactics> cp;
  std::vector<int32_t> selected, selected_index, roles, initial_distance;
  std::vector<CaiUnit> initial_units;
  std::vector<int32_t> initial_enemies;
  CaiSearchStats stats{};
  bool depth_first = false, exhausted = false;
  int32_t best = -1, best_score = std::numeric_limits<int32_t>::min();
  float bias = 1.f;

  TacticalView view(const Branch& branch) const { return TacticalView(branch,request.side); }
  std::unique_ptr<TacticalPosition> copy(const TacticalPosition& parent) {
    auto child = std::make_unique<TacticalPosition>();
    child->branch = parent.branch->clone();
    child->assignments = parent.assignments;
    child->plots = parent.plots;
    child->done = parent.done; child->moved = parent.moved;
    child->last_kill = parent.last_kill; child->touched = parent.touched;
    child->killed = parent.killed; child->attack_score = parent.attack_score;
    child->discovered = parent.discovered;
    child->generation = parent.generation + 1;
    return child;
  }
  bool apply(TacticalPosition& child, Command command, int i, int score) {
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
    int killed = -1;
    if (command.kind == MELEE || command.kind == RANGED) {
      int target = old.occupied[command.target];
      if (target >= 0 && next.units[target].hp == 0) {
        killed = target;
        child.killed.push_back(target);
        std::sort(child.killed.begin(), child.killed.end());
      }
      child.attack_score += score;
    }
    // Only new enemy contact makes an ordinary move forced (see below).
    child.moved[i] = command.kind == SWAP ? 2 : int(command.kind == MOVE);
    child.last_kill[i] = killed >= 0;
    child.touched[i] = 1;
    child.done[i] = now.hp <= 0 || now.moves <= 0 || now.waiting;
    int to = command.kind == MELEE || command.kind == RANGED ? command.target : now.tile;
    child.assignments.push_back({kind, u.id, u.tile, to, now.moves, roles[i], score,
                                 transition.result.damage, transition.result.retaliation, killed, command.target});
    if (partner >= 0) {
      int j = selected_index[partner];
      if (j < 0) return false;
      child.moved[j] = 2;
      child.touched[j] = 1; child.last_kill[j] = 0;
      child.done[j] = next.units[partner].moves <= 0;
      child.assignments.push_back({CAI_SWAP_REVERSE, partner, old.units[partner].tile,
        next.units[partner].tile, next.units[partner].moves, roles[j], 0, 0, 0, -1, -1});
    }
    if (!transition.newly_visible.empty()) {
      auto revealed = adapter->prospective_units(*child.branch, transition.newly_visible);
      bool enemy = std::any_of(revealed.begin(), revealed.end(), [&](const CaiUnit& x) {
        return x.side != request.side && x.hp > 0;
      });
      child.discovered = child.discovered || enemy;
      child.pending_restart = child.pending_restart || enemy;
      if (enemy && command.kind == MOVE) child.moved[i] = 0;
      cp->extend_plots(child,transition.newly_visible);
    }
    return true;
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
    out.insert(out.end(), p.plots.begin(), p.plots.end());
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
  bool terminal(const TacticalPosition& p) const {
    bool all_done = std::all_of(p.done.begin(), p.done.end(), [](int x) { return x != 0; });
    return all_done || (!initial_enemies.empty() && initial_enemies.size()==p.killed.size());
  }
  bool complete(TacticalPosition& p, const TacticalView& v, int& score) {
    if (p.finalized) { score = p.final_score; return true; }
    int losses = 0, positive = 0, negative = 0, target_before = 0, target_after = 0;
    score = p.rank;
    auto board = cp->board(p);
    for (int i = 0; i < int(selected.size()); ++i) {
      int id = selected[i];
      const auto& u = v.units[id];
      // CP addAvailableUnit excludes root-unavailable units. They remain in
      // the physical state as occupancy and support, without a new finish order.
      if (p.done[i] == 2 || initial_units[id].moves <= 0 || initial_units[id].waiting) continue;
      if (u.hp <= 0) { score -= initial_units[id].hp; continue; }
      // Early finish leaves untouched, still-available units to later tasks.
      if (!p.done[i]) { if (!p.touched[i]) score += 123; continue; }
      bool acceptable = false;
      int end = cp->move_score(p, board, i, u.tile, 0, true, &acceptable).score;
      if (!acceptable) ++losses;
      auto latest = std::find_if(p.assignments.rbegin(),p.assignments.rend(),[&](const CaiAssignment& a) {
        return a.unit == id && (a.kind == CAI_MOVE || a.kind == CAI_SWAP ||
               (a.kind == CAI_MELEE && a.killed_unit >= 0));
      });
      score += latest == p.assignments.rend() ? end : std::max(0,end-latest->score);
      int distance = board.plots[u.tile].distance;
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
      p.assignments.push_back({CAI_FINISH, id, u.tile, u.tile, u.moves, roles[i], end, 0, 0, -1, -1});
    }
    if (losses > config.risk_budget_base + config.risk_budget_per_kill * int(p.killed.size())) return false;
    bool attacked=std::any_of(p.assignments.begin(),p.assignments.end(),[](const CaiAssignment& a) {
      return a.kind==CAI_MELEE || a.kind==CAI_RANGED;
    });
    bool improved = !p.killed.empty() || p.discovered ||
                    (!board.enemies.empty() ? (positive > negative ||
                     (positive == negative && attacked)) : target_after < target_before);
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
        selected(r.selected_units, r.selected_units + r.selected_count),
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
    for (int i = 0; i < int(selected.size()); ++i) {
      int id = selected[i]; selected_index[id] = i;
      root->done.push_back(initial.units[id].moves <= 0 || initial.units[id].waiting ? 2 : 0);
      root->moved.push_back(0); root->last_kill.push_back(0); root->touched.push_back(0);
    }
    cp = std::make_unique<CpTactics>(*adapter,*root->branch,config,request);
    cp->initialize(*root);
    roles = cp->roles; initial_distance = cp->initial_distance;
    initial_enemies = cp->initial_enemies; bias = cp->bias;
    auto board = cp->board(*root);
    for (int i = 0; i < int(selected.size()); ++i) {
      int id = selected[i], tile = initial.units[id].tile;
      int score = cp->move_score(*root,board,i,tile,0,true,nullptr,true).score;
      root->assignments.push_back({CAI_INITIAL,id,tile,tile,initial.units[id].moves,roles[i],score,0,0,-1,-1});
      if (root->done[i] == 2)
        root->assignments.push_back({CAI_BLOCKED,id,tile,tile,initial.units[id].moves,roles[i],0,0,0,-1,-1});
    }
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
      if (terminal(parent)) {
        int score;
        if (complete(parent, current_view, score)) {
          ++stats.completed;
          if (best < 0 || score > best_score) { best = current; best_score = score; }
          if (stats.completed >= config.max_completed && !work.open.empty()) { exhausted = true; break; }
        } else ++stats.pruned;
        continue;
      }
      if (stats.generated >= config.position_capacity) { exhausted = true; break; }
      auto choices = cp->candidates(parent,depth_first);
      int children = 0;
      for (auto& candidate : choices) {
        if (children >= (depth_first ? 1 : config.max_branches)) break;
        if (stats.generated >= config.position_capacity) { exhausted = true; break; }
        auto child = copy(parent);
        bool accepted = true;
        for (auto move : candidate.moves) {
          int i = selected_index[move.unit], id = move.unit;
          if (move.kind == CP_SWAP_REVERSE) {
            child->assignments.back().score = move.score;
            continue;
          }
          if (move.kind == CP_FINISH_TEMP || move.kind == CP_BLOCKED) {
            child->done[i] = move.kind == CP_BLOCKED ? 2 : 1;
            child->last_kill[i] = 0;
            child->assignments.push_back({move.kind == CP_BLOCKED ? CAI_BLOCKED : CAI_FINISH_TEMP,
              id, current_view.units[id].tile,current_view.units[id].tile,
              current_view.units[id].moves,roles[i],move.score,0,0,-1,-1});
          } else {
            int kind = move.kind == CP_MELEE ? MELEE : move.kind == CP_RANGED ? RANGED :
                       move.kind == CP_SWAP ? SWAP : MOVE;
            if (!apply(*child,{kind,id,move.tile},i,move.score)) { accepted=false; break; }
            if (move.kind == CP_FORCED) child->moved[i] = 0;
          }
        }
        if (!accepted) { ++stats.pruned; continue; }
        if (child->pending_restart) {
          int id = candidate.moves.front().unit;
          child->assignments.push_back({CAI_RESTART,id,-1,-1,0,roles[selected_index[id]],0,0,0,-1,-1});
          child->pending_restart = false;
        }
        child->rank = parent.rank;
        for (auto move : candidate.moves)
          if (move.kind != CP_BLOCKED && move.kind != CP_FINISH_TEMP) child->rank += move.score;
        auto next = view(*child->branch);
        auto identity = key(*child, next);
        auto found = work.equivalent.find(identity);
        if (found != work.equivalent.end() && found->second.first >= child->attack_score) {
          ++stats.pruned; continue;
        }
        bool finished = terminal(*child);
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
