// Local CP tactical search, adapted from CvTacticalAI.cpp:6634-10594.
// Current behavior: docs/combat-ai-contract.md; CP mapping: combat-ai-cp-transfer.md.
#include "planner.h"
#include "cp_tactical.h"
#include "../combat_state.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>

namespace combat_ai {
namespace {
void require_invariant(bool condition, const char* message) {
  // Also enforce internal invariants in release builds, before indexing arrays.
  if (!condition) throw std::logic_error(message);
  assert(condition);
}

void set_swap_reverse_score(TacticalPosition& child, const CpMove& move) {
  // Contract: docs/combat-ai-contract.md#combo-invariants.
  require_invariant(child.assignments.size() >= 2,"swap reverse without a swap");
  const auto& forward = child.assignments[child.assignments.size()-2];
  auto& reverse = child.assignments.back();
  require_invariant(forward.kind == CAI_SWAP && reverse.kind == CAI_SWAP_REVERSE &&
      reverse.unit == move.unit && reverse.to_tile == move.tile &&
      forward.from_tile == reverse.to_tile && forward.to_tile == reverse.from_tile,
      "swap reverse does not match the physical swap");
  assert(child.assignments.back().kind == CAI_SWAP_REVERSE);
  reverse.score = move.score;
}
} // namespace

SelectedUnits::SelectedUnits(const int32_t* selected, int32_t count, int32_t roster_size)
    : slots(size_t(roster_size),SlotIdx{}) {
  for (int32_t slot_index = 0; slot_index < count; ++slot_index) {
    UnitId unit{selected[slot_index]};
    require_invariant(unit.value >= 0 && unit.value < roster_size,"invalid selected unit ID");
    require_invariant(!slots[unit].valid(),"duplicate selected unit ID");
    units.push_back(unit);
    slots[unit] = SlotIdx{slot_index};
  }
}
std::optional<SlotIdx> SelectedUnits::find_slot(UnitId unit) const {
  require_invariant(unit.value >= 0 && size_t(unit.value) < slots.size(),"unit ID outside roster");
  SlotIdx slot = slots[unit];
  return slot.valid() ? std::optional<SlotIdx>{slot} : std::nullopt;
}
SlotIdx SelectedUnits::slot_of(UnitId unit) const {
  auto slot = find_slot(unit);
  require_invariant(slot.has_value(),"candidate uses a unit outside the selected group");
  return *slot;
}

std::optional<int32_t> latest_move_score(const TacticalPosition& position, UnitId unit) {
  auto latest = std::find_if(position.assignments.rbegin(),position.assignments.rend(),
      [&](const CaiAssignment& record) {
    return record.unit == unit.value && (record.kind == CAI_MOVE || record.kind == CAI_SWAP ||
           (record.kind == CAI_MELEE && record.killed_unit >= 0));
  });
  if (latest == position.assignments.rend()) return std::nullopt;
  return latest->score;
}

bool has_attack_assignment(const TacticalPosition& position) {
  return std::any_of(position.assignments.begin(),position.assignments.end(),
      [](const CaiAssignment& record) { return record.kind == CAI_MELEE || record.kind == CAI_RANGED; });
}

std::vector<int32_t> search_position_key(const TacticalPosition& position,
    const TacticalView& view, int32_t side, const SelectedUnits& selected) {
  // Contract: docs/combat-ai-contract.md#candidate-order.
  std::vector<int32_t> result;
  for (const auto& unit : view.units) {
    if (unit.side != side && !view.visibility[unit.tile]) continue;
    result.insert(result.end(),{unit.id,unit.tile,unit.hp,unit.moves,unit.attacked,unit.acted,unit.fortified,unit.waiting});
  }
  result.push_back(-2);
  result.insert(result.end(),view.visibility.begin(),view.visibility.end());
  result.insert(result.end(),position.plots.begin(),position.plots.end());
  for (int32_t slot_index = 0; slot_index < int32_t(selected.size()); ++slot_index) {
    SlotIdx slot{slot_index};
    auto last_move = latest_move_score(position,selected.unit_at(slot));
    result.insert(result.end(),{int32_t(position.slot_states[slot]),int32_t(position.move_flags[slot]),
        position.last_kill[slot],position.touched[slot],int32_t(last_move.has_value()),last_move.value_or(0)});
  }
  // Equal physical swaps can have different final contributions: SWAP_REVERSE
  // is not a last move in CP's accounting. Absence also differs from score 0.
  result.insert(result.end(),{position.rank,int32_t(has_attack_assignment(position)),int32_t(position.discovered)});
  result.push_back(int32_t(position.killed.size()));
  result.insert(result.end(),position.killed.begin(),position.killed.end());
  const auto& memory = position.branch->contacts();
  result.insert(result.end(),{memory.turn,memory.side});
  for (auto contact : memory.known_enemies)
    result.insert(result.end(),{contact.id,contact.last_seen_own_turn});
  return result;
}

bool terminal_position(const TacticalPosition& position, const std::vector<int32_t>& initial_enemy_ids) {
  // Contract: docs/combat-ai-contract.md#early-finish.
  bool all_finished = std::all_of(position.slot_states.begin(),position.slot_states.end(),
      [](SlotState state) { return state != SlotState::Pending; });
  if (all_finished) return true;
  if (initial_enemy_ids.empty()) return false;
  // Fix to CP isEarlyFinish (:8733): newly discovered casualties cannot replace
  // an initial target. The early-finish objective remains the ORIGINAL set.
  return std::all_of(initial_enemy_ids.begin(),initial_enemy_ids.end(),[&](int32_t enemy_id) {
    return std::find(position.killed.begin(),position.killed.end(),enemy_id) != position.killed.end();
  });
}

void finalize_assignments(TacticalPosition& position, const FinalEvaluation& evaluation) {
  require_invariant(evaluation.accepted,"cannot finalize a rejected position");
  for (const auto& finish : evaluation.finishes) {
    require_invariant(finish.kind == CAI_FINISH,"invalid final assignment kind");
    require_invariant(std::none_of(position.assignments.begin(),position.assignments.end(),
        [&](const CaiAssignment& record) { return record.kind == CAI_FINISH && record.unit == finish.unit; }),
        "unit already finalized");
  }
  position.assignments.erase(std::remove_if(position.assignments.begin(),position.assignments.end(),
      [&](const CaiAssignment& record) {
    return record.kind == CAI_FINISH_TEMP && std::any_of(evaluation.finishes.begin(),evaluation.finishes.end(),
        [&](const CaiAssignment& finish) { return finish.unit == record.unit; });
  }),position.assignments.end());
  position.assignments.insert(position.assignments.end(),evaluation.finishes.begin(),evaluation.finishes.end());
}

bool CandidateExecutor::apply_command(TacticalPosition& child, Command command, SlotIdx slot, int32_t score) {
  using namespace combat_rules;
  auto old = TacticalView(*child.branch,side);
  UnitId unit_id{command.unit};
  const auto before = old.units[unit_id.value];
  int32_t partner_id = command.kind == SWAP ? old.occupied[command.target] : CAI_NO_ID;
  auto transition = adapter.apply(*child.branch,command);
  if (!transition.accepted) return false;
  auto next = TacticalView(*child.branch,side);
  const auto after = next.units[unit_id.value];
  if (command.kind == MOVE && after.tile == before.tile && after.moves == before.moves &&
      transition.newly_visible.empty()) return false;

  CaiAssignmentKind kind = command.kind == MOVE ? CAI_MOVE : command.kind == SWAP ? CAI_SWAP :
                           command.kind == MELEE ? CAI_MELEE : CAI_RANGED;
  int32_t killed_unit = CAI_NO_ID;
  if (command.kind == MELEE || command.kind == RANGED) {
    int32_t target_id = old.occupied[command.target];
    // State::kill normalizes HP to zero; roots reject negative HP.
    if (target_id >= 0 && next.units[target_id].hp == 0) {
      killed_unit = target_id;
      child.killed.push_back(target_id);
      std::sort(child.killed.begin(),child.killed.end());
    }
    child.attack_score += score;
  }
  // Contract: docs/combat-ai-contract.md#continuation.
  child.move_flags[slot] = command.kind == SWAP ? MoveFlag::Swapped :
                           command.kind == MOVE ? MoveFlag::Moved : MoveFlag::None;
  child.last_kill[slot] = killed_unit >= 0;
  child.touched[slot] = 1;
  child.slot_states[slot] = after.hp <= 0 || after.moves <= 0 || after.waiting
      ? SlotState::Finished : SlotState::Pending;

  auto record = stationary_assignment(kind,before,tactics.roles[slot],score);
  record.to_tile = command.kind == MELEE || command.kind == RANGED ? command.target : after.tile;
  record.remaining_moves = after.moves;
  record.damage = transition.result.damage;
  record.self_damage = transition.result.retaliation;
  record.killed_unit = killed_unit;
  record.command_target = command.target;
  child.assignments.push_back(record);

  if (partner_id >= 0) {
    UnitId partner{partner_id};
    SlotIdx partner_slot = selected.slot_of(partner);
    child.move_flags[partner_slot] = MoveFlag::Swapped;
    child.touched[partner_slot] = 1;
    child.last_kill[partner_slot] = 0;
    child.slot_states[partner_slot] = next.units[partner.value].moves <= 0
        ? SlotState::Finished : SlotState::Pending;
    auto reverse = stationary_assignment(CAI_SWAP_REVERSE,old.units[partner.value],tactics.roles[partner_slot]);
    reverse.to_tile = next.units[partner.value].tile;
    reverse.remaining_moves = next.units[partner.value].moves;
    child.assignments.push_back(reverse);
  }

  if (!transition.newly_visible.empty()) {
    auto revealed = adapter.prospective_units(*child.branch,transition.newly_visible);
    bool enemy_contact = false;
    for (const auto& enemy : revealed) {
      if (enemy.side == side || enemy.hp <= 0) continue;
      enemy_contact = true;
      if (child.restart_source) continue;
      UnitId observer = unit_id;
      // A physical swap updates both sight sources. Attribute a contact to a
      // participant which actually sees it, rather than the combo's first ID.
      if (partner_id >= 0 && !adapter.can_see(*child.branch,after.tile,enemy.tile) &&
          adapter.can_see(*child.branch,next.units[partner_id].tile,enemy.tile))
        observer = UnitId{partner_id};
      child.restart_source = observer;
    }
    // Contract: docs/combat-ai-contract.md#continuation and #restart.
    // Empty newly-visible plots extend the area but do not reset ordinary moves.
    child.discovered = child.discovered || enemy_contact;
    if (enemy_contact && command.kind == MOVE) child.move_flags[slot] = MoveFlag::None;
    tactics.extend_plots(child,transition.newly_visible);
  }
  return true;
}

bool CandidateExecutor::apply(TacticalPosition& child, const CpCombo& candidate) {
  // Contract: docs/combat-ai-contract.md#combo-invariants.
  using namespace combat_rules;
  require_invariant(!candidate.moves.empty() && candidate.moves.size() <= 2,"invalid combo size");
  require_invariant(!child.restart_source,"unflushed contact in parent position");
  // Check IDs and pair structure before the first physical mutation. The
  // reverse record itself is checked after executing the physical swap.
  const int32_t tile_count = child.branch->state().tile_count;
  for (size_t move_index = 0; move_index < candidate.moves.size(); ++move_index) {
    const auto& move = candidate.moves[move_index];
    selected.slot_of(UnitId{move.unit});
    require_invariant(move.kind >= CP_MOVE && move.kind <= CP_SWAP_REVERSE,"invalid internal move kind");
    require_invariant(move.tile >= 0 && move.tile < tile_count,"invalid internal target tile");
    if (move_index > 0)
      require_invariant(move.unit != candidate.moves.front().unit,"two assignments for the same unit in a combo");
    if (move.kind == CP_SWAP_REVERSE)
      require_invariant(move_index == 1 && candidate.moves.front().kind == CP_SWAP,"orphan swap reverse");
    if (move.kind == CP_SWAP)
      require_invariant(move_index == 0 && candidate.moves.size() == 2 &&
          candidate.moves[1].kind == CP_SWAP_REVERSE,"swap lacks reverse bookkeeping");
  }

  for (const auto& move : candidate.moves) {
    UnitId unit_id{move.unit};
    SlotIdx slot = selected.slot_of(unit_id);
    if (move.kind == CP_SWAP_REVERSE) {
      set_swap_reverse_score(child,move);
      continue;
    }
    if (move.kind == CP_FINISH_TEMP || move.kind == CP_BLOCKED) {
      const auto current = child.branch->state().units[unit_id.value];
      child.slot_states[slot] = move.kind == CP_BLOCKED ? SlotState::Blocked : SlotState::Finished;
      child.last_kill[slot] = 0;
      auto kind = move.kind == CP_BLOCKED ? CAI_BLOCKED : CAI_FINISH_TEMP;
      child.assignments.push_back(stationary_assignment(kind,current,tactics.roles[slot],move.score));
      continue;
    }
    int32_t kind = move.kind == CP_MELEE ? MELEE : move.kind == CP_RANGED ? RANGED :
                   move.kind == CP_SWAP ? SWAP : MOVE;
    if (!apply_command(child,{kind,unit_id.value,move.tile},slot,move.score)) return false;
    if (move.kind == CP_FORCED) child.move_flags[slot] = MoveFlag::None;
  }

  // Contract: docs/combat-ai-contract.md#restart.
  // CP makeNextAssignments appends RESTART after both actions. Preserve atomic
  // order, but record the first actual observer rather than an arbitrary anchor.
  if (child.restart_source) {
    UnitId observer = *child.restart_source;
    SlotIdx slot = selected.slot_of(observer);
    const auto current = child.branch->state().units[observer.value];
    auto restart = stationary_assignment(CAI_RESTART,current,tactics.roles[slot]);
    restart.from_tile = CAI_NO_ID;
    restart.to_tile = CAI_NO_ID;
    restart.remaining_moves = 0;
    child.assignments.push_back(restart);
    child.restart_source.reset();
  }
  return true;
}

void SearchWorkspace::clear() {
  positions.clear();
  open.clear();
  equivalent.clear();
}
namespace {
constexpr int32_t kBreadthFirstGenerations = 3; // CP TACTSIM_BREADTH_FIRST_GENERATIONS.
class Search {
  SearchWorkspace& work;
  const CaiConfig& config;
  const CaiSearchRequest& request;
  std::unique_ptr<RulesAdapter> adapter;
  std::unique_ptr<CpTactics> tactics;
  SelectedUnits selected;
  SlotValues<int32_t> roles;
  UnitValues<CaiUnit> initial_units;
  std::vector<int32_t> initial_enemy_ids;
  CaiSearchStats stats{};
  bool depth_first = false;
  bool exhausted = false;
  int32_t best_position = -1;
  int32_t best_score = std::numeric_limits<int32_t>::min();

  std::unique_ptr<TacticalPosition> copy(const TacticalPosition& parent) {
    require_invariant(!parent.restart_source,"queued parent has an unflushed contact");
    auto child = std::make_unique<TacticalPosition>();
    child->branch = parent.branch->clone();
    child->assignments = parent.assignments;
    child->plots = parent.plots;
    child->slot_states = parent.slot_states;
    child->move_flags = parent.move_flags;
    child->last_kill = parent.last_kill;
    child->touched = parent.touched;
    child->killed = parent.killed;
    child->attack_score = parent.attack_score;
    child->discovered = parent.discovered;
    child->generation = parent.generation + 1;
    return child;
  }
  std::vector<int32_t> key(const TacticalPosition& position, const TacticalView& view) const {
    return search_position_key(position,view,request.side,selected);
  }
  bool heap_compare(int32_t left, int32_t right) const {
    const auto& a = *work.positions[left];
    const auto& b = *work.positions[right];
    if (a.generation != b.generation)
      return depth_first ? a.generation < b.generation : a.generation > b.generation;
    if (a.rank != b.rank) return a.rank < b.rank;
    return left > right;
  }
  void record_completion(int32_t position_index, int32_t score) {
    // Contract: docs/combat-ai-contract.md#search-budgets.
    ++stats.completed;
    if (best_position < 0 || score > best_score) {
      best_position = position_index;
      best_score = score;
    }
    // max_completed is a search budget too. A valid best plan may therefore
    // accompany BUDGET_EXHAUSTED, even when this completion empties the heap.
    if (stats.completed >= config.max_completed) exhausted = true;
  }
  Plan result() const {
    Plan plan{};
    plan.result.reason = exhausted ? CAI_BUDGET_EXHAUSTED : best_position < 0 ? CAI_NO_PLAN : CAI_COMPLETED;
    plan.result.stats = stats;
    if (best_position >= 0) plan.assignments = work.positions[best_position]->assignments;
    plan.result.assignment_count = int32_t(plan.assignments.size());
    return plan;
  }
 public:
  Search(SearchWorkspace& workspace, const CaiConfig& settings, const CaiSearchRequest& input)
      : work(workspace), config(settings), request(input), adapter(make_rules_adapter(*input.state,input.side)),
        selected(input.selected_units,input.selected_count,input.state->unit_count) {}
  Plan run() {
    work.clear();
    ContactMemory memory{request.contacts.turn,request.contacts.side,{}};
    for (int32_t contact_index = 0; contact_index < request.contacts.known_enemy_count; ++contact_index)
      memory.known_enemies.push_back({request.contacts.known_enemies[contact_index].id,
                                     request.contacts.known_enemies[contact_index].last_seen_own_turn});
    auto root = std::make_unique<TacticalPosition>();
    root->branch = adapter->root(*request.state,memory);
    auto initial = TacticalView(*root->branch,request.side);
    initial_units = UnitValues<CaiUnit>{initial.units};
    for (int32_t slot_index = 0; slot_index < int32_t(selected.size()); ++slot_index) {
      SlotIdx slot{slot_index};
      UnitId unit_id = selected.unit_at(slot);
      const auto& unit = initial_units[unit_id];
      root->slot_states.push_back(unit.moves <= 0 || unit.waiting ? SlotState::ExcludedAtRoot : SlotState::Pending);
      root->move_flags.push_back(MoveFlag::None);
      root->last_kill.push_back(0);
      root->touched.push_back(0);
    }
    tactics = std::make_unique<CpTactics>(*adapter,*root->branch,config,request);
    tactics->initialize(*root);
    roles = tactics->roles;
    initial_enemy_ids = tactics->initial_enemies;
    auto board = tactics->board(*root);
    for (int32_t slot_index = 0; slot_index < int32_t(selected.size()); ++slot_index) {
      SlotIdx slot{slot_index};
      UnitId unit_id = selected.unit_at(slot);
      const auto& unit = initial_units[unit_id];
      int32_t score = tactics->move_score(*root,board,slot,unit.tile,0,MoveEvaluation::Initial).move.score;
      root->assignments.push_back(stationary_assignment(CAI_INITIAL,unit,roles[slot],score));
      if (root->slot_states[slot] == SlotState::ExcludedAtRoot)
        root->assignments.push_back(stationary_assignment(CAI_BLOCKED,unit,roles[slot]));
    }
    work.equivalent.emplace(key(*root,initial),std::make_pair(0,0));
    work.positions.push_back(std::move(root));
    stats.generated = 1;
    stats.peak_positions = 1;

    // Terminal children are evaluated before enqueueing. Only the root needs
    // this separate check; every node subsequently popped is non-terminal.
    if (terminal_position(*work.positions.front(),initial_enemy_ids)) {
      ++stats.checked;
      auto evaluation = tactics->evaluate_final_position(*work.positions.front());
      if (evaluation.accepted) {
        finalize_assignments(*work.positions.front(),evaluation);
        record_completion(0,evaluation.score);
      } else ++stats.pruned;
      return result();
    }
    work.open.push_back(0);
    CandidateExecutor executor{*adapter,*tactics,selected,request.side};
    auto compare = [&](int32_t left, int32_t right) { return heap_compare(left,right); };
    while (!work.open.empty() && !exhausted) {
      if (!depth_first && work.positions[work.open.front()]->generation > kBreadthFirstGenerations) {
        depth_first = true;
        std::make_heap(work.open.begin(),work.open.end(),compare);
      }
      std::pop_heap(work.open.begin(),work.open.end(),compare);
      int32_t current_index = work.open.back();
      work.open.pop_back();
      const auto& parent = *work.positions[current_index];
      assert(!terminal_position(parent,initial_enemy_ids));
      auto current_view = TacticalView(*parent.branch,request.side);
      auto equivalent = work.equivalent.find(key(parent,current_view));
      if (equivalent != work.equivalent.end() && equivalent->second.second != current_index) {
        ++stats.pruned;
        continue;
      }
      ++stats.checked;
      if (stats.generated >= config.position_capacity) {
        exhausted = true;
        break;
      }
      auto choices = tactics->candidates(parent,depth_first);
      int32_t children = 0;
      for (const auto& candidate : choices) {
        if (children >= (depth_first ? 1 : config.max_branches)) break;
        if (stats.generated >= config.position_capacity) {
          exhausted = true;
          break;
        }
        auto child = copy(parent);
        if (!executor.apply(*child,candidate)) {
          ++stats.pruned;
          continue;
        }
        // Contract: docs/combat-ai-contract.md#candidate-order.
        child->rank = parent.rank;
        for (const auto& move : candidate.moves)
          if (move.kind != CP_BLOCKED && move.kind != CP_FINISH_TEMP) child->rank += move.score;
        auto next = TacticalView(*child->branch,request.side);
        auto identity = key(*child,next);
        auto found = work.equivalent.find(identity);
        if (found != work.equivalent.end() && found->second.first >= child->attack_score) {
          ++stats.pruned;
          continue;
        }
        bool finished = terminal_position(*child,initial_enemy_ids);
        FinalEvaluation evaluation;
        if (finished) {
          evaluation = tactics->evaluate_final_position(*child);
          if (!evaluation.accepted) {
            ++stats.pruned;
            continue;
          }
          finalize_assignments(*child,evaluation);
        }
        int32_t child_index = int32_t(work.positions.size());
        work.equivalent[std::move(identity)] = {child->attack_score,child_index};
        work.positions.push_back(std::move(child));
        ++stats.generated;
        ++children;
        // Peak retained pool occupancy. With the current monotonic pool it
        // equals generated; temporary rejected children are outside this pool.
        stats.peak_positions = std::max(stats.peak_positions,int32_t(work.positions.size()));
        if (finished) {
          record_completion(child_index,evaluation.score);
          if (exhausted) break;
        } else {
          work.open.push_back(child_index);
          std::push_heap(work.open.begin(),work.open.end(),compare);
        }
      }
    }
    return result();
  }
};
} // namespace
Plan find_assignments(SearchWorkspace& workspace, const CaiConfig& config, const CaiSearchRequest& request) {
  return Search(workspace,config,request).run();
}
} // namespace combat_ai
