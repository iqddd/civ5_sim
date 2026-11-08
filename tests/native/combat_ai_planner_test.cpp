// End-to-end local-search properties. Replay uses the rules adapter, not the
// planner's candidate/scoring implementation, and verifies the resulting state.
#include "../../native/combat_ai/planner.h"
#include "../../native/combat_ai/cp_tactical.h"
#include "../../native/combat_state.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>
#include <cstring>
#include <type_traits>

using namespace combat_ai;
using namespace combat_rules;
#define CHECK(x) do { if (!(x)) throw std::runtime_error( \
  std::string(__func__) + ":" + std::to_string(__LINE__) + ": " #x); } while (false)

bool diagnostic_parity = false;

struct Fixture {
  int w = 7, h = 5, fog = 0, parity = 0, wrap = 0, turn = 0;
  std::vector<CaiTile> tiles = std::vector<CaiTile>(35, CaiTile{});
  std::vector<CaiUnit> units;
  std::vector<int32_t> visibility, selected;
  CaiBattleState state{};
  CaiConfig config = cai_default_config();
  std::unique_ptr<RulesAdapter> adapter;
  std::unique_ptr<Branch> root;
  Fixture() { config.position_capacity = 1000; config.max_completed = 80; }
  void add(int type, int side, int tile, int hp = 100) {
    int id = int(units.size());
    units.push_back({id,type,side,tile,hp,moves[type],0,0,0,0});
    if (side == 0) selected.push_back(id);
  }
  void corridor(std::initializer_list<int> passable) {
    for (auto& tile : tiles) tile.elevation = 2;
    for (int tile : passable) tiles[tile].elevation = 0;
  }
  void start() {
    std::vector<Tile> ts;
    std::vector<Unit> us;
    for (auto t : tiles) ts.push_back({t.terrain,t.elevation,t.feature,t.rivers});
    for (auto u : units) us.push_back({u.type,u.side,u.tile,u.hp,u.moves,u.attacked,u.acted,u.fortified,u.waiting,
                                      0,u.damage_taken_last_turn});
    auto geometry = std::make_shared<Geometry>(w,h,parity,wrap,ts.data());
    State physical(geometry); physical.units = us; physical.fog = fog; physical.rebuild();
    visibility.resize(tiles.size());
    for (int i = 0; i < int(tiles.size()); ++i) visibility[i] = physical.known(0,i);
    state = {w,h,parity,wrap,tiles.data(),int32_t(tiles.size()),units.data(),int32_t(units.size()),
             visibility.data(),int32_t(visibility.size()),0,turn,fog};
    adapter = make_rules_adapter(state,0); root = adapter->root(state,{});
  }
  Plan search(int target) {
    SearchWorkspace work;
    CaiSearchRequest request{&state,0,turn,target,selected.data(),int32_t(selected.size()),{0,-1,nullptr,0}};
    auto plan = find_assignments(work,config,request);
    if (diagnostic_parity) {
      SearchWorkspace traced;
      traced.trace.enabled = true;
      auto repeated = find_assignments(traced,config,request);
      CHECK(std::memcmp(&plan.result,&repeated.result,sizeof(plan.result)) == 0);
      CHECK(plan.assignments.size() == repeated.assignments.size());
      for (size_t i = 0; i < plan.assignments.size(); ++i)
        CHECK(std::memcmp(&plan.assignments[i],&repeated.assignments[i],sizeof(CaiAssignment)) == 0);
      CHECK(traced.trace.data.find("search_result") != std::string::npos);
      CHECK(!traced.trace.truncated);
    }
    return plan;
  }
  Plan support(int target, int mode) {
    CaiSearchRequest request{&state,0,turn,target,selected.data(),int32_t(selected.size()),{0,-1,nullptr,0}};
    return support_assignments(request,mode);
  }
  std::unique_ptr<Branch> replay(const Plan& plan) {
    auto branch = root->clone();
    std::vector<bool> finished(units.size(),false);
    for (auto a : plan.assignments) {
      CHECK(std::find(selected.begin(), selected.end(), a.unit) != selected.end());
      if (a.kind == CAI_INITIAL || a.kind == CAI_RESTART || a.kind == CAI_SWAP_REVERSE || a.kind == CAI_BLOCKED) continue;
      if (a.kind == CAI_FINISH) { CHECK(!finished[a.unit]); finished[a.unit] = true; continue; }
      CHECK(a.kind != CAI_FINISH_TEMP);
      CHECK(!finished[a.unit]);
      auto before = branch->state();
      CHECK(before.units[a.unit].tile == a.from_tile);
      int kind = a.kind == CAI_MOVE ? MOVE : a.kind == CAI_MELEE ? MELEE :
                 a.kind == CAI_RANGED ? RANGED : SWAP;
      auto transition = adapter->apply(*branch,{kind,a.unit,a.command_target});
      CHECK(transition.accepted);
      CHECK(transition.result.damage == a.damage);
      CHECK(transition.result.retaliation == a.self_damage);
      auto after = branch->state();
      CHECK(after.units[a.unit].moves == a.remaining_moves);
      if (a.killed_unit >= 0) CHECK(after.units[a.killed_unit].hp == 0);
    }
    return branch;
  }
};

void attacks() {
  Fixture f; f.add(1,0,16); f.add(1,0,9); f.add(0,1,18,50); f.start();
  auto p = f.search(18); CHECK(!p.assignments.empty());
  auto after = f.replay(p);
  CHECK(after->state().units[2].hp == 0);
  int shots = 0;
  for (auto a : p.assignments) if (a.kind == CAI_RANGED) { CHECK(a.to_tile == 18); ++shots; }
  CHECK(shots == 2);
  Fixture finish; finish.add(0,0,16); finish.add(0,1,17,10); finish.start();
  auto kill = finish.search(17); CHECK(!kill.assignments.empty());
  CHECK(finish.replay(kill)->state().units[1].hp == 0);
  CHECK(std::any_of(kill.assignments.begin(),kill.assignments.end(),[](auto a) { return a.unit == 0 && a.killed_unit == 1; }));
}
void swaps_and_freeing() {
  Fixture f; f.add(1,0,16,50); f.add(3,0,15); f.add(0,1,17,50);
  f.corridor({15,16,17}); f.start();
  auto plan = f.search(17); CHECK(!plan.assignments.empty());
  CHECK(std::any_of(plan.assignments.begin(),plan.assignments.end(),[](auto a) { return a.kind == CAI_SWAP; }));
  CHECK(f.replay(plan)->state().units[2].hp == 0);
  Fixture free; free.add(1,0,16); free.add(3,0,15); free.add(0,1,17,50);
  free.corridor({9,15,16,17}); free.start();
  auto p = free.search(17); CHECK(!p.assignments.empty());
  CHECK(free.replay(p)->state().units[2].hp == 0);
  // An omitted friendly remains an obstacle and is never moved by a swap.
  f.selected = {1};
  auto blocked = f.search(17);
  CHECK(blocked.assignments.empty());
}
void risk() {
  Fixture f; f.add(1,0,16,10); f.add(0,1,17,1); f.add(3,1,18);
  f.corridor({16,17,18}); f.start();
  auto model = make_danger_model(*f.adapter);
  TacticalView v(*f.root,0);
  auto score = score_turn_end(f.config,*f.adapter,*model,*f.root,v,0,false,false,false,1.f);
  CHECK(!score.acceptable); CHECK(score.score < 0);
  f.config.risk_budget_per_kill = 0;
  CHECK(f.search(17).assignments.empty());
  f.config.risk_budget_per_kill = 1;
  auto trade = f.search(17); CHECK(!trade.assignments.empty());
  CHECK(f.replay(trade)->state().units[1].hp == 0);
  CHECK(std::any_of(trade.assignments.begin(),trade.assignments.end(),[](auto a) {
    return a.kind == CAI_FINISH && a.score < 60;
  }));
  f.config.risk_budget_per_kill = 0; f.config.risk_budget_base = 1;
  CHECK(!f.search(17).assignments.empty());
}
void cover_and_retreat() {
  Fixture f; f.add(1,0,15,35); f.add(0,0,16); f.add(3,1,17); f.start();
  auto model = make_danger_model(*f.adapter);
  TacticalView screened(*f.root,0);
  auto cover = score_turn_end(f.config,*f.adapter,*model,*f.root,screened,0,false,false,true,1.f);
  auto exposed = f.root->clone();
  CHECK(f.adapter->apply(*exposed,{MOVE,1,9}).accepted);
  TacticalView open(*exposed,0);
  auto no_cover = score_turn_end(f.config,*f.adapter,*model,*exposed,open,0,false,false,true,1.f);
  CHECK(cover.score > no_cover.score);
  Fixture horse; horse.add(4,0,16); horse.add(0,1,17,1); horse.add(3,1,18);
  horse.start(); auto p = horse.search(17); CHECK(!p.assignments.empty()); horse.replay(p);
  bool hit = false, retreat = false;
  for (auto a : p.assignments) {
    if (a.kind == CAI_MELEE) { hit = true; CHECK(a.remaining_moves > 0); }
    if (hit && a.kind == CAI_MOVE) retreat = true;
  }
  // Exact move-after-attack state is available even if staying is equally good.
  CHECK(hit); (void)retreat;
}
void budgets_and_determinism() {
  Fixture f; f.add(1,0,16); f.add(1,0,9); f.add(0,1,18,50); f.start();
  auto a = f.search(18), b = f.search(18);
  CHECK(a.result.reason == b.result.reason);
  CHECK(a.result.stats.generated == b.result.stats.generated);
  CHECK(a.result.stats.pruned == b.result.stats.pruned);
  CHECK(a.result.stats.pruned > 0);
  CHECK(a.assignments.size() == b.assignments.size());
  for (size_t i=0;i<a.assignments.size();++i) {
    const auto& x=a.assignments[i]; const auto& y=b.assignments[i];
    CHECK(x.kind==y.kind && x.unit==y.unit && x.to_tile==y.to_tile && x.score==y.score);
  }
  f.config.position_capacity = 1;
  auto tiny=f.search(18); CHECK(tiny.result.reason==CAI_BUDGET_EXHAUSTED);
  CHECK(tiny.assignments.empty()); CHECK(tiny.result.stats.generated==1);
  f.config.position_capacity = 1000; f.config.max_completed = 1;
  auto limited=f.search(18); CHECK(limited.result.reason==CAI_BUDGET_EXHAUSTED);
  CHECK(limited.result.stats.completed==1); CHECK(!limited.assignments.empty()); f.replay(limited);
  f.config.aggression=CAI_NONE;
  auto peaceful=f.search(18);
  for (auto x:peaceful.assignments) CHECK(x.kind!=CAI_MELEE && x.kind!=CAI_RANGED);
}
void contact() {
  Fixture f; f.w=12; f.h=5; f.tiles.resize(60); f.fog=1;
  f.add(4,0,24); f.add(0,1,28,1); f.start();
  CHECK(!f.visibility[28]);
  auto p=f.search(28); CHECK(!p.assignments.empty());
  bool restart=false, continued=false;
  for(auto a:p.assignments) {
    if(a.kind==CAI_RESTART) restart=true;
    if(restart && (a.kind==CAI_MOVE || a.kind==CAI_MELEE)) continued=true;
  }
  CHECK(restart); CHECK(continued); f.replay(p);
}
void random_replay() {
  std::mt19937 random(42);
  for (int test = 0; test < 32; ++test) {
    Fixture f; f.w=12; f.h=8; f.tiles.resize(96);
    f.fog=test&1; f.parity=(test>>1)&1; f.wrap=(test>>2)&1;
    f.turn=test*2;
    f.config.position_capacity=160; f.config.max_completed=12;
    std::vector<int> places(96);
    std::iota(places.begin(),places.end(),0);
    std::shuffle(places.begin(),places.end(),random);
    for (int i=0;i<10;++i) f.add(random()%5,i<4?0:1,places[i],20+random()%81);
    f.start(); auto p=f.search(places[4]);
    CHECK(p.result.stats.generated<=f.config.position_capacity);
    CHECK(p.result.stats.completed<=f.config.max_completed);
    if (!p.assignments.empty()) f.replay(p);
  }
}
void scale_and_last_tile() {
  Fixture f; f.w=16; f.h=16; f.tiles.resize(256);
  f.add(0,0,253);
  for (int i=1;i<14;++i) f.add(1,0,223+i);
  f.add(0,1,254,1);
  for (int i=0;i<19;++i) f.add(0,1,i);
  f.selected.resize(13); // Fourteen own, twenty enemy, thirteen selected.
  f.config.max_completed=1;
  f.start(); auto p=f.search(254);
  CHECK(!p.assignments.empty());
  CHECK(f.replay(p)->state().units[14].hp==0);
  for(auto a:p.assignments) CHECK(a.unit>=0 && a.unit<13);
  Fixture last; last.w=16; last.h=16; last.tiles.resize(256);
  last.add(0,0,254); last.add(0,1,255,1); last.start();
  auto kill=last.search(255); CHECK(!kill.assignments.empty());
  CHECK(last.replay(kill)->state().units[0].tile==255);
}
void support_healing() {
  Fixture safe; safe.add(0,0,0,25); safe.add(0,1,34); safe.start();
  auto p = safe.support(0,CAI_HEAL_BEFORE);
  CHECK(p.assignments.size()==1 && p.assignments[0].kind==CAI_FINISH);
  auto after = safe.replay(p);
  CHECK(safe.adapter->apply(*after,{END_TURN,-1,-1}).accepted);
  CHECK(safe.adapter->apply(*after,{END_TURN,-1,-1}).accepted);
  CHECK(after->state().units[0].hp==35);
  Fixture moved; moved.add(0,0,0,25); moved.add(0,1,34); moved.units[0].acted=1; moved.start();
  auto waited=moved.replay(moved.support(0,CAI_HEAL_AFTER));
  CHECK(moved.adapter->apply(*waited,{END_TURN,-1,-1}).accepted);
  CHECK(moved.adapter->apply(*waited,{END_TURN,-1,-1}).accepted);
  CHECK(waited->state().units[0].hp==25);
  Fixture flee; flee.add(1,0,16,10); flee.add(0,1,17); flee.start();
  auto danger=make_danger_model(*flee.adapter);
  int before=danger->incoming(*flee.root,0,16);
  auto retreat=flee.support(16,CAI_HEAL_BEFORE);
  auto escaped=flee.replay(retreat);
  CHECK(escaped->state().units[0].tile!=16);
  CHECK(danger->incoming(*escaped,0,escaped->state().units[0].tile)<before);
  Fixture ready; ready.add(0,0,16,25); ready.add(0,1,17); ready.start();
  CHECK(ready.support(16,CAI_HEAL_BEFORE).assignments.empty()); // Can still attack.
}
void support_approach() {
  Fixture detour; detour.add(0,0,14); detour.add(0,1,20);
  detour.corridor({14,7,8,9,16,17,18,19,20}); detour.start();
  auto plan=detour.support(20,CAI_APPROACH);
  CHECK(!plan.assignments.empty());
  CHECK(detour.replay(plan)->state().units[0].tile!=14);
  Fixture blocked; blocked.add(0,0,14); blocked.add(0,0,15); blocked.add(0,1,20);
  blocked.selected={0}; blocked.corridor({14,15,16,17,18,19,20}); blocked.start();
  auto p=blocked.support(20,CAI_APPROACH);
  if(!p.assignments.empty()) CHECK(blocked.replay(p)->state().units[1].tile==15);
  Fixture last; last.w=16; last.h=16; last.tiles.resize(256);
  last.add(0,0,254); last.add(0,1,0); last.start();
  auto end=last.support(255,CAI_APPROACH);
  CHECK(last.replay(end)->state().units[0].tile==255);
  CHECK(last.support(254,CAI_APPROACH).assignments.empty());
}
struct TacticalFixture {
  TacticalPosition position;
  std::unique_ptr<CpTactics> tactics;
  TacticalFixture(Fixture& f, int target) {
    position.branch=f.root->clone();
    position.slot_states.assign(f.selected.size(),SlotState::Pending);
    position.move_flags.assign(f.selected.size(),MoveFlag::None);
    position.last_kill.assign(f.selected.size(),0);
    position.touched.assign(f.selected.size(),0);
    CaiSearchRequest request{&f.state,0,f.turn,target,f.selected.data(),int32_t(f.selected.size()),{0,-1,nullptr,0}};
    tactics=std::make_unique<CpTactics>(*f.adapter,*f.root,f.config,request);
    tactics->initialize(position);
  }
};
void check_root_exclusion(const Fixture& fixture, TacticalFixture& context, SlotIdx slot) {
  int unit=fixture.selected[slot.value];
  CHECK(context.position.slot_states[slot]==SlotState::ExcludedAtRoot);
  CHECK(std::find(context.tactics->active_order.begin(),context.tactics->active_order.end(),slot)==
        context.tactics->active_order.end());
  auto board=context.tactics->board(context.position);
  CHECK(context.tactics->preferred(context.position,board,slot,5).empty());
  for (bool depth_first : {false,true})
    for (const auto& candidate : context.tactics->candidates(context.position,depth_first))
      for (const auto& move : candidate.moves) CHECK(move.unit!=unit);
  auto evaluation=context.tactics->evaluate_final_position(context.position);
  CHECK(std::none_of(evaluation.finishes.begin(),evaluation.finishes.end(),
      [unit](const CaiAssignment& record) { return record.unit==unit; }));
}
void cp_area_and_preferences() {
  // GCC can silently select the wrong column when clamp and conditional
  // array selection share one expression. These scores must be table lookups.
  CHECK(formation_score(CAI_SECOND_LINE,2,false)==30);
  CHECK(formation_score(CAI_FIRST_LINE,3,false)==30);
  CHECK(formation_score(CAI_THIRD_LINE,2,false)==20);
  CHECK(formation_score(CAI_SECOND_LINE,-1,false)==80);
  CHECK(formation_score(CAI_SECOND_LINE,8,false)==10);
  CHECK(formation_score(CAI_SECOND_LINE,2,true)==120);
  Fixture distant; distant.w=16; distant.h=1; distant.tiles.resize(16);
  distant.add(4,0,0); distant.add(0,1,9); distant.start();
  TacticalFixture f(distant,9);
  auto board=f.tactics->board(f.position);
  CHECK(board.plots[0].valid); CHECK(!board.plots[2].valid);
  auto preferred=f.tactics->preferred(f.position,board,SlotIdx{0},9);
  CHECK(std::any_of(preferred.begin(),preferred.end(),[](CpMove m) { return m.kind==CP_MOVE && m.tile==4; }));
  CHECK(preferred.back().kind==CP_BLOCKED && preferred.back().score==-10);
  for (auto m : preferred) if (m.kind==CP_MOVE) CHECK(distant.adapter->distance(m.tile,9)<=6);
  // Starting outside radius six does not exclude a participant with a useful
  // reachable destination. Conversely, all far intents produce only BLOCKED.
  TacticalFixture far(distant,15);
  auto blocked=far.tactics->preferred(far.position,far.tactics->board(far.position),SlotIdx{0},9);
  CHECK(blocked.size()==1 && blocked[0].kind==CP_BLOCKED);
  Fixture ties; ties.add(0,0,17); ties.start();
  TacticalFixture equal(ties,17);
  auto top=equal.tactics->preferred(equal.position,equal.tactics->board(equal.position),SlotIdx{0},2);
  CHECK(top.size()>3); // More than two actions plus BLOCKED: cutoff ties survive.
  int cutoff=top[1].score;
  for (size_t i=2; i+1<top.size(); ++i) CHECK(top[i].score==cutoff);
  Fixture visibility; visibility.w=16; visibility.h=5; visibility.tiles.resize(80); visibility.fog=1;
  visibility.add(0,0,32); visibility.add(0,1,47); visibility.start();
  TacticalFixture fog(visibility,47);
  CHECK(!fog.position.plots[47]);
  CHECK(!fog.tactics->board(fog.position).plots[47].valid);
  // Root-only safe count and fixed danger index are not overwritten by a child.
  auto safe=fog.tactics->safe;
  auto child=fog.position.branch->clone();
  CHECK(visibility.adapter->apply(*child,{MOVE,0,33}).accepted);
  CHECK(fog.tactics->safe==safe);
}
void cp_blocker_combinations() {
  Fixture corridor; corridor.config.choices_per_unit=9;
  corridor.add(4,0,14); corridor.add(0,0,16); corridor.add(1,1,17,1);
  corridor.corridor({14,15,16,17,18}); corridor.start();
  TacticalFixture f(corridor,17);
  auto before=f.position.branch->state();
  auto old_units=std::vector<CaiUnit>(before.units,before.units+before.unit_count);
  auto board=f.tactics->board(f.position);
  auto choices=f.tactics->preferred(f.position,board,SlotIdx{0},9);
  CHECK(std::any_of(choices.begin(),choices.end(),[](CpMove m) { return m.kind==CP_MOVE && m.tile==16; }));
  auto after=f.position.branch->state();
  for (int id=0; id<after.unit_count; ++id)
    CHECK(after.units[id].hp==old_units[id].hp && after.units[id].tile==old_units[id].tile);
  CHECK(board.view.occupied[16]==1); CHECK(board.plots[15].friends==2);
  bool freeing=false;
  for (auto combo : f.tactics->candidates(f.position,false)) if (combo.moves.size()==2 && combo.moves[0].kind==CP_FORCED) {
    auto a=combo.moves[0], b=combo.moves[1];
    CHECK(old_units[a.unit].tile==b.tile); CHECK(a.unit!=b.unit);
    CHECK(board.view.occupied[a.tile]<0);
    if (b.unit==0 && b.tile==16) {
      CHECK(corridor.adapter->distance(old_units[b.unit].tile,b.tile)==2); freeing=true;
    }
  }
  CHECK(freeing); // A real blocker, even when the follower is not adjacent.
  corridor.selected={0};
  TacticalFixture omitted(corridor,17);
  auto blocked=omitted.tactics->preferred(omitted.position,omitted.tactics->board(omitted.position),SlotIdx{0},9);
  CHECK(std::none_of(blocked.begin(),blocked.end(),[](CpMove m) { return m.kind==CP_MOVE && m.tile==16; }));
  Fixture swap; swap.add(1,0,16,50); swap.add(3,0,15); swap.add(0,1,17,50);
  swap.corridor({15,16,17}); swap.start(); TacticalFixture reciprocal(swap,17);
  auto pairs=reciprocal.tactics->candidates(reciprocal.position,false);
  CHECK(std::any_of(pairs.begin(),pairs.end(),[](const CpCombo& c) {
    return c.moves.size()==2 && c.moves[0].kind==CP_SWAP && c.moves[1].kind==CP_SWAP_REVERSE;
  }));
  // BLOCKED does not promise cover at turn end; FINISH_TEMP does.
  f.position.slot_states[SlotIdx{1}]=SlotState::Blocked;
  CHECK(!f.tactics->board(f.position).plots[16].end);
  f.position.slot_states[SlotIdx{1}]=SlotState::Finished;
  auto ended=f.tactics->board(f.position);
  CHECK(ended.plots[16].end && ended.plots[15].friends_end==1);
}
void cp_predicted_kill_end_danger() {
  // A wounded shooter can remove the only source of danger. The victim must
  // not reject its own killing shot before candidate execution (338 report).
  Fixture f; f.config.risk_budget_per_kill=0;
  f.add(1,0,16,22); f.add(3,1,17,4); f.start();
  TacticalFixture context(f,17); SlotIdx slot{0};
  CHECK(context.tactics->safe[slot]>0);
  CHECK(context.tactics->danger(0,16,0,{})>22);
  CHECK(context.tactics->danger(0,16,0,{1})==0);
  CHECK(context.position.slot_states[slot]==SlotState::Pending);
  auto board=context.tactics->board(context.position);
  auto choices=context.tactics->preferred(context.position,board,slot,40);
  CHECK(std::any_of(choices.begin(),choices.end(),[](CpMove move) {
    return move.kind==CP_RANGED && move.tile==17 && move.kill && move.near_kill==1;
  }));
  CHECK(std::any_of(choices.begin(),choices.end(),[&](CpMove move) {
    return move.kind==CP_MOVE && move.remaining>0 && move.near_kill==1 &&
           context.tactics->danger(0,move.tile,0,{})>22;
  })); // Move+kill must use the restored CP potential-victim output too.
  auto plan=f.search(17);
  auto killed=f.replay(plan);
  CHECK(killed->state().units[0].hp==22 && killed->state().units[1].hp==0);
  CHECK(std::any_of(plan.assignments.begin(),plan.assignments.end(),[](CaiAssignment a) {
    return a.kind==CAI_RANGED && a.killed_unit==1;
  }));
  CHECK(f.root->state().units[1].hp==4 && f.root->state().units[0].moves==120);

  // Removing the proposed victim must retain threats from other enemies.
  Fixture remaining; remaining.config.risk_budget_per_kill=0;
  remaining.add(1,0,16,22); remaining.add(3,1,17,4); remaining.add(3,1,10);
  remaining.start(); TacticalFixture unsafe(remaining,17);
  CHECK(unsafe.tactics->safe[slot]>0);
  CHECK(unsafe.tactics->danger(0,16,0,{1})>22);
  // Admission may already exclude a unit with no acceptable attack. Inspect
  // its preferences separately from that root pruning.
  unsafe.position.slot_states[slot]=SlotState::Pending;
  auto dangerous=unsafe.tactics->preferred(unsafe.position,unsafe.tactics->board(unsafe.position),slot,40);
  CHECK(std::none_of(dangerous.begin(),dangerous.end(),[](CpMove move) {
    return move.kind==CP_RANGED && move.tile==17;
  }));

  // An ordinary nonlethal shot does not remove its target from danger.
  Fixture nonlethal; nonlethal.config.risk_budget_per_kill=0;
  nonlethal.add(1,0,16,22); nonlethal.add(3,1,17); nonlethal.start();
  TacticalFixture ordinary(nonlethal,17);
  ordinary.position.slot_states[slot]=SlotState::Pending;
  auto no_kill=ordinary.tactics->preferred(ordinary.position,ordinary.tactics->board(ordinary.position),slot,40);
  CHECK(std::none_of(no_kill.begin(),no_kill.end(),[](CpMove move) { return move.kind==CP_RANGED; }));

  // CP's near-kill assumption is preliminary only. An actual survivor must
  // still participate in the final check of an executed search branch.
  Fixture near; near.config.risk_budget_per_kill=0;
  near.add(1,0,16,22); near.add(3,1,17,28); near.start();
  TacticalFixture almost(near,17);
  CHECK(almost.tactics->danger(0,16,0,{})>22);
  auto near_choices=almost.tactics->preferred(almost.position,almost.tactics->board(almost.position),slot,40);
  CHECK(std::any_of(near_choices.begin(),near_choices.end(),[](CpMove move) {
    return move.kind==CP_RANGED && !move.kill && move.near_kill==1;
  }));
  auto near_plan=near.search(17);
  CHECK(std::none_of(near_plan.assignments.begin(),near_plan.assignments.end(),[](CaiAssignment a) {
    return a.kind==CAI_RANGED && a.from_tile==16;
  }));
  near.replay(near_plan);
}
void cp_forecast_and_danger() {
  // Cross-check the pure strength query against the unchanged physical
  // kernel, across types/wounds/terrain. Uncapped preview damage is separate.
  for (int a=0; a<5; ++a) for (int d=0; d<5; ++d) for (int hp : {1,50,100}) {
    Fixture compare;
    compare.tiles[16].elevation=1;
    compare.tiles[15].rivers=1<<1;
    compare.add(a,0,15,hp); compare.add(d,1,16,100-hp/2);
    if (d!=2 && d!=4) compare.units[1].fortified=1;
    compare.start();
    auto pure=compare.adapter->forecast(*compare.root,0,15,1,16,0);
    auto physical=compare.adapter->preview(*compare.root,{combat_rules::ranged[a]?RANGED:MELEE,0,16});
    CHECK(physical.code==0 && pure.code==0);
    CHECK(pure.attacker_strength==physical.attacker_strength);
    CHECK(pure.defender_strength==physical.defender_strength);
  }
  Fixture f; f.add(0,0,15); f.add(0,1,16); f.add(1,0,8); f.start();
  auto melee=f.adapter->forecast(*f.root,0,15,1,16,0);
  auto extra=f.adapter->forecast(*f.root,0,15,1,16,20);
  // Contract: extra-defender-damage. Fix CP's wrong melee argument: only
  // the defender weakens, just as for the ranged forecast below.
  CHECK(extra.attacker_strength==melee.attacker_strength);
  CHECK(extra.defender_strength<melee.defender_strength);
  auto shot=f.adapter->forecast(*f.root,2,8,1,16,0);
  auto wounded=f.adapter->forecast(*f.root,2,8,1,16,20);
  CHECK(wounded.attacker_strength==shot.attacker_strength);
  CHECK(wounded.defender_strength<shot.defender_strength);
  CHECK(f.root->state().units[1].hp==100);
  Fixture line; line.add(0,0,15); line.add(0,0,16); line.add(4,1,14);
  line.corridor({14,15,16,17,18,19,20}); line.start();
  auto covered=line.adapter->threatened_tiles(*line.root,2);
  CHECK(std::find(covered.begin(),covered.end(),18)!=covered.end());
  CHECK(std::find(covered.begin(),covered.end(),20)==covered.end());
  TacticalFixture danger(line,14);
  CHECK(danger.tactics->danger(0,18,0,{})>0);
  CHECK(danger.tactics->danger(0,18,0,{2})==0);
  CHECK(danger.tactics->danger(0,18,0,{})>0); // The killed-set cache is separate.
}
void cp_history_and_admission() {
  Fixture f; f.w=16; f.h=16; f.tiles.resize(256);
  f.add(0,0,0); f.add(0,1,1); f.start();
  {
    TacticalFixture absent(f,240);
    CHECK(absent.position.slot_states[SlotIdx{0}]==SlotState::Pending);
    CHECK(absent.tactics->roles[SlotIdx{0}]==CAI_FIRST_LINE);
    CHECK(absent.position.plots[0] && absent.position.plots[1]);
  }
  f.units[0].damage_taken_last_turn=100;
  f.start();
  {
    TacticalFixture equality(f,240);
    CHECK(equality.position.slot_states[SlotIdx{0}]==SlotState::Pending); // Strict >, not >=.
  }
  f.units[0].damage_taken_last_turn=101;
  f.start();
  TacticalFixture excluded(f,240);
  check_root_exclusion(f,excluded,SlotIdx{0});
  CHECK(!excluded.position.plots[0] && !excluded.position.plots[1]);
  CHECK(excluded.tactics->initial_enemies.empty());
  // A remote unrelated enemy cannot admit the unit by being added through
  // its neighborhood before admission. Near the real target it is admitted.
  TacticalFixture frontline(f,1);
  CHECK(frontline.position.slot_states[SlotIdx{0}]==SlotState::Pending);
  CHECK(frontline.tactics->roles[SlotIdx{0}]==CAI_THIRD_LINE);
  CHECK(movement_role(f.units[0],100)==CAI_FIRST_LINE);
  CHECK(movement_role(f.units[0],101)==CAI_THIRD_LINE);
  for (int type : {1,2}) {
    Fixture ranged_fixture;
    ranged_fixture.add(type,0,0);
    ranged_fixture.add(0,1,1);
    ranged_fixture.units[0].damage_taken_last_turn=101;
    ranged_fixture.start();
    TacticalFixture second_line(ranged_fixture,1);
    CHECK(second_line.position.slot_states[SlotIdx{0}]==SlotState::Pending);
    CHECK(second_line.tactics->roles[SlotIdx{0}]==CAI_SECOND_LINE);
  }
}
void cp_area_capacity() {
  Fixture f; f.w=48; f.h=24; f.tiles.resize(f.w*f.h);
  for (int y : {2,6,10}) for (int x : {34,38,42,46}) f.add(0,0,y*f.w+x);
  int target=12*f.w+12, horse=12*f.w+21;
  f.add(4,0,horse);
  std::vector<Tile> tiles(f.tiles.size());
  Geometry geometry(f.w,f.h,0,0,tiles.data());
  for (int t=0; t<int(tiles.size()); ++t) if (geometry.distance(target,t)==5) f.add(0,1,t);
  f.start();
  TacticalFixture cap(f,target);
  CHECK(std::accumulate(cap.position.plots.begin(),cap.position.plots.end(),0)==255);
  CHECK(cap.position.plots[horse]);
  auto choices=cap.tactics->preferred(cap.position,cap.tactics->board(cap.position),SlotIdx{12},9);
  CHECK(!choices.empty() && choices.back().kind==CP_BLOCKED);
  CHECK(std::any_of(choices.begin(),choices.end(),[](CpMove m) { return m.kind==CP_MOVE; }));
}
void planner_invariants() {
  static_assert(!std::is_convertible_v<UnitId,SlotIdx>);
  static_assert(!std::is_convertible_v<SlotIdx,UnitId>);
  static_assert(!std::is_convertible_v<UnitId,int32_t>);
  TacticalPosition position;
  position.slot_states.assign(1,SlotState::Pending);
  position.killed={3,9};
  CHECK(!terminal_position(position,{3,4})); // Same count, original #4 survives.
  position.killed={9};
  CHECK(!terminal_position(position,{4}));
  position.killed={3,4,9};
  CHECK(terminal_position(position,{3,4})); // Extra casualties do not prevent completion.
  CHECK(!terminal_position(position,{}));
  position.slot_states[SlotIdx{0}]=SlotState::Finished;
  CHECK(terminal_position(position,{}));

  Fixture f; f.add(0,0,16); f.add(0,0,9); f.selected={0}; f.start();
  TacticalFixture tactical(f,18);
  SelectedUnits selection(f.selected.data(),int32_t(f.selected.size()),int32_t(f.units.size()));
  CandidateExecutor executor(*f.adapter,*tactical.tactics,selection,0);
  for (int bad_id : {-1,1,999}) {
    CpMove bad;
    bad.unit=bad_id;
    bad.tile=15;
    CpCombo candidate{{bad},0};
    bool rejected=false;
    try { executor.apply(tactical.position,candidate); }
    catch (const std::logic_error&) { rejected=true; }
    CHECK(rejected);
    CHECK(tactical.position.assignments.empty());
    CHECK(tactical.position.branch->state().units[0].tile==16);
  }
  CpMove orphan;
  orphan.unit=0;
  orphan.tile=16;
  orphan.kind=CP_SWAP_REVERSE;
  bool rejected=false;
  try { executor.apply(tactical.position,{{orphan},0}); }
  catch (const std::logic_error&) { rejected=true; }
  CHECK(rejected && tactical.position.assignments.empty());

  CpMove unknown_kind;
  unknown_kind.unit=0;
  unknown_kind.tile=15;
  unknown_kind.kind=static_cast<CpMoveKind>(999);
  bool invalid_kind=false;
  try { executor.apply(tactical.position,{{unknown_kind},0}); }
  catch (const std::logic_error&) { invalid_kind=true; }
  CHECK(invalid_kind && tactical.position.assignments.empty());

  CpMove move;
  move.unit=0;
  move.tile=15;
  CHECK(executor.apply(tactical.position,{{move},0}));
  CpMove finish;
  finish.unit=0;
  finish.tile=16; // Deliberately stale proposal; bookkeeping reads the child.
  finish.kind=CP_FINISH_TEMP;
  CHECK(executor.apply(tactical.position,{{finish},0}));
  CHECK(tactical.position.assignments.back().from_tile==15);
  CHECK(tactical.position.assignments.back().to_tile==15);
}
void planner_finalization() {
  Fixture unsafe; unsafe.config.risk_budget_base=0; unsafe.config.risk_budget_per_kill=0;
  unsafe.add(1,0,16,10); unsafe.add(0,1,17); unsafe.start();
  TacticalFixture f(unsafe,17);
  f.position.slot_states[SlotIdx{0}]=SlotState::Finished;
  f.position.touched[SlotIdx{0}]=1;
  f.position.discovered=true; // Ensure danger, not lack of improvement, rejects it.
  f.position.assignments.push_back(stationary_assignment(CAI_FINISH_TEMP,unsafe.units[0],CAI_SECOND_LINE));
  auto before=f.position.assignments;
  auto states=f.position.slot_states;
  auto evaluation=f.tactics->evaluate_final_position(f.position);
  CHECK(!evaluation.accepted);
  CHECK(f.position.slot_states==states);
  CHECK(f.position.assignments.size()==before.size());
  CHECK(std::memcmp(before.data(),f.position.assignments.data(),before.size()*sizeof(CaiAssignment))==0);
  bool rejected=false;
  try { finalize_assignments(f.position,evaluation); }
  catch (const std::logic_error&) { rejected=true; }
  CHECK(rejected);
  CHECK(std::memcmp(before.data(),f.position.assignments.data(),before.size()*sizeof(CaiAssignment))==0);

  Fixture safe; safe.add(1,0,16); safe.add(0,1,17,1); safe.start();
  TacticalFixture accepted(safe,17);
  SelectedUnits selection(safe.selected.data(),int32_t(safe.selected.size()),int32_t(safe.units.size()));
  CandidateExecutor executor(*safe.adapter,*accepted.tactics,selection,0);
  CpMove attack;
  attack.unit=0;
  attack.tile=17;
  attack.kind=CP_RANGED;
  CHECK(executor.apply(accepted.position,{{attack},0}));
  auto actions_before=accepted.position.assignments.size();
  auto valid=accepted.tactics->evaluate_final_position(accepted.position);
  CHECK(valid.accepted && valid.finishes.size()==1);
  CHECK(accepted.position.assignments.size()==actions_before);
  finalize_assignments(accepted.position,valid);
  CHECK(accepted.position.assignments.size()==actions_before+1);
  CHECK(accepted.position.assignments.back().kind==CAI_FINISH);
  auto finalized_count=accepted.position.assignments.size();
  bool repeated=false;
  try { finalize_assignments(accepted.position,valid); }
  catch (const std::logic_error&) { repeated=true; }
  CHECK(repeated && accepted.position.assignments.size()==finalized_count);
}
void planner_root_exclusion() {
  static_assert(int(SlotState::Pending)==0);
  static_assert(int(SlotState::Finished)==1);
  static_assert(int(SlotState::Blocked)==2);
  static_assert(int(SlotState::ExcludedAtRoot)==3);
  for (bool waiting : {false,true}) {
    Fixture f;
    f.add(0,0,8);
    f.add(1,0,28);
    f.add(0,1,29,1);
    if (waiting) f.units[0].waiting=1;
    else f.units[0].moves=0;
    f.start();
    TacticalFixture context(f,29);
    check_root_exclusion(f,context,SlotIdx{0});
    auto plan=f.search(29);
    CHECK(!plan.assignments.empty());
    CHECK(std::count_if(plan.assignments.begin(),plan.assignments.end(),
        [](const CaiAssignment& record) { return record.unit==0 && record.kind==CAI_BLOCKED; })==1);
    CHECK(std::none_of(plan.assignments.begin(),plan.assignments.end(),
        [](const CaiAssignment& record) { return record.unit==0 && record.kind==CAI_FINISH; }));
    CHECK(f.replay(plan)->state().units[2].hp==0);
  }

  // dropSuperfluousUnits can exclude an admitted, physically available unit.
  // Its neighborhood/occupancy must retain the original admission semantics.
  Fixture pruned;
  pruned.w=16; pruned.h=1; pruned.tiles.resize(16);
  pruned.add(0,0,0,39);
  pruned.add(0,1,4);
  pruned.start();
  TacticalFixture context(pruned,4);
  CHECK(pruned.units[0].moves>0 && !pruned.units[0].waiting);
  check_root_exclusion(pruned,context,SlotIdx{0});
  auto board=context.tactics->board(context.position);
  CHECK(board.plots[0].valid && !board.plots[0].blocked && !board.plots[0].end);
  CHECK(terminal_position(context.position,context.tactics->initial_enemies));
}
void planner_blocked_handoff() {
  // Compatibility with CP's handoff assumption, not a safety guarantee.
  Fixture f;
  f.config.risk_budget_base=0;
  f.config.risk_budget_per_kill=0;
  f.add(4,0,8,20);
  f.add(1,0,28);
  f.add(0,1,10);
  f.add(0,1,3);
  f.add(0,1,17);
  f.add(0,1,29,1);
  f.start();
  TacticalFixture context(f,10);
  SelectedUnits selection(f.selected.data(),int32_t(f.selected.size()),int32_t(f.units.size()));
  CandidateExecutor executor(*f.adapter,*context.tactics,selection,0);
  auto apply_generated=[&](CpMoveKind kind, int unit, int tile) {
    auto candidates=context.tactics->candidates(context.position,false);
    auto found=std::find_if(candidates.begin(),candidates.end(),[&](const CpCombo& candidate) {
      return candidate.moves.size()==1 && candidate.moves[0].kind==kind &&
             candidate.moves[0].unit==unit && candidate.moves[0].tile==tile;
    });
    CHECK(found!=candidates.end());
    CHECK(executor.apply(context.position,*found));
    if (kind!=CP_BLOCKED && kind!=CP_FINISH_TEMP) context.position.rank+=found->score;
  };
  apply_generated(CP_MOVE,0,9);
  CHECK(context.position.move_flags[SlotIdx{0}]==MoveFlag::Moved);
  CHECK(context.position.slot_states[SlotIdx{0}]==SlotState::Pending);
  CHECK(context.position.branch->state().units[0].moves==180);
  apply_generated(CP_RANGED,1,29);
  CHECK(context.position.killed==std::vector<int>{5});
  apply_generated(CP_BLOCKED,0,9);
  CHECK(context.position.slot_states[SlotIdx{0}]==SlotState::Blocked);
  const auto& blocked=context.position.assignments.back();
  CHECK(blocked.kind==CAI_BLOCKED && blocked.unit==0 && blocked.to_tile==9);
  CHECK(blocked.remaining_moves==180);
  auto accepted=context.tactics->evaluate_final_position(context.position);
  CHECK(accepted.accepted);
  CHECK(accepted.finishes.size()==1 && accepted.finishes[0].unit==1);

  // Change only bookkeeping: checking the same physical position rejects it.
  context.position.slot_states[SlotIdx{0}]=SlotState::Finished;
  CHECK(!context.tactics->evaluate_final_position(context.position).accepted);
  context.position.slot_states[SlotIdx{0}]=SlotState::Blocked;
  finalize_assignments(context.position,accepted);
  CHECK(std::none_of(context.position.assignments.begin(),context.position.assignments.end(),
      [](const CaiAssignment& record) { return record.unit==0 && record.kind==CAI_FINISH; }));
  Plan plan;
  plan.assignments=context.position.assignments;
  auto after=f.replay(plan);
  CHECK(after->state().units[0].tile==9 && after->state().units[0].moves==180);
  CHECK(after->state().units[5].hp==0);
}
void planner_combo_contacts() {
  // Contract: docs/combat-ai/combat-ai-contract.md#restart and #combo-invariants.
  Fixture f; f.w=10; f.h=7; f.tiles.resize(70); f.fog=1;
  f.add(0,0,46); f.add(4,0,42); f.add(0,1,24); f.start();
  CHECK(!f.visibility[24]);
  TacticalFixture context(f,24);
  SelectedUnits selection(f.selected.data(),int32_t(f.selected.size()),int32_t(f.units.size()));
  CandidateExecutor executor(*f.adapter,*context.tactics,selection,0);
  CpMove freeing;
  freeing.unit=0;
  freeing.tile=47;
  freeing.kind=CP_FORCED;
  CpMove follower;
  follower.unit=1;
  follower.tile=46;
  CHECK(executor.apply(context.position,{{freeing,follower},0}));
  const auto& records=context.position.assignments;
  CHECK(records.size()==3);
  CHECK(records[0].unit==0 && records[0].kind==CAI_MOVE);
  CHECK(records[1].unit==1 && records[1].kind==CAI_MOVE);
  CHECK(records[1].to_tile!=46 && records[1].command_target==46);
  CHECK(context.position.branch->state().visibility[24]);
  CHECK(records[2].kind==CAI_RESTART && records[2].unit==1);
  CHECK(!context.position.restart_source);

  Fixture swap; swap.add(1,0,16,50); swap.add(3,0,15); swap.add(0,1,17,50);
  swap.corridor({15,16,17}); swap.start();
  TacticalFixture reciprocal(swap,17);
  SelectedUnits swapped(swap.selected.data(),int32_t(swap.selected.size()),int32_t(swap.units.size()));
  CandidateExecutor swap_executor(*swap.adapter,*reciprocal.tactics,swapped,0);
  CpMove forward;
  forward.unit=0;
  forward.tile=15;
  forward.kind=CP_SWAP;
  forward.score=11;
  CpMove reverse;
  reverse.unit=1;
  reverse.tile=16;
  reverse.kind=CP_SWAP_REVERSE;
  reverse.score=7;
  CHECK(swap_executor.apply(reciprocal.position,{{forward,reverse},18}));
  CHECK(reciprocal.position.assignments[0].score==11);
  CHECK(reciprocal.position.assignments[1].kind==CAI_SWAP_REVERSE);
  CHECK(reciprocal.position.assignments[1].score==7);
}

void apply_contract_action(Fixture& fixture, TacticalFixture& context,
                           CpMoveKind kind, int unit, int tile) {
  SelectedUnits selection(fixture.selected.data(),int32_t(fixture.selected.size()),int32_t(fixture.units.size()));
  CandidateExecutor executor(*fixture.adapter,*context.tactics,selection,0);
  CpMove move;
  move.unit=unit;
  move.tile=tile;
  move.kind=kind;
  move.score=0;
  if (!executor.apply(context.position,{{move},0}))
    throw std::runtime_error("contract action rejected: kind="+std::to_string(kind)+
        " unit="+std::to_string(unit)+" tile="+std::to_string(tile));
}
void contract_continuation() {
  // Contract: docs/combat-ai/combat-ai-contract.md#continuation and #restart.
  for (bool enemy_contact : {false,true}) {
    Fixture f;
    f.w=12; f.h=5; f.tiles.resize(60); f.fog=1;
    f.add(4,0,24);
    if (enemy_contact) f.add(0,1,28);
    f.start();
    TacticalFixture context(f,28);
    auto before=context.tactics->board(context.position);
    CHECK(!before.plots[27].valid);
    apply_contract_action(f,context,CP_MOVE,0,26);
    CHECK(context.position.branch->state().units[0].tile==26);
    CHECK(context.position.slot_states[SlotIdx{0}]==SlotState::Pending);
    CHECK(context.position.branch->state().units[0].moves>0);
    CHECK(context.tactics->board(context.position).plots[27].valid);
    CHECK(context.position.discovered==enemy_contact);
    CHECK(context.position.move_flags[SlotIdx{0}]==(enemy_contact?MoveFlag::None:MoveFlag::Moved));
    int restarts=int(std::count_if(context.position.assignments.begin(),context.position.assignments.end(),
        [](const CaiAssignment& record) { return record.kind==CAI_RESTART; }));
    CHECK(restarts==int(enemy_contact));
    if (enemy_contact) {
      const auto& restart=context.position.assignments.back();
      CHECK(restart.kind==CAI_RESTART && restart.unit==0);
      CHECK(restart.from_tile==CAI_NO_ID && restart.to_tile==CAI_NO_ID && restart.remaining_moves==0);
    }
    auto choices=context.tactics->preferred(context.position,context.tactics->board(context.position),SlotIdx{0},9);
    bool second_move=std::any_of(choices.begin(),choices.end(),[](CpMove move) { return move.kind==CP_MOVE; });
    CHECK(second_move==enemy_contact);
  }
  Fixture forced;
  forced.add(4,0,16); forced.add(0,1,18); forced.start();
  TacticalFixture freed(forced,18);
  apply_contract_action(forced,freed,CP_FORCED,0,15);
  CHECK(freed.position.move_flags[SlotIdx{0}]==MoveFlag::None);
  auto choices=freed.tactics->preferred(freed.position,freed.tactics->board(freed.position),SlotIdx{0},9);
  CHECK(std::any_of(choices.begin(),choices.end(),[](CpMove move) { return move.kind==CP_MOVE; }));

  Fixture attack;
  attack.add(4,0,16); attack.add(0,1,18); attack.start();
  TacticalFixture fought(attack,18);
  apply_contract_action(attack,fought,CP_MOVE,0,17);
  CHECK(fought.position.move_flags[SlotIdx{0}]==MoveFlag::Moved);
  apply_contract_action(attack,fought,CP_MELEE,0,18);
  CHECK(fought.position.move_flags[SlotIdx{0}]==MoveFlag::None);
}
void contract_plan_acceptance() {
  // Contract: docs/combat-ai/combat-ai-contract.md#plan-acceptance.
  // A generous risk budget isolates the formation/attack predicate.
  for (bool worsened : {false,true}) {
    Fixture f;
    f.config.risk_budget_base=13;
    f.add(1,0,16); f.add(0,1,18); f.start();
    TacticalFixture context(f,18);
    CHECK(context.tactics->initial_distance[SlotIdx{0}]==2);
    if (worsened) apply_contract_action(f,context,CP_MOVE,0,17);
    apply_contract_action(f,context,CP_RANGED,0,18);
    CHECK(context.position.killed.empty());
    CHECK(context.position.slot_states[SlotIdx{0}]==SlotState::Finished);
    CHECK(context.tactics->evaluate_final_position(context.position).accepted==!worsened);
  }
  Fixture improved;
  improved.config.risk_budget_base=13;
  improved.add(1,0,16); improved.add(0,1,17); improved.start();
  TacticalFixture formation(improved,17);
  CHECK(formation.tactics->initial_distance[SlotIdx{0}]==1);
  apply_contract_action(improved,formation,CP_MOVE,0,15);
  apply_contract_action(improved,formation,CP_FINISH_TEMP,0,15);
  CHECK(formation.tactics->board(formation.position).plots[15].distance==2);
  CHECK(formation.tactics->evaluate_final_position(formation.position).accepted);

  Fixture idle;
  idle.add(1,0,16); idle.add(0,1,18); idle.start();
  TacticalFixture unchanged(idle,18);
  apply_contract_action(idle,unchanged,CP_FINISH_TEMP,0,16);
  CHECK(!unchanged.tactics->evaluate_final_position(unchanged.position).accepted);
  unchanged.position.discovered=true;
  CHECK(unchanged.tactics->evaluate_final_position(unchanged.position).accepted);

  Fixture empty;
  empty.add(4,0,16); empty.start();
  TacticalFixture approach(empty,20);
  apply_contract_action(empty,approach,CP_MOVE,0,17);
  apply_contract_action(empty,approach,CP_FINISH_TEMP,0,17);
  CHECK(approach.tactics->evaluate_final_position(approach.position).accepted);

  // Kills and discovery override the formation predicate, never the risk budget.
  Fixture unsafe;
  unsafe.config.risk_budget_base=0; unsafe.config.risk_budget_per_kill=0;
  unsafe.add(1,0,16,10); unsafe.add(0,1,17,1); unsafe.add(0,1,18); unsafe.start();
  TacticalFixture killed(unsafe,17);
  apply_contract_action(unsafe,killed,CP_RANGED,0,17);
  CHECK(killed.position.killed==std::vector<int>{1});
  CHECK(!killed.tactics->evaluate_final_position(killed.position).accepted);
  killed.position.discovered=true;
  CHECK(!killed.tactics->evaluate_final_position(killed.position).accepted);

  TacticalFixture discovered(unsafe,17);
  apply_contract_action(unsafe,discovered,CP_FINISH_TEMP,0,16);
  discovered.position.discovered=true;
  CHECK(discovered.position.killed.empty());
  CHECK(!discovered.tactics->evaluate_final_position(discovered.position).accepted);
}
void contract_early_finish() {
  // Contract: docs/combat-ai/combat-ai-contract.md#early-finish and #slot-states.
  Fixture f;
  f.w=16; f.h=1; f.tiles.resize(16);
  f.config.risk_budget_base=0; f.config.risk_budget_per_kill=0;
  f.add(4,0,8,20); f.add(1,0,0); f.add(0,0,4);
  f.add(4,1,10); f.add(0,1,2,1); f.start();
  TacticalFixture context(f,2);
  CHECK(context.tactics->initial_enemies==std::vector<int>{4});
  apply_contract_action(f,context,CP_MOVE,0,7);
  apply_contract_action(f,context,CP_RANGED,1,2);
  CHECK(terminal_position(context.position,context.tactics->initial_enemies));
  CHECK(context.position.slot_states[SlotIdx{0}]==SlotState::Pending);
  CHECK(context.position.touched[SlotIdx{0}]);
  CHECK(!context.position.touched[SlotIdx{2}]);
  auto accepted=context.tactics->evaluate_final_position(context.position);
  CHECK(accepted.accepted);
  CHECK(accepted.finishes.size()==1 && accepted.finishes[0].unit==1);
  context.position.touched[SlotIdx{2}]=1;
  CHECK(context.tactics->evaluate_final_position(context.position).score==accepted.score-123);
  context.position.touched[SlotIdx{2}]=0;
  context.position.touched[SlotIdx{0}]=0;
  CHECK(context.tactics->evaluate_final_position(context.position).score==accepted.score+123);
  context.position.touched[SlotIdx{0}]=1;
  context.position.slot_states[SlotIdx{0}]=SlotState::Finished;
  CHECK(!context.tactics->evaluate_final_position(context.position).accepted);
  context.position.slot_states[SlotIdx{0}]=SlotState::Pending;
  finalize_assignments(context.position,accepted);
  Plan plan;
  plan.assignments=context.position.assignments;
  auto after=f.replay(plan);
  CHECK(after->state().units[0].tile==7 && after->state().units[0].moves==180);
  CHECK(after->state().units[3].hp>0 && after->state().units[4].hp==0);
}
void contract_combo_invariants() {
  // Contract: docs/combat-ai/combat-ai-contract.md#combo-invariants.
  Fixture f;
  f.add(4,0,16); f.start();
  TacticalFixture context(f,18);
  SelectedUnits selection(f.selected.data(),int32_t(f.selected.size()),int32_t(f.units.size()));
  CandidateExecutor executor(*f.adapter,*context.tactics,selection,0);
  for (auto kind : {CP_FINISH_TEMP,CP_BLOCKED}) {
    CpMove move{0,15,CP_MOVE,0};
    CpMove finish{0,15,kind,0};
    bool rejected=false;
    try { executor.apply(context.position,{{move,finish},0}); }
    catch (const std::logic_error&) { rejected=true; }
    CHECK(rejected);
    CHECK(context.position.assignments.empty());
    CHECK(context.position.branch->state().units[0].tile==16);
    CHECK(context.position.branch->state().units[0].moves==240);
    CHECK(context.position.slot_states[SlotIdx{0}]==SlotState::Pending);
    CHECK(context.position.move_flags[SlotIdx{0}]==MoveFlag::None);
    CHECK(!context.position.touched[SlotIdx{0}]);
  }
}
void contract_search_budgets() {
  // Contract: docs/combat-ai/combat-ai-contract.md#search-budgets.
  // First child kills the only target while the root has been removed from
  // the otherwise empty queue. Hitting max_completed still exhausts the budget.
  Fixture f;
  f.config.max_completed=1;
  f.add(1,0,16); f.add(0,1,18,1); f.start();
  auto plan=f.search(18);
  CHECK(plan.result.reason==CAI_BUDGET_EXHAUSTED);
  CHECK(plan.result.stats.checked==1 && plan.result.stats.generated==2);
  CHECK(plan.result.stats.completed==1);
  CHECK(f.replay(plan)->state().units[1].hp==0);
}
void contract_move_destinations() {
  // Contract: docs/combat-ai/combat-ai-contract.md#combo-invariants.
  // Exercise both stationary paths: FINISH_TEMP and ranged attacks, then
  // repeat on a moved branch rather than checking only the initial tile.
  for (int type=0; type<5; ++type) for (int fog : {0,1})
    for (auto aggression : {CAI_NONE,CAI_MEDIUM}) {
      Fixture f;
      f.fog=fog;
      f.config.aggression=aggression;
      f.add(type,0,16); f.add(0,1,18); f.start();
      TacticalFixture context(f,18);
      auto check=[&]() {
        auto board=context.tactics->board(context.position);
        int current_tile=board.view.units[0].tile;
        for (int limit : {2,5,9,100})
          for (const auto& move : context.tactics->preferred(context.position,board,SlotIdx{0},limit)) {
            if (move.kind==CP_MOVE) CHECK(move.tile!=current_tile);
            if (move.kind==CP_FINISH_TEMP || move.kind==CP_BLOCKED) CHECK(move.tile==current_tile);
          }
        for (bool depth_first : {false,true})
          for (const auto& candidate : context.tactics->candidates(context.position,depth_first)) {
            CHECK(!candidate.moves.empty() && candidate.moves.size()<=2);
            if (candidate.moves.size()==2) CHECK(candidate.moves[0].unit!=candidate.moves[1].unit);
            for (const auto& move : candidate.moves) {
              if (move.kind==CP_MOVE) CHECK(move.tile!=board.view.units[move.unit].tile);
              if (move.kind==CP_FORCED)
                CHECK(candidate.moves.size()==2 && candidate.moves[1].kind==CP_MOVE);
            }
          }
      };
      check();
      auto stationary=context.tactics->preferred(context.position,context.tactics->board(context.position),SlotIdx{0},100);
      CpMoveKind expected=aggression==CAI_NONE || (type!=1 && type!=2) ? CP_FINISH_TEMP : CP_RANGED;
      CHECK(std::any_of(stationary.begin(),stationary.end(),[expected](const CpMove& move) {
        return move.kind==expected;
      }));
      apply_contract_action(f,context,CP_MOVE,0,15);
      CHECK(context.position.branch->state().units[0].tile==15);
      check();
    }
}
void contract_swap_equivalence() {
  // Contract: docs/combat-ai/combat-ai-contract.md#candidate-order.
  Fixture f;
  f.add(1,0,16,50); f.add(3,0,15); f.add(0,1,17,50);
  f.corridor({15,16,17}); f.start();
  TacticalFixture origin(f,17);
  auto candidates=origin.tactics->candidates(origin.position,false);
  SelectedUnits selected(f.selected.data(),int32_t(f.selected.size()),int32_t(f.units.size()));
  SearchWorkspace workspace;
  std::vector<CaiUnit> first_physical;
  std::vector<CaiUnit> first_final;
  std::vector<int32_t> scores;
  for (const auto& combo : candidates) if (combo.moves[0].kind==CP_SWAP) {
    TacticalFixture child(f,17);
    CandidateExecutor executor(*f.adapter,*child.tactics,selected,0);
    CHECK(executor.apply(child.position,combo));
    child.position.rank=combo.score;
    auto state=child.position.branch->state();
    std::vector<CaiUnit> physical(state.units,state.units+state.unit_count);
    if (first_physical.empty()) first_physical=physical;
    else CHECK(std::memcmp(first_physical.data(),physical.data(),physical.size()*sizeof(CaiUnit))==0);
    for (int slot=0; slot<2; ++slot) {
      CHECK(child.position.slot_states[SlotIdx{slot}]==SlotState::Pending);
      CHECK(child.position.move_flags[SlotIdx{slot}]==MoveFlag::Swapped);
      auto preferred=child.tactics->preferred(child.position,child.tactics->board(child.position),SlotIdx{slot},100);
      CHECK(std::none_of(preferred.begin(),preferred.end(),[](const CpMove& move) { return move.kind==CP_MOVE; }));
    }
    auto view=TacticalView(*child.position.branch,0);
    auto key=search_position_key(child.position,view,0,selected);
    CHECK(workspace.equivalent.emplace(key,std::make_pair(child.position.attack_score,int(scores.size()))).second);
    ++child.position.rank;
    CHECK(search_position_key(child.position,view,0,selected)!=key);
    --child.position.rank;
    CHECK(latest_move_score(child.position,UnitId{combo.moves[0].unit}).has_value());
    CHECK(!latest_move_score(child.position,UnitId{combo.moves[1].unit}).has_value());

    // Both orientations follow the same actually generated continuation.
    for (auto spec : {std::make_pair(CP_RANGED,0),std::make_pair(CP_FINISH_TEMP,1)}) {
      auto next=child.tactics->candidates(child.position,false);
      auto action=std::find_if(next.begin(),next.end(),[&](const CpCombo& candidate) {
        return candidate.moves.size()==1 && candidate.moves[0].kind==spec.first && candidate.moves[0].unit==spec.second;
      });
      CHECK(action!=next.end());
      CHECK(executor.apply(child.position,*action));
      if (spec.first!=CP_FINISH_TEMP) child.position.rank+=action->score;
    }
    auto final=child.tactics->evaluate_final_position(child.position);
    CHECK(final.accepted);
    scores.push_back(final.score);
    finalize_assignments(child.position,final);
    Plan plan;
    plan.assignments=child.position.assignments;
    auto replay=f.replay(plan);
    auto after=replay->state();
    std::vector<CaiUnit> units(after.units,after.units+after.unit_count);
    if (first_final.empty()) first_final=units;
    else CHECK(std::memcmp(first_final.data(),units.data(),units.size()*sizeof(CaiUnit))==0);
  }
  CHECK(scores.size()==2 && scores[0]!=scores[1]);
  CHECK(workspace.equivalent.size()==2); // Neither orientation may be lost.
}
void contract_combo_mode() {
  // Contract: docs/combat-ai/combat-ai-contract.md#combo-invariants.
#if defined(CAI_CP_COMBOS) && CAI_CP_COMBOS == 0
  constexpr bool enabled=false;
#else
  constexpr bool enabled=true;
#endif
  Fixture swap;
  swap.add(1,0,16,50); swap.add(3,0,15); swap.add(0,1,17,50);
  swap.corridor({15,16,17}); swap.start();
  TacticalFixture context(swap,17);
  auto board=context.tactics->board(context.position);
  bool swap_found=false;
  for (const auto& candidate : context.tactics->candidates(context.position,false)) {
    swap_found=swap_found || candidate.moves[0].kind==CP_SWAP;
    if (!enabled) {
      CHECK(candidate.moves.size()==1);
      if (candidate.moves[0].kind==CP_MOVE) CHECK(board.view.occupied[candidate.moves[0].tile]<0);
    }
  }
  CHECK(swap_found==enabled);

  Fixture idle;
  idle.config.aggression=CAI_NONE;
  idle.add(0,0,16); idle.add(0,0,9); idle.start();
  TacticalFixture finishes(idle,16);
  bool paired=false;
  for (const auto& candidate : finishes.tactics->candidates(finishes.position,true)) {
    paired=paired || candidate.moves.size()==2;
    if (!enabled) CHECK(candidate.moves.size()==1);
  }
  CHECK(paired==enabled);
}
void cp_hypothetical_flanking() {
  // Contract: docs/combat-ai/combat-ai-contract.md#hypothetical-flanking.
  // Check actual occupancy, not a prescribed plan or an assumed optimal move.
  for (int scenario : {0,1,2}) {
    Fixture f;
    f.config.aggression=CAI_BRAVEHEART;
    if (scenario==0) {
      f.w=8; f.h=1; f.tiles.resize(8);
      f.add(0,0,0); f.add(0,1,2);
    } else {
      f.add(0,0,scenario==1 ? 16 : 15);
      f.add(0,1,17);
    }
    f.start();
    TacticalFixture context(f,scenario==0 ? 2 : 17);
    if (scenario==2) {
      // A moved branch differs from root: counting must use current occupancy.
      CHECK(f.adapter->apply(*context.position.branch,{MOVE,0,16}).accepted);
    }
    auto board=context.tactics->board(context.position);
    const auto& attacker=board.view.units[0];
    const auto& enemy=board.view.units[1];
    CHECK(board.plots[enemy.tile].friends==(scenario==0 ? 0 : 1));
    auto preferred=context.tactics->preferred(context.position,board,SlotIdx{0},100);
    bool checked=false;
    for (auto reach : f.adapter->move_intents(*context.position.branch,0)) {
      if (reach.tile==attacker.tile || board.view.occupied[reach.tile]>=0 ||
          f.adapter->distance(reach.tile,enemy.tile)!=1 || !board.plots[reach.tile].valid) continue;
      auto candidate=std::find_if(preferred.begin(),preferred.end(),[&](const CpMove& move) {
        return move.kind==CP_MOVE && move.tile==reach.tile;
      });
      if (candidate==preferred.end()) continue;
      // All fixtures have no other friendly unit, so the attacker's real
      // position must contribute zero supporting friends, in every scenario.
      auto forecast=f.adapter->forecast(*f.root,0,reach.tile,1,enemy.tile,0,true,false);
      CHECK(forecast.code==0);
      int delta=-board.plots[reach.tile].enemies;
      int attack_score=forecast.damage+delta*forecast.damage/10;
      CHECK(attack_score<enemy.hp-1); // No kill/near-kill bonus in this fixture.
      int attacks=std::min(1,(reach.remaining_moves+59)/60);
      int potential=attacks>0 ? attacks*attack_score/2 : attack_score/4;
      auto base=context.tactics->move_score(context.position,board,SlotIdx{0},reach.tile,
          reach.remaining_moves,MoveEvaluation::Intermediate);
      CHECK(base.acceptable);
      CHECK(candidate->score==base.move.score+potential*10);
      CHECK(candidate->near_kill==-1);
      checked=true;
    }
    CHECK(checked);
    CHECK(board.view.units[0].tile==(scenario==0 ? 0 : 16));
    CHECK(board.view.units[0].hp==100 && board.view.units[1].hp==100);
  }
}

void cp_average_personality() {
  // CP's midpoint offense=5 gives a strict HP<40 pruning threshold.
  for (int hp : {39,40,49,50}) {
    Fixture f;
    f.w=16; f.h=1; f.tiles.resize(16);
    f.add(0,0,0,hp); f.add(0,1,4); f.start();
    TacticalFixture context(f,4);
    if (hp<40) {
      check_root_exclusion(f,context,SlotIdx{0});
    } else {
      CHECK(context.position.slot_states[SlotIdx{0}]==SlotState::Pending);
      CHECK(context.tactics->active_order==std::vector<SlotIdx>{SlotIdx{0}});
      auto board=context.tactics->board(context.position);
      auto choices=context.tactics->preferred(context.position,board,SlotIdx{0},1);
      CHECK(!choices.empty() && choices.front().kind==CP_MOVE);
    }
    CHECK(context.position.branch->state().units[0].hp==hp);
  }
}

void cp_storage_caches() {
  // A freshly initialized context is a cache-empty oracle for the SAME
  // heuristic, not an assertion that its preferred plan is optimal.
  auto compare = [](Fixture& f, TacticalFixture& context, int target) {
    TacticalFixture empty(f,target);
    for (bool depth : {false,true}) {
      auto actual=context.tactics->candidates(context.position,depth);
      auto expected=empty.tactics->candidates(context.position,depth);
      CHECK(actual.size()==expected.size());
      for (size_t i=0;i<actual.size();++i) {
        CHECK(actual[i].score==expected[i].score);
        CHECK(actual[i].moves.size()==expected[i].moves.size());
        for (size_t j=0;j<actual[i].moves.size();++j) {
          const auto& a=actual[i].moves[j]; const auto& b=expected[i].moves[j];
          CHECK(a.unit==b.unit && a.tile==b.tile && a.kind==b.kind && a.score==b.score);
          CHECK(a.remaining==b.remaining && a.self_damage==b.self_damage);
          CHECK(a.near_kill==b.near_kill && a.kill==b.kill);
        }
      }
    }
  };
  for (bool fog : {false,true}) {
    Fixture f; f.w=12; f.h=5; f.tiles.resize(60); f.fog=fog;
    f.add(4,0,24); f.add(1,0,12); f.add(0,1,28,1); f.start();
    TacticalFixture context(f,28);
    compare(f,context,28);
    compare(f,context,28); // repeated lookups in the same position
    apply_contract_action(f,context,CP_MOVE,0,26);
    compare(f,context,28); // movement, new observation, tactical-area growth
    apply_contract_action(f,context,CP_FORCED,0,27);
    apply_contract_action(f,context,CP_MELEE,0,28);
    CHECK(context.position.killed==std::vector<int32_t>{2});
    compare(f,context,28); // freed plot and removed enemy ZOC
  }
}

void check_tactical_board(TacticalFixture& context) {
  const auto& actual=context.tactics->board(context.position);
  auto expected=context.tactics->rebuild_board(context.position);
  CHECK(actual.enemies==expected.enemies);
  CHECK(actual.view.occupied==expected.view.occupied);
  CHECK(actual.view.visibility==expected.view.visibility);
  CHECK(actual.plots.size()==expected.plots.size());
  for (size_t tile=0;tile<actual.plots.size();++tile) {
    const auto& a=actual.plots[tile]; const auto& b=expected.plots[tile];
    CHECK(a.valid==b.valid && a.blocked==b.blocked && a.end==b.end && a.edge==b.edge);
    CHECK(a.enemy==b.enemy && a.distance==b.distance && a.enemies==b.enemies);
    CHECK(a.friends==b.friends && a.friends_end==b.friends_end);
  }
}

void cp_storage_boards() {
  for (int seed=0;seed<40;++seed) {
    Fixture f; f.w=8; f.h=6; f.tiles.resize(48);
    f.fog=seed&1; f.parity=(seed>>1)&1; f.wrap=(seed>>2)&1;
    f.add(seed%5,0,16); f.add((seed+1)%5,0,17); f.add((seed+2)%5,0,24);
    f.add(0,1,20,seed%3==0?1:100); f.add(3,1,28); f.add(1,1,37);
    f.start(); TacticalFixture context(f,20);
    SelectedUnits selected(f.selected.data(),int32_t(f.selected.size()),int32_t(f.units.size()));
    CandidateExecutor executor(*f.adapter,*context.tactics,selected,0);
    for (int step=0;step<12;++step) {
      check_tactical_board(context);
      auto choices=context.tactics->candidates(context.position,step>3);
      bool advanced=false;
      for (const auto& candidate : choices) {
        advanced=executor.apply(context.position,candidate);
        // A rejected attempt ends this disposable scenario. Its physical
        // state still has to agree with a complete property rebuild.
        break;
      }
      check_tactical_board(context);
      if (!advanced) break;
    }
  }
}

void cp_storage_pool() {
  SearchWorkspace reused;
  for (int iteration=0;iteration<16;++iteration) {
    Fixture f;
    if ((iteration/2)&1) { f.w=16; f.h=16; f.tiles.resize(256); }
    f.fog=(iteration/2)&1; f.parity=(iteration>>2)&1; f.wrap=(iteration>>3)&1;
    f.add(iteration%5,0,0); f.add(1,0,1); f.add(0,1,3); f.add(3,1,4);
    f.start();
    f.config.position_capacity=(iteration%3==0?13:iteration%3==1?180:1000);
    CaiSearchRequest request{&f.state,0,0,3,f.selected.data(),int32_t(f.selected.size()),{0,-1,nullptr,0}};
    SearchWorkspace fresh;
    auto actual=find_assignments(reused,f.config,request);
    auto expected=find_assignments(fresh,f.config,request);
    CHECK(actual.result.reason==expected.result.reason);
    CHECK(std::memcmp(&actual.result.stats,&expected.result.stats,sizeof(CaiSearchStats))==0);
    CHECK(actual.assignments.size()==expected.assignments.size());
    for (size_t i=0;i<actual.assignments.size();++i)
      CHECK(std::memcmp(&actual.assignments[i],&expected.assignments[i],sizeof(CaiAssignment))==0);
    CHECK(reused.used_positions==actual.result.stats.generated);
    CHECK(reused.positions.size()==size_t(f.config.position_capacity));
    for (size_t i=size_t(reused.used_positions);i<reused.positions.size();++i)
      CHECK(!reused.positions[i].branch);
    f.replay(actual);
  }
  // Same geometry, other player: CP requests a hard wipe of every position.
  Fixture switched; switched.w=16; switched.h=16; switched.tiles.resize(256);
  switched.parity=1; switched.wrap=1;
  switched.add(0,0,0); switched.add(1,0,1); switched.add(4,1,3); switched.add(0,1,4);
  switched.start(); switched.state.active_side=1;
  switched.selected={2,3}; switched.config.max_completed=1;
  switched.config.position_capacity=1000;
  reused.positions.back().assignments.reserve(200);
  CaiSearchRequest other_side{&switched.state,1,0,0,switched.selected.data(),2,{0,-1,nullptr,0}};
  find_assignments(reused,switched.config,other_side);
  CHECK(reused.previous_side==1 && reused.used_positions<999);
  CHECK(reused.positions.back().assignments.capacity()==0);
  // Explicit CP hard reset destroys retained containers, but not array capacity.
  size_t capacity=reused.positions.capacity();
  reused.clear(true);
  CHECK(reused.positions.empty() && reused.positions.capacity()==capacity);
  CpPlots plots;
  plots.resize(4096);
  for (int i=0;i<255;++i) plots.add(size_t(i+254)).valid=true;
  CHECK(plots.local_size()==255 && plots[255].valid && !plots[0].valid);
  bool rejected=false;
  try { plots.add(4095); } catch(const std::logic_error&) { rejected=true; }
  CHECK(rejected && !plots[4095].valid);
}

void cp_context_invariants() {
  Fixture f; f.add(0,0,16); f.add(0,1,18); f.start();
  SelectedUnits selected(f.selected.data(),int32_t(f.selected.size()),int32_t(f.units.size()));
  CHECK(selected.find_slot(UnitId{0})==std::optional<SlotIdx>{SlotIdx{0}});
  CHECK(!selected.find_slot(UnitId{1}));
  TacticalFixture context(f,18);
  auto states=context.position.slot_states;
  auto order=context.tactics->active_order;
  auto distances=context.tactics->initial_distance;
  auto safe=context.tactics->safe;
  bool rejected=false;
  try { context.tactics->initialize(context.position); }
  catch (const std::logic_error&) { rejected=true; }
  CHECK(rejected);
  CHECK(context.position.slot_states==states && context.tactics->active_order==order);
  CHECK(context.tactics->initial_distance==distances && context.tactics->safe==safe);
  CHECK(context.tactics->danger(0,16,0,{1,1})==context.tactics->danger(0,16,0,{1}));
  CaiSearchRequest request{&f.state,0,f.turn,18,f.selected.data(),int32_t(f.selected.size()),{0,-1,nullptr,0}};
  CpTactics uninitialized(*f.adapter,*f.root,f.config,request);
  auto board=context.tactics->board(context.position);
  rejected=false;
  try { uninitialized.move_score(context.position,board,SlotIdx{0},16,120,MoveEvaluation::Intermediate); }
  catch (const std::logic_error&) { rejected=true; }
  CHECK(rejected);
  std::unique_ptr<CaiContext,decltype(&cai_free)> api(cai_new(&f.config),cai_free);
  CHECK(api);
  CaiAssignment assignments[128]{}; CaiSearchResult result{};
  CHECK(cai_search(api.get(),&request,assignments,128,&result)==CAI_OK);
  CaiSearchDiagnostics before{},after{};
  CHECK(cai_search_diagnostics(api.get(),&before)==CAI_OK);
  int32_t output=999,count=-1;
  CHECK(cai_attack_candidates(api.get(),&request,&output,0,&count)==CAI_BUFFER_TOO_SMALL);
  CHECK(count==1 && output==999); // No partial output on insufficient capacity.
  CHECK(cai_attack_candidates(api.get(),&request,&output,1,&count)==CAI_OK);
  CHECK(count==1 && output==0);
  CHECK(cai_search_diagnostics(api.get(),&after)==CAI_OK);
  CHECK(std::memcmp(&before,&after,sizeof(before))==0);
  request.expected_turn++;
  CHECK(cai_attack_candidates(api.get(),&request,&output,1,&count)==CAI_BAD_ARGUMENT && count==0);
  CHECK(cai_attack_candidates(nullptr,&request,&output,1,&count)==CAI_BAD_ARGUMENT && count==0);
}

void diagnostic_trace() {
  diagnostic_parity = true;
  attacks();
  swaps_and_freeing();
  risk();
  contact();
  planner_root_exclusion();
  planner_blocked_handoff();
  planner_combo_contacts();
  budgets_and_determinism();
  random_replay();
  diagnostic_parity = false;
  DecisionTrace trace;
  trace.enabled = true;
  trace.limit = 1;
  trace.event("test");
  CHECK(trace.truncated && trace.data.empty());
  trace.reset();
  CHECK(!trace.truncated && trace.data.empty());
  CHECK(cai_trace_enable(nullptr,1,100) == CAI_BAD_ARGUMENT);
  CHECK(cai_trace_data(nullptr) == nullptr && cai_trace_status(nullptr) == -1);
}

int main(int argc,char** argv) {
  try {
    std::string name=argc>1?argv[1]:"";
    if(name=="diagnostic_trace") diagnostic_trace();
    else if(name=="attacks") attacks();
    else if(name=="swaps_and_freeing") swaps_and_freeing();
    else if(name=="risk") risk();
    else if(name=="cover_and_retreat") cover_and_retreat();
    else if(name=="budgets_and_determinism") budgets_and_determinism();
    else if(name=="contact") contact();
    else if(name=="random_replay") random_replay();
    else if(name=="scale_and_last_tile") scale_and_last_tile();
    else if(name=="support_healing") support_healing();
    else if(name=="support_approach") support_approach();
    else if(name=="cp_area_and_preferences") cp_area_and_preferences();
    else if(name=="cp_blocker_combinations") cp_blocker_combinations();
    else if(name=="cp_forecast_and_danger") cp_forecast_and_danger();
    else if(name=="cp_predicted_kill_end_danger") cp_predicted_kill_end_danger();
    else if(name=="cp_history_and_admission") cp_history_and_admission();
    else if(name=="cp_area_capacity") cp_area_capacity();
    else if(name=="planner_invariants") planner_invariants();
    else if(name=="planner_finalization") planner_finalization();
    else if(name=="planner_root_exclusion") planner_root_exclusion();
    else if(name=="planner_blocked_handoff") planner_blocked_handoff();
    else if(name=="planner_combo_contacts") planner_combo_contacts();
    else if(name=="contract_continuation") contract_continuation();
    else if(name=="contract_plan_acceptance") contract_plan_acceptance();
    else if(name=="contract_early_finish") contract_early_finish();
    else if(name=="contract_combo_invariants") contract_combo_invariants();
    else if(name=="contract_search_budgets") contract_search_budgets();
    else if(name=="contract_move_destinations") contract_move_destinations();
    else if(name=="contract_swap_equivalence") contract_swap_equivalence();
    else if(name=="contract_combo_mode") contract_combo_mode();
    else if(name=="cp_hypothetical_flanking") cp_hypothetical_flanking();
    else if(name=="cp_context_invariants") cp_context_invariants();
    else if(name=="cp_storage_caches") cp_storage_caches();
    else if(name=="cp_storage_boards") cp_storage_boards();
    else if(name=="cp_storage_pool") cp_storage_pool();
    else if(name=="cp_average_personality") cp_average_personality();
    else throw std::runtime_error("unknown test group");
    std::cout<<name<<" passed\n"; return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
