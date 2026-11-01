// End-to-end local-search properties. Replay uses the rules adapter, not the
// planner's candidate/scoring implementation, and verifies the resulting state.
#include "../../native/combat_ai/planner.h"
#include "../../native/combat_state.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>

using namespace combat_ai;
using namespace combat_rules;
#define CHECK(x) do { if (!(x)) throw std::runtime_error( \
  std::string(__func__) + ":" + std::to_string(__LINE__) + ": " #x); } while (false)

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
    for (auto u : units) us.push_back({u.type,u.side,u.tile,u.hp,u.moves,u.attacked,u.acted,u.fortified,u.waiting});
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
    return find_assignments(work,config,request);
  }
  std::unique_ptr<Branch> replay(const Plan& plan) {
    auto branch = root->clone();
    std::vector<bool> finished(units.size(),false);
    for (auto a : plan.assignments) {
      CHECK(std::find(selected.begin(), selected.end(), a.unit) != selected.end());
      if (a.kind == CAI_INITIAL || a.kind == CAI_RESTART || a.kind == CAI_SWAP_REVERSE) continue;
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
  Fixture f; f.add(1,0,16,10); f.add(3,0,15); f.add(0,1,17,50);
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
  f.add(4,0,24); f.add(0,1,29,1); f.start();
  CHECK(!f.visibility[29]);
  auto p=f.search(29); CHECK(!p.assignments.empty());
  bool restart=false, continued=false, interrupted=false;
  for(auto a:p.assignments) {
    if(a.kind==CAI_RESTART) restart=true;
    if(restart && (a.kind==CAI_MOVE || a.kind==CAI_MELEE)) continued=true;
    if(a.kind==CAI_MOVE && a.command_target!=a.to_tile) interrupted=true;
  }
  CHECK(restart); CHECK(continued); CHECK(interrupted); f.replay(p);
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
int main(int argc,char** argv) {
  try {
    std::string name=argc>1?argv[1]:"";
    if(name=="attacks") attacks();
    else if(name=="swaps_and_freeing") swaps_and_freeing();
    else if(name=="risk") risk();
    else if(name=="cover_and_retreat") cover_and_retreat();
    else if(name=="budgets_and_determinism") budgets_and_determinism();
    else if(name=="contact") contact();
    else if(name=="random_replay") random_replay();
    else if(name=="scale_and_last_tile") scale_and_last_tile();
    else throw std::runtime_error("unknown test group");
    std::cout<<name<<" passed\n"; return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
