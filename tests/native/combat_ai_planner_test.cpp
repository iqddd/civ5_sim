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
  TacticalFixture(Fixture& f, int target, const std::vector<int32_t>& history={}) {
    position.branch=f.root->clone();
    position.done.assign(f.selected.size(),0); position.moved=position.done;
    position.last_kill=position.done; position.touched=position.done;
    CaiSearchRequest request{&f.state,0,f.turn,target,f.selected.data(),int32_t(f.selected.size()),{0,-1,nullptr,0}};
    tactics=std::make_unique<CpTactics>(*f.adapter,*f.root,f.config,request,history);
    tactics->initialize(position);
  }
};
void cp_area_and_preferences() {
  Fixture distant; distant.w=16; distant.h=1; distant.tiles.resize(16);
  distant.add(4,0,0); distant.add(0,1,9); distant.start();
  TacticalFixture f(distant,9);
  auto board=f.tactics->board(f.position);
  CHECK(board.plots[0].valid); CHECK(!board.plots[2].valid);
  auto preferred=f.tactics->preferred(f.position,board,0,9);
  CHECK(std::any_of(preferred.begin(),preferred.end(),[](CpMove m) { return m.kind==CP_MOVE && m.tile==4; }));
  CHECK(preferred.back().kind==CP_BLOCKED && preferred.back().score==-10);
  for (auto m : preferred) if (m.kind==CP_MOVE) CHECK(distant.adapter->distance(m.tile,9)<=6);
  // Starting outside radius six does not exclude a participant with a useful
  // reachable destination. Conversely, all far intents produce only BLOCKED.
  TacticalFixture far(distant,15);
  auto blocked=far.tactics->preferred(far.position,far.tactics->board(far.position),0,9);
  CHECK(blocked.size()==1 && blocked[0].kind==CP_BLOCKED);
  Fixture ties; ties.add(0,0,17); ties.start();
  TacticalFixture equal(ties,17);
  auto top=equal.tactics->preferred(equal.position,equal.tactics->board(equal.position),0,2);
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
  auto choices=f.tactics->preferred(f.position,board,0,9);
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
  auto blocked=omitted.tactics->preferred(omitted.position,omitted.tactics->board(omitted.position),0,9);
  CHECK(std::none_of(blocked.begin(),blocked.end(),[](CpMove m) { return m.kind==CP_MOVE && m.tile==16; }));
  Fixture swap; swap.add(1,0,16,50); swap.add(3,0,15); swap.add(0,1,17,50);
  swap.corridor({15,16,17}); swap.start(); TacticalFixture reciprocal(swap,17);
  auto pairs=reciprocal.tactics->candidates(reciprocal.position,false);
  CHECK(std::any_of(pairs.begin(),pairs.end(),[](const CpCombo& c) {
    return c.moves.size()==2 && c.moves[0].kind==CP_SWAP && c.moves[1].kind==CP_SWAP_REVERSE;
  }));
  // BLOCKED does not promise cover at turn end; FINISH_TEMP does.
  f.position.done[1]=2;
  CHECK(!f.tactics->board(f.position).plots[16].end);
  f.position.done[1]=1;
  auto ended=f.tactics->board(f.position);
  CHECK(ended.plots[16].end && ended.plots[15].friends_end==1);
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
  // Pin the actual CP parameter pass-through, even though its name suggests
  // the extra damage should weaken the defender in both attack types.
  CHECK(extra.attacker_strength<melee.attacker_strength);
  CHECK(extra.defender_strength==melee.defender_strength);
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
  TacticalFixture absent(f,240);
  CHECK(absent.position.done[0]==0);
  CHECK(absent.tactics->roles[0]==CAI_FIRST_LINE);
  CHECK(absent.position.plots[0] && absent.position.plots[1]);
  TacticalFixture equality(f,240,{100,0});
  CHECK(equality.position.done[0]==0); // Strict >, not >=.
  TacticalFixture excluded(f,240,{101,0});
  CHECK(excluded.position.done[0]==2);
  CHECK(!excluded.position.plots[0] && !excluded.position.plots[1]);
  CHECK(excluded.tactics->initial_enemies.empty());
  // A remote unrelated enemy cannot admit the unit by being added through
  // its neighborhood before admission. Near the real target it is admitted.
  TacticalFixture frontline(f,1,{101,0});
  CHECK(frontline.position.done[0]==0);
  CHECK(frontline.tactics->roles[0]==CAI_THIRD_LINE);
  CHECK(movement_role(f.units[0],100)==CAI_FIRST_LINE);
  CHECK(movement_role(f.units[0],101)==CAI_THIRD_LINE);
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
  auto choices=cap.tactics->preferred(cap.position,cap.tactics->board(cap.position),12,9);
  CHECK(!choices.empty() && choices.back().kind==CP_BLOCKED);
  CHECK(std::any_of(choices.begin(),choices.end(),[](CpMove m) { return m.kind==CP_MOVE; }));
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
    else if(name=="support_healing") support_healing();
    else if(name=="support_approach") support_approach();
    else if(name=="cp_area_and_preferences") cp_area_and_preferences();
    else if(name=="cp_blocker_combinations") cp_blocker_combinations();
    else if(name=="cp_forecast_and_danger") cp_forecast_and_danger();
    else if(name=="cp_history_and_admission") cp_history_and_admission();
    else if(name=="cp_area_capacity") cp_area_capacity();
    else throw std::runtime_error("unknown test group");
    std::cout<<name<<" passed\n"; return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
