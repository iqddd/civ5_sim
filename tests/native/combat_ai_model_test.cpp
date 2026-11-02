// Behavioral/differential checks: the public simulator ABI is the transition oracle.
#include "../../native/combat_ai/model.h"
#include "../../native/combat_state.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>

using namespace combat_ai;
using namespace combat_rules;
#define CHECK(x) do { if (!(x)) throw std::runtime_error( \
  std::string(__func__) + ":" + std::to_string(__LINE__) + ": " #x); } while (false)

Unit unit(int type = 0, int side = 0, int tile = 26, int hp = 100) {
  return {type, side, tile, hp, moves[type], 0, 0, 0, 0};
}
bool same(const Unit& a, const CaiUnit& b) {
  return std::tie(a.type,a.side,a.tile,a.hp,a.moves,a.attacked,a.acted,a.fortified,a.waiting,a.damage_taken_last_turn)
      == std::tie(b.type,b.side,b.tile,b.hp,b.moves,b.attacked,b.acted,b.fortified,b.waiting,b.damage_taken_last_turn);
}
auto command_key(Command a) { return std::make_tuple(a.kind, a.unit, a.target); }

struct Fixture {
  int w = 12, h = 8, parity = 0, wrap = 0, fog = 0;
  std::vector<Tile> tiles = std::vector<Tile>(96, Tile{});
  std::vector<Unit> units;
  std::vector<CaiTile> input_tiles;
  std::vector<CaiUnit> input_units;
  std::vector<int32_t> visibility;
  CaiBattleState input{};
  void* real = nullptr;
  int observer = 0;
  std::unique_ptr<RulesAdapter> adapter;
  std::unique_ptr<Branch> branch;
  ~Fixture() { sim_free(real); }
  void start(ContactMemory memory = {}, int random = 0, int observing_side = 0) {
    observer = observing_side;
    real = sim_new(w,h,parity,wrap,tiles.data(),units.data(),int(units.size()),fog,random,42);
    CHECK(real);
    input_tiles.clear(); input_units.clear();
    for (auto t : tiles) input_tiles.push_back({t.terrain,t.elevation,t.feature,t.rivers});
    for (int i=0;i<int(units.size());++i) {
      const auto& u = units[i];
      input_units.push_back({i,u.type,u.side,u.tile,u.hp,u.moves,u.attacked,u.acted,u.fortified,u.waiting,
                            u.damage_taken_last_turn});
    }
    visibility.resize(tiles.size());
    sim_visibility(real,observer,visibility.data());
    input = {w,h,parity,wrap,input_tiles.data(),int32_t(tiles.size()),input_units.data(),
             int32_t(units.size()),visibility.data(),int32_t(visibility.size()),0,0,fog};
    adapter = make_rules_adapter(input,observer);
    branch = adapter->root(input,memory);
    equal();
  }
  void equal() const {
    const auto& s = branch->state();
    std::vector<Unit> actual(s.unit_count);
    CHECK(sim_units(real,-1,actual.data()) == s.unit_count);
    for (int i=0;i<s.unit_count;++i) { CHECK(s.units[i].id == i); CHECK(same(actual[i],s.units[i])); }
    std::vector<int> vis(s.tile_count);
    sim_visibility(real,observer,vis.data());
    CHECK(std::equal(vis.begin(),vis.end(),s.visibility));
    Info info{}; sim_info(real,&info);
    CHECK(info.side == s.active_side && info.turn == s.turn && info.fog == s.fog);
    int count = sim_actions(real,nullptr,0);
    std::vector<Action> expected(count); sim_actions(real,expected.data(),count);
    auto commands = adapter->actions(*branch);
    CHECK(commands.size() == expected.size());
    for (size_t i=0;i<commands.size();++i)
      CHECK(command_key(commands[i]) == command_key({expected[i].kind,expected[i].unit,expected[i].target}));
  }
  Transition step(Command c) {
    auto before = branch->clone();
    const auto expected = sim_step(real,{c.kind,c.unit,c.target});
    if (c.kind == MELEE || c.kind == RANGED) {
      const auto p = adapter->preview(*branch,c);
      CHECK(p.code == expected.code);
      if (!expected.code) {
        CHECK(p.damage == expected.damage && p.retaliation == expected.retaliation);
        CHECK(p.attacker_strength == expected.attacker_strength && p.defender_strength == expected.defender_strength);
      }
    }
    auto t = adapter->apply(*branch,c);
    CHECK(t.accepted == (expected.code == 0));
    CHECK(t.result.code == expected.code && t.result.moved == expected.moved);
    CHECK(t.result.damage == expected.damage && t.result.retaliation == expected.retaliation);
    CHECK(t.result.attacker_strength == expected.attacker_strength && t.result.defender_strength == expected.defender_strength);
    if (!t.accepted) {
      const auto& a = before->state(); const auto& b = branch->state();
      CHECK(a.turn == b.turn && a.active_side == b.active_side);
      for (int i=0;i<a.unit_count;++i) {
        const auto& u = a.units[i];
        Unit v{u.type,u.side,u.tile,u.hp,u.moves,u.attacked,u.acted,u.fortified,u.waiting};
        CHECK(same(v,b.units[i]));
      }
      CHECK(t.newly_visible.empty() && !t.contact_interrupted);
    }
    equal();
    return t;
  }
  int remaining(int id, int tile) const {
    for (auto r : adapter->reachable(*branch,id)) if (r.tile == tile) return r.remaining_moves;
    return -1;
  }
};

void movement() {
  Fixture f; f.units = {unit(4),unit(0,0,27)}; f.start();
  CHECK(f.remaining(0,26) == 240 && f.remaining(0,27) == -1 && f.remaining(0,28) == 120);
  CHECK(!f.step({MOVE,0,27}).accepted);
  CHECK(f.step({MOVE,0,28}).accepted);
  CHECK(f.branch->state().units[0].tile == 28);
  CHECK(f.step({MOVE,1,26}).accepted);
  CHECK(f.remaining(0,27) == 60);
  Fixture fractional; auto u=unit(4); u.moves=1; fractional.units={u};
  fractional.tiles[27].elevation=1; fractional.tiles[27].feature=1;
  fractional.start(); CHECK(fractional.remaining(0,27)==0);
  fractional.step({MOVE,0,27}); CHECK(fractional.branch->state().units[0].moves==0);
  CHECK(fractional.adapter->reachable(*fractional.branch,0).empty());
}

void terrain_and_zoc() {
  for (int parity : {0,1}) {
    Fixture f; f.parity=parity;
    Geometry g(f.w,f.h,parity,0,f.tiles.data());
    int north=g.neighbors[26][0];
    f.units={unit(4),unit(0,1,north)}; f.start();
    CHECK(f.remaining(0,27)==0); f.step({MOVE,0,27});
    CHECK(f.branch->state().units[0].moves==0);
    Fixture escape; escape.parity=parity; escape.units={unit(4),unit(0,1,north)}; escape.start();
    escape.step({MOVE,0,25}); CHECK(escape.branch->state().units[0].moves==180);
  }
  Fixture river; river.tiles[26].rivers=1<<1; river.tiles[27].rivers=1<<4;
  river.units={unit(4)}; river.start(); CHECK(river.remaining(0,27)==0);
  river.step({MOVE,0,27}); CHECK(river.branch->state().units[0].moves==0);
  Fixture rough; rough.tiles[27].feature=1; rough.units={unit(2)}; rough.start();
  rough.step({MOVE,0,27}); CHECK(rough.branch->state().units[0].moves==0);
}

void swaps() {
  for (int a=0;a<5;++a) for (int b=0;b<5;++b) {
    Fixture f; auto partner=unit(b,0,27); partner.waiting=1; partner.attacked=1;
    f.units={unit(a),partner}; f.tiles[26].feature=1; f.start();
    f.step({SWAP,0,27});
    const auto& s=f.branch->state();
    CHECK(s.units[0].tile==27 && s.units[1].tile==26);
    CHECK(s.units[1].waiting==0 && s.units[1].attacked==1);
    CHECK(s.units[0].acted && s.units[1].acted);
  }
  Fixture partial; auto a=unit(),b=unit(0,0,27); a.moves=1;b.moves=30;
  a.fortified=b.fortified=1;partial.units={a,b};partial.start();partial.step({SWAP,0,27});
  CHECK(partial.branch->state().units[0].moves==0 && partial.branch->state().units[1].moves==0);
  CHECK(!partial.branch->state().units[0].fortified && !partial.branch->state().units[1].fortified);
}

void combat() {
  Fixture f; f.units={unit(4),unit(0,1,27,1),unit(0,0,25)}; f.start({0,0,{{1,0}}});
  f.step({MELEE,0,27}); const auto& s=f.branch->state();
  CHECK(s.units[1].hp==0 && s.units[0].tile==27 && s.units[0].moves==180);
  CaiUnit dead{}; CHECK(!f.adapter->known_enemy(*f.branch,1,dead));
  CHECK(f.remaining(2,26)>=0);
  CHECK(!f.step({MELEE,0,28}).accepted);
  f.step({MOVE,0,28}); CHECK(f.branch->state().units[0].moves==120);
  Fixture shot; shot.units={unit(1),unit(0,1,28,1),unit(4,0,25)}; shot.start();
  shot.step({RANGED,0,28}); CHECK(shot.branch->state().units[1].hp==0);
  CHECK(shot.branch->state().units[0].tile==26 && shot.remaining(2,28)>=0);
  for (int hp : {1,10,45,100}) {
    Fixture wounded; auto a=unit(3,0,26,hp),d=unit(4,1,27,hp);
    wounded.units={a,d}; wounded.tiles[26].rivers=1<<1; wounded.tiles[27].elevation=1;
    wounded.start(); wounded.step({MELEE,0,27});
    CHECK(wounded.branch->state().units[0].hp>0 || wounded.branch->state().units[1].hp>0);
  }
  for (int a=0;a<5;++a) for (int d=0;d<5;++d) for (int hp : {1,40,100}) {
    Fixture pair;auto defender=unit(d,1,27,hp);
    defender.fortified=(d!=2 && d!=4);pair.units={unit(a,0,26,hp),defender};
    pair.tiles[27].elevation=1;pair.tiles[27].feature=1;pair.start();
    pair.step({ranged[a]?RANGED:MELEE,0,27});
    CHECK(pair.branch->state().units[0].attacked);
  }
}

void turn_and_healing() {
  Fixture f; f.fog=1;f.units={unit(0,0,26,50),unit(4,1,80,60)};f.start();
  f.step({FORTIFY,0,-1}); CHECK(f.branch->state().units[0].waiting);
  CHECK(f.adapter->reachable(*f.branch,0).empty());
  f.step({END_TURN,-1,-1}); CHECK(f.branch->state().units[0].hp==50);
  f.step({SKIP,1,-1}); f.step({END_TURN,-1,-1});
  CHECK(f.branch->state().units[0].hp==60 && f.branch->state().units[0].fortified);
  f.step({MOVE,0,27}); f.step({END_TURN,-1,-1}); f.step({END_TURN,-1,-1});
  CHECK(f.branch->state().units[0].hp==60 && !f.branch->state().units[0].fortified);
  CHECK(f.branch->contacts().turn==0);
}

void contact() {
  Fixture f; f.fog=1;
  for (int i=0;i<96;++i) if (i/12!=2) f.tiles[i].elevation=3;
  auto enemy=unit(0,1,30,1); enemy.acted=1;
  f.units={unit(4,0,25),enemy,unit(0,1,33)};f.start();
  auto sibling=f.branch->clone();
  CaiUnit out{};CHECK(!f.adapter->known_enemy(*f.branch,1,out));
  auto t=f.step({MOVE,0,29}); CHECK(t.contact_interrupted);
  CHECK(f.branch->state().units[0].tile==28 && f.branch->state().units[0].moves==60);
  CHECK(f.adapter->known_enemy(*f.branch,1,out) && out.tile==30);
  CHECK(!f.adapter->known_enemy(*sibling,1,out));
  CHECK(!f.adapter->known_enemy(*f.branch,2,out));
  auto found=f.adapter->prospective_units(*f.branch,t.newly_visible);
  CHECK(std::any_of(found.begin(),found.end(),[](CaiUnit u){return u.id==1;}));
  CHECK(!f.step({MOVE,0,-1}).accepted);
  CHECK(f.adapter->prospective_units(*f.branch,t.newly_visible).size()==found.size());
  CHECK(!f.step({MOVE,0,29}).contact_interrupted);
  CHECK(f.branch->state().units[0].tile==29 && f.branch->state().units[0].moves==0);
  f.step({END_TURN,-1,-1}); f.step({END_TURN,-1,-1});
  f.step({MELEE,0,30}); CHECK(f.branch->state().units[1].hp==0);
  t=f.step({MOVE,0,32}); CHECK(t.contact_interrupted);
  CHECK(f.branch->state().units[0].tile==31 && f.branch->state().units[0].moves==120);
  CHECK(f.adapter->known_enemy(*f.branch,2,out));
}

void visibility() {
  Fixture f;f.fog=1;f.tiles[27].feature=1;f.units={unit(1),unit(0,1,28)};f.start();
  CHECK(!f.adapter->can_see(*f.branch,26,28));
  CHECK(!f.adapter->can_shoot(*f.branch,0,28));
  CHECK(!f.step({RANGED,0,28}).accepted);
  Fixture hill; hill.fog=1;hill.tiles[26].elevation=1;hill.tiles[27].feature=1;
  hill.units={unit(1),unit(0,1,28)};hill.start();
  CHECK(hill.adapter->can_see(*hill.branch,26,28) && hill.adapter->can_shoot(*hill.branch,0,28));
  hill.step({RANGED,0,28});
  CHECK(hill.adapter->can_shoot(*hill.branch,0,28)); // geometry remains usable after spending attack
  Fixture memory;memory.fog=1;memory.units={unit(),unit(0,1,80)};memory.start({0,0,{{1,0}}});
  CaiUnit out{};CHECK(memory.adapter->known_enemy(*memory.branch,1,out) && out.tile==80);
  bool rejected=false;
  try { memory.adapter->prospective_units(*memory.branch,{80}); }
  catch(const std::invalid_argument&) { rejected=true; }
  CHECK(rejected);
  Fixture hidden;hidden.fog=1;
  for(int i=0;i<96;++i) if(i/12!=2) hidden.tiles[i].elevation=3;
  hidden.units={unit(4,0,25),unit(0,1,29)};hidden.start();
  // A never-seen occupied target must remain indistinguishable in reachability.
  CHECK(hidden.remaining(0,29)==0);
  CHECK(hidden.step({MOVE,0,29}).contact_interrupted);
  CHECK(hidden.branch->state().units[0].tile==27);
}

void ownership() {
  Fixture f;f.fog=1;f.units={unit(4),unit(0,1,80)};f.start({},1);
  Info before{};sim_info(f.real,&before);
  auto sibling=f.branch->clone();
  CHECK(sibling->state().tiles==f.branch->state().tiles);
  CHECK(sibling->state().units!=f.branch->state().units);
  CHECK(sibling->state().visibility!=f.branch->state().visibility);
  auto original=sibling->state().units[0];
  f.input_tiles.clear(); f.input_tiles.shrink_to_fit();
  f.input_units.clear(); f.input_units.shrink_to_fit();
  f.visibility.clear();f.visibility.shrink_to_fit();f.tiles.clear();f.units.clear();
  f.adapter->apply(*f.branch,{MOVE,0,27});
  CHECK(sibling->state().units[0].tile==original.tile && sibling->state().units[0].moves==original.moves);
  Info after{};sim_info(f.real,&after);CHECK(before.rng==after.rng && before.turn==after.turn);
  std::vector<Unit> actual(2);sim_units(f.real,-1,actual.data());CHECK(same(actual[0],original));
  auto survivor=f.branch->clone();f.branch.reset();sibling.reset();
  sim_free(f.real);f.real=nullptr;
  CHECK(f.adapter->apply(*survivor,{MOVE,0,28}).accepted);
  f.adapter.reset();CHECK(survivor->state().units[0].tile==28);
  Fixture attack;attack.units={unit(),unit(0,1,27)};attack.start({},1);
  sim_info(attack.real,&before);
  auto predicted=attack.adapter->preview(*attack.branch,{MELEE,0,27});
  auto outcome=attack.adapter->apply(*attack.branch,{MELEE,0,27});
  CHECK(predicted.damage==outcome.result.damage && predicted.retaliation==outcome.result.retaliation);
  sim_info(attack.real,&after);CHECK(before.rng==after.rng);
  sim_units(attack.real,-1,actual.data());CHECK(actual[0].hp==100 && actual[1].hp==100);
}

void bounds_and_invalid() {
  Fixture f;f.w=f.h=16;f.tiles.resize(256);f.units={unit(4,0,255),unit(0,1,0)};f.start();
  CHECK(f.adapter->distance(0,255)>0);
  CHECK(f.remaining(0,254)==180);f.step({MOVE,0,254});
  for (Command c : std::vector<Command>{{-1,0,0},{MOVE,-1,0},{MOVE,512,0},{MOVE,0,4096},
       {MOVE,0,-1},{SWAP,0,0},{RANGED,0,0},{FORTIFY,0,-1}}) CHECK(!f.step(c).accepted);
  auto bad=f.input;bad.visibility_count=255;
  bool rejected=false;try{f.adapter->root(bad,{});}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
  bad=f.input;auto mask=f.visibility;mask[0]=0;bad.visibility=mask.data();
  rejected=false;try{f.adapter->root(bad,{});}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
  for(auto contacts : std::vector<ContactMemory>{{-1,-1,{}},{1,-1,{}},{0,0,{{0,0}}},
       {0,0,{{1,0},{1,0}}},{0,0,{{-1,0}}},{0,0,{{1,1}}},{0,1,{}}}) {
    rejected=false;try{f.adapter->root(f.input,contacts);}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
  }
  auto another=make_rules_adapter(f.input,0);auto foreign=another->root(f.input,{});
  rejected=false;try{f.adapter->actions(*foreign);}catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
  Fixture observer;observer.fog=1;observer.units={unit(4),unit(0,1,80)};observer.start({},0,1);
  CHECK(!observer.branch->state().visibility[26] && observer.branch->state().visibility[80]);
  observer.step({MOVE,0,27});observer.step({END_TURN,-1,-1});
  CHECK(observer.branch->state().visibility[80]);
  Fixture empty;empty.start();CHECK(empty.adapter->actions(*empty.branch).size()==1);
  empty.step({END_TURN,-1,-1});
  Fixture maximum;maximum.units={unit()};maximum.start();
  Info info{};sim_info(maximum.real,&info);info.turn=std::numeric_limits<int32_t>::max();
  CHECK(sim_reset(maximum.real,maximum.units.data(),1,&info)==0);
  maximum.input.turn=info.turn;maximum.branch=maximum.adapter->root(maximum.input,{});
  CHECK(!maximum.step({END_TURN,-1,-1}).accepted);
  CHECK(maximum.branch->state().turn==info.turn);
  for (auto c : maximum.adapter->actions(*maximum.branch)) CHECK(c.kind!=END_TURN);
  for (int parity : {0,1}) for (int wrap : {0,1}) {
    Fixture edge;edge.parity=parity;edge.wrap=wrap;edge.units={unit(4,0,24)};edge.start();
    CHECK(edge.adapter->distance(24,35)==(wrap?1:11));
    if(wrap) edge.step({MOVE,0,35}); else CHECK(!edge.step({MOVE,0,35}).accepted);
  }
}

void random_sequences() {
  std::mt19937 rng(1907);
  for(int fog : {0,1}) for(int parity : {0,1}) for(int wrap : {0,1}) {
    Fixture f;f.fog=fog;f.parity=parity;f.wrap=wrap;
    for(int i=0;i<10;++i) f.units.push_back(unit(i%5,i/5,(i<5?25+i:62+i-5)));
    f.tiles[39].elevation=1;f.tiles[40].feature=1;
    f.tiles[51].rivers=1<<1;f.tiles[52].rivers=1<<4;f.start();
    for(int i=0;i<160;++i) {
      auto commands=f.adapter->actions(*f.branch);
      f.step(commands[rng()%commands.size()]);
    }
  }
}

void corridor(Fixture& f) {
  for (int i=0;i<int(f.tiles.size());++i) if (i/f.w!=2) f.tiles[i].elevation=3;
}

// Independent combat oracle: execute a single attack at a prescribed pose via
// the public ABI, without calling the danger model or its internal helpers.
Result attack_oracle(const Fixture& f, int defender, int attacker, int from, int target) {
  auto units=f.units;
  if (units[defender].tile!=target) units[defender].fortified=0;
  units[defender].tile=target;
  if (units[attacker].tile!=from) units[attacker].fortified=0;
  units[attacker].tile=from;
  units[attacker].moves=moves[units[attacker].type];
  units[attacker].attacked=units[attacker].waiting=0;
  void* real=sim_clone(f.real); CHECK(real);
  Info info{};sim_info(real,&info);info.side=units[attacker].side;info.fog=info.random=0;
  int reset=sim_reset(real,units.data(),int(units.size()),&info);
  if (reset) {sim_free(real);CHECK(!reset);}
  auto result=sim_step(real,{ranged[units[attacker].type]?RANGED:MELEE,attacker,target});
  sim_free(real);CHECK(result.code==OK);return result;
}

std::vector<int> disk(const RulesAdapter& adapter,int origin,int radius) {
  std::vector<int> out{origin};
  size_t begin=0,end=1;
  for(int r=0;r<radius;++r) {
    for(size_t i=begin;i<end;++i) for(int n:adapter.neighbors(out[i]))
      if(std::find(out.begin(),out.end(),n)==out.end()) out.push_back(n);
    begin=end;end=out.size();
  }
  return out;
}

int fog_oracle(const Fixture& f,int target) {
  int count=0;
  const auto& s=f.branch->state();
  for(int id=0;id<s.unit_count;++id) {
    auto u=s.units[id];
    if(u.hp<=0 || u.side==f.observer || !s.visibility[u.tile]) continue;
    for(int hidden:disk(*f.adapter,u.tile,2)) {
      if(s.visibility[hidden] || s.tiles[hidden].elevation>=2 || s.tiles[hidden].feature==5) continue;
      auto affected=disk(*f.adapter,hidden,2);
      if(std::find(affected.begin(),affected.end(),target)!=affected.end()) ++count;
    }
  }
  return count;
}

void danger_damage() {
  for(int a=0;a<5;++a) for(int d=0;d<5;++d) for(int hp:{20,100}) {
    Fixture f;corridor(f);f.units={unit(d,0,26,hp),unit(a,1,ranged[a]?28:27)};
    f.units[0].fortified=(d!=2 && d!=4);
    f.tiles[26].elevation=1;f.tiles[26].feature=1;f.start();
    auto danger=make_danger_model(*f.adapter);
    auto expected=attack_oracle(f,0,1,f.units[1].tile,26);
    int contribution=expected.damage;
    if(expected.retaliation>=f.units[1].hp) contribution/=2;
    CHECK(danger->incoming(*f.branch,0,26)==contribution);
    CHECK(danger->incoming(*f.branch,0,24)==0);
    // A second model uses the same valid branch-owned cache.
    CHECK(make_danger_model(*f.adapter)->incoming(*f.branch,0,26)==contribution);
  }
  Fixture sum;sum.units={unit(0,0,26),unit(1,1,27),unit(1,1,28)};sum.start();
  auto danger=make_danger_model(*sum.adapter);
  CHECK(danger->incoming(*sum.branch,0,26)==
        attack_oracle(sum,0,1,27,26).damage+attack_oracle(sum,0,2,28,26).damage);
  Fixture move;corridor(move);move.units={unit(0,0,25),unit(0,1,28)};
  move.units[0].fortified=1;move.start();
  CHECK(make_danger_model(*move.adapter)->incoming(*move.branch,0,26)==attack_oracle(move,0,1,27,26).damage);
}

void danger_movement() {
  for(int a=0;a<5;++a) {
    Fixture f;corridor(f);f.units={unit(0,0,26),unit(a,1,30)};f.start();
    auto danger=make_danger_model(*f.adapter);
    CHECK((danger->incoming(*f.branch,0,26)>0)==(a==2 || a==4));
  }
  Fixture archer;corridor(archer);archer.units={unit(0,0,26),unit(1,1,29)};archer.start();
  CHECK(make_danger_model(*archer.adapter)->incoming(*archer.branch,0,26)==attack_oracle(archer,0,1,28,26).damage);
  Fixture rough;corridor(rough);rough.tiles[28].feature=1;rough.units=archer.units;rough.start();
  CHECK(make_danger_model(*rough.adapter)->incoming(*rough.branch,0,26)==0);
  Fixture river;corridor(river);river.tiles[29].rivers=1<<4;river.tiles[28].rivers=1<<1;
  river.units=archer.units;river.start();
  CHECK(make_danger_model(*river.adapter)->incoming(*river.branch,0,26)==0);
  // The enemy must spend all movement crossing between the same defender's
  // ZOC cells 28 -> 27; killing that covering unit opens the attack.
  Fixture zoc;corridor(zoc);zoc.tiles[15].elevation=0;
  zoc.units={unit(0,0,26),unit(4,1,29),unit(0,0,15)};zoc.start();
  CHECK(make_danger_model(*zoc.adapter)->incoming(*zoc.branch,0,26)==0);
  Fixture no_zoc;corridor(no_zoc);no_zoc.tiles[15].elevation=0;no_zoc.units=zoc.units;
  no_zoc.units[2].hp=0;no_zoc.start();
  CHECK(make_danger_model(*no_zoc.adapter)->incoming(*no_zoc.branch,0,26)>0);
  Fixture blocked;corridor(blocked);blocked.units={unit(0,0,26),unit(0,1,28),unit(0,0,27)};blocked.start();
  CHECK(make_danger_model(*blocked.adapter)->incoming(*blocked.branch,0,26)==0);
  Fixture los;corridor(los);los.tiles[27].elevation=2;los.units={unit(0,0,26),unit(1,1,28)};los.start();
  CHECK(make_danger_model(*los.adapter)->incoming(*los.branch,0,26)==0);
  // Current exhaustion does not suppress next-turn threat, and estimation
  // must not heal the attacker as END_TURN would.
  Fixture spent;corridor(spent);spent.units={unit(0,0,26),unit(0,1,27,30)};
  spent.units[1].moves=0;spent.units[1].waiting=spent.units[1].attacked=1;spent.start();
  auto expected=attack_oracle(spent,0,1,27,26);
  CHECK(make_danger_model(*spent.adapter)->incoming(*spent.branch,0,26)==expected.damage/(expected.retaliation>=30?2:1));
}

void danger_heuristics() {
  // No visible enemy sources => no fog contribution and no known enemy sight.
  Fixture hidden;corridor(hidden);hidden.fog=1;
  hidden.units={unit(0,0,26),unit(1,1,29)};hidden.start({0,0,{{1,0}}});
  CHECK(!hidden.branch->state().visibility[29]);
  auto damage=attack_oracle(hidden,0,1,28,26).damage;
  CHECK(make_danger_model(*hidden.adapter)->incoming(*hidden.branch,0,26)==damage*80/100);
  // The hidden archer really sees 27, but its privileged position must not
  // become a known sight source. Hypothetical approach positions must not
  // become known sight sources either.
  CHECK(hidden.adapter->can_see(*hidden.branch,29,27));
  CHECK(make_danger_model(*hidden.adapter)->incoming(*hidden.branch,0,27)==
        attack_oracle(hidden,0,1,29,27).damage*80/100);
  Fixture suicide;corridor(suicide);suicide.fog=1;
  suicide.units={unit(3,0,26),unit(0,1,29,1)};suicide.start({0,0,{{1,0}}});
  // Move the defender into reach without adding any real or hypothetical
  // sight source: querying a tile itself does not discover the hidden enemy.
  auto suicidal=attack_oracle(suicide,0,1,28,27);
  CHECK(suicidal.retaliation==1);
  CHECK(make_danger_model(*suicide.adapter)->incoming(*suicide.branch,0,27)==suicidal.damage*80/100/2);
  for(int parity:{0,1}) for(int wrap:{0,1}) {
    Fixture fog;fog.w=5;fog.h=6;fog.tiles.resize(30);fog.fog=1;fog.parity=parity;fog.wrap=wrap;
    for(int i=5;i<10;++i) fog.tiles[i].feature=1;
    fog.units={unit(0,0,10),unit(0,1,12),unit(1,1,11)};fog.start();
    auto danger=make_danger_model(*fog.adapter);
    int expected=fog_oracle(fog,10);
    CHECK(expected>0);
    // Melee at 12 cannot finish its one-step approach at the occupied 11.
    // The archer attacks once, and its damage does not depend on shot origin.
    CHECK(danger->incoming(*fog.branch,0,10)==expected+attack_oracle(fog,0,2,11,10).damage);
  }
}

void danger_cache_and_contacts() {
  Fixture f;corridor(f);f.units={unit(3,0,26),unit(0,1,27,1)};f.start({},1);
  auto danger=make_danger_model(*f.adapter);
  Info before{};sim_info(f.real,&before);
  int incoming=danger->incoming(*f.branch,0,26);CHECK(incoming>0);
  auto sibling=f.branch->clone();
  CHECK(f.adapter->apply(*f.branch,{MELEE,0,27}).accepted);
  CHECK(f.branch->state().units[1].hp==0);
  CHECK(danger->incoming(*f.branch,0,27)==0);
  CHECK(danger->incoming(*f.branch,0,26)==0); // same cached key as before the kill
  CHECK(danger->incoming(*sibling,0,26)==incoming);
  Info after{};sim_info(f.real,&after);CHECK(before.rng==after.rng);
  std::vector<Unit> real(f.units.size());sim_units(f.real,-1,real.data());
  CHECK(real[1].hp==1 && real[0].tile==26);
  // A never-seen source and the same source absent produce identical danger.
  Fixture hidden;corridor(hidden);hidden.fog=1;hidden.units={unit(0,0,26),unit(1,1,29)};hidden.start();
  auto hidden_danger=make_danger_model(*hidden.adapter);
  CHECK(hidden_danger->incoming(*hidden.branch,0,27)==0);
  auto known=hidden.adapter->root(hidden.input,{0,0,{{1,0}}});
  CHECK(hidden_danger->incoming(*known,0,27)>0);
  // A default root cannot inherit contacts or cached values from another root.
  CHECK(hidden_danger->incoming(*hidden.adapter->root(hidden.input,{}),0,27)==0);
  for(int t:{2,4}) {
    auto input=hidden.input;input.turn=t;
    auto aged=hidden.adapter->root(input,{0,0,{{1,0}}});
    CaiUnit u{};CHECK(hidden.adapter->known_enemy(*aged,1,u)==(t==2));
    CHECK((hidden_danger->incoming(*aged,0,27)>0)==(t==2));
  }
  auto branch=hidden.adapter->root(hidden.input,{0,0,{{1,0}}});
  CHECK(hidden_danger->incoming(*branch,0,27)>0);
  for(int i=0;i<4;++i) CHECK(hidden.adapter->apply(*branch,{END_TURN,-1,-1}).accepted);
  CHECK(hidden_danger->incoming(*branch,0,27)==0);
  CHECK(branch->contacts().turn==0);
  auto discovered=hidden.branch->clone();
  CHECK(hidden_danger->incoming(*discovered,0,27)==0);
  CHECK(hidden.adapter->apply(*discovered,{MOVE,0,27}).accepted);
  CHECK(hidden_danger->incoming(*discovered,0,27)>0);
  CHECK(hidden_danger->incoming(*hidden.branch,0,27)==0);
  CHECK(discovered->contacts().turn==0 && hidden.branch->contacts().known_enemies.empty());
  // Perturb a never-seen unit on a candidate attack origin, on the queried
  // destination itself, and beside the destination (possible support). It
  // must be indistinguishable from a dead roster slot in every query.
  Fixture unseen,absent;
  corridor(unseen);corridor(absent);unseen.fog=absent.fog=1;
  unseen.units={unit(0,0,24),unit(4,1,29),unit(0,1,27)};
  absent.units=unseen.units;absent.units[2].hp=0;
  unseen.start({0,0,{{1,0}}});absent.start({0,0,{{1,0}}});
  CHECK(!unseen.branch->state().visibility[27]);
  auto ud=make_danger_model(*unseen.adapter),ad=make_danger_model(*absent.adapter);
  for(int target:{26,27,28}) {
    int expected=ad->incoming(*absent.branch,0,target);CHECK(expected>0);
    CHECK(ud->incoming(*unseen.branch,0,target)==expected);
  }
  // Query the exact same key before and after an actual branch move.
  Fixture move;corridor(move);move.units={unit(4,0,26),unit(0,1,29)};move.start();
  auto md=make_danger_model(*move.adapter);int fortified=md->incoming(*move.branch,0,27);
  CHECK(move.adapter->apply(*move.branch,{MOVE,0,27}).accepted);
  CHECK(md->incoming(*move.branch,0,27)==fortified);
}

void danger_bounds() {
  for(int observer:{0,1}) for(int parity:{0,1}) for(int wrap:{0,1}) {
    Fixture f;f.w=f.h=16;f.tiles.resize(256);f.observer=observer;
    f.parity=parity;f.wrap=wrap;f.units={unit(0,observer,255),unit(0,1-observer,254)};
    f.start({},0,observer);auto danger=make_danger_model(*f.adapter);
    CHECK(danger->incoming(*f.branch,0,255)>0);
    for(auto query:std::vector<std::pair<int,int>>{{-1,255},{1,255},{0,-1},{0,256},{0,254}}) {
      bool rejected=false;try{danger->incoming(*f.branch,query.first,query.second);}
      catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
    }
    auto other=make_rules_adapter(f.input,observer);auto foreign=other->root(f.input,{});
    bool rejected=false;try{danger->incoming(*foreign,0,255);}
    catch(const std::invalid_argument&){rejected=true;}CHECK(rejected);
  }
  CHECK(own_turn(std::numeric_limits<int32_t>::max(),0,0)==1073741824);
}

int main(int argc,char** argv) {
  const std::map<std::string,std::function<void()>> cases={
    {"movement",movement},{"terrain_and_zoc",terrain_and_zoc},{"swaps",swaps},
    {"combat",combat},{"turn_and_healing",turn_and_healing},{"contact",contact},
    {"visibility",visibility},{"ownership",ownership},{"bounds_and_invalid",bounds_and_invalid},
    {"random_sequences",random_sequences},{"danger_damage",danger_damage},
    {"danger_movement",danger_movement},{"danger_heuristics",danger_heuristics},
    {"danger_cache_and_contacts",danger_cache_and_contacts},{"danger_bounds",danger_bounds}};
  try {
    if(argc==2) cases.at(argv[1])();
    else for(const auto& entry:cases) entry.second();
    std::cout<<"CP-03 checks passed\n";return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
