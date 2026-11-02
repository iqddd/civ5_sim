#include "cp_tactical.h"
#include "../combat_state.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#ifndef CAI_CP_AREA
#define CAI_CP_AREA 1
#endif
#ifndef CAI_CP_COMBOS
#define CAI_CP_COMBOS 1
#endif

namespace combat_ai {
namespace {
constexpr int impossible = -1000;
// CvTacticalAI.cpp:addFinishMovesIfAcceptable: reward an untouched unit left
// available after early completion. This is a CP score, not a movement cost.
constexpr int kUntouchedEarlyFinishBonus = 123;
bool passable(const CaiBattleState& s, int t) {
  return s.tiles[t].elevation < 2 && s.tiles[t].feature != 5;
}
bool ranged(int type) { return combat_rules::ranged[type] != 0; }
bool better(const CpMove& a, const CpMove& b) { return a.score > b.score; }
}
CpTactics::CpTactics(const RulesAdapter& a, const Branch& b, const CaiConfig& c,
                     const CaiSearchRequest& r, const std::vector<int32_t>& previous_damage)
    : adapter(a), root(b), config(c), map(b.state()), side(r.side), target(r.target),
      selected(r.selected_units,r.selected_units+r.selected_count),
      index(r.state->unit_count,-1), admitted(r.state->unit_count,0),
      initial(b,r.side) {
  damage_history=previous_damage;
  if (damage_history.empty()) damage_history.assign(initial.units.size(),0);
  if (damage_history.size()!=initial.units.size() ||
      std::any_of(damage_history.begin(),damage_history.end(),[](int n) { return n<0; }))
    throw std::invalid_argument("invalid previous-turn damage history");
  root_visible = initial.visibility;
  for (int i=0; i<int(selected.size()); ++i) {
    index[selected[i]]=i;
    const auto& u=initial.units[selected[i]];
    admitted[u.id]=u.hp>0 && u.moves>0 && !u.waiting;
  }
  // CvDangerPlots::UpdateDangerSingleUnit: fixed possible-attacker lists.
  attackers.resize(r.state->tile_count); fog_danger.assign(r.state->tile_count,0);
  std::vector<bool> known(initial.units.size(),false);
  for (auto contact : b.contacts().known_enemies) known[contact.id]=true;
  for (const auto& u : initial.units) {
    if (u.hp<=0 || u.side==side || !known[u.id]) continue;
    for (int t : adapter.threatened_tiles(b,u.id)) attackers[t].push_back(u.id);
    // Existing CP-only AddFogDanger(origin, enemy, 2, false), coefficient 1.
    if (r.state->fog && root_visible[u.tile])
      for (int hidden=0; hidden<r.state->tile_count; ++hidden)
        if (!root_visible[hidden] && passable(*r.state,hidden) && adapter.distance(u.tile,hidden)<=2)
          for (int t=0; t<r.state->tile_count; ++t)
            if (adapter.distance(hidden,t)<=2) ++fog_danger[t];
  }
}
Damage CpTactics::damage(int attacker, int from, int defender, int to, int extra,
                         bool unknown_origin) {
  std::vector<int> key{attacker,from,defender,to,extra,int(unknown_origin)};
  auto it=attack_cache.find(key);
  if (it!=attack_cache.end()) return it->second;
  auto value=adapter.forecast(root,attacker,from,defender,to,extra,true,unknown_origin);
  attack_cache.emplace(std::move(key),value);
  return value;
}
int CpTactics::danger(int unit, int tile, int self_damage, const std::vector<int>& killed) {
  std::vector<int> key{unit,tile,self_damage}; key.insert(key.end(),killed.begin(),killed.end());
  auto cached=danger_cache.find(key);
  if (cached!=danger_cache.end()) return cached->second;
  int total=fog_danger[tile];
  bool visible=!map.fog;
  for (int enemy : initial.enemies)
    if (adapter.can_see(root,initial.units[enemy].tile,tile)) visible=true;
  for (int enemy : attackers[tile]) {
    const auto& u=initial.units[enemy];
    if (u.tile==tile || std::find(killed.begin(),killed.end(),enemy)!=killed.end()) continue;
    bool unknown=adapter.distance(u.tile,tile)>(ranged(u.type)?2:1);
    auto d=damage(enemy,u.tile,unit,tile,self_damage,unknown);
    int contribution=d.damage;
    if (!visible) contribution=contribution*80/100;
    if (d.retaliation>=u.hp) contribution/=2;
    total+=contribution;
  }
  danger_cache.emplace(std::move(key),total);
  return total;
}
CpBoard CpTactics::board(const TacticalPosition& p) const {
  CpBoard b(*p.branch,side);
  b.plots.resize(b.view.visibility.size());
  for (int t=0; t<int(b.plots.size()); ++t) if (p.plots[t]) {
    auto& plot=b.plots[t]; plot.valid=true;
    int id=b.view.occupied[t];
    if (id>=0) {
      const auto& u=b.view.units[id];
      if (u.side!=side) { plot.enemy=id; b.enemies.push_back(id); }
      else {
        SlotIdx i{index[id]};
        plot.blocked=!admitted[id];
        plot.end=admitted[id] ? p.slot_states[i]==SlotState::Finished : bool(initial.units[id].waiting);
      }
    }
  }
  std::vector<int> distance_enemies=b.enemies;
  // refreshVolatilePlotProperties/checkEdgePlotsForSurprises: visible
  // enemies just outside the tactical area affect distances, not targets.
  for (int t=0; t<int(b.plots.size()); ++t) if (b.plots[t].valid) {
    for (int n : adapter.neighbors(t)) {
      if (adapter.distance(n,target)<adapter.distance(t,target) || b.plots[n].valid) continue;
      if (!b.view.visibility[n]) {
        if (passable(map,n)) b.plots[t].edge=true;
      } else {
        int id=b.view.occupied[n];
        if (id>=0 && b.view.units[id].side!=side &&
            std::find(distance_enemies.begin(),distance_enemies.end(),id)==distance_enemies.end())
          distance_enemies.push_back(id);
      }
    }
  }
  for (int t=0; t<int(b.plots.size()); ++t) if (b.plots[t].valid) {
    auto& plot=b.plots[t];
    for (int id : distance_enemies) {
      int d=adapter.distance(t,b.view.units[id].tile);
      plot.distance=std::min(plot.distance,d); plot.enemies+=d==1;
    }
    for (int n : adapter.neighbors(t)) if (b.plots[n].valid) {
      int id=b.view.occupied[n];
      if (id>=0 && b.view.units[id].side==side) {
        ++plot.friends; plot.friends_end+=b.plots[n].end;
      }
    }
  }
  return b;
}
bool CpTactics::cover(const CpBoard& b, int t) const {
  const auto& plot=b.plots[t];
  if (plot.distance<2) return false;
  if (plot.distance>3) return true;
  for (int n : adapter.neighbors(t))
    if (b.plots[n].valid && b.plots[n].distance<plot.distance && !b.plots[n].end) return false;
  return true;
}
EndScore CpTactics::turn_end(const TacticalPosition& p, const CpBoard& b, SlotIdx i, int t,
    int incoming, bool kill, int extra_self, bool final, bool relaxed, bool full_moves) const {
  const auto& u=initial.units[selected[i.value]];
  const auto& plot=b.plots[t];
  int friends=(final?plot.friends_end:plot.friends)+int(relaxed);
  bool acceptable=!(plot.enemies>3 || (plot.enemies==3 && bias<1.f));
  if (cover(b,t)) incoming/=2;
  if (plot.edge) incoming=std::max(50,incoming);
  int result=0, remaining=std::max(1,b.view.units[u.id].hp-extra_self);
  if (incoming>0) {
    if (config.aggression!=CAI_BRAVEHEART) {
      if (ranged(u.type) && incoming>remaining && (plot.enemies>0 || friends==0)) acceptable=false;
      int magic=kill || safe[i]==0 ? 23 : 42;
      if (b.enemies.size()==1 && kill) magic=12;
      if (remaining*(friends+1)<magic*incoming/remaining) acceptable=false;
    }
    incoming/=friends+1;
    int scaled=incoming*100/std::max(1,u.hp);
    if (scaled>225) scaled=15*int(std::sqrt(double(scaled)));
    result-=scaled;
  }
  // Pinned ScorePlotForCombatUnitMove passes remaining==maxMoves directly
  // as bHasMoved. Preserve that call convention, including its inverted name.
  if (!full_moves) {
    if (u.hp<100) ++result;
    if (plot.distance<3 && u.type!=2 && u.type!=4) ++result;
  }
  if (friends>0) result+=7;
  bool visible=!map.fog;
  for (int e : initial.enemies) if (adapter.can_see(root,initial.units[e].tile,t)) visible=true;
  if (!visible) ++result;
  result+=adapter.defense(t)/5;
  (void)p;
  return {result,acceptable};
}
CpMove CpTactics::move_score(const TacticalPosition& p, const CpBoard& b, SlotIdx i,
                            int t, int remaining, bool final, bool* acceptable, bool initial_eval) {
  const auto& u=b.view.units[selected[i.value]];
  CpMove move{u.id,t,CP_MOVE,impossible,remaining};
  if (acceptable) *acceptable=false;
  if (!b.plots[t].valid) return move;
  int base=formation_score(roles[i],b.enemies.empty()?adapter.distance(t,target):b.plots[t].distance,!b.enemies.empty());
  if (!b.enemies.empty() && p.last_kill[i]) base=std::max(60,base);
  int self=initial.units[u.id].hp-u.hp;
  int incoming=danger(u.id,t,self,p.killed), penalty=0;
  if (final) {
    auto end=turn_end(p,b,i,t,incoming,p.last_kill[i]!=0,0,!initial_eval,false,remaining==combat_rules::moves[u.type]);
    if (!end.acceptable) return move;
    penalty=end.score-self;
  } else {
    int overkill=100*incoming/std::max(1,u.hp);
    penalty=-overkill/20;
    if (overkill>300 && remaining<180 && u.moves>180) return move;
  }
  if (ranged(u.type)) penalty+=visibility_score.empty()?0:visibility_score[t];
  move.score=base+penalty+remaining/60;
  if (acceptable) *acceptable=true;
  return move;
}
CpMove CpTactics::attack(const TacticalPosition& p, const CpBoard& b, SlotIdx i, int from, int enemy) {
  const auto& u=b.view.units[selected[i.value]]; const auto& e=b.view.units[enemy];
  CpMove result{u.id,e.tile,ranged(u.type)?CP_RANGED:CP_MELEE,impossible};
  auto d=damage(u.id,from,enemy,e.tile,initial.units[enemy].hp-e.hp);
  if (d.code) return result;
  int dealt=d.damage, received=d.retaliation;
  if (!ranged(u.type)) {
    int friends=b.plots[e.tile].friends-(adapter.distance(from,e.tile)==1?1:0);
    int delta=friends-b.plots[from].enemies;
    dealt+=delta*dealt/10; received-=delta*received/10;
  }
  result.near_kill=dealt>=e.hp-10 ? enemy : -1;
  result.kill=dealt>=e.hp-1;
  if (result.kill) dealt=std::min(dealt,e.hp+10);
  if (received>0) {
    constexpr float factors[5]={0.f,.7f,1.1f,2.3f,4.2f};
    // This is the actual pinned CP initializer/indexing, not a shifted table.
    constexpr int hp_limit[5]={70,40,20,-5,0};
    auto ignored=p.killed;
    if (result.near_kill>=0) ignored.push_back(result.near_kill);
    int incoming=danger(u.id,result.kill?e.tile:from,received,ignored);
    int delta=int(dealt*bias*factors[config.aggression]+.5f)-received;
    if (config.aggression==CAI_NONE ||
        ((incoming>100 || initial.units[u.id].hp-received<hp_limit[config.aggression]) && delta<0 && !result.kill) ||
        (initial.units[u.id].hp-received<3 && config.aggression!=CAI_BRAVEHEART)) return result;
  }
  result.score=dealt; result.self_damage=received;
  return result;
}
int CpTactics::potential(const TacticalPosition& p, const CpBoard& b, SlotIdx i, int t,
                         int num_attacks, int& near_kill) {
  const auto& u=b.view.units[selected[i.value]];
  int best=0; near_kill=-1;
  for (int id : b.enemies) {
    int to=b.view.units[id].tile, distance=adapter.distance(t,to);
    if (distance<1 || distance>(ranged(u.type)?2:1) ||
        (ranged(u.type) && !adapter.can_target(t,to))) continue;
    auto value=attack(p,b,i,t,id);
    if (value.kill && num_attacks>0 && value.score>0) {
      value.score+=100; near_kill=value.near_kill;
    }
    best=std::max(best,value.score);
  }
  return num_attacks>0 ? num_attacks*best/2 : best/4;
}
bool CpTactics::can_end(const TacticalPosition& p, const CpBoard& b, SlotIdx i, const CpMove& m, int t) {
  if (!b.plots[t].valid) return false;
  if (safe[i]==0) return true;
  int self=initial.units[m.unit].hp-b.view.units[m.unit].hp;
  int incoming=danger(m.unit,t,self,p.killed);
  int count=0; for (SlotState state : p.slot_states) count+=state==SlotState::Pending;
  return turn_end(p,b,i,t,incoming,m.kill,m.self_damage,true,count>1,
                  m.remaining==combat_rules::moves[b.view.units[m.unit].type]).acceptable;
}
std::vector<CpMove> CpTactics::preferred(const TacticalPosition& p, const CpBoard& b, SlotIdx i, int limit) {
  std::vector<CpMove> moves, shots;
  const auto& u=b.view.units[selected[i.value]];
  if (p.slot_states[i]!=SlotState::Pending || !b.plots[u.tile].valid) return moves;
  for (auto reach : adapter.move_intents(*p.branch,u.id)) {
    int t=reach.tile;
    if ((CAI_CP_AREA && adapter.distance(target,t)>6) || !b.plots[t].valid || b.plots[t].blocked) continue;
    if (b.plots[t].enemy>=0) {
      if (config.aggression==CAI_NONE || ranged(u.type) || u.attacked || adapter.distance(u.tile,t)>1) continue;
      auto a=attack(p,b,i,u.tile,b.plots[t].enemy);
      if (a.score<0) continue;
      a.remaining=adapter.attack_moves(*p.branch,{combat_rules::MELEE,u.id,t});
      int after=a.kill?t:u.tile;
      if (a.kill && b.plots[t].enemies>3) continue;
      if (a.remaining==0 && !can_end(p,b,i,a,after)) continue;
      a.score=(a.score+8)*10+(a.kill?1000:0)+(t==target?3:0);
      a.score+=move_score(p,b,i,after,a.remaining,false).score;
      moves.push_back(a); continue;
    }
    auto m=move_score(p,b,i,t,reach.remaining_moves,false);
    if (t==u.tile) {
      if (config.aggression!=CAI_NONE && ranged(u.type) && !u.attacked && u.moves>0) {
        for (int enemy : b.enemies) {
          int to=b.view.units[enemy].tile;
          if (!adapter.can_shoot(*p.branch,u.id,to)) continue;
          auto a=attack(p,b,i,t,enemy);
          if (a.score<0) continue;
          a.remaining=0; // All supported ranged types end their turn on attack.
          if (!can_end(p,b,i,a,t)) continue;
          a.score*=10;
          if (u.moves>60 && b.plots[t].distance==1 && roles[i]!=CAI_FIRST_LINE) a.score/=2;
          a.score+=(a.kill?1000:0)+(to==target?2:0)+m.score;
          shots.push_back(a);
        }
      }
      if (shots.empty() && can_end(p,b,i,m,t)) {
        if (reach.remaining_moves==combat_rules::moves[u.type]) {
          if (initial.units[u.id].hp<95) m.score+=std::min(23,100-initial.units[u.id].hp);
          if (u.type!=2 && u.type!=4) m.score+=23;
        }
        int ignored=-1; m.score+=potential(p,b,i,t,1,ignored)*10;
        m.kind=CP_FINISH_TEMP; moves.push_back(m);
      }
    } else {
      if (p.move_flags[i]!=MoveFlag::None || b.plots[t].end) continue;
      int attacks=u.attacked?0:std::min(1,(reach.remaining_moves+59)/60);
      m.score+=potential(p,b,i,t,attacks,m.near_kill)*10;
      if ((m.remaining>60 && u.type==4) || can_end(p,b,i,m,t)) {
        m.near_kill=-1; moves.push_back(m);
      }
    }
  }
  if (int(shots.size())>=limit && limit>1) {
    std::stable_sort(shots.begin(),shots.end(),better); shots.resize(limit-1);
  }
  moves.insert(moves.end(),shots.begin(),shots.end());
  std::stable_sort(moves.begin(),moves.end(),better);
  if (int(moves.size())>limit) {
    int count=limit;
    while (count<int(moves.size()) && moves[count].score==moves[count-1].score) ++count;
    moves.resize(count);
  }
  moves.push_back({u.id,u.tile,CP_BLOCKED,-10,u.moves});
  return moves;
}
void CpTactics::initialize(TacticalPosition& p) {
  const auto& s=map;
  p.plots.assign(s.tile_count,0);
  int plot_count=0;
  auto add_plot=[&](int t) { if (!p.plots[t] && plot_count<255) { p.plots[t]=1; ++plot_count; } };
  std::vector<int> first_enemies;
  for (int t=0; t<s.tile_count; ++t) if (root_visible[t] && adapter.distance(t,target)<=5) {
    add_plot(t); int id=initial.occupied[t];
    if (id>=0 && initial.units[id].side!=side) first_enemies.push_back(id);
  }
  for (int id : first_enemies) for (int t=0; t<s.tile_count; ++t)
    if (root_visible[t] && adapter.distance(initial.units[id].tile,t)<=2) add_plot(t);
  // FindBestUnitAssignments counts target-area enemies BEFORE addAvailableUnit.
  // Only admitted participants may subsequently add their neighborhoods.
  bool have_target_enemies=!board(p).enemies.empty();
  roles.assign(selected.size(),CAI_FIRST_LINE); safe.assign(selected.size(),0);
  for (int slot_index=0; slot_index<int(selected.size()); ++slot_index) {
    SlotIdx i{slot_index};
    const auto& u=initial.units[selected[i.value]];
    roles[i]=movement_role(u,damage_history[u.id]);
    bool projected_to_die=damage_history[u.id]>u.hp;
    if (!admitted[u.id] || (!have_target_enemies && projected_to_die)) {
      p.slot_states[i]=SlotState::Blocked; admitted[u.id]=0;
    }
  }
  for (int id : selected) if (admitted[id]) {
    int center=initial.units[id].tile;
    // iterateRingPlots starts with ring zero. At the 255-plot boundary a
    // neighbor must not consume the last slot before the participant itself.
    add_plot(center);
    for (int n : adapter.neighbors(center)) add_plot(n);
  }
  if (!CAI_CP_AREA) std::fill(p.plots.begin(),p.plots.end(),1); // Benchmark ablation only.
  auto b=board(p); initial_enemies=b.enemies;
  int ours=0;
  for (int id : selected) ours+=admitted[id];
  for (int t=0; t<s.tile_count; ++t)
    if (b.plots[t].blocked && b.plots[t].distance<3) ++ours;
  bias=std::max(.9f,std::sqrt(float(ours)/std::max(1,int(initial_enemies.size()))));
  visibility_score.assign(s.tile_count,0);
  for (int t=0; t<s.tile_count; ++t) if (p.plots[t])
    for (int n=0; n<s.tile_count; ++n)
      if (p.plots[n] && b.plots[n].distance<=3 && adapter.distance(t,n)==2 && adapter.can_see(root,t,n))
        ++visibility_score[t];
  for (int slot_index=0; slot_index<int(selected.size()); ++slot_index) {
    SlotIdx i{slot_index};
    const auto& u=initial.units[selected[i.value]];
    initial_distance.push_back(b.plots[u.tile].distance);
    if (!admitted[u.id]) { p.slot_states[i]=SlotState::Blocked; continue; }
    // Root-only safe-plot count, before the tactical/non-sim plot filter.
    for (auto reach : adapter.move_intents(root,u.id))
      if ((!CAI_CP_AREA || adapter.distance(reach.tile,target)<=6) &&
          (initial.occupied[reach.tile]<0 || initial.occupied[reach.tile]==u.id) &&
          danger(u.id,reach.tile,0,{})<u.hp) ++safe[i];
    active_order.push_back(i);
  }
  // dropSuperfluousUnits: importance is the first preferred action, including
  // BLOCKED (-10). Retain stable input order on equal scores.
  SlotValues<int32_t> importance(selected.size(),0);
  int keep=std::min(int(CAI_MAX_SELECTED),int(active_order.size()));
  for (SlotIdx i : active_order) {
    if (keep<=0) break;
    auto choices=preferred(p,b,i,1);
    importance[i]=choices.empty()?impossible:choices.front().score;
    const auto& u=initial.units[selected[i.value]];
    if (u.hp<50) { // Default leader offense=0; no leader metadata in the ABI.
      if (!choices.empty() && choices.front().kind==CP_MOVE && u.moves<180) {
        importance[i]=impossible; --keep;
      } else importance[i]/=2;
    }
  }
  std::stable_sort(active_order.begin(),active_order.end(),[&](SlotIdx a,SlotIdx b) { return importance[a]>importance[b]; });
  keep=std::max(0,keep);
  for (int j=keep; j<int(active_order.size()); ++j) p.slot_states[active_order[j]]=SlotState::Blocked;
  active_order.resize(keep);
}
void CpTactics::extend_plots(TacticalPosition& p, const std::vector<int32_t>& added) const {
  int count=std::accumulate(p.plots.begin(),p.plots.end(),0);
  for (int t : added) if (!p.plots[t] && count<255) { p.plots[t]=1; ++count; }
}
std::vector<CpCombo> CpTactics::candidates(const TacticalPosition& p, bool depth_first) {
  auto b=board(p);
  std::vector<CpMove> overall;
  for (SlotIdx i : active_order) if (p.slot_states[i]==SlotState::Pending) {
    auto choices=preferred(p,b,i,depth_first?2:config.choices_per_unit);
    overall.insert(overall.end(),choices.begin(),choices.end());
  }
  std::stable_sort(overall.begin(),overall.end(),better);
  std::vector<CpCombo> result;
  auto add=[&](CpMove a, CpMove second=CpMove{}) {
    CpCombo combo{{a},a.score};
    if (second.unit>=0 && second.unit!=a.unit) { combo.moves.push_back(second); combo.score+=second.score; }
    result.push_back(std::move(combo));
  };
  size_t max_candidates=std::max(size_t((depth_first?1:config.max_branches)*5),overall.size()/2);
  for (size_t k=0; k<std::min(overall.size(),max_candidates); ++k) {
    auto primary=overall[k];
    if (primary.kind!=CP_MOVE) {
      if (depth_first && k+1<overall.size() && overall[k+1].unit!=primary.unit &&
          (overall[k+1].kind==CP_FINISH_TEMP || overall[k+1].kind==CP_BLOCKED)) add(primary,overall[++k]);
      else add(primary);
      continue;
    }
    int blocker=b.view.occupied[primary.tile];
    if (blocker<0) { add(primary); continue; }
    if (!CAI_CP_COMBOS || index[blocker]<0 || p.slot_states[SlotIdx{index[blocker]}]!=SlotState::Pending) continue;
    for (auto freeing : overall) if (freeing.unit==blocker && freeing.kind==CP_MOVE) {
      if (primary.score+freeing.score < -10) continue;
      if (freeing.tile==b.view.units[primary.unit].tile) {
        if (p.move_flags[SlotIdx{index[blocker]}]==MoveFlag::Swapped) continue;
        // Simulator swaps have explicit physical restrictions in addition to
        // CP's reciprocal intentions (notably adjacency and available moves).
        if (!adapter.can_execute(*p.branch,{combat_rules::SWAP,primary.unit,primary.tile})) continue;
        primary.kind=CP_SWAP; freeing.kind=CP_SWAP_REVERSE; add(primary,freeing); break;
      }
      if (b.view.occupied[freeing.tile]<0) {
        freeing.kind=CP_FORCED; add(freeing,primary); break;
      }
    }
  }
  std::stable_sort(result.begin(),result.end(),[](const CpCombo& a,const CpCombo& b) { return a.score>b.score; });
  return result;
}
FinalEvaluation CpTactics::evaluate_final_position(const TacticalPosition& position) {
  FinalEvaluation evaluation;
  evaluation.score = position.rank;
  auto current = board(position);
  int unsafe_positions = 0;
  int positive_changes = 0;
  int negative_changes = 0;
  int target_distance_before = 0;
  int target_distance_after = 0;

  for (int slot_index = 0; slot_index < int(selected.size()); ++slot_index) {
    SlotIdx slot{slot_index};
    UnitId unit_id{selected[slot.value]};
    const auto& unit = current.view.units[unit_id.value];
    const auto& original = initial.units[unit_id.value];
    if (position.slot_states[slot] == SlotState::Blocked || original.moves <= 0 || original.waiting)
      continue;
    if (unit.hp <= 0) {
      evaluation.score -= original.hp;
      continue;
    }
    if (position.slot_states[slot] == SlotState::Pending) {
      if (!position.touched[slot]) evaluation.score += kUntouchedEarlyFinishBonus;
      continue;
    }

    bool safe_end = false;
    int end_score = move_score(position,current,slot,unit.tile,0,true,&safe_end).score;
    if (!safe_end) ++unsafe_positions;
    auto latest_move = std::find_if(position.assignments.rbegin(),position.assignments.rend(),
        [&](const CaiAssignment& record) {
      return record.unit == unit_id.value && (record.kind == CAI_MOVE || record.kind == CAI_SWAP ||
             (record.kind == CAI_MELEE && record.killed_unit >= 0));
    });
    evaluation.score += latest_move == position.assignments.rend()
        ? end_score : std::max(0,end_score-latest_move->score);

    int distance = current.plots[unit.tile].distance;
    if (roles[slot] == CAI_THIRD_LINE) {
      positive_changes += initial_distance[slot] == 1 && distance > 1;
      negative_changes += initial_distance[slot] > 1 && distance == 1;
    } else {
      int ideal_distance = roles[slot] == CAI_FIRST_LINE ? 1 : 2;
      positive_changes += initial_distance[slot] != ideal_distance && distance == ideal_distance;
      negative_changes += initial_distance[slot] == ideal_distance && distance != ideal_distance;
    }
    target_distance_before += adapter.distance(original.tile,target);
    target_distance_after += adapter.distance(unit.tile,target);
    evaluation.finishes.push_back(stationary_assignment(CAI_FINISH,unit,roles[slot],end_score));
  }

  int loss_limit = config.risk_budget_base + config.risk_budget_per_kill * int(position.killed.size());
  bool attacked = std::any_of(position.assignments.begin(),position.assignments.end(),
      [](const CaiAssignment& record) { return record.kind == CAI_MELEE || record.kind == CAI_RANGED; });
  bool improved = !position.killed.empty() || position.discovered ||
      (!current.enemies.empty() ? (positive_changes > negative_changes ||
       (positive_changes == negative_changes && attacked)) : target_distance_after < target_distance_before);
  evaluation.accepted = unsafe_positions <= loss_limit && improved;
  return evaluation;
}
} // namespace combat_ai
