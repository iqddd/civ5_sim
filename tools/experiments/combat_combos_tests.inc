void combo_mechanism() {
  // A desired occupied endpoint must survive proposal generation even when
  // the incoming unit starts two cells away and arrives with no movement left.
  Fixture f; f.add(0,0,14); f.add(1,0,16); f.add(0,1,20);
  f.corridor({14,15,16,17,18,19,20}); f.start();
  auto intents=f.adapter->move_intents(*f.root,0);
  auto intent=std::find_if(intents.begin(),intents.end(),[](ReachableTile t){return t.tile==16;});
  CHECK(intent!=intents.end());CHECK(intent->remaining_moves==0);
  auto legal=f.adapter->reachable(*f.root,0);
  CHECK(std::none_of(legal.begin(),legal.end(),[](ReachableTile t){return t.tile==16;}));
  CHECK(!f.adapter->apply(*f.root,{MOVE,0,16}).accepted);
  auto hypothetical=f.adapter->assume_move(*f.root,0,16,0);
  auto projected=hypothetical->state();
  CHECK(projected.units[0].tile==16);CHECK(projected.units[0].moves==0);
  auto physical=f.root->state();
  CHECK(physical.units[0].tile==14);CHECK(physical.units[1].tile==16);
  CHECK(physical.units[1].hp==100);
  CHECK(std::equal(f.visibility.begin(),f.visibility.end(),projected.visibility));
  CHECK(hypothetical->contacts().known_enemies.size()==f.root->contacts().known_enemies.size());
  // An omitted or exhausted blocker stays a real obstacle. No proposal may
  // turn its hypothetical vacancy into a physical order to that unit.
  f.selected={0};
  auto omitted=f.search(20);
  for(auto a:omitted.assignments) CHECK(a.unit==0);
  CHECK(f.replay(omitted)->state().units[1].tile==16);
  f.units[1].moves=0;f.selected={0,1};f.start();
  auto exhausted=f.search(20);
  for(auto a:exhausted.assignments) CHECK(a.unit!=1 || a.kind==CAI_INITIAL);
  CHECK(f.replay(exhausted)->state().units[1].tile==16);
  // Vacating a bottleneck must allow a mounted follower from two cells away
  // to enter it and kill the enemy during this turn. Replay validates every
  // physical action and its damage, without using proposal assumptions.
  Fixture free; free.add(4,0,14);free.add(1,0,16);free.add(0,1,17,40);
  free.corridor({14,15,16,17,9});free.start();
  auto plan=free.search(17);CHECK(!plan.assignments.empty());
  CHECK(free.replay(plan)->state().units[2].hp==0);
}
