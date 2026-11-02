  std::vector<ReachableTile> move_intents(const Branch& base, int32_t id) const override {
    const auto& b = branch(base); unit(b, id);
    const auto& u = b.rules.units[id];
    if (u.hp <= 0 || u.moves <= 0 || u.waiting || u.side != b.rules.side) return {};
    // Same movement/ZOC/known-enemy rules as State::paths. The sole difference
    // is permitting a zero-remainder endpoint occupied by our own unit.
    std::vector<int> remaining(geo->tiles.size(), -1);
    std::priority_queue<std::pair<int, int>> queue;
    remaining[u.tile] = u.moves;
    queue.push({u.moves, u.tile});
    while (!queue.empty()) {
      auto [left, from] = queue.top(); queue.pop();
      if (left != remaining[from] || left <= 0) continue;
      for (int to : geo->neighbors[from]) if (geo->passable(to)) {
        int occupied = b.rules.known_unit(to, u.side);
        if (occupied >= 0 && b.rules.units[occupied].side != u.side) continue;
        int after = left - b.rules.movement(u, from, to, left, u.side);
        if (after > remaining[to]) {
          remaining[to] = after; queue.push({after, to});
        }
      }
    }
    std::vector<ReachableTile> result;
    for (int tile = 0; tile < int(remaining.size()); ++tile) if (remaining[tile] >= 0) {
      int left = remaining[tile];
      if (tile != u.tile && geo->distance(u.tile, tile) == 1)
        left = u.moves - b.rules.movement(u, u.tile, tile, u.moves, u.side);
      result.push_back({tile, left});
    }
    return result;
  }
  std::unique_ptr<Branch> assume_move(const Branch& base, int32_t id,
                                    int32_t target, int32_t moves) const override {
    const auto& b = branch(base); unit(b, id); tile(target);
    require(moves >= 0 && moves <= b.rules.units[id].moves, "invalid assumed movement");
    auto copy = std::make_unique<OwnedBranch>(b);
    copy->danger_cache.clear();
    auto& s = copy->rules;
    int blocker = s.known_unit(target, observer);
    require(blocker >= 0 && blocker != id && s.units[blocker].side == observer,
            "assumed move needs a friendly blocker");
    // This branch only estimates the incoming unit's placement. The blocker
    // is absent from this hypothetical vacancy, not killed in the real branch.
    // Keep parent knowledge fixed: this estimate must not reveal hidden units.
    s.units[blocker].hp = 0;
    s.occupied[s.units[id].tile] = -1;
    s.occupied[target] = id;
    s.units[id].tile = target;
    s.units[id].moves = moves;
    s.units[id].acted = 1;
    s.units[id].fortified = 0;
    return copy;
  }
