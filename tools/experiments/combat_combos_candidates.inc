  std::vector<Candidate> candidates(const TacticalPosition& parent) {
    auto old = view(*parent.branch);
    std::vector<std::vector<Command>> per_unit(selected.size());
    std::vector<std::vector<int>> assumed_moves(selected.size(),
                                                std::vector<int>(old.visibility.size(), -1));
    std::vector<Command> legal_swaps;
    for (Command c : adapter->actions(*parent.branch)) {
      if (c.kind == END_TURN || c.unit < 0) continue;
      int i = selected_index[c.unit];
      if (i < 0 || parent.done[i] || c.kind == FORTIFY || c.kind == SKIP) continue;
      if ((c.kind == MELEE || c.kind == RANGED) && config.aggression == CAI_NONE) continue;
      if (c.kind == MOVE && (parent.moved[i] || !in_area(old, c.target))) continue;
      // Swaps are resolved from reciprocal preferred movement intents below.
      if (c.kind == SWAP) { legal_swaps.push_back(c); continue; }
      per_unit[i].push_back(c);
    }
    for (int i = 0; i < int(selected.size()); ++i) {
      if (parent.done[i] || parent.moved[i]) continue;
      for (auto target : adapter->move_intents(*parent.branch, selected[i])) {
        int blocker = old.occupied[target.tile];
        if (blocker < 0 || blocker == selected[i] || old.units[blocker].side != request.side) continue;
        int j = selected_index[blocker];
        if (j < 0 || parent.done[j] || parent.moved[j] || !in_area(old, target.tile)) continue;
        per_unit[i].push_back({MOVE, selected[i], target.tile});
        assumed_moves[i][target.tile] = target.remaining_moves;
      }
    }
    auto compare = [](const Candidate& a, const Candidate& b) {
      return std::make_tuple(-a.score, a.selected_order, a.kind, a.tile) <
             std::make_tuple(-b.score, b.selected_order, b.kind, b.tile);
    };
    std::vector<Candidate> overall;
    for (int i = 0; i < int(selected.size()); ++i) {
      if (parent.done[i]) continue;
      std::vector<Candidate> choices;
      int initial_score = plot_score(parent, old, i, false);
      for (Command command : per_unit[i]) {
        auto child = copy(parent);
        if (command.kind == MOVE && assumed_moves[i][command.target] >= 0) {
          child->branch = adapter->assume_move(*parent.branch, selected[i], command.target,
                                               assumed_moves[i][command.target]);
        } else if (!apply(*child, command, i)) { ++stats.pruned; continue; }
        auto next = view(*child->branch);
        int gain = plot_score(*child, next, i, false) - initial_score +
                   child->attack_score - parent.attack_score;
        if (command.kind == MOVE) gain += potential(*child, selected[i]);
        choices.push_back({{command}, gain, i, command.kind, command.target});
      }
      std::stable_sort(choices.begin(), choices.end(), compare);
      int limit = depth_first ? 2 : config.choices_per_unit;
      if (int(choices.size()) > limit) {
        auto move = std::find_if(choices.begin() + limit, choices.end(), [](const Candidate& c) {
          return c.kind == MOVE;
        });
        bool have_move = std::any_of(choices.begin(), choices.begin() + limit, [](const Candidate& c) {
          return c.kind == MOVE;
        });
        if (!have_move && move != choices.end()) std::swap(choices[limit - 1], *move);
        choices.resize(limit);
      }
      for (auto& c : choices) overall.push_back(std::move(c));
      bool acceptable;
      int finish_gain = plot_score(parent, old, i, true, &acceptable) - initial_score;
      if (!parent.touched[i]) {
        finish_gain += std::min(23, 100 - old.units[selected[i]].hp);
        if (old.units[selected[i]].type != 2 && old.units[selected[i]].type != 4) finish_gain += 23;
      }
      finish_gain += potential(parent, selected[i], true);
      overall.push_back({{}, finish_gain, i, SKIP, old.units[selected[i]].tile});
    }
    std::stable_sort(overall.begin(), overall.end(), compare);
    std::vector<Candidate> result;
    // CvTacticalPosition::makeNextAssignments: start with preferred desired
    // moves. Only a desired occupied destination triggers blocker resolution.
    size_t limit = std::max(size_t((depth_first ? 1 : config.max_branches) * 5),
                            overall.size() / 2);
    for (size_t index = 0; index < std::min(overall.size(), limit); ++index) {
      const auto& primary = overall[index];
      if (primary.kind != MOVE) { result.push_back(primary); continue; }
      int blocker = old.occupied[primary.tile];
      if (blocker < 0) { result.push_back(primary); continue; }
      int j = selected_index[blocker];
      if (j < 0 || parent.done[j] || parent.moved[j]) continue;
      for (const auto& freeing : overall) {
        if (freeing.selected_order != j || freeing.kind != MOVE) continue;
        int combined = primary.score + freeing.score;
        if (combined < -10) continue; // CP allows mildly negative frontline rotations.
        int original = old.units[selected[primary.selected_order]].tile;
        if (freeing.tile == original) {
          // Reciprocal desired destinations: physical SWAP moves both units.
          bool legal = std::any_of(legal_swaps.begin(), legal_swaps.end(), [&](Command c) {
            return c.unit == selected[primary.selected_order] && c.target == primary.tile;
          });
          if (!legal) continue;
          result.push_back({{{SWAP, selected[primary.selected_order], primary.tile}},
                             combined, primary.selected_order, SWAP, primary.tile});
          break;
        }
        if (old.occupied[freeing.tile] >= 0) continue; // Only one blocker level.
        result.push_back({{freeing.commands.front(), primary.commands.front()},
                           combined, j, MOVE, primary.tile});
        // selected_order names the forced blocker, which may move again later.
        break; // Best non-blocked preferred move, not every freeing alternative.
      }
    }
    std::stable_sort(result.begin(), result.end(), compare);
    return result;
  }
