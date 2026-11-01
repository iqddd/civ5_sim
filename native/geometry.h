#pragma once
#include <array>
#include <vector>

#include "combat.h"
struct Geometry {
  int w, h, parity, wrap;
  std::vector<Tile> tiles;
  std::vector<std::array<int, 6>> neighbors;
  std::vector<std::vector<int>> sight, targets;
  Geometry(int width, int height, int row_parity, int wrap_x,
           const Tile* input);
  int index(int x, int global_y) const;
  int distance(int a, int b) const;
  int defense(int tile) const;
  int cost(int tile) const;
  bool passable(int tile) const;
  bool rough(int tile) const;
};
