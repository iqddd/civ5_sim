// Geometry adapter for the locally extracted CP sight and targeting functions.
#include "geometry.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
using std::abs;
using std::vector;
using TeamTypes = int;
using InvisibleTypes = int;
using DirectionTypes = int;
using PlayerTypes = int;
struct CvUnit {};
struct CvVisibilitySources {};
enum {
  NO_DIRECTION = -1,
  DIRECTION_NORTHEAST,
  DIRECTION_EAST,
  DIRECTION_SOUTHEAST,
  DIRECTION_SOUTHWEST,
  DIRECTION_WEST,
  DIRECTION_NORTHWEST
};
#define UNREACHABLE() throw std::logic_error("Invalid sight direction")
#define TRACK_LAST_TARGET_PATH_ENTRY(x)
#define CLEAR_TARGET_PATH_DEBUG
static constexpr int qdir[6] = {0, 1, 1, 0, -1, -1},
                     rdir[6] = {1, 0, -1, -1, 0, 1};
static int floor2(int y) { return int(std::floor(y / 2.0)); }
static int xToHexspaceX(int x, int y) { return x - floor2(y); }
static int hexspaceXToX(int x, int y) { return x + floor2(y); }
static int hexDistance(int q, int r) {
  return std::max({abs(q), abs(r), abs(q + r)});
}
struct CvMap;
struct CvPlot {
  CvMap* map;
  int id;
  int getX() const;
  int getY() const;
  int GetPlotIndex() const { return id; }
  int seeFromLevel(int) const;
  int seeThroughLevel(bool shrub = true) const;
  bool shouldProcessDisplacementPlot(int, int, int, int) const { return true; }
  void changeSightVisibility(int, int, int, bool, bool, CvUnit*, int,
                             CvVisibilitySources*);
  void changeAdjacentSight(int, int, bool, int, int, CvUnit*, int,
                           CvVisibilitySources*);
};
struct CvMap {
  Geometry* g;
  vector<CvPlot> plots;
  vector<int> scratch, collected;
  explicit CvMap(Geometry* p) : g(p), scratch(p->tiles.size(), 0) {
    plots.reserve(g->tiles.size());
    for (int i = 0; i < int(g->tiles.size()); ++i) plots.push_back({this, i});
  }
  CvPlot* plot(int x, int y) {
    int i = g->index(x, y);
    return i < 0 ? nullptr : &plots[i];
  }
  vector<int>& GetVisibilityScratchpad() { return scratch; }
};
static thread_local CvMap* current_map = nullptr;
struct Globals {
  CvMap& getMap() { return *current_map; }
  const int* getPlotDirectionX() { return qdir; }
  const int* getPlotDirectionY() { return rdir; }
  int getTurnRightDirection(int d) { return (d + 1) % 6; }
} GC;
int CvPlot::getX() const { return id % map->g->w; }
int CvPlot::getY() const { return id / map->g->w + map->g->parity; }
int CvPlot::seeFromLevel(int) const {
  const auto& t = map->g->tiles[id];
  return t.elevation == 3 ? 1
                          : 1 + (t.elevation == 1   ? 1
                                 : t.elevation == 2 ? 2
                                                    : 0);
}
int CvPlot::seeThroughLevel(bool shrub) const {
  const auto& t = map->g->tiles[id];
  return (t.elevation == 3 ? 1
                           : 1 + (t.elevation == 1   ? 1
                                  : t.elevation == 2 ? 2
                                                     : 0)) +
         (shrub && (t.feature == 1 || t.feature == 2) ? 1 : 0);
}
void CvPlot::changeSightVisibility(int, int change, int, bool, bool, CvUnit*,
                                   int, CvVisibilitySources*) {
  if (change > 0) map->collected.push_back(id);
}
static CvPlot* plotDirection(int x, int y, int d) {
  return GC.getMap().plot(
      hexspaceXToX(xToHexspaceX(x, y) + qdir[d], y + rdir[d]), y + rdir[d]);
}
static int plotDistance(int x, int y, int a, int b) {
  int dq = xToHexspaceX(a, b) - xToHexspaceX(x, y), dr = b - y;
  int best = hexDistance(dq, dr);
  if (current_map->g->wrap)
    for (int s : {-1, 1})
      best = std::min(best, hexDistance(dq + s * current_map->g->w, dr));
  return best;
}
static CvPlot* plotXYWithRangeCheck(int x, int y, int dq, int dr, int range) {
  return hexDistance(dq, dr) > range
             ? nullptr
             : GC.getMap().plot(hexspaceXToX(xToHexspaceX(x, y) + dq, y + dr),
                                y + dr);
}
struct CvTargeting {
  static bool CanSeeDisplacementPlot(int, int, int, int, int);
};
// Generated from the pinned local CP source, never fetched at runtime.
#include "cp_geometry.inc"

Geometry::Geometry(int width, int height, int row_parity, int wrap_x,
                   const Tile* input)
    : w(width),
      h(height),
      parity(row_parity),
      wrap(wrap_x),
      tiles(input, input + width * height),
      neighbors(width * height),
      sight(width * height),
      targets(width * height) {
  CvMap adapter(this);
  current_map = &adapter;
  for (int i = 0; i < w * h; ++i) {
    int x = i % w, y = i / w + parity;
    for (int d = 0; d < 6; ++d)
      neighbors[i][d] = index(
          hexspaceXToX(xToHexspaceX(x, y) + qdir[d], y + rdir[d]), y + rdir[d]);
    adapter.collected.clear();
    adapter.plots[i].changeAdjacentSight(0, 2, true, 0, NO_DIRECTION, nullptr,
                                         0, nullptr);
    auto& seen = adapter.collected;
    std::sort(seen.begin(), seen.end());
    seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
    sight[i] = seen;
    for (int dr = -2; dr <= 2; ++dr)
      for (int dq = -2; dq <= 2; ++dq) {
        if (hexDistance(dq, dr) > 2 || (!dq && !dr)) continue;
        int j = index(hexspaceXToX(xToHexspaceX(x, y) + dq, y + dr), y + dr);
        if (j >= 0 && CvTargeting::CanSeeDisplacementPlot(
                          x, y, dq, dr, adapter.plots[i].seeFromLevel(0)))
          targets[i].push_back(j);
      }
    std::sort(targets[i].begin(), targets[i].end());
    targets[i].erase(std::unique(targets[i].begin(), targets[i].end()),
                     targets[i].end());
  }
  current_map = nullptr;
}
int Geometry::index(int x, int y) const {
  y -= parity;
  if (y < 0 || y >= h) return -1;
  if (wrap) x = (x % w + w) % w;
  return x < 0 || x >= w ? -1 : y * w + x;
}
int Geometry::distance(int a, int b) const {
  int ya = a / w + parity, yb = b / w + parity;
  int dq = xToHexspaceX(b % w, yb) - xToHexspaceX(a % w, ya), dr = yb - ya;
  int best = hexDistance(dq, dr);
  if (wrap)
    for (int s : {-1, 1}) best = std::min(best, hexDistance(dq + s * w, dr));
  return best;
}
bool Geometry::passable(int i) const {
  return i >= 0 && i < int(tiles.size()) && tiles[i].elevation < 2 &&
         tiles[i].feature != 5;
}
bool Geometry::rough(int i) const {
  return tiles[i].elevation == 1 || tiles[i].feature == 1 ||
         tiles[i].feature == 2;
}
int Geometry::defense(int i) const {
  const auto& t = tiles[i];
  return (t.elevation == 1 ? 25 : 0) + (t.feature == 1 || t.feature == 2   ? 25
                                        : t.feature == 3 || t.feature == 7 ? -15
                                        : t.feature == 4 || t.feature == 6 ? -10
                                                                           : 0);
}
int Geometry::cost(int i) const {
  const auto& t = tiles[i];
  return 60 * ((t.feature == 3                                       ? 3
                : t.feature == 1 || t.feature == 2 || t.feature == 7 ? 2
                                                                     : 1) +
               (t.elevation == 1 ? 1 : 0));
}
