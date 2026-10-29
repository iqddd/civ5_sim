"""Subset of the Civ V map API used by the original terrain pipeline.

No game units, starts, resources, natural wonders or economy are simulated here.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from lupa.lua54 import LuaRuntime

from .native import ROOT, Native

DIRECTIONS = ((0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1))


def neighbor(x, y, direction):
    dq, dr = DIRECTIONS[int(direction)]
    return x - y // 2 + dq + (y + dr) // 2, y + dr


def distance(dq, dr):
    return max(abs(dq), abs(dr), abs(dq + dr))


def scalar(value):
    if value in ("true", "false"):
        return value == "true"
    try:
        return int(value)
    except (ValueError, TypeError):
        return value


class Area:
    def __init__(self, ident, plots):
        self.ident, self.plots = ident, plots

    def GetID(self):
        return self.ident

    def GetNumTiles(self):
        return len(self.plots)

    def GetNumRiverEdges(self):
        return sum(sum(f >= 0 for f in p.rivers) for p in self.plots)


class Plot:
    def __init__(self, world, x, y):
        self.world, self.x, self.y = world, x, y
        self.kind, self.terrain, self.feature, self.area = 3, -1, -1, -1
        # Owned edges: E, SE, SW; flow direction -1 means absent.
        self.rivers = [-1, -1, -1]

    def GetX(self):
        return self.x

    def GetY(self):
        return self.y

    def GetPlotType(self):
        return self.kind

    def GetTerrainType(self):
        return self.terrain

    def GetFeatureType(self):
        return self.feature

    def GetArea(self):
        return self.area

    def Area(self):
        return self.world.areas[self.area]

    def SetArea(self, value):
        self.area = int(value)

    def IsWater(self):
        return self.kind == 3

    def IsMountain(self):
        return self.kind == 0

    def IsHills(self):
        return self.kind == 1

    def IsFlatlands(self):
        return self.kind == 2

    def IsImpassable(self):
        return self.IsMountain() or (
            self.feature >= 0
            and self.world.features[self.feature].get("Impassable", False)
        )

    def SetTerrainType(self, value, *_):
        self.terrain = int(value)

    def SetFeatureType(self, value, *_):
        self.feature = int(value)

    def IsWOfRiver(self):
        return self.rivers[0] >= 0

    def IsNWOfRiver(self):
        return self.rivers[1] >= 0

    def IsNEOfRiver(self):
        return self.rivers[2] >= 0

    def SetWOfRiver(self, value, flow):
        self.rivers[0] = int(flow) if value else -1

    def SetNWOfRiver(self, value, flow):
        self.rivers[1] = int(flow) if value else -1

    def SetNEOfRiver(self, value, flow):
        self.rivers[2] = int(flow) if value else -1

    def adjacent(self):
        return [
            p
            for d in range(6)
            if (p := self.world.direction(self.x, self.y, d)) is not None
        ]

    def SetPlotType(self, value, recalculate=True, *_):
        # The SDK's BasicLuaMethod integer conversion maps nil to zero.
        value = 0 if value is None else int(value)
        if self.kind != value:
            was_water = self.IsWater()
            if self.kind == 3 or value == 3:
                self.feature, self.rivers = -1, [-1, -1, -1]
            self.kind = value
            types = self.world.types
            if (
                self.terrain == -1
                or (self.terrain in (types["TERRAIN_COAST"], types["TERRAIN_OCEAN"]))
                != self.IsWater()
            ):
                name = (
                    ("TERRAIN_COAST" if self.IsAdjacentToLand() else "TERRAIN_OCEAN")
                    if self.IsWater()
                    else "TERRAIN_GRASS"
                )
                self.terrain = types[name]
            if recalculate:
                if was_water != self.IsWater():
                    for p in self.adjacent():
                        if p.IsWater():
                            p.terrain = types[
                                "TERRAIN_COAST"
                                if p.IsAdjacentToLand()
                                else "TERRAIN_OCEAN"
                            ]
                self.world.recalculate()

    def IsAdjacentToLand(self):
        return any(not p.IsWater() for p in self.adjacent())

    def IsAdjacentToShallowWater(self):
        return any(
            p.terrain == self.world.types["TERRAIN_COAST"] for p in self.adjacent()
        )

    def IsLake(self):
        return (
            self.IsWater()
            and self.area >= 0
            and self.Area().GetNumTiles() <= self.world.defines["LAKE_MAX_AREA_SIZE"]
        )

    def IsCoastalLand(self):
        return not self.IsWater() and any(
            p.IsWater() and not p.IsLake() for p in self.adjacent()
        )

    def IsRiver(self):
        if any(f >= 0 for f in self.rivers):
            return True
        # Opposite edges belong to NE, W, NW neighbors respectively.
        return any(
            p is not None and p.rivers[e] >= 0
            for d, e in ((0, 2), (4, 0), (5, 1))
            if (p := self.world.direction(self.x, self.y, d)) is not None
        )

    def IsFreshWater(self):
        if self.IsWater() or self.IsImpassable():
            return False
        return self.IsRiver() or any(
            p.IsLake() or p.feature == self.world.types["FEATURE_OASIS"]
            for p in [self] + self.adjacent()
        )

    def GetInlandCorner(self):
        order = list(range(4))
        for i in range(4):
            j = i + self.world.native.rand(4 - i)
            order[i], order[j] = order[j], order[i]
        for i in order:
            p = (
                self
                if i == 0
                else self.world.direction(self.x, self.y, {1: 0, 2: 5, 3: 4}[i])
            )
            if (
                p is not None
                and not p.IsWater()
                and not any(
                    n is not None and n.IsWater()
                    for n in [
                        self.world.direction(p.x, p.y, 1),
                        self.world.direction(p.x, p.y, 2),
                    ]
                )
            ):
                return p
        return None

    def CanHaveFeature(self, ident):
        ident = int(ident)
        if ident == -1:
            return True
        if self.feature != -1 or self.IsMountain():
            return False
        info = self.world.features[ident]
        if self.terrain not in self.world.feature_terrains.get(ident, set()):
            return False
        if info.get("NoCoast") and self.IsCoastalLand():
            return False
        if info.get("NoRiver") and self.IsRiver():
            return False
        if info.get("RequiresFlatlands") and self.IsHills():
            return False
        if info.get("RequiresRiver") and not self.IsRiver():
            return False
        return not (
            info.get("NoAdjacent") and any(p.feature == ident for p in self.adjacent())
        )


class World:
    def __init__(self, game, seed=42, size=3, options=(2, 2, 2, 2, 2)):
        self.game, self.seed, self.size, self.options = (
            Path(game),
            seed,
            size,
            tuple(options),
        )
        self.sources, self.loaded = {}, set()
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        self.native = Native(seed)
        self.types = {}
        self.worlds = self.rows(
            "DLC/Expansion2/Gameplay/XML/GameInfo/CIV5Worlds.xml", "Worlds"
        )
        terrains = self.rows("Gameplay/XML/Terrain/CIV5Terrains.xml", "Terrains")
        self.features = self.rows(
            "DLC/Expansion2/Gameplay/XML/Terrain/CIV5Features.xml", "Features"
        )
        for rows in (self.worlds, terrains, self.features):
            for row in rows:
                self.types[row["Type"]] = row["ID"]
        self.terrains = terrains
        feature_xml = ET.fromstring(
            self.read(
                self.game
                / "Assets/DLC/Expansion2/Gameplay/XML/Terrain/CIV5Features.xml"
            )
        )
        self.feature_terrains = {}
        for row in feature_xml.findall("Feature_TerrainBooleans/Row"):
            f, t = (
                self.types[row.findtext("FeatureType")],
                self.types[row.findtext("TerrainType")],
            )
            self.feature_terrains.setdefault(f, set()).add(t)
        defines_xml = ET.fromstring(
            self.read(self.game / "Assets/Gameplay/XML/GlobalDefines.xml")
        )
        self.defines = {
            r.get("Name"): scalar(r.findtext("Value"))
            for r in defines_xml.findall("Defines/Row")
        }
        self.defines.update(
            SHALLOW_WATER_TERRAIN=self.types["TERRAIN_COAST"],
            DEEP_WATER_TERRAIN=self.types["TERRAIN_OCEAN"],
        )
        self.w, self.h = self.worlds[size]["GridWidth"], self.worlds[size]["GridHeight"]
        self.plots = [Plot(self, x, y) for y in range(self.h) for x in range(self.w)]
        self.areas = []
        self.recalculate()
        self.bind()

    def read(self, path):
        data = path.read_bytes()
        self.sources[str(path.relative_to(self.game))] = hashlib.sha256(
            data
        ).hexdigest()
        return data.decode("utf-8-sig")

    def rows(self, relative, table):
        root = ET.fromstring(self.read(self.game / "Assets" / relative))
        rows = []
        for i, row in enumerate(root.findall(f"{table}/Row")):
            values = {node.tag: scalar(node.text) for node in row}
            values.setdefault("ID", i)
            rows.append(values)
        return rows

    def get(self, x, y):
        x, y = int(x), int(y)
        return self.plots[y * self.w + x % self.w] if 0 <= y < self.h else None

    def direction(self, x, y, d):
        return self.get(*neighbor(int(x), int(y), int(d)))

    def recalculate(self):
        self.areas = []
        seen = set()
        for start in self.plots:
            if start in seen:
                continue
            group, pending = [], [start]
            seen.add(start)
            while pending:
                p = pending.pop()
                p.area = len(self.areas)
                group.append(p)
                for n in p.adjacent():
                    if (
                        n not in seen
                        and n.IsWater() == start.IsWater()
                        and n.IsImpassable() == start.IsImpassable()
                    ):
                        seen.add(n)
                        pending.append(n)
            self.areas.append(Area(len(self.areas), group))

    def biggest(self, water):
        return max(
            (a for a in self.areas if a.plots[0].IsWater() == water),
            key=lambda a: len(a.plots),
            default=None,
        )

    def find_water(self, plot, radius, fresh):
        radius = int(radius)
        q = plot.x - plot.y // 2
        for dq in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                if distance(dq, dr) > radius:
                    continue
                y = plot.y + dr
                p = self.get(q + dq + y // 2, y)
                if p is not None and (p.IsFreshWater() if fresh else p.IsWater()):
                    return True
        return False

    def table(self, data):
        return self.lua.table_from(data, recursive=True)

    def info_table(self, rows):
        t = self.lua.table()
        for row in rows:
            item = self.table(row)
            t[row["ID"]] = item
            t[row["Type"]] = item
        self.lua.eval(
            "function(t,n) setmetatable(t,{__call=function() local i=-1; return function() i=i+1; if i<n then return t[i] end end end}); return t end"
        )(t, len(rows))
        return t

    def include(self, name):
        name = name.removesuffix(".lua")
        if name in self.loaded:
            return
        if name == "AssignStartingPlots":
            # Explicitly excluded pipeline stage; invoking it remains an error.
            return
        paths = [
            self.game / "Assets" / prefix / (name + ".lua")
            for prefix in (
                "DLC/Expansion2/Gameplay/Lua",
                "DLC/Expansion/Gameplay/Lua",
                "Gameplay/Lua",
                "Maps",
            )
        ]
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Lua include: {name}")
        self.loaded.add(name)
        self.lua.execute(self.read(path), name=str(path))

    def bind(self):
        g = self.lua.globals()
        g.include = self.include
        g.print = lambda *args: None
        g.GameInfoTypes = self.table(self.types)
        g.GameDefines = self.table(self.defines)
        g.GameInfo = self.table({})
        g.GameInfo.Worlds = self.info_table(self.worlds)
        g.GameInfo.Terrains = self.info_table(self.terrains)
        g.GameInfo.Features = self.info_table(self.features)
        g.GameInfo.Climates = self.table({0: {}})
        g.PlotTypes = self.table(
            {
                "PLOT_MOUNTAIN": 0,
                "PLOT_HILLS": 1,
                "PLOT_LAND": 2,
                "PLOT_OCEAN": 3,
                "NUM_PLOT_TYPES": 4,
            }
        )
        g.TerrainTypes = self.table(
            {
                "NO_TERRAIN": -1,
                **{k: v for k, v in self.types.items() if k.startswith("TERRAIN_")},
            }
        )
        g.FeatureTypes = self.table(
            {
                "NO_FEATURE": -1,
                **{k: v for k, v in self.types.items() if k.startswith("FEATURE_")},
            }
        )
        names = ["NORTHEAST", "EAST", "SOUTHEAST", "SOUTHWEST", "WEST", "NORTHWEST"]
        g.DirectionTypes = self.table(
            {
                **{"DIRECTION_" + n: i for i, n in enumerate(names)},
                "NUM_DIRECTION_TYPES": 6,
                "NO_DIRECTION": -1,
            }
        )
        g.FlowDirectionTypes = self.table(
            {
                **{
                    "FLOWDIRECTION_" + n: i
                    for i, n in enumerate(
                        [
                            "NORTH",
                            "NORTHEAST",
                            "SOUTHEAST",
                            "SOUTH",
                            "SOUTHWEST",
                            "NORTHWEST",
                        ]
                    )
                },
                "NO_FLOWDIRECTION": -1,
                "NUM_FLOWDIRECTION_TYPES": 6,
            }
        )
        g.Map = self.table(
            {
                "GetGridSize": lambda: (self.w, self.h),
                "GetNumPlots": lambda: len(self.plots),
                "GetPlot": self.get,
                "GetPlotByIndex": lambda i: self.plots[int(i)],
                "GetWorldSize": lambda: self.size,
                "GetClimate": lambda: 0,
                "GetCustomOption": lambda i: self.options[int(i) - 1],
                "GetFractalFlags": lambda: self.table({"FRAC_WRAP_X": True}),
                "IsWrapX": lambda: True,
                "IsWrapY": lambda: False,
                "Rand": self.native.rand,
                "PlotDirection": self.direction,
                "RecalculateAreas": self.recalculate,
                "CalculateAreas": self.recalculate,
                "FindBiggestArea": self.biggest,
                "GetArea": lambda i: self.areas[int(i)],
                "FindWater": self.find_water,
            }
        )
        g.Fractal = self.table(
            {
                "Create": self.native.create,
                "CreateRifts": lambda w, h, grain, flags, rifts, xe=None, ye=None: (
                    self.native.create(w, h, grain, flags, xe, ye, rifts)
                ),
            }
        )
        self.lua.execute(
            "function table.fill(v,n) local t={} for i=1,n do t[i]=v end return t end"
        )
        self.lua.execute(
            "function math.clamp(v,lo,hi) return math.max(lo,math.min(hi,v)) end"
        )
        self.lua.execute(
            'function table.maxn(t) local n=0; for k in pairs(t) do if type(k)=="number" and k>n then n=k end end; return n end; unpack=table.unpack'
        )

    def generate(self):
        self.include("Pangaea")
        g = self.lua.globals()
        for stage in (
            "GeneratePlotTypes",
            "GenerateTerrain",
            "AddRivers",
            "AddLakes",
            "AddFeatures",
        ):
            print(f"  {stage}", flush=True)
            g[stage]()
            if stage in ("GenerateTerrain", "AddLakes", "AddFeatures"):
                self.recalculate()
        return self.export()

    def export(self):
        return {
            "schema": 1,
            "width": self.w,
            "height": self.h,
            "row_parity": 0,
            "wrap_x": True,
            "metadata": {
                "generator": "civ5-pangaea-lua-sdk-v1",
                "lua_runtime": "Lupa 2.8 / Lua 5.4",
                "profile": "BNW terrain only; not Community Patch",
                "seed": self.seed,
                "world_size": self.size,
                "options": list(self.options),
                "sources": self.sources,
                "adapter_sources": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.glob("*.py")
                },
                "native": json.loads((ROOT / "build/native_sources.json").read_text()),
                "excluded_stages": [
                    "StartPlotSystem",
                    "AddGoodies",
                    "DetermineContinents",
                ],
            },
            "tiles": [
                {
                    "terrain": self.terrains[p.terrain]["Type"]
                    .removeprefix("TERRAIN_")
                    .lower(),
                    "elevation": ["mountain", "hills", "flat", "water"][p.kind],
                    "feature": None
                    if p.feature < 0
                    else self.features[p.feature]["Type"]
                    .removeprefix("FEATURE_")
                    .lower(),
                    "rivers": list(p.rivers),
                }
                for p in self.plots
            ],
        }

    def close(self):
        self.native.close()
        self.lua = None


def generate(game, seed=42, size=3, options=(2, 2, 2, 2, 2)):
    if not (0 <= seed <= 0xFFFFFFFF and size in range(6)):
        raise ValueError("Invalid seed or world size")
    if len(options) != 5 or any(v not in range(1, 5) for v in options[:4]):
        raise ValueError("Invalid map options")
    world = World(game, seed, size, options)
    try:
        return world.generate()
    finally:
        world.close()
