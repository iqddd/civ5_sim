"""Versioned ctypes binding; no Python objects are passed into the kernel."""

import ctypes as C
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Tile(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ("terrain", "elevation", "feature", "rivers")]


class Unit(C.Structure):
    _fields_ = [
        (n, C.c_int32)
        for n in (
            "type",
            "side",
            "tile",
            "hp",
            "moves",
            "attacked",
            "acted",
            "fortified",
            "waiting",
        )
    ]


class Action(C.Structure):
    _fields_ = [(n, C.c_int32) for n in ("kind", "unit", "target")]


class Result(C.Structure):
    _fields_ = [
        (n, C.c_int32)
        for n in (
            "code",
            "moved",
            "damage",
            "retaliation",
            "attacker_strength",
            "defender_strength",
        )
    ]


class Info(C.Structure):
    _fields_ = [
        (n, C.c_int32) for n in ("side", "turn", "fog", "random", "alive0", "alive1")
    ]
    _fields_ += [("rng", C.c_uint64)]


@lru_cache(maxsize=1)
def library():
    path = ROOT / "build" / ("combat.dll" if os.name == "nt" else "libcombat.so")
    if not path.exists():
        raise RuntimeError("Build the kernel first: python tools/build_combat.py")
    lib = C.CDLL(str(path))
    for name, arguments, result in (
        ("sim_abi", [], C.c_int),
        ("sim_rules", [], C.c_char_p),
        (
            "sim_new",
            [C.c_int] * 4
            + [C.POINTER(Tile), C.POINTER(Unit)]
            + [C.c_int] * 3
            + [C.c_uint64],
            C.c_void_p,
        ),
        ("sim_free", [C.c_void_p], None),
        ("sim_clone", [C.c_void_p], C.c_void_p),
        ("sim_reset", [C.c_void_p, C.POINTER(Unit), C.c_int, C.POINTER(Info)], C.c_int),
        ("sim_units", [C.c_void_p, C.c_int, C.POINTER(Unit)], C.c_int),
        ("sim_visibility", [C.c_void_p, C.c_int, C.POINTER(C.c_int)], C.c_int),
        ("sim_info", [C.c_void_p, C.POINTER(Info)], C.c_int),
        ("sim_actions", [C.c_void_p, C.POINTER(Action), C.c_int], C.c_int),
        ("sim_step", [C.c_void_p, Action], Result),
        ("sim_preview", [C.c_void_p, Action], Result),
        (
            "sim_batch",
            [C.POINTER(C.c_void_p), C.POINTER(Action), C.POINTER(Result), C.c_int],
            None,
        ),
        ("sim_damage", [C.c_int] * 3, C.c_int),
        ("sim_memory", [C.c_void_p, C.c_int], C.c_uint64),
    ):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = arguments, result
    if lib.sim_abi() != 1:
        raise RuntimeError("Unsupported combat kernel ABI; rebuild the kernel")
    return lib
