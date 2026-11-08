"""Owned ctypes handles for the SDK fractal and its shared map RNG."""

import ctypes as C
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_library():
    path = ROOT / "build" / ("fractal.dll" if os.name == "nt" else "libfractal.so")
    if not path.exists():
        raise RuntimeError("Run python tools/build_native.py first")
    lib = C.CDLL(str(path))
    for name, args, result in [
        ("rng_new", [C.c_uint32], C.c_void_p),
        ("rng_free", [C.c_void_p], None),
        ("rng_get", [C.c_void_p, C.c_int], C.c_int),
        (
            "frac_new",
            [C.c_void_p] + [C.c_int] * 4 + [C.c_void_p] + [C.c_int] * 2,
            C.c_void_p,
        ),
        ("frac_free", [C.c_void_p], None),
        ("frac_height", [C.c_void_p, C.c_int, C.c_int], C.c_int),
        ("frac_percent", [C.c_void_p, C.c_int], C.c_int),
        ("frac_ridges", [C.c_void_p, C.c_void_p] + [C.c_int] * 4, None),
    ]:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    return lib


def flags(table):
    names = [
        "FRAC_WRAP_X",
        "FRAC_WRAP_Y",
        "FRAC_PERCENT",
        "FRAC_POLAR",
        "FRAC_CENTER_RIFT",
        "FRAC_INVERT_HEIGHTS",
    ]
    # SDK HasFlag checks presence, including a false-valued field.
    return sum(1 << i for i, key in enumerate(names) if table[key] is not None)


class Native:
    def __init__(self, seed):
        self.lib = load_library()
        self.rng = self.lib.rng_new(seed)
        self.fractals = []
        self.calls = 0

    def rand(self, n, _label=""):
        self.calls += 1
        if self.calls > 2_000_000:
            raise RuntimeError("Map generation RNG budget exceeded")
        n = int(n)
        if not 0 <= n <= 65535:
            raise ValueError(f"Invalid RNG range: {n}")
        return self.lib.rng_get(self.rng, n)

    def create(self, w, h, grain, flag_table, xe=None, ye=None, rifts=None):
        if len(self.fractals) >= 512:
            raise RuntimeError("Map generation fractal/retry budget exceeded")
        xe, ye = (
            7 if xe is None or xe < 0 else int(xe),
            6 if ye is None or ye < 0 else int(ye),
        )
        if not (1 <= w < 255 and 1 <= h < 255 and 1 <= xe <= 7 and 1 <= ye <= 7):
            raise ValueError("Unsupported native fractal dimensions")
        handle = self.lib.frac_new(
            self.rng,
            int(w),
            int(h),
            int(grain),
            flags(flag_table),
            rifts.handle if rifts else None,
            xe,
            ye,
        )
        result = Fractal(self, handle, int(w), int(h))
        self.fractals.append(result)
        return result

    def close(self):
        for f in self.fractals:
            self.lib.frac_free(f.handle)
        self.fractals.clear()
        if self.rng:
            self.lib.rng_free(self.rng)
            self.rng = None


class Fractal:
    def __init__(self, native, handle, w, h):
        self.native, self.handle, self.w, self.h = native, handle, w, h

    def GetHeight(self, x, y=None):
        if y is None:
            return self.native.lib.frac_percent(self.handle, int(x))
        x, y = int(x), int(y)
        if not (0 <= x < self.w and 0 <= y < self.h):
            raise ValueError("Fractal sample outside bounds")
        return self.native.lib.frac_height(self.handle, x, y)

    def BuildRidges(self, n, table, a, b):
        if not (3 <= int(n) <= 128):
            raise ValueError("Unsupported ridge seed count")
        self.native.lib.frac_ridges(
            self.handle, self.native.rng, int(n), flags(table), int(a), int(b)
        )
