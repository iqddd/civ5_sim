"""Build the installed SDK's CvFractal without modifying the SDK."""

import argparse
import hashlib
import json
import re
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools._build_common import compile_shared, extract_cpp_function


def build(sdk):
    out = ROOT / "build"
    out.mkdir(exist_ok=True)
    paths = [
        sdk / n
        for n in [
            "CvFractal.cpp",
            "CvFractal.h",
            "CvGameCoreUtils.h",
            "CvGameCoreUtils.cpp",
            "CvRandom.cpp",
        ]
    ]
    source = {p.name: p.read_text(encoding="cp1252") for p in paths}
    header = source["CvFractal.h"]
    cpp = re.sub(r"^#include[^\n]*", "", source["CvFractal.cpp"], flags=re.MULTILINE)
    cpp = cpp.replace(
        "FStaticVector<VoronoiSeed,256, true, c_eCiv5GameplayDLL, 0>",
        "std::vector<VoronoiSeed>",
    )
    prelude = """#include <algorithm>
#include <vector>
#include <cstdint>
#include <climits>
#include <cassert>
#include <cstdlib>
using byte = unsigned char;
using BYTE = unsigned char;
using DirectionTypes = int;
constexpr int NUM_DIRECTION_TYPES = 6;
#define FAssert(x) assert(x)
#define CvAssertMsg(x, msg) assert(x);
#define DEBUG_VARIABLE(x) ((void)(x))
template<class T> T range(T x, T lo, T hi) { return std::clamp(x,lo,hi); }
class CvRandom {
public:
 uint32_t seed;
 explicit CvRandom(uint32_t s): seed(s) {}
 unsigned short get(unsigned short n, const char*) {
   seed = 1103515245u * seed + 12345u;
   return static_cast<unsigned short>(((seed >> 16) * uint32_t(n)) / 65536u);
 }
};
"""
    utils = "\n".join(
        [
            extract_cpp_function(source["CvGameCoreUtils.h"], "inline int xToHexspaceX("),
            extract_cpp_function(source["CvGameCoreUtils.h"], "inline int hexDistance("),
            extract_cpp_function(source["CvGameCoreUtils.cpp"], "DirectionTypes estimateDirection("),
        ]
    )
    exports = """
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
API CvRandom* rng_new(uint32_t seed) { return new CvRandom(seed); }
API void rng_free(CvRandom* r) { delete r; }
API int rng_get(CvRandom* r, int n) { return r->get((unsigned short)n, "standalone"); }
API CvFractal* frac_new(CvRandom* r,int w,int h,int grain,int flags,CvFractal* rifts,int xe,int ye) {
 auto f = new CvFractal(); f->fracInit(w,h,grain,*r,flags,rifts,xe,ye); return f;
}
API void frac_free(CvFractal* f) { delete f; }
API int frac_height(CvFractal* f,int x,int y) { return f->getHeight(x,y); }
API int frac_percent(CvFractal* f,int p) { return f->getHeightFromPercent(p); }
API void frac_ridges(CvFractal* f,CvRandom* r,int n,int flags,int a,int b) { f->ridgeBuilder(*r,n,flags,a,b); }
"""
    (out / "fractal.cpp").write_text(
        prelude + utils + "\n" + header + "\n" + cpp + exports, encoding="utf-8"
    )
    library = "fractal.dll" if os.name == "nt" else "libfractal.so"
    compile_shared([out / "fractal.cpp"], out / library, script_name="compile.cmd")
    manifest = {
        "sdk": str(sdk.resolve()),
        "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "native_sources.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sdk", type=Path, default=Path("E:/civ5/CvGameCoreDLL_Expansion2")
    )
    build(parser.parse_args().sdk)
