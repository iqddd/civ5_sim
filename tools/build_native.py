"""Build the installed SDK's CvFractal without modifying the SDK."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract(text, signature):
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


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
            extract(source["CvGameCoreUtils.h"], "inline int xToHexspaceX("),
            extract(source["CvGameCoreUtils.h"], "inline int hexDistance("),
            extract(source["CvGameCoreUtils.cpp"], "DirectionTypes estimateDirection("),
        ]
    )
    exports = """
#define API extern "C" __declspec(dllexport)
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
    vswhere = Path(
        "C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
    )
    vs = subprocess.check_output(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        text=True,
    ).strip()
    if not vs:
        raise RuntimeError("Visual Studio C++ x64 build tools are required")
    vcvars = Path(vs) / "VC/Auxiliary/Build/vcvars64.bat"
    (out / "compile.cmd").write_text(
        f'@call "{vcvars}" >nul\n@cl /nologo /std:c++17 /EHsc /O2 /LD /utf-8 fractal.cpp /link /OUT:fractal.dll\n',
        encoding="utf-8",
    )
    subprocess.run(["cmd", "/c", "compile.cmd"], cwd=out, check=True)
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
