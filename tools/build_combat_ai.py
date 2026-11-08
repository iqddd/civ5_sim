"""Build the CP planner and branch tests using the prepared combat geometry.

Run tools/build_combat.py once to prepare the extracted rules/geometry inputs.
The resulting planner does not link to the combat DLL or GameCore.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools._build_common import compile_shared


def build(*, profile=False, dll_only=False):
    out = ROOT / "build"
    out.mkdir(exist_ok=True)
    missing = [name for name in ("combat_rules.h", "cp_geometry.inc")
               if not (out / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Missing prepared combat inputs: {', '.join(missing)}. "
            "Run python tools/build_combat.py first."
        )
    source = ROOT / "native/combat_ai/combat_ai.cpp"
    common = [ROOT / "native/combat_ai/model.cpp",
              ROOT / "native/combat_state.cpp", ROOT / "native/geometry.cpp"]
    planner_sources = [ROOT / "native/combat_ai/planner.cpp",
                       ROOT / "native/combat_ai/cp_tactical.cpp",
                       ROOT / "native/combat_ai/support.cpp",
                       ROOT / "native/combat_ai/tactical.cpp"]
    test_sources = [ROOT / "tests/native/combat_ai_model_test.cpp",
                    ROOT / "native/combat.cpp", *common]
    planner_tests = [ROOT / "tests/native/combat_ai_planner_test.cpp", source,
                     *common, *planner_sources]
    # Compile the C++ adapter declarations as part of every build as well.
    check = out / "combat_ai_header_check.cpp"
    check.write_text(
        '#include "model.h"\n#include <type_traits>\n'
        "static_assert(std::is_abstract_v<combat_ai::RulesAdapter>);\n",
        encoding="utf-8",
    )
    include = source.parent
    library_name = "combat_ai_profile" if profile else "combat_ai"
    windows = os.name == "nt"
    library = library_name + ".dll" if windows else "lib" + library_name + ".so"
    suffix = ".exe" if windows else ""
    executables = [] if dll_only else [
        (out / ("combat_ai_model_test" + suffix), test_sources),
        (out / ("combat_ai_planner_test" + suffix), planner_tests),
    ]
    compile_shared(
        [source, check, *common, *planner_sources], out / library,
        include_dirs=[include, out], defines=["CAI_PROFILE"] if profile else [],
        executables=executables,
        script_name="compile_combat_ai_profile.cmd" if profile else "compile_combat_ai.cmd",
    )
    print(f"Built {library_name}" + (" and native checks" if not dll_only else ""))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true", help="Separate instrumented library; production is unchanged")
    parser.add_argument("--dll-only", action="store_true", help="Build only the library")
    args = parser.parse_args()
    build(profile=args.profile, dll_only=args.dll_only)
