"""Build the CP planner and branch tests using the prepared combat geometry.

Run tools/build_combat.py once to prepare the extracted rules/geometry inputs.
The resulting planner does not link to the combat DLL or GameCore.
"""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    profile_flags = "/DCAI_PROFILE " if profile else ""
    if os.name == "nt":
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
            raise RuntimeError("Visual Studio C++ x64 Build Tools are required")
        vcvars = Path(vs) / "VC/Auxiliary/Build/vcvars64.bat"
        command = (
            f'@call "{vcvars}" >nul\n'
            f'@cl /nologo /std:c++17 /EHsc /O2 /LD /utf-8 /W4 {profile_flags}/I"{include}" /I"{out}" '
            + " ".join(f'"{p}"' for p in [source, check, *common, *planner_sources])
            + f' /link /OUT:{library_name}.dll /IMPLIB:{library_name}.lib\n@if errorlevel 1 exit /b %errorlevel%\n'
            + f'@cl /nologo /std:c++17 /EHsc /O2 /utf-8 /W4 /I"{include}" /I"{out}" '
            + " ".join(f'"{p}"' for p in test_sources)
            + ' /link /OUT:combat_ai_model_test.exe\n'
            + '@if errorlevel 1 exit /b %errorlevel%\n'
            + f'@cl /nologo /std:c++17 /EHsc /O2 /utf-8 /W4 /I"{include}" /I"{out}" '
            + " ".join(f'"{p}"' for p in planner_tests)
            + ' /link /OUT:combat_ai_planner_test.exe\n'
        )
        if dll_only:
            command = command[:command.index("@cl", command.index("@cl") + 1)]
        script = "compile_combat_ai_profile.cmd" if profile else "compile_combat_ai.cmd"
        (out / script).write_text(command, encoding="utf-8")
        subprocess.run(["cmd", "/c", script], cwd=out, check=True)
    else:
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O3",
                "-fPIC",
                "-shared",
                "-Wall",
                "-Wextra",
                *(["-DCAI_PROFILE"] if profile else []),
                "-I",
                str(include),
                "-I",
                str(out),
                str(source),
                str(check),
                *(str(p) for p in common),
                *(str(p) for p in planner_sources),
                "-o",
                str(out / f"lib{library_name}.so"),
            ],
            check=True,
        )
        if dll_only:
            print(f"Built {library_name}")
            return
        subprocess.run(
            ["c++", "-std=c++17", "-O3", "-Wall", "-Wextra",
             "-I", str(include), "-I", str(out),
             *(str(p) for p in planner_tests),
             "-o", str(out / "combat_ai_planner_test")],
            check=True,
        )
        subprocess.run(
            ["c++", "-std=c++17", "-O3", "-Wall", "-Wextra",
             "-I", str(include), "-I", str(out),
             *(str(p) for p in test_sources),
             "-o", str(out / "combat_ai_model_test")],
            check=True,
        )
    print(f"Built {library_name}" + (" and native checks" if not dll_only else ""))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true", help="Separate instrumented library; production is unchanged")
    parser.add_argument("--dll-only", action="store_true", help="Build only the library")
    args = parser.parse_args()
    build(profile=args.profile, dll_only=args.dll_only)
