"""Build the standalone CP-01 planner; no game checkout or combat DLL required."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    out = ROOT / "build"
    out.mkdir(exist_ok=True)
    source = ROOT / "native/combat_ai/combat_ai.cpp"
    # Compile the C++ adapter declarations as part of every build as well.
    check = out / "combat_ai_header_check.cpp"
    check.write_text(
        '#include "model.h"\n#include <type_traits>\n'
        "static_assert(std::is_abstract_v<combat_ai::RulesAdapter>);\n",
        encoding="utf-8",
    )
    include = source.parent
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
            f'@cl /nologo /std:c++17 /EHsc /O2 /LD /utf-8 /W4 /I"{include}" '
            f'"{source}" "{check}" /link /OUT:combat_ai.dll\n'
        )
        (out / "compile_combat_ai.cmd").write_text(command, encoding="utf-8")
        subprocess.run(["cmd", "/c", "compile_combat_ai.cmd"], cwd=out, check=True)
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
                "-I",
                str(include),
                str(source),
                str(check),
                "-o",
                str(out / "libcombat_ai.so"),
            ],
            check=True,
        )
    print("Built standalone combat AI scaffold (CP-01, ABI 1)")


if __name__ == "__main__":
    build()
