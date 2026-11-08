"""C++ source extraction and platform-specific compilation for build CLIs."""

import os
import shlex
import subprocess
from pathlib import Path


def extract_cpp_function(text: str, signature: str) -> str:
    start = text.index(signature)
    opening = text.index("{", start)
    depth, end = 1, opening + 1
    while depth and end < len(text):
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        raise ValueError(f"Unclosed C++ function: {signature}")
    return text[start:end]


def find_vcvars64() -> Path:
    vswhere = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
    installation = subprocess.check_output(
        [str(vswhere), "-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        text=True,
    ).strip()
    vcvars = Path(installation) / "VC/Auxiliary/Build/vcvars64.bat"
    if not installation or not vcvars.is_file():
        raise RuntimeError("Visual Studio C++ x64 Build Tools are required")
    return vcvars


def compile_shared(sources, out_name, include_dirs=(), defines=(), executables=(), *, script_name=None):
    """Build a shared library plus optional (output path, sources) executables.

    Retain /O2 on MSVC and -O3 on Linux. Each compilation has its own object
    directory so unrelated builds do not overwrite one another's .obj files.
    """
    output = Path(out_name).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    includes = [Path(p).resolve() for p in include_dirs]
    jobs = [(output, [Path(p).resolve() for p in sources], list(defines), True)]
    jobs += [(Path(path).resolve(), [Path(p).resolve() for p in files], [], False)
             for path, files in executables]
    if os.name == "nt":
        vcvars = find_vcvars64()
        lines = [f'@call "{vcvars}" >nul', '@if errorlevel 1 exit /b %errorlevel%']
        for target, files, flags, shared in jobs:
            target.parent.mkdir(parents=True, exist_ok=True)
            objects = target.parent / (target.stem + "-objects")
            objects.mkdir(exist_ok=True)
            command = ["cl", "/nologo", "/std:c++17", "/EHsc", "/O2", "/utf-8", "/W4"]
            if shared:
                command.append("/LD")
            command += [f"/D{flag}" for flag in flags]
            command += [f"/I{p}" for p in includes]
            command += [f"/Fo{objects}\\", *map(str, files), "/link", f"/OUT:{target}"]
            if shared:
                command.append(f"/IMPLIB:{target.with_suffix('.lib')}")
            lines += ["@" + subprocess.list2cmdline(command), '@if errorlevel 1 exit /b %errorlevel%']
        script = output.parent / (script_name or f"compile_{output.stem}.cmd")
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        subprocess.run(["cmd", "/c", str(script)], cwd=output.parent, check=True)
    else:
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        for target, files, flags, shared in jobs:
            target.parent.mkdir(parents=True, exist_ok=True)
            command = [*compiler, "-std=c++17", "-O3", "-Wall", "-Wextra"]
            if shared:
                command += ["-fPIC", "-shared"]
            command += [f"-D{flag}" for flag in flags]
            command += [item for p in includes for item in ("-I", str(p))]
            subprocess.run([*command, *map(str, files), "-o", str(target)], check=True)
    return output
