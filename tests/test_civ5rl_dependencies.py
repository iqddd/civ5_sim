"""Enforce layer boundaries and keep CLI tools out of package dependencies."""

import ast
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "civ5rl"
LOWER_LAYERS = {"core", "provenance", "protocols", "engine"}


def imported_modules(tree, package):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                yield node.lineno, item.name
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            base = resolve_name(name, package) if node.level else name
            yield node.lineno, base
            for item in node.names:
                yield node.lineno, f"{base}.{item.name}"
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else getattr(function, "attr", None)
            )
            argument = node.args[0]
            if (
                name in {"import_module", "__import__"}
                and isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
            ):
                yield node.lineno, resolve_name(argument.value, package)


def test_civ5rl_dependency_direction():
    violations = []
    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT)
        layer = relative.parts[0]
        package = ".".join((ROOT.name, *relative.parent.parts))
        forbidden = ["tests", "civ5rl.tests", "tools", "civ5rl.tools"]
        if layer in LOWER_LAYERS | {"reporting"}:
            forbidden.append("civ5rl.experiments")
        if layer in LOWER_LAYERS:
            forbidden.append("civ5rl.reporting")
        if layer == "engine":
            forbidden.extend(("deploy", "civ5rl.deploy"))
        if relative.parts[:2] == ("engine", "imitation"):
            forbidden.extend(("civ5rl.engine.ppo", "civ5rl.ppo", "civ5rl.ppo_fast"))
        for line, module in imported_modules(ast.parse(path.read_bytes()), package):
            if any(
                module == name or module.startswith(name + ".") for name in forbidden
            ):
                violations.append(f"{relative.as_posix()}:{line} imports {module}")
    assert not violations, "\n".join(violations)
