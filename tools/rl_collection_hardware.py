"""Read effective resources, including Linux container limits, without torch."""

import os
import platform
import re
import sys
from collections import Counter
from pathlib import Path


def cpu_set(value):
    result = set()
    for part in value.strip().split(","):
        if part:
            ends = [int(v) for v in part.split("-")]
            result.update(range(ends[0], ends[-1] + 1))
    return result


def _read(path):
    try:
        return path.read_text().strip()
    except (OSError, PermissionError):
        return None


def cgroup_limits(proc=Path("/proc")):
    """Resolve this process's v1/v2 mounts and all visible ancestor limits."""
    groups = {}
    for line in (_read(proc / "self/cgroup") or "").splitlines():
        _, controllers, group = line.split(":", 2)
        for controller in controllers.split(","):
            groups[controller] = group
    quotas, cpus, memory, sources = [], None, [], []
    for line in (_read(proc / "self/mountinfo") or "").splitlines():
        before, after = line.split(" - ", 1)
        fields, mount = before.split(), after.split()
        if mount[0] not in ("cgroup", "cgroup2"):
            continue
        # mountinfo escapes spaces/backslashes using octal codes.
        unescape = lambda text: re.sub(
            r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), text
        )
        root, point = Path(unescape(fields[3])), Path(unescape(fields[4]))
        controllers = [""] if mount[0] == "cgroup2" else mount[2].split(",")
        for controller in controllers:
            if controller not in groups:
                continue
            group = Path(groups[controller])
            try:
                relative = group.relative_to(root)
            except ValueError:
                # A cgroup namespace can report paths relative to its own root.
                relative = group.relative_to("/")
            current = point / relative
            while current.is_relative_to(point):
                if controller in ("", "cpu"):
                    if controller == "":
                        raw = _read(current / "cpu.max")
                        values = raw.split() if raw else []
                    else:
                        values = [
                            _read(current / name)
                            for name in ("cpu.cfs_quota_us", "cpu.cfs_period_us")
                        ]
                    if len(values) == 2 and values[0] not in (None, "max", "-1"):
                        quota, period = map(int, values)
                        if quota > 0 and period > 0:
                            quotas.append(quota / period)
                            sources.append(str(current))
                if controller in ("", "cpuset"):
                    raw = _read(current / "cpuset.cpus.effective") or _read(
                        current / "cpuset.cpus"
                    )
                    if raw:
                        parsed = cpu_set(raw)
                        cpus = parsed if cpus is None else cpus & parsed
                if controller in ("", "memory"):
                    limit_name, used_name = (
                        ("memory.max", "memory.current")
                        if controller == ""
                        else ("memory.limit_in_bytes", "memory.usage_in_bytes")
                    )
                    limit, used = (
                        _read(current / limit_name),
                        _read(current / used_name),
                    )
                    if limit not in (None, "max") and used is not None:
                        memory.append(max(0, int(limit) - int(used)))
                if current == point:
                    break
                current = current.parent
    return {
        "cpu_quota": min(quotas, default=None),
        "cpuset": sorted(cpus) if cpus is not None else None,
        "memory_available_bytes": min(memory, default=None),
        "quota_sources": sorted(set(sources)),
    }


def effective_cpu_budget(affinity, quota=None, cpuset=None):
    available = set(affinity)
    if cpuset is not None:
        available &= set(cpuset)
    if not available:
        raise ValueError("No available CPUs in affinity/cpuset")
    return (
        min(float(len(available)), quota)
        if quota is not None
        else float(len(available))
    )


def hardware():
    import psutil

    name = platform.processor()
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    else:
        for line in (_read(Path("/proc/cpuinfo")) or "").splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    process = psutil.Process()
    affinity = (
        process.cpu_affinity()
        if hasattr(process, "cpu_affinity")
        else list(range(os.cpu_count() or 1))
    )
    limits = cgroup_limits() if sys.platform == "linux" else {}
    available = sorted(set(affinity) & set(limits.get("cpuset") or affinity))
    topology = {}
    core_types = Counter()
    for cpu in available:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        core, package = _read(path / "core_id"), _read(path / "physical_package_id")
        if core is not None and package is not None:
            topology.setdefault((package, core), []).append(cpu)
        kind = _read(path / "core_type")
        if kind is not None:
            core_types[kind] += 1
    memory = psutil.virtual_memory().available
    if limits.get("memory_available_bytes") is not None:
        memory = min(memory, limits["memory_available_bytes"])
    return {
        "CPU": name,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_threads": psutil.cpu_count(),
        "cpu_affinity": affinity,
        "available_logical_threads": len(available),
        "cgroup": limits,
        "effective_cpu_budget": effective_cpu_budget(
            affinity, limits.get("cpu_quota"), limits.get("cpuset")
        ),
        "available_core_siblings": list(topology.values()),
        "available_core_types": dict(core_types),
        "available_RAM_bytes": memory,
        "platform": platform.platform(),
        "python": sys.version,
    }
