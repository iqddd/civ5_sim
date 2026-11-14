"""Verified transport packets, separate from scientific run directories."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from civ5rl.deploy.remote import _read, _sha, _write

JSON_MIN_FILES = 1000
JSON_MIN_BYTES = 64 * 1024 * 1024
PPMD = ["-t7z", "-m0=PPMd:mem=256m:o=32", "-ms=4g"]


def seven_zip():
    for name in ("7zz", "7z", "7za"):
        executable = shutil.which(name)
        if executable:
            return executable
    if os.name == "nt":
        executable = (
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "7-Zip/7z.exe"
        )
        if executable.is_file():
            return str(executable)
    raise RuntimeError("Archive transport requires 7zz/7z (7-Zip) on both hosts")


def _name(value):
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or any(part in (".", "..") for part in path.parts)
        or any(c in value for c in ("\\", ":", "\n", "\r", "\x00"))
    ):
        raise ValueError(f"Unsafe transport path: {value!r}")
    return value


def _inventory(root):
    files, directories = {}, []
    if not root.exists():
        return files, directories
    if not root.is_dir():
        raise ValueError(f"Transport root is not a directory: {root}")
    if root.is_symlink() or getattr(root.lstat(), "st_file_attributes", 0) & 0x400:
        raise ValueError(f"Symlink/reparse transport root: {root}")

    def scan_error(error):
        raise error

    # Walk without following links, including Windows junctions/reparse points.
    for current, dirs, names in os.walk(root, followlinks=False, onerror=scan_error):
        for name in sorted([*dirs, *names]):
            path = Path(current) / name
            rel = _name(path.relative_to(root).as_posix())
            stat = path.lstat()
            if path.is_symlink() or getattr(stat, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"Symlink/reparse transport input: {rel}")
            if path.is_dir():
                directories.append(rel)
            elif path.is_file():
                files[rel] = {"size": stat.st_size, "sha256": _sha(path)}
            else:
                raise ValueError(f"Special transport input: {rel}")
    return dict(sorted(files.items())), sorted(directories)


def needs_archive(root):
    count = size = 0
    for current, _, names in os.walk(root, followlinks=False):
        for name in names:
            if Path(name).suffix.lower() in (".json", ".jsonl"):
                count += 1
                size += (Path(current) / name).stat().st_size
    return count >= JSON_MIN_FILES or size >= JSON_MIN_BYTES


def _separate(source, packet):
    if (
        source == packet
        or source.is_relative_to(packet)
        or packet.is_relative_to(source)
    ):
        raise ValueError("Transport packet and scientific directory must not overlap")
    if source == Path(source.anchor) or packet == Path(packet.anchor):
        raise ValueError("Transport roots cannot be filesystem roots")


def _root(path):
    original = Path(path).absolute()
    resolved = original.resolve()
    if original != resolved:
        raise ValueError(
            f"Transport root contains a symlink/reparse or unnormalized path: {original}"
        )
    return resolved


def pack(source, packet, *, token, direction, force=False):
    source, packet = _root(source), _root(packet)
    _separate(source, packet)
    if direction not in ("upload", "download") or not token:
        raise ValueError("Transport requires a token and direction")
    files, directories = _inventory(source)
    json_files = [
        name for name in files if Path(name).suffix.lower() in (".json", ".jsonl")
    ]
    compressed = (
        json_files
        if (
            force or len(json_files) >= JSON_MIN_FILES
            or sum(files[name]["size"] for name in json_files) >= JSON_MIN_BYTES
        )
        else []
    )
    packet.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".packet-", dir=packet.parent) as temporary:
        staged = Path(temporary) / "packet"
        staged.mkdir()
        archived = set(compressed)
        for name in files:
            if name not in archived:
                target = staged / "files" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
        if compressed:
            listing = Path(temporary) / "members.txt"
            listing.write_text(
                "".join(name + "\n" for name in compressed), encoding="utf-8"
            )
            subprocess.run(
                [
                    seven_zip(),
                    "a",
                    *PPMD,
                    "-scsUTF-8",
                    "-spd",
                    "-r-",
                    "-bso0",
                    "-bsp0",
                    str(staged / "json.7z"),
                    "@" + str(listing),
                ],
                cwd=source,
                check=True,
            )
        # Detect mutation during copying/compression instead of publishing a mixed snapshot.
        if _inventory(source) != (files, directories):
            raise RuntimeError(
                "Transport source changed during packaging; retry after writer stops"
            )
        manifest = dict(
            schema=1,
            token=token,
            direction=direction,
            files=files,
            directories=directories,
            archived=compressed,
            compression=PPMD if compressed else None,
            archive_sha256=_sha(staged / "json.7z") if compressed else None,
        )
        _write(staged / "manifest.json", manifest)
        # This target was resolved and checked against the source above; only staging is replaced.
        if packet.exists():
            shutil.rmtree(packet)
        staged.replace(packet)
    return manifest


def _extract(packet, staging, manifest):
    files = manifest["files"]
    directories = manifest["directories"]
    for name in [*files, *directories, *manifest["archived"]]:
        _name(name)
    archived = set(manifest["archived"])
    if not archived.issubset(files) or len(archived) != len(manifest["archived"]):
        raise ValueError("Invalid archived file inventory")
    if archived:
        archive = packet / "json.7z"
        if _sha(archive) != manifest["archive_sha256"]:
            raise ValueError("Transport archive SHA-256 mismatch")
        # Validate names and links before extraction; do not let archive entries escape staging.
        result = subprocess.run(
            [seven_zip(), "l", "-slt", "-ba", "-sccUTF-8", str(archive)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        members = set()
        for block in result.stdout.replace("\r\n", "\n").split("\n\n"):
            fields = dict(
                line.split(" = ", 1) for line in block.splitlines() if " = " in line
            )
            if not fields:
                continue
            name = _name(fields["Path"].replace("\\", "/"))
            if (
                name in members
                or name not in archived
                or fields.get("Folder") == "+"
                or "Symbolic Link" in fields
                or "Hard Link" in fields
                or any(
                    part.startswith("l")
                    for part in fields.get("Attributes", "").split()
                )
            ):
                raise ValueError(f"Unexpected archive entry: {name}")
            members.add(name)
        if members != archived:
            raise ValueError("Transport archive member inventory differs")
        subprocess.run(
            [
                seven_zip(),
                "x",
                "-y",
                "-bso0",
                "-bsp0",
                "-o" + str(staging),
                str(archive),
            ],
            check=True,
        )
    loose, _ = _inventory(packet / "files")
    if loose != {name: facts for name, facts in files.items() if name not in archived}:
        raise ValueError("Transport loose file SHA-256/inventory mismatch")
    for name in loose:
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(packet / "files" / name, target)
    for name in directories:
        (staging / name).mkdir(parents=True, exist_ok=True)
    if _inventory(staging) != (files, directories):
        raise ValueError("Extracted file SHA-256/inventory mismatch")


def unpack(packet, destination, *, token, direction):
    packet, destination = _root(packet), _root(destination)
    _separate(destination, packet)
    manifest = _read(packet / "manifest.json")
    if (
        manifest.get("schema") != 1
        or manifest.get("token") != token
        or manifest.get("direction") != direction
    ):
        raise ValueError("Transport packet belongs to another operation/direction")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".unpack-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        _extract(packet, staging, manifest)
        # Validate the existing destination before changing or deleting anything in it.
        existing, old_dirs = _inventory(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name in sorted(old_dirs, key=lambda n: n.count("/"), reverse=True):
            if name not in manifest["directories"]:
                # Remove stale children first, including a directory replaced by a file.
                for file in list(existing):
                    if file.startswith(name + "/"):
                        (destination / file).unlink()
                        del existing[file]
                (destination / name).rmdir()
        for name in existing:
            if name not in manifest["files"]:
                (destination / name).unlink()
        for name in manifest["directories"]:
            (destination / name).mkdir(parents=True, exist_ok=True)
        for name, facts in manifest["files"].items():
            if existing.get(name) == facts:
                continue
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / name, target)
        if _inventory(destination) != (manifest["files"], manifest["directories"]):
            raise ValueError(
                "Final destination SHA-256/inventory mismatch; instance must be retained"
            )
    return dict(
        status="verified",
        verified_files=len(manifest["files"]),
        bytes=sum(item["size"] for item in manifest["files"].values()),
        archived_files=len(manifest["archived"]),
        compression=manifest["compression"],
    )
