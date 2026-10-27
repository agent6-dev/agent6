"""Carry global and repository state through a workspace snapshot."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import errno
import hashlib
import json
import os
import pathlib
import shutil
import socket
import stat
import tempfile
from collections.abc import Iterator
from typing import Literal

import pydantic

from agent6 import _pack_manifest, errors, paths, portable

Outcome = Literal["move", "keep", "conflict"]


@dataclasses.dataclass(frozen=True, slots=True)
class FileFacts:
    """Hold a preflight decision and the paths it governs."""

    name: str
    packed: pathlib.Path
    home: pathlib.Path
    sha256: str
    mtime: datetime.datetime
    home_mtime: datetime.datetime | None
    outcome: Outcome


@dataclasses.dataclass(frozen=True, slots=True)
class ImportResult:
    """Report the files moved or kept and their original workspace."""

    origin: _pack_manifest.Origin
    moved: tuple[pathlib.Path, ...]
    kept: tuple[pathlib.Path, ...]


def choose_file(packed: str, base: str | None, home: str | None) -> Outcome:
    """Apply the three-way rule to hashes without touching the filesystem.

    Args:
        packed: The hash of the pack file.
        base: The last shared hash, or None.
        home: The home file hash, or None when absent.

    Returns:
        Move, keep or conflict.
    """
    if home is None:
        return "move"
    if home == packed:
        return "keep"
    if base is not None and home == base:
        return "move"
    if base is not None and packed == base:
        return "keep"
    return "conflict"


def _files(root: pathlib.Path, *, exclude: tuple[str, ...] = ()) -> Iterator[pathlib.Path]:
    """Walk regular files without following links or omitting unreadable trees.

    Args:
        root: The resolved project root.
        exclude: Top-level names to omit.

    Yields:
        Every regular file, in path order.

    Raises:
        PackError: A tree contains a link or special file.
    """
    _pack_manifest.check_path(root)
    if not root.exists():
        return
    for path in sorted(root.iterdir()):
        if path.name in exclude:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            yield from _files(path)
        elif stat.S_ISREG(mode):
            yield path
        else:
            raise errors.PackError(f"{path}: symlink or special file cannot be transferred")


def _hash(path: pathlib.Path) -> str:
    """Hash file bytes with bounded memory.

    Args:
        path: The file or directory to inspect.

    Returns:
        The SHA256 hex digest.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _mtime(path: pathlib.Path) -> datetime.datetime:
    """Return the file's modification time in UTC."""
    return datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.UTC)


def _home(root: pathlib.Path, name: str) -> pathlib.Path:
    """Map a validated pack name into this home's state.

    Args:
        root: The resolved project root.
        name: The validated pack-relative file name.

    Returns:
        The destination path.
    """
    area, rest = name.split("/", 1)
    bases = {
        "config": paths.global_config_dir(),
        "skills": paths.data_dir() / "skills",
        "repo": paths.state_dir(root, allow_pack=True),
    }
    return bases[area] / rest


def _copy(source: pathlib.Path, target: pathlib.Path, *, secret: bool) -> None:
    """Copy bytes and timestamps, creating secrets through a private descriptor."""
    paths.mkdir_for_real_user(target.parent)
    if not secret:
        shutil.copy2(source, target)
        return
    fd, name = tempfile.mkstemp(dir=target.parent)
    temp = pathlib.Path(name)
    try:
        with os.fdopen(fd, "wb") as dest, source.open("rb") as src:
            shutil.copyfileobj(src, dest)
        info = source.stat()
        os.utime(temp, ns=(info.st_atime_ns, info.st_mtime_ns))
        paths.chown_to_real_user(temp)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def pack(root: pathlib.Path) -> _pack_manifest.Manifest:
    """Copy this home's global and repository state into a private workspace pack.

    Args:
        root: The resolved project root.

    Returns:
        The written manifest.

    Raises:
        PackError: A foreign pack, unsafe path or filesystem error prevents the snapshot.
    """
    root = root.resolve()
    directory = root / ".agent6"
    _pack_manifest.check_path(directory)
    if directory.exists() and not _pack_manifest.is_own(
        _pack_manifest.read(root).origin, paths.effective_user().home
    ):
        raise errors.PackError(f"{directory}: pack belongs to another origin; unpack first")
    try:
        base_path = paths.state_dir(root, allow_pack=True) / "pack-base.json"
        _pack_manifest.check_path(base_path, regular=base_path.exists())
        bases = pydantic.TypeAdapter(
            dict[_pack_manifest.PackName, _pack_manifest.Hash]
        ).validate_json(base_path.read_bytes() if base_path.exists() else b"{}")
        sources = {
            f"config/{name}": paths.global_config_dir() / name
            for name in _pack_manifest.CONFIG_FILES
            if (paths.global_config_dir() / name).exists()
        }
        for area, source in (
            ("skills", paths.data_dir() / "skills"),
            ("repo", paths.state_dir(root, allow_pack=True)),
        ):
            sources.update(
                {
                    f"{area}/{file.relative_to(source).as_posix()}": file
                    for file in _files(
                        source, exclude=("locks", "pack-base.json") if area == "repo" else ()
                    )
                }
            )
        paths.mkdir_for_real_user(directory)
        directory.chmod(0o700)
        with tempfile.TemporaryDirectory(dir=directory) as stage_name:
            stage = pathlib.Path(stage_name)
            for area in ("config", "repo", "skills"):
                (stage / area).mkdir(mode=0o700)
            files: dict[str, _pack_manifest.PackedFile] = {}
            for name, source in sorted(sources.items()):
                _pack_manifest.check_path(source, regular=True)
                target = stage / name
                _copy(source, target, secret=name == "config/secrets.toml")
                files[name] = _pack_manifest.PackedFile(
                    sha256=_hash(target), mtime=_mtime(target), base=bases.get(name)
                )
            manifest = _pack_manifest.Manifest(
                version=1,
                packed_at=datetime.datetime.now(datetime.UTC),
                origin=_pack_manifest.Origin(
                    host=socket.gethostname(),
                    home=str(paths.effective_user().home),
                    repo_id=paths.repo_id(root),
                    workspace=str(root),
                ),
                files=files,
            )
            (stage / "pack.json").write_text(manifest.model_dump_json(indent=2) + "\n")
            for old in directory.iterdir():
                if old == stage:
                    continue
                _pack_manifest.check_path(old)
                if old.is_dir():
                    shutil.rmtree(old)
                else:
                    old.unlink()
            for child in stage.iterdir():
                child.replace(directory / child.name)
        paths.chown_to_real_user(directory)
        return manifest
    except (OSError, pydantic.ValidationError) as exc:
        raise errors.PackError(f"pack {directory}: {exc}") from exc


def _preflight(root: pathlib.Path, manifest: _pack_manifest.Manifest) -> tuple[FileFacts, ...]:
    """Plan every import before moving any file.

    Args:
        root: The resolved project root.
        manifest: The validated pack manifest.

    Returns:
        The per-file decisions.

    Raises:
        PackError: A file is unlisted, unsafe, or conflicts with the home's copy.
    """
    directory = root / ".agent6"
    present = {p.relative_to(directory).as_posix() for p in _files(directory)} - {"pack.json"}
    unlisted = present - manifest.files.keys()
    if unlisted:
        raise errors.PackError(
            f"{directory}: files absent from manifest: {', '.join(sorted(unlisted))}"
        )
    facts: list[FileFacts] = []
    for name in sorted(present):
        packed, home = directory / name, _home(root, name)
        _pack_manifest.check_path(home)
        if home.exists() and not home.is_file():
            raise errors.PackError(f"{home}: expected a regular file")
        digest = _hash(packed)
        facts.append(
            FileFacts(
                name,
                packed,
                home,
                digest,
                _mtime(packed),
                _mtime(home) if home.exists() else None,
                choose_file(
                    digest, manifest.files[name].base, _hash(home) if home.exists() else None
                ),
            )
        )
    conflicts = [f for f in facts if f.outcome == "conflict"]
    if conflicts:
        lines = ["Pack conflicts; nothing moved:"]
        for file in conflicts:
            lines.extend(
                (
                    f"  home: {file.home} (mtime {file.home_mtime})",
                    f"  pack: {file.packed} (mtime {file.mtime.isoformat()})",
                )
            )
        lines.append(
            "Hand-merge the files or delete the side to replace, then rerun agent6 unpack."
        )
        raise errors.PackError("\n".join(lines))
    return tuple(facts)


def _move(file: FileFacts) -> None:
    """Move a file, copying across filesystems and recreating secrets as the importer.

    Args:
        file: The preflight file decision.

    Raises:
        OSError: The filesystem cannot complete the move.
    """
    paths.mkdir_for_real_user(file.home.parent)
    if file.name == "config/secrets.toml":
        _copy(file.packed, file.home, secret=True)
        file.packed.unlink()
    else:
        try:
            file.packed.rename(file.home)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            _copy(file.packed, file.home, secret=False)
            file.packed.unlink()
        paths.chown_to_real_user(file.home)


def unpack(root: pathlib.Path) -> ImportResult:
    """Move a conflict-free pack into this home, keyed to the current project root.

    Args:
        root: The resolved project root.

    Returns:
        The moved and kept files, with their origin.

    Raises:
        PackError: A conflict, invalid manifest or filesystem error prevents the import.
    """
    root = root.resolve()
    manifest = _pack_manifest.read(root)
    try:
        with contextlib.ExitStack() as locks:
            # Config and secrets writers use the same target-keyed lock.
            for name in (
                "config/config.toml",
                "config/secrets.toml",
                "config/ui.toml",
                "repo/config.toml",
            ):
                if name not in manifest.files:
                    continue
                target = _home(root, name)
                _pack_manifest.check_path(target)
                paths.mkdir_for_real_user(target.parent)
                if not locks.enter_context(portable.locked_file(target)):
                    raise errors.PackError(f"{target}: cannot take writer lock; nothing moved")
            facts = _preflight(root, manifest)
            for file in facts:
                if file.outcome == "move":
                    _move(file)
                elif file.name == "config/secrets.toml":
                    _copy(file.home, file.home, secret=True)
            state = paths.state_dir(root, allow_pack=True)
            paths.mkdir_for_real_user(state)
            portable.atomic_write(
                state / "pack-base.json",
                json.dumps({f.name: f.sha256 for f in facts}, indent=2) + "\n",
                mode=0o600,
            )
            paths.chown_to_real_user(state / "pack-base.json")
            shutil.rmtree(root / ".agent6")
            return ImportResult(
                manifest.origin,
                tuple(f.home for f in facts if f.outcome == "move"),
                tuple(f.home for f in facts if f.outcome == "keep"),
            )
    except OSError as exc:
        raise errors.PackError(f"unpack {root / '.agent6'}: {exc}") from exc
