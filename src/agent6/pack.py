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
from collections.abc import Callable, Iterator
from typing import Literal

import pydantic

from agent6 import errors, pack_manifest, paths, portable
from agent6.config import io, layer, layer_data, write
from agent6.sessions import lock

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

    origin: pack_manifest.Origin
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
    pack_manifest.check_path(root)
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


def _fold_global_config(root: pathlib.Path) -> tuple[str, ...]:
    """Copy inherited global leaves into the repo under one validated config write.

    Args:
        root: The project whose workspace lease is held exclusively.

    Returns:
        The copied dotted keys, sorted.

    Raises:
        PackError: A path is unsafe, the writer lock is unavailable or the write fails.
        ConfigError: The source or merged config is invalid.
    """
    source = paths.global_config_path()
    target = paths.repo_config_path(root, allow_pack=True)
    for path in (source, target):
        pack_manifest.check_path(path, regular=path.exists())
    paths.mkdir_for_real_user(target.parent)
    with write.writing_config(target) as held:
        if not held:
            raise errors.PackError(f"{target}: cannot take writer lock; nothing folded")
        prior = errors.read_operator_file(target) if target.is_file() else None
        merged, sources = layer_data.merge_layers(
            [
                layer_data.Layer("global", source, io.read_toml_file(source)),
                layer_data.Layer("repo", target, io.read_toml_file(target)),
            ]
        )
        leaves = layer_data.flatten_leaves(merged)
        moved = tuple(sorted(key for key, origin in sources.items() if origin == "global"))
        layer.load_effective(root, allow_pack=True)
        try:
            for key in moved:
                owner = io.undeclared_table_ancestor(target, key) or key
                io.upsert_toml_leaf(target, owner, io.read_toml_leaf(merged, owner))
        except (OSError, errors.OperatorError) as exc:
            raise errors.PackError(
                write.keep_or_rollback(target, prior, str(exc), held=held)
            ) from exc
        if error := write.revalidate_write(
            root,
            target,
            prior,
            was_valid=True,
            held=held,
            written=[(key, leaves[key]) for key in moved],
            to_repo=True,
            allow_pack=True,
        ):
            raise errors.PackError(error)
        return moved


def _sources(
    root: pathlib.Path, *, repo_config: bool, on_fold: Callable[[str], None] | None
) -> dict[str, pathlib.Path]:
    """Fold settings when requested and collect the files the snapshot carries.

    Args:
        root: The resolved project root.
        repo_config: Fold global settings and omit global config and UI files.
        on_fold: Report each copied dotted key.

    Returns:
        The source paths keyed by pack-relative name.
    """
    if repo_config:
        for key in _fold_global_config(root):
            if on_fold is not None:
                on_fold(key)
    sources = {
        f"config/{name}": paths.global_config_dir() / name
        for name in pack_manifest.CONFIG_FILES
        if not repo_config or name == "secrets.toml"
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
    return sources


def pack(
    root: pathlib.Path,
    *,
    repo_config: bool = False,
    on_fold: Callable[[str], None] | None = None,
) -> pack_manifest.Manifest:
    """Copy this home's global and repository state into a private workspace pack.

    Args:
        root: The resolved project root.
        repo_config: Fold global settings into repo config and omit global config and UI files.
        on_fold: Report each dotted key copied into repo config after the validated write.

    Returns:
        The written manifest.

    Raises:
        PackError: A foreign pack, unsafe path or filesystem error prevents the snapshot.
    """
    root = root.resolve()
    directory = root / ".agent6"
    pack_manifest.check_path(directory)
    own = [directory / name for name in ("pack.json", *pack_manifest.AREAS)]
    if any(path.exists() for path in own) and not pack_manifest.is_own(
        pack_manifest.read(root).origin, paths.effective_user().home
    ):
        raise errors.PackError(f"{directory}: pack belongs to another origin; unpack first")
    held = _exclusive_leases(root)
    try:
        base_path = paths.state_dir(root, allow_pack=True) / "pack-base.json"
        pack_manifest.check_path(base_path, regular=base_path.exists())
        bases = pydantic.TypeAdapter(
            dict[pack_manifest.PackName, pack_manifest.Hash]
        ).validate_json(base_path.read_bytes() if base_path.exists() else b"{}")
        sources = _sources(root, repo_config=repo_config, on_fold=on_fold)
        paths.mkdir_for_real_user(directory)
        with tempfile.TemporaryDirectory(dir=directory) as stage_name:
            stage = pathlib.Path(stage_name)
            for area in pack_manifest.AREAS:
                (stage / area).mkdir(mode=0o700)
            files: dict[str, pack_manifest.PackedFile] = {}
            for name, source in sorted(sources.items()):
                pack_manifest.check_path(source, regular=True)
                target = stage / name
                _copy(source, target, secret=name == "config/secrets.toml")
                files[name] = pack_manifest.PackedFile(
                    sha256=_hash(target), mtime=_mtime(target), base=bases.get(name)
                )
            manifest = pack_manifest.Manifest(
                version=1,
                packed_at=datetime.datetime.now(datetime.UTC),
                origin=pack_manifest.Origin(
                    host=socket.gethostname(),
                    home=str(paths.effective_user().home),
                    repo_id=paths.repo_id(root),
                    workspace=str(root),
                ),
                files=files,
            )
            (stage / "pack.json").write_text(manifest.model_dump_json(indent=2) + "\n")
            _remove_own(directory)
            for child in stage.iterdir():
                child.replace(directory / child.name)
        paths.chown_to_real_user(directory)
        return manifest
    except (OSError, pydantic.ValidationError) as exc:
        raise errors.PackError(f"pack {directory}: {exc}") from exc
    finally:
        for fd in held:
            lock.release_single_writer(fd)


def _remove_own(directory: pathlib.Path) -> None:
    """Remove a pack's own names from `.agent6/`, leaving every other name there.

    Args:
        directory: The `.agent6/` directory.

    Raises:
        PackError: A name is a link.
    """
    for name in ("pack.json", *pack_manifest.AREAS):
        old = directory / name
        if not old.exists() and not old.is_symlink():
            continue
        pack_manifest.check_path(old)
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()


def _exclusive_leases(root: pathlib.Path) -> list[int]:
    """Take the workspace lease of the root and of every checkout above it, alone.

    A live run, exec or machine of any of them holds the lease shared, and its sandbox could
    read the secrets a pack writes; the parents count because a nested checkout sits inside
    their workspace.

    Args:
        root: The resolved project root.

    Returns:
        The held fds, released when the pack is written.

    Raises:
        PackError: A run, exec or machine of one of the checkouts is live.
    """
    held: list[int] = []
    for checkout in lock.checkouts_from(root):
        fd = lock.exclude_workspace(checkout)
        if fd is None:
            for earlier in held:
                lock.release_single_writer(earlier)
            raise errors.PackError(
                f"{checkout}: a run, exec or machine of this checkout is live; stop it or wait"
                " for it, then pack"
            )
        held.append(fd)
    return held


def _preflight(root: pathlib.Path, manifest: pack_manifest.Manifest) -> tuple[FileFacts, ...]:
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
    present = {
        path.relative_to(directory).as_posix()
        for area in pack_manifest.AREAS
        for path in _files(directory / area)
    }
    unlisted = present - manifest.files.keys()
    if unlisted:
        raise errors.PackError(
            f"{directory}: files absent from manifest: {', '.join(sorted(unlisted))}"
        )
    facts: list[FileFacts] = []
    for name in sorted(present):
        packed, home = directory / name, _home(root, name)
        pack_manifest.check_path(home)
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
    manifest = pack_manifest.read(root)
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
                pack_manifest.check_path(target)
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
            _remove_own(root / ".agent6")
            with contextlib.suppress(OSError):  # other names live there: worktrees, plans
                (root / ".agent6").rmdir()
            return ImportResult(
                manifest.origin,
                tuple(f.home for f in facts if f.outcome == "move"),
                tuple(f.home for f in facts if f.outcome == "keep"),
            )
    except OSError as exc:
        raise errors.PackError(f"unpack {root / '.agent6'}: {exc}") from exc
