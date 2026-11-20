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
import tomllib
from collections.abc import Callable, Iterator
from typing import Literal

import pydantic

from agent6 import errors, pack_manifest, paths, portable
from agent6.config import io, layer, layer_data, write
from agent6.machine import schema as machine_schema
from agent6.sessions import event_log, journal, layout, lock, process
from agent6.sessions import manifest as sessions_manifest
from agent6.viewmodel import listing
from agent6.viewmodel.machine import state as machine_state
from agent6.viewmodel.session import status as session_status

Outcome = Literal["move", "keep", "conflict"]
# The packed files a config or secrets writer also writes; `unpack` lists what each one sets.
_CONFIG_NAMES = (
    "config/config.toml",
    "config/secrets.toml",
    "config/ui.toml",
    "repo/config.toml",
)


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
class Setting:
    """Hold one leaf a packed config file sets.

    Attributes:
        key: The dotted key.
        value: The value as JSON; "" for a secret, which is never shown.
        operator_only: The key is one no machine overlay may carry, or a `sandbox` key.
    """

    key: str
    value: str
    operator_only: bool


@dataclasses.dataclass(frozen=True, slots=True)
class Preview:
    """Hold what an unpack would do, read before anything is written.

    Attributes:
        origin: Where the manifest says the pack is from.
        packed_at: When the manifest says it was packed.
        files: Every packed file with its destination and outcome.
        settings: Every leaf of each config file the unpack would install, by pack name.
    """

    origin: pack_manifest.Origin
    packed_at: str
    files: tuple[FileFacts, ...]
    settings: dict[str, tuple[Setting, ...]]


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
        text = write.editable_text(target)
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
                owner = io.undeclared_table_ancestor(text, key) or key
                text = io.upsert_leaf_text(
                    text, owner, io.read_toml_leaf(merged, owner), path=target
                )
            error = write.write_validated(
                root,
                target,
                text,
                written=[(key, leaves[key]) for key in moved],
                to_repo=True,
                allow_pack=True,
            )
        except (OSError, errors.OperatorError) as exc:
            raise errors.PackError(str(exc)) from exc
        if error:
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
                    source,
                    exclude=("locks", "pack-base.json", "worktrees") if area == "repo" else (),
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


def _exclusive_leases(root: pathlib.Path, verb: str = "pack") -> list[int]:
    """Take the workspace lease of the root and of every checkout above it, alone.

    A live run, exec or machine of any of them holds the lease shared. Its sandbox could
    read the secrets a pack writes, and its commands can rewrite a pack being unpacked; the
    parents count because a nested checkout sits inside their workspace.

    Args:
        root: The resolved project root.
        verb: The command the refusal names: `pack` or `unpack`.

    Returns:
        The held fds, released when the pack is written or moved.

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
                f" for it, then {verb}"
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


def _settings(file: FileFacts) -> tuple[Setting, ...]:
    """Return every leaf a packed config file sets, from the bytes the plan hashed.

    Args:
        file: The file's preflight decision.

    Returns:
        The leaves in key order, each marked when the key is the operator's alone: one
        `machine_schema.protected_overlay_key_error` refuses in an overlay, or a `sandbox` key.

    Raises:
        PackError: The file changed since it was hashed, or is not valid TOML.
    """
    data = file.packed.read_bytes()
    if hashlib.sha256(data).hexdigest() != file.sha256:
        raise errors.PackError(f"{file.packed}: changed while unpack read it; nothing moved")
    try:
        table = tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise errors.PackError(
            f"{file.packed}: not valid TOML, so unpack cannot list what it sets: {exc}"
        ) from exc
    secret = file.name == "config/secrets.toml"
    return tuple(
        Setting(
            key,
            "" if secret else json.dumps(value, default=str),
            key.split(".", 1)[0] == "sandbox"
            or machine_schema.protected_overlay_key_error(key) is not None,
        )
        for key, value in sorted(layer_data.flatten_leaves(table).items())
    )


def preview(root: pathlib.Path) -> Preview:
    """Read what an unpack would install, writing nothing.

    The list is complete by construction: every packed file, and every leaf of each config
    file among those to install. No list of risky keys decides what is shown.

    Args:
        root: The resolved project root.

    Returns:
        The files, their destinations and outcomes, and the settings.

    Raises:
        PackError: A session of the checkout is live, or the pack is missing, invalid,
            unreadable or in conflict with this home.
    """
    root = root.resolve()
    held = _exclusive_leases(root, "unpack")
    try:
        manifest = pack_manifest.read(root)
        facts = _preflight(root, manifest)
        return Preview(
            manifest.origin,
            manifest.model_dump(mode="json")["packed_at"],
            facts,
            {
                file.name: _settings(file)
                for file in facts
                if file.outcome == "move" and file.name in _CONFIG_NAMES
            },
        )
    except OSError as exc:
        raise errors.PackError(f"unpack {root / '.agent6'}: {exc}") from exc
    finally:
        for fd in held:
            lock.release_single_writer(fd)


def unpack(root: pathlib.Path, *, shown: Preview | None = None) -> ImportResult:
    """Move a conflict-free pack into this home, keyed to the current project root.

    The files are trusted as they are: no hash in the manifest is compared with them. A
    caller that asked the operator first passes what it showed, and the import refuses a
    pack or a home that differs from it.

    Args:
        root: The resolved project root.
        shown: The preview the operator agreed to; None for a caller that asked nothing.

    Returns:
        The moved and kept files, with their origin.

    Raises:
        PackError: A live session of the checkout, a conflict, an invalid manifest, a change
            since the preview or a filesystem error prevents the import.
    """
    root = root.resolve()
    held = _exclusive_leases(root, "unpack")
    try:
        manifest = pack_manifest.read(root)
        with contextlib.ExitStack() as locks:
            # Config and secrets writers use the same target-keyed lock.
            for name in _CONFIG_NAMES:
                if name not in manifest.files:
                    continue
                target = _home(root, name)
                pack_manifest.check_path(target)
                paths.mkdir_for_real_user(target.parent)
                if not locks.enter_context(portable.locked_file(target)):
                    raise errors.PackError(f"{target}: cannot take writer lock; nothing moved")
            facts = _preflight(root, manifest)
            if shown is not None and facts != shown.files:
                raise errors.PackError(
                    f"{root / '.agent6'}: the pack or this home changed after unpack listed"
                    " them; nothing moved. Run agent6 unpack again."
                )
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
    finally:
        for fd in held:
            lock.release_single_writer(fd)


def _set_aside(directory: pathlib.Path) -> tuple[str, list[str], list[str]]:
    """Move a pack's names out of `.agent6` into a new `pack.invalid` directory there.

    Every name is relative to a descriptor opened without following a link, so nothing
    outside `.agent6` is read, written or deleted. A name already taken, by a link too, is
    left as it is: the first free `pack.invalid.N` is made.

    Args:
        directory: The `.agent6` directory.

    Returns:
        The new directory's name, the areas moved with the manifest, and the areas that
        would not move, each with its error.

    Raises:
        OSError: `.agent6` is a link, or the manifest is gone or cannot be moved.
    """
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.lstat("pack.json", dir_fd=fd)
        taken = 0
        while True:
            name = "pack.invalid" + (f".{taken}" if taken else "")
            try:
                os.mkdir(name, 0o700, dir_fd=fd)
                break
            except FileExistsError:
                taken += 1
        aside = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        try:
            user = paths.effective_user()
            if user.via_sudo:
                os.chown(aside, user.uid, user.gid)
            try:
                os.rename("pack.json", "pack.json.invalid", src_dir_fd=fd, dst_dir_fd=aside)
            except OSError:
                with contextlib.suppress(OSError):
                    os.rmdir(name, dir_fd=fd)
                raise
            moved: list[str] = []
            stuck: list[str] = []
            for area in pack_manifest.AREAS:
                try:
                    os.rename(area, area, src_dir_fd=fd, dst_dir_fd=aside)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    # A directory moves only when it is writable: its `..` entry changes.
                    stuck.append(f"`{area}/` ({exc})")
                else:
                    moved.append(f"`{area}/`")
        finally:
            os.close(aside)
    finally:
        os.close(fd)
    return name, moved, stuck


def _listed(names: list[str]) -> str:
    """Return the names as one phrase: `a`, `a and b`, `a, b and c`."""
    return " and ".join((", ".join(names[:-1]), names[-1])) if len(names) > 1 else names[0]


# Why a pack an ending operation finds is its own; `_writer` gives the start check's.
_ENDING = "No agent6 command starts beside a pack, so this operation"


def quarantine(
    root: pathlib.Path,
    *,
    warn: Callable[[str], None],
    events: event_log.EventSink | None = None,
    premise: str = _ENDING,
) -> None:
    """Set aside a pack that `agent6 pack` did not write, and say so.

    For a caller that knows whose the pack is. An ending operation started beside no pack
    and holds the checkout's lease, which `pack` takes alone, or it saw none before its
    landing; the start check names the session (`check_start`).
    The manifest and the `config/`, `repo/` and `skills/` beside it move into one new
    directory, `.agent6/pack.invalid`: a later `agent6 pack` refuses to write beside those
    names, and a second `pack.json` would make them a pack again. A link moves as a link,
    and a linked `.agent6` is reported and left as it is.

    Args:
        root: The directory whose `.agent6` holds the pack.
        warn: Receives the operator's warning when no journal takes it.
        events: The session's journal, whose row every view of the session shows; None
            outside a session.
        premise: Who wrote the pack and why that follows, as the warning's subject.
    """
    directory = root / ".agent6"
    found = directory / "pack.json"
    # The start check's own view, through a linked `.agent6` too; a stat writes nothing there.
    if not os.path.lexists(found):
        return
    renamed: pathlib.Path | None = None
    try:
        name, moved, stuck = _set_aside(directory)
    except OSError as exc:
        if isinstance(exc, FileNotFoundError) and not os.path.lexists(found):
            return  # gone between the stat and the move
        if directory.is_symlink():
            warning = (
                f"{directory} is a symlink that leads to a pack.json. {premise} made the link"
                " or the file. agent6 follows no link there and changed nothing: remove the"
                " link. Do not run `agent6 unpack` there."
            )
        else:
            warning = (
                f"found {found}. {premise} created it; `agent6 pack` did not. agent6 could"
                f" not move it ({exc}): delete it. Do not run `agent6 unpack` on it."
            )
    else:
        renamed = directory / name / "pack.json.invalid"
        beside = f", with {_listed(moved)}" if moved else ""
        left = f" It could not move {_listed(stuck)}: delete that." if stuck else ""
        warning = (
            f"found {found}. {premise} created it; `agent6 pack` did not. agent6 moved it to"
            f" `{name}/pack.json.invalid`{beside}.{left} Do not run `agent6 unpack` on it."
        )
    if events is not None:
        # Said once: a dead journal leaves the warning to `warn`.
        with contextlib.suppress(event_log.EventWriteError):
            events.emit(
                journal.PackQuarantined(
                    path=str(found),
                    renamed=None if renamed is None else str(renamed),
                    warning=warning,
                )
            )
            return
    warn(warning)


def quarantine_tree(
    root: pathlib.Path,
    *,
    warn: Callable[[str], None],
    events: event_log.EventSink | None = None,
) -> None:
    """Set aside every pack in a checkout's tree, for an ending operation that held its lease.

    The run preflight refused each pack its walk found (`pack_manifest.pack_dirs`), a nested
    directory's too, so one this walk finds appeared under the operation.

    Args:
        root: The checkout the operation ran in.
        warn: Receives each warning no journal takes.
        events: The session's journal; None outside a session.
    """
    for directory in pack_manifest.pack_dirs(root, onerror=lambda _exc: None):
        quarantine(directory, warn=warn, events=events)


def _writer(directory: pathlib.Path) -> str | None:
    """Return the session that wrote a pack in the directory, as a warning's premise.

    The one rule the start check applies: a pack beside a session of the checkout that is
    live, or that died without an end, was written by that session. No session starts beside
    a pack, `agent6 pack` writes beside no live one, and a session that reaches its end sets
    aside what it wrote. A checkout above the directory counts, since the directory sits
    inside its workspace; a session that works in a worktree or a clone of its own does not.

    Args:
        directory: The directory whose `.agent6` holds the pack.

    Returns:
        The premise, or None when no such session exists.
    """
    for checkout in lock.checkouts_from(directory):
        state = paths.state_dir(checkout, allow_pack=True)
        for session_dir in listing.session_dirs(state, layout.SESSION_BUCKETS):
            # A session clears its `worker.pid` at its end, after its own check.
            if process.read_worker_pid(session_dir) is None:
                continue
            try:
                manifest = sessions_manifest.read_manifest(session_dir)
            except sessions_manifest.ManifestError:
                manifest = None  # nothing says it works elsewhere
            works_elsewhere = manifest is not None and (
                manifest.worktree is not None or manifest.parallel is not None
            )
            if works_elsewhere:
                continue
            word = listing.summarize_session_dir(session_dir).status
            if word in session_status.LIVE_STATUS_WORDS:
                return f"Session {session_dir.name} is live in {checkout}, so that session"
            if word == "stale":
                return (
                    f"Session {session_dir.name} died in {checkout} without an end, so that session"
                )
        for machine_dir in machine_state.machine_instance_dirs(state):
            if process.read_worker_pid(machine_dir) is None:
                continue
            how = "is live" if process.worker_is_alive(machine_dir) else "died"
            end = "" if how == "is live" else " without an end"
            return f"Machine {machine_dir.name} {how} in {checkout}{end}, so that machine"
    return None


def check_start(directory: pathlib.Path, *, warn: Callable[[str], None]) -> None:
    """Set aside a pack that a session of the checkout wrote, so the command starts.

    `paths.project_root` refuses every pack. Where a command begins, this first applies the
    end check's rule, move and warning to a pack a session wrote (`_writer`); a pack beside
    no such session stays for the refusal, which names `agent6 unpack`.

    Args:
        directory: The directory whose `.agent6` may hold a pack: a project root, or a
            directory the run preflight's walk found.
        warn: Receives the operator's warning.
    """
    if not os.path.lexists(directory / ".agent6" / "pack.json"):
        return
    premise = _writer(directory)
    if premise is not None:
        quarantine(directory, warn=warn, premise=premise)
