# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Resolve operator tools inside the jail.

The PATH a jailed command gets, the real tool dirs mounted read+exec for it, and
the notes a surface prints about tools it cannot reach or mounts that expose
more than a tool.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib

from agent6 import paths

# The jail mounts only these system roots; a tool elsewhere needs a read+exec mount of its
# real dir, or a jailed command dies 127. One owner, so dispatch, machine.engine and
# `machine check` resolve tools identically.
_JAIL_BASE_PATH_DIRS = ("/usr/bin", "/bin")
_SYSTEM_ROOTS = (
    pathlib.Path("/usr"),
    pathlib.Path("/bin"),
    pathlib.Path("/sbin"),
    pathlib.Path("/lib"),
    pathlib.Path("/lib64"),
    pathlib.Path("/etc"),
    pathlib.Path("/dev"),
)


def _under_system_root(p: pathlib.Path) -> bool:
    """Return whether the path lies under a root the jail already mounts."""
    return any(p.is_relative_to(r) for r in _SYSTEM_ROOTS)


def _never_mounted(p: pathlib.Path) -> bool:
    """Return whether a dir must never be a jail mount, however a tool symlink resolves.

    Refused by identity, never by content: $HOME and its ancestors (a mount there
    hands the jail `~/.ssh` and every credential), and agent6's private dirs
    (`agent6.paths.private_dirs`: `secrets.toml`, memory, transcripts) in either
    direction, since a mount above a private dir grants the same reads. A dir below
    home stays allowed; that keeps `~/.local/bin` tools working.
    """
    if pathlib.Path.home().is_relative_to(p):
        return True
    return any(p.is_relative_to(d) or d.is_relative_to(p) for d in paths.private_dirs())


def operator_tool_paths() -> tuple[str, tuple[pathlib.Path, ...]]:
    """Return the jail's PATH and the real-location dirs to mount read+exec.

    Recomputed per call, so a tool just installed is picked up. A bin dir under a
    mounted system root only joins PATH; a dir outside one, and the real dir a
    symlink resolves to, also needs the mount.

    Returns:
        The PATH string and the sorted mount dirs.
    """
    path_dirs: list[str] = list(_JAIL_BASE_PATH_DIRS)
    candidates = _tool_bin_dirs()
    mounts: set[pathlib.Path] = set()
    for d in candidates:
        if not d.is_dir():
            continue
        path_dirs.append(str(d))
        if not _under_system_root(d) and not _never_mounted(d):
            mounts.add(d)
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_symlink():
                continue
            try:
                real = entry.resolve()
            except OSError:
                continue
            if real.is_file() and not _under_system_root(real) and not _never_mounted(real.parent):
                mounts.add(real.parent)
    # uv-managed CPython lives under XDG data; without this mount an in-jail `uv run` sees the
    # venv's interpreter missing and recreates the operator's .venv. A mount, never a PATH entry.
    data_home = pathlib.Path(
        os.environ.get("XDG_DATA_HOME") or pathlib.Path.home() / ".local/share"
    )
    uv_pythons = data_home / "uv" / "python"
    if uv_pythons.is_dir():
        mounts.add(uv_pythons)
    return ":".join(path_dirs), tuple(sorted(mounts))


def _tool_bin_dirs() -> tuple[pathlib.Path, ...]:
    """Return the bin dirs scanned for operator-installed tools."""
    home = pathlib.Path.home()
    return (
        pathlib.Path("/usr/local/bin"),
        pathlib.Path("/usr/local/sbin"),
        home / ".local/bin",
        home / ".cargo/bin",
        pathlib.Path("/opt/homebrew/bin"),
        pathlib.Path("/snap/bin"),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ToolMountNotes:
    """How the operator's bin dirs resolve into the jail, for the once-per-run preflight.

    Both lists hold `"<link> -> <target>"` strings; the notes change no mount decision.

    Attributes:
        unreachable: Symlinks whose target dir is never mounted, so the tool is absent
            in the jail and would die 127 with nothing naming the reason.
        exposes_home_dir: Symlinks resolving out of their bin dir into another dir
            under $HOME, which is therefore mounted read-only into the jail.
    """

    unreachable: tuple[str, ...] = ()
    exposes_home_dir: tuple[str, ...] = ()


def tool_mount_notes() -> ToolMountNotes:
    """Report the tools the jail cannot reach and the home dirs a tool drags into it.

    Returns:
        The notes over the bin dirs the jail puts on PATH.
    """
    home = pathlib.Path.home()
    bin_dirs = _tool_bin_dirs()
    unreachable: list[str] = []
    exposes: list[str] = []
    for d in bin_dirs:
        try:
            entries = list(d.iterdir()) if d.is_dir() else []
        except OSError:
            continue
        for entry in entries:
            if not entry.is_symlink():
                continue
            try:
                real = entry.resolve()
            except OSError:
                continue
            if not real.is_file() or _under_system_root(real):
                continue
            parent = real.parent
            if _never_mounted(parent):
                unreachable.append(f"{entry} -> {real}")
            elif parent.is_relative_to(home) and parent not in bin_dirs:
                exposes.append(f"{entry} -> {real}")
    return ToolMountNotes(tuple(sorted(unreachable)), tuple(sorted(exposes)))


def jail_search_path() -> str:
    """Return the PATH a jailed command resolves against, for host-side probes.

    Advisory only; the jail recomputes its own per call.
    """
    return operator_tool_paths()[0]
