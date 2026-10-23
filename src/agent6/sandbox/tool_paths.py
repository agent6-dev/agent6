# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Resolve operator tools inside the jail.

The PATH a jailed command gets, the real tool dirs mounted read+exec for it, and
the notes a surface prints about tools it cannot reach or mounts that expose
more than a tool.
"""

from __future__ import annotations

import dataclasses
import pathlib
import shutil

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

    Args:
        p: The dir a mount would bind.

    Returns:
        True for $HOME and its ancestors, and for a dir inside or above one of agent6's
        private dirs (`agent6.paths.private_dirs`).
    """
    home = pathlib.Path.home()
    if home.is_relative_to(p) or home.resolve().is_relative_to(p):
        return True
    return any(
        p.is_relative_to(d) or d.is_relative_to(p)
        for private in paths.private_dirs()
        for d in (private, private.resolve())
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ToolMount:
    """A real dir the jail mounts read+exec for operator tools.

    Attributes:
        path: The dir, at its real location.
        on_path: The directory is on the jail's PATH.
        links: Symlink sources whose targets live under this directory.
        root_refusal: The refused install root, with the PATH reason when applicable.
    """

    path: pathlib.Path
    on_path: bool = False
    links: tuple[pathlib.Path, ...] = ()
    root_refusal: str = ""


def operator_tool_mounts() -> tuple[ToolMount, ...]:
    """Return the dirs the jail mounts read+exec, each with its reason, sorted by path.

    Recomputed per call, so a tool just installed is picked up. A symlink into
    `<root>/bin` grants the root only if it holds no jail PATH directory and passes
    the mount safety checks. Other links grant the target's directory.

    Returns:
        The mounts, with every discovery source retained for each directory.
    """
    bin_dirs = tuple(d for d in _tool_bin_dirs() if d.is_dir())
    path_dirs = tuple(pathlib.Path(p) for p in _JAIL_BASE_PATH_DIRS) + bin_dirs
    on_path: set[pathlib.Path] = set()
    links: dict[pathlib.Path, list[pathlib.Path]] = {}
    root_refusals: dict[pathlib.Path, str] = {}
    for d in bin_dirs:
        if not _under_system_root(d) and not _never_mounted(d):
            on_path.add(d)
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for entry in sorted(entries):
            if not entry.is_symlink():
                continue
            try:
                real = entry.resolve()
            except OSError:
                continue
            parent = real.parent
            if not real.is_file() or _never_mounted(parent):
                continue
            if parent.name == "bin":
                root = parent.parent
                if _never_mounted(root):
                    root_refusals[parent] = str(root)
                elif any(p.is_relative_to(root) for p in path_dirs):
                    root_refusals[parent] = f"{root} holds a PATH directory"
                else:
                    parent = root
            if not _under_system_root(parent) or parent in root_refusals:
                links.setdefault(parent, []).append(entry)
    return tuple(
        ToolMount(path, path in on_path, tuple(links.get(path, ())), root_refusals.get(path, ""))
        for path in sorted(on_path | links.keys())
    )


def operator_tool_paths() -> tuple[str, tuple[pathlib.Path, ...]]:
    """Return the jail's PATH and the real-location dirs to mount read+exec.

    Returns:
        The PATH string and the sorted mount dirs.
    """
    path_dirs = [*_JAIL_BASE_PATH_DIRS, *(str(d) for d in _tool_bin_dirs() if d.is_dir())]
    return ":".join(path_dirs), tuple(
        m.path for m in operator_tool_mounts() if not _under_system_root(m.path)
    )


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

    Returns:
        `/usr/bin:/bin`, then each existing bin dir scanned for operator tools, such as
        `~/.local/bin`.
    """
    return operator_tool_paths()[0]


def jail_resolves(exe: str) -> bool:
    """Return whether a command's first word resolves in the jail, by the same advisory probe.

    Args:
        exe: The command's first word.

    Returns:
        True when `exe` contains a slash or names an executable on `jail_search_path()`.
    """
    return "/" in exe or shutil.which(exe, path=jail_search_path()) is not None
