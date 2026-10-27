"""Inspect submodule dirt without recursive Git commands or index writes."""

from __future__ import annotations

import pathlib

from agent6 import errors
from agent6.tools import _path_safety
from agent6.vcs import _git_command


def dirty_paths(
    root: pathlib.Path, selected: tuple[str, ...], workspace: _path_safety.Workspace
) -> tuple[str, ...]:
    """Find selected submodules with unreviewed worktree content without writing their indexes.

    Args:
        root: The superproject root.
        selected: The repository-relative paths selected for review.
        workspace: The review's hidden-path policy.

    Returns:
        The selected, non-hidden dirty gitlink paths in the superproject.

    Raises:
        GitError: A submodule is symlinked or its status cannot be read.
    """
    selected_names = set(selected)
    pending = [(root, "")]
    dirty: set[str] = set()
    while pending:
        directory, owner = pending.pop()
        if owner in dirty or workspace.is_denied(directory):
            continue
        if owner:
            if directory.resolve() != directory:
                raise errors.GitError("Review cannot inspect a symlinked submodule.")
            if not (directory / ".git").exists():
                continue
            exclusions = [
                f":(top,exclude,icase,literal){'/'.join(denied.parts[len(directory.parts) :])}"
                for denied in sorted(set(workspace.denied))
                if _path_safety.path_within(denied, directory)
            ]
            # Each repository needs its own driver overrides; Git's recursion bypasses them.
            if _git_command.run(
                directory,
                "status",
                "--porcelain",
                "--ignore-submodules=dirty",
                "--untracked-files=normal",
                "--",
                ":/",
                *exclusions,
                env_extra={"GIT_OPTIONAL_LOCKS": "0"},
            ).stdout:
                dirty.add(owner)
                continue
        for entry in _git_command.nul_fields(directory, "ls-files", "--stage", "-z"):
            if not entry.startswith("160000 "):
                continue
            name = entry.partition("\t")[2]
            if (owner or name in selected_names) and (directory / name).is_dir():
                pending.append((directory / name, owner or name))
    return tuple(sorted(dirty))
