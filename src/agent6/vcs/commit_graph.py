"""Read the commit relationships used by reversal guidance."""

from __future__ import annotations

import pathlib
import re

from agent6 import errors
from agent6.vcs import _git_command


def parents(root: pathlib.Path, commit: str) -> tuple[str, ...]:
    """Return a recorded commit's parents, refusing names and missing objects.

    Args:
        root: The repository.
        commit: A full recorded object id.

    Returns:
        Parent ids in git's recorded order.

    Raises:
        GitError: The id is invalid or its commit cannot be read.
    """
    if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", commit):
        raise errors.GitError("invalid recorded commit id")
    result = _git_command.run(root, "show", "-s", "--format=%P", f"{commit}^{{commit}}", "--")
    return tuple(result.stdout.strip().split())


def landed_commits(root: pathlib.Path, before: str, after: str) -> tuple[str, ...]:
    """Return the landed range in reverse topological order.

    Args:
        root: The repository.
        before: The validated starting commit.
        after: The validated ending commit.

    Returns:
        Commits to consider from newest to oldest.
    """
    result = _git_command.run(root, "rev-list", "--topo-order", f"{before}..{after}", "--")
    return tuple(result.stdout.split())
