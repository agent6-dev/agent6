# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Complete a workspace path at an `@` in a composer."""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
import re
from collections.abc import Iterable, Sequence

from agent6 import errors
from agent6.config import layer
from agent6.sessions import manifest
from agent6.vcs.git import files as git_files
from agent6.vcs.git import read as git_read

_REF_CHAR = r"[A-Za-z0-9_./\-]"
_FRAGMENT_RE = re.compile(rf"(?<![\w@/])@({_REF_CHAR}*)\Z")
_REF_PATH_RE = re.compile(rf"{_REF_CHAR}+")


REF_SUGGESTIONS_MAX = 8  # the paths one completion offers


def common_prefix(texts: Sequence[str]) -> str:
    """Return the leading characters every text shares, as a completion extends to."""
    if not texts:
        return ""
    low, high = min(texts), max(texts)
    for at, char in enumerate(low):
        if char != high[at]:
            return low[:at]
    return low


def session_workspace(
    repo: pathlib.Path,
    config_paths: tuple[pathlib.Path, ...] = (),
    session_dir: pathlib.Path | None = None,
) -> tuple[pathlib.Path, tuple[str, ...]]:
    """Return the root and the hidden set for path completion.

    Args:
        repo: The repository the front-end serves.
        config_paths: The front-end's `--config` files.
        session_dir: The session the words go to; None for new work, which starts in `repo`.

    Returns:
        The session's worktree, or `repo` without one, and `[sandbox].hide_paths` there.

    Raises:
        ConfigError: The config does not load.
    """
    cfg = layer.load_effective(repo, config_paths).config
    worktree: pathlib.Path | None = None
    if session_dir is not None:
        with contextlib.suppress(manifest.ManifestError):
            worktree = manifest.read_manifest(session_dir).worktree
    if worktree is None:
        return repo, tuple(cfg.sandbox.hide_paths)
    return worktree, tuple(cfg.with_hidden_in_checkout(repo, worktree).sandbox.hide_paths)


def ref_fragment(before_caret: str) -> str | None:
    """Return the `@path` fragment at the end of the text.

    Args:
        before_caret: The composer's text up to the caret.

    Returns:
        The characters after the `@`, "" right after a bare one; None outside a reference.
    """
    match = _FRAGMENT_RE.search(before_caret)
    return match.group(1) if match else None


@dataclasses.dataclass(frozen=True, slots=True)
class RefSuggestions:
    """The workspace files a fragment completes to.

    Attributes:
        paths: The first matches, root-relative: those starting with the fragment, then
            those containing it.
        more: How many matches `paths` leaves out.
        common: The longest prefix every path starting with the fragment shares; the
            fragment itself when none starts with it.
    """

    paths: tuple[str, ...]
    more: int
    common: str


def suggest_refs(
    root: pathlib.Path,
    hide: Iterable[str],
    fragment: str,
    *,
    limit: int = REF_SUGGESTIONS_MAX,
) -> RefSuggestions:
    """Return the working-tree files an `@fragment` completes to.

    Git lists the working-tree files, excluding ignored files. Completion skips hidden paths,
    symlinks, submodules and names outside the fragment character set. The match ignores case.

    Args:
        root: The workspace root.
        hide: `[sandbox].hide_paths`; agent6's private dirs are hidden without it.
        fragment: The characters typed after the `@`.
        limit: The most paths to return.

    Returns:
        The matches; none outside a git repository.
    """
    try:
        entries = git_files.worktree_tree_files(root)
    except errors.GitError:
        return RefSuggestions((), 0, fragment)
    hidden = git_read.hidden_in_repo(root, tuple(hide))
    names = [
        entry.path
        for entry in entries
        if entry.kind == "file"
        and _REF_PATH_RE.fullmatch(entry.path)
        and not git_read.excluded(entry.path, hidden)
    ]
    wanted = fragment.lower()
    starts = [name for name in names if name.lower().startswith(wanted)]
    matches = starts + [
        name for name in names if wanted in name.lower() and not name.lower().startswith(wanted)
    ]
    common = common_prefix(starts) if starts else fragment
    return RefSuggestions(tuple(matches[:limit]), max(len(matches) - limit, 0), common)
