# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Inline and complete the `@path` file references in words the operator typed."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pathlib
import re
from collections.abc import Iterable

from agent6 import errors, path_safety
from agent6.config import layer
from agent6.sessions import manifest
from agent6.vcs.git import files as git_files
from agent6.vcs.git import read as git_read

_REF_CHAR = r"[A-Za-z0-9_./\-]"
_TASK_FILE_REF_RE = re.compile(rf"(?<![\w@/])@({_REF_CHAR}+)")
_FRAGMENT_RE = re.compile(rf"(?<![\w@/])@({_REF_CHAR}*)\Z")
_REF_PATH_RE = re.compile(rf"{_REF_CHAR}+")


TASK_FILE_REF_MAX_BYTES = 64 * 1024  # per file; a bigger read needs an explicit tool call
REF_SUGGESTIONS_MAX = 8  # the paths one completion offers


def expand_task_file_refs(task: str, root: pathlib.Path, hide: Iterable[str] = ()) -> str:
    """Inline the `@path` references in the task that name readable workspace files.

    A match starts at a word boundary that excludes `@` and `/`, so `user@example.com`
    is untouched. A path resolves inside the host-side read boundary
    (`path_safety.host_workspace`): a regular file inside the root, outside `.git` and the
    hidden set, read through a descriptor walked from the root. Any other reference is left as
    typed. At most `TASK_FILE_REF_MAX_BYTES` are read, and decoded lossily. Never raises.

    Args:
        task: The words the operator typed; text a model or a repository file wrote is never
            passed, since a reference in it would read a file the operator did not name.
        root: The workspace root.
        hide: `[sandbox].hide_paths`; agent6's private dirs are hidden without it.

    Returns:
        The task with a `<file>` block after each resolved reference, which stays as typed.
    """
    workspace = path_safety.host_workspace(root, hide)

    def _replace(match: re.Match[str]) -> str:
        rel = match.group(1)
        try:
            fd = path_safety.open_contained(workspace.resolve_read(rel), os.O_RDONLY)
            with os.fdopen(fd, "rb") as handle:
                raw = handle.read(TASK_FILE_REF_MAX_BYTES)
                omitted = os.fstat(fd).st_size - len(raw)
        except (OSError, path_safety.ContainmentError):
            return match.group(0)
        text = raw.decode("utf-8", errors="replace")
        suffix = ""
        if omitted > 0:
            suffix = (
                f"\n... (truncated, {omitted} bytes omitted; "
                "use read_file with an explicit range for the rest)"
            )
        return f'{match.group(0)}\n<file path="{rel}">\n{text}{suffix}\n</file>\n'

    return _TASK_FILE_REF_RE.sub(_replace, task)


def session_workspace(
    repo: pathlib.Path,
    config_paths: tuple[pathlib.Path, ...] = (),
    session_dir: pathlib.Path | None = None,
) -> tuple[pathlib.Path, tuple[str, ...]]:
    """Return the root and the hidden set an `@path` resolves against.

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
    """Return the `@path` fragment the text ends in, as `expand_task_file_refs` would match it.

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

    The files are git's working-tree listing, which leaves out ignored files. A path under
    the hidden set, a symlink, a submodule, and a path `expand_task_file_refs` cannot match
    are left out. The match ignores case.

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
    common = os.path.commonprefix(starts) if starts else fragment  # noqa: RUF071 - a character prefix
    return RefSuggestions(tuple(matches[:limit]), max(len(matches) - limit, 0), common)
