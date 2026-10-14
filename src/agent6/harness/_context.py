# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Repo context for the system prompt: AGENTS.md discovery and the repo summary.

The summary holds the header line, the top-level listing, the file map and
recent commits.
"""

from __future__ import annotations

import pathlib

from agent6 import git_ops, kinds
from agent6.tools import _path_safety

_REPO_MAP_MAX_LINES = 60
_REPO_MAP_MAX_FILES_PER_DIR = 6
# AGENTS.md is injected whole; past this size the operator is warned at session start.
AGENTS_MD_WARN_CHARS = 40_000


def _agents_md_sources(
    root: pathlib.Path, hide: tuple[str, ...]
) -> tuple[tuple[pathlib.Path, str], ...]:
    """Return the readable AGENTS.md files from the git root through root, in order.

    A file whose real path leaves the git root (the root itself outside git), or enters
    `.git` or a hidden path, reads as absent.

    Args:
        root: The run's working directory.
        hide: `[sandbox].hide_paths`.

    Returns:
        (path, text) pairs, toplevel first.
    """
    base, directories = root, [pathlib.PurePath()]
    top = git_ops.toplevel(root)
    if top is not None and root.resolve().is_relative_to(top.resolve()):
        base = top.resolve()
        for part in root.resolve().relative_to(base).parts:
            directories.append(directories[-1] / part)
    names = [(directory / "AGENTS.md").as_posix() for directory in directories]
    return tuple(
        (base / name, text)
        for name in names
        if (text := _path_safety.read_workspace_text(base, name, hide))
    )


def agents_md_text(root: pathlib.Path, hide: tuple[str, ...] = ()) -> str:
    """Return the AGENTS.md text a session at root injects, whole.

    When root sits below a git toplevel, every ancestor file loads from the
    toplevel down; a file below the toplevel carries a heading naming its
    directory.

    Args:
        root: The run's working directory.
        hide: `[sandbox].hide_paths`.

    Returns:
        The files' texts joined by blank lines, "" when none is readable.
    """
    top = git_ops.toplevel(root)
    parts: list[str] = []
    for path, text in _agents_md_sources(root, hide):
        directory = path.parent.resolve()
        if top is None or directory == top.resolve():
            parts.append(text)
            continue
        label = f"{directory.relative_to(top.resolve()).as_posix()}/"
        if directory == root.resolve():
            label += " (this run's working directory)"
        parts.append(f"# AGENTS.md in {label}\n\n{text}")
    return "\n\n".join(parts)


def agents_md_notices(root: pathlib.Path, hide: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Return the session-start operator lines about the injected AGENTS.md.

    They name the files loaded when starting from a subdirectory, and warn on an
    oversize total (the text is injected whole; the remedy is trimming the file).

    Args:
        root: The run's working directory.
        hide: `[sandbox].hide_paths`.

    Returns:
        The lines, empty when there is nothing to say.
    """
    out: list[str] = []
    top = git_ops.toplevel(root)
    sources = _agents_md_sources(root, hide)
    if top is not None and top.resolve() != root.resolve():
        own = any(path.parent.resolve() == root.resolve() for path, _ in sources)
        ancestors = [path for path, _ in sources if path.parent.resolve() != root.resolve()]
        if ancestors:
            also = " plus this directory's" if own else ""
            root_loaded = any(path.parent.resolve() == top.resolve() for path in ancestors)
            location = (
                f", including the repo root's ({top})"
                if root_loaded
                else f" below the repo root ({top})"
            )
            files = f"{len(ancestors)} ancestor AGENTS.md file{'s' if len(ancestors) != 1 else ''}"
            out.append(f"loading {files}{location}{also}")
    text = agents_md_text(root, hide)
    total = len(text)
    if total > AGENTS_MD_WARN_CHARS:
        out.append(
            f"WARNING: AGENTS.md totals {total // 1000}k chars and rides in the"
            " system prompt of every model call; consider trimming it."
        )
    return tuple(out)


def _build_repo_map(tracked: tuple[str, ...]) -> str:
    """Return the compact `path/  (N files: a, b, ...)` directory map.

    The map is capped at `_REPO_MAP_MAX_LINES` rows plus one summary line past
    the cap, so it never dominates the system prompt.

    Args:
        tracked: The git-tracked paths, resolved once and shared with `file_count`.

    Returns:
        One row per directory, "" for an empty list.
    """
    if not tracked:
        return ""
    by_dir: dict[str, list[str]] = {}
    for rel in tracked:
        parent, _, name = rel.rpartition("/")
        key = parent or "."
        by_dir.setdefault(key, []).append(name)
    keys = sorted(by_dir.keys(), key=lambda k: (k != ".", k))
    rows: list[str] = []
    for idx, key in enumerate(keys):
        files = sorted(by_dir[key])
        shown = files[:_REPO_MAP_MAX_FILES_PER_DIR]
        suffix = (
            ""
            if len(files) <= _REPO_MAP_MAX_FILES_PER_DIR
            else f", +{len(files) - _REPO_MAP_MAX_FILES_PER_DIR} more"
        )
        rows.append(f"  {key}/  ({len(files)} files: {', '.join(shown)}{suffix})")
        if len(rows) >= _REPO_MAP_MAX_LINES:
            remaining = len(keys) - idx - 1
            if remaining > 0:
                rows.append(f"  ... ({remaining} more directories)")
            break
    return "\n".join(rows)


def load_repo_summary(root: pathlib.Path, hide: tuple[str, ...] = ()) -> kinds.RepoSummary:
    """Build the workspace's `RepoSummary`, shared by every mode.

    Outside a git repository (`agent6 ask` runs anywhere; run and plan refuse up
    front) the git-derived fields stay empty and the top-level listing is the
    model's starting point. There is no recursive walk substitute: an unbounded
    crawl of an arbitrary directory is what the tracked-files count avoids.

    Args:
        root: The workspace directory.
        hide: `[sandbox].hide_paths`, which no AGENTS.md is read from.

    Returns:
        The layout, AGENTS.md, recent commits and the repo map.
    """
    in_git = git_ops.is_git_repo(root)
    st = git_ops.status(root) if in_git else None
    top = tuple(
        sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in root.iterdir()
            if not p.name.startswith(".")
        )
    )
    # An unfiltered rglob would count .git/.venv/build junk and walk the tree every startup.
    tracked = git_ops.tracked_files(root) if in_git else ()
    return kinds.RepoSummary(
        root=root,
        branch=st.branch if st is not None else "",
        head_sha=st.head_sha if st is not None else "",
        file_count=len(tracked),
        top_level=top,
        agents_md=agents_md_text(root, hide),
        recent_log=git_ops.recent_log(root, n=20) if in_git else "",
        repo_map=_build_repo_map(tracked),
        is_git=in_git,
    )
