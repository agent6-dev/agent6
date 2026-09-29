# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Repo context for the system prompt: AGENTS.md discovery and the repo
summary (the header line, the top-level listing, the file map, recent
commits)."""

from __future__ import annotations

from pathlib import Path

from agent6.git_ops import is_git_repo, recent_log, status, toplevel, tracked_files
from agent6.types import RepoSummary

_REPO_MAP_MAX_LINES = 60
_REPO_MAP_MAX_FILES_PER_DIR = 6
# AGENTS.md is injected whole (pi and Claude Code both do); past this size the
# operator is warned at session start instead of the text being clipped.
AGENTS_MD_WARN_CHARS = 40_000


def _read_text(path: Path) -> str:
    """Tolerant read: a Windows-1252 byte or a permission-denied file must
    degrade, not crash the run AFTER session.start with no session.end (a dead
    run that listed as running)."""
    try:
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    except OSError:
        return ""


def _agents_md_sources(root: Path) -> tuple[tuple[Path, str], ...]:
    """Readable AGENTS.md files from the git root through *root*, in order."""
    top = toplevel(root)
    if top is None:
        candidates = (root / "AGENTS.md",)
    else:
        top = top.resolve()
        current = root.resolve()
        try:
            relative = current.relative_to(top)
        except ValueError:
            candidates = (root / "AGENTS.md",)
        else:
            directories = [top]
            for part in relative.parts:
                directories.append(directories[-1] / part)
            candidates = tuple(directory / "AGENTS.md" for directory in directories)
    return tuple((path, text) for path in candidates if (text := _read_text(path)))


def agents_md_text(root: Path) -> str:
    """The AGENTS.md text a session at *root* injects, whole (never clipped).

    When *root* sits below a git toplevel, every ancestor file loads from the
    toplevel down; a file below the toplevel carries a heading naming its
    directory, so a subdirectory start gets the same layered conventions as pi
    and Claude Code."""
    top = toplevel(root)
    parts: list[str] = []
    for path, text in _agents_md_sources(root):
        directory = path.parent.resolve()
        if top is None or directory == top.resolve():
            parts.append(text)
            continue
        label = f"{directory.relative_to(top.resolve()).as_posix()}/"
        if directory == root.resolve():
            label += " (this run's working directory)"
        parts.append(f"# AGENTS.md in {label}\n\n{text}")
    return "\n\n".join(parts)


def agents_md_notices(root: Path) -> tuple[str, ...]:
    """Session-start operator lines about the injected AGENTS.md: which files
    load when starting from a subdirectory, and an oversize warning (the text
    is injected whole; the remedy is trimming the file)."""
    out: list[str] = []
    top = toplevel(root)
    sources = _agents_md_sources(root)
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
    text = agents_md_text(root)
    total = len(text)
    if total > AGENTS_MD_WARN_CHARS:
        out.append(
            f"WARNING: AGENTS.md totals {total // 1000}k chars and rides in the"
            " system prompt of every model call; consider trimming it."
        )
    return tuple(out)


def _build_repo_map(tracked: tuple[str, ...]) -> str:
    """Compact `path/  (N files: a, b, ...)` directory map from git ls-files.

    Takes the already-resolved tracked-file list (shared with `file_count` so
    git ls-files runs once). Returns an empty string for an empty list. Output is
    capped at `_REPO_MAP_MAX_LINES` rows (plus one ``... (K more
    directories)`` summary line past the cap) so it never dominates the
    system prompt.
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


def load_repo_summary(root: Path) -> RepoSummary:
    """Build a `RepoSummary` for the workspace rooted at `root`: the layout,
    AGENTS.md, recent commits and the repo map, shared by every mode.

    Outside a git repository (`agent6 ask` runs anywhere; run/plan refuse up
    front) the git-derived fields stay empty: the top-level listing is the
    model's starting point and it lists/reads deeper on demand. No recursive
    walk substitute: an unbounded crawl of an arbitrary directory (say $HOME)
    is exactly what the tracked-files count exists to avoid.
    """
    in_git = is_git_repo(root)
    st = status(root) if in_git else None
    top = tuple(
        sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in root.iterdir()
            if not p.name.startswith(".")
        )
    )
    # Count git-tracked files: an unfiltered rglob would count .git/.venv/build
    # junk (a misleading number to the model) and traverse the whole tree every
    # startup.
    tracked = tracked_files(root) if in_git else ()
    return RepoSummary(
        root=root,
        branch=st.branch if st is not None else "",
        head_sha=st.head_sha if st is not None else "",
        file_count=len(tracked),
        top_level=top,
        agents_md=agents_md_text(root),
        recent_log=recent_log(root, n=20) if in_git else "",
        repo_map=_build_repo_map(tracked),
        is_git=in_git,
    )
