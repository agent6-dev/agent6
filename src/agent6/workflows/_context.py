# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared helpers used by multiple workflows."""

from __future__ import annotations

from pathlib import Path

from agent6.git_ops import recent_log, status, tracked_files
from agent6.types import RepoSummary

_REPO_MAP_MAX_LINES = 60
_REPO_MAP_MAX_FILES_PER_DIR = 6


def _build_repo_map(root: Path) -> str:
    """Compact `path/  (N files: a, b, ...)` directory map from git ls-files.

    Returns an empty string outside a git repo or when ls-files fails.
    Output is hard-capped at ``_REPO_MAP_MAX_LINES`` rows so it never
    dominates the system prompt; directories beyond the cap are summarised
    as a single ``... (K more directories)`` line.
    """
    tracked = tracked_files(root)
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
    """Build a `RepoSummary` for the workspace rooted at ``root``.

    Used by both the implement workflow and plan-mode workflow so they see
    the same view of the repo (top-level layout, AGENTS.md, recent commits).
    """
    st = status(root)
    top = tuple(
        sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in root.iterdir()
            if not p.name.startswith(".")
        )
    )
    file_count = sum(1 for p in root.rglob("*") if p.is_file())
    agents_md_path = root / "AGENTS.md"
    agents_md = agents_md_path.read_text(encoding="utf-8") if agents_md_path.is_file() else ""
    return RepoSummary(
        root=root,
        branch=st.branch,
        head_sha=st.head_sha,
        file_count=file_count,
        top_level=top,
        agents_md=agents_md,
        recent_log=recent_log(root, n=20),
        repo_map=_build_repo_map(root),
    )
