# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared helpers used by multiple workflows."""

from __future__ import annotations

from pathlib import Path

from agent6.git_ops import recent_log, status
from agent6.types import RepoSummary


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
    )
