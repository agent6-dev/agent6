# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared helpers used by multiple workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agent6.budget import BudgetExceeded
from agent6.git_ops import co_change_pairs, is_git_repo, recent_log, status, tracked_files
from agent6.types import CoChangePair, HotSymbol, RepoSummary
from agent6.workflows._symbol_outline import build_symbol_outline_block

if TYPE_CHECKING:
    from agent6.tools.dispatch import ToolDispatcher

_REPO_MAP_MAX_LINES = 60
_REPO_MAP_MAX_FILES_PER_DIR = 6
# Bound the AGENTS.md injected into every turn's prompt. Generous enough to
# carry a normal conventions file whole; truncates a pathological 50KB one so it
# can't dominate the prefix. The model is pointed at read_file for the rest.
_AGENTS_MD_MAX_CHARS = 16000


def _build_repo_map(tracked: tuple[str, ...]) -> str:
    """Compact `path/  (N files: a, b, ...)` directory map from git ls-files.

    Takes the already-resolved tracked-file list (shared with ``file_count`` so
    git ls-files runs once). Returns an empty string for an empty list. Output is
    hard-capped at ``_REPO_MAP_MAX_LINES`` rows so it never dominates the system
    prompt; directories beyond the cap are summarised as a single
    ``... (K more directories)`` line.
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


def load_repo_summary(root: Path, *, dispatcher: ToolDispatcher | None = None) -> RepoSummary:
    """Build a `RepoSummary` for the workspace rooted at ``root``.

    Base view (layout, AGENTS.md, recent commits, repo map) is shared by the
    implement and plan-mode workflows. When *dispatcher* is given (the run loop,
    and ``agent6 prompt show``), ALSO enrich with structural priors: hot symbols
    (cross-file reference hot spots), git co-change pairs, and the tree-sitter
    symbol outline. Enrichment is best-effort -- a parser or git-history hiccup
    must not block the run -- but BudgetExceeded / KeyboardInterrupt propagate so
    the loop's budget guarantee and abort path stay intact.

    Outside a git repository (``agent6 ask`` runs anywhere; run/plan refuse up
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
    # Count git-tracked files, not an unfiltered rglob: the old walk counted
    # .git/.venv/build junk (a misleading number to the model) and traversed the
    # whole tree every startup. tracked is reused by _build_repo_map below.
    tracked = tracked_files(root) if in_git else ()
    file_count = len(tracked)
    agents_md_path = root / "AGENTS.md"
    agents_md = agents_md_path.read_text(encoding="utf-8") if agents_md_path.is_file() else ""
    if len(agents_md) > _AGENTS_MD_MAX_CHARS:
        agents_md = (
            agents_md[:_AGENTS_MD_MAX_CHARS]
            + "\n... (AGENTS.md truncated here; use read_file for the full text)\n"
        )
    hot: tuple[HotSymbol, ...] = ()
    co_change: tuple[CoChangePair, ...] = ()
    symbol_outline = ""
    if dispatcher is not None:
        try:
            hot = tuple(
                HotSymbol(*t)
                for t in dispatcher.hot_symbols(max_symbols=20, min_files_referenced=2)
            )
        except (BudgetExceeded, KeyboardInterrupt):
            raise
        except Exception:
            hot = ()
        if in_git:
            try:
                co_change = tuple(CoChangePair(*t) for t in co_change_pairs(root, n_commits=200))
            except (BudgetExceeded, KeyboardInterrupt):
                raise
            except Exception:
                co_change = ()
        try:
            symbol_outline = build_symbol_outline_block(dispatcher.file_outlines(), root=root)
        except (BudgetExceeded, KeyboardInterrupt):
            raise
        except Exception:
            symbol_outline = ""
    return RepoSummary(
        root=root,
        branch=st.branch if st is not None else "",
        head_sha=st.head_sha if st is not None else "",
        file_count=file_count,
        top_level=top,
        agents_md=agents_md,
        recent_log=recent_log(root, n=20) if in_git else "",
        repo_map=_build_repo_map(tracked),
        co_change_pairs=co_change,
        hot_symbols=hot,
        symbol_outline=symbol_outline,
        is_git=in_git,
    )
