# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Reader for agent6's own bundled docs, backing the `agent6_docs` ask tool."""

from __future__ import annotations

from pathlib import Path

# One uppercase name per doc the wheel bundles into agent6/_docs/ (pyproject's force-include list).
# A dev checkout keeps README and AGENTS at the repo root and the rest lowercase under docs/.
AGENT6_DOC_FILES = (
    "README.md",
    "AGENTS.md",
    "USAGE.md",
    "INSTALLATION.md",
    "CONFIG.md",
    "SECURITY.md",
    "ARCHITECTURE.md",
    "STATE-MACHINES.md",
    "ACP.md",
    "WEB.md",
    "TERMINAL.md",
)


def agent6_docs_dirs() -> list[Path]:
    """Return the directories searched for a doc: the wheel bundle, then the dev checkout."""
    base = Path(__file__).resolve()
    repo_root = base.parents[3]
    return [base.parents[1] / "_docs", repo_root, repo_root / "docs"]


def _locate(fname: str) -> Path | None:
    """Return the on-disk path of a canonical doc, trying the exact name then its lowercase form."""
    for d in agent6_docs_dirs():
        for cand in (fname, fname.lower()):
            p = d / cand
            if p.is_file():
                return p
    return None


def list_agent6_docs() -> list[str]:
    """Return the names (without `.md`) of the docs present on disk."""
    return [n[:-3] for n in AGENT6_DOC_FILES if _locate(n) is not None]


def read_agent6_doc(name: str) -> str | None:
    """Return a doc's text, or None for an unknown name or a doc missing on disk."""
    fname = name if name.endswith(".md") else f"{name}.md"
    if fname not in AGENT6_DOC_FILES:
        return None
    p = _locate(fname)
    return p.read_text(encoding="utf-8", errors="replace") if p else None
