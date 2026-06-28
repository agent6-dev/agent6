# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Reader for agent6's own bundled docs, backing the `agent6_docs` ask tool."""

from __future__ import annotations

from pathlib import Path

# Canonical (uppercase) doc names the tool exposes. The wheel bundles all five
# under these names in agent6/_docs/. In a dev checkout README/AGENTS sit at the
# repo root under these names too, while the reference docs live lowercase under
# docs/ (the site's convention) -- _locate() handles that case difference.
AGENT6_DOC_FILES = ("README.md", "CONFIG.md", "SECURITY.md", "AGENTS.md", "ARCHITECTURE.md")


def agent6_docs_dirs() -> list[Path]:
    base = Path(__file__).resolve()  # .../agent6/tools/_agent6_docs.py
    repo_root = base.parents[3]
    # Bundled wheel layout (everything under _docs/), then the dev checkout: the
    # repo root (README/AGENTS) and docs/ (the reference docs).
    return [base.parents[1] / "_docs", repo_root, repo_root / "docs"]


def _locate(fname: str) -> Path | None:
    """The on-disk path of a canonical doc, or None. Tries the exact name then
    its lowercase form, so the uppercase bundle name resolves to a lowercase
    docs/ source file in a dev checkout."""
    for d in agent6_docs_dirs():
        for cand in (fname, fname.lower()):
            p = d / cand
            if p.is_file():
                return p
    return None


def list_agent6_docs() -> list[str]:
    return [n[:-3] for n in AGENT6_DOC_FILES if _locate(n) is not None]


def read_agent6_doc(name: str) -> str | None:
    fname = name if name.endswith(".md") else f"{name}.md"
    if fname not in AGENT6_DOC_FILES:
        return None
    p = _locate(fname)
    return p.read_text(encoding="utf-8", errors="replace") if p else None
