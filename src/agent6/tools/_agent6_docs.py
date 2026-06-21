# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Reader for agent6's own bundled docs, backing the `agent6_docs` ask tool."""

from __future__ import annotations

from pathlib import Path

AGENT6_DOC_FILES = ("README.md", "CONFIG.md", "SECURITY.md", "AGENTS.md", "ARCHITECTURE.md")


def agent6_docs_dirs() -> list[Path]:
    base = Path(__file__).resolve()  # .../agent6/tools/dispatch.py
    return [base.parents[1] / "_docs", base.parents[3]]  # bundled, then dev repo-root


def list_agent6_docs() -> list[str]:
    for d in agent6_docs_dirs():
        if d.is_dir():
            found = [n[:-3] for n in AGENT6_DOC_FILES if (d / n).is_file()]
            if found:
                return found
    return []


def read_agent6_doc(name: str) -> str | None:
    fname = name if name.endswith(".md") else f"{name}.md"
    if fname not in AGENT6_DOC_FILES:
        return None
    for d in agent6_docs_dirs():
        p = d / fname
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
    return None
