# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The wheel bundles every doc the agent6_docs tool advertises.

The force-include list is the one source of what ships, so it is held to the advertised surface.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from agent6.tools.schema import Agent6DocsInput

_ROOT = Path(__file__).resolve().parents[2]


def _bundled() -> dict[str, str]:
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]


def test_every_bundled_doc_source_exists() -> None:
    for src, dest in _bundled().items():
        assert (_ROOT / src).is_file(), f"{src} is force-included but missing"
        assert dest.startswith("agent6/_docs/"), f"{src} bundles outside _docs: {dest}"


def test_the_advertised_docs_are_bundled() -> None:
    """Every doc name the tool description offers as an example ships."""
    names = {Path(dest).stem for dest in _bundled().values()}
    import re

    advertised = set(re.findall(r"[A-Z][A-Z-]+", Agent6DocsInput.TOOL_DESCRIPTION))
    advertised -= {"OWN", "USE", "CLI"}  # emphasis and prose, not doc names
    missing = advertised - names
    assert not missing, f"description advertises unbundled docs: {sorted(missing)}"
    assert {"USAGE", "STATE-MACHINES", "ACP", "INSTALLATION", "WEB"} <= names


def test_the_reader_serves_every_bundled_doc() -> None:
    """The reader's name list covers every bundled doc."""
    from agent6.tools._agent6_docs import AGENT6_DOC_FILES

    bundled = {Path(dest).name for dest in _bundled().values()}
    assert bundled == set(AGENT6_DOC_FILES), (
        f"bundle and reader disagree: only bundled {sorted(bundled - set(AGENT6_DOC_FILES))}, "
        f"only in the reader {sorted(set(AGENT6_DOC_FILES) - bundled)}"
    )


def test_every_operator_facing_nav_page_is_servable() -> None:
    """Every page in docs/mkdocs.yml's nav is a name the reader serves.

    index.md (the home page) and data-contracts.md (a generated wire-schema reference) are not
    names the reader carries.
    """
    from agent6.tools._agent6_docs import read_agent6_doc

    mkdocs = (_ROOT / "docs" / "mkdocs.yml").read_text(encoding="utf-8")
    nav_pages = re.findall(r"^\s*-\s+.+:\s*(\S+\.md)\s*$", mkdocs, re.M)
    assert nav_pages
    skip = {"index.md", "data-contracts.md"}
    for page in {p for p in nav_pages if p not in skip}:
        assert read_agent6_doc(Path(page).stem.upper()) is not None, f"{page} is not servable"
