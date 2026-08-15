# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The wheel bundles every doc the agent6_docs tool advertises.

The tool description promised machines/CLI/budget answers while the bundle
held only five reference docs; live reads of GETTING-STARTED and
STATE-MACHINES returned missing. The force-include list is the one source of
what ships, so it is held to the advertised surface here."""

from __future__ import annotations

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
    """Every doc name the tool description offers as an example must ship;
    the usage promise (machines, the CLI, budgets) rides on GETTING-STARTED
    and STATE-MACHINES being present."""
    names = {Path(dest).stem for dest in _bundled().values()}
    import re

    advertised = set(re.findall(r"[A-Z][A-Z-]+", Agent6DocsInput.TOOL_DESCRIPTION))
    advertised -= {"OWN", "USE", "CLI"}  # emphasis and prose, not doc names
    missing = advertised - names
    assert not missing, f"description advertises unbundled docs: {sorted(missing)}"
    assert {"GETTING-STARTED", "STATE-MACHINES", "ACP", "INSTALLATION", "WEB"} <= names
