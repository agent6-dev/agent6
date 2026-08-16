# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Staleness pin for the generated config page.

``docs/config.md`` is RENDERED from ``docs/config_template.md`` plus the config
model by ``docs/gen_config.py``: the key, the default and the
``Field(description=...)`` of every leaf. This regenerates it in-memory and
asserts the committed file matches, so a field added, renamed, removed or
re-defaulted without regenerating fails here. The fix is never to edit the
page: run ``uv run python docs/gen_config.py``.

Both directions hold by construction, which is what the two loose checks this
replaced could only approximate: an undocumented leaf cannot exist (every leaf
renders a row) and neither can a row for a key that does not."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[2]


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_config", _ROOT / "docs" / "gen_config.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_page_is_not_stale() -> None:
    gen = _load_generator()
    template = (_ROOT / "docs" / "config_template.md").read_text(encoding="utf-8")
    generated: str = gen.render(template)
    committed = (_ROOT / "docs" / "config.md").read_text(encoding="utf-8")
    assert generated == committed, (
        "docs/config.md is stale; regenerate it with: uv run python docs/gen_config.py"
    )


def test_every_leaf_has_a_description() -> None:
    """A leaf with no description renders an empty Meaning cell -- a row that
    says nothing is worse than a missing one, because the page looks complete."""
    gen = _load_generator()
    undescribed = sorted(path for path, (_, desc) in gen.leaves().items() if not desc.strip())
    assert not undescribed, f"config leaves with no Field(description=...): {undescribed}"


def test_every_leaf_reaches_the_page() -> None:
    """Every config leaf appears as a row in the rendered page.

    The staleness pin above compares the page against a re-render of the same
    template, so it cannot see a `<!-- config-table: ... -->` marker that stops
    matching: both sides then lose the same rows. That happened to
    `models.worker`, dropping ten documented fields off the page while every
    check stayed green. A leaf is documented or this fails.
    """
    gen = _load_generator()
    page = (_ROOT / "docs" / "config.md").read_text(encoding="utf-8")

    def documented(path: str) -> bool:
        parts = path.split(".")
        return any(f"`{'.'.join(parts[i:])}`" in page for i in range(len(parts)))

    missing = sorted(path for path in gen.leaves() if not documented(path))
    assert not missing, f"config leaves with no row on the page: {missing}"
