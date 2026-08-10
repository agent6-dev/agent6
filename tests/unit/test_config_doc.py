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
