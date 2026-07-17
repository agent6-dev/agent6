# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Staleness pin for the generated data-contracts page.

``docs/data-contracts.md`` is DERIVED from the contract modules' docstrings and
the source tree by ``docs/gen_contracts.py``; a hand edit or a docstring change
that shifts a card leaves it stale. This regenerates the markdown in-memory and
asserts the committed file matches, same shape as
``tests/security/test_subprocess_allowlist.py``. The fix is never to edit the
page: run ``uv run python docs/gen_contracts.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[2]


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "gen_contracts", _ROOT / "docs" / "gen_contracts.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's `from __future__ import annotations`
    # dataclasses can resolve their own stringized annotations via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_data_contracts_page_is_not_stale() -> None:
    generated: str = _load_generator().build_markdown()
    committed = (_ROOT / "docs" / "data-contracts.md").read_text(encoding="utf-8")
    assert generated == committed, (
        "docs/data-contracts.md is stale; regenerate it with: uv run python docs/gen_contracts.py"
    )
