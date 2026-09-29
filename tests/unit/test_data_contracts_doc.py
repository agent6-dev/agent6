# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Staleness pin for the generated data-contracts page.

`docs/data-contracts.md` is derived from the contract modules' docstrings by
`docs/gen_contracts.py`; this regenerates it in memory and asserts the committed file matches. The
fix is never to edit the page: run `uv run python docs/gen_contracts.py`.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_generator() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "gen_contracts", _ROOT / "docs" / "gen_contracts.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so the module's stringized dataclass annotations resolve.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_data_contracts_page_is_not_stale() -> None:
    generated: str = _load_generator().build_markdown()
    committed = (_ROOT / "docs" / "data-contracts.md").read_text(encoding="utf-8")
    assert generated == committed, (
        "docs/data-contracts.md is stale; regenerate it with: uv run python docs/gen_contracts.py"
    )
