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
import re
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


def test_every_contract_has_one_support_level() -> None:
    """Every contract card is classified once in the support policy."""
    generator = _load_generator()
    generated: str = generator.build_markdown()
    assert "## Support levels\n\n" in generated
    support = generated.split("## Support levels\n\n", 1)[1].split("\n## ", 1)[0]
    assert re.findall(r"^- \*\*([^*]+):\*\*", support, re.MULTILINE) == [
        "Public",
        "Durable but private",
        "Disposable",
    ]
    classified = re.findall(r"\[([^]]+)\]\(#[^)]+\)", support)
    assert sorted(classified) == sorted(contract.title for contract in generator.CONTRACTS)


def test_public_cards_link_to_formats_without_field_tables() -> None:
    """The developer map links public fields instead of copying their reference tables."""
    page = _load_generator().build_markdown()
    public_cards = [card for card in page.split("\n## ") if "[Formats](formats.md#" in card]
    assert len(public_cards) >= 9
    assert all("| field | type | default |" not in card for card in public_cards)


def test_function_only_module_has_no_class_kind() -> None:
    """A function-only wire builder must not be described as a plain class."""
    generator = _load_generator()
    facts = generator._module_facts("agent6.viewmodel.session.view")
    assert not facts.class_fields
    assert "session_state_as_dict" in facts.class_docs
    assert generator._kind(facts, "session_state_as_dict") == ""


def test_function_only_card_has_no_kind_separator() -> None:
    """A function-only card's module link has no empty kind or trailing whitespace."""
    generated: str = _load_generator().build_markdown()
    link = next(
        line
        for line in generated.splitlines()
        if line.startswith("[`agent6.viewmodel.session.view`]")
    )
    assert link == (
        "[`agent6.viewmodel.session.view`](https://github.com/agent6-dev/agent6/"
        "blob/master/src/agent6/viewmodel/session/view.py)"
    )
    assert all(line == line.rstrip() for line in generated.splitlines())
