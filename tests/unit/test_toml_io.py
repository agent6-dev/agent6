# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.config.io scalar/table serialization + leaf surgery."""

from __future__ import annotations

import tomllib
from pathlib import Path

from agent6.config.io import (
    _toml_repr,  # pyright: ignore[reportPrivateUsage]
    parse_cli_value,  # pyright: ignore[reportPrivateUsage]
    remove_toml_table,
    upsert_toml_leaf,  # pyright: ignore[reportPrivateUsage]
)


def test_remove_toml_table_drops_header_body_and_subtables(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text(
        '[cli]\ninput = "bar"\n[cli.sub]\nx = 1\n[budget]\nmax_usd = 1.0\n', encoding="utf-8"
    )
    assert remove_toml_table(path, "cli") is True
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert "cli" not in data  # header, body, and [cli.sub] all gone
    assert data["budget"] == {"max_usd": 1.0}  # the sibling table is untouched


def test_remove_toml_table_absent_returns_false(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[budget]\nmax_usd = 1.0\n", encoding="utf-8")
    assert remove_toml_table(path, "cli") is False
    assert path.read_text(encoding="utf-8") == "[budget]\nmax_usd = 1.0\n"


def test_toml_repr_serializes_nested_dict_as_inline_table() -> None:
    # An OpenRouter routing value round-trips through the inline-table form.
    val = {"provider": {"sort": "throughput"}}
    rendered = _toml_repr(val)  # pyright: ignore[reportPrivateUsage]
    assert rendered == '{ provider = { sort = "throughput" } }'
    assert tomllib.loads(f"x = {rendered}")["x"] == val


def test_toml_repr_empty_dict() -> None:
    assert _toml_repr({}) == "{}"  # pyright: ignore[reportPrivateUsage]


def test_config_set_whole_extra_body_value_round_trips(tmp_path: Path) -> None:
    # The natural way to set a table-valued config (extra_body) is the whole
    # value at table granularity — the deep-leaf path would collide with the
    # inline parent. Setting the whole value must produce valid, re-parseable
    # TOML even when the section already has the key.
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[providers.openrouter]\napi_format = "openai"\n'
        'extra_body = { provider = { sort = "throughput" } }\n',
        encoding="utf-8",
    )
    value = parse_cli_value('{ provider = { sort = "latency" } }')  # pyright: ignore[reportPrivateUsage]
    upsert_toml_leaf(  # pyright: ignore[reportPrivateUsage]
        cfg, "providers.openrouter.extra_body", value
    )
    parsed = tomllib.loads(cfg.read_text(encoding="utf-8"))
    assert parsed["providers"]["openrouter"]["extra_body"] == {"provider": {"sort": "latency"}}
    # the sibling key survived the surgery
    assert parsed["providers"]["openrouter"]["api_format"] == "openai"
