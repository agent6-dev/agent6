# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.config.io scalar/table serialization + leaf surgery."""

from __future__ import annotations

import tomllib
from pathlib import Path

from agent6.config.io import (
    format_toml_value,
    parse_cli_value,  # pyright: ignore[reportPrivateUsage]
    remove_toml_leaf,
    remove_toml_table,
    upsert_toml_leaf,  # pyright: ignore[reportPrivateUsage]
)


def test_remove_toml_leaf_deletes_whole_multiline_array(tmp_path: Path) -> None:
    """A multi-line array value must be removed whole. Deleting only the opening
    `leaf = [` line orphaned the continuation lines, leaving unparseable TOML
    (and `config fix` then reported the file it 'repaired' as invalid)."""
    path = tmp_path / "c.toml"
    path.write_text(
        '[sandbox]\nallow_urls = [\n  "http://x",\n  "http://y",\n]\ntool_network = "block"\n'
    )
    assert remove_toml_leaf(path, "sandbox.allow_urls") is True
    out = path.read_text()
    tomllib.loads(out)  # must stay valid TOML
    assert "allow_urls" not in out
    assert 'tool_network = "block"' in out  # sibling + header preserved


def test_remove_toml_leaf_deletes_whole_multiline_string(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('[a]\nx = """\nmultiline\nstring\n"""\ny = 1\n')
    assert remove_toml_leaf(path, "a.x") is True
    out = path.read_text()
    assert tomllib.loads(out) == {"a": {"y": 1}}


def test_remove_toml_leaf_multiline_last_leaf_drops_header(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[sandbox]\nallow_urls = [\n  1,\n]\n")
    assert remove_toml_leaf(path, "sandbox.allow_urls") is True
    out = path.read_text()
    tomllib.loads(out)
    assert "[sandbox]" not in out  # empty section header dropped


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


def test_format_toml_value_round_trips_through_parse_cli_value() -> None:
    # The serializer is the exact inverse of parse_cli_value: what an editor
    # prefills from it must save back unchanged (the TUI edit box relies on
    # this for list/dict fields).
    assert parse_cli_value(format_toml_value(("uv", "run", "pytest"))) == ["uv", "run", "pytest"]
    assert parse_cli_value(format_toml_value([])) == []


def test_toml_repr_serializes_nested_dict_as_inline_table() -> None:
    # An OpenRouter routing value round-trips through the inline-table form.
    val = {"provider": {"sort": "throughput"}}
    rendered = format_toml_value(val)
    assert rendered == '{ provider = { sort = "throughput" } }'
    assert tomllib.loads(f"x = {rendered}")["x"] == val


def test_toml_repr_empty_dict() -> None:
    assert format_toml_value({}) == "{}"


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
