# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config write surgery is crash-safe: writers publish through atomic_write, never in place."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import portable
from agent6.config import io


def test_writers_go_through_atomic_write_and_never_truncate(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "config.toml"
    original = "[sandbox]\nprotect_git = true\n"
    cfg.write_text(original, encoding="utf-8")

    def boom(_path: pathlib.Path, _text: str) -> None:
        raise RuntimeError("simulated crash during publish")

    # path.write_text would truncate cfg before any rename; atomic_write fails before it.
    monkeypatch.setattr(portable, "atomic_write", boom)
    with pytest.raises(RuntimeError):
        io.upsert_toml_leaf(cfg, "sandbox.protect_git", False)
    assert cfg.read_text(encoding="utf-8") == original  # not truncated


def test_write_leaves_no_temp_siblings(tmp_path: pathlib.Path) -> None:
    cfg = tmp_path / "config.toml"
    io.upsert_toml_leaf(cfg, "sandbox.network", "auto")
    io.upsert_toml_leaf(cfg, "sandbox.protect_git", False)
    assert 'network = "auto"' in cfg.read_text(encoding="utf-8")
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]  # tmp files cleaned up


def test_a_quoted_leaf_key_is_the_same_leaf(tmp_path: pathlib.Path) -> None:
    """`"protect_git" = true` is valid TOML naming the same leaf, and the surgery matches it."""
    path = tmp_path / "config.toml"
    path.write_text('[sandbox]\n"protect_git" = true\nhome = "tmp"\n', encoding="utf-8")

    io.upsert_toml_leaf(path, "sandbox.protect_git", False)

    assert io.read_toml_file(path) == {"sandbox": {"protect_git": False, "home": "tmp"}}
    assert path.read_text(encoding="utf-8").count("protect_git") == 1
