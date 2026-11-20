# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The config writers are crash-safe: they publish through atomic_write, never in place."""

from __future__ import annotations

import pathlib
import tomllib

import pytest

from agent6 import paths, portable
from agent6.config import io, write


def test_writers_go_through_atomic_write_and_never_truncate(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = paths.global_config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    original = '[sandbox]\nprotect_git = "on"\n'
    cfg.write_text(original, encoding="utf-8")

    def boom(_path: pathlib.Path, _text: str) -> None:
        raise RuntimeError("simulated crash during publish")

    # path.write_text would truncate cfg before any rename; atomic_write fails before it.
    monkeypatch.setattr(portable, "atomic_write", boom)
    with pytest.raises(RuntimeError):
        write.set_config_value(tmp_path, "sandbox.protect_git", "off")
    assert cfg.read_text(encoding="utf-8") == original  # not truncated


def test_write_leaves_no_temp_siblings(tmp_path: pathlib.Path) -> None:
    assert write.set_config_value(tmp_path, "sandbox.network", "auto") is None
    assert write.set_config_value(tmp_path, "sandbox.protect_git", "off") is None
    cfg = paths.global_config_path()
    assert 'network = "auto"' in cfg.read_text(encoding="utf-8")
    assert [p.name for p in cfg.parent.iterdir()] == ["config.toml"]  # tmp files cleaned up


def test_a_quoted_leaf_key_is_the_same_leaf(tmp_path: pathlib.Path) -> None:
    """`"protect_git" = true` is valid TOML naming the same leaf, and the surgery matches it."""
    text = '[sandbox]\n"protect_git" = true\nhome = "tmp"\n'

    edited = io.upsert_leaf_text(text, "sandbox.protect_git", False, path=tmp_path / "c.toml")

    assert tomllib.loads(edited) == {"sandbox": {"protect_git": False, "home": "tmp"}}
    assert edited.count("protect_git") == 1
