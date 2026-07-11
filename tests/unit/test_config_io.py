# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config write surgery is crash-safe: writers publish through atomic_write
(tmp + rename), never truncating the live file in place."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import io


def test_writers_go_through_atomic_write_and_never_truncate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "config.toml"
    original = "[sandbox]\nprotect_git = true\n"
    cfg.write_text(original, encoding="utf-8")

    def boom(_path: Path, _text: str) -> None:
        raise RuntimeError("simulated crash during publish")

    # If a writer still called path.write_text, it would truncate cfg before any
    # rename and this patch would never fire; going through atomic_write means
    # the failure happens before the rename and the live file is untouched.
    monkeypatch.setattr(io, "atomic_write", boom)
    with pytest.raises(RuntimeError):
        io.upsert_toml_leaf(cfg, "sandbox.protect_git", False)
    assert cfg.read_text(encoding="utf-8") == original  # not truncated


def test_write_leaves_no_temp_siblings(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    io.upsert_toml_leaf(cfg, "sandbox.agent_network", "providers")
    io.upsert_toml_leaf(cfg, "sandbox.protect_git", False)
    assert 'agent_network = "providers"' in cfg.read_text(encoding="utf-8")
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]  # tmp files cleaned up
