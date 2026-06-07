# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 init`."""

from __future__ import annotations

from pathlib import Path

from agent6.init import init_workspace


def test_init_creates_files_in_empty_dir(tmp_path: Path) -> None:
    rc = init_workspace(tmp_path, force=False)
    assert rc == 0
    assert (tmp_path / ".agent6" / "config.toml").is_file()
    assert (tmp_path / "AGENTS.md").is_file()
    gi = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    for entry in (".env", ".env.*", "secrets/", "*.pem", "*.key", ".agent6/"):
        assert entry in gi


def test_init_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    (tmp_path / ".agent6").mkdir()
    (tmp_path / ".agent6" / "config.toml").write_text("# mine\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# mine\n", encoding="utf-8")
    rc = init_workspace(tmp_path, force=False)
    assert rc == 0
    assert (tmp_path / ".agent6" / "config.toml").read_text(encoding="utf-8") == "# mine\n"
    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8") == "# mine\n"
    assert (tmp_path / ".agent6" / "config.toml.suggested").is_file()
    assert (tmp_path / "AGENTS.md.suggested").is_file()


def test_init_force_overwrites(tmp_path: Path) -> None:
    (tmp_path / ".agent6").mkdir()
    (tmp_path / ".agent6" / "config.toml").write_text("# mine\n", encoding="utf-8")
    rc = init_workspace(tmp_path, force=True)
    assert rc == 0
    assert "verify_command" in (tmp_path / ".agent6" / "config.toml").read_text(encoding="utf-8")


def test_init_gitignore_is_idempotent(tmp_path: Path) -> None:
    init_workspace(tmp_path, force=False)
    first = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    # Drop the suggested files so the second init isn't a no-op for the config/AGENTS path.
    (tmp_path / ".agent6" / "config.toml.suggested").unlink(missing_ok=True)
    (tmp_path / "AGENTS.md.suggested").unlink(missing_ok=True)
    init_workspace(tmp_path, force=False)
    second = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert first == second


def test_init_gitignore_preserves_existing(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("# pre-existing\nmy-secret-file\n", encoding="utf-8")
    init_workspace(tmp_path, force=False)
    gi = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "# pre-existing" in gi
    assert "my-secret-file" in gi
    assert ".agent6/" in gi
