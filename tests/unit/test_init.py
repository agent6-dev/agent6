# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 init`."""

from __future__ import annotations

from pathlib import Path

from agent6.init import init_workspace
from agent6.paths import repo_config_path


def test_init_creates_files_in_empty_dir(tmp_path: Path) -> None:
    rc = init_workspace(tmp_path, force=False)
    assert rc == 0
    # The per-repo config lives OUT of the workspace, under the state dir.
    assert repo_config_path(tmp_path).is_file()
    assert not (tmp_path / ".agent6").exists()
    assert (tmp_path / "AGENTS.md").is_file()
    gi = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    for entry in (".env", ".env.*", "secrets/", "*.pem", "*.key"):
        assert entry in gi


def test_init_py_verify_command_is_jail_compatible(tmp_path: Path) -> None:
    """The py default must run inside the jail: no `uv` (it lives under $HOME,
    invisible to the sandbox), and the scaffold must explain the constraint."""
    init_workspace(tmp_path, force=False, profile="py")
    cfg = repo_config_path(tmp_path).read_text(encoding="utf-8")
    assert 'verify_command = [".venv/bin/python", "-m", "pytest", "-x"]' in cfg
    assert '"uv"' not in cfg
    assert "INSIDE the sandbox" in cfg  # the jail-execution warning is present


def test_init_py_gitignores_build_artifacts(tmp_path: Path) -> None:
    """The py profile ignores bytecode so the verify run's __pycache__ is not
    swept into agent6's per-step commits."""
    init_workspace(tmp_path, force=False, profile="py")
    gi = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "__pycache__/" in gi
    assert "*.pyc" in gi


def test_init_profiles_gitignore_their_ecosystem(tmp_path: Path) -> None:
    rust = tmp_path / "r"
    rust.mkdir()
    init_workspace(rust, force=False, profile="rust")
    assert "target/" in (rust / ".gitignore").read_text(encoding="utf-8")
    node = tmp_path / "n"
    node.mkdir()
    init_workspace(node, force=False, profile="node")
    assert "node_modules/" in (node / ".gitignore").read_text(encoding="utf-8")


def test_init_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    cfgp = repo_config_path(tmp_path)
    cfgp.parent.mkdir(parents=True, exist_ok=True)
    cfgp.write_text("# mine\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# mine\n", encoding="utf-8")
    rc = init_workspace(tmp_path, force=False)
    assert rc == 0
    assert cfgp.read_text(encoding="utf-8") == "# mine\n"
    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8") == "# mine\n"
    assert cfgp.with_name("config.toml.suggested").is_file()
    assert (tmp_path / "AGENTS.md.suggested").is_file()


def test_init_force_overwrites(tmp_path: Path) -> None:
    cfgp = repo_config_path(tmp_path)
    cfgp.parent.mkdir(parents=True, exist_ok=True)
    cfgp.write_text("# mine\n", encoding="utf-8")
    rc = init_workspace(tmp_path, force=True)
    assert rc == 0
    assert "verify_command" in cfgp.read_text(encoding="utf-8")


def test_init_gitignore_is_idempotent(tmp_path: Path) -> None:
    init_workspace(tmp_path, force=False)
    first = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    # Drop the suggested files so the second init isn't a no-op for the config/AGENTS path.
    repo_config_path(tmp_path).with_name("config.toml.suggested").unlink(missing_ok=True)
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
    assert "secrets/" in gi
