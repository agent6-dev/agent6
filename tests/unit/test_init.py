# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the `agent6 init` setup wizard (granular + idempotent)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config_layer import load_effective, repo_config_path_for
from agent6.init import init_workspace


@pytest.fixture(autouse=True)
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the per-repo config (out of the workspace) inside tmp_path.
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))


def _repo(tmp_path: Path, name: str = "repo") -> Path:
    p = tmp_path / name
    p.mkdir()
    return p


def test_init_empty_dir_creates_scaffold(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    rc = init_workspace(repo)  # default is non-interactive: accept defaults
    assert rc == 0
    assert repo_config_path_for(repo).is_file()  # config lives OUT of the workspace
    assert not (repo / ".agent6").exists()
    assert (repo / "AGENTS.md").is_file()
    gi = (repo / ".gitignore").read_text(encoding="utf-8")
    for entry in (".env", "secrets/", "*.pem", "*.key"):
        assert entry in gi


def test_init_infers_verify_for_python_repo(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    init_workspace(repo)
    cfg = load_effective(repo).config
    # No .venv in this fresh repo -> python3 on PATH (the .venv/bin/python default
    # is only used when that interpreter actually exists; see verify_infer).
    assert cfg.workflow.verify_command == ("python3", "-m", "pytest", "-q")


def test_init_verify_from_agents_md(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "AGENTS.md").write_text("## Verify\n\n```bash\nmake test\n```\n", encoding="utf-8")
    init_workspace(repo)
    assert load_effective(repo).config.workflow.verify_command == ("make", "test")


def test_init_detects_ecosystem_for_gitignore(tmp_path: Path) -> None:
    py = _repo(tmp_path, "py")
    (py / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    init_workspace(py)
    assert "__pycache__/" in (py / ".gitignore").read_text(encoding="utf-8")

    rust = _repo(tmp_path, "rust")
    (rust / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    init_workspace(rust)
    assert "target/" in (rust / ".gitignore").read_text(encoding="utf-8")


def test_init_never_overwrites_or_writes_suggested(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    cfgp = repo_config_path_for(repo)
    cfgp.parent.mkdir(parents=True, exist_ok=True)
    cfgp.write_text('[workflow]\nverify_command = ["my-test"]\n', encoding="utf-8")
    (repo / "AGENTS.md").write_text("# mine\n", encoding="utf-8")

    init_workspace(repo)

    # Existing content untouched, NO .suggested siblings, verify not clobbered.
    assert (repo / "AGENTS.md").read_text(encoding="utf-8") == "# mine\n"
    assert not cfgp.with_name("config.toml.suggested").is_file()
    assert not (repo / "AGENTS.md.suggested").is_file()
    assert load_effective(repo).config.workflow.verify_command == ("my-test",)


def test_init_gitignore_is_idempotent(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    init_workspace(repo)
    first = (repo / ".gitignore").read_text(encoding="utf-8")
    init_workspace(repo)
    assert (repo / ".gitignore").read_text(encoding="utf-8") == first


def test_init_gitignore_preserves_existing(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text("# pre-existing\nmy-secret-file\n", encoding="utf-8")
    init_workspace(repo)
    gi = (repo / ".gitignore").read_text(encoding="utf-8")
    assert "# pre-existing" in gi and "my-secret-file" in gi and "secrets/" in gi
