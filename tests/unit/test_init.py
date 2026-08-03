# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the `agent6 init` setup wizard (granular + idempotent)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config.layer import load_effective, repo_config_path_for
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


def test_cmd_init_reports_invalid_config_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pre-existing INVALID (but TOML-parseable) config makes `agent6 init` exit
    2 with a clean ERROR plus init's own repair pointer, not a crash through the
    generic "unexpected ... please report this" handler. init loads the effective
    config to infer a verify command; it is the user's setup to fix, and init is
    the repair command."""
    from agent6.ui.cli import cli_main

    repo = _repo(tmp_path)
    # Valid global with a configured provider, so the cross-field validator has a
    # non-empty "known providers" set to reject the typo against. AGENT6_CONFIG_HOME
    # (set by the isolated_state fixture) points at the agent6 dir itself, so the
    # global config is <cfg>/config.toml.
    global_cfg = tmp_path / "cfg" / "config.toml"
    global_cfg.parent.mkdir(parents=True, exist_ok=True)
    global_cfg.write_text(
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n',
        encoding="utf-8",
    )
    # Invalid per-repo config: worker references a provider that does not exist.
    cfgp = repo_config_path_for(repo)
    cfgp.parent.mkdir(parents=True, exist_ok=True)
    cfgp.write_text('[models.worker]\nprovider = "typoprovider"\nmodel = "x/y"\n', encoding="utf-8")

    monkeypatch.chdir(repo)
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    rc = cli_main(["init", "--yes"])
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ")
    assert "typoprovider" in err  # the precise validation reason is surfaced
    assert "Fix or delete" in err  # init's own repair pointer survives
    assert "report this" not in err


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


def test_init_never_rewrites_an_agents_md_it_cannot_decode(tmp_path: Path) -> None:
    """The append read the file with errors="replace" and wrote that back, so
    every non-ASCII byte in a non-UTF-8 AGENTS.md became U+FFFD, silently."""
    from agent6.init import _setup_agents_md  # pyright: ignore[reportPrivateUsage]

    p = tmp_path / "AGENTS.md"
    original = "# Notes pour l'\xe9quipe\n".encode("latin-1")
    p.write_bytes(original)
    _setup_agents_md(tmp_path, ecosystem="python", ask=lambda _p, _d: True)
    assert p.read_bytes() == original


def test_init_still_appends_to_a_utf8_agents_md(tmp_path: Path) -> None:
    """The converse: a legitimate file keeps its accents and gains the section."""
    from agent6.init import _setup_agents_md  # pyright: ignore[reportPrivateUsage]

    p = tmp_path / "AGENTS.md"
    p.write_text("# Team notes\n\nCafé.\n", encoding="utf-8")
    _setup_agents_md(tmp_path, ecosystem="python", ask=lambda _p, _d: True)
    text = p.read_text(encoding="utf-8")
    assert "Verify command" in text
    assert "Café" in text
