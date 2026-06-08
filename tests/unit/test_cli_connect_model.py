# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 connect` / `agent6 model` / `agent6 config` CLI flows."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from agent6 import secrets
from agent6.cli import main


@pytest.fixture
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.chdir(tmp_path)
    return tmp_path / "g"


def test_connect_stores_key_and_provider_and_never_execs(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("agent6.cli.connect.getpass.getpass", lambda prompt="": "sk-ant-FAKE")
    # Security: connect must NEVER run a subprocess (no remote-supplied command).
    calls: list[object] = []

    def _record_run(*args: object, **kwargs: object) -> None:
        calls.append(args)

    monkeypatch.setattr("subprocess.run", _record_run)

    rc = main(["connect", "--provider", "anthropic"])
    assert rc == 0

    sp = tmp_path / "g" / "secrets.toml"
    assert sp.is_file()
    assert stat.S_IMODE(sp.stat().st_mode) == 0o600
    assert secrets.resolve_api_key("anthropic", None) == "sk-ant-FAKE"

    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    assert "[providers.anthropic]" in gc
    assert calls == []


def test_connect_prints_post_entry_key_summary(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Simulate Python < 3.14 getpass (no echo_char): the helper must print a
    # length + last-four summary so the operator can tell the paste landed.
    def _fake_getpass(prompt: str = "", **kwargs: object) -> str:
        if "echo_char" in kwargs:
            raise TypeError("echo_char unsupported")
        return "sk-ant-0123456789wxyz"

    monkeypatch.setattr("agent6.cli.connect.getpass.getpass", _fake_getpass)
    rc = main(["connect", "--provider", "anthropic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Captured key: 21 chars, ending …wxyz" in out
    # The key itself is never echoed in full.
    assert "sk-ant-0123456789wxyz" not in out


def test_connect_short_key_summary_omits_tail(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _fake_getpass(prompt: str = "", **kwargs: object) -> str:
        if "echo_char" in kwargs:
            raise TypeError("echo_char unsupported")
        return "short"

    monkeypatch.setattr("agent6.cli.connect.getpass.getpass", _fake_getpass)
    rc = main(["connect", "--provider", "anthropic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Captured key: 5 chars." in out
    assert "ending" not in out


def test_connect_masked_echo_skips_summary(
    iso: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Simulate Python 3.14+ getpass that accepts echo_char: no post-entry
    # summary is printed because the keystrokes were already masked live.
    def _fake_getpass(prompt: str = "", **kwargs: object) -> str:
        return "sk-ant-0123456789wxyz"

    monkeypatch.setattr("agent6.cli.connect.getpass.getpass", _fake_getpass)
    rc = main(["connect", "--provider", "anthropic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Captured key:" not in out


def test_connect_local_endpoint_no_key(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("agent6.cli.connect.getpass.getpass", lambda prompt="": "")
    monkeypatch.setattr("builtins.input", lambda prompt="": "")  # accept default base_url
    rc = main(["connect", "--provider", "ollama"])
    assert rc == 0
    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    assert "[providers.ollama]" in gc
    # No key entered -> no secrets file required.
    assert not (tmp_path / "g" / "secrets.toml").is_file()


def test_model_set_and_show(iso: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["model", "worker", "anthropic", "claude-x", "--thinking", "medium"])
    assert rc == 0
    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    assert "[models.worker]" in gc
    assert "claude-x" in gc
    assert "medium" in gc

    capsys.readouterr()
    rc = main(["model"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "worker" in out
    assert "claude-x" in out


def test_model_rejects_unknown_role(iso: Path) -> None:
    # argparse `choices` validates the role positional (and feeds argcomplete).
    with pytest.raises(SystemExit) as exc:
        main(["model", "bogus", "anthropic", "claude-x"])
    assert exc.value.code == 2


def test_model_aborts_without_provider(iso: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Role given but provider omitted and none connected: the prompt gets an
    # empty answer and the command refuses rather than writing a bad config.
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert main(["model", "worker"]) == 2


def test_model_interactive_prefill(
    iso: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A provider is connected; the model list is served live (mocked). The
    # operator picks the provider by default and the model by number.
    (tmp_path / "g").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "config.toml").write_text(
        '[providers.anthropic]\nkind = "anthropic"\n', encoding="utf-8"
    )

    def fake_list_models(*a: object, **k: object) -> list[str]:
        return ["claude-a", "claude-b"]

    monkeypatch.setattr("agent6.cli.model.list_models", fake_list_models)
    answers = iter(["", "2"])  # provider default, then model #2
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    rc = main(["model", "worker"])
    assert rc == 0
    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    assert "[models.worker]" in gc
    assert "claude-b" in gc


def test_model_repo_scope_writes_repo(iso: Path, tmp_path: Path) -> None:
    rc = main(["model", "reviewer", "anthropic", "claude-o", "--repo"])
    assert rc == 0
    repo_cfg = (tmp_path / ".agent6" / "config.toml").read_text(encoding="utf-8")
    assert "[models.reviewer]" in repo_cfg


def test_config_fill_writes_global(iso: Path, tmp_path: Path) -> None:
    rc = main(["config", "fill"])
    assert rc == 0
    gc = (tmp_path / "g" / "config.toml").read_text(encoding="utf-8")
    assert "[sandbox]" in gc
    assert "[budget]" in gc


def test_config_show_runs(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert "[sandbox]" in out
    assert "source:" in out


def test_config_path_runs(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "path"]) == 0
    out = capsys.readouterr().out
    assert "global config" in out
    assert "secrets" in out
