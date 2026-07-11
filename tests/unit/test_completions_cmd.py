# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 completions` installs shell tab-completion idempotently."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.ui.cli.completions_cmd import cmd_completions, detect_shell


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / ".config" / "agent6"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("ZDOTDIR", raising=False)
    return tmp_path


def test_print_emits_the_registration(capsys: pytest.CaptureFixture[str], home: Path) -> None:
    for shell in ("bash", "zsh", "fish"):
        assert cmd_completions(shell, print_only=True) == 0
        out = capsys.readouterr().out
        assert "agent6" in out and out.strip()  # the shellcode names the executable


def test_bash_install_is_idempotent(capsys: pytest.CaptureFixture[str], home: Path) -> None:
    assert cmd_completions("bash", print_only=False) == 0
    script = home / ".config" / "agent6" / "completions.bash"
    rc = home / ".bashrc"
    assert "agent6" in script.read_text(encoding="utf-8")
    first = rc.read_text(encoding="utf-8")
    assert first.count(">>> agent6 completions >>>") == 1
    assert str(script) in first  # the guarded source line points at the script
    # Rerunning refreshes the script but never duplicates the rc block.
    assert cmd_completions("bash", print_only=False) == 0
    assert rc.read_text(encoding="utf-8") == first
    assert "activate now" in capsys.readouterr().out


def test_zsh_respects_zdotdir(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    zdot = home / "zdot"
    zdot.mkdir()
    monkeypatch.setenv("ZDOTDIR", str(zdot))
    assert cmd_completions("zsh", print_only=False) == 0
    assert ">>> agent6 completions >>>" in (zdot / ".zshrc").read_text(encoding="utf-8")
    capsys.readouterr()


def test_fish_writes_native_completions_file(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cmd_completions("fish", print_only=False) == 0
    target = home / ".config" / "fish" / "completions" / "agent6.fish"
    assert "agent6" in target.read_text(encoding="utf-8")
    assert "fish loads it automatically" in capsys.readouterr().out


def test_unknown_shell_is_a_clear_error(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SHELL", "/bin/tcsh")
    assert detect_shell() == "tcsh"
    assert cmd_completions("", print_only=False) == 2
    err = capsys.readouterr().err
    assert "tcsh" in err and "bash|zsh|fish" in err


def test_detects_shell_from_env(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    assert cmd_completions("", print_only=True) == 0
    assert "agent6" in capsys.readouterr().out
