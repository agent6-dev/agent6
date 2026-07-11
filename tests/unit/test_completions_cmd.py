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
    # Point the process-tree walk at an empty dir so detection falls back to
    # $SHELL deterministically (the real tree ends in whatever shell runs pytest).
    monkeypatch.setattr("agent6.ui.cli.completions_cmd._PROC", tmp_path / "no-proc")
    return tmp_path


def test_print_emits_the_registration(capsys: pytest.CaptureFixture[str], home: Path) -> None:
    for shell in ("bash", "zsh", "fish", "xonsh"):
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
    out = capsys.readouterr().out
    assert "fish loads it automatically" in out
    assert "activate now" not in out  # fish needs no activation step


def test_unknown_shell_is_a_clear_error(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SHELL", "/bin/tcsh")
    assert detect_shell() == "tcsh"
    assert cmd_completions(None, print_only=False) == 2
    err = capsys.readouterr().err
    assert "tcsh" in err and "bash|zsh|fish|xonsh" in err


def test_detects_shell_from_env(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    assert cmd_completions(None, print_only=True) == 0
    assert "agent6" in capsys.readouterr().out


def test_xonsh_writes_autoloaded_completer(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cmd_completions("xonsh", print_only=False) == 0
    target = home / ".config" / "xonsh" / "rc.d" / "agent6.xsh"
    code = target.read_text(encoding="utf-8")
    # The completer drives the argcomplete protocol against the live agent6,
    # so the file must parse as Python and set the protocol request.
    import ast

    ast.parse(code)
    assert "_ARGCOMPLETE_STDOUT_FILENAME" in code
    assert "COMP_LINE" in code
    assert 'add_one_completer("agent6"' in code
    # Candidates with shell-hostile characters are quoted before insertion,
    # and a missing/hung agent6 yields no candidates instead of a traceback.
    assert "shlex.quote" in code
    assert "TimeoutExpired" in code
    out = capsys.readouterr().out
    assert "xonsh loads it automatically" in out
    assert "activate now" not in out  # rc.d needs no activation step


def test_xonsh_detected_in_process_walk(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proc = home / "proc"
    for pid, comm, ppid in ((50, "uv", 40), (40, "xonsh", 30), (30, "bash", 1)):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "comm").write_text(comm + "\n", encoding="utf-8")
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 0 0 0", encoding="utf-8")
    monkeypatch.setattr("agent6.ui.cli.completions_cmd._PROC", proc)
    monkeypatch.setattr("os.getppid", lambda: 50)
    monkeypatch.setenv("SHELL", "/bin/bash")
    assert detect_shell() == "xonsh"


def test_detects_shell_from_process_tree(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """$SHELL is the login shell, not the running one (a fish started from
    bash keeps $SHELL=bash). The walk returns the nearest shell ancestor,
    skipping non-shell wrappers like uv."""
    proc = home / "proc"
    # agent6 <- uv(50) <- fish(40) <- bash(30) <- init
    for pid, comm, ppid in ((50, "uv", 40), (40, "fish", 30), (30, "bash", 1)):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "comm").write_text(comm + "\n", encoding="utf-8")
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 0 0 0", encoding="utf-8")
    monkeypatch.setattr("agent6.ui.cli.completions_cmd._PROC", proc)
    monkeypatch.setattr("os.getppid", lambda: 50)
    monkeypatch.setenv("SHELL", "/bin/bash")
    assert detect_shell() == "fish"
