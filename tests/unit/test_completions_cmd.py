# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 completions` installs shell tab-completion idempotently."""

from __future__ import annotations

import pathlib

import pytest

from agent6.ui.cli import completions_cmd


@pytest.fixture
def home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.delenv("ZDOTDIR", raising=False)
    # The process-tree walk points at an empty dir, so detection falls back to $SHELL.
    monkeypatch.setattr("agent6.ui.cli.completions_cmd._PROC", tmp_path / "no-proc")
    return tmp_path


def test_print_emits_the_registration(
    capsys: pytest.CaptureFixture[str], home: pathlib.Path
) -> None:
    for shell in ("bash", "zsh", "fish", "xonsh"):
        assert completions_cmd.cmd_completions(shell, print_only=True) == 0
        out = capsys.readouterr().out
        assert "agent6" in out and out.strip()  # the shellcode names the executable


def test_bash_install_is_idempotent(capsys: pytest.CaptureFixture[str], home: pathlib.Path) -> None:
    assert completions_cmd.cmd_completions("bash", print_only=False) == 0
    script = home / ".config" / "agent6" / "completions.bash"
    rc = home / ".bashrc"
    assert "agent6" in script.read_text(encoding="utf-8")
    first = rc.read_text(encoding="utf-8")
    assert first.count(">>> agent6 completions >>>") == 1
    assert str(script) in first  # the guarded source line points at the script
    # Rerunning refreshes the script but never duplicates the rc block.
    assert completions_cmd.cmd_completions("bash", print_only=False) == 0
    assert rc.read_text(encoding="utf-8") == first
    assert "activate now" in capsys.readouterr().out


def test_bash_block_does_not_execute_a_path_with_shell_metacharacters(
    home: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A `$(...)` in XDG_CONFIG_HOME is inert in the rc text the operator's shell sources."""
    import shlex
    import subprocess

    # A config dir named by a command substitution; executed on source it creates PWNED in cwd.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "$(touch PWNED)" / "agent6"))
    assert completions_cmd.cmd_completions("bash", print_only=False) == 0
    rc = home / ".bashrc"
    subprocess.run(["bash", "-c", f"source {shlex.quote(str(rc))}"], cwd=tmp_path, check=False)
    assert not (tmp_path / "PWNED").exists()


def test_zsh_respects_zdotdir(
    home: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    zdot = home / "zdot"
    zdot.mkdir()
    monkeypatch.setenv("ZDOTDIR", str(zdot))
    assert completions_cmd.cmd_completions("zsh", print_only=False) == 0
    assert ">>> agent6 completions >>>" in (zdot / ".zshrc").read_text(encoding="utf-8")
    capsys.readouterr()


def test_fish_writes_native_completions_file(
    home: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert completions_cmd.cmd_completions("fish", print_only=False) == 0
    target = home / ".config" / "fish" / "completions" / "agent6.fish"
    assert "agent6" in target.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "fish loads it automatically" in out
    assert "activate now" not in out  # fish needs no activation step


def test_unknown_shell_is_a_clear_error(
    home: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SHELL", "/bin/tcsh")
    assert completions_cmd.detect_shell() == "tcsh"
    assert completions_cmd.cmd_completions(None, print_only=False) == 2
    err = capsys.readouterr().err
    assert "tcsh" in err and "bash|zsh|fish|xonsh" in err


def test_detects_shell_from_env(
    home: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    assert completions_cmd.cmd_completions(None, print_only=True) == 0
    assert "agent6" in capsys.readouterr().out


def test_xonsh_writes_autoloaded_completer(
    home: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert completions_cmd.cmd_completions("xonsh", print_only=False) == 0
    target = home / ".config" / "xonsh" / "rc.d" / "agent6.xsh"
    code = target.read_text(encoding="utf-8")
    # The completer drives the argcomplete protocol against the live agent6, so the file must parse.
    import ast

    ast.parse(code)
    assert "_ARGCOMPLETE_STDOUT_FILENAME" in code
    assert "COMP_LINE" in code
    assert 'add_one_completer("agent6"' in code
    # Shell-hostile candidates are quoted; a missing or hung agent6 yields no candidates.
    assert "shlex.quote" in code
    assert "TimeoutExpired" in code
    out = capsys.readouterr().out
    assert "xonsh loads it automatically" in out
    assert "activate now" not in out  # rc.d needs no activation step


def test_xonsh_detected_in_process_walk(
    home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = home / "proc"
    for pid, comm, ppid in ((50, "uv", 40), (40, "xonsh", 30), (30, "bash", 1)):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "comm").write_text(comm + "\n", encoding="utf-8")
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 0 0 0", encoding="utf-8")
    monkeypatch.setattr("agent6.ui.cli.completions_cmd._PROC", proc)
    monkeypatch.setattr("os.getppid", lambda: 50)
    monkeypatch.setenv("SHELL", "/bin/bash")
    assert completions_cmd.detect_shell() == "xonsh"


def test_detects_shell_from_process_tree(
    home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shell is the nearest shell ancestor in the process tree, not $SHELL (the login shell)."""
    proc = home / "proc"
    # the parent chain: agent6 under uv (50), under fish (40), under bash (30), under init
    for pid, comm, ppid in ((50, "uv", 40), (40, "fish", 30), (30, "bash", 1)):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "comm").write_text(comm + "\n", encoding="utf-8")
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 0 0 0", encoding="utf-8")
    monkeypatch.setattr("agent6.ui.cli.completions_cmd._PROC", proc)
    monkeypatch.setattr("os.getppid", lambda: 50)
    monkeypatch.setenv("SHELL", "/bin/bash")
    assert completions_cmd.detect_shell() == "fish"


def test_bash_install_refuses_an_unreadable_rc(
    home: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bash install refuses an unreadable rc through the boundary and never appends blind."""
    from agent6 import errors

    rc = home / ".bashrc"
    rc.write_text("# mine\n", encoding="utf-8")
    rc.chmod(0o000)
    try:
        with pytest.raises(errors.OperatorError, match="could not read"):
            completions_cmd.cmd_completions("bash", print_only=False)
    finally:
        rc.chmod(0o600)
    assert rc.read_text(encoding="utf-8") == "# mine\n"


def test_a_moved_config_home_updates_the_stale_source_block(
    home: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """After the config home moves, install rewrites the stale source block.

    "Already sourced" over a block naming the old script path would leave completions broken.
    """
    assert completions_cmd.cmd_completions("bash", print_only=False) == 0
    rc = home / ".bashrc"
    old_block = rc.read_text(encoding="utf-8")
    moved = home / "elsewhere"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(moved))
    assert completions_cmd.cmd_completions("bash", print_only=False) == 0
    text = rc.read_text(encoding="utf-8")
    assert str(moved / "agent6" / "completions.bash") in text
    assert str(home / ".config" / "agent6" / "completions.bash") not in text
    assert text.count(">>> agent6 completions >>>") == 1
    assert text != old_block
    assert "updated the source path" in capsys.readouterr().out


def test_malformed_completion_markers_are_refused_untouched(
    home: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A duplicated or mangled marker set is refused untouched; agent6 edits only its one block."""
    rc = home / ".bashrc"
    rc.write_text(
        "# >>> agent6 completions >>>\nx\n# <<< agent6 completions <<<\n"
        "# >>> agent6 completions >>>\ny\n# <<< agent6 completions <<<\n",
        encoding="utf-8",
    )
    before = rc.read_text(encoding="utf-8")
    assert completions_cmd.cmd_completions("bash", print_only=False) == 2
    assert rc.read_text(encoding="utf-8") == before
    err = capsys.readouterr().err
    assert err.startswith("ERROR:") and "malformed agent6 completion markers" in err
