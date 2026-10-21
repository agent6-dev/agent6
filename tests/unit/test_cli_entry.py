# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The guarded console-script entry point `cli_main` sorts failures by fault.

An OperatorError refuses at exit 2 with no traceback; a closed stdout ends quietly at 141;
anything else crash-reports at exit 1.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

from agent6 import errors
from agent6.config import ConfigError
from agent6.ui.cli import cli_main, entry

_CRASH_MARKERS = ("unexpected", "full traceback", "report this")


def test_cli_main_passes_through_return_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def _ok(_argv: list[str] | None = None) -> int:
        return 3

    monkeypatch.setattr(entry, "main", _ok)
    assert cli_main() == 3


def test_cli_main_converts_unexpected_exception_to_friendly_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(_argv: list[str] | None = None) -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(entry, "main", _boom)
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    rc = cli_main()
    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR: unexpected RuntimeError: kaboom" in err
    # Points at a saved traceback that actually exists and contains the stack.
    tb_line = next(line for line in err.splitlines() if "full traceback:" in line)
    tb_path = pathlib.Path(tb_line.split("full traceback:", 1)[1].strip())
    assert tb_path.is_file()
    assert "RuntimeError: kaboom" in tb_path.read_text(encoding="utf-8")
    tb_path.unlink()


def test_an_operator_error_refuses_without_a_crash_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An operator error is `ERROR:` and exit 2; everything else is a crash report."""

    def _bad(_argv: list[str] | None = None) -> int:
        raise errors.OperatorError("no such machine file: overlay.toml")

    monkeypatch.setattr(entry, "main", _bad)
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    assert cli_main() == 2
    captured = capsys.readouterr()
    assert captured.err == "ERROR: no such machine file: overlay.toml\n"


def test_a_config_error_is_an_operator_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ConfigError subclasses OperatorError, so every reader gets the refusal surface for free."""

    def _bad(_argv: list[str] | None = None) -> int:
        raise ConfigError("Config file is not valid TOML (/x/config.toml): line 1")

    monkeypatch.setattr(entry, "main", _bad)
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    assert cli_main() == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ")
    assert "not valid TOML" in err
    assert not any(marker in err for marker in _CRASH_MARKERS)


def test_an_unreadable_config_file_refuses_end_to_end(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unreadable config file refuses with the file name and the OS reason, end to end."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "s"))
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    bad = tmp_path / "c.toml"
    bad.write_text("x = 1\n", encoding="utf-8")
    bad.chmod(0o000)
    try:
        rc = cli_main(["--config", str(bad), "config", "show"])
    finally:
        bad.chmod(0o600)
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ")
    assert "c.toml" in err
    assert not any(marker in err for marker in _CRASH_MARKERS)


def test_a_bad_budget_flag_refuses_end_to_end(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run --max-usd inf` names the flag it refuses at exit 2, not a saved traceback."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)  # past the git wall
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "s"))
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    rc = cli_main(["run", "task", "--max-usd", "inf"])
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ")
    assert "--max-usd" in err
    assert not any(marker in err for marker in _CRASH_MARKERS)


def test_cli_main_reraises_under_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_argv: list[str] | None = None) -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(entry, "main", _boom)
    monkeypatch.setenv("AGENT6_DEBUG", "1")
    with pytest.raises(RuntimeError, match="kaboom"):
        cli_main()


def test_cli_main_handles_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _interrupt(_argv: list[str] | None = None) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(entry, "main", _interrupt)
    assert cli_main() == 130
    assert "interrupted" in capsys.readouterr().err


def test_a_closed_stdout_ends_quietly(tmp_path: pathlib.Path) -> None:
    """A write after the reader closed ends at 141 like any SIGPIPE, with no crash report."""
    read_end, write_end = os.pipe()
    os.close(read_end)
    env = os.environ | {
        "XDG_CONFIG_HOME": str(tmp_path / "g"),
        "XDG_STATE_HOME": str(tmp_path / "s"),
        "TMPDIR": str(tmp_path),
    }
    env.pop("AGENT6_DEBUG", None)
    script = "import sys; from agent6.ui.cli import cli_main; sys.exit(cli_main())"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, "config", "show"],
            cwd=tmp_path,
            env=env,
            stdout=write_end,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
    finally:
        os.close(write_end)
    assert (proc.returncode, proc.stderr) == (141, "")
    assert not list(tmp_path.glob("agent6-crash-*"))


def test_a_short_answer_to_a_closed_stdout_ends_at_141_too(tmp_path: pathlib.Path) -> None:
    """`--version` fit stdout's buffer, so the closed pipe failed its flush at exit: 120."""
    read_end, write_end = os.pipe()
    os.close(read_end)
    env = os.environ | {
        "XDG_CONFIG_HOME": str(tmp_path / "g"),
        "XDG_STATE_HOME": str(tmp_path / "s"),
    }
    env.pop("AGENT6_DEBUG", None)
    script = "import sys; from agent6.ui.cli import cli_main; sys.exit(cli_main())"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, "--version"],
            cwd=tmp_path,
            env=env,
            stdout=write_end,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
    finally:
        os.close(write_end)
    assert (proc.returncode, proc.stderr) == (141, "")


def test_a_closed_stderr_ends_at_141_too(tmp_path: pathlib.Path) -> None:
    """With `2>&1 | head` the exit flush of stderr failed after the handler: exit 120."""
    read_end, write_end = os.pipe()
    os.close(read_end)
    env = os.environ | {
        "XDG_CONFIG_HOME": str(tmp_path / "g"),
        "XDG_STATE_HOME": str(tmp_path / "s"),
        "TMPDIR": str(tmp_path),
    }
    env.pop("AGENT6_DEBUG", None)
    script = "import sys; from agent6.ui.cli import cli_main; sys.exit(cli_main())"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, "sessions", "show", "no-such-run"],
            cwd=tmp_path,
            env=env,
            stdout=write_end,
            stderr=write_end,
            timeout=60,
            check=False,
        )
    finally:
        os.close(write_end)
    assert proc.returncode == 141
    assert not list(tmp_path.glob("agent6-crash-*"))


def test_a_plan_task_that_names_an_existing_plan_points_at_run_from(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 plan run <plan-id>` started a planning session whose task was the id."""
    from agent6 import paths
    from agent6.sessions import layout as sessions_layout

    monkeypatch.chdir(tmp_path)
    plan = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="quiet-lake-AAAA11", subdir="plans"
    )
    plan.ensure()
    plan.manifest_path.write_text('{"version": 3, "mode": "plan", "session_id": "x"}')

    rc = cli_main(["plan", "run", "quiet-lake-AAAA11"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "names an existing plan" in err and "agent6 run --from quiet-lake-AAAA11" in err


def test_a_long_plan_task_reaches_the_preflight(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The existing-plan check stats the task as a session dir name.

    Once a plan exists, a task past 255 bytes with no slash would raise `File name too long` there.
    """
    for argv in (["init", "-q", "-b", "main"], ["commit", "-q", "--allow-empty", "-m", "c1"]):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *argv], cwd=tmp_path, check=True
        )
    monkeypatch.chdir(tmp_path)
    from agent6 import paths
    from agent6.sessions import layout as sessions_layout

    # A repository that already holds a plan: the stat reaches the long name.
    sessions_layout.bucket_dir(paths.state_dir(tmp_path), "plans").mkdir(parents=True)
    rc = cli_main(["plan", "word " * 60 + "x" * 300])
    err = capsys.readouterr().err
    assert rc == 2, err
    assert "No providers configured" in err
