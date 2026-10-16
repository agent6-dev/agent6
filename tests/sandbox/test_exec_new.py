# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 exec --new`, live: the real CLI on a pipe and on a pty."""

from __future__ import annotations

import os
import pathlib
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import termios
import time
from collections.abc import Iterator

import pytest

_CLI = (sys.executable, "-c", "from agent6.ui.cli import entry; raise SystemExit(entry.cli_main())")


def _exec_new(*args: str, config: pathlib.Path | None = None) -> list[str]:
    return [*_CLI, *(("--config", str(config)) if config else ()), "exec", "--new", *args]


@pytest.mark.needs_namespaces
def test_a_pipe_passes_through_and_the_exit_code_is_the_commands(tmp_path: pathlib.Path) -> None:
    """`echo hi | agent6 exec --new cat` prints hi: stdin and stdout are the caller's."""
    done = subprocess.run(
        _exec_new("cat"), input=b"hi\n", capture_output=True, cwd=tmp_path, check=False
    )
    assert (done.returncode, done.stdout) == (0, b"hi\n"), done.stderr
    failed = subprocess.run(
        _exec_new("--", "sh", "-c", "exit 7"),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        cwd=tmp_path,
        check=False,
    )
    assert failed.returncode == 7, failed.stderr


@pytest.fixture
def granted_config_home(monkeypatch: pytest.MonkeyPatch) -> Iterator[pathlib.Path]:
    """An `XDG_CONFIG_HOME` outside /tmp, which full replaces whole: only the mask can hide it."""
    base = pathlib.Path("/var/tmp")
    if not (base.is_dir() and os.access(base, os.W_OK)):
        pytest.skip("/var/tmp is not writable here")
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="agent6-exec-new-", dir=base))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(scratch))
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


@pytest.mark.needs_namespaces
def test_the_config_dir_is_masked_under_a_grant_that_holds_it(
    tmp_path: pathlib.Path, granted_config_home: pathlib.Path
) -> None:
    """A read grant over the config dir's parent shows the parent and never `secrets.toml`."""
    private = granted_config_home / "agent6"
    private.mkdir()
    (private / "secrets.toml").write_text("", encoding="utf-8")
    (granted_config_home / "beside.txt").write_text("", encoding="utf-8")
    config = tmp_path / "grant.toml"
    config.write_text(
        f'[sandbox]\nisolation = "full"\nextra_read_paths = ["{granted_config_home}"]\n',
        encoding="utf-8",
    )
    done = subprocess.run(
        _exec_new("--", "sh", "-c", f"ls -A {granted_config_home} {private}", config=config),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        cwd=tmp_path,
        check=False,
    )
    assert b"beside.txt" in done.stdout, done.stderr
    assert b"secrets.toml" not in done.stdout + done.stderr


def _read_until(fd: int, pattern: bytes, timeout_s: float = 20.0) -> bytes:
    """Return the pty's output up to the first match of the pattern, failing at the timeout."""
    seen = b""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        seen += chunk
        if re.search(pattern, seen):
            return seen
    pytest.fail(f"never saw {pattern!r} in {seen!r}")


def _on_a_pty(
    argv: list[str], cwd: pathlib.Path, env: dict[str, str]
) -> tuple[subprocess.Popen[bytes], int]:
    """Start the CLI with a pty as its terminal; the master is the test's keyboard and screen."""
    master, slave = pty.openpty()
    try:
        proc = subprocess.Popen(argv, stdin=slave, stdout=slave, stderr=slave, cwd=cwd, env=env)
    finally:
        os.close(slave)
    return proc, master


_UNSANDBOXED = {"AGENT6_DISABLE_SANDBOX": "1"}


@pytest.mark.parametrize(
    ("level", "isolation"),
    [
        pytest.param(_UNSANDBOXED, None, id="none"),
        pytest.param({}, "partial", id="partial"),
        pytest.param({}, None, id="default", marks=pytest.mark.needs_namespaces),
    ],
)
def test_a_terminal_gets_a_shell_with_job_control(
    tmp_path: pathlib.Path, level: dict[str, str], isolation: str | None
) -> None:
    """On a terminal the shell reads what is typed, Ctrl-C ends only its job, and exit 3 is 3.

    The launcher starts a command in a process group of its own: left in the background of
    its terminal, the shell stops at its first read.
    """
    env = {**os.environ, **level, "SHELL": "/bin/sh", "PS1": "$ "}
    config = None
    if isolation is not None:
        config = tmp_path / "level.toml"
        config.write_text(f'[sandbox]\nisolation = "{isolation}"\n', encoding="utf-8")
    proc, master = _on_a_pty(_exec_new(config=config), tmp_path, env)
    try:
        _read_until(master, rb"/bin/sh in a fresh sandbox, isolation \w+\.")
        os.write(master, b"echo MARK-$((6*7))\n")
        _read_until(master, rb"MARK-42")
        os.write(master, b"echo STARTED-$((1+1)); sleep 30\n")
        _read_until(master, rb"STARTED-2")
        time.sleep(0.5)
        os.write(master, b"\x03")
        os.write(master, b"echo ALIVE-$((2+3))\n")
        _read_until(master, rb"ALIVE-5")
        os.write(master, b"exit 3\n")
        assert proc.wait(timeout=20) == 3
    finally:
        proc.kill()
        proc.wait()
        os.close(master)


def test_a_redirected_stdout_gets_the_commands_bytes_alone(tmp_path: pathlib.Path) -> None:
    """Typed at a terminal, `agent6 exec --new CMD > file` writes the file no CR and no stderr."""
    master, slave = pty.openpty()
    try:
        done = subprocess.run(
            _exec_new("--", "sh", "-c", "echo out; echo err >&2; exit 4"),
            stdin=slave,
            capture_output=True,
            cwd=tmp_path,
            env={**os.environ, **_UNSANDBOXED},
            check=False,
        )
    finally:
        os.close(slave)
        os.close(master)
    assert (done.returncode, done.stdout) == (4, b"out\n"), done.stderr
    assert done.stderr.endswith(b"err\n")


def test_a_closed_stdout_ends_the_command_without_a_crash_report(tmp_path: pathlib.Path) -> None:
    """Typed at a terminal, `agent6 exec --new seq 1 2000000 | head -1` exits 141, as seq does."""
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        _exec_new("seq", "1", "2000000"),
        stdin=slave,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env={**os.environ, **_UNSANDBOXED},
    )
    try:
        assert proc.stdout is not None and proc.stderr is not None
        assert proc.stdout.readline() == b"1\n"
        proc.stdout.close()
        code = proc.wait(timeout=20)
        err = proc.stderr.read()
    finally:
        proc.kill()
        proc.wait()
        os.close(slave)
        os.close(master)
    assert code == 128 + signal.SIGPIPE, err
    assert b"unexpected" not in err


def test_an_inherited_terminal_gets_its_modes_back(tmp_path: pathlib.Path) -> None:
    """A command that leaves the terminal on its stdout raw does not leave the operator's raw."""
    master, slave = pty.openpty()
    try:
        before = termios.tcgetattr(slave)
        done = subprocess.run(
            _exec_new("--", "sh", "-c", "stty raw -echo <&1"),
            stdin=subprocess.DEVNULL,
            stdout=slave,
            stderr=slave,
            cwd=tmp_path,
            env={**os.environ, **_UNSANDBOXED},
            check=False,
        )
        after = termios.tcgetattr(slave)
    finally:
        os.close(slave)
        os.close(master)
    assert done.returncode == 0
    assert after == before


def test_sigquit_ends_the_command_and_restores_the_terminal(tmp_path: pathlib.Path) -> None:
    """A SIGQUIT to agent6 during the relay exits 131 and leaves the terminal as it was."""
    proc, master = _on_a_pty(_exec_new("sleep", "30"), tmp_path, {**os.environ, **_UNSANDBOXED})
    try:
        before = termios.tcgetattr(master)
        deadline = time.monotonic() + 20
        while termios.tcgetattr(master) == before and time.monotonic() < deadline:
            time.sleep(0.05)
        proc.send_signal(signal.SIGQUIT)
        assert proc.wait(timeout=20) == 128 + signal.SIGQUIT
        assert termios.tcgetattr(master) == before
    finally:
        proc.kill()
        proc.wait()
        os.close(master)
