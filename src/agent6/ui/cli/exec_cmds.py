# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 exec --new`, and the policy and stdio every `agent6 exec` form shares.

`exec_new` builds a fresh sandbox from the current config, with no session, so an operator
tests their sandbox settings by hand. `exec_policy` and `run_attached` also serve
`net_cmds.exec_in_session`, which runs a command in a live run's sandbox.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import pathlib
import pty
import select
import shutil
import signal
import sys
import termios
import tty
import types
from typing import Any

from agent6 import child_env, kinds
from agent6.app import _setup, confine
from agent6.config import Config
from agent6.sandbox import detect, jail, tool_paths
from agent6.tools import policy as tools_policy
from agent6.ui.cli import _common

# The relay wakes this often to notice the command ended while a child of it holds the pty.
_RELAY_POLL_S = 0.05


def resolve_shell() -> str:
    """Return the shell `exec --new` opens: `$SHELL` when the jail's PATH holds it, else /bin/sh."""
    name = pathlib.Path(os.environ.get("SHELL", "")).name
    found = shutil.which(name, path=tool_paths.jail_search_path()) if name else None
    return found or "/bin/sh"


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte; `os.write` may take fewer."""
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _relay(master: int, proc: jail.JailedProcess) -> None:
    """Copy bytes between the operator's terminal and the pty until the command ends.

    The terminal is raw for the relay, so Ctrl-C and the other control keys reach the pty as
    bytes, never this process as signals. The pty follows the terminal's size.

    Args:
        master: The pty's master side.
        proc: The command.
    """

    def _resize(_signum: int = 0, _frame: types.FrameType | None = None) -> None:
        with contextlib.suppress(OSError):
            size = fcntl.ioctl(0, termios.TIOCGWINSZ, bytes(8))
            fcntl.ioctl(master, termios.TIOCSWINSZ, size)

    saved = termios.tcgetattr(0)
    _resize()
    old_winch = signal.signal(signal.SIGWINCH, _resize)
    try:
        tty.setraw(0, termios.TCSADRAIN)  # the default flush would drop what was typed ahead
        watched = [master, 0]
        while proc.popen.poll() is None:
            ready, _, _ = select.select(watched, [], [], _RELAY_POLL_S)
            if master in ready:
                try:
                    shown = os.read(master, 65536)
                except OSError:  # EIO: every holder of the slave is gone
                    return
                _write_all(1, shown)
            if 0 in ready:
                typed = os.read(0, 4096)
                if typed:
                    _write_all(master, typed)
                else:
                    watched.remove(0)
        # What the command wrote before it ended; a child still holding the slave is not waited on.
        os.set_blocking(master, False)
        with contextlib.suppress(OSError):
            while tail := os.read(master, 65536):
                _write_all(1, tail)
    finally:
        signal.signal(signal.SIGWINCH, old_winch)
        with contextlib.suppress(termios.error):  # the terminal itself is gone on a hangup
            termios.tcsetattr(0, termios.TCSADRAIN, saved)


def _exit_on_signal(signum: int, _frame: types.FrameType | None) -> None:
    """Turn a terminating signal into an exit, so `run_attached` cleans up.

    Args:
        signum: The signal received.

    Raises:
        SystemExit: Always, with 128 plus the signal.
    """
    raise SystemExit(128 + signum)


def exec_policy(
    cwd: pathlib.Path,
    cfg: Config,
    isolation: kinds.IsolationLevel,
    argv: tuple[str, ...],
    network: kinds.NetworkMode | None = None,
    *,
    worktree_git_dir: pathlib.Path | None = None,
) -> kinds.JailPolicy:
    """Build the policy of the operator's command: a run command's, with no time limit.

    Unsandboxed, its environment is the one `run_in_jail` gives a command at that level: this
    process's minus the provider keys, under the policy's own values.

    Args:
        cwd: The workspace.
        cfg: The effective config.
        isolation: The resolved isolation level.
        argv: The command.
        network: The run's recorded network; the config's when None.
        worktree_git_dir: The repository git dir recorded for a session whose workspace is
            a linked worktree.

    Returns:
        The policy.

    Raises:
        JailUnavailableError: The workspace is under a hidden path, the jail's HOME is refused
            or cannot be created, or the worktree's `.git` pointer no longer names the recorded
            git dir.
    """
    # A run refuses this at its start; the config read for a live run can have the entry since.
    if (hidden := confine.check_workspace_outside_hidden_paths(cfg, cwd)) is not None:
        raise jail.JailUnavailableError(hidden)
    env_base = child_env.without_provider_keys(dict(os.environ)) if isolation == "none" else None
    return tools_policy.jail_policy(
        cwd,
        cfg,
        isolation,
        argv,
        network=network,
        timeout_s=0.0,
        env_base=env_base,
        worktree_git_dir=worktree_git_dir,
    )


def run_attached(policy: kinds.JailPolicy, session_net: jail.SessionNetwork | None = None) -> int:
    """Run the policy's command on the operator's stdio and return its exit code.

    With stdin and stdout both on a terminal the command gets a pty of its own and this process
    relays the bytes: nothing inside holds the operator's terminal. The pty is the controlling
    terminal of the command's session and the command runs in its foreground group, so the
    pty's signals and job control reach it. Otherwise stdin, stdout and stderr pass straight
    through, so pipes and redirects work, and a terminal among them gets back the modes it had.
    The command and everything it started end with this call, however it returns.

    Args:
        policy: The jail policy.
        session_net: The session network the command joins, when its policy names one.

    Returns:
        The command's exit code; 128 plus the signal for a command a signal ended.

    Raises:
        JailUnavailableError: The launcher is missing or cannot be executed.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    old_handlers = {
        sig: signal.signal(sig, _exit_on_signal)
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)
    }
    proc: jail.JailedProcess | None = None
    master = -1
    modes: dict[int, list[Any]] = {}
    try:
        if os.isatty(0) and os.isatty(1):
            master, slave = pty.openpty()
            try:
                proc = jail.spawn_in_jail(
                    policy,
                    stdin=slave,
                    stdout=slave,
                    # A redirected stderr stays the operator's file.
                    stderr=slave if os.isatty(2) else None,
                    session_net=session_net,
                    controlling_terminal=True,
                )
            finally:
                os.close(slave)
            _relay(master, proc)
        else:
            modes = {fd: termios.tcgetattr(fd) for fd in (0, 1, 2) if os.isatty(fd)}
            proc = jail.spawn_in_jail(policy, session_net=session_net)
        code = proc.popen.wait()
        return code if code >= 0 else 128 - code
    finally:
        left = proc.close() if proc is not None else frozenset[int]()
        if master >= 0:
            os.close(master)
        for fd, saved in modes.items():
            with contextlib.suppress(termios.error):  # the terminal itself is gone on a hangup
                # Only a changed terminal is written: a write from the background stops agent6.
                if termios.tcgetattr(fd) != saved:
                    termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if left:
            _common.error(jail.survivors_message(left))


def exec_new(cfg: Config, cwd: pathlib.Path, argv: tuple[str, ...]) -> int:
    """Run a command, or the operator's shell, in a fresh sandbox built from the config.

    No session: the policy a run's commands get (mounts, hidden paths, network, isolation
    level), after the refusals and warnings a run prints at start, gone when the command ends.

    Args:
        cfg: The effective config.
        cwd: The workspace.
        argv: The command; empty opens the shell `resolve_shell` names.

    Returns:
        The command's exit code; 2 when the host or the config refuses the sandbox.
    """
    try:
        env = _setup.detect_env()
        isolation = detect.resolve_isolation(cfg.sandbox.isolation, env)
        refusal = confine.check_network_support(cfg, isolation) or confine.config_refusal(
            cfg, isolation, cwd
        )
        if refusal is not None:
            _common.refuse(refusal)
            return 2
        confine.warn_sandbox_gaps(isolation, env, cfg, root=cwd)
        if not argv:
            argv = (resolve_shell(),)
            _common.note(f"{argv[0]} in a fresh sandbox, isolation {isolation}.")
        policy = exec_policy(cwd, cfg, isolation, argv)
        # The network a run opens for its commands, for this one command.
        session_net = jail.SessionNetwork.open() if policy.network == "session" else None
    except (jail.JailUnavailableError, detect.IsolationUnavailableError) as exc:
        _common.refuse(f"{exc}")
        return 2
    try:
        return run_attached(policy, session_net)
    except jail.JailUnavailableError as exc:
        _common.error(f"{exc}")
        return 2
    finally:
        if session_net is not None:
            session_net.close()
