# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Watch a live run: the dashboard co-process, the stream modes and the harness's logger."""

from __future__ import annotations

import contextlib
import os
import pathlib
import signal
import subprocess
import sys
from collections.abc import Callable, Generator

from agent6.ui.cli import _console_view


def loop_logger(mode: str, console_view: _console_view.ConsoleView | None) -> Callable[[str], None]:
    """Return the harness's text logger for the mode.

    With a live console view, notices go through it, under its lock and after its
    spinner line is cleared; the loop's state narration and the lines the stream
    already shows are suppressed unless `AGENT6_DEBUG=1`. Headless and `ask` keep
    the full trace on their own stream.

    Args:
        mode: The session mode.
        console_view: The live view, when the run has one.

    Returns:
        The logger.
    """
    if console_view is None:
        # Redirected to a file, the stream is block-buffered; unflushed, the run reads as dead.
        return _eprint if mode == "ask" else _print_flush
    debug = os.environ.get("AGENT6_DEBUG") == "1"

    def _filtered(msg: str) -> None:
        """Pass a notice to the view, dropping narration unless debugging."""
        stripped = msg.removeprefix("[agent6] ")
        narration = (
            "compaction",
            "  tool_error:",
            "  auto-commit:",
            "  final checkpoint:",
            "STEER:",
            "  injecting steering instruction",
            "  ask answered",
        )
        if not debug and ("LOOP:" in msg or stripped.startswith(narration)):
            return
        console_view.notice(msg)

    return _filtered


def _print_flush(msg: str) -> None:
    """Print to stdout and flush, so a redirected log shows the trace live."""
    print(msg, flush=True)


def _eprint(msg: str) -> None:
    """Print to stderr and flush, for `ask`, whose stdout is the answer."""
    print(msg, file=sys.stderr, flush=True)


def _tui_available() -> bool:
    """Return whether textual is installed."""
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("textual") is not None


def should_spawn_tui(*, tui: bool, interactive: bool, mode: str) -> bool:
    """Return whether the run opens the dashboard TUI, warning when `--tui` cannot run.

    The dashboard needs a real TTY, is for `run` and `plan` (an ask's answer is its
    deliverable), and excludes `-i`.

    Args:
        tui: The `--tui` flag.
        interactive: The `-i` flag.
        mode: The session mode.
    """
    if not tui:
        return False
    if interactive or mode not in ("run", "plan"):
        print("[agent6] --tui is not available here; continuing in CLI mode.", file=sys.stderr)
        return False
    if not sys.stdout.isatty():
        print("[agent6] --tui needs a TTY; continuing in CLI mode.", file=sys.stderr)
        return False
    if not _tui_available():
        print(
            "[agent6] --tui needs 'textual' (part of the base install; this environment"
            " is missing it); continuing in CLI mode.",
            file=sys.stderr,
        )
        return False
    return True


def stream_modes(*, tui_enabled: bool) -> tuple[bool, bool]:
    """Return `(stream_text, console_stream)` for the worker provider.

    `stream_text` makes the provider emit the delta events every live view renders;
    `console_stream` also subscribes a `ConsoleView` to render them on stderr.
    Streaming is on for a stderr TTY, under `AGENT6_FORCE_STREAM=1` (a gateway that
    corrupts a non-streaming body), and under `AGENT6_STREAM_TO_LOG=1`, which the
    `tui` hub sets for a run it watches on the dashboard: deltas without the echo.

    Args:
        tui_enabled: The dashboard owns the terminal.
    """
    stream_to_log = os.environ.get("AGENT6_STREAM_TO_LOG") == "1"
    stream_text = (
        sys.stderr.isatty() or os.environ.get("AGENT6_FORCE_STREAM") == "1" or stream_to_log
    )
    # Echo only when a console reads it: not under the TUI, not for a hub-watched run.
    console_stream = stream_text and not tui_enabled and not stream_to_log
    return stream_text, console_stream


@contextlib.contextmanager
def tui_session(session_dir: pathlib.Path, *, enabled: bool) -> Generator[None]:
    """Run the dashboard TUI as a co-process that owns the terminal for the block.

    This process's console goes to `<session_dir>/tui_console.log` meanwhile; the
    TUI tails the log and approvals cross the file bridge. The TUI holds the finished
    dashboard until the operator leaves, and the block waits for that. A spawn
    failure degrades to a run without the TUI.

    Args:
        session_dir: The run's dir.
        enabled: Whether to spawn at all.

    Yields:
        Nothing; the run proceeds inside the block.

    Raises:
        KeyboardInterrupt: The operator interrupted; re-raised after the TUI is down.
    """
    if not enabled:
        yield
        return
    try:
        # Opened before the spawn: a failure after it would orphan a TUI holding the terminal.
        log_fh = (session_dir / "tui_console.log").open("w", encoding="utf-8")
    except OSError as exc:
        print(f"[agent6] could not start TUI ({exc}); continuing without it.", file=sys.stderr)
        yield
        return
    try:
        # -P keeps the workspace off sys.path, so a top-level `agent6/` there cannot shadow ours.
        proc = subprocess.Popen(
            [
                sys.executable,
                "-P",
                "-m",
                "agent6.ui.tui",
                "--watch",
                str(session_dir),
                "--exit-on-end",
            ]
        )
    except OSError as exc:
        log_fh.close()
        print(f"[agent6] could not start TUI ({exc}); continuing without it.", file=sys.stderr)
        yield
        return
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = log_fh
    sys.stderr = log_fh
    interrupted = False
    try:
        yield
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        # Wait for the operator to leave, not a deadline; Ctrl-C still tears everything down.
        # A dashboard gone before the run ended left the terminal silent for the rest of it.
        gone_early = not interrupted and proc.poll() is not None
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        finally:
            # Whatever the teardown does, this process gets its console back.
            sys.stdout, sys.stderr = orig_out, orig_err
            with contextlib.suppress(Exception):
                log_fh.close()
        if gone_early:
            how = "closed" if proc.returncode == 0 else f"exited with code {proc.returncode}"
            print(
                f"[agent6] the dashboard {how} before the run ended; the run's console"
                f" output is in {log_fh.name}",
                file=sys.stderr,
            )
