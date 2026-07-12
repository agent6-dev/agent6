# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Mid-run steering: a SIGINT handler that lets the operator pause the loop
and inject a one-shot instruction (or abort), plus interactive revised-prompt
selection. Independent of the run command; run.py wires it in.
"""

from __future__ import annotations

import contextlib
import io
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import termios
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent6.events import EventSink
from agent6.ui.bridge.approval import (
    clear_steer_answer,
    clear_steer_request,
    frontend_is_live,
    read_steer_answer,
    steer_answer_is_abort,
    steer_request_pending,
)
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._menu_input import menu_capable
from agent6.ui.cli._steer_menu import normalize_steer_choice, pause_menu


@dataclass
class SteerState:
    requested: Callable[[], bool]
    clear: Callable[[], None]
    prompt: Callable[[], str | None]
    restore: Callable[[], None]
    # Polled during a streaming call so a Stop interrupts a long turn promptly.
    abort_pending: Callable[[], bool]
    # Polled during a streaming call: True aborts the in-flight model call so
    # the steer prompt runs now instead of at the next between-step boundary.
    interrupt: Callable[[], bool]


def select_revised_prompt(
    original: str,
    revised: str,
    questions: tuple[str, ...],
) -> str | None:
    """Interactive accept/edit/skip prompt for prompt.revise_prompt."""
    print("\n[agent6] prompt revision proposed:", file=sys.stderr)
    print("\n--- revised ---", file=sys.stderr)
    print(revised, file=sys.stderr)
    if questions:
        print("\n--- clarifying questions ---", file=sys.stderr)
        for question in questions:
            print(f"- {question}", file=sys.stderr)
    print("\n--- original ---", file=sys.stderr)
    print(original, file=sys.stderr)
    while True:
        try:
            choice = (
                input("[agent6] revise_prompt: [a]ccept, [o]riginal, [e]dit, [q]uit? ")
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            return None
        if choice in {"", "a", "accept", "y", "yes"}:
            return revised
        if choice in {"o", "orig", "original", "s", "skip"}:
            return original
        if choice in {"q", "quit", "abort"}:
            return None
        if choice in {"e", "edit"}:
            # $EDITOR may be a command with flags ("code --wait"); split it,
            # and a missing binary is a choose-again, not a run-killing crash.
            editor = os.environ.get("EDITOR", "vi")
            editor_argv = shlex.split(editor) or ["vi"]
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                prefix="agent6-revised-task-",
                suffix=".md",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(revised.rstrip() + "\n")
            try:
                try:
                    result = subprocess.run([*editor_argv, str(tmp_path)], check=False)
                except OSError as exc:
                    print(
                        f"[agent6] cannot run $EDITOR ({editor!r}): {exc}; choose again.",
                        file=sys.stderr,
                    )
                    continue
                if result.returncode != 0:
                    print(
                        f"[agent6] editor exited {result.returncode}; choose again.",
                        file=sys.stderr,
                    )
                    continue
                edited = tmp_path.read_text(encoding="utf-8").strip()
            finally:
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
            if edited:
                return edited
            print("[agent6] edited prompt was empty; choose again.", file=sys.stderr)
            continue
        print("[agent6] choose accept, original, edit, or quit.", file=sys.stderr)


def tty_message(text: str) -> None:
    """Print to the controlling terminal directly, bypassing any stdout/stderr
    redirection (the TUI redirects the run's std streams to a log file)."""
    try:
        with open("/dev/tty", "w", encoding="utf-8") as tty:  # noqa: PTH123
            tty.write(text)
            tty.flush()
            return
    except OSError:
        with contextlib.suppress(Exception):
            print(text, file=sys.stderr, flush=True)


def tty_prompt(text: str, *, fall_back_to_stdin: bool = True) -> str | None:
    """Prompt on the controlling terminal directly (see ``tty_message``).
    Falls back to stdin when there is no controlling terminal, unless the
    caller must never consume piped stdin (``fall_back_to_stdin=False``:
    return None)."""
    try:
        # The getpass recipe: O_RDWR on the device + an unbuffered FileIO.
        # A plain open("/dev/tty", "r+") NEVER works -- buffered update mode
        # requires a seekable stream and a tty is not -- so every /dev/tty
        # prompt silently used the stdin fallback (or, without the fallback,
        # returned no answer at all).
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
        # Discard type-ahead before prompting (the sudo/ssh rule): text typed
        # before this prompt existed was aimed at something else -- e.g. a
        # pause-menu command typed during the "pausing after this step" window
        # must not ride into a run_command [y/N/a] approval as its answer.
        with contextlib.suppress(Exception):
            termios.tcflush(fd, termios.TCIFLUSH)
        tty = io.TextIOWrapper(
            io.FileIO(fd, "r+"), encoding="utf-8", errors="replace", write_through=True
        )
    except OSError:
        if not fall_back_to_stdin:
            return None
        try:
            return input(text)
        except (EOFError, KeyboardInterrupt):
            return None
    try:
        with tty:
            tty.write(text)
            line = tty.readline()
            return line.rstrip("\n") if line else None
    except OSError:
        # The terminal vanished mid-prompt; the text already printed, so do
        # not prompt again on stdin.
        return None


def install_steer_sigint(
    events: EventSink, run_dir: Path, console_view: ConsoleView | None = None
) -> SteerState:
    """Install a SIGINT handler with escalating stages.

    * 1st Ctrl-C: pause at the next safe boundary (between steps; the
      in-flight model call finishes first). Emits ``run.steer_requested``;
      the prompt is a TUI modal when the TUI is live, otherwise the
      interactive pause menu; with redirected std streams the menu cannot
      own the line, so a plain prompt goes to the controlling terminal
      (``/dev/tty``) instead.
    * 2nd Ctrl-C: interrupt the in-flight model call and prompt now.
    * 3rd Ctrl-C (or Ctrl-C at the pause prompt itself): KeyboardInterrupt,
      stopping the run (resumable with ``agent6 resume``).

    ``console_view``, when given, has its heartbeat spinner suspended for the
    prompt's duration: the spinner's per-tick line-erase otherwise wipes the
    pause-menu line and its Tab preview.

    Returns callables for the workflow plus a ``restore`` hook to put the
    previous handler back when the run is done.
    """
    state: dict[str, Any] = {"stage": 0, "prompting": False}

    def _handler(_signum: int, _frame: Any) -> None:
        # A boundary can be a whole model response away (a reasoning model may
        # think for 30-60s), hence the escalation; at the pause prompt itself a
        # Ctrl-C stops the run, as the pause banner promised.
        if state["prompting"] or state["stage"] >= 2:
            raise KeyboardInterrupt
        if state["stage"] == 1:
            state["stage"] = 2
            if not frontend_is_live(run_dir):
                tty_message("\n[agent6] interrupting this step. Ctrl-C again to stop the run.\n")
            return
        state["stage"] = 1
        # Drop a STALE answer file (one without a request marker) so it is not
        # instantly consumed as this new prompt's answer. An answer with a
        # pending request is a live front-end steer the loop has not consumed
        # yet; deleting it would silently discard the operator's instruction.
        if not steer_request_pending(run_dir):
            clear_steer_answer(run_dir)
        events.emit("run.steer_requested", source="sigint")
        # With the TUI up, the steer prompt is a modal, don't scribble on the
        # terminal it owns. Otherwise tell the user a prompt is coming.
        if not frontend_is_live(run_dir):
            tty_message(
                "\n[agent6] pausing after this step: steer / continue / stop / detach."
                " Ctrl-C again to interrupt now.\n"
            )

    previous = signal.signal(signal.SIGINT, _handler)

    def requested() -> bool:
        # Either a Ctrl-C (any stage) OR a front-end steer request marker.
        return state["stage"] >= 1 or steer_request_pending(run_dir)

    def interrupt() -> bool:
        # A double Ctrl-C aborts the in-flight call; so does a front-end steer
        # (the instruction is already typed, injecting it beats finishing a
        # step it may contradict).
        return state["stage"] >= 2 or steer_request_pending(run_dir)

    def clear() -> None:
        state["stage"] = 0
        clear_steer_answer(run_dir)
        clear_steer_request(run_dir)

    def prompt() -> str | None:
        # TUI live: the user answers a modal; read its file-bridge result.
        if frontend_is_live(run_dir):
            answer = read_steer_answer(run_dir)
            # A dismissed/abandoned modal yields None (read_steer_answer timed out
            # or the TUI died). Clear the request marker on THIS no-answer path so a
            # persisting `steer.request` cannot re-trigger another 600s blocking
            # read at the very next boundary, looping the run. A genuinely-answered
            # steer leaves clearing to the caller's clear() (with the answer already
            # consumed). The SIGINT stage is also cleared so a stale Ctrl-C request
            # doesn't immediately re-arm the same dead prompt.
            if answer is None:
                state["stage"] = 0
                clear_steer_request(run_dir)
            return answer
        pause = console_view.pause if console_view is not None else contextlib.nullcontext
        state["prompting"] = True
        try:
            with pause():
                if menu_capable():
                    # The interactive pause menu: line editing, history, and a
                    # fish-style Tab preview of the slash commands.
                    return pause_menu(run_dir)
                return normalize_steer_choice(
                    tty_prompt(
                        "[agent6] paused: [enter] continue · type to steer · q stop · d detach: "
                    )
                )
        finally:
            state["prompting"] = False

    def restore() -> None:
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGINT, previous)

    return SteerState(
        requested=requested,
        clear=clear,
        prompt=prompt,
        restore=restore,
        abort_pending=lambda: steer_answer_is_abort(run_dir),
        interrupt=interrupt,
    )


def file_bridge_steer(run_dir: Path) -> SteerState:
    """Steer for a run with no controlling terminal (detached spawn from the
    TUI hub or the web UI): no SIGINT handler, requests and answers travel
    only over the front-end file bridge. Without this, a hub-spawned run
    would never poll the ``steer.request`` marker and every web/TUI steer
    would be silently lost."""

    def prompt() -> str | None:
        answer = read_steer_answer(run_dir)
        # No answer (front-end died or abandoned the prompt): clear the
        # request marker so it cannot re-trigger another blocking read at the
        # very next boundary, looping the run.
        if answer is None:
            clear_steer_request(run_dir)
        return answer

    def clear() -> None:
        clear_steer_answer(run_dir)
        clear_steer_request(run_dir)

    return SteerState(
        requested=lambda: steer_request_pending(run_dir),
        clear=clear,
        prompt=prompt,
        restore=lambda: None,
        abort_pending=lambda: steer_answer_is_abort(run_dir),
        interrupt=lambda: steer_request_pending(run_dir),
    )


def make_steer_state(
    events: EventSink, run_dir: Path, console_view: ConsoleView | None = None
) -> SteerState:
    """Install the steer SIGINT handler when a controlling terminal exists
    (covers run/plan/ask with or without the TUI); else steer purely over the
    front-end file bridge (detached runs)."""
    try:
        with open("/dev/tty", encoding="utf-8"):  # noqa: PTH123
            pass
    except OSError:
        return file_bridge_steer(run_dir)
    return install_steer_sigint(events, run_dir, console_view)
