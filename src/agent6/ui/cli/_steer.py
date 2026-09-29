# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Mid-run steering: the escalating Ctrl-C handler, the /dev/tty writers and the prompt review."""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import termios
from collections.abc import Callable, Generator
from typing import Any

from agent6 import budget
from agent6 import events as agent6_events
from agent6.app import frontend
from agent6.sessions import ipc
from agent6.ui import steer
from agent6.ui.cli import _common, _console_view, _menu_input, _steer_menu
from agent6.viewmodel import transcript


@contextlib.contextmanager
def idle_prompt_sigint() -> Generator[None]:
    """Restore the default Ctrl-C for the block, an idle prompt where one press simply raises.

    No step is in flight there, so the escalating handler would lie, PEP 475 would
    retry the interrupted `input()`, and the armed stage would open a phantom menu.

    Yields:
        Nothing; the handler is restored after the block.
    """
    prev = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev)


def select_revised_prompt(
    original: str,
    revised: str,
    questions: tuple[str, ...],
    console_view: _console_view.ConsoleView | None = None,
) -> str | None:
    """Ask the operator to accept, keep, edit or quit a revised prompt.

    Args:
        original: The prompt as typed.
        revised: The model's revision.
        questions: The model's open questions about the task.
        console_view: The live view, paused for the exchange.

    Returns:
        The prompt to run; None (quit, Ctrl-C, Ctrl-D) stops the run.
    """
    pause = console_view.pause if console_view is not None else contextlib.nullcontext
    with pause():
        return _select_revised_prompt(original, revised, questions)


def _select_revised_prompt(
    original: str,
    revised: str,
    questions: tuple[str, ...],
) -> str | None:
    """Return the chosen prompt; see `select_revised_prompt`."""
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
            with idle_prompt_sigint():
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
            edited = _edit_in_editor(revised)
            if edited is not None:
                return edited
            continue
        print("[agent6] choose accept, original, edit, or quit.", file=sys.stderr)


def _edit_in_editor(revised: str) -> str | None:
    """Return the text after one `$EDITOR` round trip.

    Args:
        revised: The text to edit.

    Returns:
        The saved text, or None with the reason on stderr for an operator-fixable
        failure: a missing editor, a non-zero exit, a non-UTF-8 or empty save.
    """
    argv = _common.editor_argv()
    if argv is None:
        print("[agent6] choose again.", file=sys.stderr)
        return None
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="agent6-revised-task-",
        suffix=".md",
        delete=False,
    ) as tmp:
        tmp_path = pathlib.Path(tmp.name)
        tmp.write(revised.rstrip() + "\n")
    try:
        try:
            result = subprocess.run([*argv, str(tmp_path)], check=False)
        except OSError as exc:
            print(
                f"[agent6] cannot run $EDITOR ({argv[0]!r}): {exc}; choose again.", file=sys.stderr
            )
            return None
        if result.returncode != 0:
            print(f"[agent6] editor exited {result.returncode}; choose again.", file=sys.stderr)
            return None
        try:
            edited = tmp_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            print(f"[agent6] edited prompt is not UTF-8: {exc}; choose again.", file=sys.stderr)
            return None
    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
    if edited:
        return edited
    print("[agent6] edited prompt was empty; choose again.", file=sys.stderr)
    return None


# The controlling terminal, which the writers below reach past a TUI's stream redirect.
TTY_PATH = "/dev/tty"


def tty_message(text: str) -> None:
    """Print to the controlling terminal directly, through the same scrubber as stdout."""
    try:
        with open(TTY_PATH, "w", encoding="utf-8") as tty:  # noqa: PTH123
            tty.write(transcript.scrub_terminal_output(text))
            tty.flush()
            return
    except OSError:
        with contextlib.suppress(Exception):
            print(text, file=sys.stderr, flush=True)


def tty_prompt(
    text: str,
    *,
    fall_back_to_stdin: bool = True,
    plain: str | None = None,
    until: Callable[[], bool] | None = None,
) -> str | None:
    """Prompt on the controlling terminal directly, like `tty_message`.

    Args:
        text: The prompt.
        fall_back_to_stdin: Prompt on stdin without a controlling terminal; False
            returns None instead, for a caller that must never consume piped stdin.
        plain: The prompt without terminal escapes, for the stdin fallback.
        until: Polled while the prompt waits; once it holds, a partly typed line is
            discarded and the prompt ends.

    Returns:
        The line, or None: no terminal, EOF, or `until` held.
    """
    try:
        # The getpass recipe: a buffered "r+" open needs a seekable stream, which a tty is not.
        fd = os.open(TTY_PATH, os.O_RDWR | os.O_NOCTTY)
        # Type-ahead was aimed at something else: a menu command must not answer an approval.
        with contextlib.suppress(Exception):
            termios.tcflush(fd, termios.TCIFLUSH)
        tty = io.TextIOWrapper(
            io.FileIO(fd, "r+"), encoding="utf-8", errors="replace", write_through=True
        )
    except OSError:
        if not fall_back_to_stdin:
            return None
        try:
            if until is None:
                return input(text if plain is None else plain)
            sys.stdout.write(text if plain is None else plain)
            sys.stdout.flush()
            return _menu_input.read_line_until(sys.stdin.fileno(), until)
        except (EOFError, KeyboardInterrupt, OSError, ValueError):
            return None
    try:
        with tty:
            tty.write(transcript.scrub_terminal_output(text))
            line = _menu_input.read_line_until(fd, until)
            if line is None and until is not None:
                # Whatever was typed was aimed at a prompt that is over.
                with contextlib.suppress(Exception):
                    termios.tcflush(fd, termios.TCIFLUSH)
                tty.write("\n")
            return line
    except OSError:
        # The terminal vanished mid-prompt; the text already printed, so no stdin retry.
        return None


def format_session_facts(facts: frontend.SessionFacts) -> str:
    """Return the one-line status the pause banner and Ctrl-Z print, spend first."""
    return (
        f"{budget.format_usd(facts.spend_usd, partial=facts.spend_partial)}"
        f" · {facts.model} · commands {facts.run_commands} · {facts.isolation}"
    )


def _status_suffix(session_facts: Callable[[], frontend.SessionFacts] | None) -> str:
    """Return the indented status line under the pause banner, or "" without facts."""
    if session_facts is None:
        return ""
    return f"          {format_session_facts(session_facts())}\n"


_JOB_CONTROL_HINT = (
    "[agent6] job control is unavailable here; Ctrl-C then /detach keeps it running.\n"
)


def _install_status_signal(
    state: dict[str, Any], session_facts: Callable[[], frontend.SessionFacts] | None
) -> Any:
    """Install the Ctrl-Z handler: print the run's state and stand down an unopened pause.

    It replaces SIGTSTP's default: a suspended agent's live provider stream is killed
    mid-response by the server, corrupting the run rather than pausing it.

    Args:
        state: The ladder's shared stage dict.
        session_facts: Reads the facts the status line prints.

    Returns:
        The previous handler, or None where the signal does not exist.
    """
    if not hasattr(signal, "SIGTSTP"):
        return None

    def _handler(_signum: int, _frame: Any) -> None:
        """Print the status and cancel a pause whose menu has not opened."""
        line = _status_suffix(session_facts).strip() or "no live facts for this execution"
        # An open menu stands on stage 1: its action is the next boundary's answer.
        if state["stage"] == 1 and not state["prompting"]:
            state["stage"] = 0
            tty_message(
                f"\n[agent6] {line}\n[agent6] pause cancelled; the run continues.\n"
                + _JOB_CONTROL_HINT
            )
        else:
            tty_message(f"\n[agent6] {line}\n" + _JOB_CONTROL_HINT)

    return signal.signal(signal.SIGTSTP, _handler)


def install_steer_sigint(  # noqa: C901, PLR0915  # a closure factory over one shared stage dict
    events: agent6_events.EventSink,
    session_dir: pathlib.Path,
    console_view: _console_view.ConsoleView | None = None,
    session_facts: Callable[[], frontend.SessionFacts] | None = None,
    btw_runner: _steer_menu.BtwRunner | None = None,
    config_path: pathlib.Path | None = None,
) -> steer.SteerState:
    """Install the escalating SIGINT handler and return the harness's steer callables.

    The first Ctrl-C pauses at the next boundary and emits `session.steer_requested`;
    the prompt is a front-end's modal when one is live, else the pause menu, else a
    plain prompt on the controlling terminal. The second interrupts the in-flight
    call and prompts now. The third, or one at the prompt itself, stops the run.
    Ctrl-Z prints the status and cancels an unopened pause (`_install_status_signal`).

    Args:
        events: Where the request is journaled.
        session_dir: The run's dir, holding the steer files.
        console_view: The live view, paused for the prompt.
        session_facts: Reads the facts the pause banner prints.
        btw_runner: Starts a btw for the menu.
        config_path: The invocation's `--config`.

    Returns:
        The steer state; its `restore` puts the previous handlers back.
    """
    state: dict[str, Any] = {"stage": 0, "prompting": False}

    def _handler(_signum: int, _frame: Any) -> None:
        """Climb one stage.

        Raises:
            KeyboardInterrupt: At the prompt, or past stage two: the run stops.
        """
        # A boundary can be a whole model response away, hence the escalation.
        if state["prompting"] or state["stage"] >= 2:
            raise KeyboardInterrupt
        if state["stage"] == 1:
            state["stage"] = 2
            if not ipc.frontend_is_live(session_dir):
                tty_message("\n[agent6] interrupting this step. Ctrl-C again to stop the run.\n")
            return
        state["stage"] = 1
        # A stale answer file would answer this prompt; one with a pending request is a live steer.
        if not ipc.steer_request_pending(session_dir):
            ipc.clear_steer_answer(session_dir)
        events.emit("session.steer_requested", source="sigint")
        # A live front-end prompts in its own modal; its terminal is not scribbled on.
        if not ipc.frontend_is_live(session_dir):
            tty_message(
                "\n[agent6] pausing after this step: Enter continues, type to steer,"
                " /stop ends it, /detach backgrounds it. Ctrl-C again to interrupt now.\n"
                + _status_suffix(session_facts)
            )

    previous = signal.signal(signal.SIGINT, _handler)
    previous_tstp = _install_status_signal(state, session_facts)

    def requested() -> bool:
        """Return whether a Ctrl-C or a front-end's request marker asks for a pause."""
        return state["stage"] >= 1 or ipc.steer_request_pending(session_dir)

    def interrupt() -> bool:
        """Return whether the in-flight call is to be aborted: a double Ctrl-C or `steer --now`."""
        return state["stage"] >= 2 or ipc.steer_interrupt_pending(session_dir)

    def clear() -> None:
        """Reset the stage and the steer files."""
        state["stage"] = 0
        ipc.clear_steer_answer(session_dir)
        ipc.clear_steer_request(session_dir)

    def prompt() -> str | None:
        """Return the steer: an answer already on disk, a front-end's, or the terminal's."""
        seeded = ipc.take_steer_answer(session_dir)
        if seeded is not None:
            return seeded
        if ipc.frontend_is_live(session_dir):
            answer = ipc.read_steer_answer(session_dir)
            # An abandoned modal: a persisting request would block again at the next boundary.
            if answer is None:
                state["stage"] = 0
                ipc.clear_steer_request(session_dir)
            return answer
        return _menu()

    def _menu() -> str | None:
        """Return the terminal's answer: the menu where termios owns the line, else a plain one."""
        pause = console_view.pause if console_view is not None else contextlib.nullcontext
        state["prompting"] = True
        try:
            with pause():
                if _menu_input.menu_capable():
                    return _steer_menu.pause_menu(
                        session_dir, btw_runner=btw_runner, config_path=config_path
                    )
                typed = tty_prompt(
                    "[agent6] paused: [enter] continue · type to steer · /stop · /exit · /detach: ",
                    until=lambda: ipc.steer_answer_written(session_dir),
                )
                if typed is None and ipc.steer_answer_written(session_dir):
                    tty_message("[agent6] a steer arrived from a front-end; taking it\n")
                    return ipc.take_steer_answer(session_dir)
                return _steer_menu.pause_line(
                    typed, session_dir, btw_runner=btw_runner, config_path=config_path
                )
        finally:
            state["prompting"] = False

    def armed() -> bool:
        """Return whether a pause is armed."""
        return state["stage"] >= 1

    def prompt_now() -> None:
        """Open the menu right after an operator prompt's answer, which counts as a boundary.

        The action seeds the answer the next boundary consumes; an empty one disarms.
        """
        action = _menu()
        if action is None or not action.strip():
            state["stage"] = 0
            return
        # The request marker keeps the action alive across a Ctrl-Z and the next Ctrl-C.
        if not ipc.submit_steer(session_dir, action):
            state["stage"] = 0
            tty_message("[agent6] could not write the steer request\n")

    def restore() -> None:
        """Put the previous SIGINT and SIGTSTP handlers back."""
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGINT, previous)
        if previous_tstp is not None:
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGTSTP, previous_tstp)

    def reset_stage() -> None:
        """Disarm without touching the steer files."""
        state["stage"] = 0

    return steer.SteerState(
        requested=requested,
        clear=clear,
        prompt=prompt,
        restore=restore,
        abort_pending=lambda: ipc.steer_answer_is_abort(session_dir),
        interrupt=interrupt,
        reset_stage=reset_stage,
        armed=armed,
        prompt_now=prompt_now,
    )


def make_steer_state(
    events: agent6_events.EventSink,
    session_dir: pathlib.Path,
    console_view: _console_view.ConsoleView | None = None,
    session_facts: Callable[[], frontend.SessionFacts] | None = None,
    btw_runner: _steer_menu.BtwRunner | None = None,
    config_path: pathlib.Path | None = None,
) -> steer.SteerState:
    """Return the steer state: the SIGINT ladder with a controlling terminal, else the file bridge.

    Args:
        events: Where the request is journaled.
        session_dir: The run's dir.
        console_view: The live view, paused for the prompt.
        session_facts: Reads the facts the pause banner prints.
        btw_runner: Starts a btw for the menu.
        config_path: The invocation's `--config`.
    """
    try:
        with open("/dev/tty", encoding="utf-8"):  # noqa: PTH123
            pass
    except OSError:
        return steer.file_bridge_steer(session_dir)
    return install_steer_sigint(
        events, session_dir, console_view, session_facts, btw_runner, config_path
    )
