# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Mid-run steering: a SIGINT handler that lets the operator pause the loop
and inject a one-shot instruction (or abort), plus interactive revised-prompt
selection. Independent of the run command; run.py wires it in.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent6.events import EventSink
from agent6.ui.approval import (
    clear_steer_answer,
    read_steer_answer,
    tui_is_live,
)


@dataclass
class SteerState:
    requested: Callable[[], bool]
    clear: Callable[[], None]
    prompt: Callable[[], str | None]
    restore: Callable[[], None]


def select_revised_prompt(
    original: str,
    revised: str,
    questions: tuple[str, ...],
) -> str | None:
    """Interactive accept/edit/skip prompt for workflow.revise_prompt."""
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
            editor = os.environ.get("EDITOR", "vi")
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
                result = subprocess.run([editor, str(tmp_path)], check=False)
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


def tty_prompt(text: str) -> str | None:
    """Prompt on the controlling terminal directly (see ``tty_message``).
    Falls back to stdin when /dev/tty is unavailable."""
    try:
        with open("/dev/tty", "r+", encoding="utf-8") as tty:  # noqa: PTH123
            tty.write(text)
            tty.flush()
            line = tty.readline()
            return line.rstrip("\n") if line else None
    except OSError:
        try:
            return input(text)
        except (EOFError, KeyboardInterrupt):
            return None


def install_steer_sigint(events: EventSink, run_dir: Path) -> SteerState:
    """Install a SIGINT handler that asks the workflow to steer.

    * 1st SIGINT, set the "steer requested" flag and emit
      ``run.steer_requested``. The workflow notices at its next safe boundary
      (between steps) and prompts: through a TUI modal when the TUI is live,
      otherwise on the controlling terminal (``/dev/tty``, so the prompt is
      visible even when the TUI has redirected the run's std streams to a log).
    * 2nd SIGINT within 2 s, raise KeyboardInterrupt to abort the run.

    Returns callables for the workflow plus a ``restore`` hook to put the
    previous handler back when the run is done.
    """
    state: dict[str, Any] = {"requested": False, "last_ts": 0.0}
    window_s = 2.0

    def _handler(_signum: int, _frame: Any) -> None:
        now = time.monotonic()
        if state["requested"]:
            # Second Ctrl-C aborts: within the 2s window always, or any time a
            # steer is still pending in TUI mode (no terminal feedback there to
            # re-arm against). Otherwise just refresh the timestamp, crucially
            # WITHOUT re-clearing the answer file, which the TUI may have just
            # written (re-clearing it would strand read_steer_answer for 600s).
            if (now - state["last_ts"]) < window_s or tui_is_live(run_dir):
                raise KeyboardInterrupt
            state["last_ts"] = now
            return
        state["requested"] = True
        state["last_ts"] = now
        clear_steer_answer(run_dir)
        events.emit("run.steer_requested", source="sigint")
        # With the TUI up, the steer prompt is a modal, don't scribble on the
        # terminal it owns. Otherwise tell the user a prompt is coming.
        if not tui_is_live(run_dir):
            tty_message(
                "\n[agent6] steer requested — finishing current step, then will"
                " prompt. Press Ctrl-C again to abort.\n"
            )

    previous = signal.signal(signal.SIGINT, _handler)

    def requested() -> bool:
        return bool(state["requested"])

    def clear() -> None:
        state["requested"] = False
        clear_steer_answer(run_dir)

    def prompt() -> str | None:
        # TUI live: the user answers a modal; read its file-bridge result.
        if tui_is_live(run_dir):
            return read_steer_answer(run_dir)
        return tty_prompt("[agent6] steer (blank=continue, 'abort'=stop, else=instruction): ")

    def restore() -> None:
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGINT, previous)

    return SteerState(requested=requested, clear=clear, prompt=prompt, restore=restore)


# Used when there is no controlling terminal at all (fully non-interactive):
# no SIGINT handler, default Ctrl-C behaviour.
NULL_STEER = SteerState(
    requested=lambda: False,
    clear=lambda: None,
    prompt=lambda: None,
    restore=lambda: None,
)


def make_steer_state(events: EventSink, run_dir: Path) -> SteerState:
    """Install the steer SIGINT handler when a controlling terminal exists
    (covers run/plan/ask with or without the TUI); else a no-op."""
    try:
        with open("/dev/tty", encoding="utf-8"):  # noqa: PTH123
            pass
    except OSError:
        return NULL_STEER
    return install_steer_sigint(events, run_dir)


# inline-resolved file references in user task strings.
#
# A token of the form `@PATH` that resolves to a regular file inside `root`
# is replaced with the file's contents wrapped in a `<file path=...>` block.
# Anything that doesn't match (missing files, paths that escape root, email
# addresses, decorators copied from code, etc.) is left untouched so the
# transformation never corrupts a hand-written task string.
