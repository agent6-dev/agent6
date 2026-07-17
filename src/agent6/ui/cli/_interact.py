# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Operator interaction for a live run: the `run_command` approver, the
`ask_user` questioner, their /dev/tty fallbacks, and the detach away-mode
(deny / wait / spawn the background resume)."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from agent6.events import EventSink
from agent6.runs.ipc import (
    away_mode,
    clear_answer,
    clear_question_answers,
    frontend_is_live,
    read_answer,
    read_question_answers,
    session_allow_set,
    set_away_mode,
    set_session_allow,
    steer_answer_is_abort,
)
from agent6.tools.schema import UserQuestion
from agent6.ui.cli._steer import (
    tty_message as _tty_message,
)
from agent6.ui.cli._steer import (
    tty_prompt as _tty_prompt,
)

if TYPE_CHECKING:
    from agent6.ui.cli._console_view import ConsoleView


def _pause(cv: ConsoleView | None) -> contextlib.AbstractContextManager[None]:
    """Pause the live console spinner around an interactive /dev/tty prompt so it
    cannot erase the question and the operator's keystrokes. No-op when headless
    (no ConsoleView: a TUI-bridged, detached, or piped run)."""
    return cv.pause() if cv is not None else contextlib.nullcontext()


def _has_controlling_tty() -> bool:
    """True iff a controlling terminal exists (so the stdin approver can actually
    prompt). A foreground run has one; a web/hub-spawned or fully headless run
    does not, and there falls back to waiting for a front-end rather than a
    no-terminal deny."""
    try:
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError:
        return False
    os.close(fd)
    return True


def default_stdin_approver(prompt: str) -> str:
    """Plain-terminal fallback for tool approval (no live TUI, or its answer
    timed out). Returns "yes", "no", or "session" (allow all for the rest of this
    run). Routed via /dev/tty so the prompt stays visible when a TUI has redirected
    the std streams to its console log; plain stdin without one."""
    ans = _tty_prompt(f"{prompt} [y/N/a]  (a = allow all this session): ")
    if ans is None:
        return "no"
    ans = ans.strip().lower()
    if ans in {"a", "all", "always", "session"}:
        return "session"
    return "yes" if ans in {"y", "yes"} else "no"


def prompt_detach_away_mode(run_dir: Path) -> None:
    """On detach with run_commands=ask, ask how approvals/questions should be
    handled while nothing is watching, and record it for the background run.

    The default is WAIT: a deny throws away the run's work (the model's commands
    are refused and it flails, burning tokens for nothing), while wait pauses
    cleanly at the approval and is resumable -- re-attach with `agent6 attach`
    and answer. Non-interactive (no tty) also defaults to wait."""
    if not sys.stdin.isatty():
        set_away_mode(run_dir, "wait")
        return
    print(
        "[agent6] Detaching with run_commands=ask -- nothing will be watching to approve.",
        file=sys.stderr,
    )
    ans = _tty_prompt(
        "  While away: [w]ait for a reattached front-end / [a]pprove all / [d]eny all? [w]: ",
        fall_back_to_stdin=False,
    )
    choice = (ans or "").strip().lower()
    if choice in {"a", "approve"}:
        set_session_allow(run_dir)  # reuse the session-allow marker
        print("  -> approving every run_command.", file=sys.stderr)
    elif choice in {"d", "deny"}:
        set_away_mode(run_dir, "deny")
        print("  -> denying run_commands until you reattach.", file=sys.stderr)
    else:
        set_away_mode(run_dir, "wait")
        print("  -> waiting; reattach (agent6 attach / the TUI) to approve.", file=sys.stderr)


def _wait_for_reply(run_dir: Path, read_once: Callable[[], object | None]) -> object | None:
    """Detach 'wait' mode: block until a reattached front-end supplies an answer
    (``read_once`` returns non-None) or stops the run. ``read_once`` polls a live
    front-end for up to its own (short) timeout; between calls, when no front-end is
    attached yet, we poll for one. A reattached front-end's Stop lands as a steer
    abort, which breaks the wait so the run can end. Returns the reply, or None on
    stop."""
    while True:
        if steer_answer_is_abort(run_dir):
            return None
        if frontend_is_live(run_dir):
            reply = read_once()
            if reply is not None:
                return reply
        else:
            time.sleep(1.0)  # no front-end yet; poll for one to reattach


def build_approver(
    run_dir: Path, events: EventSink, console_view: ConsoleView | None = None
) -> Callable[[str], bool]:
    """Build the `run_command` approver, bridged to a live TUI when present.

    Emits an `approval.prompt` event; if a TUI is live (it wrote `frontend.pid`) the
    answer comes from its Allow/Deny modal via the file bridge
    (`approvals/<id>.answer`), otherwise -- or if the TUI dies / times out -- it
    falls back to the stdin `[y/N]` prompt. Emits `approval.answer` either way.
    This is what actually wires the watch/auto-spawn TUI to run_command approval
    (previously the modal's answer was written but never read)."""
    counter = {"n": 0}

    def approve(prompt: str) -> bool:
        counter["n"] += 1
        prompt_id = f"approval-{counter['n']}"
        # Already granted for the session (this run + its resumes) -> auto-pass.
        if session_allow_set(run_dir):
            events.emit("approval.answer", id=prompt_id, approved=True, source="session")
            return True
        # Clear any premature answer for this id, then emit the prompt so ANY live
        # front-end (a re-attached `agent6 attach`, the TUI, the web) can render and
        # answer it. clear_answer stops a pre-written answer (a premature approve
        # POST, ids being predictable) from silently auto-passing.
        clear_answer(run_dir, prompt_id)
        events.emit("approval.prompt", id=prompt_id, prompt=prompt)
        # A live front-end ALWAYS gets asked, in its own UI, regardless of the
        # detach away-mode: away-mode governs only the window when nothing is
        # attached. (A foreground run writes no frontend.pid, so it falls through
        # to the stdin prompt below.)
        if frontend_is_live(run_dir):
            answer = read_answer(run_dir, prompt_id)  # the front-end wrote "allow session" itself
            if answer is not None:
                events.emit("approval.answer", id=prompt_id, approved=answer, source="frontend")
                return answer
        # Nothing attached (or the front-end died mid-prompt): the detached run's
        # chosen away-mode governs. deny/wait are only reached headless.
        away = away_mode(run_dir)
        if away == "deny":
            events.emit("approval.answer", id=prompt_id, approved=False, source="away-deny")
            return False
        wait_for_frontend = away == "wait" or not _has_controlling_tty()
        if wait_for_frontend:
            # away="wait", OR an unattended run with no away-mode and no terminal
            # (a web/hub-spawned run whose viewers have all left): block until a
            # front-end attaches and answers, rather than deny. Deny discards the
            # run's work; wait pauses cleanly and is resumable (the default).
            reply = _wait_for_reply(
                run_dir, lambda: read_answer(run_dir, prompt_id, timeout_s=20.0, dead_grace_s=8.0)
            )
            approved = bool(reply)
            events.emit("approval.answer", id=prompt_id, approved=approved, source="await-frontend")
            return approved
        # Foreground (a controlling tty, no away-mode): prompt on it directly.
        with _pause(console_view):
            answer_s = default_stdin_approver(prompt)
        if answer_s == "session":
            set_session_allow(run_dir)
        approved = answer_s != "no"
        events.emit("approval.answer", id=prompt_id, approved=approved, source="stdin")
        return approved

    return approve


def build_questioner(
    run_dir: Path, events: EventSink, console_view: ConsoleView | None = None
) -> Callable[[tuple[UserQuestion, ...]], tuple[str, ...]]:
    """Build the `ask_user` questioner, bridged to a live TUI when present.

    Emits a `question.prompt` event; if a TUI is live the answer comes from its
    question modal via `questions/<id>.answer`, otherwise (or if the TUI dies /
    times out) it falls back to a numbered stdin prompt. A headless run (no TUI,
    no TTY) gets an empty answer rather than hanging. Emits `question.answer`."""
    counter = {"n": 0}

    def ask(questions: tuple[UserQuestion, ...]) -> tuple[str, ...]:
        counter["n"] += 1
        question_id = f"question-{counter['n']}"
        # Clear a pre-written answer before emitting (see build_approver): an
        # ask_user answer that arrived before the prompt must not be consumed.
        clear_question_answers(run_dir, question_id)
        events.emit(
            "question.prompt",
            id=question_id,
            questions=[{"question": q.question, "options": list(q.options)} for q in questions],
        )
        answers: tuple[str, ...] | None = None
        source = "stdin"
        # A live front-end (re-attached CLI watch, TUI, web) always gets asked,
        # whatever the away-mode; away-mode is the no-front-end fallback.
        if frontend_is_live(run_dir):
            answers = read_question_answers(run_dir, question_id)
            if answers is not None:
                source = "frontend"
        if answers is None and away_mode(run_dir) == "wait":
            # Detached 'wait', nothing attached: block until a front-end answers.
            reply = _wait_for_reply(
                run_dir,
                lambda: read_question_answers(
                    run_dir, question_id, timeout_s=20.0, dead_grace_s=8.0
                ),
            )
            answers = reply if isinstance(reply, tuple) else tuple("" for _ in questions)
            source = "away-wait"
        if answers is None:
            with _pause(console_view):
                stdin_answers = default_stdin_questioner(questions)
                if stdin_answers is None:
                    # No front-end and no controlling terminal: nobody saw the
                    # question. Answer empty so the run never hangs, and say so
                    # where a watcher will see it instead of failing silently.
                    answers = tuple("" for _ in questions)
                    source = "headless-default"
                    _tty_message(
                        "[agent6] ask_user: no front-end attached and no terminal;"
                        " returning empty answers\n"
                    )
                else:
                    answers = stdin_answers
        events.emit("question.answer", id=question_id, answers=list(answers), source=source)
        return answers

    return ask


def ask_one_stdin(q: UserQuestion, prefix: str = "") -> str | None:
    """Prompt one question on /dev/tty; a digit picks an option, else free text.
    None means no terminal (headless)."""
    lines = [
        f"{prefix}{q.question}",
        *(f"  {i}) {opt}" for i, opt in enumerate(q.options, start=1)),
    ]
    ans = _tty_prompt("\n".join(lines) + "\n> ", fall_back_to_stdin=False)
    if ans is None:
        return None
    ans = ans.strip()
    if ans.isdigit() and 1 <= int(ans) <= len(q.options):
        return q.options[int(ans) - 1]
    return ans


def default_stdin_questioner(questions: tuple[UserQuestion, ...]) -> tuple[str, ...] | None:
    """Ask each question on /dev/tty (visible under a TUI's stream redirect). For a
    series, print a summary afterwards and let the operator revise any answer (type
    its number) before submitting (blank). Returns None without a controlling
    terminal (headless) so the caller can answer empty -- never hanging or eating
    piped stdin -- and say so."""
    answers: list[str] = []
    multi = len(questions) > 1
    for i, q in enumerate(questions, start=1):
        prefix = f"[{i}/{len(questions)}] " if multi else ""
        ans = ask_one_stdin(q, prefix)
        if ans is None:
            return None  # no tty: never block
        answers.append(ans)
    while multi:  # review + revise loop; blank submits
        summary = "\n".join(
            f"  {n}) {q.question} -> {a or '(empty)'}"
            for n, (q, a) in enumerate(zip(questions, answers, strict=True), start=1)
        )
        pick = _tty_prompt(
            f"Review:\n{summary}\nEnter to submit, or a number to change that answer: ",
            fall_back_to_stdin=False,
        )
        if pick is None or not pick.strip():
            break
        if pick.strip().isdigit() and 1 <= int(pick.strip()) <= len(questions):
            j = int(pick.strip()) - 1
            revised = ask_one_stdin(questions[j])
            if revised is not None:
                answers[j] = revised
    return tuple(answers)
