# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Operator interaction for a live run: the `run_command` approver, the
`ask_user` questioner, their /dev/tty fallbacks, and the detach away-mode
(deny / wait / spawn the background resume)."""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from agent6.events import EventSink
from agent6.tools.schema import UserQuestion
from agent6.ui.bridge.approval import (
    away_mode,
    frontend_is_live,
    read_answer,
    read_question_answers,
    session_allow_set,
    set_away_mode,
    set_session_allow,
    steer_answer_is_abort,
)
from agent6.ui.bridge.spawn import spawn_detached_resume
from agent6.ui.cli._ptk_reader import ask_navigate, on_tty, ptk_prompt, radio_select
from agent6.ui.cli._steer import (
    tty_message as _tty_message,
)
from agent6.ui.cli._steer import (
    tty_prompt as _tty_prompt,
)
from agent6.ui.cli.egress import EgressGuard

if TYPE_CHECKING:
    from agent6.ui.cli._bar import BarController
    from agent6.ui.cli._console_view import ConsoleView


def _pause(cv: ConsoleView | None) -> contextlib.AbstractContextManager[None]:
    """Pause the live console spinner around an interactive /dev/tty prompt so it
    cannot erase the question and the operator's keystrokes. No-op when headless
    (no ConsoleView: a TUI-bridged, detached, or piped run)."""
    return cv.pause() if cv is not None else contextlib.nullcontext()


def _prompt_terminal[T](
    bar: BarController | None, cv: ConsoleView | None, fn: Callable[[], T]
) -> T:
    """Run a blocking terminal prompt `fn`. In bar mode route it through the bar's
    main thread (with the bar suspended) so it owns the terminal; otherwise run it
    under a spinner pause. Keeps the questioner/approver agnostic of the live view."""
    if bar is not None:
        return bar.prompt(fn)
    with _pause(cv):
        return fn()


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
    """On detach with run_commands=ask, ask how approvals/questions should be handled
    while nothing is watching, and record it for the background run. Non-interactive
    (no tty) -> deny, the safe default."""
    if not sys.stdin.isatty():
        set_away_mode(run_dir, "deny")
        return
    print(
        "[agent6] Detaching with run_commands=ask -- nothing will be watching to approve.",
        file=sys.stderr,
    )
    ans = _tty_prompt(
        "  While away: [a]pprove all / [d]eny all / [w]ait for a reattached front-end? [d]: ",
        fall_back_to_stdin=False,
    )
    choice = (ans or "").strip().lower()
    if choice in {"a", "approve"}:
        set_session_allow(run_dir)  # reuse the session-allow marker
        print("  -> approving every run_command.", file=sys.stderr)
    elif choice in {"w", "wait"}:
        set_away_mode(run_dir, "wait")
        print("  -> waiting; reattach (agent6 watch / the TUI) to approve.", file=sys.stderr)
    else:
        set_away_mode(run_dir, "deny")
        print("  -> denying run_commands until you reattach.", file=sys.stderr)


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
    run_dir: Path,
    events: EventSink,
    console_view: ConsoleView | None = None,
    bar: BarController | None = None,
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
        # A detached run chose how to handle approvals while away (deny / wait).
        away = away_mode(run_dir)
        if away == "deny":
            events.emit("approval.answer", id=prompt_id, approved=False, source="away-deny")
            return False
        events.emit("approval.prompt", id=prompt_id, prompt=prompt)
        if away == "wait":  # block until a reattached front-end answers (or stops)
            reply = _wait_for_reply(
                run_dir, lambda: read_answer(run_dir, prompt_id, timeout_s=20.0, dead_grace_s=8.0)
            )
            approved = bool(reply)
            events.emit("approval.answer", id=prompt_id, approved=approved, source="away-wait")
            return approved
        approved: bool | None = None
        source = "stdin"
        if frontend_is_live(run_dir):
            # A front-end that chose "allow session" set the marker itself, so this
            # answer is just yes; the check above auto-passes every later prompt.
            approved = read_answer(run_dir, prompt_id)
            if approved is not None:
                source = "tui"
        if approved is None:
            answer = _prompt_terminal(bar, console_view, lambda: default_stdin_approver(prompt))
            if answer == "session":
                set_session_allow(run_dir)
            approved = answer != "no"
        events.emit("approval.answer", id=prompt_id, approved=approved, source=source)
        return approved

    return approve


def spawn_detached(guard: EgressGuard, cwd: Path, run_id: str) -> str:
    """Spawn the detached background resume for this run.

    Under network isolation a direct spawn inherits the empty namespace and the
    resume's provider egress is dead on arrival, so it goes through the
    pre-forked host spawner; without isolation the direct spawn is fine."""
    if guard.detach_spawner is not None:
        return guard.detach_spawner.spawn_resume(cwd, run_id)
    return spawn_detached_resume(cwd, run_id)


def build_questioner(
    run_dir: Path,
    events: EventSink,
    console_view: ConsoleView | None = None,
    bar: BarController | None = None,
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
        events.emit(
            "question.prompt",
            id=question_id,
            questions=[{"question": q.question, "options": list(q.options)} for q in questions],
        )
        answers: tuple[str, ...] | None = None
        source = "stdin"
        if away_mode(run_dir) == "wait":
            # Detached 'wait': block until a reattached front-end answers (or stops).
            reply = _wait_for_reply(
                run_dir,
                lambda: read_question_answers(
                    run_dir, question_id, timeout_s=20.0, dead_grace_s=8.0
                ),
            )
            answers = reply if isinstance(reply, tuple) else tuple("" for _ in questions)
            source = "away-wait"
        elif frontend_is_live(run_dir):
            answers = read_question_answers(run_dir, question_id)
            if answers is not None:
                source = "tui"
        if answers is None:
            stdin_answers = _prompt_terminal(
                bar, console_view, lambda: default_stdin_questioner(questions)
            )
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
    """Prompt one question. On a tty the prompt_toolkit widgets own it: options are
    an inline arrow-key radio (always with a "type your own answer" free-text escape),
    free text is a line editor, and a cancel (esc / Ctrl-C, including cancelling the
    free-text) returns None so the caller (e.g. the navigator) treats it as
    unanswered rather than re-asking. Off a tty it falls back to a numbered / plain
    /dev/tty prompt. None means no answer (cancelled / headless)."""
    if on_tty():
        if q.options:
            picked = radio_select(q.question, q.options, prefix=prefix)
            if picked is not None:
                _tty_message(f"{prefix}{q.question} -> {picked}\n")  # land it in scrollback
            return picked  # None == cancelled: a real cancel, not a fallthrough to re-ask
        typed = ptk_prompt(f"{prefix}{q.question}\n> ")
        return typed.strip() if typed is not None else None
    # No tty for the widgets (e.g. a TUI stream redirect): the numbered /dev/tty prompt.
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
    """Ask a question series on the terminal. A single question is a radio (always
    with a "type your own answer" escape) or a line editor. A series uses the inline
    forward/back navigator (``ask_navigate``): move between questions freely, answer
    in any order, go back to change one, then submit. Without a tty for the navigator
    (e.g. a TUI stream redirect) it falls back to sequential /dev/tty prompts + a
    numbered review; fully headless returns None so the caller answers empty rather
    than hanging or eating piped stdin."""
    if len(questions) <= 1:
        # A single question (or, degenerately, none): ask it directly, no navigator.
        single: list[str] = []
        for q in questions:
            ans = ask_one_stdin(q)
            if ans is None:
                return None
            single.append(ans)
        return tuple(single)
    # Multiple on a tty: the inline forward/back navigator, answering each question
    # through the same radio / line editor (with its "type your own" escape).
    navigated = ask_navigate([q.question for q in questions], lambda i: ask_one_stdin(questions[i]))
    if navigated is not None:
        return tuple(navigated)
    # No tty for the navigator: sequential /dev/tty asks, then a numbered review; a
    # fully headless run returns None on the first ask.
    answers: list[str] = []
    for i, q in enumerate(questions, start=1):
        ans = ask_one_stdin(q, f"[{i}/{len(questions)}] ")
        if ans is None:
            return None  # no tty at all: never block
        answers.append(ans)
    while True:  # numbered review + revise loop; blank submits
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
