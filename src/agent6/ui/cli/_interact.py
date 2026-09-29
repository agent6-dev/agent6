# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Operator interaction for a live run: the approver, the questioner and the away-mode."""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from agent6.portable import has_controlling_tty
from agent6.sessions.ipc import (
    AWAY_MODES,
    AwayMode,
    answer_written,
    await_frontend_reply,
    away_mode,
    frontend_is_live,
    question_answers_written,
    read_answer,
    read_question_answers,
    record_answer,
    set_away_mode,
    set_session_allow,
)
from agent6.tools.operator_prompts import (
    ApprovalAnswer,
    ApprovalRequest,
    Approver,
    QuestionAnswer,
    Questioner,
    QuestionRequest,
    Source,
)
from agent6.tools.schema import UserQuestion
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._steer import (
    tty_message,
    tty_prompt,
)
from agent6.ui.keymap import answer_for, approval_prompt_suffix
from agent6.ui.steer import SteerState
from agent6.viewmodel import approval_parts
from agent6.viewmodel.transcript import scrub_terminal_controls


def _pause(cv: ConsoleView | None) -> contextlib.AbstractContextManager[None]:
    """Return a pause of the console's spinner around a prompt; a no-op without a view."""
    return cv.pause() if cv is not None else contextlib.nullcontext()


def lane_away_mode() -> AwayMode:
    """Return the away-mode a fan-out's lanes run with.

    The coordinator's own marker when a hub set one, else `wait` with a terminal to
    attach from, else `deny`.
    """
    marker = os.environ.get("AGENT6_DETACHED_AWAY", "")
    if marker in AWAY_MODES:
        return marker
    return "wait" if has_controlling_tty() else "deny"


def default_stdin_approver(
    prompt: str, *, standing: bool = True, until: Callable[[], bool] | None = None
) -> str | None:
    """Ask for a tool approval on /dev/tty, the fallback when no front-end answers.

    The payload renders on its own indented lines, the answer line dim below it.

    Args:
        prompt: `Allow <tool>: <payload>`.
        standing: The gate offers the two session answers; `fetch` has none to give.
        until: Ends the wait early once it holds (the answer arrived by another route).

    Returns:
        "yes", "no", "session" or "session-deny"; None when nothing was typed.
    """
    suffix = approval_prompt_suffix(standing=standing)
    bold, dim, yellow, reset = "\033[1m", "\033[2m", "\033[33m", "\033[0m"
    # The text under judgment carries no sequence at all: conceal (SGR 8) hides what it wraps.
    head, payload = approval_parts(scrub_terminal_controls(prompt))
    if payload:
        body = "\n".join(f"    {ln}" for ln in payload.splitlines())
        rendered = (
            f"{bold}{yellow}?{reset} {bold}{head}:{reset}\n\n{body}\n\n  {dim}{suffix}{reset}"
        )
        plain = f"? {head}:\n\n{body}\n\n  {suffix}"
    else:
        rendered = plain = f"{prompt} {suffix}"
    # The stdin fallback prints to stdout, which may be a pipe, so it gets no escapes.
    ans = tty_prompt(rendered, plain=plain, until=until)
    if ans is None:
        return None
    return answer_for(ans, standing=standing)


def prompt_detach_away_mode(session_dir: Path, scopes: tuple[str, ...]) -> None:
    """Ask, on detach, how prompts are handled while nothing watches, and record the answer.

    The default is wait: a deny throws the run's work away, wait pauses at the prompt
    and is resumable. Without a controlling terminal it is wait.

    Args:
        session_dir: The run's dir.
        scopes: Every scope in play; "approve all" grants each, since one blocked
            prompt from another would stall the run.
    """
    if not has_controlling_tty():
        set_away_mode(session_dir, "wait")
        return
    print(
        "[agent6] Detaching with run_commands=ask; nothing will be watching to approve.",
        file=sys.stderr,
    )
    ans = tty_prompt(
        "  While away: [w]ait for a reattached front-end / [a]pprove all / [d]eny all? [w]: ",
        fall_back_to_stdin=False,
    )
    choice = (ans or "").strip().lower()
    if choice in {"a", "approve"}:
        for scope in scopes:
            set_session_allow(session_dir, scope)
        covered = "run_command and MCP tool call" if len(scopes) > 1 else "run_command"
        print(f"  -> approving every {covered}.", file=sys.stderr)
    elif choice in {"d", "deny"}:
        set_away_mode(session_dir, "deny")
        print("  -> denying run_commands until you reattach.", file=sys.stderr)
    else:
        set_away_mode(session_dir, "wait")
        print("  -> waiting; reattach (agent6 attach / the TUI) to approve.", file=sys.stderr)


def build_approver(
    session_dir: Path,
    console_cell: Sequence[ConsoleView | None] | None = None,
    steer_cell: Sequence[SteerState | None] | None = None,
) -> Approver:
    """Build the command approver, bridged to a live front-end when one is attached.

    A live front-end answers through the file bridge; otherwise the terminal prompt
    asks, reading the same file while it waits, so an answer by another route
    lands too.

    Args:
        session_dir: The run's dir.
        console_cell: Holds the console view once it exists; paused around the prompt.
        steer_cell: Holds the steer state once it exists; an armed pause opens its
            menu right after the answer.

    Returns:
        The approver.
    """

    def approve(request: ApprovalRequest, /) -> ApprovalAnswer:
        """Return one approval's answer: the front-end's, the away-mode's, or the terminal's."""
        # A live front-end is always asked; away-mode governs only when nothing is attached.
        if frontend_is_live(session_dir):
            answer = read_answer(session_dir, request.id)
            if answer is not None:
                return ApprovalAnswer(record_answer(session_dir, answer, request.scope), "frontend")
        # Nothing attached: the detached run's away-mode governs.
        away = away_mode(session_dir)
        if away == "deny":
            return ApprovalAnswer(False, "away-deny")
        wait_for_frontend = away == "wait" or not has_controlling_tty()
        if wait_for_frontend:
            # Block until a front-end answers; a deny would discard the run's work.
            tty_message(
                f"[agent6] waiting: an approval awaits a front-end; answer it with:"
                f" agent6 attach {session_dir.name}\n"
            )
            reply = await_frontend_reply(
                session_dir,
                lambda: read_answer(session_dir, request.id, timeout_s=20.0, dead_grace_s=8.0),
            )
            approved = reply is not None and record_answer(session_dir, reply, request.scope)
            return ApprovalAnswer(approved, "await-frontend")
        steer = steer_cell[0] if steer_cell else None
        # The view is attached after the prompts are built, so it is read now.
        with _pause(console_cell[0] if console_cell else None):
            if steer is not None and steer.armed():
                tty_message("\n[agent6] pause armed: the menu opens after this answer.\n")
            answer_s = default_stdin_approver(
                request.prompt,
                standing=bool(request.scope),
                until=lambda: answer_written(session_dir, request.id),
            )
        source: Source = "stdin"
        if answer_s is None:
            filed = read_answer(session_dir, request.id, timeout_s=0.0)
            if filed is None:
                answer_s = "no"
            else:
                tty_message("[agent6] answered elsewhere.\n")
                answer_s, source = filed, "frontend"
        # A session choice persists across resumes; session-deny withdraws the scope's tools.
        approved = record_answer(session_dir, answer_s, request.scope)
        if steer is not None and steer.armed():
            steer.prompt_now()
        return ApprovalAnswer(approved, source)

    return approve


def build_questioner(
    session_dir: Path, console_cell: Sequence[ConsoleView | None] | None = None
) -> Questioner:
    """Build the `ask_user` questioner, bridged to a live front-end when one is attached.

    Like `build_approver`; a headless run gets empty answers rather than hanging.

    Args:
        session_dir: The run's dir.
        console_cell: Holds the console view once it exists; paused around the prompt.

    Returns:
        The questioner.
    """

    def ask(request: QuestionRequest, /) -> QuestionAnswer:
        """Return one question prompt's answers: the front-end's, the away-mode's, or the tty's."""
        questions = request.questions
        # A live front-end is always asked; away-mode is the fallback without one.
        if frontend_is_live(session_dir):
            answers = read_question_answers(session_dir, request.id)
            if answers is not None:
                return QuestionAnswer(answers, "frontend")
        if away_mode(session_dir) == "wait":
            # Block until a front-end answers.
            tty_message(
                f"[agent6] waiting: a question awaits a front-end; answer it with:"
                f" agent6 attach {session_dir.name}"
                f" (or, with no terminal: agent6 answer {session_dir.name} TEXT)\n"
            )
            reply = await_frontend_reply(
                session_dir,
                lambda: read_question_answers(
                    session_dir, request.id, timeout_s=20.0, dead_grace_s=8.0
                ),
            )
            if isinstance(reply, tuple):
                return QuestionAnswer(reply, "frontend")
            return QuestionAnswer(tuple("" for _ in questions), "away-wait", unseen=True)
        with _pause(console_cell[0] if console_cell else None):
            stdin_answers = default_stdin_questioner(
                questions, until=lambda: question_answers_written(session_dir, request.id)
            )
        if stdin_answers is None:
            filed = read_question_answers(session_dir, request.id, timeout_s=0.0)
            if filed is not None:
                tty_message("[agent6] answered elsewhere.\n")
                return QuestionAnswer(filed, "frontend")
            # Nobody saw the question: answer empty so the run never hangs, and say so.
            tty_message(
                "[agent6] no front-end attached and no terminal to answer the"
                " question; returning empty answers\n"
            )
            return QuestionAnswer(tuple("" for _ in questions), "headless-default", unseen=True)
        return QuestionAnswer(stdin_answers, "stdin")

    return ask


def ask_one_stdin(
    q: UserQuestion, prefix: str = "", until: Callable[[], bool] | None = None
) -> str | None:
    """Ask one question on /dev/tty; a digit picks an option, else the text is the answer.

    Args:
        q: The question.
        prefix: The number in a series.
        until: Ends the wait early once it holds.

    Returns:
        The answer; None without a terminal or once `until` held.
    """
    lines = [
        f"{prefix}{q.question}",
        *(f"  {i}) {opt}" for i, opt in enumerate(q.options, start=1)),
    ]
    ans = tty_prompt("\n".join(lines) + "\n> ", fall_back_to_stdin=False, until=until)
    if ans is None:
        return None
    ans = ans.strip()
    if ans.isdigit() and 1 <= int(ans) <= len(q.options):
        return q.options[int(ans) - 1]
    return ans


def default_stdin_questioner(
    questions: tuple[UserQuestion, ...], until: Callable[[], bool] | None = None
) -> tuple[str, ...] | None:
    """Ask each question on /dev/tty, then let the operator revise any answer in a series.

    Args:
        questions: The questions.
        until: Ends the wait early once it holds.

    Returns:
        The answers; None without a controlling terminal or once `until` held.
    """
    answers: list[str] = []
    multi = len(questions) > 1
    for i, q in enumerate(questions, start=1):
        prefix = f"[{i}/{len(questions)}] " if multi else ""
        ans = ask_one_stdin(q, prefix, until)
        if ans is None:
            return None
        answers.append(ans)
    while multi:  # blank submits
        summary = "\n".join(
            f"  {n}) {q.question} -> {a or '(empty)'}"
            for n, (q, a) in enumerate(zip(questions, answers, strict=True), start=1)
        )
        pick = tty_prompt(
            f"Review:\n{summary}\nEnter to submit, or a number to change that answer: ",
            fall_back_to_stdin=False,
            until=until,
        )
        if pick is None:
            if until is not None and until():
                return None
            break
        if not pick.strip():
            break
        if pick.strip().isdigit() and 1 <= int(pick.strip()) <= len(questions):
            j = int(pick.strip()) - 1
            revised = ask_one_stdin(questions[j], until=until)
            if revised is None:
                return None
            answers[j] = revised
    return tuple(answers)
