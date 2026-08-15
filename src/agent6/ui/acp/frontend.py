# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `SessionFrontend` an ACP client provides.

Every prompt the lifecycle raises becomes a `session/request_permission` to
the editor; everything a terminal front-end would draw becomes nothing, because
an ACP client renders from `session/update` instead.

A client that declared it cannot be asked is never asked: the answer comes from
the CAUTIOUS default rather than a hang or an invented yes. That is what
`FrontendCapabilities` is for, and why an editor with no way to show a prompt
still gets a working session -- one where the model simply has fewer powers.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.app.frontend import FrontendCapabilities, SessionFacts, SessionFrontend, SteerHooks
from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.events import EventSink
from agent6.sessions.layout import SessionLayout
from agent6.tools.schema import UserQuestion
from agent6.types import IsolationLevel
from agent6.workflows.loop import SessionResult, Workflow

# What the client is asked, and what an unaskable client is assumed to have
# said. Every one of these is the CAUTIOUS answer: a session that cannot ask
# is a session that does less, never one that does something unwatched.
# (prompt, options, standing) -> the chosen option, or None for no answer.
# `standing` is None for a QUESTION, whose options the model wrote: an
# answer among several is not a permission, and must never be offered as
# one the editor may remember.
Asker = Callable[[str, tuple[str, ...], bool | None], str | None]


def _false() -> bool:
    return False


def _nothing() -> None:
    return None


@dataclass(slots=True)
class _NoSteer:
    """No pause menu: ACP steers by prompting into a live session instead.

    `SteerHooks` is a Protocol of callable ATTRIBUTES, so this holds fields
    rather than defining methods."""

    requested: Callable[[], bool] = _false
    clear: Callable[[], None] = _nothing
    prompt: Callable[[], str | None] = _nothing
    restore: Callable[[], None] = _nothing
    abort_pending: Callable[[], bool] = _false
    interrupt: Callable[[], bool] = _false
    reset_stage: Callable[[], None] = _nothing


def acp_frontend(
    *,
    ask: Asker,
    capabilities: FrontendCapabilities,
    agent6_exe: Callable[[], str],
    spawn_detached_resume: Callable[[Path, str], str],
) -> SessionFrontend:
    """Wire the lifecycle to one ACP client."""

    def _approve(prompt: str, /, *, scope: str | None = None) -> bool:
        if not capabilities.can_ask:
            return False  # nobody to ask, so the answer is no
        # No scope means an "always allow" the editor remembers must NOT cover
        # this one -- the fetch tool's off-list host, where a GET can carry data
        # out in its path. The option names carry it, because an editor that
        # offers "always" needs something to key that decision on.
        standing = scope is not None
        options = ("allow", "deny") if standing else ("allow once", "deny")
        answer = ask(prompt, options, standing)
        return bool(answer) and answer.startswith("allow")

    def _questioner(questions: tuple[UserQuestion, ...]) -> tuple[str, ...]:
        answers: list[str] = []
        for question in questions:
            reply = ask(question.question, question.options, None) if capabilities.can_ask else None
            # An unanswered question becomes an empty string, which the loop
            # already treats as "the operator said nothing", not as a value.
            answers.append(reply or "")
        return tuple(answers)

    def _confirm_unconfined(isolation: IsolationLevel, cfg: Config) -> bool:
        """Only ask when it is actually true.

        The lifecycle calls this on EVERY run; the "is this dangerous" test
        lives in the answer, not the call. Asking regardless told the editor a
        confined run was unsandboxed -- a false statement about the run, on the
        one approval that must never become reflexive.
        """
        if isolation != "none" or cfg.sandbox.run_commands != "yes":
            return True
        # No scope: docs/security.md documents this as a ONE-TIME gate, and
        # ACP's `allow_always` is exactly the button that would let one click
        # silence it for every later session.
        return _approve("Run commands UNSANDBOXED on this host, with no per-command prompt?")

    def _steer(
        _events: EventSink, _session_dir: Path, _facts: Callable[[], SessionFacts]
    ) -> SteerHooks:
        return _NoSteer()

    def _no_repl(
        _session_dir: Path, _budget: BudgetTracker, _task: str, _mcp: object
    ) -> Callable[[int, str], Literal["continue", "stop"]]:
        # ACP has its own turn loop; an interactive REPL inside it would be a
        # second one, with two things reading the same stdin. The hook exists
        # and always continues.
        return lambda _iteration, _summary: "continue"

    def _no_ask_repl(
        _wf: Workflow, _budget: BudgetTracker, _layout: SessionLayout, _task: str
    ) -> SessionResult:
        raise RuntimeError("an ACP session drives its own turns; the ask REPL is not used")

    return SessionFrontend(
        capabilities=capabilities,
        should_spawn_tui=lambda _tui, _interactive, _mode: False,
        # Stream the deltas as events (session/update reads them) without
        # echoing to a console nobody is watching.
        stream_modes=lambda _tui_enabled: (True, False),
        attach_console_view=lambda _events: None,
        close_console_view=lambda: None,
        loop_logger=lambda _mode: lambda _line: None,
        tui_session=lambda _session_dir, _enabled: _nullcontext(),
        build_approver=lambda _session_dir, _events: _approve,
        build_questioner=lambda _session_dir, _events: _questioner,
        make_steer_state=_steer,
        confirm_unconfined_autorun=_confirm_unconfined,
        confirm_run_on_run_branch=lambda branch: _approve(
            f"Continue this run on {branch!r}, which is already a run branch?"
        ),
        confirm_replay_after_crash=lambda iteration, tools: _approve(
            f"The previous run died mid-turn (iteration {iteration};"
            f" {', '.join(tools) or 'unknown tools'}). Its tools may have partially"
            " applied; replaying can repeat a non-idempotent effect. Re-run the turn?"
        ),
        prompt_detach_away_mode=lambda _session_dir, _scopes: None,
        select_revised_prompt=lambda _original, _revised, _notes: None,
        build_repl_hook=_no_repl,
        run_ask_repl=_no_ask_repl,
        save_ask_transcript=lambda _layout, _question, _answer: None,
        build_coordinator_spawner=_no_coordinator,
        agent6_exe=agent6_exe,
        spawn_detached_resume=spawn_detached_resume,
    )


def _no_coordinator(
    _cfg: Config,
    _cwd: Path,
    _state_dir: Path,
    _mode: str,
    _session_id: str,
    _max_usd: float | None,
    _auto_approve: bool,
) -> None:
    """`/parallel` fans out sibling runs, which need somewhere to be watched.
    An ACP client renders ONE session; lanes would run invisibly."""
    return None


def _nullcontext() -> AbstractContextManager[None]:
    return nullcontext()
