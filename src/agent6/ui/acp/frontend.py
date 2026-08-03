# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `RunFrontend` an ACP client provides.

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

from agent6.app.preflight import BranchChoice
from agent6.app.run import FrontendCapabilities, RunFacts, RunFrontend, SteerHooks
from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.events import EventSink
from agent6.runs.layout import RunLayout
from agent6.tools.schema import UserQuestion
from agent6.types import IsolationLevel
from agent6.workflows.loop import RunResult, Workflow

# What the client is asked, and what an unaskable client is assumed to have
# said. Every one of these is the CAUTIOUS answer: a session that cannot ask
# is a session that does less, never one that does something unwatched.
Asker = Callable[[str, tuple[str, ...]], str | None]


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
) -> RunFrontend:
    """Wire the lifecycle to one ACP client."""

    def _approve(prompt: str, /, *, standing: bool = True) -> bool:
        if not capabilities.can_ask:
            return False  # nobody to ask, so the answer is no
        # `standing=False` means an "always allow" the editor remembers must
        # NOT cover this one -- the fetch tool's off-list host, where a GET can
        # carry data out in its path. The option names carry it, because an
        # editor that offers "always" needs something to key that decision on.
        options = ("allow", "deny") if standing else ("allow once", "deny")
        answer = ask(prompt, options)
        return bool(answer) and answer.startswith("allow")

    def _questioner(questions: tuple[UserQuestion, ...]) -> tuple[str, ...]:
        answers: list[str] = []
        for question in questions:
            reply = ask(question.question, question.options) if capabilities.can_ask else None
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
        return _approve("Run commands UNSANDBOXED on this host, with no per-command prompt?")

    def _branch_choice(cfg: Config, _cwd: Path, base: str) -> BranchChoice:
        """`git.branch_from`, honoured rather than discarded.

        `None` means "stack on whatever is checked out", which for
        `branch_from = "base"` is the opposite of what the operator configured
        -- and can carry another run's unmerged branch into this run's diff.
        With no terminal to ask on, `ask` takes the CLI's own headless
        fallback: the clean base.
        """
        return BranchChoice(start_point=None if cfg.git.branch_from == "current" else base)

    def _steer(_events: EventSink, _run_dir: Path, _facts: Callable[[], RunFacts]) -> SteerHooks:
        return _NoSteer()

    def _no_repl(
        _run_dir: Path, _budget: BudgetTracker, _task: str, _mcp: object
    ) -> Callable[[int, str], Literal["continue", "stop"]]:
        # ACP has its own turn loop; an interactive REPL inside it would be a
        # second one, with two things reading the same stdin. The hook exists
        # and always continues.
        return lambda _iteration, _summary: "continue"

    def _no_ask_repl(
        _wf: Workflow, _budget: BudgetTracker, _layout: RunLayout, _task: str
    ) -> RunResult:
        raise RuntimeError("an ACP session drives its own turns; the ask REPL is not used")

    return RunFrontend(
        capabilities=capabilities,
        should_spawn_tui=lambda _tui, _interactive, _mode: False,
        # Stream the deltas as events (session/update reads them) without
        # echoing to a console nobody is watching.
        stream_modes=lambda _tui_enabled: (True, False),
        attach_console_view=lambda _events: None,
        close_console_view=lambda: None,
        loop_logger=lambda _mode: lambda _line: None,
        tui_session=lambda _run_dir, _enabled: _nullcontext(),
        build_approver=lambda _run_dir, _events: _approve,
        build_questioner=lambda _run_dir, _events: _questioner,
        make_steer_state=_steer,
        confirm_unconfined_autorun=_confirm_unconfined,
        confirm_run_on_run_branch=lambda branch: _approve(
            f"Continue this run on {branch!r}, which is already a run branch?"
        ),
        choose_branch_start_point=_branch_choice,
        prompt_detach_away_mode=lambda _run_dir: None,
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
    _run_id: str,
    _max_usd: float | None,
    _auto_approve: bool,
) -> None:
    """`/parallel` fans out sibling runs, which need somewhere to be watched.
    An ACP client renders ONE session; lanes would run invisibly."""
    return None


def _nullcontext() -> AbstractContextManager[None]:
    return nullcontext()
