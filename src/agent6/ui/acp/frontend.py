# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the `SessionFrontend` an ACP client provides.

Every prompt the lifecycle raises becomes a `session/request_permission`; what a
terminal would draw becomes nothing, since the editor renders `session/update`.
A client that declared it cannot be asked gets the cautious default instead.
"""

from __future__ import annotations

import contextlib
import pathlib
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from agent6 import budget, events, kinds
from agent6.app import frontend
from agent6.config import Config
from agent6.harness import _snapshot, loop
from agent6.sessions import ipc, layout
from agent6.tools import operator_prompts
from agent6.ui import steer

# Silence past the wait is the cautious answer: an approval denies, a question has none.
PERMISSION_TIMEOUT_S = 300.0


class Asker(Protocol):
    """Ask the editor one prompt.

    Args:
        prompt: The text.
        options: The choices, in order.
        standing: Whether "always" may be offered; None for a question, whose
            options the model wrote and which is never a permission.
        call_id: The dispatcher's stamp on the gated tool call, or None for a
            prompt that gates no call.
        until: Polled while the answer is pending; True ends the wait with None,
            the prompt having been answered by another route.

    Returns:
        The chosen option, or None for no answer.
    """

    def __call__(  # noqa: D102  # the protocol's docstring documents the call
        self,
        prompt: str,
        options: tuple[str, ...],
        standing: bool | None,
        call_id: int | None,
        until: Callable[[], bool] | None = None,
        /,
    ) -> str | None: ...


def acp_frontend(  # noqa: C901  # one callable per ACP capability, built in one place
    *,
    ask: Asker,
    capabilities: frontend.FrontendCapabilities,
    agent6_exe: Callable[[], str],
    spawn_detached_resume: Callable[[pathlib.Path, str, Sequence[str]], str],
) -> frontend.SessionFrontend:
    """Wire the lifecycle to one ACP client.

    Args:
        ask: Asks the editor one prompt.
        capabilities: What the client said it can do.
        agent6_exe: The agent6 executable a spawn uses.
        spawn_detached_resume: Spawns a detached resume.

    Returns:
        The front-end the lifecycle drives.
    """

    def _approve(
        prompt: str,
        /,
        *,
        scope: str | None = None,
        call_id: int | None = None,
        until: Callable[[], bool] | None = None,
    ) -> bool | None:
        """Return the editor's verdict, or None when it gave none in time."""
        if not capabilities.can_ask:
            return False
        # No scope means an "always allow" must not cover this one; the option names carry it.
        standing = scope is not None
        options = ("allow", "deny") if standing else ("allow once", "deny")
        answer = ask(prompt, options, standing, call_id, until)
        return None if answer is None else answer.startswith("allow")

    def _build_approver(session_dir: pathlib.Path) -> operator_prompts.Approver:
        def approve(
            request: operator_prompts.ApprovalRequest, /
        ) -> operator_prompts.ApprovalAnswer:
            if not capabilities.can_ask:
                return operator_prompts.ApprovalAnswer(False, "headless")
            approved = _approve(
                request.prompt,
                scope=request.scope,
                call_id=request.call_id,
                until=lambda: ipc.answer_written(session_dir, request.id),
            )
            if approved is None:
                filed = ipc.read_answer(session_dir, request.id, timeout_s=0.0)
                if filed is not None:
                    return operator_prompts.ApprovalAnswer(
                        ipc.record_answer(session_dir, filed, request.scope), "frontend"
                    )
                return operator_prompts.ApprovalAnswer(False, "acp")
            return operator_prompts.ApprovalAnswer(approved, "acp")

        return approve

    def _build_questioner(session_dir: pathlib.Path) -> operator_prompts.Questioner:
        def ask_questions(
            request: operator_prompts.QuestionRequest, /
        ) -> operator_prompts.QuestionAnswer:
            if not capabilities.can_ask:
                return operator_prompts.QuestionAnswer(
                    tuple("" for _ in request.questions), "headless", unseen=True
                )
            if not all(question.options for question in request.questions):
                filed = ipc.read_question_answers(session_dir, request.id, timeout_s=0.0)
                if filed is not None:
                    return operator_prompts.QuestionAnswer(filed, "frontend")
                # ACP v1 renders option buttons only, so a free-form question reached nobody.
                return operator_prompts.QuestionAnswer(
                    tuple("" for _ in request.questions), "headless", unseen=True
                )
            # One deadline for the request, or N questions wait N times the bound.
            deadline = time.monotonic() + PERMISSION_TIMEOUT_S
            answers: list[str] = []
            for question in request.questions:
                answer = ask(
                    question.question,
                    question.options,
                    None,
                    request.call_id,
                    lambda: (
                        ipc.question_answers_written(session_dir, request.id)
                        or time.monotonic() >= deadline
                    ),
                )
                if answer is None:
                    filed = ipc.read_question_answers(session_dir, request.id, timeout_s=0.0)
                    if filed is not None:
                        return operator_prompts.QuestionAnswer(filed, "frontend")
                answers.append(answer or "")
            return operator_prompts.QuestionAnswer(tuple(answers), "acp")

        return ask_questions

    def _confirm_unconfined(isolation: kinds.IsolationLevel, cfg: Config) -> bool:
        """Ask only when the run is unconfined, so the approval never becomes reflexive.

        Returns:
            Whether the run may go ahead.
        """
        if isolation != "none" or cfg.sandbox.run_commands != "yes":
            return True
        # No scope: docs/security.md makes this a one-time gate, never an "always".
        return bool(_approve("Run commands UNSANDBOXED on this host, with no per-command prompt?"))

    def _steer(
        _events: events.EventSink,
        session_dir: pathlib.Path,
        _facts: Callable[[], frontend.SessionFacts],
    ) -> frontend.SteerHooks:
        # A later prompt resumes the run with its text seeded through the steer files.
        return steer.file_bridge_steer(session_dir)

    def _no_repl(
        _session_dir: pathlib.Path, _budget: budget.BudgetTracker, _task: str, _mcp: object
    ) -> Callable[[int, str], kinds.AutoCommitDirective]:
        # ACP has its own turn loop; a REPL inside it would be a second reader of stdin.
        return lambda _iteration, _summary: "continue"

    def _no_ask_repl(
        _wf: loop.Harness, _budget: budget.BudgetTracker, _layout: layout.SessionLayout, _task: str
    ) -> _snapshot.SessionResult:
        raise RuntimeError("an ACP session drives its own turns; the ask REPL is not used")

    return frontend.SessionFrontend(
        capabilities=capabilities,
        should_spawn_tui=lambda _tui, _interactive, _mode: False,
        # The deltas stream as events; the editor is the live view.
        stream_modes=lambda _tui_enabled: (True, True),
        attach_console_view=lambda _events: None,
        close_console_view=lambda: None,
        loop_logger=lambda _mode: lambda _line: None,
        tui_session=lambda _session_dir, _enabled: contextlib.nullcontext(),
        build_approver=_build_approver,
        build_questioner=_build_questioner,
        make_steer_state=_steer,
        confirm_unconfined_autorun=_confirm_unconfined,
        confirm_run_on_run_branch=lambda branch: bool(
            _approve(f"Continue this run on {branch!r}, which is already a run branch?")
        ),
        confirm_replay_after_crash=lambda iteration, tools: bool(
            _approve(
                f"The previous run died mid-turn (iteration {iteration};"
                f" {', '.join(tools) or 'unknown tools'}). Its tools may have partially"
                " applied; replaying can repeat a non-idempotent effect. Re-run the turn?"
            )
        ),
        prompt_detach_away_mode=lambda _session_dir, _scopes: None,
        select_revised_prompt=None,
        build_repl_hook=_no_repl,
        run_ask_repl=_no_ask_repl,
        save_ask_transcript=lambda _layout, _question, _answer: None,
        build_coordinator_spawner=_no_coordinator,
        agent6_exe=agent6_exe,
        spawn_detached_resume=spawn_detached_resume,
    )


def _no_coordinator(
    _cfg: Config,
    _cwd: pathlib.Path,
    _state_dir: pathlib.Path,
    _mode: str,
    _session_id: str,
    _max_usd: float | None,
    _auto_approve: bool,
) -> None:
    """Refuse `/parallel`: an ACP client renders one session, so lanes would run unseen."""
    return None
