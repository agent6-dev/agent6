# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Subprocess entry: run ONE machine `agent` state, self-confined.

A machine run's engine is a thin supervisor that stays in the host network
namespace and makes no network calls itself. Each `agent` state runs *here*, in
its own fresh process, so it can confine its OWN egress per
`sandbox.agent_network` (the broker on `strict`, Landlock on `hardened`),
independently of the engine and of sibling `tool` states. That is what lets a
machine run a broker-confined agent alongside an operator-reviewed, network-carved-out
tool in the same run.

Invoked as ``python -m agent6.ui.cli.machine_agent <request.json> <result.json>``.
It reads a request, sets up the sandbox while still single-threaded, runs the
agent loop to completion, and writes the result. The engine enforces the
timeout by killing this process, which gives true mid-call cancellation.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent6.budget import BudgetTracker
from agent6.config.layer import load_effective_with_overlay
from agent6.events import EventSink
from agent6.git_ops import set_repo_hook_policy
from agent6.providers import TranscriptSink
from agent6.sandbox.detect import detect
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.schema import UserQuestion
from agent6.types import SandboxProfile
from agent6.ui.bridge.approval import (
    clear_answer,
    clear_pending_answers,
    clear_question_answers,
    clear_steer_answer,
    clear_steer_request,
    frontend_is_live,
    read_answer,
    read_question_answers,
    read_steer_answer,
    session_allow_set,
    steer_request_pending,
)
from agent6.ui.cli._common import _state_dir
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli.egress import (
    _check_network_profile,
    _maybe_apply_agent_landlock,
    _maybe_start_egress,
    _stop_egress,
)
from agent6.ui.cli.providers import (
    _build_role_provider,
    _InstrumentedProvider,
    resolve_compaction_thresholds,
    resolve_decompose,
)
from agent6.workflows.loop import Workflow


def _result(
    reason: str, payload: dict[str, Any] | None, budget: BudgetTracker | None
) -> dict[str, Any]:
    usd = 0.0
    inp = out = 0
    if budget is not None:
        usd, _ = budget.estimate_usd()
        snap = budget.snapshot()
        inp, out = snap.input_total, snap.output_total
    return {
        "reason": reason,
        "payload": payload,
        "usd": usd,
        "input_tokens": inp,
        "output_tokens": out,
    }


@dataclass
class _MachineBridges:
    """The interactivity bridges for one machine `agent` state.

    Answers are read from the per-state dir, but a front-end registers
    `frontend.pid` on the instance dir, so the liveness gate probes the instance
    dir (`live_dir`). Prompt/answer events go to the per-state log the front-end
    already tails, so its RunState fold surfaces them like a run's.
    """

    approve: Callable[[str], bool]
    ask: Callable[[tuple[UserQuestion, ...]], tuple[str, ...]]
    steer_requested: Callable[[], bool]
    steer_clear: Callable[[], None]
    steer_prompt: Callable[[], str | None]


def _build_machine_bridges(
    instance_dir: Path, state_dir: Path, events: EventSink
) -> _MachineBridges:
    """Wire run-level approval/question/steer bridges to a machine agent state.

    A no live front-end (`frontend.pid` on the instance dir) makes each bridge a
    safe headless default: deny an approval, answer a question with "", no steer.
    """
    # Crash recovery re-executes the same `<seq>-<state>` dir and its prompt-id
    # counters restart at 1, so an answer file left by the aborted attempt would
    # satisfy this execution's first prompt unseen. Drop the stale bridge state
    # first (frontend.pid lives on the instance dir, so this touches none).
    clear_pending_answers(state_dir)
    counters = {"approval": 0, "question": 0}

    def approve(prompt: str) -> bool:
        counters["approval"] += 1
        prompt_id = f"approval-{counters['approval']}"
        # A prior "allow session" for this agent state auto-passes every later prompt.
        if session_allow_set(state_dir):
            events.emit("approval.answer", id=prompt_id, approved=True, source="session")
            return True
        # Clear a pre-written answer for this id before emitting (see the run
        # approver): a premature /api/machine/<name>/approve must not auto-pass.
        clear_answer(state_dir, prompt_id)
        events.emit("approval.prompt", id=prompt_id, prompt=prompt)
        approved: bool | None = None
        source = "headless"
        if frontend_is_live(instance_dir):
            approved = read_answer(state_dir, prompt_id, live_dir=instance_dir)
            if approved is not None:
                source = "frontend"
        if approved is None:
            approved = False  # headless machine: no operator to ask, deny safely
        events.emit("approval.answer", id=prompt_id, approved=approved, source=source)
        return approved

    def ask(questions: tuple[UserQuestion, ...]) -> tuple[str, ...]:
        counters["question"] += 1
        question_id = f"question-{counters['question']}"
        clear_question_answers(state_dir, question_id)  # drop any premature answer
        events.emit(
            "question.prompt",
            id=question_id,
            questions=[{"question": q.question, "options": list(q.options)} for q in questions],
        )
        answers: tuple[str, ...] | None = None
        source = "headless"
        if frontend_is_live(instance_dir):
            answers = read_question_answers(state_dir, question_id, live_dir=instance_dir)
            if answers is not None:
                source = "frontend"
        if answers is None:
            answers = tuple("" for _ in questions)
        events.emit("question.answer", id=question_id, answers=list(answers), source=source)
        return answers

    def steer_requested() -> bool:
        return steer_request_pending(state_dir)

    def steer_clear() -> None:
        clear_steer_answer(state_dir)
        clear_steer_request(state_dir)

    def steer_prompt() -> str | None:
        if not frontend_is_live(instance_dir):
            clear_steer_request(state_dir)
            return None
        answer = read_steer_answer(state_dir, live_dir=instance_dir)
        if answer is None:
            clear_steer_request(state_dir)
        return answer

    return _MachineBridges(approve, ask, steer_requested, steer_clear, steer_prompt)


def _run_one(req: dict[str, Any]) -> dict[str, Any]:
    cwd = Path(req["cwd"])
    profile: SandboxProfile = req["profile"]
    root = Path(req["root"])
    transcript_dir = Path(req["transcript_dir"])
    r = req["request"]
    cfg = load_effective_with_overlay(cwd, req["overlay"]).config.with_machine_agent_overrides(
        provider=r["provider"],
        model=r["model"],
        thinking=r["thinking"],
        temperature=r["temperature"],
        max_usd=r["max_usd"],
        max_input_tokens=r["max_input_tokens"],
        max_output_tokens=r["max_output_tokens"],
    )
    set_repo_hook_policy(cfg.git.run_repo_hooks)
    # A mode="run" state commits its work, but this confined process can't read
    # ~/.gitconfig (not a Landlock read root). The engine resolved the identity on
    # the host and passed it down; export it so git uses it regardless of where
    # the config lives. None for read-only states (no commits).
    ident = req.get("commit_identity")
    if isinstance(ident, dict):
        if name := ident.get("name"):
            os.environ["GIT_AUTHOR_NAME"] = os.environ["GIT_COMMITTER_NAME"] = str(name)
        if email := ident.get("email"):
            os.environ["GIT_AUTHOR_EMAIL"] = os.environ["GIT_COMMITTER_EMAIL"] = str(email)
    # Confine THIS process's egress per sandbox.agent_network (single-threaded
    # here, as required by unshare). The engine already validated the combo, but
    # re-check defensively and fail closed.
    net_err = _check_network_profile(cfg, profile)
    if net_err is not None:
        print(f"REFUSING: {net_err}", file=sys.stderr)
        return _result("error", None, None)
    egress_guard, egress_err = _maybe_start_egress(cfg, profile)
    if egress_err is not None:
        print(f"REFUSING: {egress_err}", file=sys.stderr)
        return _result("error", None, None)
    budget: BudgetTracker | None = None
    try:
        landlock_err = _maybe_apply_agent_landlock(cfg, profile, detect())
        if landlock_err is not None:
            print(f"REFUSING: {landlock_err}", file=sys.stderr)
            return _result("error", None, None)
        budget = BudgetTracker(
            max_input_tokens=cfg.budget.max_input_tokens,
            max_output_tokens=cfg.budget.max_output_tokens,
            max_usd=cfg.budget.best_effort_usd_limit,
        )
        inner_provider = _build_role_provider(
            cfg, "worker", transcript_sink=TranscriptSink(transcript_dir), budget=budget
        )
        # An EventSink only when the caller passes events_log: the machine
        # supervisor points it at this state's per-state
        # `states/<seq>-<name>/logs.jsonl` (and `machine create` at the draft's
        # logs.jsonl), so the TUI/web can watch the agent's reasoning + tool
        # calls live, exactly like a run. None only when no log path is given.
        events_log = req.get("events_log")
        events_sink = EventSink(Path(events_log)) if isinstance(events_log, str) else None
        # stream_text: ALWAYS use the streaming transport. Machine agents run
        # headless (cron / nohup) and produce long generations; the
        # non-streaming path drops the connection mid-body on OpenRouter-style
        # gateways (SSE heartbeats corrupt it, observed as "incomplete chunked
        # read" on ~14k-token authoring calls). It is also what feeds the
        # role.*_delta events to the sink above.
        # At a TTY (or forced), render the live conversation to stderr so
        # `machine create` and live `agent` states are watchable; it consumes
        # the same events the sink records.
        rm = cfg.models.resolve("worker")
        if events_sink is not None and (
            sys.stderr.isatty() or os.environ.get("AGENT6_FORCE_STREAM") == "1"
        ):
            events_sink.subscribe(ConsoleView(sys.stderr))
        provider = _InstrumentedProvider(
            inner=inner_provider,
            role=r.get("role_label", "agent"),
            model=rm.model if rm is not None else "",
            provider_name=rm.provider if rm is not None else "",
            events=events_sink,
            budget=budget,
            stream_text=True,
        )
        # Re-confirm the cwd-containment invariant at the subprocess boundary
        # (defense in depth, the engine already filtered these).
        root_r = root.resolve()
        protect = tuple(
            rp
            for p in req.get("protect_paths", [])
            if (rp := Path(p).resolve()).is_relative_to(root_r)
        )
        # "machine" (the `machine create` authoring agent) and "agent" (a
        # running machine's `agent` state, unless it opted into mode="run") are
        # read-only structured-output loops: the dispatcher refuses edits AND
        # run_command/run_verify (defense in depth alongside the read-only tool
        # list) and the loop uses a finish_run-focused prompt.
        mode = r.get("mode", "agent")
        read_only = mode in ("machine", "agent")
        # Bridge run-level interactivity (approve/ask_user/steer) to a front-end
        # watching this machine: answers land in the per-state dir, the liveness
        # gate probes the instance dir where the front-end registers frontend.pid.
        # Needs a per-state log (events_sink) for the front-end to see the prompt.
        bridges: _MachineBridges | None = None
        if events_sink is not None and events_log is not None:
            state_dir = Path(events_log).parent
            instance_dir = transcript_dir.parent
            bridges = _build_machine_bridges(instance_dir, state_dir, events_sink)
        dispatcher = ToolDispatcher(
            root=root,
            config=cfg,
            sandbox_profile=profile,
            approver=bridges.approve if bridges is not None else None,
            questioner=bridges.ask if bridges is not None else None,
            events=events_sink,
            graph_client=None,
            run_root_node_id=None,
            mcp_manager=None,
            extra_protect_paths=protect,
            mode="machine" if read_only else "run",
            # The REPO's state dir (not this state's per-state dir above): a
            # mode="run" agent state participates in cross-run memory like any
            # other run; for read-only states the dispatcher mode guard and
            # the machine/agent prompt assembly keep it inert.
            state_dir=_state_dir(root),
        )
        compact_drop, compact_summarise = resolve_compaction_thresholds(
            cfg, rm, log=lambda msg: print(msg, file=sys.stderr)
        )
        cfg = resolve_decompose(cfg, rm, log=lambda msg: print(msg, file=sys.stderr))
        wf = Workflow(
            root=root,
            config=cfg,
            provider=provider,
            dispatcher=dispatcher,
            logger=lambda msg: print(msg, file=sys.stderr),
            mode=mode if mode in ("machine", "agent") else "run",
            state_dir=_state_dir(root),
            compact_drop_at_chars=compact_drop,
            compact_summarise_at_chars=compact_summarise,
            context_summary_max_tokens=cfg.context.summary_max_tokens,
            compact_elision_gists=cfg.context.elision_gists,
            steer_requested=bridges.steer_requested if bridges is not None else (lambda: False),
            steer_clear=bridges.steer_clear if bridges is not None else (lambda: None),
            steer_prompt=bridges.steer_prompt if bridges is not None else (lambda: None),
        )
        result = wf.run(r["prompt"])
        payload = result.finish_payload if result.reason == "finish_run" else None
        return _result(result.reason, payload, budget)
    finally:
        _stop_egress(egress_guard)


def main() -> int:
    req = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = _run_one(req)
    Path(sys.argv[2]).write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
