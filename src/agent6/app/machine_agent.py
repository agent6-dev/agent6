# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The two sides of a machine `agent` state: the host-side launcher that spawns
the confined subprocess, and the confined runner itself.

A machine run's engine is a thin supervisor that stays in the host network
namespace and makes no network calls itself. Each `agent` state runs in its own
fresh process (`ui/cli/machine_agent` is the `python -m` entry), independent of
the engine and of sibling `tool` states; like every agent process it runs
unconfined, and the jail bounds the commands it dispatches.

`build_machine_agent_runner` (host side) builds the callable an `agent` state
fires: it spawns the subprocess with a fixed argv, hands it the request via a
temp file, and enforces the timeout by killing the process group. `run_one`
(subprocess side) reads that request, sets up the sandbox while still
single-threaded, runs the agent loop to completion, and writes the result.
`MachineAgentRequest` (here) owns the `request.json` file shape and
`AgentExecResult` (machine/engine.py) owns `result.json`: both sides
serialize/validate through the models, per the IPC rule
(`tests/unit/test_machine_agent_ipc.py` pins the bytes). The live conversation
view is the one presentation piece: `ui/cli` injects `attach_console` so this
module never imports `agent6.ui`.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from agent6.app._setup import apply_git_egress_policy
from agent6.app.confine import check_hide_paths_support, check_network_support
from agent6.app.machine._spend import Spend, read_budget_totals
from agent6.app.providers import (
    InstrumentedProvider,
    build_role_provider,
    build_summariser_provider,
    resolve_compaction_thresholds,
    resolve_decompose,
)
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective_with_overlay, resolved_state_dir
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    render_commit_trailer,
)
from agent6.git_ops import status as git_status
from agent6.machine import AgentExecResult, AgentRequest
from agent6.providers import Provider, TranscriptSink
from agent6.sessions.ipc import (
    clear_answer,
    clear_pending_answers,
    clear_question_answers,
    clear_steer_answer,
    clear_steer_request,
    frontend_is_live,
    read_answer,
    read_question_answers,
    read_steer_answer,
    record_answer,
    session_allow_set,
    steer_request_pending,
)
from agent6.tools.dispatch import Approver, ToolDispatcher
from agent6.tools.schema import UserQuestion
from agent6.types import IsolationLevel
from agent6.workflows.loop import Workflow


def _no_console(_events: EventSink) -> None:
    """The headless default when no front-end injects a live view."""


class MachineAgentRequest(BaseModel):
    """The ``request.json`` envelope of the machine-agent subprocess IPC.

    The host runner (`build_machine_agent_runner`) serializes it into the temp
    file the fixed argv (``python -m agent6.ui.cli.machine_agent <request.json>
    <result.json>``) names; the subprocess validates it back and hands it to
    `run_one`. One owner of the file shape per the IPC rule; ``result.json`` is
    owned the same way by `AgentExecResult`. The files are transient
    per-invocation (both sides are always the same install), and the bytes are
    pinned by ``tests/unit/test_machine_agent_ipc.py``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cwd: Path
    root: Path
    # The machine's `[config]` overlay, applied over the effective config.
    overlay: dict[str, Any]
    isolation: IsolationLevel
    transcript_dir: Path
    # When set, the subprocess writes a watchable logs.jsonl here (role.*_delta
    # + tool.* events), so `machine create` and live `agent` states are
    # followable in the TUI/web dashboard exactly like a run.
    events_log: Path | None = None
    protect_paths: tuple[Path, ...] = ()
    # Resolved on the host (pre-Landlock, so it sees global git config); the
    # confined subprocess can't read ~/.gitconfig, so its mode="run" commits
    # would otherwise fail with "Author identity unknown". None for read-only
    # (mode="agent"/"machine") states.
    commit_identity: CommitIdentity | None = None
    request: AgentRequest


def _machine_head_sha(root: Path) -> str | None:
    """HEAD at state start, the chain's first parent; None when unreadable
    (an unborn repo roots the chain)."""
    try:
        return git_status(root).head_sha or None
    except (GitError, OSError):
        return None


def _result(
    reason: str, payload: dict[str, Any] | None, budget: BudgetTracker | None
) -> AgentExecResult:
    usd = 0.0
    partial = False
    inp = out = 0
    if budget is not None:
        usd, partial = budget.estimate_usd()
        snap = budget.snapshot()
        inp, out = snap.input_total, snap.output_total
    return AgentExecResult(
        reason=reason,
        payload=payload,
        usd=usd,
        usd_partial=partial,
        input_tokens=inp,
        output_tokens=out,
    )


def _apply_operator_env_grants(cfg: Config) -> Config:
    """The supervisor's `machine run --auto-approve` grant, carried by env like
    the sandbox setter (operator-only, structurally LLM-unreachable; a machine
    [config] overlay must not and cannot set sandbox.*). Upgrades run_commands
    ask -> yes, never a withheld no (`with_sandbox_overrides`)."""
    return cfg.with_sandbox_overrides(auto_approve=os.environ.get("AGENT6_AUTO_APPROVE") == "1")


@dataclass(frozen=True, slots=True)
class _MachineBridges:
    """The interactivity bridges for one machine `agent` state.

    Answers are read from the per-state dir, but a front-end registers
    a `frontends/` claim on the instance dir, so the liveness gate probes the instance
    dir (`live_dir`). Prompt/answer events go to the per-state log the front-end
    already tails, so its SessionState fold surfaces them like a run's.
    """

    approve: Approver
    ask: Callable[[tuple[UserQuestion, ...]], tuple[str, ...]]
    steer_requested: Callable[[], bool]
    steer_clear: Callable[[], None]
    steer_prompt: Callable[[], str | None]


def _build_machine_bridges(
    instance_dir: Path, state_dir: Path, events: EventSink
) -> _MachineBridges:
    """Wire run-level approval/question/steer bridges to a machine agent state.

    No live front-end (a `frontends/` claim on the instance dir) makes each bridge a
    safe headless default: deny an approval, answer a question with "", no steer.
    """
    # Crash recovery re-executes the same `<seq>-<state>` dir and its prompt-id
    # counters restart at 1, so an answer file left by the aborted attempt would
    # satisfy this execution's first prompt unseen. Drop the stale bridge state
    # first (front-end claims live on the instance dir, so this touches none).
    clear_pending_answers(state_dir)
    counters = {"approval": 0, "question": 0}

    def approve(prompt: str, /, *, scope: str | None = None) -> bool:
        counters["approval"] += 1
        prompt_id = f"approval-{counters['approval']}"
        # A prior "allow session" for THIS scope auto-passes its later prompts.
        if scope and session_allow_set(state_dir, scope):
            events.emit("approval.answer", id=prompt_id, approved=True, source="session")
            return True
        # Clear a pre-written answer for this id before emitting (see the run
        # approver): a premature /api/machine/<name>/approve must not auto-pass.
        clear_answer(state_dir, prompt_id)
        events.emit("approval.prompt", id=prompt_id, prompt=prompt, standing=bool(scope))
        approved: bool | None = None
        source = "headless"
        if frontend_is_live(instance_dir):
            answer = read_answer(state_dir, prompt_id, live_dir=instance_dir)
            if answer is not None:
                approved = record_answer(state_dir, answer, scope)
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


def _build_agent_providers(
    cfg: Config,
    req: MachineAgentRequest,
    *,
    budget: BudgetTracker,
    attach_console: Callable[[EventSink], None],
) -> tuple[InstrumentedProvider, Provider, EventSink | None]:
    """The agent state's worker provider (instrumented), its reviewer-role
    summariser, and the optional event sink.

    One TranscriptSink for both (its seq counter is per-instance), each seat
    stamped. An EventSink only when the caller passes events_log; the console
    attach is injected so this module never imports `agent6.ui`.

    ALWAYS streams: machine agents run headless and generate long, and
    OpenRouter-style gateways' SSE heartbeats corrupt a non-streaming body
    mid-read. Streaming also feeds the role.*_delta events."""
    transcript_sink = TranscriptSink(req.transcript_dir)
    inner_provider = build_role_provider(
        cfg, "worker", transcript_sink=transcript_sink, budget=budget
    )
    events_sink = EventSink(req.events_log) if req.events_log is not None else None
    rm = cfg.models.resolve("worker")
    if events_sink is not None:
        attach_console(events_sink)
    provider = InstrumentedProvider(
        inner=inner_provider,
        role="agent",
        model=rm.model if rm is not None else "",
        provider_name=rm.provider if rm is not None else "",
        events=events_sink,
        budget=budget,
        stream_text=True,
    )
    summariser_provider = build_summariser_provider(
        cfg, transcript_sink=transcript_sink, budget=budget, events=events_sink
    )
    return provider, summariser_provider, events_sink


def run_one(
    req: MachineAgentRequest,
    *,
    attach_console: Callable[[EventSink], None] = _no_console,
    reporter: Reporter = STDIO_REPORTER,
) -> AgentExecResult:
    isolation = req.isolation
    r = req.request
    # Config load + per-state overrides run FIRST, and can raise (a bad overlay,
    # or an override naming a provider that isn't configured). Salvage that into
    # a clean AgentExecResult the subprocess writes to result.json -- otherwise
    # the exception escapes to a pydantic traceback + a non-zero exit, and the
    # host runner only recovers it via its missing-result fallback.
    try:
        cfg = load_effective_with_overlay(req.cwd, req.overlay).config.with_machine_agent_overrides(
            provider=r.provider,
            model=r.model,
            thinking=r.thinking,
            temperature=r.temperature,
            max_usd=r.max_usd,
            max_tokens_fallback=r.max_tokens_fallback,
        )
        cfg = _apply_operator_env_grants(cfg)
    except (ConfigError, ValidationError) as exc:
        reporter.err(f"REFUSING: machine agent config error: {exc}")
        return _result("error", None, None)
    apply_git_egress_policy(cfg)
    # A mode="run" state commits its work, but this confined process can't read
    # ~/.gitconfig (not a Landlock read root): export the host-resolved identity
    # so git uses it regardless of where the config lives.
    if req.commit_identity is not None:
        if name := req.commit_identity.name:
            os.environ["GIT_AUTHOR_NAME"] = os.environ["GIT_COMMITTER_NAME"] = name
        if email := req.commit_identity.email:
            os.environ["GIT_AUTHOR_EMAIL"] = os.environ["GIT_COMMITTER_EMAIL"] = email
    # The engine already validated the isolation against the config; re-check
    # defensively and fail closed.
    net_err = check_network_support(cfg, isolation)
    if net_err is not None:
        reporter.err(f"REFUSING: {net_err}")
        return _result("error", None, None)
    hide_err = check_hide_paths_support(cfg, isolation)
    if hide_err is not None:
        reporter.err(f"REFUSING: {hide_err}")
        return _result("error", None, None)
    budget = BudgetTracker(
        max_usd=cfg.budget.max_usd,
        max_tokens_fallback=cfg.budget.max_tokens_fallback,
    )
    provider, summariser_provider, events_sink = _build_agent_providers(
        cfg, req, budget=budget, attach_console=attach_console
    )
    # Re-confirm the cwd-containment invariant at the subprocess boundary
    # (defense in depth, the engine already filtered these).
    root_r = req.root.resolve()
    protect = tuple(rp for p in req.protect_paths if (rp := p.resolve()).is_relative_to(root_r))
    # "machine" (the `machine create` authoring agent) and "agent" (a
    # running machine's `agent` state, unless it opted into mode="run") are
    # read-only structured-output loops: the dispatcher refuses edits AND
    # run_command/run_verify (defense in depth alongside the read-only tool
    # list) and the loop uses a finish_session-focused prompt.
    mode = r.mode
    read_only = mode in ("machine", "agent")
    # Bridge run-level interactivity (approve/ask_user/steer) to a front-end
    # watching this machine: answers land in the per-state dir, the liveness
    # gate probes the instance dir where the front-end registers its claim.
    # Needs a per-state log (events_sink) for the front-end to see the prompt.
    bridges: _MachineBridges | None = None
    if events_sink is not None and req.events_log is not None:
        state_dir = req.events_log.parent
        instance_dir = req.transcript_dir.parent
        bridges = _build_machine_bridges(instance_dir, state_dir, events_sink)
    dispatcher = ToolDispatcher(
        root=req.root,
        config=cfg,
        isolation=isolation,
        approver=bridges.approve if bridges is not None else None,
        questioner=bridges.ask if bridges is not None else None,
        events=events_sink,
        curator=None,
        run_root_node_id=None,
        mcp_manager=None,
        extra_protect_paths=protect,
        mode="machine" if read_only else "run",
        # The REPO's state dir (not this state's per-state dir above): a
        # mode="run" agent state participates in cross-run memory like any
        # other run; for read-only states the dispatcher mode guard and
        # the machine/agent prompt assembly keep it inert.
        state_dir=resolved_state_dir(req.root),
    )
    rm = cfg.models.resolve("worker")
    compact_drop, compact_summarise = resolve_compaction_thresholds(cfg, rm, log=reporter.err)
    cfg = resolve_decompose(cfg, rm, log=reporter.err)
    wf = Workflow(
        root=req.root,
        config=cfg,
        commit_trailer=render_commit_trailer(
            cfg.git.commit.trailer, models=(rm.model if rm is not None else "",)
        ),
        provider=provider,
        summariser_provider=summariser_provider,
        dispatcher=dispatcher,
        logger=reporter.err,
        mode=mode if mode in ("machine", "agent") else "run",
        # A mode="run" state commits on its own chain like any run; the
        # instance dir name is its session-unique id. Read-only states never
        # commit (mode gate), so the refs stay None there.
        chain_ref=(
            f"refs/agent6/machine-{req.transcript_dir.parent.name}" if not read_only else None
        ),
        chain_fallback_parent=_machine_head_sha(req.root) if not read_only else None,
        commit_per_step=cfg.git.commit_per_step,
        state_dir=resolved_state_dir(req.root),
        compact_drop_at_chars=compact_drop,
        compact_summarise_at_chars=compact_summarise,
        context_summary_max_tokens=cfg.context.summary_max_tokens,
        compact_elision_gists=cfg.context.elision_gists,
        steer_requested=bridges.steer_requested if bridges is not None else (lambda: False),
        steer_clear=bridges.steer_clear if bridges is not None else (lambda: None),
        steer_prompt=bridges.steer_prompt if bridges is not None else (lambda: None),
    )
    result = wf.run(r.prompt)
    payload = result.finish_payload if result.reason == "finish_session" else None
    return _result(result.reason, payload, budget)


def build_machine_agent_runner(
    overlay: dict[str, Any],
    cwd: Path,
    isolation: IsolationLevel,
    transcript_dir: Path,
    protect_paths: tuple[Path, ...] = (),
    commit_identity: CommitIdentity | None = None,
) -> Callable[[AgentRequest, Path | None], AgentExecResult]:
    """Build the host-side runner an `agent` state uses to drive a confined loop.

    The machine engine is a host-netns supervisor; each `agent` state runs in
    its OWN subprocess (`agent6.ui.cli.machine_agent`) driving the loop
    (`run_one` above), independently of the engine and of sibling `tool`
    states. Like every agent process it runs unconfined; the jail bounds the
    commands it dispatches. The subprocess is
    spawned with a fixed argv (no LLM-derived content) and handed the request via
    a temp file; the operator-authored prompt travels in that file, never on the
    command line. ``timeout_secs`` is enforced by killing the subprocess's whole
    process group (true mid-call cancellation, and the per-agent broker is torn
    down with it).

    ``events_log`` is per CALL: the live World passes each agent-state execution
    its own ``<instance>/states/<seq>-<state>/logs.jsonl`` and `machine create`
    passes the draft log, so the subprocess writes a watchable event stream there.
    """

    def run_agent(request: AgentRequest, events_log: Path | None = None) -> AgentExecResult:
        # The salvage below must see only THIS call's events: machine create
        # shares one draft log across attempts, and an attempt that died before
        # its first budget.update would otherwise salvage the PRIOR attempt's
        # cumulative totals and double-book them. (Per-state logs are fresh per
        # execution, so the offset is 0 there.)
        start_offset = 0
        if events_log is not None:
            with contextlib.suppress(OSError):
                start_offset = events_log.stat().st_size

        def salvaged(reason: str) -> AgentExecResult:
            # No result.json (killed/timed-out/crashed): recover the loop's
            # running budget.update totals from the state's own event log, else a
            # timed-out state books $0 and the budget guard never trips.
            spend = (
                read_budget_totals(events_log, from_offset=start_offset)
                if events_log is not None
                else Spend()
            )
            return AgentExecResult(
                reason=reason,
                payload=None,
                usd=spend.usd,
                usd_partial=spend.partial,
                input_tokens=spend.input_tokens,
                output_tokens=spend.output_tokens,
            )

        payload = MachineAgentRequest(
            cwd=cwd,
            root=cwd,
            overlay=overlay,
            isolation=isolation,
            transcript_dir=transcript_dir,
            events_log=events_log,
            protect_paths=protect_paths,
            commit_identity=commit_identity,
            request=request,
        )
        with tempfile.TemporaryDirectory(prefix="agent6-machine-agent-") as td:
            req_file = Path(td) / "request.json"
            out_file = Path(td) / "result.json"
            req_file.write_text(payload.model_dump_json(), encoding="utf-8")
            argv = [
                sys.executable,
                "-m",
                "agent6.ui.cli.machine_agent",
                str(req_file),
                str(out_file),
            ]
            # Own session/process group so the timeout kill takes the agent
            # subprocess AND its broker/jail children with it.
            proc = subprocess.Popen(argv, start_new_session=True)
            try:
                proc.wait(timeout=request.timeout_s)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    # By pid: start_new_session made it the group leader, and an
                    # unreaped child's pgid cannot have been recycled. Looking
                    # it up first left a window where, under sudo, an unrelated
                    # group could be killed as root.
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return salvaged("timeout")
            if proc.returncode != 0 or not out_file.is_file():
                return salvaged("error")
            try:
                return AgentExecResult.model_validate_json(out_file.read_text(encoding="utf-8"))
            except (OSError, ValidationError):
                # A malformed result.json is treated like a missing one: the
                # spend salvage keeps the budget honest.
                return salvaged("error")

    return run_agent
