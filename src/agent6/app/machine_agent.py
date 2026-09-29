# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The two sides of a machine `agent` state: the host-side launcher and the runner.

The machine engine is a supervisor that makes no network calls itself. Each
`agent` state runs the loop in its own process (`ui/cli/machine_agent` is the
`python -m` entry), unconfined like every agent process, with the jail bounding
the commands it dispatches. `build_machine_agent_runner` spawns that process with
a fixed argv, hands it the request through a temp file and enforces the timeout
by killing the process group; `run_one` reads the request, validates it, runs the
loop and writes the result. `MachineAgentRequest` owns the `request.json` shape
and `AgentExecResult` owns `result.json`. The live view is injected as
`attach_console`, so this module never imports `agent6.ui`.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from agent6.app._session import tool_result_cap_bytes
from agent6.app._setup import apply_git_ops_policy, budget_tracker
from agent6.app.confine import check_hide_paths_support, check_network_support
from agent6.app.providers import (
    InstrumentedProvider,
    build_role_provider,
    resolve_compaction_thresholds,
    resolve_decompose,
    reviewer_seat_provider,
)
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.commit_message import render_commit_trailer
from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective_with_overlay
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    chain_tip,
    checkout_detached,
    fetch_branch,
    machine_branch_for,
    machine_chain_ref_for,
)
from agent6.git_ops import status as git_status
from agent6.harness._chain import RunChain, commit_identity
from agent6.harness._compaction import CompactionSettings
from agent6.harness._operator import OperatorBridge
from agent6.harness.loop import Harness
from agent6.harness.subrun import SubrunError, clone_workspace
from agent6.kinds import IsolationLevel
from agent6.machine import AgentExecResult, AgentRequest, validate_record_payload
from agent6.paths import state_dir
from agent6.providers import Provider, TranscriptSink
from agent6.sandbox.jail import die_with_parent
from agent6.sessions.ipc import (
    await_frontend_reply,
    away_mode,
    clear_pending_answers,
    clear_steer_answer,
    clear_steer_request,
    frontend_is_live,
    read_answer,
    read_question_answers,
    read_steer_answer,
    record_answer,
    steer_request_pending,
)
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.operator_prompts import (
    ApprovalAnswer,
    ApprovalRequest,
    OperatorPrompts,
    QuestionAnswer,
    QuestionRequest,
)
from agent6.viewmodel.machine_state import Spend, read_budget_totals


def _no_console(_events: EventSink) -> None:
    """Attach no live view: the headless default."""


class MachineAgentRequest(BaseModel):
    """The `request.json` envelope of the machine-agent subprocess IPC.

    Both sides are the same install, and the bytes are pinned by
    `tests/unit/test_machine_agent_ipc.py`.

    Attributes:
        cwd: The operator's checkout; config layers and cross-run memory resolve from it.
        root: Where the harness, the dispatcher and the jail work: the clone for a
            `mode="run"` state, else `cwd`.
        overlay: The machine's `[config]` overlay, applied over the effective config.
        isolation: The isolation level the engine validated.
        transcript_dir: Where the subprocess records transcripts.
        events_log: Where the subprocess writes a watchable event log, when set.
        protect_paths: The bundle paths the state may not rewrite.
        commit_identity: The host-resolved git identity for a `mode="run"` state, since
            the confined subprocess cannot read `~/.gitconfig`; None for read-only states.
        request: The state's request.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cwd: Path
    root: Path
    overlay: dict[str, Any]
    isolation: IsolationLevel
    transcript_dir: Path
    events_log: Path | None = None
    protect_paths: tuple[Path, ...] = ()
    commit_identity: CommitIdentity | None = None
    request: AgentRequest


def _machine_head_sha(root: Path) -> str | None:
    """Read HEAD at state start, the chain's first parent; None when unreadable.

    Args:
        root: The working tree.

    Returns:
        The sha, or None for an unborn or unreadable repo.
    """
    try:
        return git_status(root).head_sha or None
    except (GitError, OSError):
        return None


def _finish_validator(r: AgentRequest) -> Callable[[dict[str, Any] | None], list[str]] | None:
    """Build the finish-payload check for a state that declares an output schema.

    Args:
        r: The state's request.

    Returns:
        The check the engine also judges the recorded fact with, or None.
    """
    name = r.output_schema
    if name is None:
        return None

    def check(payload: dict[str, Any] | None) -> list[str]:
        return validate_record_payload(r.schemas, name, payload, where="finish_session payload")

    return check


def _task_with_contract(r: AgentRequest) -> str:
    """Render the state prompt plus its finish contract as `field: type` lines.

    Args:
        r: The state's request.

    Returns:
        The task text the execution runs.
    """
    if r.output_schema is None:
        return r.prompt
    lines = [
        f"{r.prompt}",
        "",
        "finish_session must include `result` as a JSON object matching schema"
        f" {r.output_schema!r}:",
    ]
    seen: set[str] = set()
    queue = [r.output_schema]
    while queue:
        name = queue.pop(0)
        if name in seen or name not in r.schemas:
            continue
        seen.add(name)
        parts = []
        for fname, f in r.schemas[name].items():
            t = f.type + (" (optional)" if f.optional else "")
            if f.enum is not None:
                t += " one of [" + ", ".join(f.enum) + "]"
            parts.append(f"{fname}: {t}")
            if f.type in r.schemas:
                queue.append(f.type)
        lines.append(f"  {name} = {{{'; '.join(parts)}}}")
    return "\n".join(lines)


def _result(
    reason: str, payload: dict[str, Any] | None, budget: BudgetTracker | None
) -> AgentExecResult:
    """Build the state's result.

    Args:
        reason: The end reason.
        payload: The finish payload, when the state finished with one.
        budget: The tracker whose spend the result carries; None books zero.

    Returns:
        The result.
    """
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
    """Apply the supervisor's `--auto-approve` and `--no-commands` choices from the env.

    The env is operator-only; a machine `[config]` overlay cannot set `sandbox.*`.

    Args:
        cfg: The state's config.

    Returns:
        The config with the overrides applied.
    """
    return cfg.with_sandbox_overrides(
        auto_approve=os.environ.get("AGENT6_AUTO_APPROVE") == "1",
        no_commands=os.environ.get("AGENT6_NO_COMMANDS") == "1",
    )


@dataclass(frozen=True, slots=True)
class _MachineBridges:
    """The interactivity bridges for one machine `agent` state.

    Answers are read from the per-state dir; the liveness gate probes the instance
    dir, where a front-end registers its claim.

    Attributes:
        prompts: The approval and question gate.
        steer_requested: Whether a steer request is pending.
        steer_clear: Clears the steer request and its answer.
        steer_prompt: Reads the steer answer, or None.
    """

    prompts: OperatorPrompts
    steer_requested: Callable[[], bool]
    steer_clear: Callable[[], None]
    steer_prompt: Callable[[], str | None]


def _build_machine_bridges(
    instance_dir: Path, agent_state: Path, events: EventSink
) -> _MachineBridges:
    """Wire the approval, question and steer bridges to a machine agent state.

    A live front-end is asked in its own UI; otherwise the instance's away-mode
    governs as for a detached run, and a headless machine denies an approval,
    answers a question with "" and takes no steer.

    Args:
        instance_dir: The instance dir, where a front-end registers its claim.
        agent_state: The per-state dir the answers land in.
        events: The per-state log the prompt events go to.

    Returns:
        The bridges.
    """
    # Crash recovery reuses the state dir and its prompt ids, so stale answers go first.
    clear_pending_answers(agent_state, started_at=time.time())

    def approve(request: ApprovalRequest, /) -> ApprovalAnswer:
        if frontend_is_live(instance_dir):
            answer = read_answer(agent_state, request.id, live_dir=instance_dir)
            if answer is not None:
                return ApprovalAnswer(record_answer(agent_state, answer, request.scope), "frontend")
        if away_mode(instance_dir) == "wait":
            reply = await_frontend_reply(
                instance_dir,
                lambda: read_answer(
                    agent_state, request.id, timeout_s=20.0, dead_grace_s=8.0, live_dir=instance_dir
                ),
            )
            approved = reply is not None and record_answer(agent_state, reply, request.scope)
            return ApprovalAnswer(approved, "await-frontend")
        return ApprovalAnswer(False, "headless")  # no operator to ask

    def ask(request: QuestionRequest, /) -> QuestionAnswer:
        empty = tuple("" for _ in request.questions)
        if frontend_is_live(instance_dir):
            answers = read_question_answers(agent_state, request.id, live_dir=instance_dir)
            if answers is not None:
                return QuestionAnswer(answers, "frontend")
        if away_mode(instance_dir) == "wait":
            # Park for the front-end rather than inventing "".
            reply = await_frontend_reply(
                instance_dir,
                lambda: read_question_answers(
                    agent_state, request.id, timeout_s=20.0, dead_grace_s=8.0, live_dir=instance_dir
                ),
            )
            if isinstance(reply, tuple):
                return QuestionAnswer(reply, "frontend")
            return QuestionAnswer(empty, "await-frontend", unseen=True)
        return QuestionAnswer(empty, "headless", unseen=True)

    prompts = OperatorPrompts(
        approver=approve, questioner=ask, journal=events.emit, session_dir=agent_state
    )

    def steer_requested() -> bool:
        return steer_request_pending(agent_state)

    def steer_clear() -> None:
        clear_steer_answer(agent_state)
        clear_steer_request(agent_state)

    def steer_prompt() -> str | None:
        if not frontend_is_live(instance_dir):
            clear_steer_request(agent_state)
            return None
        answer = read_steer_answer(agent_state, live_dir=instance_dir)
        if answer is None:
            clear_steer_request(agent_state)
        return answer

    return _MachineBridges(prompts, steer_requested, steer_clear, steer_prompt)


def _build_agent_providers(
    cfg: Config,
    req: MachineAgentRequest,
    *,
    budget: BudgetTracker,
    attach_console: Callable[[EventSink], None],
) -> tuple[InstrumentedProvider, Provider, EventSink | None]:
    """Build the state's worker provider, its summariser and its event sink.

    The worker always streams: machine agents run headless and generate long, and
    a gateway's SSE heartbeats corrupt a non-streaming body mid-read.

    Args:
        cfg: The state's config.
        req: The state's request envelope.
        budget: The tracker both providers bill.
        attach_console: Given the event sink, attaches the live view.

    Returns:
        The instrumented worker, the summariser, and the sink or None without a log.
    """
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
        role="worker",
        model=rm.model if rm is not None else "",
        provider_name=rm.provider if rm is not None else "",
        events=events_sink,
        budget=budget,
        stream_text=True,
    )
    summariser_provider = reviewer_seat_provider(
        cfg, "summariser", transcript_sink=transcript_sink, budget=budget, events=events_sink
    )
    return provider, summariser_provider, events_sink


def run_one(
    req: MachineAgentRequest,
    *,
    attach_console: Callable[[EventSink], None] = _no_console,
    reporter: Reporter = STDIO_REPORTER,
) -> AgentExecResult:
    """Run one machine `agent` state to completion inside its subprocess.

    Args:
        req: The request envelope.
        attach_console: Given the event sink, attaches the live view.
        reporter: Receives the refusals and the loop's log lines.

    Returns:
        The state's result; a config or isolation refusal is an error result.
    """
    isolation = req.isolation
    r = req.request
    # A config error becomes an error result, not a traceback the host must salvage.
    try:
        cfg = load_effective_with_overlay(req.cwd, req.overlay).config.with_machine_agent_overrides(
            provider=r.provider,
            model=r.model,
            effort=r.effort,
            temperature=r.temperature,
            max_usd=r.max_usd,
            max_tokens_fallback=r.max_tokens_fallback,
        )
        cfg = _apply_operator_env_grants(cfg)
    except (ConfigError, ValidationError) as exc:
        reporter.refuse(f"machine agent config error: {exc}")
        return _result("error", None, None)
    apply_git_ops_policy(cfg)
    # The confined process cannot read ~/.gitconfig, so the host-resolved identity is exported.
    if req.commit_identity is not None:
        if name := req.commit_identity.name:
            os.environ["GIT_AUTHOR_NAME"] = os.environ["GIT_COMMITTER_NAME"] = name
        if email := req.commit_identity.email:
            os.environ["GIT_AUTHOR_EMAIL"] = os.environ["GIT_COMMITTER_EMAIL"] = email
    # The engine validated the isolation already; re-check and fail closed.
    net_err = check_network_support(cfg, isolation)
    if net_err is not None:
        reporter.refuse(net_err)
        return _result("error", None, None)
    hide_err = check_hide_paths_support(cfg, isolation, req.root)
    if hide_err is not None:
        reporter.refuse(hide_err)
        return _result("error", None, None)
    budget = budget_tracker(cfg)
    provider, summariser_provider, events_sink = _build_agent_providers(
        cfg, req, budget=budget, attach_console=attach_console
    )
    # Re-confirm the containment invariant at the subprocess boundary.
    root_r = req.root.resolve()
    protect = tuple(rp for p in req.protect_paths if (rp := p.resolve()).is_relative_to(root_r))
    # "machine" and "agent" are read-only loops: the dispatcher refuses edits and commands.
    mode = r.mode
    read_only = mode in ("machine", "agent")
    # The bridges need a per-state log for the front-end to see the prompt.
    bridges: _MachineBridges | None = None
    if events_sink is not None and req.events_log is not None:
        agent_state = req.events_log.parent
        instance_dir = req.transcript_dir.parent
        bridges = _build_machine_bridges(instance_dir, agent_state, events_sink)
    dispatcher = ToolDispatcher(
        root=req.root,
        config=cfg,
        isolation=isolation,
        prompts=bridges.prompts if bridges is not None else None,
        events=events_sink,
        curator=None,
        run_root_node_id=None,
        mcp_manager=None,
        extra_protect_paths=protect,
        mode="machine" if read_only else "run",
        # The repo's state dir, keyed on the checkout: a clone has no memory of its own.
        state_dir=state_dir(req.cwd),
    )
    rm = cfg.models.resolve("worker")
    compact_drop, compact_summarise, keep_recent = resolve_compaction_thresholds(
        cfg, rm, log=reporter.err
    )
    cfg = resolve_decompose(cfg, rm, log=reporter.err)
    wf = Harness(
        # A run-mode state commits on its own chain, named by the instance dir.
        chain=RunChain(
            req.root,
            ref=machine_chain_ref_for(req.transcript_dir.parent.name) if not read_only else None,
            fallback_parent=_machine_head_sha(req.root) if not read_only else None,
            identity=commit_identity(
                cfg.git.commit,
                render_commit_trailer(
                    cfg.git.commit.trailer, models=(rm.model if rm is not None else "",)
                ),
            ),
            per_step=cfg.git.commit_per_step,
        ),
        config=cfg,
        max_iterations=cfg.harness.max_iterations,
        provider=provider,
        dispatcher=dispatcher,
        logger=reporter.err,
        mode="agent" if mode == "agent" else "run",
        state_dir=state_dir(req.cwd),
        compaction=CompactionSettings(
            drop_at_chars=compact_drop,
            summarise_at_chars=compact_summarise,
            tool_result_cap_bytes=tool_result_cap_bytes(cfg, "worker"),
            keep_recent_chars=keep_recent,
            keep_thinking_turns=cfg.context.keep_thinking_turns,
            elision_gists=cfg.context.elision_gists,
            summary_max_tokens=cfg.context.summary_max_tokens,
            summariser=summariser_provider,
        ),
        bridge=OperatorBridge(
            steer_requested=bridges.steer_requested,
            steer_clear=bridges.steer_clear,
            steer_prompt=bridges.steer_prompt,
        )
        if bridges is not None
        else OperatorBridge(),
        finish_validator=_finish_validator(r),
    )
    result = wf.run(_task_with_contract(r))
    payload = result.finish_payload if result.reason == "finish_session" else None
    return _result(result.reason, payload, budget)


def build_machine_agent_runner(
    overlay: dict[str, Any],
    cwd: Path,
    isolation: IsolationLevel,
    transcript_dir: Path,
    protect_paths: tuple[Path, ...] = (),
    commit_identity: CommitIdentity | None = None,
    machine_id: str | None = None,
    clone_root: Path | None = None,
) -> Callable[[AgentRequest, Path | None], AgentExecResult]:
    """Build the host-side runner an `agent` state fires.

    The runner spawns the subprocess with a fixed argv, hands it the request through
    a temp file (the prompt never rides the command line) and enforces the timeout
    by killing the whole process group. With `machine_id` and `clone_root` set,
    every state executes in a fresh clone checked out at the machine chain's tip,
    and a run-mode state's commits land back on the chain ref and the visible
    machine branch; the operator's checkout is never touched.

    Args:
        overlay: The machine's `[config]` overlay.
        cwd: The operator's checkout.
        isolation: The isolation level the engine validated.
        transcript_dir: Where the subprocess records transcripts.
        protect_paths: The bundle paths a state may not rewrite.
        commit_identity: The host-resolved git identity for run-mode states.
        machine_id: The machine's id, when its states run in clones.
        clone_root: Where the per-state clones live, when they do.

    Returns:
        The runner: given a request and an optional per-call event log, the result.
    """

    def run_agent(request: AgentRequest, events_log: Path | None = None) -> AgentExecResult:
        # The salvage reads this call's events only: `machine create` shares one draft log.
        start_offset = 0
        if events_log is not None:
            with contextlib.suppress(OSError):
                start_offset = events_log.stat().st_size

        def salvaged(reason: str) -> AgentExecResult:
            # Without a result file the spend comes from the event log, so the guard trips.
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

        clone: Path | None = None
        chain = machine_chain_ref_for(machine_id) if machine_id is not None else None
        if chain is not None and clone_root is not None:
            clone = clone_root / f"state-{request.step_seq:04d}"
            try:
                clone_at_machine_chain(cwd, clone, chain)
            except (SubrunError, GitError) as exc:
                return salvaged(f"error: clone for machine {machine_id!r} failed: {exc}")
        workdir = clone or cwd
        payload = MachineAgentRequest(
            # `cwd` stays the checkout: config and memory load from it; `root` is the work.
            cwd=cwd,
            root=workdir,
            overlay=overlay,
            isolation=isolation,
            transcript_dir=transcript_dir,
            events_log=events_log,
            # The clone's copy of the bundle is protected too, though the origin's executes.
            protect_paths=tuple(
                workdir / p.relative_to(cwd) if clone is not None and p.is_relative_to(cwd) else p
                for p in protect_paths
            ),
            commit_identity=commit_identity,
            request=request,
        )
        with tempfile.TemporaryDirectory(prefix="agent6-machine-agent-") as td:
            req_file = Path(td) / "request.json"
            out_file = Path(td) / "result.json"
            req_file.write_text(payload.model_dump_json(), encoding="utf-8")
            argv = [
                sys.executable,
                # -P keeps the workspace off sys.path, so a planted `agent6/` cannot shadow.
                "-P",
                "-m",
                "agent6.ui.cli.machine_agent",
                str(req_file),
                str(out_file),
            ]
            # An own process group so the timeout kill takes the jail children too; PDEATHSIG
            # so the tree dies with the supervisor. The preexec hook is async-signal-minimal.
            try:
                proc = subprocess.Popen(
                    argv,
                    start_new_session=True,
                    env={**os.environ, "AGENT6_SUBRUN": "1"},
                    preexec_fn=die_with_parent(os.getpid()),  # noqa: PLW1509
                )
            except OSError as exc:
                result = salvaged(f"error: machine agent failed to start: {exc}")
            else:
                try:
                    proc.wait(timeout=request.timeout_s)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        # By pid: the unreaped leader's pgid cannot have been recycled.
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    result = salvaged("timeout")
                else:
                    if proc.returncode != 0 or not out_file.is_file():
                        result = salvaged("error")
                    else:
                        try:
                            result = AgentExecResult.model_validate_json(out_file.read_bytes())
                        except (OSError, ValidationError):
                            # A malformed result file counts as missing.
                            result = salvaged("error")
        if clone is not None and chain is not None and machine_id is not None:
            result = _land_machine_clone(cwd, clone, chain, machine_branch_for(machine_id), result)
        return result

    return run_agent


def clone_at_machine_chain(origin: Path, dest: Path, chain_ref: str) -> None:
    """Make a fresh clone checked out at the machine chain's tip.

    A clone copies branches, not `refs/agent6/*`, so the chain ref is fetched in;
    with no chain yet the clone's own HEAD is the start.

    Args:
        origin: The repository to clone.
        dest: Where the clone goes; an existing one is replaced.
        chain_ref: The machine's chain ref.

    Raises:
        SubrunError: The clone failed.
        GitError: The fetch or checkout failed.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    clone_workspace(origin, dest)
    tip = chain_tip(origin, chain_ref)
    if tip is not None:
        fetch_branch(dest, origin, f"{chain_ref}:{chain_ref}")
        checkout_detached(dest, tip)


def _land_machine_clone(
    origin: Path, clone: Path, chain_ref: str, branch: str, result: AgentExecResult
) -> AgentExecResult:
    """Land the state's work back in the origin on every outcome, and drop the clone.

    Serial states make both ref updates fast-forwards. An import failure keeps the
    clone, the only copy, and routes the state as failed with no payload.

    Args:
        origin: The repository the work lands in.
        clone: The state's clone.
        chain_ref: The machine's chain ref.
        branch: The visible machine branch.
        result: The state's result.

    Returns:
        The result, or a failed one when the import failed.
    """
    advanced = chain_tip(clone, chain_ref)
    if advanced is None or advanced == chain_tip(origin, chain_ref):
        shutil.rmtree(clone, ignore_errors=True)
        return result
    try:
        fetch_branch(origin, clone, f"{chain_ref}:{chain_ref}")
        fetch_branch(origin, clone, f"{chain_ref}:refs/heads/{branch}")
    except GitError as exc:
        return result.model_copy(
            update={"reason": f"import of machine work failed: {exc}", "payload": None}
        )
    shutil.rmtree(clone, ignore_errors=True)
    return result
