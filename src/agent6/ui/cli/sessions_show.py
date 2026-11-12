# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions show`: one-shot liveness and progress of a session from its dir.

Reads worker.pid, the log scan and the manifest's branch facts; text or `--json`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import time
from collections.abc import Mapping
from typing import Any, TypedDict

from agent6 import kinds, terminal
from agent6.app import merge, resume
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.sessions import process
from agent6.ui.cli import common
from agent6.vcs.git import read as git_read
from agent6.viewmodel import (
    LogScan,
    SessionSummary,
    comparison,
    existing_run_branch,
    format,
    listing,
    policy,
    receipt,
    scan_session_log,
    summarize_session_dir,
    tail,
    transcript,
    wire,
)
from agent6.viewmodel import lanes as vm_lanes
from agent6.viewmodel.session import view as state_view


def _fmt_dur(seconds: float | None) -> str:
    """Return seconds as a short duration, or "?" for None."""
    if seconds is None:
        return "-"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _print_lineage(manifest: sessions_manifest.SessionManifest) -> None:
    """Print the session `--from` seeded this one from, and a fork's lineage and worktree."""
    if manifest.source_session_id:
        print(f"seeded from: {manifest.source_session_id}")
    lineage = format.format_lineage(
        manifest.parent_session_id, manifest.forked_from_turn, manifest.forked_from_sha
    )
    if lineage:
        print(f"forked from: {lineage}")
    if manifest.worktree is not None:
        missing = sessions_manifest.worktree_blocker(manifest.worktree)
        print(missing or f"worktree:   {manifest.worktree}")


def _print_parallel_compare(
    manifest: sessions_manifest.SessionManifest, session_dir: pathlib.Path
) -> None:
    """Print a lane's place in its group; a no-op for any other run.

    The parent it nests under, then the compare outcome once there is one: its place,
    whether it won, judged or mechanical, and the judge's rationale.

    Args:
        manifest: The run's manifest.
        session_dir: The run's directory, used to read its parent.
    """
    if (lineage := manifest.parallel) is not None:
        print(f"lane of:    {lineage.parent} (lane {lineage.lane} of group {lineage.group})")
    formatted = format.format_compare(
        manifest.placement, comparison.session_comparison(session_dir)
    )
    if formatted is None:
        return
    headline, rationale = formatted
    print(f"compare:    {headline}")
    if rationale:
        print(f"  judge: {rationale}")


def _lanes(layout: sessions_layout.SessionLayout, tips: Mapping[str, str]) -> list[SessionSummary]:
    """Return the session's lanes by lineage with their unmerged marks.

    Args:
        layout: The session's layout.
        tips: The branch-tips snapshot.

    Returns:
        The lanes in lane order; empty for a session without any.
    """
    return listing.lanes_of(layout.state_dir, layout.session_id, branch_tips=tips)


def _won(manifest: sessions_manifest.SessionManifest | None) -> bool:
    """Return whether the manifest's compare stamp names it the winner."""
    return manifest is not None and manifest.placement is not None and manifest.placement.winner


def _lane_manifests(
    state: pathlib.Path, lanes: list[SessionSummary]
) -> dict[str, sessions_manifest.SessionManifest]:
    """Return each readable lane manifest, for its route and compare stamp."""
    manifests: dict[str, sessions_manifest.SessionManifest] = {}
    for lane in lanes:
        lane_layout = sessions_layout.session_layout(state, lane.session_id)
        if lane_layout is not None:
            with contextlib.suppress(sessions_manifest.ManifestError):
                manifests[lane.session_id] = sessions_manifest.read_manifest(
                    lane_layout.session_dir
                )
    return manifests


def _print_lanes(
    manifest: sessions_manifest.SessionManifest,
    lanes: list[SessionSummary],
    lane_manifests: Mapping[str, sessions_manifest.SessionManifest],
    outcomes: Mapping[str, str],
) -> None:
    """Print the group line for a group, the lane count otherwise, then one line per lane.

    Args:
        manifest: The session's manifest.
        lanes: Its lanes by lineage.
        lane_manifests: Each readable lane manifest, for its compare stamp.
        outcomes: How each recorded lane joined, by session id.
    """
    if lanes:
        print(f"lanes:      {len(lanes)}")
    idents = {
        lane.session_id: lane.id_cell(winner=_won(lane_manifests.get(lane.session_id)))
        for lane in lanes
    }
    width = max((len(ident) for ident in idents.values()), default=0)
    for lane in lanes:
        lane_manifest = lane_manifests.get(lane.session_id)
        stamp = lane_manifest.placement if lane_manifest is not None else None
        place = f"rank {stamp.rank}/{stamp.of}" if stamp is not None else f"lane {lane.lane}"
        model = lane.model or "?"
        cost = lane.usage_cell
        label = format.listing_status_label(
            lane.mode, lane.status, lane.detail, unmerged=lane.unmerged
        )
        outcome = outcomes.get(lane.session_id, "")
        joined = f"  {outcome}" if outcome else ""
        ident = idents[lane.session_id]
        print(f"  {place:<10} {ident:<{width}}  {model}  {label}  {cost}{joined}".rstrip())


def lane_outcomes(session_dir: pathlib.Path) -> dict[str, str]:
    """Return each recorded lane's join as one phrase, by session id.

    Args:
        session_dir: The session's dir.

    Returns:
        `joined at <sha>`, `no changes` or `<outcome>: <detail>` per lane; empty without records.
    """
    phrases: dict[str, str] = {}
    for child in vm_lanes.children_of(session_dir):
        if child.outcome == "joined":
            phrases[child.session_id] = f"joined at {child.join_sha[:12]}"
        elif child.outcome == "no_changes":
            phrases[child.session_id] = "no changes"
        elif child.outcome:
            detail = f": {child.error}" if child.error else ""
            phrases[child.session_id] = f"{child.outcome}{detail}"
    return phrases


def _status_state(
    row: SessionSummary, scan: LogScan, *, last_age: float | None
) -> tuple[str, str, str]:
    """Return the run's state as `(status, label, detail)`.

    The text render joins them; `--json` emits all three, so a script never parses prose.

    Args:
        row: The listing's fold of this run.
        scan: The log scan.
        last_age: Seconds since the last event, when there is one.

    Returns:
        The status is the listing's own word and the label its own rendering, so no second
        rule can disagree with the listing about the mode fold or the unmerged mark. The
        detail is this surface's diagnostic: what to do, or why the word applies; "" when
        there is none.
    """
    word, reason = row.status, row.detail
    cell = format.listing_status_label(row.mode, row.status, row.detail, unmerged=row.unmerged)
    if word == "unreadable":
        # The record, not the journal, decides: a newer or corrupt manifest names itself.
        return word, cell, format.dead_run_note(word, reason)[0]
    if scan.finished:
        if scan.end_summary:
            # The `ended:` line carries the run's own sentence; the raw reason would repeat it.
            return word, cell, ""
        if scan.stop_reason == "provider_error":
            # The label names the failure; the provider's own text, on one clean line, says why.
            error = terminal.scrub_terminal_controls(scan.provider_error)
            return word, cell, " ".join(error.split())
        # The raw end reason is the diagnostic, unless the label already carries it.
        end = "" if scan.stop_reason in (word, *reason.split(" · ")) else scan.stop_reason
        return word, cell, end
    detail = {
        "waiting": "needs answer; attach to respond",
        "stale": format.dead_run_note("stale", "")[0],
        "parked": f"{reason}; resume to start" if reason else "resume to start",
        # A log holding preflight events from a worker that died launching is "never started".
        "created": "no events yet" if scan.last_type is None else "never started",
    }.get(word, "")
    if word == "running" and last_age is not None and last_age > 120:
        detail = _IN_FLIGHT.get(scan.last_type or "", "long step")
    return word, cell, detail


# The status words of a session that stopped short of its end, which `agent6 resume` continues.
_RESUMABLE = frozenset({"stale", "stopped", "failed"})

# What a long-running step is doing, by the event the journal recorded last.
_IN_FLIGHT = {
    "verify.start": "verify command running",
    "tool.call": "tool call running",
    "model.call": "provider call in flight",
    "model.text_delta": "provider call in flight",
    "model.thinking_delta": "provider call in flight",
}


def _pid_note(pid: int | None, *, alive: bool, finished: bool) -> str:
    """Return the status line's worker suffix: alive, recycled, or not running."""
    if alive:
        return f"  (worker pid {pid} alive)"
    if pid is None or finished:
        return ""
    # Liveness matches the recorded start time, so a recycled pid reads dead, not "not running".
    return (
        f"  (worker pid {pid} was recycled)"
        if process.pid_alive(pid)
        else f"  (worker pid {pid} not running)"
    )


# What the run's last end said the gate made of the final tree.
_VERIFY_LINES = {
    "passed": "green on the final tree",
    "failed": "red on the final tree",
    "timed_out": "timed out on the final tree",
    "unverified": "not verified: no verify covered the final tree",
    "not_applicable": "no gate",
}


def _print_end_facts(
    scan: LogScan,
    session_id: str,
    *,
    gated: bool,
    resume: str | None,
    attempts: tuple[str, ...],
) -> None:
    """Print unanswered questions, the gate's word and an unfinished run's resume command.

    Args:
        scan: The log scan.
        session_id: The session.
        gated: The session is a run, the one mode a gate judges.
        resume: "" when the session stopped short of its end and a bare resume continues it,
            why a resume would refuse when it stopped short but cannot, None for an end that
            needs no resume.
        attempts: The transcript's verification attempt lines.
    """
    failed = scan.stop_reason == "setup_command_failed"
    setup = scan.setup_output
    if scan.finished and scan.end_summary:
        ended = terminal.scrub_terminal_controls(scan.end_summary)
        if failed:
            # The diagnostic is the sentence, then the output: each prints once.
            ended, _, setup = ended.partition("\n")
        # The run's own end sentence, on one clean line.
        print(f"ended:      {' '.join(ended.split())}")
    if setup:
        # A setup that worked is one line; a failed one keeps its output, the diagnostic.
        lines = [line.strip() for line in setup.splitlines()] if failed else [setup.splitlines()[0]]
        shown = "\n            ".join(lines)
        print(f"setup:      {shown.removeprefix('setup command: ')}")
    if n := scan.unattended_questions:
        # Answered empty by the harness in this execution; a resume starts the count again.
        print(
            f"questions:  {n} unanswered (nobody was attached):"
            f" agent6 sessions transcript {session_id}"
        )
    # An end with no word (a crash, a log written before the field) prints no line.
    if gated and (verify := _VERIFY_LINES.get(scan.gate_word())):
        if scan.baseline_ok is not None:
            verify += f" ({'green' if scan.baseline_ok else 'red'} before the run)"
        print(f"verify:     {verify}")
    elif attempts:
        print("verify:")
    for attempt in attempts:
        print("            " + attempt.replace("\n", "\n" + " " * 12))
    if resume == "":
        print(f"resume with:  agent6 resume {session_id}")
    elif resume is not None:
        print(f"resume:     {resume}")


def _usage_line(scan: LogScan) -> str:
    """Return the run's tokens and cost line.

    Token counters and the points against the cap are per execution and the cost is banked
    across executions, so a resumed run says which is which.

    Args:
        scan: The run's log scan.

    Returns:
        `in` and `out`, the cached side when the journal recorded it, the plan movement a
        percent-metered execution observed against its cap, then the usage as the listing's
        cell spells it (blank for a clean $0).
    """
    run = scan.budget.execution
    tokens = f"in={run.input_tokens or 0} out={run.output_tokens or 0}"
    if run.cache_read_tokens is not None or run.cache_write_tokens is not None:
        tokens += f" cache_r={run.cache_read_tokens or 0}"
        tokens += f" cache_w={run.cache_write_tokens or 0}"
    if run.plan_cap is not None and run.plan_cap > 0:
        tokens += f" plan moved={run.plan_observed_points:g}/{run.plan_cap:g}pt"
    execution_s = " (latest execution)" if scan.executions > 1 else ""
    cell = scan.usage_cell
    executions_s = f" (all {scan.executions} executions)" if scan.executions > 1 else ""
    cost_s = f"  usage {cell}{executions_s}" if cell else ""
    return f"{tokens}{execution_s}{cost_s}"


class StatusJSON(TypedDict):
    """The status payload, with meanings on the command."""

    session_id: str
    mode: str | None
    task: str
    continued_task: str
    task_line: str
    model: str
    models: dict[str, Any]
    model_from_flag: bool
    preset: str | None
    status: str
    status_label: str
    status_detail: str
    end_summary: str | None
    alive: bool
    pid: int | None
    turn: int | None
    last_event: str | None
    last_event_age_s: float | None
    elapsed_s: float | None
    stop_reason: str | None
    baseline_ok: bool | None
    verification: kinds.Verification | None
    budget: dict[str, Any]
    unattended_questions: int
    setup_output: str | None
    environment_evidence: dict[str, Any]
    source_session_id: str | None
    parent_session_id: str | None
    forked_from_turn: int | None
    forked_from_sha: str | None
    worktree: str | None
    placement: dict[str, Any] | None
    comparison: dict[str, Any] | None
    parallel: dict[str, Any] | None
    lanes: list[listing.SummaryRow]
    run_branch: str | None
    base_branch: str | None
    merged_into: str | None
    pins: list[str]


def _cmd_status(session_id: str, *, as_json: bool = False) -> int:
    """Print a one-shot liveness and progress summary for a run.

    Answered from the run dir alone: worker.pid probed with signal 0, so liveness is known
    even while the worker is blocked in a long provider call, plus the last event, the
    turn and the elapsed time from logs.jsonl. `agent6 attach` is the live follower.

    Attributes:
        mode: The session mode.
        task: The operator's words, whole.
        continued_task: The latest resume steer, whole; empty without a continuation.
        task_line: The task clipped to 200 characters, then a newline and `continued: <steer>`,
            clipped the same way, when a resume steer exists.
        model: The resolved provider/model route, or ? when unknown.
        models: The recorded driver, reviewer, helper and review seats, with effort and origin.
        model_from_flag: Whether a model flag selected the route.
        preset: The launch preset, or null when absent.
        status: The lifecycle status word shared with listings.
        status_label: The rendered status with mode and unmerged marks.
        status_detail: The diagnostic or next action explaining the status.
        end_summary: The journaled closing sentence, or null when absent.
        alive: Whether the recorded worker identity is alive.
        pid: The recorded worker process identifier, or null when unreadable.
        turn: The latest turn number, or null before a turn.
        last_event: The latest event name, or null without a log.
        last_event_age_s: Seconds since the latest event, or null when unknown.
        elapsed_s: Seconds from the first start to the last event or current time.
        stop_reason: The journaled end reason, or null while unfinished.
        baseline_ok: Whether baseline verification passed, or null when unrecorded.
        verification: The final gate word, or null when unrecorded.
        budget: The BudgetView with its rendered `usd_text` and `plan_text`, as `attach --json`
            carries it.
        unattended_questions: Questions answered empty because nobody was attached.
        setup_output: Retained setup output, or null when absent.
        environment_evidence: Recorded targets, setup and lifecycle observations.
        source_session_id: The source session used by run --from, or null.
        parent_session_id: The fork parent, or null.
        forked_from_turn: The fork's source turn, or null.
        forked_from_sha: The fork's source commit, or null.
        worktree: The linked worktree path, or null.
        placement: The lane's rank and group id, or null outside a compared lane.
        comparison: The group record on its parent, or null for other sessions.
        parallel: The ParallelLineage, or null outside a lane.
        lanes: The parent's summary_row objects.
        run_branch: The existing run branch, or null.
        base_branch: The base branch, or null when unknown.
        merged_into: The merge destination, or null when not merged.
        pins: The operator's active pinned instructions.

        session_id: The session identifier.

    Args:
        session_id: The session, or "" for the newest.
        as_json: Print one JSON object.

    Returns:
        The exit code; 2 when the session cannot be resolved.
    """
    layout = common.resolve_target(session_id)
    if layout is None:
        return 2
    target = layout.session_dir

    loaded: sessions_manifest.SessionManifest | None = None
    with contextlib.suppress(sessions_manifest.ManifestError):
        loaded = sessions_manifest.read_manifest(target)
    # A missing manifest still renders, with `mode` as "?" rather than the model default.
    manifest = loaded or sessions_manifest.SessionManifest()
    mode_display = loaded.mode if loaded is not None else None

    scan = scan_session_log(layout.logs_path) if layout.logs_path.is_file() else LogScan()

    pid = process.read_worker_pid(target)
    alive = process.worker_is_alive(target)
    last_age = (time.time() - scan.last_ep) if scan.last_ep is not None else None
    # A live run is still elapsing; a finished or dead one stopped at its last event.
    elapsed = (
        ((time.time() if alive and not scan.finished else scan.last_ep) - scan.start_ep)
        if scan.last_ep is not None and scan.start_ep is not None
        else None
    )

    model = format.format_model_route(manifest.models.driver) or "?"
    model_from_flag = manifest.models.driver_from_flag
    changes = _changes(target.name, manifest, undone=scan.finished and scan.stop_reason == "undone")
    tips = git_read.run_ref_tips(pathlib.Path.cwd())
    summary = summarize_session_dir(target, branch_tips=tips)
    lanes = _lanes(layout, tips)
    lane_manifests = _lane_manifests(layout.state_dir, lanes)
    status, status_cell, status_detail = _status_state(
        summary,
        scan,
        last_age=last_age,
    )
    state = f"{status_cell} ({status_detail})" if status_detail else status_cell

    def status_json() -> StatusJSON:
        """Return the public status payload."""
        return {
            "session_id": target.name,
            "mode": mode_display,
            "task": summary.task_label.task,
            "continued_task": summary.task_label.continued_task,
            "task_line": summary.task_label.line(),
            "model": model,
            "models": manifest.models.model_dump(mode="json"),
            "model_from_flag": model_from_flag,
            "preset": manifest.harness.preset or None,
            "status": status,
            "status_label": status_cell,
            "status_detail": status_detail,
            "end_summary": scan.end_summary or None,
            "alive": alive,
            "pid": pid,
            "turn": scan.turn,
            "last_event": scan.last_type,
            "last_event_age_s": round(last_age, 1) if last_age is not None else None,
            "elapsed_s": round(elapsed, 1) if elapsed is not None else None,
            "stop_reason": scan.stop_reason if scan.finished else None,
            "baseline_ok": scan.baseline_ok,
            "verification": scan.verification or None,
            "budget": state_view.budget_dict(scan.budget),
            "unattended_questions": scan.unattended_questions,
            "setup_output": scan.setup_output or None,
            "environment_evidence": receipt.environment_dict(scan),
            "source_session_id": manifest.source_session_id,
            "parent_session_id": manifest.parent_session_id,
            "forked_from_turn": manifest.forked_from_turn,
            "forked_from_sha": manifest.forked_from_sha,
            "worktree": str(manifest.worktree) if manifest.worktree else None,
            "placement": manifest.placement.model_dump(mode="json") if manifest.placement else None,
            "comparison": manifest.comparison.model_dump(mode="json")
            if manifest.comparison
            else None,
            "parallel": manifest.parallel.model_dump(mode="json") if manifest.parallel else None,
            "lanes": [
                listing.summary_row(ln, winner=_won(lane_manifests.get(ln.session_id)))
                for ln in lanes
            ],
            "run_branch": existing_run_branch(manifest, pathlib.Path.cwd()) or None,
            "base_branch": manifest.base_branch or None,
            "merged_into": changes.merged_into or None,
            "pins": list(scan.pins),
        }

    if as_json:
        print(json.dumps(status_json()))
        return 0

    pid_note = _pid_note(pid, alive=alive, finished=scan.finished)
    print(f"session:    {target.name}  (mode={mode_display or '?'})")
    for prefix, task in [
        ("task:       ", summary.task_label.task),
        *(
            ("            continued: ", words)
            for words in (
                [
                    str(event.get("text", ""))
                    for event in tail.tail_events(layout.logs_path, follow=False)
                    if event.get("type") == "steer.injected"
                ]
                or [summary.task_label.continued_task]
            )
            if words
        ),
    ]:
        print(prefix + task.replace("\n", "\n" + " " * 12))
    _print_lineage(manifest)
    _print_parallel_compare(manifest, target)
    _print_lanes(manifest, lanes, lane_manifests, lane_outcomes(target))
    print(
        f"model:      {model}{' (from --model)' if model_from_flag else ''}\n"
        f"preset:     {manifest.harness.preset or '-'}"
        + "".join(f"\nmodels:     {line}" for line in policy.model_lines(manifest))
    )
    print(
        terminal.scrub_terminal_controls(receipt.environment_summary(scan))
        + f"\nstatus:     {state}{pid_note}"
    )
    print(f"turn:       {scan.turn if scan.turn is not None else '-'}")
    print(
        f"last event: {scan.last_type or '-'}"
        f"{f'  ({_fmt_dur(last_age)} ago)' if last_age is not None else ''}"
    )
    # First start to last event: the time between executions is in it.
    spans = f" (wall, {scan.executions} executions)" if scan.executions > 1 else ""
    print(f"elapsed:    {_fmt_dur(elapsed)}{spans}")
    if scan.budget_seen:
        print(f"usage:      {_usage_line(scan)}")
    _print_end_facts(
        scan,
        target.name,
        gated=mode_display == "run",
        resume=listing.resume_blocker(target) if status in _RESUMABLE else None,
        attempts=tuple(
            item.body
            for item in transcript.fold_transcript(
                list(tail.tail_events(layout.logs_path, follow=False)), worker_dead=not alive
            )
            if item.kind == "verify"
        ),
    )
    if changes.line:
        print(f"changes:    {changes.line}")
    if mode_display == "plan" and (target / "plan.md").is_file():
        print(f"plan:       agent6 plan show {target.name}")
    for i, pin in enumerate(scan.pins):
        print(f"{'pins:' if i == 0 else '':<12}{pin}")
    _print_listening_ports(target)
    _print_task_tree(target)
    return 0


@dataclasses.dataclass(frozen=True, slots=True)
class _Changes:
    """Where the run's work lives, as the text row and the merge base.

    Attributes:
        line: The text row; "" for a session with no run branch.
        merged_into: The base the run branch is merged into, else "".
    """

    line: str
    merged_into: str


def _changes(
    session_id: str, manifest: sessions_manifest.SessionManifest, *, undone: bool
) -> _Changes:
    """Return where the run's work lives, checked against git as the end-of-run footer does.

    Merged into the base, on the run branch awaiting `sessions merge`, on the hidden chain
    ref alone (the branch deleted, the commits kept), nowhere (no commit was made), or
    taken back by `/undo`.

    Args:
        session_id: The session.
        manifest: Its manifest.
        undone: `/undo` took the work back, so no merge is offered.

    Returns:
        The text row ("" without a run branch) and the base the work is merged into ("" when none).
    """
    run_branch = manifest.run_branch or ""
    if manifest.git_control == "model" or not run_branch:
        ref = (
            wire.commits_ref(manifest, pathlib.Path.cwd())
            if manifest.git_control == "model"
            else ""
        )
        return _Changes(f"work is on {ref}" if ref else "", "")
    if undone:
        return _Changes(f"{run_branch} (taken back by /undo)", "")
    cwd = pathlib.Path.cwd()
    stamp = resume.covering_stamp(cwd, manifest)
    if stamp is not None:
        return _Changes(
            format.format_branch(run_branch, manifest.base_branch, stamp.into), stamp.into
        )
    merge_hint = f"merge with: {merge.merge_command(session_id, manifest.base_branch)}"
    chain = git_read.chain_ref_for(session_id)
    branch_kept = git_read.branch_exists(cwd, run_branch)
    tip = (
        git_read.branch_tip_sha(cwd, run_branch) if branch_kept else git_read.chain_tip(cwd, chain)
    )
    if tip is None or tip == manifest.base_sha:
        # A fork cut before its source's first commit sits on the base with no commit of its own.
        return _Changes("none (the run made no commit)", "")
    if not branch_kept:
        return _Changes(f"{chain} ({run_branch} is gone; the commits are kept); {merge_hint}", "")
    return _Changes(
        f"{format.format_branch(run_branch, manifest.base_branch, '')}; {merge_hint}", ""
    )


def _print_listening_ports(session_dir: pathlib.Path) -> None:
    """Print what the run is serving, and how to reach it.

    A run's commands share a network with no way in from outside, so a dev server the agent
    started is invisible here, its port included; the line names the command that opens it.

    Args:
        session_dir: The run's dir.
    """
    ports = process.listening_ports(session_dir)
    if not ports:
        return
    listed = ", ".join(str(p) for p in ports)
    print(f"serving:    {listed} (inside the run)")
    print(f"            open one: agent6 forward {session_dir.name} {ports[0]}")


def _print_task_tree(session_dir: pathlib.Path) -> None:
    """Print the run's task tree when it decomposed into subtasks; a single root is skipped."""
    from agent6.graph import storage  # noqa: PLC0415
    from agent6.ui.cli import task_tree  # noqa: PLC0415

    layout = sessions_layout.layout_of(session_dir)
    nodes = storage.load_graph(layout)
    if len(nodes) <= 1:
        return
    lines = task_tree.task_tree_lines({nid: nodes[nid].model_dump() for nid in sorted(nodes)})
    if lines:
        print("\ntasks:")
        for line in lines:
            print(f"  {line}")
