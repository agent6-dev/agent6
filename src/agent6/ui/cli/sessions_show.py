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

from agent6 import git_ops, terminal
from agent6.app import merge, resume
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.sessions import process
from agent6.ui.cli import _common
from agent6.viewmodel import (
    LogScan,
    SessionSummary,
    existing_run_branch,
    format,
    listing,
    scan_session_log,
    summarize_session_dir,
)


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
        gone = "" if manifest.worktree.is_dir() else " (gone)"
        print(f"worktree:   {manifest.worktree}{gone}")


def _print_parallel_compare(manifest: sessions_manifest.SessionManifest) -> None:
    """Print a lane's place in its fan-out; a no-op for any other run.

    The coordinator it nests under, then the compare outcome once there is one: its place,
    whether it won, judged or mechanical, and the judge's rationale.

    Args:
        manifest: The run's manifest.
    """
    if (lineage := manifest.parallel) is not None:
        print(f"lane of:    {lineage.coordinator} (lane {lineage.lane} of group {lineage.group})")
    formatted = format.format_compare(manifest.compare)
    if formatted is None:
        return
    headline, rationale = formatted
    print(f"compare:    {headline}")
    if rationale:
        print(f"  judge: {rationale}")


def _fanout_lanes(
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
    tips: Mapping[str, str],
) -> list[SessionSummary]:
    """Return a coordinator's lanes with their unmerged marks; empty for any other session."""
    if manifest.fanout is None:
        return []
    return listing.lanes_of(layout.state_dir, layout.session_id, branch_tips=tips)


def _won(manifest: sessions_manifest.SessionManifest | None) -> bool:
    """Return whether the manifest's compare stamp names it the winner."""
    return manifest is not None and manifest.compare is not None and manifest.compare.winner


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


def _print_fanout(
    manifest: sessions_manifest.SessionManifest,
    lanes: list[SessionSummary],
    lane_manifests: Mapping[str, sessions_manifest.SessionManifest],
) -> None:
    """Print a coordinator's fan-out line and one line per lane: place, id, status, cost."""
    if manifest.fanout is None:
        return
    print(
        f"fan-out:    {format.lane_count(manifest.fanout.lanes)} (--parallel "
        f"{manifest.fanout.spec})"
    )
    idents = {
        lane.session_id: lane.id_cell(winner=_won(lane_manifests.get(lane.session_id)))
        for lane in lanes
    }
    width = max((len(ident) for ident in idents.values()), default=0)
    for lane in lanes:
        lane_manifest = lane_manifests.get(lane.session_id)
        stamp = lane_manifest.compare if lane_manifest is not None else None
        place = f"rank {stamp.rank}/{stamp.of}" if stamp is not None else f"lane {lane.lane}"
        model = lane.model or "?"
        cost = lane.cost_cell
        label = format.listing_status_label(
            lane.mode, lane.status, lane.reason, unmerged=lane.unmerged
        )
        print(
            f"  {place:<10} {idents[lane.session_id]:<{width}}  {model}  {label}  {cost}".rstrip()
        )


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
    word, reason = row.status, row.reason
    cell = format.listing_status_label(row.mode, row.status, row.reason, unmerged=row.unmerged)
    if scan.finished:
        if scan.end_summary:
            # The `ended:` line carries the run's own sentence; the raw reason would repeat it.
            return word, cell, ""
        if scan.end_reason == "provider_error":
            # The label names the failure; the provider's own text, on one clean line, says why.
            error = terminal.scrub_terminal_controls(scan.provider_error)
            return word, cell, " ".join(error.split())
        # The raw end reason is the diagnostic, unless the label already carries it.
        end = "" if scan.end_reason in (word, *reason.split(" · ")) else scan.end_reason
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
    "role.call": "provider call in flight",
    "role.text_delta": "provider call in flight",
    "role.thinking_delta": "provider call in flight",
}


def _pid_note(pid: int | None, *, alive: bool, finished: bool) -> str:
    """Return the state line's worker suffix: alive, recycled, or not running."""
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


def _print_end_facts(scan: LogScan, session_id: str, *, gated: bool, resume: str | None) -> None:
    """Print unanswered questions, the gate's word and an unfinished run's resume command.

    Args:
        scan: The log scan.
        session_id: The session.
        gated: The session is a run, the one mode a gate judges.
        resume: "" when the session stopped short of its end and a bare resume continues it,
            why a resume would refuse when it stopped short but cannot, None for an end that
            needs no resume.
    """
    failed = scan.end_reason == "setup_command_failed"
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
        `in` and `out`, the cached side when the journal recorded it, the plan points a
        percent-metered execution consumed against its cap, then the cost as the listing's
        cell spells it (blank for a clean $0).
    """
    tokens = f"in={scan.input_tokens or 0} out={scan.output_tokens or 0}"
    if scan.cache_read_tokens is not None or scan.cache_creation_tokens is not None:
        tokens += f" cache_r={scan.cache_read_tokens or 0}"
        tokens += f" cache_c={scan.cache_creation_tokens or 0}"
    if scan.plan_cap > 0:
        execution_pt = scan.plan_consumed - scan.plan_prior_executions
        tokens += f" plan={execution_pt:g}/{scan.plan_cap:g}pt"
    execution_s = " (latest execution)" if scan.executions > 1 else ""
    cell = scan.cost_cell
    executions_s = f" (all {scan.executions} executions)" if scan.executions > 1 else ""
    cost_s = f"  cost {cell}{executions_s}" if cell else ""
    return f"{tokens}{execution_s}{cost_s}"


def _cmd_status(session_id: str, *, as_json: bool = False) -> int:
    """Print a one-shot liveness and progress summary for a run.

    Answered from the run dir alone: worker.pid probed with signal 0, so liveness is known
    even while the worker is blocked in a long provider call, plus the last event, the
    turn and the elapsed time from logs.jsonl. `agent6 attach` is the live follower.

    Args:
        session_id: The session, or "" for the newest.
        as_json: Print one JSON object.

    Returns:
        The exit code; 2 when the session cannot be resolved.
    """
    layout = _common.resolve_target(session_id)
    if layout is None:
        return 2
    target = layout.session_dir

    loaded: sessions_manifest.SessionManifest | None = None
    with contextlib.suppress(sessions_manifest.ManifestError):
        loaded = sessions_manifest.read_manifest(target)
    # A missing manifest still renders, with `mode` as "?" rather than the model default.
    manifest = loaded or sessions_manifest.SessionManifest()
    mode_display = loaded.mode if loaded is not None else None

    logs = target / sessions_layout.LOGS_NAME
    scan = scan_session_log(logs) if logs.is_file() else LogScan()

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
    compare_json = manifest.compare.model_dump(mode="json") if manifest.compare else None
    changes = _changes(target.name, manifest, undone=scan.finished and scan.end_reason == "undone")
    tips = git_ops.run_ref_tips(pathlib.Path.cwd())
    summary = summarize_session_dir(target, branch_tips=tips)
    lanes = _fanout_lanes(layout, manifest, tips)
    lane_manifests = _lane_manifests(layout.state_dir, lanes)
    status, status_cell, status_detail = _status_state(
        summary,
        scan,
        last_age=last_age,
    )
    state = f"{status_cell} ({status_detail})" if status_detail else status_cell

    if as_json:
        print(
            json.dumps(
                {
                    "session_id": target.name,
                    "mode": mode_display,
                    "task_clipped": summary.task_label.task,
                    "continued_task": summary.task_label.continued_task,
                    "task_line": summary.task_label.line(),
                    "model": model,
                    "model_from_flag": model_from_flag,
                    "preset": manifest.harness.preset or None,
                    "status": status,
                    "label": status_cell,
                    "detail": status_detail,
                    "end_summary": scan.end_summary or None,
                    "alive": alive,
                    "pid": pid,
                    "turn": scan.turn,
                    "last_event": scan.last_type,
                    "last_event_age_s": round(last_age, 1) if last_age is not None else None,
                    "elapsed_s": round(elapsed, 1) if elapsed is not None else None,
                    "end_reason": scan.end_reason if scan.finished else None,
                    "baseline_ok": scan.baseline_ok,
                    "verified": scan.verified or None,
                    "input_tokens": scan.input_tokens,
                    "output_tokens": scan.output_tokens,
                    "cache_read_tokens": scan.cache_read_tokens,
                    "cache_creation_tokens": scan.cache_creation_tokens,
                    "cost_usd": scan.cost_usd,
                    "plan_consumed": scan.plan_consumed,
                    "plan_prior_executions": scan.plan_prior_executions,
                    "plan_cap": scan.plan_cap,
                    "unattended_questions": scan.unattended_questions,
                    "setup_output": scan.setup_output or None,
                    # An under-estimate when some spend was unpriced; the text render marks it too.
                    "usd_partial": scan.usd_partial if scan.cost_usd is not None else None,
                    "source_session_id": manifest.source_session_id,
                    "parent_session_id": manifest.parent_session_id,
                    "forked_from_turn": manifest.forked_from_turn,
                    "forked_from_sha": manifest.forked_from_sha,
                    "worktree": str(manifest.worktree) if manifest.worktree else None,
                    "compare": compare_json,
                    "parallel": manifest.parallel.model_dump(mode="json")
                    if manifest.parallel
                    else None,
                    "fanout": manifest.fanout.model_dump(mode="json") if manifest.fanout else None,
                    "lanes": [
                        listing.summary_row(ln, winner=_won(lane_manifests.get(ln.session_id)))
                        for ln in lanes
                    ],
                    "run_branch": existing_run_branch(manifest, pathlib.Path.cwd()) or None,
                    "base_branch": manifest.base_branch or None,
                    "merged_into": changes.merged_into or None,
                    "pins": list(scan.pins),
                }
            )
        )
        return 0

    pid_note = _pid_note(pid, alive=alive, finished=scan.finished)
    print(f"session:    {target.name}  (mode={mode_display or '?'})")
    if task := summary.task_label.line():
        print("task:       " + task.replace("\n", "\n" + " " * 12))
    _print_lineage(manifest)
    _print_parallel_compare(manifest)
    _print_fanout(manifest, lanes, lane_manifests)
    print(
        f"model:      {model}{' (from --model)' if model_from_flag else ''}\n"
        f"preset:     {manifest.harness.preset or '-'}"
    )
    print(f"state:      {state}{pid_note}")
    print(f"turn:       {scan.turn if scan.turn is not None else '-'}")
    print(
        f"last event: {scan.last_type or '-'}"
        f"{f'  ({_fmt_dur(last_age)} ago)' if last_age is not None else ''}"
    )
    # First start to last event: the time between executions is in it.
    spans = f" (wall, {scan.executions} executions)" if scan.executions > 1 else ""
    print(f"elapsed:    {_fmt_dur(elapsed)}{spans}")
    if scan.input_tokens is not None or scan.cost_usd is not None:
        print(f"usage:      {_usage_line(scan)}")
    _print_end_facts(
        scan,
        target.name,
        gated=mode_display == "run",
        resume=listing.resume_blocker(target) if status in _RESUMABLE else None,
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
    if not run_branch:
        return _Changes("", "")
    if undone:
        return _Changes(f"{run_branch} (taken back by /undo)", "")
    cwd = pathlib.Path.cwd()
    stamp = resume.covering_stamp(cwd, manifest)
    if stamp is not None:
        return _Changes(
            format.format_branch(run_branch, manifest.base_branch, stamp.into), stamp.into
        )
    merge_hint = f"merge with: {merge.merge_command(session_id, manifest.base_branch)}"
    chain = git_ops.chain_ref_for(session_id)
    branch_kept = git_ops.branch_exists(cwd, run_branch)
    tip = git_ops.branch_tip_sha(cwd, run_branch) if branch_kept else git_ops.chain_tip(cwd, chain)
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
    from agent6.graph import storage  # noqa: PLC0415  # noqa: PLC0415
    from agent6.ui.cli import _task_tree  # noqa: PLC0415  # noqa: PLC0415

    layout = sessions_layout.layout_of(session_dir)
    nodes = storage.load_graph(layout)
    if len(nodes) <= 1:
        return
    lines = _task_tree.task_tree_lines({nid: nodes[nid].model_dump() for nid in sorted(nodes)})
    if lines:
        print("\ntasks:")
        for line in lines:
            print(f"  {line}")
