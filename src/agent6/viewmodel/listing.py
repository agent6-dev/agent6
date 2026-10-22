# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Scan session dirs into the listing rows every front-end's hub shows.

Directory discovery and row assembly share the journal scanner and status helpers,
so the CLI, TUI and web listings cannot disagree.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
from collections.abc import Container, Iterable, Mapping, Sequence

from agent6 import git_ops
from agent6.sessions import layout, process
from agent6.sessions import manifest as sessions_manifest
from agent6.viewmodel import _session_log, _session_status, format


def session_mtime(session_dir: pathlib.Path) -> float:
    """Return a session's last-activity time as epoch seconds.

    The journal's mtime, else the manifest's, else the dir's; never the dir's first,
    since a viewer's `frontends/` claim bumps it and would float a viewed run to newest.

    Args:
        session_dir: The session's state dir.

    Returns:
        The mtime, 0.0 when none of the three can be read.
    """
    for candidate in (
        session_dir / layout.LOGS_NAME,
        session_dir / sessions_manifest.MANIFEST_NAME,
        session_dir,
    ):
        try:
            return candidate.stat().st_mtime
        except OSError:
            continue
    return 0.0


def session_dirs(
    state_dir: pathlib.Path, buckets: Iterable[str] = layout.HUB_BUCKETS
) -> list[pathlib.Path]:
    """Return every session dir a listing shows, newest first by last activity.

    Args:
        state_dir: The repo's state dir.
        buckets: The bucket names under `sessions/` to walk; the hub's by default.

    Returns:
        The session dirs, husks skipped.
    """
    dirs: list[pathlib.Path] = []
    for name in buckets:
        bucket = layout.bucket_dir(state_dir, name)
        if bucket.is_dir():
            dirs.extend(p for p in bucket.iterdir() if p.is_dir() and not is_session_husk(p))
    dirs.sort(key=session_mtime, reverse=True)
    return dirs


def newest_session_dir(buckets: Iterable[pathlib.Path]) -> pathlib.Path | None:
    """Return the most recently active session dir across the given bucket dirs.

    Husks are skipped: a crash-orphaned dir is newer than the real runs, and returning
    it would point a bare `attach` at a phantom no listing shows.

    Args:
        buckets: The bucket dirs in scope; a missing one is skipped.

    Returns:
        The newest session dir by last activity, or None when no bucket holds one.
    """
    runs: list[pathlib.Path] = []
    for bucket in buckets:
        if bucket.is_dir():
            runs.extend(p for p in bucket.iterdir() if p.is_dir() and not is_session_husk(p))
    dirs = sorted(runs, key=session_mtime, reverse=True)
    return dirs[0] if dirs else None


def is_session_husk(session_dir: pathlib.Path) -> bool:
    """Return whether a session dir never started: no manifest, no log, no live worker.

    A dir with a live worker is a just-launched run in its pre-manifest preflight
    window and stays listed as "starting". Listings skip husks, and an id lookup must
    not let one shadow a real run of the same id in another bucket.

    Args:
        session_dir: The session's state dir.

    Returns:
        True when the dir has none of the three.
    """
    return not layout.session_has_record(session_dir) and not process.worker_is_alive(session_dir)


def session_compare(session_dir: pathlib.Path) -> sessions_manifest.CompareStamp | None:
    """Return the fan-out compare stamp on a lane's manifest.

    The event fold does not carry it, so every run view reads it from here.

    Args:
        session_dir: The session's state dir.

    Returns:
        The stamp, or None for a run outside a compared fan-out or with an unreadable
        manifest.
    """
    try:
        manifest = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError:
        return None
    return manifest.compare


def is_winner(session_dir: pathlib.Path) -> bool:
    """Return whether a run is its fan-out's compare winner."""
    compare = session_compare(session_dir)
    return compare is not None and compare.winner


@dataclasses.dataclass(frozen=True, slots=True)
class SessionSummary:
    """One listing row: everything a hub or `sessions list` needs, uncolored.

    Attributes:
        session_id: The session's id.
        mode: "run", "plan", "ask" or "?".
        task_label: The original task and latest continuation, before display clipping.
        status: The status word (created, starting, running, waiting, stale, passed,
            answered, planned, finished, stopped, undone, failed).
        reason: The detail beside the word: the end reason when failed, what it waits
            on when waiting, or a pending stop after the step.
        end_reason: The last `session.end` reason word; "" while the session runs.
        cost_usd: The cumulative spend across executions.
        usd_partial: `cost_usd` is a lower bound: some execution had unpriced spend.
        mtime: The last-activity time as epoch seconds.
        unmerged: The run's chain or branch holds commits its base does not, per the
            merge stamp and the caller's branch-tips snapshot; False without a
            snapshot, and for an undone run (its /undo child carries the mark).
        verify_ok: The gate verdict from the gate facts, not the status word: the
            compare table and the judge read it.
        coordinator: The session that dispatched this lane, the row it nests under.
        lane: The lane number within its fan-out.
        plan_consumed: The plan points consumed across executions on a plan-metered provider.
        plan_cap: The plan's points cap, `[budget].max_plan_points`.
        plan_used_percent: The account's reading; 0 unless a plan provider answered.
        model: The manifest's provider/model route for this execution.
        model_from_flag: The route came from a `--model` flag rather than config.
        worktree: The linked worktree the run works in, "" for a run in the checkout.
    """

    session_id: str
    mode: str
    task_label: _session_status.TaskLabel
    status: str
    reason: str
    cost_usd: float
    usd_partial: bool
    mtime: float
    end_reason: str = ""
    unmerged: bool = False
    verify_ok: bool | None = None
    coordinator: str = ""
    lane: int | None = None
    plan_consumed: float = 0.0
    plan_cap: float = 0.0
    plan_used_percent: float = 0.0
    model: str = ""
    model_from_flag: bool = False
    worktree: str = ""

    @property
    def cost_cell(self) -> str:
        """The session's cost as `cost_cell` spells it."""
        return _session_status.cost_cell(
            self.cost_usd,
            partial=self.usd_partial,
            plan_consumed=self.plan_consumed,
            plan_used_percent=self.plan_used_percent,
        )

    def id_cell(self, *, winner: bool) -> str:
        """Return the id cell every listing shows: the id, then the winner and worktree marks."""
        cell = f"{self.session_id} {format.WINNER_GLYPH}" if winner else self.session_id
        return f"{cell} {format.WORKTREE_MARK}" if self.worktree else cell


@dataclasses.dataclass(frozen=True, slots=True)
class ListingRow:
    """One listing row: a session and the fan-out lanes nested under it, rows themselves."""

    summary: SessionSummary
    lanes: tuple[ListingRow, ...] = ()

    @property
    def mtime(self) -> float:
        """The row's last activity: its own or a lane's, whichever is later."""
        return max((self.summary.mtime, *(lane.mtime for lane in self.lanes)))


def nested_rows(summaries: Iterable[SessionSummary]) -> list[ListingRow]:
    """Nest each lane under the listed session that dispatched it.

    Args:
        summaries: The sessions to list.

    Returns:
        The rows, newest first, every session appearing once: a lane whose coordinator
        is not listed stays a row of its own.
    """
    items = list(summaries)
    by_id = {s.session_id: s for s in items}
    children: dict[str, list[SessionSummary]] = {}
    for s in items:
        if s.coordinator and s.coordinator in by_id and s.coordinator != s.session_id:
            children.setdefault(s.coordinator, []).append(s)
    placed: set[str] = set()

    def row(s: SessionSummary) -> ListingRow:
        placed.add(s.session_id)
        lanes = sorted(children.get(s.session_id, ()), key=_lane_order)
        return ListingRow(s, tuple(row(ln) for ln in lanes if ln.session_id not in placed))

    roots = [s for s in items if not (s.coordinator in by_id and s.coordinator != s.session_id)]
    rows = [row(s) for s in roots]
    # A coordinator chain that loops reaches no root: its sessions are rows.
    rows.extend(row(s) for s in items if s.session_id not in placed)
    rows.sort(key=lambda r: r.mtime, reverse=True)
    return rows


def _lane_order(lane: SessionSummary) -> tuple[int, str]:
    """Return the sort key placing lanes in lane order, then by id."""
    return (lane.lane or 0, lane.session_id)


def lanes_of(
    state_dir: pathlib.Path, coordinator: str, *, branch_tips: Mapping[str, str] | None = None
) -> list[SessionSummary]:
    """Return the lanes whose manifests name a coordinator, in lane order.

    Args:
        state_dir: The repo's state dir.
        coordinator: The dispatching session's id.
        branch_tips: The caller's branch-tips snapshot; with it each lane carries its
            unmerged mark.

    Returns:
        The lanes' summaries.
    """
    lanes = [
        s
        for d in session_dirs(state_dir)
        if (s := summarize_session_dir(d, branch_tips=branch_tips)).coordinator == coordinator
    ]
    return sorted(lanes, key=_lane_order)


def row_json(row: ListingRow, *, winners: Container[str]) -> dict[str, object]:
    """Return a listing row and its nested lanes as JSON.

    Args:
        row: The row.
        winners: The ids of the fan-out compare winners.

    Returns:
        The `summary_row` dict with `mtime` and `when` set to the group's latest activity.
    """
    out = summary_row(
        row.summary,
        winner=row.summary.session_id in winners,
        lanes=[row_json(ln, winners=winners) for ln in row.lanes],
    )
    out["mtime"] = row.mtime
    out["when"] = format.format_when(row.mtime) if row.mtime else ""
    return out


def summary_row(
    s: SessionSummary,
    *,
    winner: bool = False,
    lanes: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    """Return one listing row as JSON, the shape `sessions list --json` and `/api/hub` share.

    Args:
        s: The row's summary.
        winner: The row is its fan-out's compare winner.
        lanes: The nested lane rows, as `row_json` builds them.

    Returns:
        The row, with the status cell rendered as `label` and `level` so a client
        needs no copy of the status maps, and `task_line` with an optional continuation line.
    """
    return {
        "session_id": s.session_id,
        "mode": s.mode,
        "task_clipped": s.task_label.task,
        "task_line": s.task_label.line(),
        "continued_task": s.task_label.continued_task,
        "status": s.status,
        "reason": s.reason,
        "end_reason": s.end_reason,
        "label": format.listing_status_label(s.mode, s.status, s.reason, unmerged=s.unmerged),
        "level": format.status_level(s.status),
        "mtime": s.mtime,
        "when": format.format_when(s.mtime) if s.mtime else "",
        "cost_usd": s.cost_usd,
        "usd_partial": s.usd_partial,
        "plan_consumed": s.plan_consumed,
        "plan_cap": s.plan_cap,
        "cost": s.cost_cell,
        "model": s.model,
        "model_from_flag": s.model_from_flag,
        "id_cell": s.id_cell(winner=winner),
        "unmerged": s.unmerged,
        "verify_ok": s.verify_ok,
        "winner": winner,
        "lane": s.lane,
        "coordinator": s.coordinator,
        "worktree": s.worktree or None,
        "lanes": list(lanes),
    }


def session_is_live(session_dir: pathlib.Path) -> bool:
    """Return whether anything will read what the operator writes to this session.

    Derived from the status word, so a surface cannot disagree with the label it shows.

    Args:
        session_dir: The session's state dir.

    Returns:
        True while the status word is in `LIVE_STATUS_WORDS`.
    """
    logs = session_dir / layout.LOGS_NAME
    facts = (
        _session_log.scan_session_log(logs).status_facts()
        if logs.is_file()
        else _session_status.StatusFacts()
    )
    return (
        _session_status.status_for_session_dir(session_dir, facts)[0]
        in _session_status.LIVE_STATUS_WORDS
    )


def needs_new_work(*, finished: bool, end_reason: str, all_passed: bool | None) -> bool:
    """Return whether a bare resume of a run in this state would have nothing to do.

    The one predicate behind the resume refusal, the web composer's hint and the
    wire's `needs_new_work`. A bare resume of such a run spends a call, answers in
    prose, records a silent_finish and leaves a passed run reading as failed.

    Args:
        finished: A session.end was seen.
        end_reason: The session.end reason word.
        all_passed: The verify tri-state.

    Returns:
        True only for a `finish_session` end over a tree the gate certified green or
        that no gate judged; every other end is what resume is for.
    """
    return finished and end_reason == "finish_session" and all_passed is not False


def finished_needs_new_work(session_dir: pathlib.Path) -> bool:
    """Return `needs_new_work` over the run's log, for a verb that holds only the dir."""
    scan = _session_log.scan_session_log(session_dir / layout.LOGS_NAME)
    return needs_new_work(
        finished=scan.finished, end_reason=scan.end_reason, all_passed=scan.all_passed
    )


def resume_blocker(session_dir: pathlib.Path) -> str:
    """Return why a bare `agent6 resume` would refuse the session, or "" when it can start.

    The resume command is printed only where it would work: a coordinator has no loop, a
    lane's tree is no checkout's once its clone is gone, a worktree run whose worktree is
    gone continues through `fork`, and a run that never reached a turn has no snapshot.

    Args:
        session_dir: The session's state dir.

    Returns:
        The reason, naming the `agent6 fork` command when one continues the run; "" when
        the resume can start.
    """
    if is_fanout(session_dir):
        return "a fan-out coordinator is not resumable"
    with contextlib.suppress(sessions_manifest.ManifestError):
        m = sessions_manifest.read_manifest(session_dir)
        if m.parallel is not None:
            return (
                f"a lane of the fan-out {m.parallel.coordinator}, whose work is on its own"
                f' branch; `agent6 fork {session_dir.name} --steer "..."` continues it in a'
                " worktree of its own"
            )
        if m.worktree is not None and not (m.worktree / ".git").exists():
            return (
                f"its worktree {m.worktree} is gone; `agent6 fork {session_dir.name}` continues"
                " it in a new worktree"
            )
    if not (session_dir / "loop_state.json").is_file():
        return (
            "it never reached its first turn, so there is nothing to resume; start the task again"
        )
    return ""


def fanout_resume_refusal(session_id: str) -> str:
    """Return the refusal every resume path gives a fan-out coordinator, which has no loop."""
    return (
        f"cannot resume {session_id}: a fan-out coordinator is not resumable;"
        f" `agent6 sessions show {session_id}` lists its lanes."
    )


def is_fanout(session_dir: pathlib.Path) -> bool:
    """Return whether the session is a fan-out coordinator; False when its manifest is unreadable.

    A hub asks before it spawns a resume that could only refuse.

    Args:
        session_dir: The session's state dir.

    Returns:
        True when its manifest carries a fan-out record.
    """
    try:
        return sessions_manifest.read_manifest(session_dir).fanout is not None
    except sessions_manifest.ManifestError:
        return False


def undone_refusal(session_id: str, child: str) -> str:
    """Return the refusal resume and merge share for a run `/undo` took back.

    Args:
        session_id: The undone run's id.
        child: The fork that continues the run.

    Returns:
        The refusal, then the `agent6 resume` command for `child`.
    """
    return (
        f"run {session_id!r} was taken back by /undo; its continuation is {child}:\n"
        f'    agent6 resume {child} --steer "<the message, edited>"'
    )


def needs_new_work_refusal(session_id: str) -> str:
    """Return the refusal `agent6 resume` and the web composer share for a finished run."""
    return (
        f"run {session_id!r} already finished (the agent called finish_session)."
        " Give it new work with:\n"
        f'    agent6 resume {session_id} --steer "<what to do next>"'
    )


def unmerged(
    manifest: sessions_manifest.SessionManifest, word: str, branch_tips: Mapping[str, str]
) -> bool:
    """Return whether the run holds work outside its merge stamp.

    Args:
        manifest: The run's manifest.
        word: The dir-aware status word; an undone run's child carries its work.
        branch_tips: The caller's `git_ops.run_ref_tips` snapshot.

    Returns:
        Whether its chain or branch has a tip beyond the base and outside the stamp.
    """
    if word == "undone":
        return False
    tips = {
        tip
        for tip in (
            branch_tips.get(git_ops.chain_ref_for(manifest.session_id)),
            branch_tips.get(manifest.run_branch or ""),
        )
        if tip
    }
    # The branch may carry the operator's own commits; a merge stamps whichever tip it merged.
    return bool(tips - {manifest.base_sha}) and (
        manifest.merged is None or manifest.merged.tip not in tips
    )


def summarize_session_dir(
    session_dir: pathlib.Path,
    *,
    branch_tips: Mapping[str, str] | None = None,
) -> SessionSummary:
    """Fold a session dir's journal and manifest into one listing row.

    The journal owns the original task and resume steer; the manifest is a fallback
    only before the journal records a task.

    Args:
        session_dir: The session's state dir.
        branch_tips: The caller's `git_ops.run_ref_tips` snapshot; with it the row says
            whether the chain or branch still holds commits no merge stamped, without
            it `unmerged` stays False (no mark, never a wrong one).

    Returns:
        The row.
    """
    logs = session_dir / layout.LOGS_NAME
    scan = _session_log.scan_session_log(logs) if logs.is_file() else _session_log.LogScan()
    manifest: sessions_manifest.SessionManifest | None = None
    with contextlib.suppress(sessions_manifest.ManifestError):
        manifest = sessions_manifest.read_manifest(session_dir)
    mode, label = scan.mode, scan.task_label
    if manifest is not None:
        if not label.task:
            label = label.fallback(manifest.user_task)
        if mode == "?":
            mode = manifest.mode or mode
    if mode == "?" and not label.task:
        label = _session_status.TaskLabel("(no logs)")
    word, reason = _session_status.status_for_session_dir(session_dir, scan.status_facts())
    lineage = manifest.parallel if manifest is not None else None
    has_unmerged = (
        unmerged(manifest, word, branch_tips)
        if branch_tips is not None and manifest is not None
        else False
    )
    return SessionSummary(
        session_id=session_dir.name,
        mode=mode,
        task_label=label,
        status=word,
        reason=reason,
        end_reason=scan.end_reason if scan.finished else "",
        unmerged=has_unmerged,
        cost_usd=scan.cost_usd or 0.0,
        usd_partial=scan.usd_partial,
        mtime=session_mtime(session_dir),
        verify_ok=scan.verify_verdict(),
        coordinator=lineage.coordinator if lineage is not None else "",
        lane=lineage.lane if lineage is not None else None,
        plan_consumed=scan.plan_consumed,
        plan_cap=scan.plan_cap,
        plan_used_percent=scan.plan_used_percent,
        model=format.format_model_route(manifest.models.driver) if manifest is not None else "",
        model_from_flag=manifest.models.driver_from_flag if manifest is not None else False,
        worktree=str(manifest.worktree) if manifest is not None and manifest.worktree else "",
    )
