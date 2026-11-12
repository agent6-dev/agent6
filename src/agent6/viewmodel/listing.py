# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Scan session dirs into the listing rows every front-end's hub shows.

Directory discovery and row assembly share the journal scanner and status helpers,
so the CLI, TUI and web listings cannot disagree.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import pathlib
from collections.abc import Callable, Container, Iterable, Mapping, Sequence
from typing import Any, NotRequired, TypedDict

from agent6 import errors, kinds, paths
from agent6.sessions import ipc, layout, process
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs import git_read
from agent6.viewmodel import comparison, events, format, review, session_log, session_status


def session_mtime(session_dir: pathlib.Path) -> float:
    """Return the last recorded event time, or the manifest start time without events."""
    last_ep = session_log.last_event_epoch(session_dir / layout.LOGS_NAME)
    if last_ep is not None:
        return last_ep
    with contextlib.suppress(sessions_manifest.ManifestError):
        if started := events.event_epoch(sessions_manifest.read_manifest(session_dir).start_ts):
            return started
    # Legacy state without recorded times has only filesystem metadata.
    for candidate in (
        session_dir / layout.LOGS_NAME,
        session_dir / sessions_manifest.MANIFEST_NAME,
        session_dir,
    ):
        with contextlib.suppress(OSError):
            return candidate.stat().st_mtime
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


@dataclasses.dataclass(frozen=True, slots=True)
class Overlap:
    """Another ended, unmerged session that changed some of the same paths.

    Attributes:
        session_id: The other session.
        paths: The repo-relative paths both changed, sorted.
    """

    session_id: str
    paths: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class SessionSummary:
    """One listing row: everything a hub or `sessions list` needs, uncolored.

    Attributes:
        session_id: The session's id.
        mode: "run", "plan", "ask", "read_only" or "?".
        task_label: The original task and latest continuation, before display clipping.
        status: The status word (created, starting, running, waiting, stale, passed,
            answered, planned, finished, stopped, undone, failed).
        detail: The detail beside the word: the end reason when failed, what it waits
            on when waiting, or a pending stop after the step.
        stop_reason: The last `session.stop` reason word; "" while the session runs.
        cost_usd: The cumulative spend across executions.
        usd_partial: `cost_usd` is a lower bound: some execution had unpriced spend.
        last_activity: The last-activity time as epoch seconds.
        unmerged: The run's chain or branch holds commits its base does not, per the
            merge stamp and the caller's branch-tips snapshot; False without a
            snapshot, and for an undone run (its /undo child carries the mark).
        verification: The gate verdict from the gate facts, not the status word: the
            compare table and the judge read it.
        coordinator: The session that dispatched this lane, the row it nests under.
        lane: The lane number within its fan-out.
        plan_observed_points: The plan movement observed across executions, in points.
        plan_cap: The plan's points cap, `[budget].max_plan_points`.
        plan_used_percent: The account's reading; 0 unless a plan provider answered.
        model: The manifest's provider/model route for this execution.
        model_from_flag: The route came from a `--model` flag rather than config.
        worktree: The linked worktree the run works in, "" for a run in the checkout.
        worktree_missing: The refusal when the recorded worktree is absent in this home.
        branch: The run's branch while the caller's branch-tips snapshot holds it; "" without
            a snapshot, a branch or a commit on it.
        base_sha: The commit the run started from, as the manifest recorded it; "" unknown.
        source_tip: The run's last diff checkpoint commit; "" before one.
        open_findings: The open grounded findings by severity, from the journal's review fold.
        overlaps: The other ended, unmerged runs that changed some of the same paths;
            landing one means resuming the others on the new base.
    """

    session_id: str
    mode: str
    task_label: session_status.TaskLabel
    status: str
    detail: str
    cost_usd: float
    usd_partial: bool
    last_activity: float
    stop_reason: str = ""
    unmerged: bool = False
    verification: kinds.Verification | None = None
    coordinator: str = ""
    lane: int | None = None
    plan_observed_points: float = 0.0
    plan_cap: float = 0.0
    plan_used_percent: float = 0.0
    model: str = ""
    model_from_flag: bool = False
    worktree: str = ""
    worktree_missing: str = ""
    branch: str = ""
    base_sha: str = ""
    source_tip: str = ""
    open_findings: review.SeverityCounts = dataclasses.field(default_factory=review.SeverityCounts)
    overlaps: tuple[Overlap, ...] = ()

    @property
    def usage_cell(self) -> str:
        """The session's cost as `usage_cell` spells it."""
        return session_status.usage_cell(
            self.cost_usd,
            partial=self.usd_partial,
            plan_observed_points=self.plan_observed_points,
            plan_used_percent=self.plan_used_percent,
        )

    def id_cell(self, *, winner: bool) -> str:
        """Return the id cell every listing shows: the id, then the winner and worktree marks."""
        cell = f"{self.session_id} {format.WINNER_GLYPH}" if winner else self.session_id
        if self.worktree:
            missing = self.worktree_missing
            return f"{cell} ({missing})" if missing else f"{cell} {format.WORKTREE_MARK}"
        return cell

    def merge_note(self) -> str:
        """Return what stands before a clean merge, the words every listing shows.

        Returns:
            `open findings: 2 block, 1 warn · same paths as <id> and 3 more`; "" with neither.
            The note names one run; `overlaps` holds every run and its paths.
        """
        notes: list[str] = []
        counts = dataclasses.asdict(self.open_findings)
        if found := ", ".join(f"{n} {severity}" for severity, n in counts.items() if n):
            notes.append(f"open findings: {found}")
        if self.overlaps:
            more = len(self.overlaps) - 1
            notes.append(
                f"same paths as {self.overlaps[0].session_id}"
                + (f" and {more} more" if more else "")
            )
        return " · ".join(notes)


@dataclasses.dataclass(frozen=True, slots=True)
class ListingRow:
    """One listing row: a session and the fan-out lanes nested under it, rows themselves."""

    summary: SessionSummary
    lanes: tuple[ListingRow, ...] = ()

    @property
    def group_last_activity(self) -> float:
        """The row's last activity: its own or a lane's, whichever is later."""
        return max((self.summary.last_activity, *(lane.group_last_activity for lane in self.lanes)))


def nested_rows(
    summaries: Iterable[SessionSummary], *, repo: pathlib.Path | None = None
) -> list[ListingRow]:
    """Nest each lane under the listed session that dispatched it, newest activity first.

    Args:
        summaries: The sessions to list.
        repo: The repository whose history tells which paths each ended, unmerged run
            changed; with it, two such runs that changed the same paths flag each other.

    Returns:
        The rows, every session appearing once: a lane whose coordinator is not listed
        stays a row of its own; lanes keep lane order under their coordinator.
    """
    items = [s for s in summaries if not is_seat_record(s)]
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
    rows.sort(key=lambda r: r.group_last_activity, reverse=True)
    return rows if repo is None else flag_overlaps(rows, changed_paths(repo))


def flag_overlaps(
    rows: list[ListingRow], touched: Callable[[SessionSummary], frozenset[str]]
) -> list[ListingRow]:
    """Flag ended, unmerged sessions that changed some of the same paths, lanes included.

    Args:
        rows: The nested listing rows.
        touched: Returns the paths a session changed between its base and its tip.

    Returns:
        The rows, each such summary's `overlaps` naming the others.
    """
    waiting: dict[str, frozenset[str]] = {}

    def collect(row: ListingRow) -> None:
        s = row.summary
        if s.unmerged and s.status in format.ENDED_WORDS:
            waiting[s.session_id] = touched(s)
        for lane in row.lanes:
            collect(lane)

    for row in rows:
        collect(row)
    if len(waiting) < 2:
        return rows

    def flagged(row: ListingRow) -> ListingRow:
        s = row.summary
        lanes = tuple(flagged(lane) for lane in row.lanes)
        mine = waiting.get(s.session_id)
        if mine is None:
            return dataclasses.replace(row, lanes=lanes)
        overlaps = tuple(
            Overlap(other, tuple(sorted(mine & theirs)))
            for other, theirs in waiting.items()
            if other != s.session_id and mine & theirs
        )
        return ListingRow(dataclasses.replace(s, overlaps=overlaps), lanes)

    return [flagged(row) for row in rows]


def changed_paths(repo: pathlib.Path) -> Callable[[SessionSummary], frozenset[str]]:
    """Return a reader of the paths a run changed, from its base to its last checkpoint.

    Args:
        repo: The repository holding the run's commits.

    Returns:
        The reader; a run without a recorded base or checkpoint changed no known path.
    """

    def touched(s: SessionSummary) -> frozenset[str]:
        if not s.base_sha or not s.source_tip:
            return frozenset()
        return _paths_between(str(repo), s.base_sha, s.source_tip)

    return touched


@functools.lru_cache(maxsize=512)
def _paths_between(repo: str, base: str, tip: str) -> frozenset[str]:
    """Return the paths two commits differ in; two shas name immutable trees, so cached."""
    try:
        return frozenset(git_read.tree_diff_paths(pathlib.Path(repo), base, tip))
    except errors.GitError:
        return frozenset()


def rows_for(
    cwd: pathlib.Path, *, buckets: Iterable[str] = layout.HUB_BUCKETS
) -> tuple[list[ListingRow], set[str]]:
    """Return a repository's listing rows and its fan-out winners' ids.

    Args:
        cwd: The repository.
        buckets: The session buckets to list; the hub's by default.

    Returns:
        The nested rows in that order, and the ids of the compare winners.
    """
    tips = git_read.run_ref_tips(cwd)
    dirs = session_dirs(paths.state_dir(cwd), buckets)
    winners = {d.name for d in dirs if comparison.is_winner(d)}
    rows = nested_rows((summarize_session_dir(d, branch_tips=tips) for d in dirs), repo=cwd)
    return rows, winners


def is_seat_record(summary: SessionSummary) -> bool:
    """Return whether a session is a bounded read-only conversation its parent dispatched.

    A review seat is evidence of its panel, reached from the receipt and the findings; it
    has no work to join, so no listing counts it among its parent's lanes or rows.

    Args:
        summary: The session.

    Returns:
        True for a dispatched session of the `read_only` kind.
    """
    return bool(summary.coordinator) and summary.mode == "read_only"


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
        and not is_seat_record(s)
    ]
    return sorted(lanes, key=_lane_order)


class SummaryRow(TypedDict):
    """The listing payload, with meanings on its builders."""

    session_id: str
    mode: str
    task: str
    task_line: str
    continued_task: str
    status: str
    status_detail: str
    stop_reason: str
    status_label: str
    status_level: str
    last_activity: float
    when: str
    cost_usd: float
    usd_partial: bool
    plan_observed_points: float
    plan_cap: float
    usage: str
    model: str
    model_from_flag: bool
    id_cell: str
    unmerged: bool
    verification: kinds.Verification | None
    winner: bool
    lane: int | None
    coordinator: str
    worktree: str | None
    branch: str | None
    open_findings: dict[str, int]
    overlaps: list[dict[str, Any]]
    merge_note: str
    lanes: list[SummaryRow]
    group_last_activity: NotRequired[float]


def row_json(row: ListingRow, *, winners: Container[str]) -> SummaryRow:
    """Return a listing row and its nested lanes as JSON.

    Attributes:
        group_last_activity: The group's latest activity time in epoch seconds.
        when: The rendered group activity time.

    Args:
        row: The row.
        winners: The ids of the fan-out compare winners.

    Returns:
        The `summary_row` dict with group activity and a rendered `when`.
    """
    out = summary_row(
        row.summary,
        winner=row.summary.session_id in winners,
        lanes=[row_json(ln, winners=winners) for ln in row.lanes],
    )
    if row.lanes:
        out["group_last_activity"] = row.group_last_activity
    out["when"] = format.format_when(row.group_last_activity) if row.group_last_activity else ""
    return out


def summary_row(
    s: SessionSummary,
    *,
    winner: bool = False,
    lanes: Sequence[SummaryRow] = (),
) -> SummaryRow:
    """Return one listing row as JSON, the shape `sessions list --json` and `/api/hub` share.

    Attributes:
        session_id: The session identifier.
        mode: The session mode.
        task: The operator's words, whole.
        task_line: The task clipped to 200 characters, then a newline and `continued: <steer>`,
            clipped the same way, when a resume steer exists.
        continued_task: The latest resume steer, whole; empty without a continuation.
        status: The lifecycle status word.
        status_detail: The detail explaining the status.
        stop_reason: The last journaled end reason, or an empty string while running.
        status_label: The rendered status with mode and unmerged marks.
        status_level: The status presentation level.
        last_activity: The last activity time in epoch seconds.
        when: The rendered last activity time.
        cost_usd: Cumulative spend in USD across executions.
        usd_partial: Whether unpriced usage makes spend a lower bound.
        plan_observed_points: The plan movement observed across executions, in points.
        plan_cap: The subscription points cap.
        usage: The rendered usage cell: plan points, dollars, or both.
        model: The resolved provider/model route.
        model_from_flag: Whether a model flag selected the route.
        id_cell: The session identifier with winner and worktree marks.
        unmerged: Whether the run holds unmerged work.
        verification: The gate verdict, or null without a verdict.
        lane: The lane number, or null outside a fan-out.
        coordinator: The session that dispatched this lane.
        worktree: The linked worktree path, or null in the operator's checkout.
        branch: The run's branch while it exists, or null.
        open_findings: The open grounded findings by severity: `block`, `warn`, `nit`.
        overlaps: The Overlap records: the other ended, unmerged runs that changed some of
            the same paths.
        merge_note: The rendered open findings and same-path runs; "" with neither.

        winner: Whether this lane won its comparison.
        lanes: Nested listing rows for this coordinator's lanes.

    Args:
        s: The row's summary.
        winner: The row is its fan-out's compare winner.
        lanes: The nested lane rows, as `row_json` builds them.

    Returns:
        The row, with the status cell rendered as `status_label` and `status_level` so a client
        needs no copy of the status maps, and `task_line` with an optional continuation line.
    """
    return {
        "session_id": s.session_id,
        "mode": s.mode,
        "task": s.task_label.task,
        "task_line": s.task_label.line(),
        "continued_task": s.task_label.continued_task,
        "status": s.status,
        "status_detail": s.detail,
        "stop_reason": s.stop_reason,
        "status_label": format.listing_status_label(
            s.mode, s.status, s.detail, unmerged=s.unmerged
        ),
        "status_level": format.status_level(s.status),
        "last_activity": s.last_activity,
        "when": format.format_when(s.last_activity) if s.last_activity else "",
        "cost_usd": s.cost_usd,
        "usd_partial": s.usd_partial,
        "plan_observed_points": s.plan_observed_points,
        "plan_cap": s.plan_cap,
        "usage": s.usage_cell,
        "model": s.model,
        "model_from_flag": s.model_from_flag,
        "id_cell": s.id_cell(winner=winner),
        "unmerged": s.unmerged,
        "verification": s.verification,
        "winner": winner,
        "lane": s.lane,
        "coordinator": s.coordinator,
        "worktree": s.worktree or None,
        "branch": s.branch or None,
        "open_findings": dataclasses.asdict(s.open_findings),
        "overlaps": [dataclasses.asdict(o) for o in s.overlaps],
        "merge_note": s.merge_note(),
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
        session_log.scan_session_log(logs).status_facts()
        if logs.is_file()
        else session_status.StatusFacts()
    )
    return (
        session_status.status_for_session_dir(session_dir, facts)[0]
        in session_status.LIVE_STATUS_WORDS
    )


def needs_new_work(
    *,
    finished: bool,
    stop_reason: str,
    verification: kinds.Verification | None,
    saved_answers: int = 0,
) -> bool:
    """Return whether a bare resume of a run in this state would have nothing to do.

    The one predicate behind the resume refusal, the web composer's hint and the
    wire's `needs_new_work`. A bare resume of such a run spends a call, answers in
    prose, records a silent_finish and leaves a passed run reading as failed.

    Args:
        finished: A session.stop was seen.
        stop_reason: The session.stop reason word.
        verification: The final verification word, or None when unknown.
        saved_answers: Answers saved on disk for the run's questions; the resume delivers them.

    Returns:
        True only for a `finish_session` end over a tree the gate certified green or
        that no gate judged, with no answer waiting; every other end is what resume is for.
    """
    return (
        finished
        and stop_reason == "finish_session"
        and verification in ("passed", "not_applicable")
        and not saved_answers
    )


def finished_needs_new_work(session_dir: pathlib.Path) -> bool:
    """Return `needs_new_work` over the run's log and dir, for a verb that holds only the dir."""
    scan = session_log.scan_session_log(session_dir / layout.LOGS_NAME)
    return needs_new_work(
        finished=scan.finished,
        stop_reason=scan.stop_reason,
        verification=scan.verification,
        saved_answers=len(ipc.saved_question_answers(session_dir)),
    )


def resume_blocker(session_dir: pathlib.Path) -> str:
    """Return why the session's state cannot be resumed at all, or "".

    A coordinator has no loop, a lane's tree is no checkout's once its clone is gone,
    a worktree run whose worktree is
    gone continues through `fork`, and a run that never reached a turn has no snapshot.

    Args:
        session_dir: The session's state dir.

    Returns:
        The reason, naming the `agent6 fork` command when one continues the run; "" when
        no such state blocker applies.
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
        if missing := sessions_manifest.worktree_blocker(m.worktree):
            return missing
    if not (session_dir / "loop_state.json").is_file():
        return (
            "it never reached its first turn, so there is nothing to resume; start the task again"
        )
    return ""


def live_sessions_in_checkout(state_dir: pathlib.Path) -> tuple[str, ...]:
    """Return the ids of the live sessions working in the repository's own checkout.

    A worktree run, a fork and a fan-out lane work elsewhere and are left out; a session
    whose manifest cannot be read counts, since nothing says it works elsewhere.

    Args:
        state_dir: The repository's state dir.

    Returns:
        The session ids, in listing order.
    """
    live: list[str] = []
    for session_dir in session_dirs(state_dir):
        if not process.worker_is_alive(session_dir):
            continue
        manifest: sessions_manifest.SessionManifest | None = None
        with contextlib.suppress(sessions_manifest.ManifestError):
            manifest = sessions_manifest.read_manifest(session_dir)
        if manifest is None or (manifest.worktree is None and manifest.parallel is None):
            live.append(session_dir.name)
    return tuple(live)


def session_action_applies(action: str, *, fanout: bool, mode: str = "run") -> bool:
    """Hide loop and commit-chain actions on groups; their event records remain reviewable.

    A bounded read-only conversation's record is read, never acted on: every action is hidden.

    Args:
        action: The session action name.
        fanout: Whether the manifest records a fan-out, not an ordinary run with lanes.
        mode: The session's mode.

    Returns:
        Whether the action applies, independent of the session's liveness or git policy.
    """
    if mode == "read_only":
        return False
    return not (fanout and action in {"compact", "fork", "merge"})


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
        branch_tips: The caller's `git_read.run_ref_tips` snapshot.

    Returns:
        Whether its chain or branch has a tip beyond the base and outside the stamp.
    """
    if word == "undone":
        return False
    tips = _run_tips(manifest, branch_tips)
    # The branch may carry the operator's own commits; a merge stamps whichever tip it merged.
    return bool(tips - {manifest.base_sha}) and (
        manifest.merged is None or manifest.merged.tip not in tips
    )


def existing_branch(
    manifest: sessions_manifest.SessionManifest | None, branch_tips: Mapping[str, str] | None
) -> str:
    """Return the run's branch while the tips snapshot holds it, else "".

    The manifest names the branch at run start and git creates it at the first commit, so
    a run that never committed has none to review or merge.

    Args:
        manifest: The run's manifest, or None when unreadable.
        branch_tips: The caller's `git_read.run_ref_tips` snapshot, or None without one.

    Returns:
        The branch name, or "" without a manifest, a snapshot, a branch or a commit on it.
    """
    if manifest is None or branch_tips is None or not manifest.run_branch:
        return ""
    return manifest.run_branch if manifest.run_branch in branch_tips else ""


def _run_tips(
    manifest: sessions_manifest.SessionManifest, branch_tips: Mapping[str, str]
) -> set[str]:
    """Return the tips of the refs holding the run's work, its chain and its branch."""
    return {
        tip
        for tip in (
            branch_tips.get(git_read.chain_ref_for(manifest.session_id)),
            branch_tips.get(manifest.run_branch or ""),
        )
        if tip
    }


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
        branch_tips: The caller's `git_read.run_ref_tips` snapshot; with it the row says
            whether the chain or branch still holds commits no merge stamped, without
            it `unmerged` stays False (no mark, never a wrong one).

    Returns:
        The row.
    """
    logs = session_dir / layout.LOGS_NAME
    scan = session_log.scan_session_log(logs) if logs.is_file() else session_log.LogScan()
    manifest: sessions_manifest.SessionManifest | None = None
    with contextlib.suppress(sessions_manifest.ManifestError):
        manifest = sessions_manifest.read_manifest(session_dir)
    mode, label = scan.mode, scan.task_label
    if manifest is not None:
        if not label.task:
            label = label.fallback(manifest.user_task, manifest.continued_task)
        if mode == "?":
            mode = manifest.mode or mode
    if mode == "?" and not label.task:
        label = session_status.TaskLabel("(no logs)")
    word, reason = session_status.status_for_session_dir(session_dir, scan.status_facts())
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
        detail=reason,
        stop_reason=scan.stop_reason if scan.finished else "",
        unmerged=has_unmerged,
        cost_usd=scan.budget.session.usd,
        usd_partial=scan.budget.session.usd_partial,
        last_activity=scan.last_ep if scan.last_ep is not None else session_mtime(session_dir),
        verification=scan.verification,
        coordinator=lineage.coordinator if lineage is not None else "",
        lane=lineage.lane if lineage is not None else None,
        plan_observed_points=scan.budget.session.plan_observed_points,
        plan_cap=scan.budget.execution.plan_cap or 0.0,
        plan_used_percent=scan.budget.account.used_percent if scan.budget.account else 0.0,
        model=format.format_model_route(manifest.models.driver) if manifest is not None else "",
        model_from_flag=manifest.models.driver_from_flag if manifest is not None else False,
        worktree=str(manifest.worktree) if manifest is not None and manifest.worktree else "",
        worktree_missing=sessions_manifest.worktree_blocker(manifest.worktree) if manifest else "",
        branch=existing_branch(manifest, branch_tips),
        base_sha=manifest.base_sha if manifest is not None else "",
        source_tip=scan.source_tip or "",
        open_findings=scan.review.open_counts,
    )
