# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The children of a group: the front-end runtime, planning, spawning, running and dispatch."""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import functools
import os
import pathlib
import threading
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from agent6 import budget as agent6_budget
from agent6 import child_records, errors, kinds, paths
from agent6.app import child_launch, child_wait, parallel_results, stop, worktrees
from agent6.app import compare as app_compare
from agent6.app import reporter as app_reporter
from agent6.config import Config, ConfigError
from agent6.context import memory
from agent6.harness import child_repo
from agent6.harness import judge as harness_judge
from agent6.models import validate
from agent6.sessions import child_dispatch, event_log, ipc, journal, lineage
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel import (
    events,
    fold_session,
    format,
    produced_result,
    session_is_live,
    summarize_session_dir,
    tail_events,
    wire,
)
from agent6.viewmodel.session import log as session_log


class ParallelError(Exception):
    """The group could not be set up: over the child cap, or a route that names nothing."""


class SpawnRun(Protocol):
    """Spawn a detached agent6 command and locate its session dir: the front-end's primitive."""

    def __call__(
        self,
        argv: list[str],
        cwd: pathlib.Path,
        *,
        before: set[pathlib.Path],
        list_dirs: Callable[[], list[pathlib.Path]],
        env: dict[str, str],
    ) -> tuple[pathlib.Path | None, int, str]:
        """Spawn the command and wait for its session dir.

        Args:
            argv: The full command: the interpreter, its flags and the bootstrap module.
            cwd: Where to spawn.
            before: The session dirs that existed before the spawn.
            list_dirs: Lists the candidate session dirs.
            env: The child's environment.

        Returns:
            `(session_dir, pid, "")` once a new dir with a log appears, else
            `(None, pid, error)`, the pid 0 when nothing started.
        """
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class ChildRuntime:
    """The front-end primitives the parallel pipeline drives.

    Attributes:
        spawn: Launches a detached agent6 run and locates its session dir.
        build_provider: Builds the reviewer provider the auto-compare's judge uses.
        judging_status: Reports the judge's progress.
    """

    spawn: SpawnRun
    build_provider: app_compare.BuildProvider
    judging_status: app_compare.JudgingStatus


def child_settings(
    cfg: Config, selectors: Sequence[str | None]
) -> list[kinds.ModelSettings | None]:
    """Resolve each child's selector to its complete settings, in child order.

    Args:
        cfg: The run's config.
        selectors: One entry per child; None is the worker's own settings.

    Returns:
        The settings in child order, None where the child runs the worker's own.

    Raises:
        ConfigError: A selector names nothing.
    """
    return [cfg.child_settings(selector) for selector in selectors]


def bridge_spawner(
    spec: child_records.ChildSpec,
    task: str,
    *,
    operator_task: str | None = None,
    pins: Sequence[str] = (),
    cfg: Config,
    explicit_leaves: frozenset[str],
    origin: pathlib.Path,
    record_state: pathlib.Path,
    max_usd: float | None,
    auto_approve: bool = False,
    child_away: ipc.AwayMode = "wait",
    at: str | None = None,
    group_id: str,
    parent: str,
    gate_source: sessions_manifest.GateSource = "",
    limits: child_records.ChildLimits | None = None,
    allowance: agent6_budget.BudgetAllowance | None = None,
    contract: kinds.ConversationContract | None = None,
    runtime: ChildRuntime,
) -> child_records.ChildResult:
    """Clone the origin and start the child bootstrap detached in the clone.

    The task, pins, caps and lineage travel in the child's launch file under the origin's
    state dir; the host argv is the fixed bootstrap.

    Args:
        spec: The child.
        task: The child's task.
        pins: The parent's pins, passed out of band of the task.
        operator_task: The words before composition; None means `task` is uncomposed.
        cfg: The origin's effective config.
        explicit_leaves: The dotted leaves the origin's config layers set.
        origin: The repository.
        record_state: The state dir the child's record goes under: its dispatcher's.
        max_usd: The child's spend cap.
        auto_approve: The group's own `--auto-approve`, forwarded.
        child_away: The child's away-mode: "wait" parks a question for `agent6 attach`,
            "deny" answers it empty.
        at: A commit to detach the clone at; the parent's chain tip.
        group_id: The group the child belongs to.
        parent: The session every listing nests the child under.
        gate_source: Where the gate in the child's config came from, "" when it has none.
        limits: The turn cap and deadline the child runs under; None for the config's own.
        allowance: What the parent had left when the child started, which sets every
            cap; None for the config's.
        contract: The harness-owned read-only conversation; None for work.
        runtime: The front-end's spawn primitive.

    Returns:
        The child's result once its session dir is located, the run still going;
        `ok=False` when the clone or spawn failed, with the child's pid when one started.
    """
    branch = git_read.run_branch_for(spec.session_id)
    try:
        child_repo.clone_worktree(origin, spec.worktree)
        if at is not None:
            # A local clone hardlinks the whole odb, so the chain's commits are present.
            git_ops.checkout_detached(spec.worktree, at)
    except (child_repo.ChildRepoError, errors.GitError) as exc:
        return child_records.ChildResult(
            spec=spec, session_dir=spec.worktree, branch=branch, ok=False, error=str(exc)
        )
    # The child's own manifest records its lineage, so a parent death orphans nothing; the
    # gate's origin was resolved in the origin, where the repo signals are.
    private = child_launch.launch_dir(paths.state_dir(origin), group_id)
    try:
        config = worktrees.write_child_config(
            cfg, spec, explicit_leaves=explicit_leaves, into=private
        )
        launch_path = child_launch.write(
            private,
            child_launch.ChildLaunch(
                worktree=spec.worktree,
                record_state=record_state,
                session_id=spec.session_id,
                task=task,
                pins=tuple(pins),
                max_usd=max_usd,
                auto_approve=auto_approve,
                away=child_away,
                config=config.path,
                notes=config.notes,
                parent=parent,
                group=group_id,
                number=spec.number,
                gate_source=gate_source,
                operator_task=operator_task,
                limits=limits,
                allowance=allowance,
                contract=contract,
            ),
        )
    except (child_launch.LaunchError, OSError) as exc:
        return child_records.ChildResult(
            spec=spec, session_dir=spec.worktree, branch=branch, ok=False, error=str(exc)
        )
    # The clone's state dir is empty; the origin's memory and rulings are seeded into it.
    memory.seed_store(paths.state_dir(origin), paths.state_dir(spec.worktree))
    # The child's record lives in the origin's state from its first event.
    child_dir = child_wait.child_dir(record_state, spec.session_id, mode=spec.mode)

    def list_dirs() -> list[pathlib.Path]:
        return [child_dir] if child_dir.is_dir() else []

    # A child's model cannot dispatch work; only the harness may dispatch reviewers.
    markers = {
        "AGENT6_STREAM_TO_LOG": "1",
        "AGENT6_CHILD": "1",
        child_launch.ENV: str(launch_path),
    }
    session_dir, pid, err = runtime.spawn(
        child_launch.bootstrap_argv(),
        spec.worktree,
        before=set(),
        list_dirs=list_dirs,
        env={**os.environ, **markers},
    )
    if session_dir is None:
        return child_records.ChildResult(
            spec=spec,
            session_dir=child_dir,
            branch=branch,
            ok=False,
            error=err,
            pid=pid,
        )
    return child_records.ChildResult(
        spec=spec, session_dir=session_dir, branch=branch, ok=True, error="", pid=pid
    )


def run_child_to_completion(
    spec: child_records.ChildSpec,
    task: str,
    *,
    pins: Sequence[str] = (),
    cfg: Config,
    explicit_leaves: frozenset[str],
    origin: pathlib.Path,
    record_state: pathlib.Path,
    group: str,
    parent: str,
    runtime: ChildRuntime,
    max_usd: float | None = None,
    auto_approve: bool = False,
    child_away: ipc.AwayMode = "wait",
    gate_source: sessions_manifest.GateSource = "",
    spawner: child_records.ChildSpawner | None = None,
    at: str | None = None,
    import_lock: threading.Lock | None = None,
    poll_interval_s: float = child_wait.POLL_INTERVAL_S,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
    should_stop: Callable[[], bool] | None = None,
    stop_is_now: Callable[[], bool] | None = None,
    hard_stop: threading.Event | None = None,
    limits: child_records.ChildLimits | None = None,
    allowance: agent6_budget.BudgetAllowance | None = None,
    contract: kinds.ConversationContract | None = None,
    clock: Callable[[], float] = time.time,
    watch: Callable[[pathlib.Path, str, str], bool] | None = None,
) -> child_records.ChildResult:
    """Run one child to its end and import it into the origin.

    The parent runs a group of these on a thread pool, so each is
    self-contained. A child that failed to start, was still running at teardown or its
    deadline, refused its import, or ended without a result comes back `ok=False`; an
    imported branch is safe in the origin either way, and a clone that was not imported
    keeps its work. The child's process is owned to its end: one that does not stop when
    asked is killed and reaped, never left running.

    Args:
        spec: The child.
        task: The child's task.
        pins: The parent's pins.
        cfg: The origin's effective config.
        explicit_leaves: The dotted leaves the origin's config layers set.
        origin: The repository.
        record_state: The state dir the run's and its children's records live under.
        group: The group the child belongs to.
        parent: The session every listing nests the child under.
        runtime: The front-end's primitives.
        max_usd: The child's spend cap.
        auto_approve: The parent's own `--auto-approve`, forwarded.
        child_away: The child's away-mode.
        gate_source: Where the gate in the config came from, "" when it has none.
        spawner: Clones and spawns the child; the bridge spawner by default.
        at: A commit to detach the clone at.
        import_lock: Serializes the import across the group, since concurrent fetches
            into one repo race.
        poll_interval_s: How often the await polls.
        reporter: Receives the notes.
        should_stop: Polled to interrupt the await; the child then gets a stop request
            with the parent's urgency.
        stop_is_now: Whether that request carries the `now` urgency; None reads as `now`. A
            plain request lets the child finish its turn and waits for it, escalating to
            `now` when one arrives; `now` hands its command back and waits a bounded grace.
        hard_stop: Set to skip the grace.
        limits: The turn cap and deadline the child runs under; None for the config's own.
        allowance: What the parent had left when the child started, as the child's caps;
            None for the config's caps.
        contract: The harness-owned read-only conversation; None for work.
        clock: The wall clock the deadline is read against.
        watch: The parent's side of the wait, called on each poll with the child's dir,
            its label and its status: it counts the child's spend and answers its prompt,
            and returns whether the parent's budget is spent. None leaves the child's
            prompt to whoever attaches to it.

    Returns:
        The child's result, its `session_dir` the imported dir on success.
    """
    if spawner is None:
        spawner = functools.partial(
            bridge_spawner,
            pins=pins,
            cfg=cfg,
            origin=origin,
            record_state=record_state,
            max_usd=max_usd,
            auto_approve=auto_approve,
            child_away=child_away,
            at=at,
            explicit_leaves=explicit_leaves,
            group_id=group,
            parent=parent,
            gate_source=gate_source,
            limits=limits,
            allowance=allowance,
            contract=contract,
            runtime=runtime,
        )
    res = spawner(spec, task)
    if not res.ok:
        # A child that never reported its session dir is still a process of ours.
        how = child_wait.stop_child(res.pid)
        if how != "gone":
            res = dataclasses.replace(res, error=f"{res.error}; its process {how}")
        return res
    deadline_at = limits.deadline_at if limits is not None else None
    overdue = over_budget = False
    label = f"{'reviewer' if contract is not None else 'child'} {spec.number} {spec.session_id}"

    def on_poll(status: str) -> bool:
        nonlocal over_budget
        over_budget = watch is not None and watch(res.session_dir, label, status)
        return over_budget

    if not child_wait.await_child(
        res,
        poll_interval_s=poll_interval_s,
        should_stop=should_stop,
        deadline_at=deadline_at,
        clock=clock,
        on_poll=on_poll,
    ):
        overdue = deadline_at is not None and clock() >= deadline_at
        immediate = overdue or over_budget or stop_is_now is None or stop_is_now()
        asked = ipc.request_stop(res.session_dir, now=immediate)
        if not immediate:

            def escalate() -> bool:
                # The child finishes its turn; a later `now` or a teardown escalates.
                torn_down = hard_stop is not None and hard_stop.is_set()
                return torn_down or (stop_is_now is not None and stop_is_now())

            if child_wait.await_child(res, poll_interval_s=poll_interval_s, should_stop=escalate):
                asked = True
            else:
                asked = ipc.request_stop(res.session_dir, now=True)
                immediate = True
        if immediate and not child_wait.drain_child(
            res, poll_interval_s=poll_interval_s, hard_stop=hard_stop
        ):
            # Not reading its requests: ended by signal. The clone keeps its branch.
            how = child_wait.stop_child(res.pid)
            return child_records.ChildResult(
                spec=spec,
                session_dir=res.session_dir,
                branch=res.branch,
                ok=False,
                error=(_stopped_by(overdue=overdue, over_budget=over_budget) or "interrupted")
                + "; child "
                + ("was asked to stop and" if asked else "could not be asked to stop and")
                + f" did not end within the grace; its process {how}; not imported"
                " (its clone keeps its work)",
                pid=res.pid,
            )
    try:
        imported = parallel_results.import_child(res, origin, import_lock)
    except child_repo.ChildRepoError as exc:
        # The clone holds the only copy of the child's work: say where it is.
        return child_records.ChildResult(
            spec=res.spec,
            session_dir=res.session_dir,
            branch=res.branch,
            ok=False,
            error=f"{exc}; its work is kept in {res.spec.worktree}",
        )
    return land_child(
        res,
        imported,
        cfg=cfg,
        origin=origin,
        group=group,
        reporter=reporter,
        stopped_by=_stopped_by(overdue=overdue, over_budget=over_budget),
    )


def _stopped_by(*, overdue: bool, over_budget: bool) -> str:
    """Return what ended a child the parent stopped, "" for a stop the operator asked."""
    if overdue:
        return "ended at its deadline"
    return "stopped at the run's budget" if over_budget else ""


def land_child(
    res: child_records.ChildResult,
    imported: child_records.Imported,
    *,
    cfg: Config,
    origin: pathlib.Path,
    group: str,
    reporter: app_reporter.Reporter,
    stopped_by: str = "",
) -> child_records.ChildResult:
    """Finish an imported child: carry its memory back, clean its clone, read its outcome.

    Args:
        res: The child as it ran.
        imported: What the import landed.
        cfg: The origin's effective config.
        origin: The repository.
        group: The group the child belongs to.
        reporter: Receives the notes.
        stopped_by: What ended the child when the parent stopped it, "" otherwise.

    Returns:
        The child's result over the imported dir, its usage read from its record.
    """
    dest = imported.session_dir
    # The dispatcher's own state takes the memory back and holds the launch files.
    own_state = paths.state_dir(origin)
    parallel_results.carry_back(
        paths.state_dir(res.spec.worktree),
        own_state,
        dest,
        number=res.spec.number,
        reporter=reporter,
    )
    # Each child removes its own dirs only; the group dir goes with whichever empties it last.
    worktrees.cleanup(
        [res.spec],
        origin=origin,
        note_keep=lambda path, why: reporter.note(f"kept {path}: {why}"),
        record_state=sessions_layout.layout_of(dest).state_dir,
        worktree_root=res.spec.worktree.parent,
        base=worktrees.worktree_base(cfg, origin),
        launch_dir=child_launch.launch_dir(own_state, group),
        cfg=cfg,
    )
    result = recorded_result(
        res.spec, res.branch, dest, imported, origin=origin, stopped_by=stopped_by
    )
    scan = session_log.scan_session_log(dest / sessions_layout.LOGS_NAME)
    if scan.stop_reason == "setup_command_failed" and scan.setup_output:
        reporter.note(f"child {res.spec.number} [{res.spec.session_id}]: {scan.setup_output}")
    return result


def recorded_result(
    spec: child_records.ChildSpec,
    branch: str,
    dest: pathlib.Path,
    imported: child_records.Imported,
    *,
    origin: pathlib.Path,
    stopped_by: str = "",
) -> child_records.ChildResult:
    """Return a child's result from its imported record: its outcome, its work, its usage.

    Args:
        spec: The child.
        branch: Its branch.
        dest: Its imported session dir.
        imported: What the import landed.
        origin: The repository.
        stopped_by: What ended the child when the parent stopped it, "" otherwise.

    Returns:
        `ok` only for a deliberate end; the usage is read either way.
    """
    summary = summarize_session_dir(dest)
    usage = child_usage(dest)
    if not produced_result(summary.status):
        # Only a deliberate end joins the parent as a success.
        return child_records.ChildResult(
            spec=spec,
            session_dir=dest,
            branch=branch,
            ok=False,
            error=(f"{stopped_by}: " if stopped_by else "")
            + f"no result ({format.status_label(summary.status, summary.detail)});"
            f" {parallel_results.imported_branch_note(origin, branch)}",
            imported_ref=imported.ref,
            imported_tip=imported.tip,
            usage=usage,
        )
    return child_records.ChildResult(
        spec=spec,
        session_dir=dest,
        branch=branch,
        ok=True,
        error="",
        imported_ref=imported.ref,
        imported_tip=imported.tip,
        has_commits=child_has_commits(dest, imported),
        usage=usage,
        report=fold_session(
            tail_events(dest / sessions_layout.LOGS_NAME, follow=False)
        ).finish_summary,
        stopped_by=stopped_by,
    )


def _child_provider(session_dir: pathlib.Path) -> str:
    """Return the provider entry that drove a child, "" when its manifest names none."""
    try:
        driver = sessions_manifest.read_manifest(session_dir).models.driver
    except sessions_manifest.ManifestError:
        return ""
    return driver.provider if driver is not None else ""


def child_usage(session_dir: pathlib.Path) -> agent6_budget.UsageSettlement | None:
    """Read what a child used from its journal, as its parent settles it.

    Args:
        session_dir: The child's imported session dir.

    Returns:
        The inclusive spend and per-provider plan observations; None when descendant
        records are incomplete or unreadable, which leaves the child unsettled.
    """
    try:
        records = child_dispatch.read_records(session_dir)
    except child_dispatch.DispatchRecordError:
        return None
    if any(
        record.state != "complete" or any(not child.settled for child in record.children)
        for record in records
    ):
        return None
    # The journal includes settled descendants already. Never sum their journals again.
    scan = session_log.scan_session_log(session_dir / sessions_layout.LOGS_NAME)
    if any(
        child.session_id not in scan.budget.session.settled_children
        for record in records
        if record.source == "panel"
        for child in record.children
    ):
        return None
    readings: list[tuple[float, str, agent6_budget.PlanUsage]] = []
    for directory in (session_dir, *lineage.descendants_of(session_dir)):
        for raw in tail_events(directory / sessions_layout.LOGS_NAME, follow=False):
            if raw.get("type") != "budget.update":
                continue
            event = events.parse_event(raw)
            if not isinstance(event, journal.BudgetUpdate) or event.account is None:
                continue
            account = event.account
            provider = account.provider or _child_provider(directory)
            if not provider or not account.windows:
                continue
            # Credits are already in the inclusive USD total; only replay plan windows.
            plan = agent6_budget.PlanUsage(
                windows=tuple(
                    agent6_budget.PlanWindow(w.name, w.used_percent, w.window_minutes, w.resets_at)
                    for w in account.windows
                )
            )
            readings.append((events.event_epoch(raw.get("ts")) or 0.0, provider, plan))
    return agent6_budget.UsageSettlement(
        usd=scan.budget.session.usd,
        usd_partial=scan.budget.session.usd_partial,
        tokens=scan.budget.session.unmetered_tokens,
        plan_readings=tuple((p, r) for _, p, r in sorted(readings, key=lambda item: item[0])),
    )


def recover_children(
    group: str,
    children: Sequence[str],
    *,
    cfg: Config,
    origin: pathlib.Path,
    record_state: pathlib.Path,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> list[child_records.ChildResult]:
    """Return an interrupted group's children: the live stopped, the finished imported.

    A child with no session dir never started. A live one is stopped as `agent6 stop` stops
    a run and waited for. One whose clone is still on disk is imported and landed; one
    already landed is read as it stands.

    Args:
        group: The group id.
        children: The children's session ids, in child order.
        cfg: The origin's effective config.
        origin: The repository.
        record_state: The state dir the run's and its children's records live under.
        reporter: Receives the notes.

    Returns:
        One result per child that has a session dir, in child order.
    """
    worktree_root = worktrees.child_worktree_root(cfg, origin, group)
    results: list[child_records.ChildResult] = []
    for child, session_id in enumerate(children, start=1):
        matches = sessions_layout.session_matches(record_state, session_id)
        if not matches:
            continue
        child_dir = matches[0].session_dir
        mode = "reviewer" if matches[0].subdir == "reviewers" else "run"
        spec = child_records.ChildSpec(
            child, session_id, worktree_root / f"child-{child}", None, mode=mode
        )
        branch = git_read.run_branch_for(session_id)
        if session_is_live(child_dir):
            outcome = stop.stop_session(child_dir, wait_s=child_wait.STOP_GRACE_S)
            reporter.note(f"child {child} [{session_id}]: {outcome.message}")
        if (spec.worktree / ".git").exists():
            res = child_records.ChildResult(spec, child_dir, branch, True, "")
            try:
                imported = parallel_results.import_child(res, origin, None)
            except child_repo.ChildRepoError as exc:
                results.append(dataclasses.replace(res, ok=False, error=str(exc)))
                continue
            results.append(
                land_child(
                    res,
                    imported,
                    cfg=cfg,
                    origin=origin,
                    group=group,
                    reporter=reporter,
                )
            )
            continue
        ref = branch if git_read.branch_exists(origin, branch) else ""
        tip = (git_read.branch_tip_sha(origin, branch) or "") if ref else ""
        imported = child_records.Imported(child_dir, ref, tip)
        results.append(recorded_result(spec, branch, child_dir, imported, origin=origin))
    return results


def build_child_recoverer(
    cfg: Config,
    origin: pathlib.Path,
    record_state: pathlib.Path,
    *,
    mode: str,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> child_records.ChildRecoverer | None:
    """Build the recoverer a resumed run's loop reconciles interrupted dispatches with.

    Args:
        cfg: The run's config.
        origin: The repository.
        record_state: The state dir the run's and its children's records live under.
        mode: The session mode; only a run dispatches.
        reporter: Receives the notes.

    Returns:
        The recoverer, or None where no dispatch could have happened.
    """
    if mode != "run":
        return None

    def recover(group: str, children: Sequence[str]) -> list[child_records.ChildResult]:
        return recover_children(
            group, children, cfg=cfg, origin=origin, record_state=record_state, reporter=reporter
        )

    return recover


def child_has_commits(session_dir: pathlib.Path, imported: child_records.Imported) -> bool:
    """Return whether the imported tip holds a commit beyond the child's recorded base.

    Args:
        session_dir: The imported child's session dir, whose manifest records its base.
        imported: What the import landed.

    Returns:
        False for a record-only import or a tip at the base; True otherwise, and when the
        base is unrecorded, so an unknown child is merged rather than dropped.
    """
    if not imported.tip:
        return False
    try:
        base = sessions_manifest.read_manifest(session_dir).base_sha
    except sessions_manifest.ManifestError:
        return True
    return imported.tip != base


def build_child_ranker(
    cfg: Config,
    origin: pathlib.Path,
    record_state: pathlib.Path,
    *,
    parent_dir: pathlib.Path,
    base_sha: str,
    build_provider: app_compare.BuildProvider,
    judging_status: app_compare.JudgingStatus,
    budget: agent6_budget.BudgetTracker,
    events: event_log.EventSink,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> child_records.ChildRanker:
    """Build the ranker a rank join asks: `[harness.parallel].judges` over the imported children.

    The judges bill the parent's own budget. Each ranking is stamped on the parent
    and the children, journaled, and printed as the ranked table.

    Args:
        cfg: The parent's config.
        origin: The repository.
        record_state: The state dir the run's and its children's records live under.
        parent_dir: The parent's session dir, which holds the judges' transcripts.
        base_sha: The commit the children were cut from when the dispatch names none.
        build_provider: Builds the reviewer provider a judge sits in.
        judging_status: Shown around the judge calls.
        budget: The parent's tracker.
        events: The parent's journal.
        reporter: Receives the ranked table and a failed judge's line.

    Returns:
        The ranker.
    """

    def rank(
        results: Sequence[child_records.ChildResult], *, group: str, base: str
    ) -> child_records.Ranking:
        rows = {res.spec.session_id: summarize_session_dir(res.session_dir) for res in results}
        candidates = [
            harness_judge.CandidateBrief(
                session_id=res.spec.session_id,
                task=app_compare.manifest_task(res.session_dir, ""),
                diff=git_read.diff_range(origin, base or base_sha, res.imported_tip)
                if res.imported_tip
                else "",
                verification=rows[res.spec.session_id].verification,
                cost_usd=rows[res.spec.session_id].cost_usd,
            )
            for res in results
        ]
        outcome = app_compare.rank(
            cfg,
            candidates,
            transcript_dir=parent_dir / "judges",
            build_provider=build_provider,
            judging_status=judging_status,
            budget=budget,
            events=events,
            reporter=reporter,
        )
        winner = parallel_results.ranked_winner(
            candidates, outcome.ranking, harness_gate=cfg.harness.verify.when != "never"
        )
        parallel_results.stamp_compare_outcomes(
            candidates,
            outcome,
            winner=winner,
            record_state=record_state,
            parent_dir=parent_dir,
            group=group,
            reporter=reporter,
        )
        events.emit(
            journal.ParallelCompared(
                group=group,
                ranked_by=outcome.ranked_by,
                rationale=outcome.rationale,
                ranking=[
                    {
                        "session_id": rid,
                        "verify": app_compare.verify_word(rows[rid].verification),
                        "cost_usd": rows[rid].cost_usd,
                        "cost": rows[rid].usage_cell,
                    }
                    for rid in outcome.ranking
                ],
                judge_cost_usd=outcome.judge_cost_usd,
                judge_cost_partial=outcome.judge_cost_partial,
            )
        )
        # A child whose model owned git has no branch of agent6's: the table names its ref.
        model_refs: dict[str, str] = {}
        for res in results:
            with contextlib.suppress(sessions_manifest.ManifestError):
                manifest = sessions_manifest.read_manifest(res.session_dir)
                if manifest.git_control == "model":
                    model_refs[res.spec.session_id] = wire.commits_ref(manifest, origin)
        app_compare.print_ranked_candidates(
            candidates,
            outcome,
            rows=rows,
            committed={res.spec.session_id: res.has_commits for res in results},
            model_refs=model_refs,
            joins=parent_dir.name,
            reporter=reporter,
        )
        return child_records.Ranking(
            order=outcome.ranking,
            ranked_by=outcome.ranked_by,
            rationale=outcome.rationale,
            winner=winner,
        )

    return rank


def build_child_spawner(
    cfg: Config,
    origin: pathlib.Path,
    record_state: pathlib.Path,
    *,
    explicit_leaves: frozenset[str],
    parent_session_id: str,
    parent_mode: str = "run",
    runtime: ChildRuntime,
    max_usd: float | None = None,
    auto_approve: bool = False,
    child_away: ipc.AwayMode = "wait",
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
    watch: Callable[[pathlib.Path, str, str], bool] | None = None,
) -> child_records.GroupChildSpawner:
    """Build the group dispatcher the loop calls at a `/parallel` boundary.

    One call runs every child on its own thread, imports each under a shared lock
    and returns the results in dispatch order. Child ids are
    `<parent>-<group>-l<i>`.

    Args:
        cfg: The parent's config.
        origin: The repository.
        record_state: The state dir the run's and its children's records live under.
        explicit_leaves: The dotted leaves the parent's config layers set.
        parent_session_id: The parent's session id.
        parent_mode: The parent's session bucket owner.
        runtime: The front-end's primitives.
        max_usd: Each child's spend cap.
        auto_approve: The parent's own `--auto-approve`, forwarded to every child.
        child_away: Every child's away-mode, without a watch.
        reporter: Receives the notes.
        watch: The parent's side of each child's wait (`run_child_to_completion`); with it
            every child waits for the parent's operator to answer its prompts.

    Returns:
        The dispatcher.
    """
    if watch is not None:
        child_away = "wait"

    def dispatch(
        children: list[child_records.ChildTask], group: str, *, at: str | None = None
    ) -> list[child_records.ChildResult]:
        # The child cap binds before any clone; the refusal reaches the parent as feedback.
        if len(children) > cfg.harness.parallel.max_children:
            raise ParallelError(
                f"/parallel requests {len(children)} children but [harness.parallel].max_children ="
                f" {cfg.harness.parallel.max_children}. Request fewer, or raise"
                " [harness.parallel].max_children."
            )
        # The routes are validated before any clone; no model cache warns and proceeds.
        try:
            settings = [child.settings or cfg.child_settings(child.model) for child in children]
        except ConfigError as exc:
            raise ParallelError(str(exc)) from exc
        verdict = validate.validate_spec_models(
            [s.route if s is not None else None for s in settings], cfg
        )
        if verdict.refused:
            raise ParallelError(validate.refusal_message(verdict, directive=True))
        if verdict.warned:
            reporter.warn(validate.warning_message(verdict))
        # One name for the worktree dir, the child ids and the stamped lineage: the sweeps derive
        # the clone path from it.
        group_id = f"{parent_session_id}-{group}"
        try:
            worktree_root = worktrees.place(cfg, origin, group_id)
        except (errors.OperatorError, errors.GitError, OSError) as exc:
            raise ParallelError(str(exc)) from exc
        specs = [
            child_records.ChildSpec(
                number=i,
                session_id=f"{group_id}-l{i}",
                worktree=worktree_root / f"child-{i}",
                settings=chosen,
                mode="reviewer" if child.contract is not None else "run",
            )
            for i, (chosen, child) in enumerate(zip(settings, children, strict=True), start=1)
        ]
        paths.mkdir_for_real_user(sessions_layout.bucket_dir(record_state, "runs"))
        import_lock = threading.Lock()
        parent_dir = (
            sessions_layout.bucket_dir(record_state, kinds.session_bucket(parent_mode))
            / parent_session_id
        )
        hard_stop = threading.Event()

        def should_stop() -> bool:
            # Both stop channels interrupt the wait; the loop consumes their markers after.
            return (
                hard_stop.is_set()
                or ipc.steer_answer_is_abort(parent_dir)
                or ipc.stop_request_pending(parent_dir)
            )

        def stop_is_now() -> bool:
            return (
                hard_stop.is_set()
                or ipc.steer_answer_is_abort(parent_dir)
                or ipc.stop_request_is_now(parent_dir)
            )

        # The parent's manifest pins the gate the children's config carries.
        gate_source = ""
        if cfg.harness.verify.command:
            try:
                gate_source = sessions_manifest.read_manifest(parent_dir).harness.gate.source
            except sessions_manifest.ManifestError:
                gate_source = ""

        def one(
            pair: tuple[child_records.ChildSpec, child_records.ChildTask],
        ) -> child_records.ChildResult:
            spec, child = pair
            return run_child_to_completion(
                spec,
                child.task,
                pins=child.pins,
                at=at,
                cfg=cfg,
                gate_source=gate_source,
                explicit_leaves=explicit_leaves,
                origin=origin,
                record_state=record_state,
                group=group_id,
                parent=parent_session_id,
                runtime=runtime,
                max_usd=max_usd,
                auto_approve=auto_approve,
                child_away=child_away,
                import_lock=import_lock,
                reporter=reporter,
                should_stop=should_stop,
                stop_is_now=stop_is_now,
                hard_stop=hard_stop,
                limits=child.limits,
                allowance=child.allowance,
                contract=child.contract,
                watch=watch,
            )

        pairs = list(zip(specs, children, strict=True))
        if len(pairs) > 1:
            # Not a with-block: its exit would join every child await on an interrupt.
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs))
            try:
                futures = [pool.submit(one, p) for p in pairs]
                # A child thread that raises (a bug, not a child failure) aborts the group now.
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_EXCEPTION
                )
                for f in done:
                    exc = f.exception()
                    if exc is not None:
                        raise exc
                results = [f.result() for f in futures]  # submit order = child order
            except BaseException:
                # The child threads request a stop and exit; the children keep running detached.
                hard_stop.set()
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            pool.shutdown(wait=True)
            return results
        return [one(p) for p in pairs]

    return dispatch


def build_parent_spawner(
    cfg: Config,
    origin: pathlib.Path,
    record_state: pathlib.Path,
    *,
    explicit_leaves: frozenset[str],
    mode: str,
    session_id: str,
    runtime: ChildRuntime,
    max_usd: float | None = None,
    auto_approve: bool = False,
    child_away: ipc.AwayMode = "wait",
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
    watch: Callable[[pathlib.Path, str, str], bool] | None = None,
) -> child_records.GroupChildSpawner | None:
    """Build the `/parallel` dispatcher for a run's loop, or None when dispatch is unavailable.

    A child worker retains the executor only for its configured harness panels.

    Args:
        cfg: The run's config.
        origin: The repository.
        record_state: The state dir the run's and its children's records live under.
        explicit_leaves: The dotted leaves the run's config layers set.
        mode: The session mode.
        session_id: The run's session id.
        runtime: The front-end's primitives.
        max_usd: Each child's spend cap.
        auto_approve: The run's own `--auto-approve`, forwarded to every child.
        child_away: Every child's away-mode, without a watch.
        reporter: Receives the notes.
        watch: The run's side of each child's wait: its prompts and its spend.

    Returns:
        The dispatcher, or None.
    """
    authority = child_launch.dispatch_authority(cfg, mode=mode, session_id=session_id, root=origin)
    if not (authority.work or authority.panels):
        return None
    return build_child_spawner(
        cfg,
        origin,
        record_state,
        explicit_leaves=explicit_leaves,
        parent_session_id=session_id,
        parent_mode=mode,
        runtime=runtime,
        max_usd=max_usd,
        auto_approve=auto_approve,
        child_away=child_away,
        reporter=reporter,
        watch=watch,
    )
