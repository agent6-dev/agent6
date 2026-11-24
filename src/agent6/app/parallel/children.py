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
import shutil
import threading
import time
from collections.abc import Callable, Generator, Sequence
from typing import Any, Protocol

from agent6 import budget as agent6_budget
from agent6 import child_records, errors, journal_tail, kinds, paths, task_text
from agent6.app import compare as app_compare
from agent6.app import reporter as app_reporter
from agent6.app import stop, worktrees
from agent6.app.parallel import launch as child_launch
from agent6.app.parallel import relay as child_watch
from agent6.app.parallel import results as parallel_results
from agent6.app.parallel import resume as child_resume
from agent6.app.parallel import wait as child_wait
from agent6.config import model as config_model
from agent6.config import validation
from agent6.context import memory
from agent6.harness.parallel import repo as child_repo
from agent6.models import validate
from agent6.sessions import child_dispatch, event_log, ipc, journal, lineage, process
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel import children as vm_children
from agent6.viewmodel import events, format, listing, tail, wire
from agent6.viewmodel.session import log as session_log
from agent6.viewmodel.session import state as session_state
from agent6.viewmodel.session import status as session_status


class ParallelError(Exception):
    """The group could not be set up: over the child cap, or a route that names nothing."""


type PollChild = Callable[
    [pathlib.Path, str, str | child_watch.ChildStartup, Callable[[], bool]], bool
]


class SpawnRun(Protocol):
    """Spawn an owned child and wait for its start report, exit or requested stop."""

    def __call__(
        self,
        argv: list[str],
        cwd: pathlib.Path,
        *,
        ready: Callable[[int], pathlib.Path | None],
        stop_reason: Callable[[int], str],
        env: dict[str, str],
    ) -> tuple[pathlib.Path | None, int, str]:
        """Spawn the command on its watch thread.

        Args:
            argv: The full command: the interpreter, its flags and the bootstrap module.
            cwd: Where to spawn.
            ready: The session dir once this process reports starting its execution.
            stop_reason: Why to stop and reap it, empty to keep waiting.
            env: The child's environment.

        Returns:
            `(session_dir, pid, "")` after the report, else `(None, pid, error)`
            after exit. The pid is zero if the command could not start.
        """
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class ChildRuntime:
    """The front-end primitives the parallel pipeline drives.

    Attributes:
        spawn: Launches a detached agent6 run and locates its session dir.
        observe: Builds a listener for the new journal events of one child execution.
    """

    spawn: SpawnRun
    observe: Callable[[child_records.ChildSpec], Callable[[dict[str, Any]], None]] | None = None

    @contextlib.contextmanager
    def observing(
        self,
        spec: child_records.ChildSpec,
        state: pathlib.Path,
        continuing: bool,
        parent_events: event_log.EventSink | None,
        watch: PollChild | None,
    ) -> Generator[PollChild | None]:
        """Give the observer a last read before the parent can merge the child.

        Args:
            spec: The child to watch.
            state: The origin's record state.
            continuing: The child resumes in the same directory.
            parent_events: The parent journal where operator steers are reported.
            watch: Count spend and relay prompts while the child runs.

        Yields:
            The watch for this execution of the child.
        """
        if self.observe is None and parent_events is None:
            yield watch
            return
        directory = child_wait.child_dir(state, spec.session_id, mode=spec.mode)
        listener = self.observe(spec) if self.observe is not None else None
        reader = tail.LogTail(directory / sessions_layout.LOGS_NAME)
        if continuing:
            reader.read()  # The next execution appends to the same journal.

        def drain() -> None:
            for row in reader.read():
                if parent_events is not None and row.get("type") in (
                    journal.SteerInjected.TYPE,
                    journal.SteerFromParent.TYPE,
                ):
                    parent_events.emit(
                        journal.ParallelChildActivity(
                            child=spec.session_id,
                            activity="steered",
                            detail=str(row.get("text", "")),
                        )
                    )
                if listener is not None:
                    listener(row)

        def poll(
            directory: pathlib.Path,
            label: str,
            status: str | child_watch.ChildStartup,
            interrupted: Callable[[], bool],
        ) -> bool:
            drain()
            return watch(directory, label, status, interrupted) if watch is not None else False

        try:
            yield poll
        finally:
            drain()


def child_settings(
    cfg: config_model.Config, selectors: Sequence[str | None]
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
    cfg: config_model.Config,
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
    continuation: str = "",
    workspace: child_resume.Workspace | None = None,
    runtime: ChildRuntime,
    stop_reason: Callable[[child_watch.ChildStartup], str] = lambda _: "",
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
        continuation: The next instruction for the child; empty starts a new child.
        workspace: The parent-validated checkout for the continuation.
        runtime: The front-end's spawn primitive.
        stop_reason: Why to end the child before its start report, empty to keep waiting.

    Returns:
        The child's result once its session dir is located, the run still going;
        `ok=False` when the clone or spawn failed, with the child's pid when one started.
    """
    branch = git_read.run_branch_for(spec.session_id)
    private = child_launch.launch_dir(paths.state_dir(origin), group_id)
    try:
        if continuation:
            clone = (
                workspace.checked(spec.worktree)
                if workspace
                else child_resume.check_workspace(
                    spec, private, parent=parent, record_state=record_state
                )
            )
        else:
            child_repo.clone_worktree(origin, spec.worktree)
            if at is not None:
                # A local clone hardlinks the whole odb, so the chain's commits are present.
                git_ops.checkout_detached(spec.worktree, at)
            clone = child_launch.CloneReceipt.capture(spec.worktree)
    except (child_launch.LaunchError, child_repo.ChildRepoError, errors.GitError) as exc:
        return child_records.ChildResult(
            spec=spec, session_dir=spec.worktree, branch=branch, ok=False, error=str(exc)
        )
    # The child's own manifest records its lineage, so a parent death orphans nothing; the
    # gate's origin was resolved in the origin, where the repo signals are.
    try:
        config = worktrees.write_child_config(
            cfg, spec, explicit_leaves=explicit_leaves, into=private
        )
        launch_path = child_launch.write(
            private,
            child_launch.ChildLaunch(
                worktree=spec.worktree,
                clone=clone,
                record_state=record_state,
                session_id=spec.session_id,
                task=task,
                task_source=spec.task_source,
                sibling_notice=spec.sibling_notice,
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
                continuation=continuation,
            ),
        )
    except (child_launch.LaunchError, OSError) as exc:
        return child_records.ChildResult(
            spec=spec, session_dir=spec.worktree, branch=branch, ok=False, error=str(exc)
        )
    # The clone's state dir is empty; the origin's memory and rulings are seeded into it.
    memory.seed_store(paths.state_dir(origin), clone.state)
    # The child's record lives in the origin's state from its first event.
    child_dir = child_wait.child_dir(record_state, spec.session_id, mode=spec.mode)
    log = child_dir / sessions_layout.LOGS_NAME
    reader = tail.LogTail(log)
    try:
        with log.open("rb") as fh:
            start_at = journal_tail.complete_end(fh)
    except FileNotFoundError:
        start_at = 0
    report = journal.SessionResume.TYPE if continuation else journal.SessionStart.TYPE

    def ready(pid: int) -> pathlib.Path | None:
        if process.read_worker_pid(child_dir) not in (None, pid):
            return None
        rows = reader.read(start_at=start_at)
        return (
            child_dir
            if any(row.get("type") == report and row.get("worker_pid") == pid for row in rows)
            else None
        )

    # A child's model cannot dispatch work; only the harness may dispatch reviewers.
    markers = {
        "AGENT6_STREAM_TO_LOG": "1",
        "AGENT6_CHILD": "1",
        child_launch.ENV: str(launch_path),
    }
    session_dir, pid, err = runtime.spawn(
        child_launch.bootstrap_argv(),
        spec.worktree,
        ready=ready,
        stop_reason=lambda pid: stop_reason(child_watch.ChildStartup(pid, start_at)),
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
    cfg: config_model.Config,
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
    continuation: str = "",
    workspace: child_resume.Workspace | None = None,
    clock: Callable[[], float] = time.time,
    watch: PollChild | None = None,
    parent_events: event_log.EventSink | None = None,
) -> child_records.ChildResult:
    """Run one child to its end and import it into the origin.

    The parent runs each of a group's children on its own thread, so each is
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
        continuation: The next instruction for the child; empty starts a new child.
        workspace: The parent-validated checkout for the continuation.
        clock: The wall clock the deadline is read against.
        parent_events: The parent execution's sink for child activity.
        watch: The parent's side of the wait, called on each poll with the child's dir,
            its label and its status: it counts the child's spend and answers its prompt,
            and returns whether the parent's budget is spent. None leaves the child's
            prompt to whoever attaches to it.

    Returns:
        The child's result, its `session_dir` the imported dir on success.
    """
    deadline_at = limits.deadline_at if limits is not None else None
    label = f"{spec.number_label} {spec.session_id}"

    startup_stop = ""

    def parent_stop_reason() -> str:
        if deadline_at is not None and clock() >= deadline_at:
            return "ended at its deadline"
        if (hard_stop is not None and hard_stop.is_set()) or (should_stop and should_stop()):
            return "stopped by its parent"
        return ""

    def interrupted() -> bool:
        return bool(parent_stop_reason())

    def stop_before_start(startup: child_watch.ChildStartup) -> str:
        nonlocal startup_stop
        startup_stop = parent_stop_reason()
        if not startup_stop and watch is not None:
            directory = child_wait.child_dir(record_state, spec.session_id, mode=spec.mode)
            if watch(directory, label, startup, interrupted):
                startup_stop = "stopped at the run's budget"
        return startup_stop

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
            continuation=continuation,
            workspace=workspace,
            runtime=runtime,
            stop_reason=stop_before_start,
        )
    res = spawner(spec, task)
    if not res.ok:
        if workspace is not None and workspace.fresh:
            # No worker started in the clone made for this continuation.
            shutil.rmtree(spec.worktree, ignore_errors=True)
        return dataclasses.replace(res, started=False)
    if parent_events is not None:
        parent_events.emit(
            journal.ParallelChildActivity(
                child=spec.session_id, activity="started", detail=task[:200]
            )
        )
    overdue = over_budget = False

    def on_poll(status: str) -> bool:
        nonlocal over_budget
        over_budget = watch is not None and watch(res.session_dir, label, status, interrupted)
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
        imported = parallel_results.import_child(
            res, origin, import_lock, cfg, continuing=bool(continuation)
        )
    except child_repo.ChildRepoError as exc:
        # The clone holds the only copy of the child's work: say where it is.
        return child_records.ChildResult(
            spec=res.spec,
            session_dir=res.session_dir,
            branch=res.branch,
            ok=False,
            error=f"{exc}; its work is kept in {res.spec.worktree}",
        )
    return finish_import(
        res,
        imported,
        cfg=cfg,
        origin=origin,
        group=group,
        reporter=reporter,
        stopped_by=startup_stop
        or _stopped_by(overdue=overdue, over_budget=over_budget)
        or ("stopped by its parent" if should_stop is not None and should_stop() else ""),
    )


def _stopped_by(*, overdue: bool, over_budget: bool) -> str:
    """Return what ended a child the parent stopped, "" for a stop the operator asked."""
    if overdue:
        return "ended at its deadline"
    return "stopped at the run's budget" if over_budget else ""


def finish_import(
    res: child_records.ChildResult,
    imported: child_records.Imported,
    *,
    cfg: config_model.Config,
    origin: pathlib.Path,
    group: str,
    reporter: app_reporter.Reporter,
    stopped_by: str = "",
) -> child_records.ChildResult:
    """Finish an imported child: carry its memory back, clean its clone, read its outcome.

    Args:
        res: The child as it ran.
        imported: What the import retained.
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
        label=res.spec.number_label,
        reporter=reporter,
    )
    result = recorded_result(
        res.spec, res.branch, dest, imported, origin=origin, stopped_by=stopped_by
    )
    # A child recorded `stopped` keeps its clone and private launch for its continuation.
    if result.ok and result.stopped_by:
        return result
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
    scan = session_log.scan_session_log(dest / sessions_layout.LOGS_NAME)
    if scan.stop_reason == "setup_command_failed" and scan.setup_output:
        reporter.note(f"{res.spec.number_label} [{res.spec.session_id}]: {scan.setup_output}")
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
        imported: What the import retained.
        origin: The repository.
        stopped_by: What ended the child when the parent stopped it, "" otherwise.

    Returns:
        `ok` only for a deliberate end; the usage is read either way.
    """
    summary = listing.summarize_session_dir(dest)
    usage = child_usage(dest)
    recorded = session_state.fold_session(
        tail.tail_events(dest / sessions_layout.LOGS_NAME, follow=False)
    )
    call = recorded.last_model_call
    context = child_records.ChildContext(
        tokens=call.context_tokens if call is not None and call.context_tokens > 0 else None,
        last_run_at=recorded.last_event_at,
    )
    ok = session_status.produced_result(summary.status)
    commits = child_commits(origin, dest, imported)
    noun = "commit" if commits == 1 else "commits"
    held = (
        f"{imported.ref} holds {commits} {noun}"
        if imported.ref
        else "no branch (the run made no commit)"
    )
    return child_records.ChildResult(
        spec=spec,
        session_dir=dest,
        branch=branch,
        ok=ok,
        error=""
        if ok
        else (
            (f"{stopped_by}: " if stopped_by else "")
            + f"no result ({format.status_label(summary.status, summary.detail)}); {held}"
        ),
        imported_ref=imported.ref,
        imported_tip=imported.tip,
        has_commits=child_has_commits(dest, imported),
        commits=commits,
        verification=summary.verification,
        usage=usage,
        report=recorded.finish_summary if ok else "",
        context=context,
        # A stop counts as a result, and one that never merges by itself: the child's own
        # record says it was one, whoever asked.
        stopped_by=stopped_by or ("stopped" if summary.status == "stopped" else ""),
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
        The inclusive cap reading, account balances and per-provider plan readings.
        Incomplete or unreadable descendant records leave the child unsettled (None).
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
        for raw in tail.tail_events(directory / sessions_layout.LOGS_NAME, follow=False):
            if raw.get("type") != "budget.update":
                continue
            event = events.parse_event(raw)
            if not isinstance(event, journal.BudgetUpdate) or event.account is None:
                continue
            account = event.account
            provider = account.provider or _child_provider(directory)
            if not provider or not account.windows:
                continue
            # Balance readings have their own account and observation time.
            plan = agent6_budget.PlanUsage(
                windows=tuple(
                    agent6_budget.PlanWindow(w.name, w.used_percent, w.window_minutes, w.resets_at)
                    for w in account.windows
                )
            )
            readings.append((events.event_epoch(raw.get("ts")) or 0.0, provider, plan))
    return agent6_budget.UsageSettlement(
        usd=scan.budget.session.usd,
        usd_observed_credits=scan.budget.session.usd_observed_credits,
        credit_readings=scan.credit_readings,
        usd_partial=scan.budget.session.usd_partial,
        tokens=scan.budget.session.unmetered_tokens,
        model_tokens=scan.budget.model_tokens.total,
        plan_readings=tuple((p, r) for _, p, r in sorted(readings, key=lambda item: item[0])),
    )


def recover_children(
    group: str,
    children: Sequence[str],
    *,
    cfg: config_model.Config,
    origin: pathlib.Path,
    record_state: pathlib.Path,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> list[child_records.ChildResult]:
    """Return an interrupted group's children: the live stopped, the finished imported.

    A child with no session dir never started. A live one is stopped as `agent6 stop` stops
    a run and waited for. One whose clone is still on disk is imported and finalized; one
    already imported is read as it stands.

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
        if listing.session_is_live(child_dir):
            outcome = stop.stop_session(child_dir, wait_s=child_wait.STOP_GRACE_S)
            reporter.note(f"child {child} [{session_id}]: {outcome.message}")
        if (spec.worktree / ".git").exists():
            res = child_records.ChildResult(spec, child_dir, branch, True, "")
            try:
                imported = parallel_results.import_child(res, origin, None, cfg, continuing=True)
            except child_repo.ChildRepoError as exc:
                results.append(dataclasses.replace(res, ok=False, error=str(exc)))
                continue
            results.append(
                finish_import(
                    res,
                    imported,
                    cfg=cfg,
                    origin=origin,
                    group=group,
                    reporter=reporter,
                )
            )
            continue
        try:
            manifest = sessions_manifest.read_manifest(child_dir)
        except sessions_manifest.ManifestError as exc:
            results.append(child_records.ChildResult(spec, child_dir, branch, False, str(exc)))
            continue
        ref = git_read.chain_ref_for(session_id) if manifest.git_control == "model" else branch
        tip = git_read.chain_tip(origin, ref) or ""
        ref = ref if tip else ""
        imported = child_records.Imported(child_dir, ref, tip)
        results.append(recorded_result(spec, branch, child_dir, imported, origin=origin))
    return results


def build_child_recoverer(
    cfg: config_model.Config,
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
        imported: What the import retained.

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


def child_commits(
    origin: pathlib.Path, session_dir: pathlib.Path, imported: child_records.Imported
) -> int:
    """Return how many commits the imported tip holds beyond the child's recorded base.

    Args:
        origin: The repository the child was imported into.
        session_dir: The imported child's session dir, whose manifest records its base.
        imported: What the import retained.

    Returns:
        The count; 0 for a record-only import, a tip at the base, or an unrecorded base.
    """
    if not imported.tip:
        return 0
    try:
        base = sessions_manifest.read_manifest(session_dir).base_sha
    except sessions_manifest.ManifestError:
        return 0
    return (git_read.commit_count(origin, base, imported.tip) or 0) if base else 0


def build_rank_recorder(
    origin: pathlib.Path,
    record_state: pathlib.Path,
    *,
    parent_dir: pathlib.Path,
    events: event_log.EventSink,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> child_records.RankRecorder:
    """Build what records a ranking the dispatcher made of a task's attempts.

    Each ranking is stamped on the parent and the children, journaled, and printed as the
    ranked table. The judges ran as the parent's reviewer children, so their spend is on its
    budget already.

    Args:
        origin: The repository.
        record_state: The state dir the run's and its children's records live under.
        parent_dir: The parent's session dir.
        events: The parent's journal.
        reporter: Receives the ranked table and why the mechanical order ranked.

    Returns:
        The recorder.
    """

    def record(
        candidates: Sequence[child_records.CandidateBrief],
        results: Sequence[child_records.ChildResult],
        ranking: child_records.Ranking,
        *,
        group: str,
    ) -> None:
        briefs = list(candidates)
        rows = {
            res.spec.session_id: listing.summarize_session_dir(res.session_dir) for res in results
        }
        outcome = app_compare.outcome_of(ranking)
        if ranking.fallback and len(briefs) > 1:
            reporter.note(f"ranked by verify result, then cost: {ranking.fallback}")
        parallel_results.stamp_compare_outcomes(
            briefs,
            outcome,
            winner=ranking.winner,
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
            briefs,
            outcome,
            rows=rows,
            committed={res.spec.session_id: res.has_commits for res in results},
            model_refs=model_refs,
            merge_parents={ranking.winner: parent_dir.name} if ranking.winner else None,
            reporter=reporter,
        )

    return record


@dataclasses.dataclass(frozen=True, slots=True)
class GroupSpawner:
    """Start groups of children in the background, and end what still runs at teardown.

    Attributes:
        start: Starts one group and returns its children at once.
        teardown: Set to end every child's watch now, without the stop's grace.
    """

    start: child_records.GroupChildSpawner
    teardown: threading.Event

    def __call__(
        self, children: list[child_records.ChildTask], group: str, *, at: str | None = None
    ) -> list[child_records.StartedChild]:
        """Start the children as one group.

        Args:
            children: The children, in child order.
            group: The group id, `p<seq>`.
            at: The commit each clone is detached at.

        Returns:
            The started children, at once.
        """
        return self.start(children, group, at=at)

    def close(self) -> None:
        """End every child still running: its watch stops it by signal and imports nothing."""
        self.teardown.set()


def _watch_child(
    run: Callable[[], child_records.ChildResult],
    result: concurrent.futures.Future[child_records.ChildResult],
    session_id: str,
    parent_events: event_log.EventSink | None,
) -> None:
    """Report the child's end before making its result available to its parent.

    Args:
        run: The child execution on this thread.
        result: The result the parent waits on.
        session_id: The child's id.
        parent_events: The parent execution's sink when it has one.
    """
    try:
        try:
            ended = run()
        except Exception as exc:
            if parent_events is not None:
                parent_events.emit(
                    journal.ParallelChildActivity(
                        child=session_id, activity="failed", detail=str(exc)
                    )
                )
            raise
        if parent_events is not None:
            parent_events.emit(
                journal.ParallelChildActivity(
                    child=session_id,
                    activity="failed"
                    if not ended.ok
                    else "stopped"
                    if ended.stopped_by
                    else "finished",
                    detail=ended.error or ended.stopped_by or ended.report,
                )
            )
        result.set_result(ended)
    except BaseException as exc:
        result.set_exception(exc)


def _specs_with_siblings(
    parent_dir: pathlib.Path,
    specs: list[child_records.ChildSpec],
    children: list[child_records.ChildTask],
) -> list[child_records.ChildSpec]:
    """Attach the parent's other children to each launch.

    Args:
        parent_dir: The parent whose records name the earlier children.
        specs: The child launches in this batch.
        children: The tasks in launch order.

    Returns:
        The specs with a notice for each child that has siblings.
    """
    siblings: dict[str, tuple[str, str]] = {}
    for row in vm_children.children_of(parent_dir):
        if row.status in {"unreadable", "unreported", "?"}:
            state = "end unknown"
        elif row.status in session_status.LIVE_STATUS_WORDS or row.status in {
            "created",
            "not started",
            "parked",
        }:
            state = "not ended"
        else:
            state = "ended"
        siblings[row.session_id] = (state, row.task)
    # A new batch has ids before any of its journals exist.
    siblings.update(
        (spec.session_id, ("not ended", child.task))
        for spec, child in zip(specs, children, strict=True)
    )
    if len(siblings) < 2:
        return specs
    rows = {
        sid: f"[{sid}] {state}: {task_text.clip_title(task_text.task_headline(task))}"
        for sid, (state, task) in siblings.items()
    }
    launches: list[child_records.ChildSpec] = []
    for spec in specs:
        lines = [line for sid, line in rows.items() if sid != spec.session_id]
        launches.append(
            dataclasses.replace(spec, sibling_notice="\n".join(["[harness siblings]", *lines]))
        )
    return launches


def build_child_spawner(
    cfg: config_model.Config,
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
    watch: PollChild | None = None,
    parent_events: event_log.EventSink | None = None,
) -> GroupSpawner:
    """Build the group dispatcher the loop calls at a `/parallel` boundary.

    One call starts every child on a watch thread of its own and returns at once; each
    watch clones, spawns and awaits its child, imports it under a lock the execution's
    children share, and resolves the child's result. Child ids are `<parent>-<group>-l<i>`.

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
        parent_events: The parent execution's sink for child activity.
        watch: The parent's side of each child's wait (`run_child_to_completion`); with it
            every child waits for the parent's operator to answer its prompts.

    Returns:
        The dispatcher.
    """
    if watch is not None:
        child_away = "wait"
    # One fetch into the repository at a time, across every group this execution starts.
    import_lock = threading.Lock()
    hard_stop = threading.Event()

    def dispatch(
        children: list[child_records.ChildTask], group: str, *, at: str | None = None
    ) -> list[child_records.StartedChild]:
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
        except validation.ConfigError as exc:
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
                task_source="panel" if child.contract is not None else child.task_source,
            )
            for position, (chosen, child) in enumerate(
                zip(settings, children, strict=True), start=1
            )
            for i in (child.continuation.number if child.continuation is not None else position,)
        ]
        workspaces: dict[int, child_resume.Workspace] = {}
        try:
            for spec, child in zip(specs, children, strict=True):
                if child.continuation is not None:
                    workspaces[spec.number] = child_resume.prepare(
                        spec,
                        child.continuation,
                        origin=origin,
                        private=child_launch.launch_dir(paths.state_dir(origin), group_id),
                        parent=parent_session_id,
                        record_state=record_state,
                    )
        except (
            child_launch.LaunchError,
            child_repo.ChildRepoError,
            errors.GitError,
            OSError,
        ) as exc:
            raise ParallelError(str(exc)) from exc
        paths.mkdir_for_real_user(sessions_layout.bucket_dir(record_state, "runs"))
        parent_dir = (
            sessions_layout.bucket_dir(record_state, kinds.session_bucket(parent_mode))
            / parent_session_id
        )
        specs = _specs_with_siblings(parent_dir, specs, children)

        # The parent's manifest pins the gate the children's config carries.
        gate_source = ""
        if cfg.harness.verify.command:
            try:
                gate_source = sessions_manifest.read_manifest(parent_dir).harness.gate.source
            except sessions_manifest.ManifestError:
                gate_source = ""

        def one(
            spec: child_records.ChildSpec, child: child_records.ChildTask, asked: threading.Event
        ) -> child_records.ChildResult:
            def should_stop() -> bool:
                # Both stop channels interrupt the wait; the loop consumes their markers after.
                return (
                    hard_stop.is_set()
                    or asked.is_set()
                    or ipc.steer_answer_is_abort(parent_dir)
                    or ipc.stop_request_pending(parent_dir)
                )

            def stop_is_now() -> bool:
                return (
                    hard_stop.is_set()
                    or asked.is_set()
                    or ipc.steer_answer_is_abort(parent_dir)
                    or ipc.stop_request_is_now(parent_dir)
                )

            with runtime.observing(
                spec, record_state, child.continuation is not None, parent_events, watch
            ) as poll:
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
                    continuation=child.continuation.instruction
                    if child.continuation is not None
                    else "",
                    workspace=workspaces.get(spec.number),
                    watch=poll,
                    parent_events=parent_events,
                )

        started: list[child_records.StartedChild] = []
        for spec, child in zip(specs, children, strict=True):
            asked = threading.Event()
            result: concurrent.futures.Future[child_records.ChildResult] = (
                concurrent.futures.Future()
            )
            # The kernel ties a child to the thread that forked it: its watch outlives it.
            threading.Thread(
                target=_watch_child,
                args=(
                    functools.partial(one, spec, child, asked),
                    result,
                    spec.session_id,
                    parent_events,
                ),
                name=f"agent6-child-{spec.session_id}",
                daemon=True,
            ).start()
            workspace = workspaces.get(spec.number)
            started.append(
                child_records.StartedChild(
                    spec.session_id, result, asked.set, workspace.notice if workspace else ""
                )
            )
        return started

    return GroupSpawner(dispatch, hard_stop)


def build_parent_spawner(
    cfg: config_model.Config,
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
    watch: PollChild | None = None,
    parent_events: event_log.EventSink | None = None,
) -> GroupSpawner | None:
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
        parent_events: The parent execution's sink for child activity.
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
        parent_events=parent_events,
    )
