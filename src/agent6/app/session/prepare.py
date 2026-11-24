# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The steps a run and a resume share before their execution body, and their one teardown.

Each lifecycle calls them in its own order; a refusal is reported here and raised as
`preflight.SessionRefusedError`, which the lifecycle turns into its exit code. `hold` releases
what the stages took, on every exit.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
from collections.abc import Callable, Collection, Generator

from agent6 import kinds, paths
from agent6.app import finalize, fork_worktrees, preflight, resume_questions, stamps
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.app.parallel import launch as child_launch
from agent6.app.session import runtime as session_runtime
from agent6.config import Config
from agent6.harness.prompt import repo_context
from agent6.providers import TranscriptSink
from agent6.sessions import event_log, lock, process
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import operator_prompts


@dataclasses.dataclass(frozen=True, slots=True)
class AddedWorktree:
    """A worktree a run's start added for itself.

    Attributes:
        repo: The repository it was added to.
        path: The worktree.
        base_sha: The commit it was added at.
    """

    repo: pathlib.Path
    path: pathlib.Path
    base_sha: str


@dataclasses.dataclass(frozen=True, slots=True)
class AutoStash:
    """The operator's edits an execution stashed at its start, and what ending the stash reads.

    Attributes:
        cwd: The checkout.
        base_branch: The base branch, used for a run without a worktree.
        run_branch: The run's branch, or None.
        auto_pop: Whether the end applies the stash back.
        in_worktree: Whether the checkout is the session's own worktree.
    """

    cwd: pathlib.Path
    base_branch: str
    run_branch: str | None
    auto_pop: bool
    in_worktree: bool


@dataclasses.dataclass(slots=True)
class Held:
    """What a lifecycle holds: each stage records what it took, and `hold` releases it.

    Attributes:
        worker_lock_fd: The session's one writer lock.
        repo_lock_fd: The checkout's writer lock.
        added_worktree: The worktree this start added, dropped unless the manifest records it.
        in_session: Whether this worker opened the session: its end then closes the console
            view, clears the worker pid and ends the stash. Before that a refusal releases
            only the locks and the worktree, so it never clears another worker's pid.
        handed_off: Whether another lifecycle started the session under the same id; the pid
            is then that lifecycle's.
        detached: Whether the execution continues in the background, which keeps the pid for
            it and leaves the stash stashed.
        stash: The stash this execution pushed.
        untracked_at_start: The operator's untracked files, left out of the stash's status check.
    """

    worker_lock_fd: int | None = None
    repo_lock_fd: int | None = None
    added_worktree: AddedWorktree | None = None
    in_session: bool = False
    handed_off: bool = False
    detached: bool = False
    stash: AutoStash | None = None
    untracked_at_start: frozenset[str] = frozenset()


def _drop_unrecorded_worktree(
    added: AddedWorktree,
    layout: sessions_layout.SessionLayout,
    *,
    reporter: app_reporter.Reporter,
) -> None:
    """Remove a worktree a start added, unless the session's manifest records it."""
    # The manifest names a worktree as agent6's, so a start refused before it leaves none.
    with contextlib.suppress(sessions_manifest.ManifestError):
        if sessions_manifest.read_manifest(layout.session_dir).worktree == added.path:
            return
    gone, why = fork_worktrees.remove_fork_worktree(added.repo, added.path, (added.base_sha,))
    if not gone:
        reporter.note(
            f"the worktree {added.path} stays: it {why}. No session names it; remove"
            f" it with `git worktree remove {added.path}`"
        )


def _end_session(
    held: Held,
    layout: sessions_layout.SessionLayout,
    *,
    frontend: app_frontend.SessionFrontend,
    reporter: app_reporter.Reporter,
) -> None:
    """Close the console view, then clear the worker pid and end the stash."""
    try:
        frontend.close_console_view()  # stops the heartbeat, clears the spinner
    finally:
        # Settled even when the close raises.
        if not held.detached and not held.handed_off:
            # A detach keeps the pid until the background `resume` claims it.
            process.clear_worker_pid(layout.session_dir)
        if held.stash is not None:
            finalize.end_auto_stash(
                held.stash.cwd,
                detached=held.detached,
                base_branch=held.stash.base_branch,
                run_branch=held.stash.run_branch,
                auto_pop=held.stash.auto_pop,
                session_id=layout.session_id,
                in_worktree=held.stash.in_worktree,
                exclude=held.untracked_at_start,
                reporter=reporter,
            )


@contextlib.contextmanager
def hold(
    layout: sessions_layout.SessionLayout,
    *,
    frontend: app_frontend.SessionFrontend,
    reporter: app_reporter.Reporter,
) -> Generator[Held]:
    """Release what a lifecycle holds, on every exit path.

    The one owner of worker.pid, both locks, the stash and an unrecorded worktree. The order
    is fixed and is not the reverse of the order taken: the session's end, both locks, the
    worktree. The locks go even when the session's end raises, since an in-process front-end
    outlives the run.

    Args:
        layout: The session's directory layout.
        frontend: The surface whose console view the session's end closes.
        reporter: Receives the stash's notes and a worktree that stays.

    Yields:
        The record the lifecycle's stages fill.
    """
    held = Held()
    try:
        yield held
    finally:
        try:
            if held.in_session:
                _end_session(held, layout, frontend=frontend, reporter=reporter)
        finally:
            lock.release_single_writer(held.repo_lock_fd)
            lock.release_single_writer(held.worker_lock_fd)
            if held.added_worktree is not None:
                _drop_unrecorded_worktree(held.added_worktree, layout, reporter=reporter)


def record_dirs(
    repo: pathlib.Path,
    launch: child_launch.ChildLaunch | None,
    *,
    reporter: app_reporter.Reporter,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Return the state directory a session's record goes under and the one it executes with.

    A child's record goes under the state of the repository that dispatched it, so its
    listings read the child from its first event; its memory and rulings stay with its clone.

    Args:
        repo: The repository the session was started in.
        launch: The child launch that binds the record; None outside one.
        reporter: Receives the refusal.

    Returns:
        The record's state directory, then the execution's.

    Raises:
        preflight.SessionRefusedError: The launch is unbound or its checkout changed.
    """
    if launch is None:
        state = paths.state_dir(repo)
        return state, state
    try:
        state = launch.checked_clone(repo).state
    except child_launch.LaunchError as exc:
        reporter.refuse(str(exc))
        raise preflight.SessionRefusedError(2) from exc
    return launch.record_state, state


def hold_workspace_and_select_isolation(
    cfg: Config,
    cwd: pathlib.Path,
    *,
    frontend: app_frontend.SessionFrontend,
    reporter: app_reporter.Reporter,
    worktree_git_dir: pathlib.Path | None,
) -> kinds.IsolationLevel:
    """Take the workspace lease, then resolve the isolation its commands run under.

    The lease comes first, so no pack can be written under a live run.

    Args:
        cfg: The run's config.
        cwd: The checkout the commands run in.
        frontend: The surface that confirms an unconfined autorun.
        reporter: Receives the refusal.
        worktree_git_dir: The repository git dir of a linked worktree, or None.

    Returns:
        The resolved isolation level.

    Raises:
        preflight.SessionRefusedError: The workspace is being packed, or the level was refused.
    """
    if not lock.hold_workspace(cwd):
        reporter.refuse(lock.WORKSPACE_PACKING.format(root=cwd))
        raise preflight.SessionRefusedError(2)
    return session_runtime.select_isolation(
        cfg,
        cwd=cwd,
        confirm_unconfined=frontend.confirm_unconfined_autorun,
        reporter=reporter,
        worktree_git_dir=worktree_git_dir,
    )


def take_single_writer(
    layout: sessions_layout.SessionLayout, *, reporter: app_reporter.Reporter
) -> int:
    """Create the session directory and take its one writer lock.

    Args:
        layout: The session's directory layout.
        reporter: Receives the refusal.

    Returns:
        The lock's file descriptor, released by the lifecycle's teardown.

    Raises:
        preflight.SessionRefusedError: Another worker holds the session.
    """
    layout.ensure()
    fd = lock.acquire_single_writer(layout.session_dir)
    if fd is None:
        reporter.err(lock.SINGLE_WRITER_BUSY.format(rid=layout.session_id))
        raise preflight.SessionRefusedError(2)
    return fd


def headless_check(
    cfg: Config,
    frontend: app_frontend.SessionFrontend,
    *,
    tui: bool,
    interactive: bool,
    mode: kinds.LoopMode,
    away: str,
    parking_away: str | None = None,
    reporter: app_reporter.Reporter,
) -> bool:
    """Refuse a session nobody could answer, and note one that will park instead.

    Args:
        cfg: The run's config.
        frontend: The surface, which says whether it can ask.
        tui: Whether `--tui` was asked for.
        interactive: Whether an operator is at the terminal.
        mode: The session mode.
        away: The away answer the refusal reads: the env, or on a resume the recorded one.
        parking_away: The away answer the parking note reads; None reads `away`.
        reporter: Receives the refusal or the note.

    Returns:
        Whether the TUI owns the terminal.

    Raises:
        preflight.SessionRefusedError: The session would wait on an answer nobody can give.
    """
    tui_enabled = frontend.should_spawn_tui(tui, interactive, mode)
    refusal = preflight.headless_approval_refusal(
        cfg,
        tui_enabled=tui_enabled,
        away=away,
        can_ask=frontend.capabilities.can_ask,
        clamped=kinds.session_kind(mode).clamps_commands,
    )
    if refusal is not None:
        reporter.refuse(refusal)
        raise preflight.SessionRefusedError(2)
    parking = preflight.headless_parking_note(
        cfg,
        tui_enabled=tui_enabled,
        away=away if parking_away is None else parking_away,
        can_ask=frontend.capabilities.can_ask,
    )
    if parking is not None:
        reporter.note(parking)
    return tui_enabled


@dataclasses.dataclass(frozen=True, slots=True)
class Channels:
    """The execution's journal, its transcripts and its one gate to the operator.

    Attributes:
        events: The run's event sink.
        transcript_sink: Where every provider call is transcribed.
        prompts: The operator gate, so a question asked before the loop and the dispatcher's
            approvals share one journal and one id sequence.
    """

    events: event_log.EventSink
    transcript_sink: TranscriptSink
    prompts: operator_prompts.OperatorPrompts


def operator_channels(
    layout: sessions_layout.SessionLayout,
    frontend: app_frontend.SessionFrontend,
    *,
    carried: resume_questions.CarriedQuestions | None = None,
    delivered: Collection[str] = (),
) -> Channels:
    """Open the journal and the transcripts, and the operator gate over the journal.

    Args:
        layout: The session's directory layout.
        frontend: The surface that answers approvals and questions.
        carried: A resume's questions from the journal; None for a fresh execution.
        delivered: The late answers the snapshot says the model already read.

    Returns:
        The three channels.
    """
    events = event_log.EventSink(layout.logs_path)
    prompts = operator_prompts.OperatorPrompts(
        approver=frontend.build_approver(layout.session_dir),
        questioner=frontend.build_questioner(layout.session_dir),
        journal=events.emit,
        session_dir=layout.session_dir,
        first_question=carried.next_number if carried is not None else 1,
        pending=carried.pending if carried is not None else (),
        undelivered=tuple(
            late
            for late in (carried.late_answers if carried is not None else ())
            if late.id not in delivered
        ),
    )
    return Channels(
        events=events,
        transcript_sink=TranscriptSink(layout.transcripts_dir, objects_dir=layout.objects_dir),
        prompts=prompts,
    )


def workspace_notices(cwd: pathlib.Path, cfg: Config, *, reporter: app_reporter.Reporter) -> None:
    """Warn about an install inside the workspace and print the AGENTS.md notices.

    Args:
        cwd: The checkout.
        cfg: The run's config, for the hidden paths.
        reporter: Receives the lines.
    """
    session_runtime.warn_install_inside_workspace(cwd, reporter=reporter)
    for line in repo_context.agents_md_notices(cwd, cfg.sandbox.hide_paths):
        reporter.note(line)


def undo_forker(
    config_paths: tuple[pathlib.Path, ...],
    session_id: str,
    *,
    cwd: pathlib.Path,
    reporter: app_reporter.Reporter,
) -> Callable[[], tuple[str, str] | None]:
    """Return the `/undo` step: fork back before the last message and rewind the checkout.

    Args:
        config_paths: The `--config` files the fork is stamped under.
        session_id: The session undone.
        cwd: The repository.
        reporter: Receives the fork's notes.

    Returns:
        The step; it returns the fork's id and the text taken back, or None.
    """

    def fork_back() -> tuple[str, str] | None:
        # Lazy: app.undo imports app.fork, which imports app.resume, which imports app.run.
        from agent6.app import undo  # noqa: PLC0415

        return undo.undo_fork(config_paths, session_id, cwd=cwd, reporter=reporter)

    return fork_back


def settle_gate(
    cfg: Config,
    layout: sessions_layout.SessionLayout,
    *,
    source: sessions_manifest.GateSource,
    events: event_log.EventSink,
    reporter: app_reporter.Reporter,
) -> tuple[Config, bool]:
    """Drop a gate commands cannot run, then pin the gate this execution runs with.

    Args:
        cfg: The config with the execution's gate resolved.
        layout: The session's directory layout.
        source: Where the gate came from: configured, inferred or adopted.
        events: The run's event sink, watched for a gate adopted mid-run.
        reporter: Where a failed pin and the drop's reason are reported.

    Returns:
        The config, and whether a set gate was withheld because commands are.
    """
    gate_before = cfg.harness.verify.command
    cfg = preflight.drop_gate_if_unrunnable(cfg, session_dir=layout.session_dir, reporter=reporter)
    withheld = bool(gate_before) and not cfg.harness.verify.command
    stamps.pin_gate(
        layout.session_dir,
        sessions_manifest.GateStamp(
            command=gate_before,
            source=source if gate_before else "",
            availability="withheld" if withheld else "ready",
        ),
        events=events,
        reporter=reporter,
    )
    return cfg, withheld
