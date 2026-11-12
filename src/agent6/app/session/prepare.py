# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The steps a run and a resume share before their execution body.

Each lifecycle calls them in its own order; a refusal is reported here and raised as
`preflight.SessionRefusedError`, which the lifecycle turns into its exit code.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable, Collection

from agent6 import kinds
from agent6.app import frontend as app_frontend
from agent6.app import preflight, resume_questions, stamps
from agent6.app import reporter as app_reporter
from agent6.app.session import runtime as session_runtime
from agent6.config import Config
from agent6.harness import repo_context
from agent6.providers import TranscriptSink
from agent6.sessions import event_log, lock
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import operator_prompts


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
