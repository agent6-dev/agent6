# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Record an escaping execution and settle the commits it leaves."""

from __future__ import annotations

import contextlib
import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from agent6 import budget as agent6_budget
from agent6 import errors
from agent6.app import (
    finalize,
)
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.app.parallel import launch as child_launch
from agent6.config import Config
from agent6.harness import loop, loop_snapshot, run_chain
from agent6.sandbox import detect, jail
from agent6.sessions import event_log, ipc, journal, process
from agent6.sessions import layout as sessions_layout
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read

if TYPE_CHECKING:
    from agent6.app import execution


def detach_to_background(
    *,
    frontend: app_frontend.SessionFrontend,
    cfg: Config,
    layout: sessions_layout.SessionLayout,
    cwd: pathlib.Path,
    flags: Sequence[str],
    reporter: app_reporter.Reporter,
) -> None:
    """Hand a detached execution to a background `resume` under this invocation's flags.

    Called once the caller has released the run's locks. Asks how approvals are answered while
    nothing watches, spawns, then prints the reattach line only for a spawn that happened.

    Args:
        frontend: The injected front-end.
        cfg: The resolved config.
        layout: The session's directory layout.
        cwd: The workspace.
        flags: This invocation's overrides as CLI options.
        reporter: Where the reattach line goes.
    """
    if cfg.sandbox.run_commands == "ask" and not ipc.session_allow_set(
        layout.session_dir, ipc.COMMAND_SCOPE
    ):
        frontend.prompt_detach_away_mode(layout.session_dir, app_frontend.approval_scopes(cfg))
    err = frontend.spawn_detached_resume(cwd, layout.session_id, flags)
    if err:
        # The handoff failed, so the pid this process kept through the spawn goes now.
        process.clear_worker_pid(layout.session_dir)
        reporter.note(err)
        return
    reporter.out(f"\n[agent6] detached: {layout.session_id} continues in the background.")
    reporter.out(f"          reattach:  agent6 attach {layout.session_id}")


def escape_reason(exc: BaseException) -> loop_snapshot.SessionEndReason:
    """Return the end reason an escaping exception stands for.

    Args:
        exc: The escaping exception.

    Returns:
        "interrupted" for Ctrl-C or a closed output pipe (`agent6 run ... | head`), the
        operator's side going away; "sandbox_unavailable" for a jail, container or isolation
        level the host refused; else "crashed".
    """
    if isinstance(exc, (KeyboardInterrupt, BrokenPipeError)):
        return "interrupted"
    if isinstance(exc, (jail.JailUnavailableError, detect.IsolationUnavailableError)):
        return "sandbox_unavailable"
    return "crashed"


def journal_escape(
    events: event_log.EventSink,
    exc: BaseException,
    *,
    execution_id: str,
    turns: int,
    tree: str = "",
) -> loop_snapshot.SessionEndReason:
    """Journal the end an escape leaves.

    Without one every surface reads the dead run as running until the silence window expires;
    a dead journal must not mask the exit code.

    Args:
        events: The run's event sink.
        exc: The escaping exception.
        execution_id: The execution that stopped.
        turns: The turns reached.
        tree: The tree the run leaves, so the next resume tells its files from the operator's.

    Returns:
        The reason journaled.
    """
    reason = escape_reason(exc)
    left: dict[str, Any] = {"tree": tree} if tree else {}
    with contextlib.suppress(event_log.EventWriteError):
        events.emit(
            journal.SessionStop(
                execution_id=execution_id,
                reason=reason,
                turns=turns,
                verification=None,
                **left,
            )
        )
    return reason


def tree_left_before_the_loop(
    cwd: pathlib.Path, inputs: execution.ExecutionInputs, hidden: frozenset[str]
) -> str:
    """Return the tree a run leaves when it escapes before its harness exists; "" outside run mode.

    Seeded on the chain tip like a chain commit, as the harness's own read is; the escape's
    end then tells the next resume the run's files from the operator's.

    Args:
        cwd: The workspace.
        inputs: The execution's inputs.
        hidden: The run's hidden paths inside the repository, left out like its untracked files.

    Returns:
        The tree's sha; "" outside run mode or when the read fails or is interrupted.
    """
    if inputs.mode != "run":
        return ""
    try:
        chain_tip = git_read.chain_tip(cwd, git_read.chain_ref_for(inputs.session_id))
        return git_read.worktree_tree(
            cwd, chain_tip or inputs.base_sha or None, inputs.untracked_at_start | hidden
        )
    except (errors.GitError, OSError, KeyboardInterrupt):
        return ""


def tree_left(wf: loop.Harness, mode: str) -> str:
    """Return the tree a run leaves, for the end an escape journals; "" outside run mode.

    The read runs git inside the escape handler, where another Ctrl-C can land: the end is
    then written without the tree rather than not at all.

    Args:
        wf: The run's harness.
        mode: The session mode.

    Returns:
        The tree's sha; "" outside run mode or when the read fails or is interrupted.
    """
    if mode != "run":
        return ""
    try:
        return wf.chain.tree_sha()
    except KeyboardInterrupt:
        return ""


def finish_chain(
    cfg: Config,
    layout: sessions_layout.SessionLayout,
    inputs: execution.ExecutionInputs,
    *,
    wf: loop.Harness | None,
    result: loop_snapshot.SessionResult | None,
    interrupted: bool,
    mirror: run_chain.MirrorTarget | None,
    budget: agent6_budget.BudgetTracker,
    events: event_log.EventSink,
    reporter: app_reporter.Reporter,
    cwd: pathlib.Path,
) -> None:
    """Copy a clone's last commits into its repository, then merge a run that finished.

    Runs once every close succeeded: no merge runs on an execution whose teardown failed.

    Args:
        cfg: The config the tools resolved.
        layout: The session's directory layout.
        inputs: What the lifecycle hands the body.
        wf: The harness, or None when the execution escaped before it was built.
        result: How the run ended, or None when it escaped.
        interrupted: Whether the operator interrupted the run.
        mirror: The repository a session in a clone copies its commits into, or None.
        budget: The run's budget, which the merge's message spends from.
        events: The run's event sink.
        reporter: The output channels.
        cwd: The workspace.
    """
    mode = inputs.mode
    if mirror is not None and mode == "run":
        # Once more at the end: a merge taken in or an undo moves the chain with no commit;
        # under `git.control = "model"` the model's own commits are the record.
        try:
            if cfg.git.control == "model":
                git_ops.mirror_session(
                    mirror.repository,
                    cwd,
                    git_read.chain_ref_for(inputs.session_id),
                    None,
                    sender=mirror.sender,
                    head=True,
                )
            elif wf is not None:
                wf.chain.mirror()
        except errors.GitError as exc:
            reporter.warn(
                f"the run's commits could not be copied into {inputs.mirror_into}: {exc};"
                f" they are in its worktree {cwd}"
            )
    # A child's base branch belongs to its throwaway clone. Its parent owns the merge.
    auto_merge = cfg.git.auto_merge and mode == "run" and not child_launch.is_child()
    if interrupted or result is None or not auto_merge:
        return
    if finalize.auto_merge_eligible(result):
        finalize.finalize_auto_merge(
            # The merge target is the repository's branch, not the clone's.
            inputs.mirror_into or cwd,
            layout=layout,
            cfg=cfg,
            reporter=reporter,
            budget=budget,
            events=events,
            worker_lock_fd=inputs.worker_lock_fd,
        )
        return
    if result.reason not in finalize.AUTO_MERGE_ENDS:
        reason = f"the run ended on {result.reason}, not finish_session or children_ranked"
    elif result.child_verification:
        child_id = result.child_verification.session_id
        reason = f"only child {child_id} verified the final tree"
    else:
        reason = f"the gate on the final tree is {result.verification}"
    reporter.note(f"auto_merge skipped: {reason}")
