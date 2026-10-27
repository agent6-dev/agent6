# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Land a run's branch, for `sessions merge` and `git.auto_merge`.

Landing is ref plumbing (`git_ops.plumb_merge`): no checkout and no clean-tree requirement,
so the worktree carrying the run's own work is never an obstacle. Both callers share the
method dispatch and the manifest record.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import pathlib
import shlex
from collections.abc import Callable
from typing import Literal

from agent6 import budget as agent6_budget
from agent6 import errors, event_log
from agent6.app import _merge_message, _setup, stamps
from agent6.config import Config
from agent6.sessions import layout as sessions_layout
from agent6.sessions import lock, process
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs import git_ops, git_read
from agent6.viewmodel import scan_session_log


@dataclasses.dataclass(frozen=True, slots=True)
class MergeOutcome:
    """Record what `execute_merge` did.

    A noop is a branch whose content the target already holds: the target stays where it was
    and the run is merged all the same. A noop over the tip already recorded leaves the record
    of the merge that did happen alone.

    Attributes:
        status: merged, noop, conflict or error.
        merged_sha: The target's tip after the merge; on a noop, its own tip.
        conflicts: The conflicting paths.
        error: Why the merge did not happen.
        stamp_error: Why the manifest stamp did not land; the merge happened either way.
        recorded: This call stamped a noop as merged, with NO_MERGE_COMMIT for the sha.
        left_behind: Paths the checkout kept its own version of (`left_behind_line`).
        target_before: The target's tip before the merge, the base of those versions.
    """

    status: Literal["merged", "noop", "conflict", "error"]
    merged_sha: str = ""
    conflicts: tuple[str, ...] = ()
    error: str = ""
    stamp_error: str = ""
    recorded: bool = False
    left_behind: tuple[str, ...] = ()
    target_before: str = ""


def record_merge_in_manifest(
    layout: sessions_layout.SessionLayout,
    *,
    merged_into: str,
    merged_sha: str,
    merged_tip: str = "",
    into_tip: str = "",
    via: str = "",
    method: Literal["ff_only", "merge", "squash"] | None = None,
    target_before: str | None = None,
) -> str:
    """Stamp the merge into the run manifest, so `prune` can tell a merged branch.

    A missing or corrupt manifest must not fail a merge that already happened.

    Args:
        layout: The run's directory layout.
        merged_into: The target branch.
        merged_sha: The merge commit, or NO_MERGE_COMMIT.
        merged_tip: The run-branch tip merged; `prune --delete-squashed` deletes only a branch
            still pointing there.
        into_tip: The target's tip a merge that added nothing saw.
        via: The run whose merge landed this one, "" for its own.
        method: The landing method, None when unrecorded.
        target_before: The target tip before landing, None when unrecorded.

    Returns:
        "" when the stamp landed, else why it did not.
    """
    try:
        m = sessions_manifest.read_manifest(layout.session_dir)
    except sessions_manifest.ManifestError as exc:
        return str(exc)
    stamped = m.model_copy(
        update={
            "merged": sessions_manifest.MergeStamp(
                into=merged_into,
                sha=merged_sha,
                tip=merged_tip,
                into_tip=into_tip,
                via=via,
                method=method,
                target_before=target_before,
                ts=datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds"),
            )
        }
    )
    # ManifestError: a manifest newer than this binary is left alone rather than downgraded.
    try:
        stamps.write_manifest(layout.manifest_path, stamped)
    except (OSError, sessions_manifest.ManifestError) as exc:
        return str(exc)
    return ""


def dispatch_merge(
    cwd: pathlib.Path,
    method: str,
    target: str,
    run_branch: str,
    base_sha: str,
    manifest: sessions_manifest.SessionManifest,
    message: str | None,
    cfg: Config,
    identity: git_read.CommitIdentity,
    *,
    transcript_dir: pathlib.Path | None = None,
    budget: agent6_budget.BudgetTracker | None = None,
    events: event_log.EventSink | None = None,
    warn: Callable[[str], None] = lambda _m: None,
    merge_base: str | None = None,
    finish_summaries: tuple[str, ...] = (),
) -> git_ops.MergeResult:
    """Run the method on the target through `plumb_merge`.

    A squash builds its message per `[git.commit.squash].message`; an operator message
    overrides any style.

    Args:
        cwd: The repository.
        method: ff_only, merge or squash.
        target: The branch to land on.
        run_branch: The run's branch or chain ref.
        base_sha: The commit the run started from.
        manifest: The run's manifest.
        message: The operator's message, else the style's.
        cfg: The resolved config.
        identity: The committer.
        transcript_dir: Where a model-written message's call is transcribed.
        budget: The run's tracker, when the merge runs inside one.
        events: The run's event sink, when the merge runs inside one.
        warn: Where a degraded message is reported.
        merge_base: The base to merge from, when git's own does not serve.
        finish_summaries: Completed executions' summaries; the last supplies the body of
            an `agent6` or `conventional` message, the tuple supplies model context.

    Returns:
        The plumbing's result.
    """
    if method == "squash" and message is None:
        # The message names what this merge lands: the commits after the landed base.
        message = _merge_message.squash_message(
            cwd,
            cfg,
            manifest,
            base_sha=merge_base or base_sha,
            run_branch=run_branch,
            finish_summaries=finish_summaries,
            transcript_dir=transcript_dir,
            budget=budget,
            events=events,
            warn=warn,
        )
    if message is None and method == "merge":
        message = f"Merge {run_branch}"
    return git_ops.plumb_merge(
        cwd,
        target,
        run_branch,
        method=method,
        message=message,
        identity=identity,
        merge_base=merge_base,
    )


def landed_base(
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
    target: str,
    tip: str,
) -> str | None:
    """Return the merge base for a run the target already holds as a squash, else None.

    Git cannot relate a squash to the chain it came from, so its own base reads every earlier
    commit as new work. The base is the merged tip when the chain continues past it, else the
    point a fork left its ancestor's chain.

    Args:
        cwd: The repository.
        layout: The run's directory layout.
        manifest: The run's manifest; its stamp, or an ancestor's, names the merged tip.
        target: The branch to land on.
        tip: The run's chain tip.

    Returns:
        The base, or None where git's own serves.
    """
    node, fork_point = manifest, ""
    for _ in range(64):  # a lineage deeper than this is not a fork chain
        stamp = node.merged
        if stamp is not None and stamp.into == target and stamp.tip:
            if git_read.is_ancestor(cwd, stamp.tip, target):
                break  # a real merge: git relates the two histories itself
            if git_read.is_ancestor(cwd, stamp.tip, tip):
                return stamp.tip
            if fork_point and git_read.is_ancestor(cwd, fork_point, stamp.tip):
                return fork_point
            break
        fork_point = node.forked_from_sha
        if not node.parent_session_id:
            break
        try:
            node = sessions_manifest.read_manifest(
                layout.session_dir.parent / node.parent_session_id
            )
        except (sessions_manifest.ManifestError, OSError):
            break
    return None


NO_BASE_SHA = "the manifest records no base_sha; nothing to merge from"


def execute_merge(
    cwd: pathlib.Path,
    *,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
    run_branch: str,
    target: str,
    base_sha: str,
    method: str,
    message: str | None,
    cfg: Config,
    identity: git_read.CommitIdentity,
    budget: agent6_budget.BudgetTracker | None = None,
    events: event_log.EventSink | None = None,
    warn: Callable[[str], None] = lambda _m: None,
    worker_lock_fd: int | None = None,
) -> MergeOutcome:
    """Land the run's branch on the target and record the merge.

    Ref plumbing only: the checkout is never switched and the worktree is never required
    clean. The existing writer lock serializes review dispositions and this mutation.

    Args:
        cwd: The repository.
        layout: The run's directory layout.
        manifest: The run's manifest.
        run_branch: The run's branch or chain ref.
        target: The branch to land on.
        base_sha: The commit the run started from.
        method: ff_only, merge or squash.
        message: The operator's message, else the style's.
        cfg: The resolved config.
        identity: The committer.
        budget: The run's tracker, when the merge runs inside one.
        events: The run's event sink, when the merge runs inside one.
        warn: Where a degraded squash message is reported.
        worker_lock_fd: The executing worker's held lock, else this call takes the lock.

    Returns:
        What the merge did.
    """
    _setup.apply_git_ops_policy(cfg)
    try:
        with contextlib.ExitStack() as cleanup:
            if worker_lock_fd is None:
                fd = lock.acquire_single_writer(layout.session_dir)
                cleanup.callback(lock.release_single_writer, fd)
                if fd is None or process.worker_is_alive(layout.session_dir):
                    return MergeOutcome(
                        "error", error="session writer is busy; wait for it to finish"
                    )
            # A mutable ref must never reach the message builder, merge plumbing or stamp.
            tip = git_read.chain_tip(cwd, run_branch)
            guard = scan_session_log(layout.logs_path).review.merge_guard
            refusal = (
                f"cannot read review evidence: {guard.read_failure}"
                if guard.read_failure
                else "open block findings: "
                + ", ".join(guard.open_block_ids)
                + "; resolve each with a reason before merging"
                if guard.open_block_ids
                else NO_BASE_SHA
                if not base_sha
                else f"target branch {target!r} does not exist"
                if not git_read.branch_exists(cwd, target)
                else f"source ref {run_branch!r} does not exist"
                if tip is None
                else ""
            )
            if refusal:
                return MergeOutcome("error", error=refusal)
            assert tip is not None
            if (
                method == "ff_only"
                and not git_read.is_ancestor(cwd, target, tip)
                and not git_read.is_ancestor(cwd, tip, target)
            ):
                return MergeOutcome(
                    "error",
                    error=(
                        f"{target!r} has moved since the run started, so a"
                        " fast-forward is impossible; merge with --method merge or"
                        " squash instead"
                    ),
                )
            target_tip_before = git_read.branch_tip_sha(cwd, target) or ""
            merge_base = landed_base(cwd, layout, manifest, target, tip)
            result = dispatch_merge(
                cwd,
                method,
                target,
                tip,
                base_sha,
                manifest,
                f"Merge {run_branch}" if method == "merge" and message is None else message,
                cfg,
                identity,
                transcript_dir=layout.session_dir / "transcripts",
                budget=budget,
                events=events,
                warn=warn,
                merge_base=merge_base,
                finish_summaries=_merge_message.finish_summaries(layout),
            )
            if result.conflicted:
                return MergeOutcome("conflict", conflicts=result.conflicts)
            noop = bool(result.merged_sha) and result.merged_sha == target_tip_before
            already_recorded = noop and manifest.merged is not None and manifest.merged.tip == tip
            stamp_error = ""
            if not already_recorded:
                stamp_error = record_merge_in_manifest(
                    layout,
                    merged_into=target,
                    merged_sha=sessions_manifest.NO_MERGE_COMMIT if noop else result.merged_sha,
                    merged_tip=tip,
                    into_tip=target_tip_before if noop else "",
                    method=method if method in ("ff_only", "merge", "squash") else None,
                    target_before=target_tip_before,
                )
            if not noop:
                _land_included_run(cwd, layout, manifest, target, result.merged_sha)
            return MergeOutcome(
                "noop" if noop else "merged",
                merged_sha=result.merged_sha,
                stamp_error=stamp_error,
                recorded=noop and not already_recorded and not stamp_error,
                left_behind=result.left_behind,
                target_before=target_tip_before,
            )
    except (errors.GitError, OSError) as exc:
        return MergeOutcome("error", error=f"merge failed: {exc}")


def _land_included_run(
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
    target: str,
    merged_sha: str,
) -> None:
    """Stamp the run this one took in under `dirty_tree = "include"` as merged with it.

    Only a run still at the tip this one started on: later commits of its own did not land.

    Args:
        cwd: The repository.
        layout: The landed run's directory layout.
        manifest: The landed run's manifest.
        target: The branch it landed on.
        merged_sha: Its merge commit.
    """
    included = manifest.includes
    if included is None:
        return
    theirs = dataclasses.replace(layout, session_id=included.session_id)
    try:
        if sessions_manifest.read_manifest(theirs.session_dir).merged is not None:
            return
    except sessions_manifest.ManifestError:
        return
    if git_read.chain_tip(cwd, git_read.chain_ref_for(included.session_id)) != included.tip:
        return
    record_merge_in_manifest(
        theirs,
        merged_into=target,
        merged_sha=merged_sha,
        merged_tip=included.tip,
        via=manifest.session_id,
    )


def merge_command(session_id: str, base_branch: str) -> str:
    """Return the command that lands a run, as a hint prints it.

    Args:
        session_id: The run to land.
        base_branch: The run's base branch; `git_read.DETACHED_HEAD` for a detached HEAD.

    Returns:
        The command; on a detached HEAD it ends in `--into BRANCH` for the operator to fill in.
    """
    into = " --into BRANCH" if base_branch == git_read.DETACHED_HEAD else ""
    return f"agent6 sessions merge {session_id}{into}"


def left_behind_line(target: str, outcome: MergeOutcome) -> str:
    """Return the line for merged files the checkout keeps its own version of, or "".

    The plumbing brings the checkout forward only where a file still matches what the branch
    held, so an operator's edit is never overwritten; the tree they then test holds the older
    content. Committed as it is, such a file would undo the merged change, so the line carries
    the two commands that merge the edit with what landed.

    Args:
        target: The branch the run landed on.
        outcome: The landed merge.

    Returns:
        The line, "" when the checkout kept no file of its own.
    """
    if not outcome.left_behind:
        return ""
    named = ", ".join(outcome.left_behind[:4]) + (", ..." if len(outcome.left_behind) > 4 else "")
    kept = f"your checkout keeps its own {named}; {target} holds what was merged"
    if not outcome.target_before:
        return kept
    paths = " ".join(shlex.quote(p) for p in outcome.left_behind)
    return (
        f"{kept}\n"
        "  to merge your edits with what landed (a conflict leaves its markers in the file):\n"
        f"    git diff --binary {outcome.target_before[:12]} -- {paths} > your-edits.patch\n"
        f"    git checkout HEAD -- {paths} && git apply --3way your-edits.patch"
    )


def noop_merge_line(run_branch: str, target: str, outcome: MergeOutcome) -> str:
    """Return the line for a merge that added nothing; a stamp error is the caller's note."""
    if outcome.recorded or outcome.stamp_error:
        recorded = f"; recorded as merged into {target}" if outcome.recorded else ""
        return f"nothing to add from {run_branch}: {target} already has its content{recorded}"
    return f"nothing left to merge from {run_branch} into {target}"
