# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Find and reopen the session record and check its snapshot against the checkout."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib

from agent6 import directive, errors, kinds
from agent6.app import (
    fork_worktrees,
    preflight,
    resume_chain,
    resume_questions,
)
from agent6.app import reporter as app_reporter
from agent6.app.parallel import launch as child_launch
from agent6.app.parallel import resume as child_resume
from agent6.app.session import prepare as session_prepare
from agent6.app.session import setup as session_setup
from agent6.config import RoleName
from agent6.harness import loop_snapshot
from agent6.sessions import child_dispatch, id, ipc
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel import listing
from agent6.viewmodel.session import log as session_log


def resumable_bucket_dirs(state_dir: pathlib.Path) -> list[pathlib.Path]:
    """List the bucket dirs holding sessions `agent6 resume` can pick up.

    Args:
        state_dir: The repo's state directory.

    Returns:
        One dir per resumable session kind.
    """
    return [
        sessions_layout.bucket_dir(state_dir, kinds.session_bucket(kind.name))
        for kind in kinds.SESSION_KINDS.values()
        if kind.resumable
    ]


def covering_stamp(
    repo: pathlib.Path, manifest: sessions_manifest.SessionManifest
) -> sessions_manifest.MergeStamp | None:
    """Return the merge stamp when it covers every commit the run made.

    A resumed run commits past its stamp, and no merge covers those commits.

    Args:
        repo: The repository.
        manifest: The session's manifest.

    Returns:
        The stamp, or None when it covers less than the chain.
    """
    stamp = manifest.merged
    if stamp is None or not stamp.into:
        return None
    holds = git_read.merge_stamp_holds(
        repo, manifest.session_id, manifest.run_branch or "", stamp.tip
    )
    return stamp if holds else None


def commits_note(repo: pathlib.Path, manifest: sessions_manifest.SessionManifest) -> str:
    """Say where a session's commits are, for a refusal that points at them.

    Args:
        repo: The repository.
        manifest: The session's manifest.

    Returns:
        The run branch, the merge that included them, the chain ref, or "it recorded
        no commits".
    """
    if manifest.run_branch and git_read.branch_exists(repo, manifest.run_branch):
        return f"its commits are on {manifest.run_branch}"
    stamp = covering_stamp(repo, manifest)
    if stamp is not None:
        return f"its commits are {stamp.description()}"
    chain = git_read.chain_ref_for(manifest.session_id)
    if git_read.chain_tip(repo, chain) is not None:
        return f"its commits are on {chain}"
    return "it recorded no commits"


def snapshot_head_mismatch(
    snapshot_path: pathlib.Path, repo_root: pathlib.Path, *, chain_ref: str
) -> tuple[str, str] | None:
    """Detect a chain tip that diverged from the run's last snapshot.

    Forward movement on the same line is the run's own per-turn commits and
    resumes cleanly; only a tip that is not a descendant of the snapshot head
    counts. Committed history only. A snapshot with no recorded head, a corrupt
    snapshot or an unborn ref is skipped.

    Args:
        snapshot_path: The run's snapshot.
        repo_root: The checkout.
        chain_ref: The chain ref resume commits on top of.

    Returns:
        `(snapshot head, resume-onto head)` on divergence, else None.
    """
    snap_head = ""
    with contextlib.suppress(OSError, ValueError):
        loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            # A raw peek that must not raise; the key is `SessionSnapshot.head_sha`.
            snap_head = str(loaded.get("head_sha") or "")
    if not snap_head:
        return None
    try:
        current_head = git_read.chain_tip(repo_root, chain_ref)
    except errors.GitError:
        return None
    if current_head is None or current_head == snap_head:
        return None
    if git_read.is_ancestor(repo_root, snap_head, current_head):
        # The tip moved forward on the same line: the run's own commits.
        return None
    return (snap_head, current_head)


@dataclasses.dataclass(frozen=True, slots=True)
class ResumeTarget:
    """The session a resume found, and what its manifest settles before any lock.

    Attributes:
        repo: The repository the resume was started in.
        state: The state directory the execution runs with: the repository's, or the clone's
            of a continued child.
        layout: The session's directory layout.
        manifest: The session's manifest, as read before the resume touched anything.
        mode: The session mode.
        route: The route this execution drives with: `--model`, else the recorded one.
        cwd: The checkout: the session's worktree, else the repository.
        budget: The budget flags over the caps the run recorded.
    """

    repo: pathlib.Path
    state: pathlib.Path
    layout: sessions_layout.SessionLayout
    manifest: sessions_manifest.SessionManifest
    mode: kinds.ResumableMode
    route: str | kinds.ModelRoute | None
    cwd: pathlib.Path
    budget: session_setup.BudgetOverrides

    @property
    def role(self) -> RoleName:
        """The role that drives the session's mode."""
        return kinds.session_kind(self.mode).role

    @property
    def snapshot_path(self) -> pathlib.Path:
        """Where the loop keeps the session's rolling snapshot."""
        return self.layout.session_dir / "loop_state.json"


def find_resume(
    repo: pathlib.Path,
    session_id: str,
    *,
    steer: str,
    model: str,
    budget_overrides: session_setup.BudgetOverrides | None,
    reporter: app_reporter.Reporter,
) -> ResumeTarget:
    """Find the session and refuse one this resume cannot continue, before any lock.

    Args:
        repo: The repository the resume was started in.
        session_id: The session's id or prefix.
        steer: The `--steer` text.
        model: The `--model` value.
        budget_overrides: The budget flags.
        reporter: Receives the refusal.

    Returns:
        The session and what its manifest settles.

    Raises:
        preflight.SessionRefusedError: The steer, the id, the manifest, the child authority
            or the worktree refused.
    """
    launch = child_launch.read()
    # A continued child writes to its bound record but executes with its clone's state.
    record, state = session_prepare.record_dirs(
        repo, launch if launch is not None and launch.continuation else None, reporter=reporter
    )
    if steer.strip() and (problem := directive.steer_problem(steer)) is not None:
        reporter.error(problem)
        raise preflight.SessionRefusedError(2)
    # One resolver across buckets: a runs/-only fallback would pick a run over an ask.
    try:
        layout = id.resolve_session(record, session_id)
    except id.SessionIdError as exc:
        reporter.error(str(exc))
        raise preflight.SessionRefusedError(2) from exc
    # The manifest first: clearing state before it would clobber a session resume refuses.
    try:
        manifest = sessions_manifest.read_manifest(layout.session_dir)
        # A resume keeps the recorded route unless it sets its own.
        recorded = manifest.models.driver
        route: str | kinds.ModelRoute | None = model or (
            kinds.ModelRoute(recorded.provider, recorded.model) if recorded is not None else None
        )
        mode = manifest.session_mode()
        # A `--max-*` set on the run replays the same way, field by field.
        caps = (budget_overrides or session_setup.BudgetOverrides()).over(manifest.budget)
    except sessions_manifest.ManifestError as exc:
        reporter.error(f"cannot resume {layout.session_id}: {exc}")
        raise preflight.SessionRefusedError(2) from exc
    owned_child = child_resume.authorized(launch, manifest, session_id=layout.session_id, root=repo)
    if launch is not None and launch.continuation and not owned_child:
        reporter.refuse("the private launch does not authorize this child continuation")
        raise preflight.SessionRefusedError(2)
    if manifest.parallel is not None and not owned_child:
        reporter.refuse(listing.manifest_resume_blocker(manifest, layout.session_id))
        raise preflight.SessionRefusedError(2)
    cwd = launch.worktree if owned_child and launch is not None else manifest.worktree or repo
    require_worktree(repo, layout.session_id, manifest, reporter=reporter)
    return ResumeTarget(
        repo=repo,
        state=state,
        layout=layout,
        manifest=manifest,
        mode=mode,
        route=route,
        cwd=cwd,
        budget=caps,
    )


def require_worktree(
    repo: pathlib.Path,
    session_id: str,
    manifest: sessions_manifest.SessionManifest,
    *,
    reporter: app_reporter.Reporter,
) -> None:
    """Refuse a session whose recorded worktree is gone or cannot be jailed.

    Args:
        repo: The repository, where the refusal looks for the session's commits.
        session_id: The session's id.
        manifest: The session's manifest.
        reporter: Receives the refusal.

    Raises:
        preflight.SessionRefusedError: The worktree is missing, or the manifest names no
            repository git dir for a linked one.
    """
    if missing := sessions_manifest.worktree_blocker(manifest.worktree):
        reporter.error(
            f"cannot resume {session_id}: {missing}; {commits_note(repo, manifest)};"
            f" `agent6 fork {session_id}` continues it in a new worktree."
        )
        raise preflight.SessionRefusedError(2)
    if (
        manifest.worktree is not None
        and manifest.worktree_git_dir is None
        and not fork_worktrees.is_clone(manifest.worktree)
    ):
        reporter.error(
            f"cannot resume {session_id}: its manifest names a worktree but not the repository"
            f" git dir it points into, so its jail cannot grant one; `agent6 fork {session_id}`"
            " continues its commits in a new worktree."
        )
        raise preflight.SessionRefusedError(2)


def reopen_record(
    target: ResumeTarget, *, steer: str, merge_from: str, reporter: app_reporter.Reporter
) -> session_log.LogScan:
    """Bring the record up to date under the writer lock, and read how the session ended.

    Args:
        target: The session.
        steer: The `--steer` text.
        merge_from: The `--merge` branch.
        reporter: Receives the refusal.

    Returns:
        The scan of the journal.

    Raises:
        preflight.SessionRefusedError: The clone lacks the run's commits, or the session
            has nothing to resume.
    """
    layout, manifest, session_id = target.layout, target.manifest, target.layout.session_id
    if fork_worktrees.mirror_origin(target.repo, target.cwd) is not None:
        # The repository's refs are the record between executions: a fork's seed, a merge.
        try:
            # The repository is the sender, as it was when the clone was made.
            git_ops.mirror_session(
                target.cwd,
                target.repo,
                git_read.chain_ref_for(session_id),
                manifest.run_branch or None,
                sender=None,
            )
        except errors.GitError as exc:
            reporter.error(
                f"cannot resume {session_id}: its worktree {target.cwd} has no copy of its"
                f" commits and git could not fetch them: {exc}"
            )
            raise preflight.SessionRefusedError(1) from exc
    # A finished run has nothing to continue without --steer; every other ending resumes.
    scan = session_log.scan_session_log(layout.logs_path)
    new_work = listing.needs_new_work(
        finished=scan.finished,
        stop_reason=scan.stop_reason,
        verification=scan.verification,
        saved_answers=len(ipc.saved_question_answers(layout.session_dir)),
    )
    if scan.finished and scan.stop_reason == "undone":
        # The fork is the continuation; resuming here would redo the taken-back edit.
        reporter.refuse(listing.undone_refusal(session_id, scan.undone_to))
        raise preflight.SessionRefusedError(2)
    if not steer.strip() and not merge_from and new_work:
        reporter.refuse(listing.needs_new_work_refusal(session_id))
        raise preflight.SessionRefusedError(2)
    return scan


def reset_bridge(
    target: ResumeTarget,
    scan: session_log.LogScan,
    *,
    steer: str,
    started_at: float,
    reporter: app_reporter.Reporter,
) -> resume_questions.CarriedQuestions:
    """Drop the bridge state the last execution left, and seed this execution's steer.

    Args:
        target: The session.
        scan: The scan of the journal.
        steer: The `--steer` text.
        started_at: The instant this execution began; the stale-state clear keeps bridge
            markers written since.
        reporter: Receives the refusal.

    Returns:
        The questions the journal carries over.

    Raises:
        preflight.SessionRefusedError: The steer could not be written.
    """
    layout = target.layout
    carried = resume_questions.carried_questions(layout.logs_path)
    # Approval ids reset on resume, so their stale answers go; a pending question's stays.
    ipc.clear_pending_answers(
        layout.session_dir,
        started_at=started_at,
        keep_questions=[request.id for request in carried.pending],
        keep_steer=scan.stop_reason == "setup_command_failed",
    )
    # The steer as a front-end recorded it before it launched the resume: its images and its id.
    launch = ipc.take_launch_input(layout.session_dir, steer, follow_up=True)
    if steer.strip() and not ipc.submit_steer(
        layout.session_dir, steer, images=launch.images, message_id=launch.message_id
    ):
        reporter.error("could not write the initial steer request")
        raise preflight.SessionRefusedError(2)
    return carried


def read_snapshot(
    target: ResumeTarget, *, reporter: app_reporter.Reporter
) -> loop_snapshot.SessionSnapshot:
    """Read the snapshot the resume continues from, before anything is touched.

    Args:
        target: The session.
        reporter: Receives the refusal.

    Returns:
        The snapshot.

    Raises:
        preflight.SessionRefusedError: The session wrote no snapshot, the checkout is no git
            repository, or the snapshot or a dispatch record is unreadable.
    """
    layout, session_id = target.layout, target.layout.session_id
    if target.manifest.worktree is not None:
        reporter.note(f"working in the worktree {target.cwd}")
    if not target.snapshot_path.is_file():
        reporter.error(
            f"{session_id} never reached its first turn: no resume snapshot was written,"
            " so there is nothing to resume. Start the task again with a new run."
        )
        raise preflight.SessionRefusedError(2)
    # An ask is read-only and may run outside a git repo, so it skips the git preflight.
    # The no-repo guard runs before any git-touching check.
    if target.mode != "ask" and not preflight.require_git_repo(target.cwd, reporter=reporter):
        raise preflight.SessionRefusedError(2)
    # An unreadable or outdated snapshot refuses here, before anything is touched.
    try:
        snapshot = loop_snapshot.load_session_snapshot(target.snapshot_path)
    except (ValueError, OSError) as exc:
        reporter.error(str(exc))
        raise preflight.SessionRefusedError(1) from exc
    # The loop settles children from these records at every turn boundary.
    try:
        child_dispatch.read_records(layout.session_dir)
    except child_dispatch.DispatchRecordError as exc:
        reporter.error(
            f"{session_id} cannot be resumed: a dispatch record is unreadable."
            f" Start a new run.\n{exc}"
        )
        raise preflight.SessionRefusedError(1) from exc
    return snapshot


def check_chain(
    target: ResumeTarget, *, force: bool, reporter: app_reporter.Reporter
) -> resume_chain.MergedStart | None:
    """Refuse a chain that left the snapshot, unless `--force` overrides.

    Args:
        target: The session.
        force: Whether to resume onto a chain that diverged, or past a branch off it.
        reporter: Receives the guard's lines and the refusal.

    Returns:
        Where a merged run continues from; None when its chain continues as it is.

    Raises:
        preflight.SessionRefusedError: The chain diverged from the snapshot, the run branch
            left the chain, or git could not read the checkout.
    """
    cwd, manifest, session_id = target.cwd, target.manifest, target.layout.session_id
    # A rewritten chain ref would leave the model reasoning about a changed record.
    mismatch = (
        snapshot_head_mismatch(
            target.snapshot_path, cwd, chain_ref=git_read.chain_ref_for(session_id)
        )
        if target.mode != "ask"
        else None
    )
    if mismatch is not None:
        snap_head, onto_head = mismatch
        reporter.err("GUARD: the code this run would resume onto diverged from its last snapshot.")
        reporter.err(f"  snapshot head: {snap_head}")
        reporter.err(f"  resume onto:   {onto_head}")
        if not force:
            reporter.err("REFUSING to resume. Re-run with --force to override.")
            raise preflight.SessionRefusedError(2)
    # A merged run can continue from HEAD. Otherwise, check for commits off the chain.
    if target.mode != "run":
        return None
    try:
        merged = resume_chain.merged_start(cwd, manifest)
        off = None if merged else resume_chain.branch_off_the_chain(cwd, manifest)
    except errors.GitError as exc:
        reporter.error(str(exc))
        raise preflight.SessionRefusedError(2) from exc
    if off is None:
        return merged
    branch, branch_tip, ahead = off
    if not force:
        reporter.refuse(resume_chain.off_chain_refusal(session_id, branch, branch_tip, ahead=ahead))
        raise preflight.SessionRefusedError(2)
    reporter.note(f"continuing on the chain; {branch} stays at {branch_tip[:12]}, off the run")
    return merged
