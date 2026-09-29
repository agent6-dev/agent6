# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions merge/prune`: land a run's chain on its base, and clean up what landed.

Prune removes the branches, chain refs, fan-out clones and fork worktrees already merged.
"""

from __future__ import annotations

import contextlib
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.app.fork_worktrees import sweep_fork_worktrees
from agent6.app.merge import NO_BASE_SHA, execute_merge, left_behind_line, noop_merge_line
from agent6.app.parallel import adopt_orphan_lane, sweep_fanout_clones
from agent6.commit_message import render_commit_trailer
from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective
from agent6.git_ops import (
    BRANCH_PREFIX,
    CommitIdentity,
    GitError,
    branch_exists,
    branch_tip_sha,
    chain_ref_for,
    chain_tip,
    delete_branch_if_merged,
    delete_ref,
    force_delete_squash_merged_branch,
    is_ancestor,
    is_git_repo,
    list_chain_refs,
    list_run_branches,
    list_run_commits,
    verify_git_identity,
)
from agent6.git_ops import status as git_status
from agent6.harness.subrun import SubrunError
from agent6.paths import state_dir
from agent6.sessions.ipc import worker_is_alive
from agent6.sessions.layout import LOGS_NAME, SessionLayout, session_layout
from agent6.sessions.manifest import (
    NO_MERGE_COMMIT,
    ManifestError,
    MergeStamp,
    SessionManifest,
    read_manifest,
)
from agent6.ui.cli._common import error, refuse, sgr
from agent6.ui.cli.sessions_cmds import (
    _commits_ref,
    _committed_nothing,
    _resolve_session_manifest,
)
from agent6.viewmodel import tail_events, worker_models


@dataclass(frozen=True, slots=True)
class _MergePlan:
    """A validated, mutation-ready merge: everything `_cmd_merge` needs after every guard passed.

    Attributes:
        layout: The run.
        manifest: Its manifest.
        run_branch: The branch or chain ref holding the work.
        target: The branch to land on.
        base_sha: The commit the run was cut from.
        strategy: The merge strategy.
        identity: The committer identity.
        cfg: The effective config.
    """

    layout: SessionLayout
    manifest: SessionManifest
    run_branch: str
    target: str
    base_sha: str
    strategy: str
    identity: CommitIdentity
    cfg: Config


def _plan_merge(  # noqa: PLR0911
    cwd: Path,
    session_id: str,
    into: str | None,
    strategy: str | None,
    *,
    config_path: Path | None,
) -> _MergePlan | int:
    """Resolve and validate everything a merge needs, without touching the repo.

    Args:
        cwd: The repo.
        session_id: The run, or "" for the newest.
        into: The target branch; None takes the branch the run was cut from.
        strategy: The merge strategy; None takes `git.merge_strategy`.
        config_path: The `--config` file, if any.

    Returns:
        The plan, or the exit code of a printed refusal: every guard fails here, before
        `_cmd_merge` mutates anything.
    """
    res = _resolve_session_manifest(cwd, session_id)
    if isinstance(res, int):
        return res
    layout, manifest = res
    # The raw pid, not session_is_live: after session.end the finalizer may still be at work.
    if worker_is_alive(layout.session_dir):
        refuse(
            f"run {session_id!r} is still live; a merge now lands only the"
            " commits so far (its later ones need another merge) and moves your"
            " index while the worker still edits. Stop it first:\n"
            f"    agent6 stop {session_id}"
        )
        return 2
    ref = _commits_ref(cwd, manifest)
    # execute_merge refuses a missing base_sha too; here it is a refusal before anything moves.
    unmergeable = (
        NO_BASE_SHA
        if not manifest.base_sha
        else f"this session has no branch to merge ({ref.reason})."
        if ref.reason
        else ""
    )
    if unmergeable:
        refuse(unmergeable)
        return 2
    run_branch = ref.head_ref
    target = into or manifest.base_branch
    if not target:
        error("no target branch (manifest has no base_branch); pass --into <branch>.")
        return 2
    if target == run_branch:
        error(f"target {target!r} is the run branch itself; pass --into <other-branch>.")
        return 2
    try:
        cfg = load_effective(cwd, config_path).config
    except ConfigError as exc:
        error(f"{exc}")
        return 2
    # An orphaned lane (its coordinator died before importing it) is adopted: fetched, then merged.
    try:
        adopted = adopt_orphan_lane(cwd, cfg, layout, manifest)
    except SubrunError as exc:
        error(f"{exc}")
        return 2
    if adopted is not None:
        print(f"[agent6] {adopted}")
        layout = SessionLayout(state_dir=layout.state_dir, session_id=layout.session_id)
    # chain_tip resolves both shapes of head_ref: a branch name and the hidden chain ref.
    if chain_tip(cwd, run_branch) is None:
        if _committed_nothing(cwd, manifest.session_id):
            # Not a failure: there is no work to land.
            print("[agent6] nothing to merge: this run committed nothing.")
            return 0
        error(
            f"run ref {run_branch!r} no longer exists; its commits survive at"
            f" {chain_ref_for(manifest.session_id)}."
        )
        return 2
    if not branch_exists(cwd, target):
        error(f"target branch {target!r} does not exist; pass --into <existing-branch>.")
        return 2
    identity = CommitIdentity(
        name=cfg.git.commit.name,
        email=cfg.git.commit.email,
        trailer=render_commit_trailer(
            cfg.git.commit.trailer,
            models=worker_models(tail_events(layout.session_dir / LOGS_NAME, follow=False))
            or ((manifest.models.driver.model,) if manifest.models.driver else ()),
        ),
    )
    try:
        verify_git_identity(cwd, identity)  # refuse cleanly before mutating anything
    except GitError as exc:
        error(f"{exc}")
        return 2
    return _MergePlan(
        layout=layout,
        manifest=manifest,
        run_branch=run_branch,
        target=target,
        base_sha=manifest.base_sha,
        strategy=strategy or cfg.git.merge_strategy,
        identity=identity,
        cfg=cfg,
    )


def _manual_merge_cmd(cwd: Path, plan: _MergePlan) -> str:
    """Return the by-hand merge for a plumbing conflict, from the checkout as it is.

    Git refuses to merge over modified tracked files, and a finished run leaves its work in
    the tree until merged, so a dirty tree is stashed first; a checkout on another branch
    moves to the target.
    """
    steps: list[str] = []
    with contextlib.suppress(GitError):
        st = git_status(cwd)
        if st.modified_count:
            steps.append("git stash")
        if st.branch != plan.target:
            steps.append(f"git checkout {plan.target}")
    squash = " --squash" if plan.strategy == "squash" else ""
    steps.append(f"git merge{squash} {plan.run_branch}")
    return " && ".join(steps)


def _cmd_merge(
    *,
    session_id: str,
    strategy: str | None,
    into: str | None,
    message: str | None,
    config_path: Path | None,
) -> int:
    """Land a run's work on a target branch with the chosen strategy.

    Ref plumbing only: the checkout, index and worktree are never the medium, so a worktree
    still carrying the run's work is no obstacle.

    Args:
        session_id: The run, or "" for the newest.
        strategy: The merge strategy; None takes `git.merge_strategy`.
        into: The target branch; None takes the branch the run was cut from.
        message: The merge commit message; None takes the default.
        config_path: The `--config` file, if any.

    Returns:
        The exit code: 0 landed or nothing to land, 1 a conflict, 2 a refusal.
    """
    cwd = Path.cwd()
    plan = _plan_merge(cwd, session_id, into, strategy, config_path=config_path)
    if isinstance(plan, int):
        return plan
    if not list_run_commits(cwd, plan.base_sha, plan.run_branch):
        # A success line here would be indistinguishable from a real merge.
        print(f"[agent6] nothing to merge: run branch {plan.run_branch} has no commits.")
        return 0
    outcome = execute_merge(
        cwd,
        layout=plan.layout,
        manifest=plan.manifest,
        run_branch=plan.run_branch,
        target=plan.target,
        base_sha=plan.base_sha,
        strategy=plan.strategy,
        message=message,
        cfg=plan.cfg,
        identity=plan.identity,
        warn=lambda m: print(f"[agent6] {m}", file=sys.stderr),
    )
    if outcome.status == "error":
        error(f"{outcome.error}")
        return 1
    if outcome.status == "conflict":
        print(
            f"CONFLICT: merging {plan.run_branch} into {plan.target} hit conflicts in "
            f"{', '.join(outcome.conflicts)}. Your checkout is untouched (no partial "
            f"merge); resolve it by hand if you want:\n"
            f"    {_manual_merge_cmd(cwd, plan)}",
            file=sys.stderr,
        )
        return 1
    note = (
        f"\n  (merge record could not be written: {outcome.stamp_error};"
        " `sessions prune` will call this branch unmerged)"
        if outcome.stamp_error
        else ""
    )
    if outcome.status == "noop":
        print(f"[agent6] {noop_merge_line(plan.run_branch, plan.target, outcome)}.{note}")
        return 0
    print(
        f"[agent6] merged {plan.run_branch} into {plan.target} "
        f"({plan.strategy}) -> {outcome.merged_sha[:12]}{note}"
    )
    if kept := left_behind_line(plan.target, outcome):
        print(f"[agent6] {kept}")
    return 0


def _session_stamp(layout: SessionLayout | None) -> tuple[MergeStamp | None, str]:
    """Return a session's recorded merge and why none could be read.

    Returns:
        `(stamp, reason)`: the reason is "no session record" or "unreadable manifest"
        (kept, never force-deleted), else ""; the stamp is None when the manifest records no
        merge.
    """
    if layout is None:
        return None, "no session record"
    try:
        return read_manifest(layout.session_dir).merged, ""
    except ManifestError:
        return None, "unreadable manifest"


def _base_gone(into: str) -> str:
    """Return the keep reason for a merge base that no longer exists."""
    return f"base {into} is gone"


def _squash_unconfirmed(cwd: Path, stamp: MergeStamp) -> str:
    """Return why a squash-merge stamp does not prove a force-delete content-safe, or "".

    The merged tip must be recorded, and the base must still hold the commit the record
    names (the merge commit, or for a merge that added nothing, the base tip that already
    held the content): a reset or rewrite of the base leaves the branch as the content's
    only holder.

    Args:
        cwd: The repo.
        stamp: The merge stamp.
    """
    if not stamp.tip:
        return "no merge tip was recorded"
    if stamp.sha == NO_MERGE_COMMIT:
        if not stamp.into_tip:
            return "the record names no commit to check"
        if not is_ancestor(cwd, stamp.into_tip, stamp.into):
            return f"{stamp.into} no longer holds its content"
    elif not is_ancestor(cwd, stamp.sha, stamp.into):
        return f"{stamp.into} no longer holds the merge commit"
    return ""


@dataclass(frozen=True, slots=True)
class Landed:
    """How a run's commits stand against its merge stamp, for both prune loops.

    Attributes:
        verdict: `merged` (its tip is an ancestor of the base), `squashed` (the stamp proves
            the base holds them and the operator asked for the force-delete), else `keep`.
        why: The keep reason each loop prints and counts.
    """

    verdict: Literal["merged", "squashed", "keep"]
    why: str = ""


def landed(cwd: Path, stamp: MergeStamp, tip: str | None, *, delete_squashed: bool) -> Landed:
    """Return the one classification of a run with a recorded merge.

    Args:
        cwd: The repo.
        stamp: The merge stamp.
        tip: The branch's or chain ref's sha; None when the branch is gone.
        delete_squashed: `--delete-squashed` was given.
    """
    if not branch_exists(cwd, stamp.into):
        return Landed("keep", _base_gone(stamp.into))
    if tip is not None and is_ancestor(cwd, tip, stamp.into):
        return Landed("merged")
    if tip is not None and stamp.tip and stamp.tip != tip:
        # A resumed run committed on after the merge: those commits are in no other ref.
        return Landed("keep", "advanced since the merge")
    if why := _squash_unconfirmed(cwd, stamp):
        return Landed("keep", why)
    return Landed("squashed") if delete_squashed else Landed("keep", "squash-merged")


def _keep_branch_line(br: str, stamp: MergeStamp, why: str) -> str:
    """Return the line naming why a run branch is kept."""
    if why == "squash-merged":
        return (
            f"[agent6] kept {br} (squash-merged into {stamp.into}, unreachable; "
            f"remove with: sessions prune --delete-squashed, or: git branch -D {br})"
        )
    at = "" if stamp.sha == NO_MERGE_COMMIT else f" at {stamp.sha[:12]}"
    return (
        f"[agent6] kept {br} (squash-merged into {stamp.into}{at}, but {why};"
        f" review, then: git branch -D {br})"
    )


def _prune_branch(cwd: Path, br: str, stamp: MergeStamp, state: Landed, current: str) -> bool:
    """Act on one run branch's classification.

    A proven squash is force-deleted with the undelete hint (the commit survives in the
    reflog until GC); the rest are kept and the reason said.

    Args:
        cwd: The repo.
        br: The branch.
        stamp: Its merge stamp.
        state: Its classification.
        current: The checked-out branch.

    Returns:
        Whether the branch was deleted.
    """
    if state.verdict == "merged":
        # `git branch -d` refused only because HEAD is not the base; delete it from there.
        print(
            f"[agent6] kept {br} (merged into {stamp.into} but not reachable from "
            f"{current!r}; re-run prune on {stamp.into}, or: git branch -D {br})"
        )
        return False
    if state.verdict == "keep":
        print(_keep_branch_line(br, stamp, state.why))
        return False
    sha = branch_tip_sha(cwd, br)
    if sha is not None and force_delete_squash_merged_branch(cwd, br):
        print(f"[agent6] deleted {br} (squash-merged into {stamp.into})")
        print(sgr(f"          undelete: git branch {br} {sha[:12]}", "2"))
        return True
    print(f"[agent6] kept {br} (squash-merged into {stamp.into}; git refused the delete)")
    return False


def _cmd_prune(*, delete_squashed: bool = False, config_path: Path | None = None) -> int:
    """Delete what `git branch -d` can safely remove, and sweep the merged clones and worktrees.

    Run branches reachable-merged into HEAD go; squash-merged and unmerged ones are
    reported. Fan-out clone dirs go when every lane branch tip exists in this repo; a clone
    holding any commit this repo lacks is kept whole. The worktree of a merged fork goes;
    an unmerged fork keeps its worktree. With `--delete-squashed`, branches and chain refs
    the manifest confirms were squash-merged into an existing base are force-deleted too,
    each deletion printing the exact command to undelete it. Unmerged runs are never
    force-deleted.

    Args:
        delete_squashed: Force-delete proven squash merges.
        config_path: The `--config` file, if any.

    Returns:
        The exit code; 2 when git refuses.
    """
    cwd = Path.cwd()
    if not is_git_repo(cwd):
        error("not a git repository")
        return 2
    branches = list_run_branches(cwd)
    try:
        current = git_status(cwd).branch
    except GitError as exc:
        error(f"{exc}")
        return 2
    repo_state = state_dir(cwd)
    deleted = squashed_deleted = merged_kept = unmerged_kept = live_kept = 0
    for br in branches:
        if br == current:
            print(f"[agent6] skipped {br} (checked out)", file=sys.stderr)
            continue
        layout = session_layout(repo_state, br.removeprefix(BRANCH_PREFIX))
        if layout is not None and worker_is_alive(layout.session_dir):
            # The run is still committing to it, whatever git makes of its tip.
            live_kept += 1
            print(f"[agent6] kept {br} (live)")
            continue
        if delete_branch_if_merged(cwd, br):
            deleted += 1
            print(f"[agent6] deleted {br} (merged)")
            continue
        stamp, why = _session_stamp(layout)
        if why:
            unmerged_kept += 1
            print(f"[agent6] kept {br} ({why}; review, then: git branch -D {br})")
            continue
        if stamp is None or not stamp.sha:
            unmerged_kept += 1
            print(f"[agent6] kept {br} (unmerged; review, then: git branch -D {br})")
            continue
        state = landed(cwd, stamp, branch_tip_sha(cwd, br), delete_squashed=delete_squashed)
        if _prune_branch(cwd, br, stamp, state, current):
            squashed_deleted += 1
        else:
            merged_kept += 1
    # Chain refs are pruned whether or not a run branch survives: `branch_per_run` off has none.
    refs_deleted, refs_kept = _prune_chain_refs(cwd, repo_state, delete_squashed=delete_squashed)
    clones_note, swept_any = _sweep_workdirs(cwd, repo_state, config_path)
    if not branches and not (refs_deleted or refs_kept or swept_any):
        print("[agent6] nothing to prune: no agent6/* run branches, no chain refs.")
        return 0
    kept = merged_kept + unmerged_kept + live_kept
    total_deleted = deleted + squashed_deleted
    squashed_note = f" ({squashed_deleted} squash-merged)" if squashed_deleted else ""
    live_note = f", {live_kept} live" if live_kept else ""
    refs_note = ""
    if refs_deleted or refs_kept:
        why = ", ".join(f"{n} {reason}" for reason, n in sorted(refs_kept.items()))
        refs_note = f"; chain refs: deleted {refs_deleted}, kept {sum(refs_kept.values())}" + (
            f" ({why})" if why else ""
        )
    print(
        f"\n[agent6] deleted {total_deleted}{squashed_note}; kept {kept} "
        f"({merged_kept} merged, {unmerged_kept} unmerged{live_note}){refs_note}{clones_note}",
    )
    return 0


def _sweep_workdirs(cwd: Path, state: Path, config_path: Path | None) -> tuple[str, bool]:
    """Sweep the fan-out clones and fork worktrees under `[parallel].workdir`.

    Prints each keep and each worktree removal.

    Args:
        cwd: The repo.
        state: The repo's state dir.
        config_path: The `--config` file, if any.

    Returns:
        The summary-line note, and whether anything was swept or kept.
    """
    try:
        cfg = load_effective(cwd, config_path).config
    except ConfigError as exc:
        print(f"[agent6] workdir sweep skipped (config unreadable: {exc})", file=sys.stderr)
        return "", False
    clones_swept, clones_kept = sweep_fanout_clones(cwd, cfg)
    if clones_kept:
        print(
            f"[agent6] kept {clones_kept} fan-out clone dir(s) holding a live lane or"
            " commits this repo lacks (merge or archive their lanes first)"
        )
    worktrees_removed, worktrees_kept = sweep_fork_worktrees(cwd, state)
    for fork_id in worktrees_removed:
        print(f"[agent6] removed {fork_id}'s worktree (merged)")
    for fork_id, why in worktrees_kept:
        print(f"[agent6] kept {fork_id}'s worktree ({why})")
    note = (
        f"; fan-out clones: swept {clones_swept}, kept {clones_kept}"
        if clones_swept or clones_kept
        else ""
    )
    return note, bool(clones_swept or clones_kept or worktrees_removed)


def _prune_chain_refs(
    cwd: Path, repo_state: Path, *, delete_squashed: bool
) -> tuple[int, Counter[str]]:
    """Drop the `refs/agent6/<id>/head` chain refs whose manifest confirms the run merged.

    The same safety rules as branches: reachable from the base deletes outright; a squash
    merge deletes only with `--delete-squashed` and only while the ref still points at the
    recorded merged tip. Live runs, unmerged runs and refs with no run manifest (machine
    chains) are kept, counted by reason and never named: an unmerged ref is the run's anchor.

    Args:
        cwd: The repo.
        repo_state: The repo's state dir.
        delete_squashed: Force-delete proven squash merges.

    Returns:
        The count deleted, and the kept refs counted by reason; every ref counted once.
    """
    refs_deleted = 0
    kept: Counter[str] = Counter()
    for sid, sha in list_chain_refs(cwd):
        layout = session_layout(repo_state, sid)
        if layout is None:
            # A machine's chain (`machine_chain_ref_for`) has no session record.
            kept["machine" if sid.startswith("machine-") else "no session record"] += 1
            continue
        if worker_is_alive(layout.session_dir):
            kept["live"] += 1
            continue
        stamp, why = _session_stamp(layout)
        if why:
            kept[why] += 1
            continue
        if stamp is None or not stamp.sha:
            kept["unmerged"] += 1
            continue
        state = landed(cwd, stamp, sha, delete_squashed=delete_squashed)
        if state.verdict == "keep":
            kept[state.why] += 1
            continue
        ref = chain_ref_for(sid)
        delete_ref(cwd, ref)
        refs_deleted += 1
        how = "merged" if state.verdict == "merged" else "squash-merged"
        print(f"[agent6] deleted {ref} ({how} into {stamp.into})")
        if state.verdict == "squashed":
            # A chain ref has no reflog: the sha is the only way back.
            print(sgr(f"          undelete: git update-ref {ref} {sha[:12]}", "2"))
    return refs_deleted, kept
