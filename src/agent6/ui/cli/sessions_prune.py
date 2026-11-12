# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions prune`: clean up what landed.

Prune removes merged branches and chain refs, imported clones and merged fork worktrees.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import pathlib
import sys
from typing import Literal

from agent6 import errors, paths
from agent6.app import fork_worktrees, merge, worktrees
from agent6.config import ConfigError, layer
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.sessions import process
from agent6.ui.cli import common
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel import scan_session_log

_UNSTARTED_FORK = "unmerged; fork has not started"
_NO_RECORD = "no session record"


def _session_stamp(
    layout: sessions_layout.SessionLayout | None,
) -> tuple[sessions_manifest.MergeStamp | None, str]:
    """Return a session's recorded merge and why none could be read.

    Args:
        layout: The run; None when it has no session record.

    Returns:
        `(stamp, reason)`: the reason names a missing or unreadable record, or an unstarted
        fork (kept, never deleted), else ""; the stamp is None when the manifest records no merge.
    """
    if layout is None:
        return None, _NO_RECORD
    try:
        manifest = sessions_manifest.read_manifest(layout.session_dir)
    except sessions_manifest.ManifestError:
        return None, "unreadable manifest"
    if manifest.parent_session_id and not scan_session_log(layout.logs_path).saw_start:
        return manifest.merged, _UNSTARTED_FORK
    return manifest.merged, ""


def _stamp_or_record(
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout | None,
    ref: str,
    tip: str | None,
    current: str,
) -> tuple[sessions_manifest.MergeStamp | None, str]:
    """Return a session's merge record, writing one first for a run landed by hand.

    A run with no record whose base branch already holds its content is recorded as merged,
    and a line says so.

    Args:
        cwd: The repo.
        layout: The run; None when it has no session record.
        ref: The branch or chain ref holding the run's commits, for the line.
        tip: Its sha; None when it is gone.
        current: The checked-out branch, checked when the run's base branch is gone.

    Returns:
        `_session_stamp`'s pair, the stamp being the record just written when one was.
    """
    stamp, why = _session_stamp(layout)
    if layout is None or stamp is not None or why or tip is None:
        return stamp, why
    stamp = merge.record_if_landed(cwd, layout, tip, fallback=current)
    if stamp is not None:
        print(
            f"[agent6] {ref}: {stamp.into} already has its content;"
            f" recorded as merged into {stamp.into}"
        )
    return stamp, why


def _base_gone(into: str) -> str:
    """Return the keep reason for a merge base that no longer exists."""
    return f"base {into} is gone"


def _squash_unconfirmed(cwd: pathlib.Path, stamp: sessions_manifest.MergeStamp) -> str:
    """Return why a squash-merge stamp does not prove a force-delete content-safe, or "".

    The merged tip must be recorded, and the base must still hold the commit the record
    names (the merge commit, or for a merge that added nothing, the base tip that already
    held the content) or the content itself: a base rewritten with the content kept
    (a reword, a rebase, a signing pass) still holds it; a reset leaves the branch as its
    only holder.

    Args:
        cwd: The repo.
        stamp: The merge stamp.

    Returns:
        The keep reason; "" when the checks pass.
    """
    if not stamp.tip:
        return "no merge tip was recorded"
    if stamp.sha == sessions_manifest.NO_MERGE_COMMIT:
        if not stamp.into_tip:
            return "the record names no commit to check"
        named, lost = stamp.into_tip, f"{stamp.into} no longer holds its content"
    else:
        named, lost = stamp.sha, f"{stamp.into} no longer holds the merge commit"
    if git_read.is_ancestor(cwd, named, stamp.into):
        return ""
    with contextlib.suppress(errors.GitError):
        if git_ops.holds_content(cwd, stamp.into, stamp.tip):
            return ""
    return lost


@dataclasses.dataclass(frozen=True, slots=True)
class Landed:
    """How a run's commits stand against its merge stamp, for both prune loops.

    Attributes:
        verdict: `merged` (its tip is an ancestor of the base), `squashed` (the stamp proves
            the base holds them and the operator asked for the force-delete), else `keep`.
        why: The keep reason each loop prints and counts.
    """

    verdict: Literal["merged", "squashed", "keep"]
    why: str = ""


def landed(
    cwd: pathlib.Path,
    stamp: sessions_manifest.MergeStamp,
    tip: str | None,
    *,
    delete_squashed: bool,
) -> Landed:
    """Return the one classification of a run with a recorded merge.

    Args:
        cwd: The repo.
        stamp: The merge stamp.
        tip: The branch's or chain ref's sha; None when the branch is gone.
        delete_squashed: `--delete-squashed` was given.

    Returns:
        The verdict, with the keep reason when it is `keep`.
    """
    if not git_read.branch_exists(cwd, stamp.into):
        return Landed("keep", _base_gone(stamp.into))
    if tip is not None and git_read.is_ancestor(cwd, tip, stamp.into):
        return Landed("merged")
    if tip is not None and stamp.tip and stamp.tip != tip:
        # A resumed run committed on after the merge: those commits are in no other ref.
        return Landed("keep", "advanced since the merge")
    if why := _squash_unconfirmed(cwd, stamp):
        return Landed("keep", why)
    return Landed("squashed") if delete_squashed else Landed("keep", "squash-merged")


def _keep_branch_line(br: str, stamp: sessions_manifest.MergeStamp, why: str) -> str:
    """Return the line naming why a run branch is kept."""
    if why == "squash-merged":
        return (
            f"[agent6] kept {br} (squash-merged into {stamp.into}, unreachable; "
            f"remove with: sessions prune --delete-squashed, or: git branch -D {br})"
        )
    at = "" if stamp.sha == sessions_manifest.NO_MERGE_COMMIT else f" at {stamp.sha[:12]}"
    return (
        f"[agent6] kept {br} (squash-merged into {stamp.into}{at}, but {why};"
        f" review, then: git branch -D {br})"
    )


def _prune_branch(
    cwd: pathlib.Path, br: str, stamp: sessions_manifest.MergeStamp, state: Landed, current: str
) -> bool:
    """Act on one run branch's classification.

    A proven squash is force-deleted with the undelete hint (its tip sha, restorable until
    gc prunes the unreachable commit); the rest are kept and the reason said.

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
    sha = git_read.branch_tip_sha(cwd, br)
    if sha is not None and git_ops.force_delete_squash_merged_branch(cwd, br):
        print(f"[agent6] deleted {br} (squash-merged into {stamp.into})")
        print(common.sgr(f"          undelete: git branch {br} {sha[:12]}", "2"))
        return True
    print(f"[agent6] kept {br} (squash-merged into {stamp.into}; git refused the delete)")
    return False


def cmd_prune(*, delete_squashed: bool = False, config_paths: tuple[pathlib.Path, ...] = ()) -> int:
    """Delete what `git branch -d` can safely remove, and sweep the merged clones and worktrees.

    Run branches reachable-merged into HEAD go; squash-merged and unmerged ones are
    reported. Group clone dirs go when every lane branch tip exists in this repo; a clone
    holding any commit this repo lacks is kept whole. The worktree of a merged fork goes;
    an unmerged fork keeps its worktree, and an unstarted fork keeps its branch too.
    With `--delete-squashed`, branches and chain refs
    the manifest confirms were squash-merged into an existing base are force-deleted too,
    each deletion printing the exact command to undelete it. Unmerged runs are never
    force-deleted.

    Args:
        delete_squashed: Force-delete proven squash merges.
        config_paths: The `--config` files in command-line order, if any.

    Returns:
        The exit code; 2 when git refuses.
    """
    cwd = fork_worktrees.sweep_checkout(pathlib.Path.cwd())
    if not git_read.is_git_repo(cwd):
        common.error("not a git repository")
        return 2
    branches = git_read.list_run_branches(cwd)
    try:
        current = git_read.status(cwd).branch
    except errors.GitError as exc:
        common.error(f"{exc}")
        return 2
    repo_state = paths.state_dir(cwd)
    deleted = squashed_deleted = merged_kept = live_kept = 0
    kept_why: collections.Counter[str] = collections.Counter()
    for br in branches:
        if br == current:
            print(f"[agent6] skipped {br} (checked out)", file=sys.stderr)
            continue
        layout = sessions_layout.session_layout(repo_state, br.removeprefix(git_read.BRANCH_PREFIX))
        if layout is not None and process.worker_is_alive(layout.session_dir):
            # The run is still committing to it, whatever git makes of its tip.
            live_kept += 1
            print(f"[agent6] kept {br} (live)")
            continue
        stamp, why = _stamp_or_record(cwd, layout, br, git_read.branch_tip_sha(cwd, br), current)
        if why != _UNSTARTED_FORK and git_ops.delete_branch_if_merged(cwd, br):
            deleted += 1
            print(f"[agent6] deleted {br} (merged)")
            continue
        if why or stamp is None or not stamp.sha:
            label = why or "unmerged"
            kept_why[label] += 1
            print(f"[agent6] kept {br} ({label}; review, then: git branch -D {br})")
            continue
        state = landed(
            cwd, stamp, git_read.branch_tip_sha(cwd, br), delete_squashed=delete_squashed
        )
        if _prune_branch(cwd, br, stamp, state, current):
            squashed_deleted += 1
        else:
            merged_kept += 1
    # Chain refs are pruned whether or not a run branch survives: `branch_per_run` off has none.
    refs_deleted, refs_kept = _prune_chain_refs(
        cwd, repo_state, current, delete_squashed=delete_squashed
    )
    clones_note, swept_any = _sweep_workdirs(cwd, repo_state, config_paths)
    if not branches and not (refs_deleted or refs_kept or swept_any):
        print("[agent6] nothing to prune: no agent6/* run branches, no chain refs.")
        return 0
    # A branch with no record cannot be confirmed merged: the stamp that held the squash is gone.
    unrecorded_kept = kept_why.pop(_NO_RECORD, 0)
    unmerged_kept = sum(kept_why.values())
    kept = merged_kept + unmerged_kept + live_kept + unrecorded_kept
    total_deleted = deleted + squashed_deleted
    squashed_note = f" ({squashed_deleted} squash-merged)" if squashed_deleted else ""
    live_note = (f", {live_kept} live" if live_kept else "") + (
        f", {unrecorded_kept} without a record" if unrecorded_kept else ""
    )
    refs_note = ""
    if refs_deleted or refs_kept:
        why = ", ".join(f"{n} {reason}" for reason, n in sorted(refs_kept.items()))
        refs_note = f"; chain refs: deleted {refs_deleted}, kept {sum(refs_kept.values())}" + (
            f" ({why})" if why else ""
        )
    print(
        f"\n[agent6] branches: deleted {total_deleted}{squashed_note}; kept {kept} "
        f"({merged_kept} merged, {unmerged_kept} unmerged{live_note}){refs_note}{clones_note}",
    )
    return 0


def _sweep_workdirs(
    cwd: pathlib.Path, state: pathlib.Path, config_paths: tuple[pathlib.Path, ...]
) -> tuple[str, bool]:
    """Sweep group and machine clones and fork worktrees under `[git.worktree].base_dir`.

    Prints each keep and each worktree removal.

    Args:
        cwd: The repo.
        state: The repo's state dir.
        config_paths: The `--config` files in command-line order, if any.

    Returns:
        The summary-line note, and whether anything was swept or kept.
    """
    try:
        cfg = layer.load_effective(cwd, config_paths).config
    except ConfigError as exc:
        print(f"[agent6] worktree sweep skipped (config unreadable: {exc})", file=sys.stderr)
        return "", False
    clones_swept, clones_kept = worktrees.sweep_group_clones(
        cwd,
        cfg,
        note_keep=lambda path, why: print(f"[agent6] kept clone group at {path} ({why})"),
    )
    worktrees_removed, worktrees_kept = fork_worktrees.sweep_fork_worktrees(cwd, state)
    for fork_id, why in worktrees_removed:
        print(f"[agent6] removed {fork_id}'s worktree ({why})")
    for fork_id, why in worktrees_kept:
        print(f"[agent6] kept {fork_id}'s worktree ({why})")
    common.note_removed_cwd(cwd)
    note = (
        f"; clone groups: swept {clones_swept}, kept {clones_kept}"
        if clones_swept or clones_kept
        else ""
    )
    if worktrees_removed or worktrees_kept:
        note += f"; worktrees: removed {len(worktrees_removed)}, kept {len(worktrees_kept)}"
    return note, bool(clones_swept or clones_kept or worktrees_removed or worktrees_kept)


def _prune_chain_refs(
    cwd: pathlib.Path, repo_state: pathlib.Path, current: str, *, delete_squashed: bool
) -> tuple[int, collections.Counter[str]]:
    """Drop the `refs/agent6/<id>/head` chain refs whose manifest confirms the run merged.

    The same safety rules as branches: reachable from the base deletes outright; a squash
    merge deletes only with `--delete-squashed` and only while the ref still points at the
    recorded merged tip. Live runs, unmerged runs and refs with no run manifest (machine
    chains) are kept, counted by reason and never named: an unmerged ref is the run's anchor.

    Args:
        cwd: The repo.
        repo_state: The repo's state dir.
        current: The checked-out branch.
        delete_squashed: Force-delete proven squash merges.

    Returns:
        The count deleted, and the kept refs counted by reason; every ref counted once.
    """
    refs_deleted = 0
    kept: collections.Counter[str] = collections.Counter()
    for sid, sha in git_read.list_chain_refs(cwd):
        layout = sessions_layout.session_layout(repo_state, sid)
        if layout is None:
            # A machine's chain (`machine_chain_ref_for`) has no session record.
            kept["machine" if sid.startswith("machine-") else "no session record"] += 1
            continue
        if process.worker_is_alive(layout.session_dir):
            kept["live"] += 1
            continue
        stamp, why = _stamp_or_record(cwd, layout, git_read.chain_ref_for(sid), sha, current)
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
        ref = git_read.chain_ref_for(sid)
        git_ops.delete_ref(cwd, ref)
        refs_deleted += 1
        how = "merged" if state.verdict == "merged" else "squash-merged"
        print(f"[agent6] deleted {ref} ({how} into {stamp.into})")
        if state.verdict == "squashed":
            # A chain ref has no reflog: the sha is the only way back.
            print(common.sgr(f"          undelete: git update-ref {ref} {sha[:12]}", "2"))
    return refs_deleted, kept
