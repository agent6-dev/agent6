# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Decide where a resumed run's chain continues from, at `agent6 resume`.

When the checkout's HEAD contains the run's merged work, the run continues there so its next
commit holds only new work. The manifest's base moves with it so `sessions diff` and `commits`,
the squash message, reviews, merges and forks all read the same `base_sha`.
If HEAD lacks the recorded target tip (as in a worktree or fork), the chain stays as it is
and the base moves to the commit the next merge starts from.
A run branch the operator committed on is off the chain, and the resume refuses.
"""

from __future__ import annotations

import dataclasses
import pathlib

from agent6 import errors, kinds, terminal
from agent6.app import merge, preflight, stamps
from agent6.app import reporter as app_reporter
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel import wire


@dataclasses.dataclass(frozen=True, slots=True)
class MergedStart:
    """The continuation point for a merged run whose work the checkout's HEAD contains.

    Attributes:
        head: The checkout's HEAD, which contains the recorded target tip.
        branch: The run's new base branch: the merge target if checked out, else the
            base branch the manifest records.
        tip: The run's merged tip; None when its refs were pruned.
        into: The branch the run merged into.
    """

    head: str
    branch: str
    tip: str | None
    into: str


def merged_start(
    cwd: pathlib.Path, manifest: sessions_manifest.SessionManifest
) -> MergedStart | None:
    """Return where the run continues from when its merge covers everything it committed.

    Args:
        cwd: The checkout.
        manifest: The session's manifest.

    Returns:
        The start, or None when the run has no recorded merge, has commits past its merge,
        or the checkout's HEAD lacks the recorded target tip. Its chain then continues as it is.

    Raises:
        GitError: git could not read the checkout.
    """
    stamp = manifest.merged
    if stamp is None or not stamp.into or manifest.git_control != "agent6":
        return None
    target_tip = stamp.into_tip if stamp.sha == sessions_manifest.NO_MERGE_COMMIT else stamp.sha
    status = git_read.status(cwd)
    if not target_tip or not status.head_sha:
        return None
    if not git_read.is_ancestor(cwd, target_tip, status.head_sha):
        return None
    # Check the ref a merge would use: the run branch while it covers the chain, else the chain.
    ref = wire.commits_ref(manifest, cwd)
    tip = git_read.chain_tip(cwd, ref) if ref else None
    if tip is not None and tip != stamp.tip:
        return None
    branch = stamp.into if status.branch == stamp.into else manifest.base_branch
    return MergedStart(head=status.head_sha, branch=branch, tip=tip, into=stamp.into)


def continue_from_head(
    cwd: pathlib.Path,
    merged: MergedStart,
    manifest: sessions_manifest.SessionManifest,
    identity: git_read.CommitIdentity,
) -> str:
    """Move the run's refs so its chain continues from the checkout's HEAD.

    Args:
        cwd: The checkout.
        merged: Where the run continues from.
        manifest: The session's manifest.
        identity: The author of the continuation commit, if one is needed.

    Returns:
        HEAD when the refs were pruned. Otherwise, the descendant of the run tip and HEAD
        if one contains the other, or a two-parent commit with HEAD's tree.

    Raises:
        GitError: A ref could not be moved.
    """
    if merged.tip is None:
        return merged.head
    return git_ops.chain_continue_from(
        cwd,
        merged.head,
        f"agent6: continue from {merged.into} after the merge",
        ref=git_read.chain_ref_for(manifest.session_id),
        tip=merged.tip,
        identity=identity,
        also_branch=wire.existing_run_branch(manifest, cwd) or None,
    )


def unmerged_base(
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
) -> str | None:
    """Return where the run's unmerged work starts when its base branch holds earlier work.

    For a run whose chain continues as it is: the tip from its own merge, else the base
    `merge.merged_base` gives a fork of a squash-merged run.

    Args:
        cwd: The checkout.
        layout: The session's directory layout.
        manifest: The session's manifest.

    Returns:
        The commit, or None when the base the manifest records already says so.
    """
    ref = wire.commits_ref(manifest, cwd)
    tip = git_read.chain_tip(cwd, ref) if ref else None
    if tip is None or not manifest.base_sha or manifest.git_control != "agent6":
        return None
    stamp, target = manifest.merged, manifest.base_branch
    own = stamp is not None and stamp.into == target and stamp.tip
    if stamp is not None and own and git_read.is_ancestor(cwd, stamp.tip, tip):
        base = stamp.tip
    else:
        base = merge.merged_base(cwd, layout, manifest, target, tip)
    if not base or base == manifest.base_sha:
        return None
    return base if git_read.is_ancestor(cwd, manifest.base_sha, base) else None


def move_base(
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
    merged: MergedStart | None,
    identity: git_read.CommitIdentity,
    *,
    mode: kinds.LoopMode,
    reporter: app_reporter.Reporter,
) -> str:
    """Move the run's base to where this execution's work starts, and record it.

    A merged run continues from HEAD. A run whose base branch holds its earlier work keeps
    its chain and starts its diff and next merge past that work.

    Args:
        cwd: The checkout.
        layout: The session's directory layout.
        manifest: The session's manifest.
        merged: Where the run continues from when HEAD holds its merged work.
        identity: The author of the continuation commit, if one is needed.
        mode: The session mode; only a run's base moves.
        reporter: Receives the refusal and the note.

    Returns:
        The base this execution runs with: the moved one, else the one the manifest records.

    Raises:
        preflight.SessionRefusedError: The run's refs could not be moved to HEAD.
    """
    if merged is not None:
        try:
            base = continue_from_head(cwd, merged, manifest, identity)
        except errors.GitError as exc:
            reporter.error(f"cannot continue this run from HEAD: {exc}")
            raise preflight.SessionRefusedError(2) from exc
        stamps.stamp_continued_from(layout.session_dir, base, merged.branch)
        reporter.note(
            f"this run's earlier work was merged into {merged.into}; continuing from HEAD"
            f" ({merged.head[:12]})"
        )
        return base
    moved = unmerged_base(cwd, layout, manifest) if mode == "run" else None
    if not moved:
        return manifest.base_sha
    # The run's base moves to where its unmerged work starts; its chain stays.
    current = sessions_manifest.read_manifest(layout.session_dir)
    stamps.write_manifest(layout.manifest_path, current.model_copy(update={"base_sha": moved}))
    reporter.note(
        f"this run's earlier work was merged into {manifest.base_branch}; its diff and"
        f" next merge start at {moved[:12]}"
    )
    return moved


def branch_off_the_chain(
    cwd: pathlib.Path, manifest: sessions_manifest.SessionManifest
) -> tuple[str, str, bool] | None:
    """Find a run branch that no longer sits at the chain tip.

    Args:
        cwd: The checkout.
        manifest: The session's manifest.

    Returns:
        The branch, its tip, and whether it still descends from the chain tip (so a merge
        includes the whole branch); None when the branch is at the tip or either is missing.
    """
    branch = wire.existing_run_branch(manifest, cwd)
    chain = git_read.chain_tip(cwd, git_read.chain_ref_for(manifest.session_id))
    tip = git_read.branch_tip_sha(cwd, branch) if branch else None
    if not branch or chain is None or tip is None or tip == chain:
        return None
    return branch, tip, git_read.is_ancestor(cwd, chain, tip)


def merged_in_line(merged: git_ops.ChainMerge) -> str:
    """Return the console line for a branch merged into the run's chain at a resume."""
    if not merged.merged:
        return f"the run's work already contains {merged.branch}; nothing merged in"
    if not merged.conflicts:
        return f"merged {merged.branch} into the run's work ({merged.sha[:12]}); no conflicts"
    shown = ", ".join(terminal.printable_name(p) for p in merged.conflicts)
    return (
        f"merged {merged.branch} into the run's work ({merged.sha[:12]}); conflicts left in"
        f" {shown} for the run to resolve"
    )


def off_chain_refusal(session_id: str, branch: str, tip: str, *, ahead: bool) -> str:
    """Return the refusal for a run branch that left the chain, naming its commit.

    Args:
        session_id: The run.
        branch: The run branch.
        tip: The branch's tip.
        ahead: Whether the branch descends from the chain tip.

    Returns:
        The refusal text with the two ways out.
    """
    first = (
        f"merge it with the run, then resume:  agent6 sessions merge {session_id}"
        if ahead
        else f"take the commit where you want it, then drop the branch:  git branch -D {branch}"
    )
    return (
        f"{branch} is at {tip[:12]}, a commit this run's chain does not hold, and the run's"
        " next commits would leave it behind.\n"
        f"    {first}\n"
        f"    or continue without it:  agent6 resume {session_id} --force"
    )
