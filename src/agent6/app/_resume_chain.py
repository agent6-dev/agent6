# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Decide where a resumed run's chain continues from, at `agent6 resume`.

A run whose work landed continues from the checkout's HEAD: the next commit holds only new
work, and the manifest's base moves there so every reader of `base_sha` (`sessions diff` and
`commits`, the squash message, the review diff, the next merge, a later fork) sees the same.
Where the workspace's HEAD does not hold the landing commit (a worktree run, a fork), the chain
stays as it is and the base moves to the commit the next merge starts from.
A run branch the operator committed on is off the chain, and the resume refuses.
"""

from __future__ import annotations

import dataclasses
import pathlib

from agent6.app import merge
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs import git_ops, git_read
from agent6.viewmodel import wire


@dataclasses.dataclass(frozen=True, slots=True)
class LandedStart:
    """A run whose work landed on the commit the checkout has out.

    Attributes:
        head: The checkout's HEAD, which holds the landing commit.
        branch: The run's base branch from here on: the branch the merge landed on when the
            checkout has it out, else the one the manifest records.
        tip: The run's tip that landed; None when its refs were pruned.
        into: The branch the merge landed on.
    """

    head: str
    branch: str
    tip: str | None
    into: str


def landed_start(
    cwd: pathlib.Path, manifest: sessions_manifest.SessionManifest
) -> LandedStart | None:
    """Return where the run continues from when its merge covers everything it committed.

    Args:
        cwd: The checkout.
        manifest: The session's manifest.

    Returns:
        The start, or None when the run has commits past its merge, never landed, or landed
        on a commit the checkout's HEAD does not hold: its chain then continues as it is.

    Raises:
        GitError: git could not read the checkout.
    """
    stamp = manifest.merged
    if stamp is None or not stamp.into or manifest.git_control != "agent6":
        return None
    landing = stamp.into_tip if stamp.sha == sessions_manifest.NO_MERGE_COMMIT else stamp.sha
    status = git_read.status(cwd)
    if not landing or not status.head_sha:
        return None
    if not git_read.is_ancestor(cwd, landing, status.head_sha):
        return None
    # What a merge would land now: the run branch while it covers the chain, else the chain.
    ref = wire.commits_ref(manifest, cwd)
    tip = git_read.chain_tip(cwd, ref) if ref else None
    if tip is not None and tip != stamp.tip:
        return None
    branch = stamp.into if status.branch == stamp.into else manifest.base_branch
    return LandedStart(head=status.head_sha, branch=branch, tip=tip, into=stamp.into)


def continue_from_head(
    cwd: pathlib.Path,
    landed: LandedStart,
    manifest: sessions_manifest.SessionManifest,
    identity: git_read.CommitIdentity,
) -> str:
    """Move the run's refs so its chain continues from the checkout's HEAD.

    Args:
        cwd: The checkout.
        landed: Where the run continues from.
        manifest: The session's manifest.
        identity: The author of the joining commit.

    Returns:
        The commit the chain continues from: HEAD when the run's refs were pruned or the
        landing was a real merge, else a commit joining the landed tip to HEAD's tree.

    Raises:
        GitError: A ref could not be moved.
    """
    if landed.tip is None:
        return landed.head
    return git_ops.chain_continue_from(
        cwd,
        landed.head,
        f"agent6: continue from {landed.into} after the merge",
        ref=git_read.chain_ref_for(manifest.session_id),
        tip=landed.tip,
        identity=identity,
        also_branch=wire.existing_run_branch(manifest, cwd) or None,
    )


def unlanded_base(
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
) -> str | None:
    """Return where the run's unlanded work starts when its base branch holds earlier work.

    For a run whose chain continues as it is: the tip its own merge landed, else the base
    `merge.landed_base` gives a fork of a squash-merged run.

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
        base = merge.landed_base(cwd, layout, manifest, target, tip)
    if not base or base == manifest.base_sha:
        return None
    return base if git_read.is_ancestor(cwd, manifest.base_sha, base) else None


def branch_off_the_chain(
    cwd: pathlib.Path, manifest: sessions_manifest.SessionManifest
) -> tuple[str, str, bool] | None:
    """Find a run branch that no longer sits at the chain tip.

    Args:
        cwd: The checkout.
        manifest: The session's manifest.

    Returns:
        The branch, its tip, and whether it still descends from the chain tip (a merge then
        lands it whole); None when the branch is at the tip or either is missing.
    """
    branch = wire.existing_run_branch(manifest, cwd)
    chain = git_read.chain_tip(cwd, git_read.chain_ref_for(manifest.session_id))
    tip = git_read.branch_tip_sha(cwd, branch) if branch else None
    if not branch or chain is None or tip is None or tip == chain:
        return None
    return branch, tip, git_read.is_ancestor(cwd, chain, tip)


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
        f"land it with the run, then resume:  agent6 sessions merge {session_id}"
        if ahead
        else f"take the commit where you want it, then drop the branch:  git branch -D {branch}"
    )
    return (
        f"{branch} is at {tip[:12]}, a commit this run's chain does not hold, and the run's"
        " next commits would leave it behind.\n"
        f"    {first}\n"
        f"    or continue without it:  agent6 resume {session_id} --force"
    )
