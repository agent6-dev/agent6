# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Clone a lane's workspace and import its finished branch.

Git plumbing over `agent6.vcs.git.ops`, with no model, UI or process spawning; `app.lanes`
drives a `LaneSpawner` over it.
"""

from __future__ import annotations

import pathlib

from agent6 import errors, lane_records
from agent6.sessions import lineage, process
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read


class ChildRepoError(Exception):
    """A lane clone or import failed."""


def clone_worktree(origin: pathlib.Path, dest: pathlib.Path) -> None:
    """Clone the origin repo into a lane's own worktree.

    Args:
        origin: The repo to clone.
        dest: The worktree's path, which must not exist.

    Raises:
        ChildRepoError: When the clone fails.
    """
    try:
        git_ops.clone_repo(origin, dest)
    except errors.GitError as exc:
        raise ChildRepoError(f"clone {origin} -> {dest} failed: {exc}") from exc


def import_run(
    origin: pathlib.Path,
    lane_repo: pathlib.Path,
    branch: str,
    lane_session_dir: pathlib.Path,
    *,
    model_control: bool = False,
) -> lane_records.Imported:
    """Land a finished lane's branch in the origin; its record is there already.

    A model-controlled lane lands HEAD on its chain ref and keeps a free branch name.
    A harness-controlled lane without a branch lands nothing.

    Args:
        origin: The parent's repo.
        lane_repo: The lane's workspace clone.
        branch: The lane's branch.
        lane_session_dir: The lane's run dir, under the origin's state.
        model_control: Import HEAD instead of a harness-owned branch.

    Returns:
        The run dir and the ref the work landed in, with its tip; a lane that committed
        nothing lands no ref.

    Raises:
        ChildRepoError: When the branch already exists in the origin, a descendant is still
            running, or a fetch fails.
    """
    if not model_control and git_read.branch_exists(origin, branch):
        raise ChildRepoError(f"branch {branch!r} already exists in {origin}")
    fetch_retained_refs(origin, lane_repo, lane_session_dir)
    ref = tip = ""
    if model_control:
        try:
            head = git_read.status(lane_repo)
            chain = git_read.chain_ref_for(lane_session_dir.name)
            git_ops.fetch_branch(origin, lane_repo, f"HEAD:{chain}")
            if head.branch and not git_read.branch_exists(origin, head.branch):
                git_ops.create_branch_at(origin, head.branch, head.head_sha)
        except errors.GitError as exc:
            raise ChildRepoError(f"import HEAD from {lane_repo} failed: {exc}") from exc
        ref, tip = chain, head.head_sha
    elif git_read.branch_exists(lane_repo, branch):
        try:
            git_ops.fetch_branch(origin, lane_repo, f"{branch}:{branch}")
        except errors.GitError as exc:
            raise ChildRepoError(f"fetch {branch!r} from {lane_repo} failed: {exc}") from exc
        ref, tip = branch, git_read.branch_tip_sha(origin, branch) or ""
    return lane_records.Imported(lane_session_dir, ref, tip)


def fetch_retained_refs(
    origin: pathlib.Path, lane_repo: pathlib.Path, lane_session_dir: pathlib.Path
) -> None:
    """Fetch a lane's and its descendants' session-owned refs before the lane's clone goes.

    Hidden target refs keep evidence reachable after the clone and its object store go.

    Args:
        origin: The receiving repository.
        lane_repo: The lane's clone, holding the retained targets.
        lane_session_dir: The lane's record.

    Raises:
        ChildRepoError: A descendant is live or a fetch fails.
    """
    descendants = lineage.descendants_of(lane_session_dir)
    if any(process.worker_is_alive(child) for child in descendants):
        raise ChildRepoError("a descendant is still running; its worker clone must be retained")
    try:
        for session in (lane_session_dir, *descendants):
            for retained in git_read.session_refs(lane_repo, session.name):
                git_ops.fetch_branch(origin, lane_repo, f"{retained}:{retained}")
    except errors.GitError as exc:
        raise ChildRepoError(
            f"import retained session targets from {lane_repo} failed: {exc}"
        ) from exc
