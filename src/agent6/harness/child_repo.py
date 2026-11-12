# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Clone a child's workspace and import its finished branch.

Git plumbing over `agent6.vcs.git.ops`, with no model, UI or process spawning; `app.children`
drives a `ChildSpawner` over it.
"""

from __future__ import annotations

import pathlib

from agent6 import child_records, errors
from agent6.sessions import lineage, process
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read


class ChildRepoError(Exception):
    """A child clone or import failed."""


def clone_worktree(origin: pathlib.Path, dest: pathlib.Path) -> None:
    """Clone the origin repo into a child's own worktree.

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
    child_worktree: pathlib.Path,
    branch: str,
    child_session_dir: pathlib.Path,
    *,
    model_control: bool = False,
) -> child_records.Imported:
    """Land a finished child's branch in the origin; its record is there already.

    A model-controlled child lands HEAD on its chain ref and keeps a free branch name.
    A harness-controlled child without a branch lands nothing.

    Args:
        origin: The parent's repo.
        child_worktree: The child's workspace clone.
        branch: The child's branch.
        child_session_dir: The child's session dir, under the origin's state.
        model_control: Import HEAD instead of a harness-owned branch.

    Returns:
        The session dir and the ref the work landed in, with its tip; a child that committed
        nothing lands no ref.

    Raises:
        ChildRepoError: When the branch already exists in the origin, a descendant is still
            running, or a fetch fails.
    """
    if not model_control and git_read.branch_exists(origin, branch):
        raise ChildRepoError(f"branch {branch!r} already exists in {origin}")
    fetch_retained_refs(origin, child_worktree, child_session_dir)
    ref = tip = ""
    if model_control:
        try:
            head = git_read.status(child_worktree)
            chain = git_read.chain_ref_for(child_session_dir.name)
            git_ops.fetch_branch(origin, child_worktree, f"HEAD:{chain}")
            if head.branch and not git_read.branch_exists(origin, head.branch):
                git_ops.create_branch_at(origin, head.branch, head.head_sha)
        except errors.GitError as exc:
            raise ChildRepoError(f"import HEAD from {child_worktree} failed: {exc}") from exc
        ref, tip = chain, head.head_sha
    elif git_read.branch_exists(child_worktree, branch):
        try:
            git_ops.fetch_branch(origin, child_worktree, f"{branch}:{branch}")
        except errors.GitError as exc:
            raise ChildRepoError(f"fetch {branch!r} from {child_worktree} failed: {exc}") from exc
        ref, tip = branch, git_read.branch_tip_sha(origin, branch) or ""
    return child_records.Imported(child_session_dir, ref, tip)


def fetch_retained_refs(
    origin: pathlib.Path, child_worktree: pathlib.Path, child_session_dir: pathlib.Path
) -> None:
    """Fetch a child's and its descendants' session-owned refs before the child's clone goes.

    Hidden target refs keep evidence reachable after the clone and its object store go.

    Args:
        origin: The receiving repository.
        child_worktree: The child's clone, holding the retained targets.
        child_session_dir: The child's record.

    Raises:
        ChildRepoError: A descendant is live or a fetch fails.
    """
    descendants = lineage.descendants_of(child_session_dir)
    if any(process.worker_is_alive(child) for child in descendants):
        raise ChildRepoError("a descendant is still running; its worker clone must be retained")
    try:
        for session in (child_session_dir, *descendants):
            for retained in git_read.session_refs(child_worktree, session.name):
                git_ops.fetch_branch(origin, child_worktree, f"{retained}:{retained}")
    except errors.GitError as exc:
        raise ChildRepoError(
            f"import retained session targets from {child_worktree} failed: {exc}"
        ) from exc
