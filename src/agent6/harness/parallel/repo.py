# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Clone a child's workspace and import its finished branch.

Git plumbing over `agent6.vcs.git.ops`, with no model, UI or process spawning;
`app.parallel.children` drives a `ChildSpawner` over it.
"""

from __future__ import annotations

import pathlib

from agent6 import child_records, errors, kinds
from agent6.config import Config
from agent6.sandbox import detect, jail
from agent6.sessions import lineage, process
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import policy as tools_policy
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


def sender_jail(
    child_worktree: pathlib.Path, child_session_dir: pathlib.Path, config: Config
) -> kinds.JailPolicy:
    """Return the jail the clone's side of a fetch runs under, no weaker than its child's.

    Args:
        child_worktree: The child's workspace clone.
        child_session_dir: The child's session dir, whose manifest records the level its
            commands last ran under.
        config: The config of the process that fetches.

    Returns:
        The policy a command in the clone gets under that config.

    Raises:
        IsolationUnavailableError: The host cannot provide the configured level.
        JailUnavailableError: The jail cannot be built, or the level this process resolves
            is weaker than the recorded one.
    """
    try:
        ran_under = sessions_manifest.read_manifest(child_session_dir).policy.isolation
    except sessions_manifest.ManifestError:
        ran_under = ""  # no readable record names a level
    return tools_policy.workspace_jail(child_worktree, config, ran_under=ran_under)


def import_run(
    origin: pathlib.Path,
    child_worktree: pathlib.Path,
    branch: str,
    child_session_dir: pathlib.Path,
    config: Config,
    *,
    model_control: bool = False,
    continuing: bool = False,
) -> child_records.Imported:
    """Import a finished child's branch in the origin; its record is there already.

    A model-controlled child imports HEAD on its chain ref and keeps a free branch name.
    A harness-controlled child without a branch imports nothing. The clone's side of every
    fetch runs under the jail a command in the clone gets.

    Args:
        origin: The parent's repo.
        child_worktree: The child's workspace clone.
        branch: The child's branch.
        child_session_dir: The child's session dir, under the origin's state.
        config: The origin's config, which the child's own is a snapshot of.
        model_control: Import HEAD instead of a harness-owned branch.
        continuing: Allow a fast-forward of an existing child ref.

    Returns:
        The session dir and the ref the work was imported into, with its tip; a child that committed
        nothing imports no ref.

    Raises:
        ChildRepoError: When the branch already exists in the origin, a descendant is still
            running, the clone's jail cannot be built or would be weaker than the child's own,
            or a fetch fails.
    """
    if not model_control and not continuing and git_read.branch_exists(origin, branch):
        raise ChildRepoError(f"branch {branch!r} already exists in {origin}")
    try:
        sender = sender_jail(child_worktree, child_session_dir, config)
    except (detect.IsolationUnavailableError, jail.JailUnavailableError) as exc:
        raise ChildRepoError(f"no jail to read {child_worktree} in: {exc}") from exc
    fetch_retained_refs(origin, child_worktree, child_session_dir, sender)
    ref = tip = ""
    if model_control:
        try:
            head = git_read.status(child_worktree)
            chain = git_read.chain_ref_for(child_session_dir.name)
            git_ops.fetch_branch(origin, child_worktree, f"HEAD:{chain}", sender=sender)
            if head.branch and not git_read.branch_exists(origin, head.branch):
                git_ops.create_branch_at(origin, head.branch, head.head_sha)
        except errors.GitError as exc:
            raise ChildRepoError(f"import HEAD from {child_worktree} failed: {exc}") from exc
        ref, tip = chain, head.head_sha
    elif git_read.branch_exists(child_worktree, branch):
        try:
            git_ops.fetch_branch(origin, child_worktree, f"{branch}:{branch}", sender=sender)
        except errors.GitError as exc:
            raise ChildRepoError(f"fetch {branch!r} from {child_worktree} failed: {exc}") from exc
        ref, tip = branch, git_read.branch_tip_sha(origin, branch) or ""
    return child_records.Imported(child_session_dir, ref, tip)


def fetch_retained_refs(
    origin: pathlib.Path,
    child_worktree: pathlib.Path,
    child_session_dir: pathlib.Path,
    sender: kinds.JailPolicy,
) -> None:
    """Fetch a child's and its descendants' session-owned refs before the child's clone goes.

    Hidden target refs keep evidence reachable after the clone and its object store go.

    Args:
        origin: The receiving repository.
        child_worktree: The child's clone, holding the retained targets.
        child_session_dir: The child's record.
        sender: The policy a command in the clone runs under.

    Raises:
        ChildRepoError: A descendant is live or a fetch fails.
    """
    descendants = lineage.descendants_of(child_session_dir)
    if any(process.worker_is_alive(child) for child in descendants):
        raise ChildRepoError("a descendant is still running; its worker clone must be retained")
    try:
        for session in (child_session_dir, *descendants):
            for retained in git_read.session_refs(child_worktree, session.name):
                git_ops.fetch_branch(
                    origin, child_worktree, f"{retained}:{retained}", sender=sender
                )
    except errors.GitError as exc:
        raise ChildRepoError(
            f"import retained session targets from {child_worktree} failed: {exc}"
        ) from exc
