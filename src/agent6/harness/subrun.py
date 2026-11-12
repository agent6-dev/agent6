# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Clone a lane's workspace and import its finished branch and run dir.

Git plumbing over `agent6.vcs.git_ops`, with no model, UI or process spawning; `app.parallel`
drives a `LaneSpawner` over it.
"""

from __future__ import annotations

import pathlib
import shutil

from agent6 import errors, lane_records, paths
from agent6.sessions import layout
from agent6.vcs import git_ops, git_read


class SubrunError(Exception):
    """A lane clone or import failed."""


def clone_workspace(origin: pathlib.Path, dest: pathlib.Path) -> None:
    """Clone the origin repo into a disposable lane workspace.

    Args:
        origin: The repo to clone.
        dest: The workspace path, which must not exist.

    Raises:
        SubrunError: When the clone fails.
    """
    try:
        git_ops.clone_repo(origin, dest)
    except errors.GitError as exc:
        raise SubrunError(f"clone {origin} -> {dest} failed: {exc}") from exc


def import_run(
    origin: pathlib.Path,
    lane_repo: pathlib.Path,
    branch: str,
    lane_session_dir: pathlib.Path,
    origin_state: pathlib.Path,
    *,
    model_control: bool = False,
) -> lane_records.Imported:
    """Land a finished lane's branch in the origin and move its run dir under the origin's state.

    A model-controlled lane lands HEAD on its chain ref and keeps a free branch name.
    A harness-controlled lane without a branch imports only its record.

    Args:
        origin: The coordinator's repo.
        lane_repo: The lane's workspace clone.
        branch: The lane's branch.
        lane_session_dir: The lane's run dir.
        origin_state: The origin's state dir.
        model_control: Import HEAD instead of a harness-owned branch.

    Returns:
        The imported run dir and the ref the work landed in, with its tip; a lane that
        committed nothing imports its record alone.

    Raises:
        SubrunError: When the branch or run dir already exists in the origin, or the fetch fails.
    """
    if not model_control and git_read.branch_exists(origin, branch):
        raise SubrunError(f"branch {branch!r} already exists in {origin}")
    dest_session_dir = layout.bucket_dir(origin_state, "runs") / lane_session_dir.name
    if dest_session_dir.exists():
        raise SubrunError(f"run dir already exists: {dest_session_dir}")
    ref = tip = ""
    if model_control:
        try:
            head = git_read.status(lane_repo)
            chain = git_read.chain_ref_for(lane_session_dir.name)
            git_ops.fetch_branch(origin, lane_repo, f"HEAD:{chain}")
            if head.branch and not git_read.branch_exists(origin, head.branch):
                git_ops.create_branch_at(origin, head.branch, head.head_sha)
        except errors.GitError as exc:
            raise SubrunError(f"import HEAD from {lane_repo} failed: {exc}") from exc
        ref, tip = chain, head.head_sha
    elif git_read.branch_exists(lane_repo, branch):
        try:
            git_ops.fetch_branch(origin, lane_repo, f"{branch}:{branch}")
        except errors.GitError as exc:
            raise SubrunError(f"fetch {branch!r} from {lane_repo} failed: {exc}") from exc
        ref, tip = branch, git_read.branch_tip_sha(origin, branch) or ""
    paths.mkdir_for_real_user(dest_session_dir.parent)
    shutil.move(str(lane_session_dir), str(dest_session_dir))
    return lane_records.Imported(dest_session_dir, ref, tip)
