# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Clone a lane's workspace and import its finished branch and run dir.

Git plumbing over `agent6.git_ops`, with no model, UI or process spawning; `app.parallel`
drives a `LaneSpawner` over it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent6.git_ops import GitError, branch_exists, clone_repo, fetch_branch
from agent6.kinds import ModelRoute
from agent6.paths import mkdir_for_real_user
from agent6.sessions.layout import bucket_dir


class SubrunError(Exception):
    """A lane clone or import failed."""


@dataclass(frozen=True, slots=True)
class LaneSpec:
    """Name one lane to run.

    Attributes:
        lane: The lane's index in its group.
        session_id: The lane's session id.
        workdir: The lane's workspace clone.
        route: The lane's model route, None for the worker's own.
    """

    lane: int
    session_id: str
    workdir: Path
    route: ModelRoute | None


@dataclass(frozen=True, slots=True)
class LaneResult:
    """Record the outcome of one lane.

    Attributes:
        spec: The lane run.
        session_dir: Where its run state lives.
        branch: Its branch.
        ok: Whether it succeeded.
        error: Why it failed, "" on success.
    """

    spec: LaneSpec
    session_dir: Path
    branch: str
    ok: bool
    error: str


@dataclass(frozen=True, slots=True)
class LaneTask:
    """Name one lane to dispatch, as the coordinator expands a `/parallel` segment.

    Attributes:
        task: The lane's task text.
        model: The lane's `[provider/]model` text, None for the worker's own route.
        pins: The operator's pins, passed beside the task so the lane's manifest task stays clean.
    """

    task: str
    model: str | None
    pins: tuple[str, ...] = ()


class LaneSpawner(Protocol):
    """Run one lane to its end."""

    def __call__(self, spec: LaneSpec, task: str) -> LaneResult:
        """Run the lane on the task and return its result."""
        ...


class GroupLaneSpawner(Protocol):
    """Run a group of sibling lanes to their end and import each into the coordinator's repo.

    One call clones, spawns, awaits and imports every lane; `app.parallel` owns that machinery
    and the coordinator supplies the tasks and the group id (`p<seq>`). A lane that failed to
    start, outlived the teardown or was refused at import has `ok` False and leaves the
    coordinator's repo untouched.
    """

    def __call__(
        self, lanes: list[LaneTask], group: str, *, at: str | None = None
    ) -> list[LaneResult]:
        """Run the lanes as one group, from the commit `at` when given, in dispatch order."""
        ...


def clone_workspace(origin: Path, dest: Path) -> None:
    """Clone the origin repo into a disposable lane workspace.

    Args:
        origin: The repo to clone.
        dest: The workspace path, which must not exist.

    Raises:
        SubrunError: When the clone fails.
    """
    try:
        clone_repo(origin, dest)
    except GitError as exc:
        raise SubrunError(f"clone {origin} -> {dest} failed: {exc}") from exc


def import_run(
    origin: Path,
    lane_repo: Path,
    branch: str,
    lane_session_dir: Path,
    origin_state: Path,
) -> Path:
    """Land a finished lane's branch in the origin and move its run dir under the origin's state.

    Both refusals are checked before the fetch or the move, so a refusal touches neither. A lane
    that never committed has no branch to land; its record imports all the same.

    Args:
        origin: The coordinator's repo.
        lane_repo: The lane's workspace clone.
        branch: The lane's branch.
        lane_session_dir: The lane's run dir.
        origin_state: The origin's state dir.

    Returns:
        The imported run dir.

    Raises:
        SubrunError: When the branch or run dir already exists in the origin, or the fetch fails.
    """
    if branch_exists(origin, branch):
        raise SubrunError(f"branch {branch!r} already exists in {origin}")
    dest_session_dir = bucket_dir(origin_state, "runs") / lane_session_dir.name
    if dest_session_dir.exists():
        raise SubrunError(f"run dir already exists: {dest_session_dir}")
    if branch_exists(lane_repo, branch):
        try:
            fetch_branch(origin, lane_repo, f"{branch}:{branch}")
        except GitError as exc:
            raise SubrunError(f"fetch {branch!r} from {lane_repo} failed: {exc}") from exc
    mkdir_for_real_user(dest_session_dir.parent)
    shutil.move(str(lane_session_dir), str(dest_session_dir))
    return dest_session_dir
