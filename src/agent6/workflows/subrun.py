# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Subordinate-run mechanics: clone a disposable lane workspace, import a
finished lane's branch and run dir back into the origin, and join a
subordinate branch into the current branch.

Pure git plumbing over `agent6.git_ops` -- no LLM, no UI, no process
spawning. Later parallel-runs work drives a `LaneSpawner` over these.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent6.git_ops import GitError, branch_exists, clone_repo, fetch_branch, merge_branch


class SubrunError(Exception):
    """Subordinate-run mechanics (clone/import/join) failed."""


@dataclass(frozen=True, slots=True)
class LaneSpec:
    """One subordinate lane to run: its own workspace clone, run id, and
    model (None = the configured worker model)."""

    lane: int
    run_id: str
    workdir: Path
    model: str | None


@dataclass(frozen=True, slots=True)
class LaneResult:
    """Outcome of running a lane: where its run state lives, its branch, and
    whether it succeeded (`error` set on failure)."""

    spec: LaneSpec
    run_dir: Path
    branch: str
    ok: bool
    error: str


@dataclass(frozen=True, slots=True)
class LaneTask:
    """One lane to dispatch: the task text and an optional per-lane model
    (`None` = the configured worker model). The coordinator expands each
    `/parallel` segment into these (spec=3 -> three, one per model in a list)."""

    task: str
    model: str | None


class LaneSpawner(Protocol):
    def __call__(self, spec: LaneSpec, task: str) -> LaneResult: ...


class GroupLaneSpawner(Protocol):
    """Dispatch a sibling group of subordinate lanes and return their results in
    dispatch order (one `LaneResult` per `LaneTask` in *lanes*).

    One call is synchronous-complete: clone + spawn each lane on its own model,
    await them all to terminal, and import each finished branch + run dir into the
    coordinator's repo. All spawn/await/import machinery is the ui side's (see
    `ui/cli/parallel.py`); the coordinator loop supplies only the per-lane tasks
    and a *group* id (`p<seq>`), so `workflows` never imports ui. On a lane that
    failed to start, is still running at teardown, or whose import was refused,
    that lane's `LaneResult.ok` is False and the coordinator's repo is left
    untouched for it."""

    def __call__(self, lanes: list[LaneTask], group: str) -> list[LaneResult]: ...


def clone_workspace(origin: Path, dest: Path) -> None:
    """Clone *origin* into *dest*, a disposable lane workspace.

    Plain `git clone` on a filesystem path (git's local-clone optimization:
    hardlinks same-filesystem, copies cross-device). Raises SubrunError on
    failure, e.g. *dest* already exists or *origin* is not a repo.
    """
    try:
        clone_repo(origin, dest)
    except GitError as exc:
        raise SubrunError(f"clone {origin} -> {dest} failed: {exc}") from exc


def import_run(
    origin: Path,
    lane_repo: Path,
    branch: str,
    lane_run_dir: Path,
    origin_state: Path,
) -> Path:
    """Land a finished lane's *branch* in *origin* and move `lane_run_dir`
    under `<origin_state>/runs/`. Returns the imported run dir.

    Refuses (SubrunError) to overwrite an existing branch in *origin* or an
    existing run dir at the destination -- checked before either the fetch or
    the move, so a refusal touches neither.
    """
    if branch_exists(origin, branch):
        raise SubrunError(f"branch {branch!r} already exists in {origin}")
    dest_run_dir = origin_state / "runs" / lane_run_dir.name
    if dest_run_dir.exists():
        raise SubrunError(f"run dir already exists: {dest_run_dir}")
    try:
        fetch_branch(origin, lane_repo, f"{branch}:{branch}")
    except GitError as exc:
        raise SubrunError(f"fetch {branch!r} from {lane_repo} failed: {exc}") from exc
    dest_run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(lane_run_dir), str(dest_run_dir))
    return dest_run_dir


def join_branch(workspace: Path, branch: str) -> str | None:
    """Merge *branch* into the current branch of *workspace*.

    Returns the merged sha, or None on conflict: the merge is aborted
    (`git merge --abort`), leaving the workspace clean.
    """
    try:
        result = merge_branch(workspace, branch)
    except GitError as exc:
        raise SubrunError(f"merge {branch!r} into {workspace} failed: {exc}") from exc
    return None if result.conflicted else result.merged_sha
