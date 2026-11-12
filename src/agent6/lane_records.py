# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The lane records every fan-out shares: what a lane runs, what it returned, how it joined.

`run --parallel`, the `/parallel` steer and the import plumbing read and write these; the
code that clones, spawns and merges lives in `harness.subrun`, `app.lanes` and
`harness.parallel_dispatch`.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Literal, Protocol

from agent6 import kinds

# Who asked for a dispatch: the operator's steer, the model's tool call, or a review panel.
DispatchSource = Literal["operator", "model", "panel"]
# How one lane's work joined the coordinator's chain.
JoinStatus = Literal["joined", "no_changes", "conflict", "failed"]


@dataclasses.dataclass(frozen=True, slots=True)
class LaneTask:
    """Name one lane to dispatch, as the coordinator expands a `/parallel` segment.

    Attributes:
        task: The lane's task text.
        model: The lane's selector (a role name or `[provider/]model`), None for the
            worker's own settings.
        pins: The operator's pins, passed beside the task so the lane's manifest task stays clean.
    """

    task: str
    model: str | None
    pins: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class LaneSpec:
    """Name one lane to run.

    Attributes:
        lane: The lane's index in its group.
        session_id: The lane's session id.
        workdir: The lane's workspace clone.
        settings: The lane's route, effort and temperature, resolved from its selector; None
            for the worker's own, which the lane's config carries unchanged.
    """

    lane: int
    session_id: str
    workdir: pathlib.Path
    settings: kinds.ModelSettings | None


@dataclasses.dataclass(frozen=True, slots=True)
class Imported:
    """What a lane's import landed in the origin.

    Attributes:
        session_dir: The lane's run dir under the origin's state.
        ref: The ref the import fetched the lane's work into; "" when only the record came.
        tip: That ref's commit; "" with no ref.
    """

    session_dir: pathlib.Path
    ref: str = ""
    tip: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class LaneResult:
    """Record the outcome of one lane.

    Attributes:
        spec: The lane run.
        session_dir: Where its run state lives.
        branch: The branch the lane was given; a name, not proof that work was imported.
        ok: Whether it ended with a result and its record was imported.
        error: Why it failed, "" on success.
        imported_ref: The ref the import landed the lane's work in; "" when none.
        imported_tip: That ref's commit; "" when none.
        has_commits: Whether the imported tip holds commits beyond the lane's base.
    """

    spec: LaneSpec
    session_dir: pathlib.Path
    branch: str
    ok: bool
    error: str
    imported_ref: str = ""
    imported_tip: str = ""
    has_commits: bool = False


class LaneSpawner(Protocol):
    """Run one lane to its end."""

    def __call__(self, spec: LaneSpec, task: str) -> LaneResult:
        """Run the lane on the task and return its result."""
        ...


class GroupLaneSpawner(Protocol):
    """Run a group of sibling lanes to their end and import each into the coordinator's repo.

    One call clones, spawns, awaits and imports every lane; `app.lanes` owns that machinery
    and the coordinator supplies the tasks and the group id (`p<seq>`). A lane that failed to
    start, outlived the teardown or was refused at import has `ok` False and leaves the
    coordinator's repo untouched.
    """

    def __call__(
        self, lanes: list[LaneTask], group: str, *, at: str | None = None
    ) -> list[LaneResult]:
        """Run the lanes as one group, from the commit `at` when given, in dispatch order."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class LaneJoin:
    """One lane's outcome in a dispatch.

    Attributes:
        session_id: The lane's session.
        branch: The lane's branch.
        status: `joined` (merged, `sha` set), `no_changes` (ended with a result and no commit
            beyond its base; nothing to merge), `conflict` (imported, the merge conflicted,
            the ref exists locally) or `failed` (no result or nothing importable).
        sha: The merge commit, or "".
        detail: The failure's reason, or "".
    """

    session_id: str
    branch: str
    status: JoinStatus
    sha: str
    detail: str


@dataclasses.dataclass(frozen=True, slots=True)
class DispatchResult:
    """What a dispatch did, for the caller to deliver as it sees fit.

    Attributes:
        group: The group id the lanes were stamped with; "" when nothing was dispatched.
        dispatched: Whether lanes ran; False for a refusal before any clone.
        lanes: Each lane's join, in dispatch order.
        message: The one notice summarizing the outcome, or the refusal.
        base: The commit the lanes were cut from; "" when none ran.
    """

    group: str
    dispatched: bool
    lanes: tuple[LaneJoin, ...]
    message: str
    base: str = ""
