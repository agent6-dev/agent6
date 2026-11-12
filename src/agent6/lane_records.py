# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The lane records every group shares: what a lane runs, what it returned, how it joined.

`run --parallel`, the `/parallel` steer and the import plumbing read and write these; the
code that clones, spawns and merges lives in `harness.child_repo`, `app.lanes` and
`harness.parallel_dispatch`.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Sequence
from typing import Literal, Protocol

from agent6 import budget, kinds

# Who asked for a dispatch: the operator's steer, the model's tool call, or a review panel.
DispatchSource = Literal["operator", "model", "panel"]
# How a dispatch lands its lanes: each merged in order, none (a panel's records), or the
# best-ranked one alone.
JoinPolicy = Literal["merge", "record_only", "rank"]
# How one lane's work joined the parent's chain.
JoinStatus = Literal["joined", "no_changes", "conflict", "failed", "ranked"]


@dataclasses.dataclass(frozen=True, slots=True)
class LaneLimits:
    """The ceiling a lane runs under.

    Attributes:
        max_turns: The turn cap the lane's `harness.max_turns` is clamped to; None keeps the
            config's value.
        deadline_at: The wall-clock second (epoch) the lane must have ended by; None for none.
            Wall-clock, since it crosses the process boundary into the lane.
    """

    max_turns: int | None = None
    deadline_at: float | None = None

    def clamp_turns(self, configured: int) -> int:
        """Return the lane's turn cap: the smaller of the configured value and the cap.

        Args:
            configured: `harness.max_turns`, -1 for unlimited.

        Returns:
            The configured value under the cap; a configured -1 cannot lift the cap.
        """
        if self.max_turns is None:
            return configured
        if configured < 0:
            return self.max_turns
        return min(configured, self.max_turns)


@dataclasses.dataclass(frozen=True, slots=True)
class DispatchAuthority:
    """Separate work dispatch from configured harness panels and bound descendants.

    Attributes:
        work: The model and steer may dispatch work.
        panels: The harness may submit bounded read-only panel leaves.
        deadline_at: The owning lane's deadline, inherited by every descendant.
    """

    work: bool = True
    panels: bool = False
    deadline_at: float | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class LaneTask:
    """Name one lane to dispatch, as the parent expands a `/parallel` segment.

    Attributes:
        task: The lane's task text.
        model: The lane's selector (a role name or `[provider/]model`), None for the
            worker's own settings.
        pins: The operator's pins, passed beside the task so the lane's manifest task stays clean.
        allowance: The lane's share of the parent's allowance; None for the config's caps.
        limits: The lane's turn cap and deadline; None for the config's own.
        contract: A harness-owned bounded conversation; None for work.
        settings: Resolved panel-seat settings; None resolves the model selector.
    """

    task: str
    model: str | None
    pins: tuple[str, ...] = ()
    allowance: budget.BudgetAllowance | None = None
    limits: LaneLimits | None = None
    contract: kinds.ConversationContract | None = None
    settings: kinds.ModelSettings | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class LaneSpec:
    """Name one lane to run.

    Attributes:
        lane: The lane's index in its group.
        session_id: The lane's session id.
        worktree: The lane's workspace clone.
        settings: The lane's route, effort and temperature, resolved from its selector; None
            for the worker's own, which the lane's config carries unchanged.
        mode: The lane's session mode, which names its bucket under the origin's state.
    """

    lane: int
    session_id: str
    worktree: pathlib.Path
    settings: kinds.ModelSettings | None
    mode: kinds.LoopMode = "run"


@dataclasses.dataclass(frozen=True, slots=True)
class Imported:
    """What a lane's import landed in the origin.

    Attributes:
        session_dir: The lane's session dir under the origin's state.
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
        pid: The lane process the spawn started, 0 when none did; a child that never
            reported its session dir is stopped through it.
        usage: What the lane used, read from its imported record; None when no record
            came back.
        report: What the lane said at its end, its finish summary; "" when it ended another
            way.
        stopped_by: What ended the lane when its parent stopped it (its deadline, the
            run's budget); "" when it ended by itself or by the operator.
    """

    spec: LaneSpec
    session_dir: pathlib.Path
    branch: str
    ok: bool
    error: str
    imported_ref: str = ""
    imported_tip: str = ""
    has_commits: bool = False
    pid: int = 0
    usage: budget.UsageSettlement | None = None
    report: str = ""
    stopped_by: str = ""


class LaneSpawner(Protocol):
    """Run one lane to its end."""

    def __call__(self, spec: LaneSpec, task: str) -> LaneResult:
        """Run the lane on the task and return its result."""
        ...


class GroupLaneSpawner(Protocol):
    """Run a group of sibling lanes to their end and import each into the parent's repo.

    One call clones, spawns, awaits and imports every lane; `app.lanes` owns that machinery
    and the parent supplies the tasks and the group id (`p<seq>`). A lane that failed to
    start, outlived the teardown or was refused at import has `ok` False and leaves the
    parent's repo untouched.
    """

    def __call__(
        self, lanes: list[LaneTask], group: str, *, at: str | None = None
    ) -> list[LaneResult]:
        """Run the lanes as one group, from the commit `at` when given, in dispatch order."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class Ranking:
    """The order a rank join lands by.

    Attributes:
        order: The ranked lanes' session ids, best first.
        ranked_by: `judge` when a judge's ranking ordered them, else `mechanical`.
        rationale: The judges' words; "" for the mechanical order.
        winner: The lane the join lands, or "" when the order crowns none: every gate ran
            red, or a lane that did not verify was ranked over one that did.
    """

    order: tuple[str, ...]
    ranked_by: Literal["judge", "mechanical"] = "mechanical"
    rationale: str = ""
    winner: str = ""


class LaneRanker(Protocol):
    """Rank a dispatch's finished lanes, best first."""

    def __call__(self, results: Sequence[LaneResult], *, group: str, base: str) -> Ranking:
        """Return the order of a group's lanes that ended with a result, cut from `base`."""
        ...


class LaneRecoverer(Protocol):
    """Find an interrupted group's lanes, stop what still runs and import what finished."""

    def __call__(self, group: str, children: Sequence[str]) -> list[LaneResult]:
        """Return a result per lane found; a lane with no session dir is missing."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class LaneJoin:
    """One lane's outcome in a dispatch.

    Attributes:
        session_id: The lane's session.
        branch: The lane's branch.
        status: `joined` (merged, `sha` set), `no_changes` (ended with a result and no commit
            beyond its base; nothing to merge), `conflict` (imported, the merge conflicted,
            the ref exists locally), `failed` (no result or nothing importable) or
            `ranked` (a rank join placed the lane and did not join it: another lane landed,
            or the run has no chain to join onto; the branch holds this one's work).
        sha: The merge commit, or "".
        detail: The failure's reason; a ranked lane's place (`2 of 3`); for a lane that
            joined or had nothing to join, what stopped it when its parent did; else "".
        report: What the lane said at its end, its finish summary; "" when it gave none.
    """

    session_id: str
    branch: str
    status: JoinStatus
    sha: str
    detail: str
    report: str = ""


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
