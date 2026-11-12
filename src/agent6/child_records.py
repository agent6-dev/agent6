# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The child records every group shares: what a child runs, what it returned, how it joined.

`run --parallel`, the `/parallel` steer and the import plumbing read and write these; the
code that clones, spawns and merges lives in `harness.child_repo`, `app.children` and
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
# How a dispatch lands its children: each merged in order, none (a panel's records), or the
# best-ranked one alone.
JoinPolicy = Literal["merge", "record_only", "rank"]
# How one child's work joined the parent's chain.
JoinStatus = Literal["joined", "no_changes", "conflict", "failed", "ranked"]


@dataclasses.dataclass(frozen=True, slots=True)
class ChildLimits:
    """The ceiling a child runs under.

    Attributes:
        max_turns: The turn cap the child's `harness.max_turns` is clamped to; None keeps the
            config's value.
        deadline_at: The wall-clock second (epoch) the child must have ended by; None for none.
            Wall-clock, since it crosses the process boundary into the child.
    """

    max_turns: int | None = None
    deadline_at: float | None = None

    def clamp_turns(self, configured: int) -> int:
        """Return the child's turn cap: the smaller of the configured value and the cap.

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
        panels: The harness may dispatch a panel's reviewers.
        deadline_at: The owning child's deadline, inherited by every descendant.
    """

    work: bool = True
    panels: bool = False
    deadline_at: float | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ChildTask:
    """Name one child to dispatch, as the parent expands a `/parallel` segment.

    Attributes:
        task: The child's task text.
        model: The child's selector (a role name or `[provider/]model`), None for the
            worker's own settings.
        pins: The operator's pins, passed beside the task so the child's manifest task stays clean.
        allowance: The child's share of the parent's allowance; None for the config's caps.
        limits: The child's turn cap and deadline; None for the config's own.
        contract: A harness-owned bounded conversation; None for work.
        settings: Resolved panel-reviewer settings; None resolves the model selector.
    """

    task: str
    model: str | None
    pins: tuple[str, ...] = ()
    allowance: budget.BudgetAllowance | None = None
    limits: ChildLimits | None = None
    contract: kinds.ConversationContract | None = None
    settings: kinds.ModelSettings | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ChildSpec:
    """Name one child to run.

    Attributes:
        number: The child's index in its group.
        session_id: The child's session id.
        worktree: The child's workspace clone.
        settings: The child's route, effort and temperature, resolved from its selector; None
            for the worker's own, which the child's config carries unchanged.
        mode: The child's session mode, which names its bucket under the origin's state.
    """

    number: int
    session_id: str
    worktree: pathlib.Path
    settings: kinds.ModelSettings | None
    mode: kinds.LoopMode = "run"


@dataclasses.dataclass(frozen=True, slots=True)
class Imported:
    """What a child's import landed in the origin.

    Attributes:
        session_dir: The child's session dir under the origin's state.
        ref: The ref the import fetched the child's work into; "" when only the record came.
        tip: That ref's commit; "" with no ref.
    """

    session_dir: pathlib.Path
    ref: str = ""
    tip: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class ChildResult:
    """Record the outcome of one child.

    Attributes:
        spec: The child run.
        session_dir: Where its run state lives.
        branch: The branch the child was given; a name, not proof that work was imported.
        ok: Whether it ended with a result and its record was imported.
        error: Why it failed, "" on success.
        imported_ref: The ref the import landed the child's work in; "" when none.
        imported_tip: That ref's commit; "" when none.
        has_commits: Whether the imported tip holds commits beyond the child's base.
        pid: The child process the spawn started, 0 when none did; a child that never
            reported its session dir is stopped through it.
        usage: What the child used, read from its imported record; None when no record
            came back.
        report: What the child said at its end, its finish summary; "" when it ended another
            way.
        stopped_by: What ended the child when its parent stopped it (its deadline, the
            run's budget); "" when it ended by itself or by the operator.
    """

    spec: ChildSpec
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


class ChildSpawner(Protocol):
    """Run one child to its end."""

    def __call__(self, spec: ChildSpec, task: str) -> ChildResult:
        """Run the child on the task and return its result."""
        ...


class GroupChildSpawner(Protocol):
    """Run a group of sibling children to their end and import each into the parent's repo.

    One call clones, spawns, awaits and imports every child; `app.children` owns that machinery
    and the parent supplies the tasks and the group id (`p<seq>`). A child that failed to
    start, outlived the teardown or was refused at import has `ok` False and leaves the
    parent's repo untouched.
    """

    def __call__(
        self, children: list[ChildTask], group: str, *, at: str | None = None
    ) -> list[ChildResult]:
        """Run the children as one group, from the commit `at` when given, in dispatch order."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class Ranking:
    """The order a rank join lands by.

    Attributes:
        order: The ranked children's session ids, best first.
        ranked_by: `judge` when a judge's ranking ordered them, else `mechanical`.
        rationale: The judges' words; "" for the mechanical order.
        winner: The child the join lands, or "" when the order crowns none: every gate ran
            red, or a child that did not verify was ranked over one that did.
    """

    order: tuple[str, ...]
    ranked_by: Literal["judge", "mechanical"] = "mechanical"
    rationale: str = ""
    winner: str = ""


class ChildRanker(Protocol):
    """Rank a dispatch's finished children, best first."""

    def __call__(self, results: Sequence[ChildResult], *, group: str, base: str) -> Ranking:
        """Return the order of a group's children that ended with a result, cut from `base`."""
        ...


class ChildRecoverer(Protocol):
    """Find an interrupted group's children, stop what still runs and import what finished."""

    def __call__(self, group: str, children: Sequence[str]) -> list[ChildResult]:
        """Return a result per child found; a child with no session dir is missing."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class ChildJoin:
    """One child's outcome in a dispatch.

    Attributes:
        session_id: The child's session.
        branch: The child's branch.
        status: `joined` (merged, `sha` set), `no_changes` (ended with a result and no commit
            beyond its base; nothing to merge), `conflict` (imported, the merge conflicted,
            the ref exists locally), `failed` (no result or nothing importable) or
            `ranked` (a rank join placed the child and did not join it: another child landed,
            or the run has no chain to join onto; the branch holds this one's work).
        sha: The merge commit, or "".
        detail: The failure's reason; a ranked child's place (`2 of 3`); for a child that
            joined or had nothing to join, what stopped it when its parent did; else "".
        report: What the child said at its end, its finish summary; "" when it gave none.
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
        group: The group id the children were stamped with; "" when nothing was dispatched.
        dispatched: Whether children ran; False for a refusal before any clone.
        children: Each child's join, in dispatch order.
        message: The one notice summarizing the outcome, or the refusal.
        base: The commit the children were cut from; "" when none ran.
    """

    group: str
    dispatched: bool
    children: tuple[ChildJoin, ...]
    message: str
    base: str = ""
