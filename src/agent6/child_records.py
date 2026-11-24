# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The child records every group shares: what a child runs, what it returned, how it merged.

`run --parallel`, the `/parallel` steer and the import plumbing read and write these; the
code that clones, spawns and merges lives in `harness.parallel.repo`, `app.children` and
`harness.parallel.dispatch`.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import pathlib
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from agent6 import budget, kinds

# Who asked for a dispatch: the operator's steer, the model's tool call, or a review panel.
DispatchSource = Literal["operator", "model", "panel"]
# How a dispatch merges its children: each merged in order, none (a panel's records), or the
# best-ranked one alone.
MergePolicy = Literal["merge", "record_only", "rank"]
# How one child's work merged into the parent's chain.
MergeStatus = Literal["merged", "no_changes", "conflict", "failed", "ranked", "stopped", "retained"]


def number_label(number: int, *, reviewer: bool) -> str:
    """Return the child's place within its task or panel, when recorded."""
    label = "reviewer" if reviewer else "attempt"
    return f"{label} {number}" if number > 0 else label


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
class ChildContinuation:
    """A parent's instruction for another execution of an existing child.

    Attributes:
        number: The child's original number in its group.
        instruction: Its next instruction.
        fresh: Restore a new clone rather than reuse the stopped child's clone.
        commits: Commits beyond the base on the retained ref.
    """

    number: int
    instruction: str
    fresh: bool = False
    commits: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class ChildTask:
    """Name one child to dispatch, as the parent expands a `/parallel` segment.

    Attributes:
        task: The child's task text.
        model: The child's selector (a role name or `provider/model`), None for the
            worker's own settings.
        pins: The operator's pins, passed beside the task so the child's manifest task stays clean.
        allowance: The child's share of the parent's allowance; None for the config's caps.
        limits: The child's turn cap and deadline; None for the config's own.
        contract: A harness-owned bounded conversation; None for work.
        settings: Resolved panel-reviewer settings; None resolves the model selector.
        continuation: The existing child and its next instruction; None starts a new child.
        task_source: Who supplied the task.
    """

    task: str
    model: str | None
    pins: tuple[str, ...] = ()
    allowance: budget.BudgetAllowance | None = None
    limits: ChildLimits | None = None
    contract: kinds.ConversationContract | None = None
    settings: kinds.ModelSettings | None = None
    continuation: ChildContinuation | None = None
    task_source: DispatchSource = "operator"


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
        task_source: Who supplied the task.
    """

    number: int
    session_id: str
    worktree: pathlib.Path
    settings: kinds.ModelSettings | None
    mode: kinds.LoopMode = "run"
    task_source: DispatchSource = "operator"

    @property
    def number_label(self) -> str:
        """Return the attempt or reviewer number for this launch."""
        return number_label(self.number, reviewer=self.mode == "reviewer")


@dataclasses.dataclass(frozen=True, slots=True)
class Imported:
    """What a child's import retained in the origin.

    Attributes:
        session_dir: The child's session dir under the origin's state.
        ref: The ref the import fetched the child's work into; "" when only the record came.
        tip: That ref's commit; "" with no ref.
    """

    session_dir: pathlib.Path
    ref: str = ""
    tip: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class ChildContext:
    """The last measured conversation size and activity of a child.

    Attributes:
        tokens: Full provider input at its last measured call; None when unknown.
        last_run_at: Epoch of its last journal activity when it ended; None when unknown.
    """

    tokens: int | None = None
    last_run_at: float | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ChildResult:
    """Record the outcome of one child.

    Attributes:
        spec: The child run.
        session_dir: Where its run state lives.
        branch: The branch the child was given; a name, not proof that work was imported.
        ok: Whether it ended with a result and its record was imported.
        error: Why it failed, "" on success.
        imported_ref: The ref the import stored the child's work in; "" when none.
        imported_tip: That ref's commit; "" when none.
        has_commits: Whether the imported tip holds commits beyond the child's base.
        commits: How many it holds; 0 when none, or when the base is unrecorded.
        verification: Its own gate's word at its end; None when it never said.
        pid: The child process the spawn started, 0 when none did; a child that never
            reported its session dir is stopped through it.
        usage: What the child used, read from its imported record; None when no record
            came back.
        report: What the child said at its end, its finish summary; "" when it ended another
            way.
        context: Its last measured conversation size and activity.
        stopped_by: What ended the child short of its own finish: its deadline, the run's
            budget, its parent's stop, or `stopped` for the operator's and for the end of the
            execution that started it; "" when it ended by itself. A stopped child's work
            never merges by itself.
        started: Whether its worker reported the start of this execution; False when the
            launch failed or was stopped first.
    """

    spec: ChildSpec
    session_dir: pathlib.Path
    branch: str
    ok: bool
    error: str
    imported_ref: str = ""
    imported_tip: str = ""
    has_commits: bool = False
    commits: int = 0
    verification: kinds.Verification | None = None
    pid: int = 0
    usage: budget.UsageSettlement | None = None
    report: str = ""
    stopped_by: str = ""
    context: ChildContext = ChildContext()
    started: bool = True


class ChildSpawner(Protocol):
    """Run one child to its end."""

    def __call__(self, spec: ChildSpec, task: str) -> ChildResult:
        """Run the child on the task and return its result."""
        ...


def _no_stop() -> None:
    """Stop nothing: the hook of a child that has already ended."""


@dataclasses.dataclass(frozen=True, slots=True)
class StartedChild:
    """One child started in the background, watched until it has ended and been imported.

    Attributes:
        session_id: The child's session id.
        result: Resolves once: with the child's result after its import, or with why it
            never ran.
        stop: Asks the child to stop now; its result still resolves, with its work imported
            and never merged.
        workspace: Where a continued child resumed, empty on first launch.
    """

    session_id: str
    result: concurrent.futures.Future[ChildResult]
    stop: Callable[[], None] = _no_stop
    workspace: str = ""


class GroupChildSpawner(Protocol):
    """Start a group of sibling children, each imported into the parent's repo at its end.

    One call returns at once; `app.children` clones, spawns, awaits and imports every child
    behind it, and the parent supplies the tasks and the group id (`p<seq>`). A child that
    failed to start, outlived the teardown or was refused at import has `ok` False and
    leaves the parent's repo untouched.
    """

    def __call__(
        self, children: list[ChildTask], group: str, *, at: str | None = None
    ) -> list[StartedChild]:
        """Start the children as one group, from the commit `at` when given, in dispatch order."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class CandidateBrief:
    """One attempt at a task, as its judges and the mechanical order read it.

    Attributes:
        session_id: The attempt's session id.
        task: The task it was given.
        diff: Its change from the commit it started at.
        verification: Its own gate's word; None when it never said.
        cost_usd: What it spent.
        ref: The branch or ref that holds its work; "" when none does.
        tip: That ref's commit; "" when none.
        summary: What it said at its end; "" when it gave none.
    """

    session_id: str
    task: str
    diff: str
    verification: kinds.Verification | None
    cost_usd: float
    ref: str = ""
    tip: str = ""
    summary: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Ranking:
    """The ranked candidates in merge order.

    Attributes:
        order: The ranked children's session ids, best first.
        ranked_by: `judge` when a judge's order ranked them, else `mechanical`.
        rationale: The judges' words; "" for the mechanical order.
        winner: The child selected to merge, or "" when the order crowns none: every gate ran
            red, or a child that did not verify was ranked over one that did.
        fallback: Why the mechanical order ranked; "" when judges did.
        judge_cost_usd: What the judges spent; 0.0 when none ran.
        judge_cost_partial: Whether that spend is a lower bound.
    """

    order: tuple[str, ...]
    ranked_by: Literal["judge", "mechanical"] = "mechanical"
    rationale: str = ""
    winner: str = ""
    fallback: str = ""
    judge_cost_usd: float = 0.0
    judge_cost_partial: bool = False


class RankRecorder(Protocol):
    """Record the order a dispatch's attempts were ranked in, outside the dispatch record."""

    def __call__(
        self,
        candidates: Sequence[CandidateBrief],
        results: Sequence[ChildResult],
        ranking: Ranking,
        *,
        group: str,
    ) -> None:
        """Stamp the ranking on the manifests, journal it and print the ranked table."""
        ...


class ChildRecoverer(Protocol):
    """Find an interrupted group's children, stop what still runs and import what finished."""

    def __call__(self, group: str, children: Sequence[str]) -> list[ChildResult]:
        """Return a result per child found; a child with no session dir is missing."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class ChildMerge:
    """One child's outcome in a dispatch.

    Attributes:
        session_id: The child's session.
        branch: The child's branch.
        status: `merged` (on the parent chain, `sha` set), `no_changes` (ended with a result
            and no commit beyond its base), `conflict` (the merge conflicted,
            the ref exists locally), `failed` (no result or nothing importable),
            `ranked` (the ranking placed the child but it did not merge: another child merged,
            or the run has no chain to merge onto; the branch holds this one's work) or
            `stopped` (the child was stopped, so nothing merged; the branch holds its work),
            or `retained` (finished under model-owned git; the imported ref holds its work).
        sha: The merge commit, or "".
        detail: The failure's reason; a conflict's files; a ranked child's place (`2 of 3`);
            what stopped a stopped child (`stopped by its parent`); else "".
        report: What the child said at its end, its finish summary; "" when it gave none.
        context: Its last measured conversation size and activity.
        commits: The commits on the retained ref beyond the child's base, or 0 without a count.
    """

    session_id: str
    branch: str
    status: MergeStatus
    sha: str
    detail: str
    report: str = ""
    commits: int = 0
    context: ChildContext = ChildContext()


@dataclasses.dataclass(frozen=True, slots=True)
class Started:
    """What a work dispatch started in the background, or why it started nothing.

    Attributes:
        children: The started children's session ids, in the order written; empty for a
            refusal or a spawner failure.
        message: What started and how to follow it, or the refusal.
        groups: The group id of each dispatch record written.
    """

    children: tuple[str, ...]
    message: str
    groups: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class DispatchResult:
    """What a completed dispatch did, for the caller to deliver as it sees fit.

    Attributes:
        group: The group id the children were stamped with; "" when nothing was dispatched.
        dispatched: Whether children ran; False for a refusal before any clone.
        children: Each child's outcome in roster order, or None for a reviewer without a result.
        message: The one notice summarizing the outcome, or the refusal.
        base: The commit the children were cut from; "" when none ran.
    """

    group: str
    dispatched: bool
    children: tuple[ChildMerge | None, ...]
    message: str
    base: str = ""
