"""Define immutable review evidence and validate review boundary payloads."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Annotated, Literal, Self

import pydantic

Severity = Literal["block", "warn", "nit"]
Reason = Annotated[
    str, pydantic.StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]
Identifier = Annotated[str, pydantic.StringConstraints(strip_whitespace=True, min_length=1)]


@dataclasses.dataclass(frozen=True, slots=True)
class RuleSource:
    """Identify one base-tree rule blob supplied whole to the reviewer.

    Attributes:
        path: The rule file's path in the base tree.
        blob_id: The immutable rule blob identifier.
        supplied_bytes: The number of rule bytes supplied to reviewers.
    """

    path: str
    blob_id: str
    supplied_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewRules:
    """Hold the complete rule text, or a refusal with no text supplied."""

    sources: tuple[RuleSource, ...] = ()
    text: str = ""
    complete: bool = True
    reason: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewTarget:
    """Identify the immutable trees and path coverage of a review.

    Attributes:
        base_commit: The base commit, or null for a tree-only comparison.
        base_tree: The immutable base tree identifier.
        head_commit: The head commit, or null for a working-tree snapshot.
        head_tree: The immutable head tree identifier.
        snapshot_commit: The commit retaining the reviewed tree.
        paths: The reviewed path scope, empty for all paths.
        source: Whether the comparison came from revisions or the working tree.
        hidden_paths: Paths withheld from the review.
        snapshot_ref: The session-owned hidden ref retaining the comparison; empty on old records.
    """

    base_commit: str | None
    base_tree: str
    head_commit: str | None
    head_tree: str
    snapshot_commit: str
    paths: tuple[str, ...]
    source: Literal["revisions", "worktree"]
    hidden_paths: tuple[str, ...] = ()
    snapshot_ref: str = ""


class ReviewLocation(pydantic.BaseModel):
    """Name a repository-relative path on one side, with an optional inclusive line range.

    Attributes:
        path: The repository-relative path.
        side: The base or head side of the comparison.
        start_line: The inclusive one-based start line, or null for a file-level finding.
        end_line: The inclusive one-based end line, or null for a file-level finding.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    path: str
    side: Literal["base", "head"]
    start_line: int | None = pydantic.Field(default=None, gt=0, strict=True)
    end_line: int | None = pydantic.Field(default=None, gt=0, strict=True)

    @pydantic.field_validator("path")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        path = pathlib.PurePosixPath(value)
        if (
            not value
            or path == pathlib.PurePosixPath()
            or path.is_absolute()
            or pathlib.PureWindowsPath(value).drive
            or ".." in path.parts
            or "\\" in value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
        ):
            raise ValueError("review path must stay within the repository")
        return path.as_posix()

    @pydantic.model_validator(mode="after")
    def _inclusive_range(self) -> Self:
        if (self.start_line is None) != (self.end_line is None):
            raise ValueError("review lines must both be set or both be unset")
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError("review end_line must not precede start_line")
        return self

    def __str__(self) -> str:
        suffix = " (base)" if self.side == "base" else ""
        if self.start_line is None:
            return f"{self.path}{suffix}"
        end = f"-{self.end_line}" if self.end_line != self.start_line else ""
        return f"{self.path}:{self.start_line}{end}{suffix}"


class Finding(pydantic.BaseModel):
    """Hold one review finding at a structured location.

    Attributes:
        category: The finding's issue category.
        severity: The grounded block, warn or nit severity.
        title: The short finding title.
        detail: The supporting explanation and evidence.
        location: The path, side and optional line range.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    category: str
    severity: Severity
    title: str = pydantic.Field(max_length=200)
    detail: str = pydantic.Field(default="", max_length=1000)
    location: ReviewLocation


class SeatReply(pydantic.BaseModel):
    """Validate the verdict and findings returned by a review seat."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    verdict: Literal["pass", "block"]
    summary: str
    findings: tuple[Finding, ...]
    risk: str = pydantic.Field(default="", max_length=1000)
    risk_truncated: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewVerdict:
    """Hold one indexed seat's verdict, including an explicit abstention on failure.

    Attributes:
        seat_index: The seat's zero-based position in the roster.
        focus: The seat's assigned review focus.
        model_route: The provider/model route that reviewed.
        verdict: The seat's pass, block or abstention decision.
        findings: The grounded findings from this seat.
        summary: The seat's closing summary.
        error: The failure diagnostic for an abstention, or null.
        risk: The seat's retained risk assessment.
        risk_truncated: Whether the risk text was truncated.
    """

    seat_index: int
    focus: str
    model_route: str
    verdict: Literal["pass", "block", "abstain"]
    findings: tuple[Finding, ...] = ()
    summary: str = ""
    error: str | None = None
    risk: str = ""
    risk_truncated: bool = False


class LineOrigin(pydantic.BaseModel):
    """Record byte-level evidence without dating a defect or changing its severity.

    Attributes:
        kind: Whether the lines are pre-existing, changed or of unknown origin.
        reason: The explanation for the attribution result.
        base_location: The mapped base range for pre-existing lines, or null.
        commits: Attribution commits for pre-existing lines.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    kind: Literal["pre_existing", "changed", "unknown"] = "unknown"
    reason: str = "Historical observations contain no line-origin evidence."
    base_location: ReviewLocation | None = None
    commits: tuple[Identifier, ...] = ()

    @pydantic.model_validator(mode="after")
    def _evidence(self) -> Self:
        if self.kind == "pre_existing":
            if (
                self.base_location is None
                or self.base_location.side != "base"
                or self.base_location.start_line is None
                or not self.commits
            ):
                raise ValueError("pre-existing lines require a base range and attribution commits")
        elif self.base_location is not None or self.commits:
            raise ValueError("only pre-existing lines carry base attribution")
        return self


@dataclasses.dataclass(frozen=True, slots=True)
class FindingObservation:
    """Associate a finding with its panel and the seats that reported it.

    Attributes:
        finding_id: The finding identifier shared across observations.
        panel_id: The panel that recorded this observation.
        finding: The grounded finding text and location.
        seat_indexes: The zero-based seats that reported the finding.
        line_origin: The line-attribution evidence for this observation.
    """

    finding_id: str
    panel_id: str
    finding: Finding
    seat_indexes: tuple[int, ...]
    line_origin: LineOrigin = dataclasses.field(default_factory=LineOrigin)


@dataclasses.dataclass(frozen=True, slots=True)
class SeatAssignment:
    """Identify the resolved route and settings of one panel seat.

    Attributes:
        seat_index: The seat's zero-based position in the roster.
        model_route: The resolved provider/model route.
        focus: The seat's assigned review focus.
        scope: Whether the seat reviews the diff or the repository.
        temperature: The configured sampling temperature, or null for the provider default.
    """

    seat_index: int
    model_route: str
    focus: str
    scope: Literal["diff", "repo"] = "diff"
    temperature: float | None = None


class ReviewPanel(pydantic.BaseModel):
    """Validate a completed panel's durable evidence, including its worker feedback.

    A journal row validates with its envelope stripped (`event_log.payload`).

    Attributes:
        panel_id: The durable panel identifier.
        execution_id: The execution that requested the panel.
        turn: The turn that triggered the panel.
        trigger: The review trigger.
        target: The immutable reviewed comparison.
        roster: The seats that sat; empty on a record written before rosters were kept.
        findings: The panel's grounded finding observations.
        injected_finding_ids: Findings supplied to the worker as feedback.
        block_on: The configured worker-finish blocking rule.
        raw_blocked: Whether the panel met the blocking rule before disarming.
        blocked: Whether the panel blocks worker completion.
        disarmed: Whether the harness disarmed the review block.
        n_block: The number of distinct blocking provider/model routes.
        n_abstain: The number of abstaining seats.
        completion: Whether the panel completed conclusively.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    panel_id: Identifier
    execution_id: Identifier
    turn: int
    trigger: str
    target: ReviewTarget
    roster: tuple[SeatAssignment, ...] = ()
    findings: tuple[FindingObservation, ...]
    injected_finding_ids: tuple[str, ...]
    block_on: Literal["none", "any", "all", "model_quorum"]
    raw_blocked: bool
    blocked: bool
    disarmed: bool
    n_block: int = pydantic.Field(ge=0)
    n_abstain: int = pydantic.Field(ge=0)
    completion: Literal["complete", "inconclusive"]


def finding_key(finding: Finding) -> str:
    """Return exact finding content and location with surrounding whitespace stripped."""
    return finding.model_copy(
        update={
            "category": finding.category.strip(),
            "title": finding.title.strip(),
            "detail": finding.detail.strip(),
        }
    ).model_dump_json()


class FindingResolution(pydantic.BaseModel):
    """Record the operator's conclusion without inferring verification success.

    The action handler must check that a duplicate names another finding in the same session.

    Attributes:
        disposition: The operator's conclusion about the finding.
        reason: The operator's explanation.
        duplicate_of: The other finding for a duplicate resolution, or null.
        commit: The associated correction commit, or null.
        verify_id: The associated verification attempt, or null.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    disposition: Literal[
        "fixed", "false_positive", "intentional", "wont_fix", "duplicate", "pre_existing"
    ]
    reason: Reason
    duplicate_of: Identifier | None = None
    commit: str | None = None
    verify_id: str | None = None

    @pydantic.model_validator(mode="after")
    def _duplicate_target(self) -> Self:
        if self.disposition == "duplicate" and self.duplicate_of is None:
            raise ValueError("duplicate resolution requires another finding ID")
        return self


class FindingAction(pydantic.BaseModel):
    """Validate an operator request against a known review-state version.

    Attributes:
        request_id: The caller's identifier for idempotent action delivery.
        expected_version: The review version the caller read.
        action: The action kind supplied by the route rather than the HTTP body.
        resolution: The conclusion for a resolve action, or null.
        reason: The explanation for a reopen action, or null.
        resume: Whether the operator approved resuming a stopped session to fix the finding.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    request_id: Identifier
    expected_version: int = pydantic.Field(ge=0, strict=True)
    action: Literal["fix", "resolve", "reopen"]
    resolution: FindingResolution | None = None
    reason: Reason | None = None
    resume: bool = False

    @pydantic.model_validator(mode="after")
    def _required_reason(self) -> Self:
        if self.action == "resolve" and self.resolution is None:
            raise ValueError("resolve requires a resolution with a reason")
        if self.action == "reopen" and self.reason is None:
            raise ValueError("reopen requires a reason")
        return self
