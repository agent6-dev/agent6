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
    """Identify one base-tree rule blob supplied whole to the reviewer."""

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
    """Identify the immutable trees and path coverage of a review."""

    base_commit: str | None
    base_tree: str
    head_commit: str | None
    head_tree: str
    snapshot_commit: str
    paths: tuple[str, ...]
    source: Literal["revisions", "worktree"]


class ReviewLocation(pydantic.BaseModel):
    """Name a repository-relative path on one side, with an optional inclusive line range."""

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
    """Hold one review finding at a structured location."""

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


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewVerdict:
    """Hold one indexed seat's verdict, including an explicit abstention on failure."""

    seat_index: int
    focus: str
    model_route: str
    verdict: Literal["pass", "block", "abstain"]
    findings: tuple[Finding, ...] = ()
    summary: str = ""
    error: str | None = None


class LineOrigin(pydantic.BaseModel):
    """Record byte-level evidence without dating a defect or changing its severity."""

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
    """Associate a finding with its panel and the seats that reported it."""

    finding_id: str
    panel_id: str
    finding: Finding
    seat_indexes: tuple[int, ...]
    line_origin: LineOrigin = dataclasses.field(default_factory=LineOrigin)


class ReviewPanel(pydantic.BaseModel):
    """Validate a completed panel's durable evidence, including its worker feedback."""

    model_config = pydantic.ConfigDict(frozen=True)

    panel_id: Identifier
    execution_id: Identifier
    turn: int
    trigger: str
    target: ReviewTarget
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
    """Validate an operator request against a known review-state version."""

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
