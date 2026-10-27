# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Aggregate the review panel's verdicts with executable grounding.

A finding is reported only when its citation is in the diff (a touched path, or a line on
either side of a hunk), and a `block` gates only in an allowed category; a block elsewhere is
downgraded to `warn`, and `warn` and `nit` never gate. Network-free; `run_panel` calls the
models.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from agent6 import review_records
from agent6.vcs import review

ReviewBlockOn = Literal["none", "any", "model_quorum", "all"]

# Only the first set may gate; the rest advise, being too noisy to block on.
ALLOWED_BLOCK_CATEGORIES: frozenset[str] = frozenset(
    {"security", "sandbox-bypass", "off-topic-edit", "data-loss", "verify-uncovered-correctness"}
)
ADVISORY_CATEGORIES: frozenset[str] = frozenset({"test-gap", "style", "over-eng", "other"})
ALL_CATEGORIES: frozenset[str] = ALLOWED_BLOCK_CATEGORIES | ADVISORY_CATEGORIES


def rules_message(rules: review_records.ReviewRules) -> str:
    """Return the whole base-tree rule text or the absence notice."""
    return "REVIEW RULES (base tree):\n" + (rules.text or "No base rules exist.")


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewContext:
    """What every seat is shown, and what the aggregator grounds findings against.

    Attributes:
        task: The run's task.
        rules: The complete base-tree rules and their sources.
        diff: The immutable delta from the run base to the chain commit.
        verify_ok: The gate's result; None when none ran (none configured, or `agent6 review`).
        verify_output: The gate's output.
        focus: The seat's focus text.
        prior_findings: Findings already injected, for dedup; they never re-count.
    """

    task: str = ""
    rules: review_records.ReviewRules = dataclasses.field(
        default_factory=review_records.ReviewRules
    )
    diff: str = ""
    verify_ok: bool | None = None
    verify_output: str = ""
    focus: str = ""
    prior_findings: tuple[review_records.Finding, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class PanelResult:
    """The panel's aggregated result.

    Attributes:
        panel_id: The panel's id.
        block_on: The gating rule applied.
        blocked: Whether the panel rejects the work.
        merged_findings: The grounded findings across seats, deduplicated, blocks first.
        per_seat: Each seat's grounded verdict.
        n_block: Distinct blocking models counted toward the gate.
        n_abstain: Seats that failed.
        skipped_reason: Why the panel did not run, or None.
    """

    panel_id: str
    block_on: ReviewBlockOn
    blocked: bool
    merged_findings: tuple[review_records.Finding, ...]
    per_seat: tuple[review_records.ReviewVerdict, ...]
    n_block: int
    n_abstain: int
    skipped_reason: str | None = None


def panel_is_inconclusive(result: PanelResult) -> bool:
    """Return whether every seat abstained, so nothing was reviewed.

    The one owner the CLI verdict and the in-loop critique both ask, so neither reads an
    all-abstain panel as a pass.

    Args:
        result: The panel result.

    Returns:
        Whether every seat abstained.
    """
    return bool(result.per_seat) and result.n_abstain == len(result.per_seat)


def inconclusive_note(result: PanelResult) -> str:
    """Return the human line for an all-abstain panel."""
    return f"review inconclusive: all {result.n_abstain} seats abstained; nothing was reviewed"


# A critique rides into the worker's next turn beside the tool results, findings first.
# `_compaction_result.CLAUDE_CODE_NOTICE_ROOM_BYTES` reserves this critique space.
REVIEW_NOTICE_BYTES = 4_000


def review_notice(text: str) -> str:
    """Return the `[review]` notice for the text, cut to `REVIEW_NOTICE_BYTES` with a marker."""
    body = text.encode()
    if len(body) <= REVIEW_NOTICE_BYTES:
        return f"[review]\n{text}"
    head = body[:REVIEW_NOTICE_BYTES].decode(errors="ignore")
    return f"[review]\n{head}\n[review: {len(body) - len(head.encode())} more bytes cut]"


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return whether two inclusive spans share a line."""
    return a[0] <= b[1] and b[0] <= a[1]


def is_grounded(location: review_records.ReviewLocation, hunks: review.Hunks) -> bool:
    """Return whether the location overlaps a hunk on its specified side.

    A file-level location grounds on a touched path on that side.

    Args:
        location: The cited path, side and optional line range.
        hunks: The diff's hunks by repository path and side.

    Returns:
        Whether the location grounds.
    """
    key = (location.path, location.side)
    if location.start_line is None or location.end_line is None:
        return key in hunks
    span = (location.start_line, location.end_line)
    return any(
        target is not None and _overlaps(span, target)
        for h in hunks.get(key, ())
        for target in (h.old if location.side == "base" else h.new,)
    )


_SEV_ORDER = {"block": 0, "warn": 1, "nit": 2}


def _ground_severity(f: review_records.Finding, ctx: ReviewContext) -> review_records.Severity:
    """Return the severity a finding keeps after grounding.

    A `block` survives only in a gating category, and `verify-uncovered-correctness` only when
    verify passed; otherwise it becomes `warn`. `warn` and `nit` pass through.

    Args:
        f: The finding.
        ctx: The review context, for the verify result.

    Returns:
        The severity.
    """
    if f.severity != "block":
        return f.severity
    coherent = f.category != "verify-uncovered-correctness" or ctx.verify_ok is True
    if f.category in ALLOWED_BLOCK_CATEGORIES and coherent:
        return "block"
    return "warn"


def _ground_seat(
    v: review_records.ReviewVerdict, ctx: ReviewContext, hunks: review.Hunks
) -> review_records.ReviewVerdict:
    """Ground one seat's findings.

    An uncited finding is dropped; a block the category cannot carry, or one on a seat whose
    verdict is pass, becomes a warn.

    Args:
        v: The seat's verdict.
        ctx: The review context.
        hunks: The diff's hunks by repo path.

    Returns:
        The verdict with its grounded findings.
    """
    out: list[review_records.Finding] = []
    for f in v.findings:
        if not is_grounded(f.location, hunks):
            continue
        sev = _ground_severity(f, ctx)
        if v.verdict != "block" and sev == "block":
            sev = "warn"
        out.append(f if sev == f.severity else f.model_copy(update={"severity": sev}))
    return dataclasses.replace(v, findings=tuple(out))


def _has_new_block(v: review_records.ReviewVerdict, prior_keys: set[str]) -> bool:
    """Return whether the seat carries a surviving block that is not a prior finding.

    A block whose key dedups away is dropped from `merged_findings`, so letting it gate would
    reject the work while reporting no blocking findings.

    Args:
        v: The seat's grounded verdict.
        prior_keys: The keys of the already-injected findings.

    Returns:
        Whether a new block survives.
    """
    return v.verdict == "block" and any(
        f.severity == "block" and review_records.finding_key(f) not in prior_keys
        for f in v.findings
    )


def _decide(
    block_on: ReviewBlockOn,
    n_block: int,
    min_blocking_models: int,
    *,
    n_seats_blocking: int,
    n_total: int,
) -> bool:
    """Apply the gating rule.

    Args:
        block_on: The rule.
        n_block: Distinct blocking models.
        min_blocking_models: The distinct blocking routes `model_quorum` needs.
        n_seats_blocking: Seats with a surviving non-prior block.
        n_total: Seats that reviewed.

    Returns:
        Whether the panel blocks.
    """
    if block_on == "none" or not n_total:
        return False
    if block_on == "any":
        return n_block >= 1
    if block_on == "model_quorum":
        return n_block >= max(1, min_blocking_models)
    if block_on == "all":
        return n_seats_blocking == n_total
    return False  # pragma: no cover - exhaustive


def aggregate_verdicts(
    per_seat: list[review_records.ReviewVerdict],
    ctx: ReviewContext,
    *,
    block_on: ReviewBlockOn,
    min_blocking_models: int,
    panel_id: str,
) -> PanelResult:
    """Fold per-seat verdicts into one panel result with executable grounding.

    Every finding whose citation is not in the diff is dropped, and a block outside a gating
    category becomes a warn. Exact normalized content and location identify a repeat.
    An already-injected block neither re-surfaces nor counts toward the gate.
    `none` never blocks; `any` blocks on any surviving block. `model_quorum` needs
    `min_blocking_models` distinct provider/model routes to block. `all` needs every seat
    to block; an abstention is not a block.

    Args:
        per_seat: The seats' verdicts.
        ctx: The review context.
        block_on: The gating rule.
        min_blocking_models: The distinct blocking routes `model_quorum` needs.
        panel_id: The panel's id.

    Returns:
        The panel result.
    """
    hunks = review.diff_hunks(ctx.diff)
    prior_keys = {review_records.finding_key(f) for f in ctx.prior_findings}

    grounded_seats: list[review_records.ReviewVerdict] = []
    blocking_models: set[str] = set()  # distinct models with >=1 surviving non-prior block
    n_abstain = n_seats_blocking = 0
    for v in per_seat:
        if v.error is not None or v.verdict == "abstain":
            n_abstain += 1
            grounded_seats.append(dataclasses.replace(v, verdict="abstain", findings=()))
            continue
        gv = _ground_seat(v, ctx, hunks)
        grounded_seats.append(gv)
        if _has_new_block(gv, prior_keys):
            blocking_models.add(v.model_route)
            n_seats_blocking += 1

    # Merge and dedup the grounded findings, dropping those already injected.
    merged: dict[str, review_records.Finding] = {}
    for v in grounded_seats:
        for f in v.findings:
            key = review_records.finding_key(f)
            if key in prior_keys:
                continue
            cur = merged.get(key)
            if cur is None or _SEV_ORDER[f.severity] < _SEV_ORDER[cur.severity]:
                merged[key] = f
    merged_findings = tuple(
        sorted(merged.values(), key=lambda f: (_SEV_ORDER[f.severity], str(f.location), f.category))
    )

    n_block = len(blocking_models)  # distinct-model blocking seats
    blocked = _decide(
        block_on,
        n_block,
        min_blocking_models,
        n_seats_blocking=n_seats_blocking,
        n_total=len(grounded_seats),
    )

    return PanelResult(
        panel_id=panel_id,
        block_on=block_on,
        blocked=blocked,
        merged_findings=merged_findings,
        per_seat=tuple(grounded_seats),
        n_block=n_block,
        n_abstain=n_abstain,
    )


def findings_within(
    findings: tuple[review_records.Finding, ...], budget: int
) -> tuple[review_records.Finding, ...]:
    """Return the longest leading run of findings whose rendering fits the budget in bytes."""
    for n in range(len(findings), 0, -1):
        if len(render_findings(findings[:n]).encode()) <= budget:
            return findings[:n]
    return ()


def render_findings(findings: tuple[review_records.Finding, ...]) -> str:
    """Return merged findings as a `[review]` block for the worker or the CLI, "" when empty."""
    if not findings:
        return ""
    lines = []
    for f in findings:
        lines.append(f"- [{f.severity}:{f.category}] ({f.location}) {f.title}")
        if f.detail.strip():
            lines.append(f"    {f.detail.strip()}")
    return "\n".join(lines)


__all__ = [
    "ALLOWED_BLOCK_CATEGORIES",
    "ALL_CATEGORIES",
    "PanelResult",
    "ReviewBlockOn",
    "ReviewContext",
    "aggregate_verdicts",
    "findings_within",
    "is_grounded",
    "render_findings",
]
