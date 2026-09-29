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
import re
from typing import Literal

Severity = Literal["block", "warn", "nit"]
Verdict = Literal["pass", "block"]
ReviewDecision = Literal["advisory", "veto", "quorum", "all"]

# Only the first set may gate; the rest advise, being too noisy to block on.
ALLOWED_BLOCK_CATEGORIES: frozenset[str] = frozenset(
    {"security", "sandbox-bypass", "off-topic-edit", "data-loss", "verify-uncovered-correctness"}
)
ADVISORY_CATEGORIES: frozenset[str] = frozenset({"test-gap", "style", "over-eng", "other"})
ALL_CATEGORIES: frozenset[str] = ALLOWED_BLOCK_CATEGORIES | ADVISORY_CATEGORIES


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """One reviewer finding.

    Attributes:
        category: One of `ALL_CATEGORIES`.
        severity: block, warn or nit.
        file_line: The citation the grounding check reads: "path:line" or "path".
        title: One line.
        detail: The rest, or "".
    """

    category: str
    severity: Severity
    file_line: str
    title: str
    detail: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewVerdict:
    """One seat's verdict.

    Attributes:
        seat: The seat's name.
        model: The model that reviewed.
        verdict: pass or block.
        findings: The seat's findings.
        summary: The seat's summary, or "".
        error: Set when the seat failed; the seat then abstains rather than passes.
    """

    seat: str
    model: str
    verdict: Verdict
    findings: tuple[Finding, ...] = ()
    summary: str = ""
    error: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewContext:
    """What every seat is shown, and what the aggregator grounds findings against.

    Attributes:
        task: The run's task.
        agents_md: The repo's AGENTS.md.
        diff: The working-tree delta since the last accepted finish.
        verify_ok: The gate's result; None when none ran (none configured, or `agent6 review`).
        verify_output: The gate's output.
        persona: The seat's persona text.
        prior_findings: Findings already injected, for dedup; they never re-count.
    """

    task: str = ""
    agents_md: str = ""
    diff: str = ""
    verify_ok: bool | None = None
    verify_output: str = ""
    persona: str = ""
    prior_findings: tuple[Finding, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class PanelResult:
    """The panel's aggregated result.

    Attributes:
        panel_id: The panel's id.
        decision: The gating rule applied.
        blocked: Whether the panel rejects the work.
        merged_findings: The grounded findings across seats, deduplicated, blocks first.
        per_seat: Each seat's grounded verdict.
        n_block: Distinct blocking models counted toward the gate.
        n_abstain: Seats that failed.
        skipped_reason: Why the panel did not run, or None.
    """

    panel_id: str
    decision: ReviewDecision
    blocked: bool
    merged_findings: tuple[Finding, ...]
    per_seat: tuple[ReviewVerdict, ...]
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
# `_compaction.CLAUDE_CODE_NOTICE_ROOM_BYTES` sizes the result cap with this much critique in mind.
REVIEW_NOTICE_BYTES = 4_000


def review_notice(text: str) -> str:
    """Return the `[review]` notice for the text, cut to `REVIEW_NOTICE_BYTES` with a marker."""
    body = text.encode()
    if len(body) <= REVIEW_NOTICE_BYTES:
        return f"[review]\n{text}"
    head = body[:REVIEW_NOTICE_BYTES].decode(errors="ignore")
    return f"[review]\n{head}\n[review: {len(body) - len(head.encode())} more bytes cut]"


# Diff grounding: the (path, line) citations this diff supports.
# Both the old-side and new-side line numbers, so a deletion grounds against the pre-image path.
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_GIT_HDR_RE = re.compile(r"^diff --git a/(.*?) b/(.*)$")


def _unquote_git_path(p: str) -> str:
    """Return the path with git's C-string quoting stripped, so it matches a citation."""
    p = p.strip()
    if len(p) >= 2 and p.startswith('"') and p.endswith('"'):
        p = p[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return p


def _hdr_path(raw: str) -> str:
    """Return the path of a `--- ` or `+++ ` header line, or "" for /dev/null."""
    target = _unquote_git_path(raw[4:].split("\t", 1)[0])
    return "" if target == "/dev/null" else re.sub(r"^[ab]/", "", target)


@dataclasses.dataclass(frozen=True, slots=True)
class Hunk:
    """One hunk as it addresses one path.

    A side with no lines spans the line the change sits at, so a citation of it grounds.

    Attributes:
        old: The pre-image lines it replaced, an inclusive span; None when that side is another
            file's (a rename's other name) or absent (a created file).
        new: The post-image lines it produced, likewise; None for a rename's other name or a
            deleted file.
    """

    old: tuple[int, int] | None
    new: tuple[int, int] | None


def _span(start: str, count: str | None) -> tuple[int, int]:
    """Return the inclusive span of a hunk side from its header start and count."""
    s, n = int(start), int(count) if count is not None else 1
    return (s, s + max(n, 1) - 1)


def diff_hunks(diff: str) -> dict[str, list[Hunk]]:
    """Map each touched path to its hunks, for grounding and dedup.

    An in-place hunk carries both sides under its path. A rename's, a created file's or a
    deleted file's hunk carries each side under that side's path, so deleted code grounds at
    its old line number and neither name grounds the other's. A `+++ ` line counts as a header
    only after a `--- ` line. A file touched without hunks (binary, a pure rename, a mode
    change) is recorded with none, so a path-only citation of it grounds and a line citation
    does not.

    Args:
        diff: The unified diff.

    Returns:
        The hunks by repo path.
    """
    hunks: dict[str, list[Hunk]] = {}
    newpath = oldpath = ""
    prev_minus = False
    lines = diff.splitlines()
    for i, raw in enumerate(lines):
        if g := _GIT_HDR_RE.match(raw):
            hunks.setdefault(_unquote_git_path(g.group(1)), [])
            hunks.setdefault(_unquote_git_path(g.group(2)), [])
            oldpath = newpath = ""
            prev_minus = False
            continue
        # A "--- " header is always followed by "+++ "; the lookahead rejects a deleted "-- " line.
        if raw.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            oldpath, newpath, prev_minus = _hdr_path(raw), "", True
            continue
        if raw.startswith("+++ ") and prev_minus:
            newpath, prev_minus = _hdr_path(raw), False
            continue
        prev_minus = False
        m = _HUNK_RE.match(raw)
        if m:
            old, new = _span(m.group(1), m.group(2)), _span(m.group(3), m.group(4))
            if newpath and newpath == oldpath:
                hunks.setdefault(newpath, []).append(Hunk(old=old, new=new))
            else:
                if newpath:
                    hunks.setdefault(newpath, []).append(Hunk(old=None, new=new))
                if oldpath:
                    hunks.setdefault(oldpath, []).append(Hunk(old=old, new=None))
    return hunks


def _split_cite(file_line: str, hunks: dict[str, list[Hunk]]) -> tuple[str, tuple[int, int] | None]:
    """Parse one citation into its repo path and cited line span.

    The forms a reviewer writes: 'foo.py', 'a/foo.py:2', 'foo.py:2-4', 'foo.py:2:' and
    'foo.py:12:5' (`path:line:col`). The path resolves against the hunks first; a leading `a/`
    or `b/` is dropped only when the unstripped path is not in the diff.

    Args:
        file_line: The citation.
        hunks: The diff's hunks by repo path.

    Returns:
        The path and the span, or None for a path-only citation; ("", None) for an empty one.
    """
    cite = file_line.strip().rstrip(":")
    if not cite:
        return "", None
    parts = cite.split(":")
    # Trailing numeric fields are position (line, then column); a drive letter is not numeric.
    nums: list[str] = []
    while len(parts) > 1 and _is_span(parts[-1]):
        nums.insert(0, parts.pop())
    raw = _unquote_git_path(":".join(parts))
    path = raw if raw in hunks else re.sub(r"^[ab]/", "", raw)
    if not nums:
        return path, None
    ends = nums[0].split("-", 1)  # "2-4" -> ["2", "4"]; "2" -> ["2"]
    lo, hi = int(ends[0]), int(ends[-1])
    return path, (min(lo, hi), max(lo, hi))  # a reversed range normalizes


def _is_span(field: str) -> bool:
    """Return whether the field is a numeric position: a line ("12") or a range ("2-4")."""
    ends = field.split("-", 1)
    return all(e.isdigit() for e in ends) and bool(ends[0])


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return whether two inclusive spans share a line."""
    return a[0] <= b[1] and b[0] <= a[1]


def is_grounded(file_line: str, hunks: dict[str, list[Hunk]]) -> bool:
    """Return whether the citation is in the diff.

    A path-only citation grounds on a touched path; a line or range grounds when it overlaps a
    hunk on either side, so a changed span whose first line is unchanged still grounds.

    Args:
        file_line: The citation.
        hunks: The diff's hunks by repo path.

    Returns:
        Whether the citation grounds.
    """
    path, span = _split_cite(file_line, hunks)
    if not path:
        return False
    if span is None:  # path-only: grounded if the diff touched that file at all
        return path in hunks
    return any(
        (h.new is not None and _overlaps(span, h.new))
        or (h.old is not None and _overlaps(span, h.old))
        for h in hunks.get(path, ())
    )


DedupKey = tuple[str, str, Hunk | tuple[int, int] | None]


def _dedup_key(f: Finding, hunks: dict[str, list[Hunk]]) -> DedupKey:
    """Return a finding's identity across seats and iterations: file, category and hunk.

    The hunk whose post-image span holds the citation decides first, else the one whose
    pre-image span does; the two sides of one hunk key alike. A citation outside every hunk
    keys on its own span, a path-only one on the file.

    Args:
        f: The finding.
        hunks: The diff's hunks by repo path.

    Returns:
        The key.
    """
    path, span = _split_cite(f.file_line, hunks)
    if span is None:
        return (path, f.category, None)
    mine = hunks.get(path, ())
    for h in mine:
        if h.new is not None and _overlaps(span, h.new):
            return (path, f.category, h)
    for h in mine:
        if h.old is not None and _overlaps(span, h.old):
            return (path, f.category, h)
    return (path, f.category, span)


_SEV_ORDER = {"block": 0, "warn": 1, "nit": 2}


def _ground_severity(f: Finding, ctx: ReviewContext, hunks: dict[str, list[Hunk]]) -> Severity:
    """Return the severity a finding keeps after grounding.

    A `block` survives only in a gating category, and `verify-uncovered-correctness` only when
    verify passed; otherwise it becomes `warn`. `warn` and `nit` pass through.

    Args:
        f: The finding.
        ctx: The review context, for the verify result.
        hunks: The diff's hunks by repo path; unread here.

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
    v: ReviewVerdict, ctx: ReviewContext, hunks: dict[str, list[Hunk]]
) -> ReviewVerdict:
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
    out: list[Finding] = []
    for f in v.findings:
        if not is_grounded(f.file_line, hunks):
            continue
        sev = _ground_severity(f, ctx, hunks)
        if v.verdict != "block" and sev == "block":
            sev = "warn"
        out.append(f if sev == f.severity else dataclasses.replace(f, severity=sev))
    return dataclasses.replace(v, findings=tuple(out))


def _has_new_block(
    v: ReviewVerdict, prior_keys: set[DedupKey], hunks: dict[str, list[Hunk]]
) -> bool:
    """Return whether the seat carries a surviving block that is not a prior finding.

    A block whose key dedups away is dropped from `merged_findings`, so letting it gate would
    reject the work while reporting no blocking findings.

    Args:
        v: The seat's grounded verdict.
        prior_keys: The keys of the already-injected findings.
        hunks: The diff's hunks by repo path.

    Returns:
        Whether a new block survives.
    """
    return v.verdict == "block" and any(
        f.severity == "block" and _dedup_key(f, hunks) not in prior_keys for f in v.findings
    )


def _decide(
    decision: ReviewDecision,
    n_block: int,
    quorum: int,
    *,
    n_seats_blocking: int,
    n_total: int,
) -> bool:
    """Apply the gating rule.

    Args:
        decision: The rule.
        n_block: Distinct blocking models.
        quorum: The blocks a `quorum` decision needs.
        n_seats_blocking: Seats with a surviving non-prior block.
        n_total: Seats that reviewed.

    Returns:
        Whether the panel blocks.
    """
    if decision == "advisory" or not n_total:
        return False
    if decision == "veto":
        return n_block >= 1
    if decision == "quorum":
        return n_block >= max(1, quorum)
    if decision == "all":
        return n_seats_blocking == n_total
    return False  # pragma: no cover - exhaustive


def aggregate_verdicts(
    per_seat: list[ReviewVerdict],
    ctx: ReviewContext,
    *,
    decision: ReviewDecision,
    quorum: int,
    panel_id: str,
) -> PanelResult:
    """Fold per-seat verdicts into one panel result with executable grounding.

    Every finding whose citation is not in the diff is dropped, and a block outside a gating
    category becomes a warn. Findings dedup across seats and against `prior_findings` by (path,
    category, hunk); an already-injected block neither re-surfaces nor counts toward the gate.
    Advisory never blocks; veto blocks on any surviving block; quorum needs `quorum` blocks
    counting at most one per distinct model; all needs every seat to block, an abstention being
    no block.

    Args:
        per_seat: The seats' verdicts.
        ctx: The review context.
        decision: The gating rule.
        quorum: The blocks a `quorum` decision needs.
        panel_id: The panel's id.

    Returns:
        The panel result.
    """
    hunks = diff_hunks(ctx.diff)
    prior_keys = {_dedup_key(f, hunks) for f in ctx.prior_findings}

    grounded_seats: list[ReviewVerdict] = []
    blocking_models: set[str] = set()  # distinct models with >=1 surviving non-prior block
    n_abstain = n_seats_blocking = 0
    for v in per_seat:
        if v.error is not None:
            n_abstain += 1
            grounded_seats.append(dataclasses.replace(v, findings=()))
            continue
        gv = _ground_seat(v, ctx, hunks)
        grounded_seats.append(gv)
        if _has_new_block(gv, prior_keys, hunks):
            blocking_models.add(v.model)
            n_seats_blocking += 1

    # Merge and dedup the grounded findings, dropping those already injected.
    merged: dict[DedupKey, Finding] = {}
    for v in grounded_seats:
        for f in v.findings:
            key = _dedup_key(f, hunks)
            if key in prior_keys:
                continue
            cur = merged.get(key)
            if cur is None or _SEV_ORDER[f.severity] < _SEV_ORDER[cur.severity]:
                merged[key] = f
    merged_findings = tuple(
        sorted(merged.values(), key=lambda f: (_SEV_ORDER[f.severity], f.file_line, f.category))
    )

    n_block = len(blocking_models)  # distinct-model blocking seats
    blocked = _decide(
        decision,
        n_block,
        quorum,
        n_seats_blocking=n_seats_blocking,
        n_total=len(grounded_seats),
    )

    return PanelResult(
        panel_id=panel_id,
        decision=decision,
        blocked=blocked,
        merged_findings=merged_findings,
        per_seat=tuple(grounded_seats),
        n_block=n_block,
        n_abstain=n_abstain,
    )


def render_findings(findings: tuple[Finding, ...]) -> str:
    """Return merged findings as a `[review]` block for the worker or the CLI, "" when empty."""
    if not findings:
        return ""
    lines = []
    for f in findings:
        lines.append(f"- [{f.severity}:{f.category}] ({f.file_line}) {f.title}")
        if f.detail.strip():
            lines.append(f"    {f.detail.strip()}")
    return "\n".join(lines)


__all__ = [
    "ALLOWED_BLOCK_CATEGORIES",
    "ALL_CATEGORIES",
    "Finding",
    "Hunk",
    "PanelResult",
    "ReviewContext",
    "ReviewDecision",
    "ReviewVerdict",
    "aggregate_verdicts",
    "diff_hunks",
    "is_grounded",
    "render_findings",
]
