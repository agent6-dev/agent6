# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-surface presentation constants shared by the CLI, TUI, and web.

The single source of truth for how run/task state reads to a human, so the same
state never renders differently across surfaces (per-front-end glyph maps
drift). The web SPA can't import Python, so it mirrors these exact
characters in ui/web/client.js; keep them in sync.
"""

from __future__ import annotations

from agent6.sessions.manifest import CompareStamp

# Task-node status glyphs. Text characters (not graphics) so every terminal font
# renders them. ruff's ambiguous-glyph rule (RUF001) flags the en-dash /
# multiplication-sign, which is the intended distinct look here.
TASK_STATUS_GLYPH = {
    "passed": "✓",
    "failed": "✗",
    "in_progress": "▸",
    "pending": "·",
    "skipped": "–",  # noqa: RUF001
    "obsolete": "×",  # noqa: RUF001
}


SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def spinner_frame(tick: int) -> str:
    """The braille spinner frame for *tick*, one owner for every surface."""
    return SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]


def format_cost(usd: float, *, partial: bool = False) -> str:
    """Render a USD cost identically on every surface: cents at >= $1, four
    decimals below (so small runs aren't all '$0.00'), with a leading '~' when
    the figure is a known under-estimate (a model without price data). The web
    SPA mirrors this in client.js's fmtUsd."""
    prefix = "~" if partial else ""
    return f"{prefix}${usd:.2f}" if usd >= 0.995 else f"{prefix}${usd:.4f}"


# The fan-out winner marker, shown on listing rows (a lane the auto-compare
# ranked first). Text glyph so every terminal font renders it; the web SPA
# mirrors it in page.py.
WINNER_GLYPH = "★"


def format_compare(compare: CompareStamp | None) -> tuple[str, str] | None:
    """A lane's fan-out compare outcome as ``(headline, rationale)``, or None when
    the run carries no ``compare`` stamp. The headline reads e.g.
    ``rank 1/2 · winner · judge ($0.0102)``; the parenthesised figure is the
    judge call's cost for the whole group, present whenever a judge call was
    made (a ``~`` marks an unpriced lower bound). The rationale is the judge's
    text, empty for a mechanical ranking. Shared by `sessions show` and the TUI run
    header; the web SPA renders the same stamp fields from the snapshot JSON."""
    if compare is None:
        return None
    parts = [f"rank {compare.rank}/{compare.of}"]
    if compare.winner:
        parts.append("winner")
    if compare.ranked_by:
        by = compare.ranked_by
        if compare.judge_cost_usd > 0 or compare.judge_cost_partial:
            cost = format_cost(compare.judge_cost_usd, partial=compare.judge_cost_partial)
            by += f" ({cost})"
        parts.append(by)
    return " · ".join(parts), compare.rationale


def status_label(status: str, reason: str = "") -> str:
    """The one human label for a run outcome: the status word (from
    ``status_word``), plus the reason with underscores spaced when there is one
    ("failed · provider error"). Shared by every hub listing, the run header, and
    the web wire form, so the same run reads the same on every surface."""
    return status if not reason else f"{status} · {reason.replace('_', ' ')}"
