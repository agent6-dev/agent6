# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-surface presentation constants shared by the CLI, TUI, and web.

The single source of truth for how run/task state reads to a human, so the same
state never renders differently across surfaces (per-front-end glyph maps
drift). The web SPA can't import Python, so it mirrors these exact
characters in ui/web/client.js; keep them in sync.
"""

from __future__ import annotations

import time
from typing import Literal

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


def format_when(epoch: float) -> str:
    """A listing's `when` column: local `MM-DD HH:MM`."""
    return time.strftime("%m-%d %H:%M", time.localtime(epoch))


def format_cost(usd: float, *, partial: bool = False) -> str:
    """Render a USD cost identically on every surface: cents at >= $1, four
    decimals below (so small runs aren't all '$0.00'), with a leading '~' when
    the figure is a known under-estimate (a model without price data). The web
    SPA mirrors this in client.js's fmtUsd."""
    prefix = "~" if partial else ""
    return f"{prefix}${usd:.2f}" if usd >= 0.995 else f"{prefix}${usd:.4f}"


def machine_state_mark(*, is_current: bool, is_visited: bool) -> str:
    """The mark before a machine state in the overview: the current state,
    a visited one, or none. Text glyphs (the task-status set's), mirrored by
    the web SPA."""
    return "▸" if is_current else ("·" if is_visited else " ")


def format_transition(seq: int, state: str, label: str, goto: str, detail: str = "") -> str:
    """One journaled machine transition as every surface prints it:
    `[seq] state --label--> goto`, the failure evidence appended when there is
    any. The web SPA mirrors the shape."""
    line = f"[{seq}] {state} --{label}--> {goto}"
    return f"{line} -- {detail}" if detail else line


def format_cost_cell(usd: float, *, partial: bool = False) -> str:
    """A listing's cost cell: blank for a genuinely clean $0, else
    `format_cost` (an all-unpriced run's `~$0.0000` is information: spend
    happened, price unknown)."""
    if usd <= 0 and not partial:
        return ""
    return format_cost(usd, partial=partial)


# The fan-out winner marker, shown on listing rows (a lane the auto-compare
# ranked first). Text glyph so every terminal font renders it; the web SPA
# mirrors it in page.py.
WINNER_GLYPH = "★"


def winner_id(session_id: str, *, winner: bool) -> str:
    """The id cell of a listing row: the winner glyph suffixed on a fan-out
    compare winner (folded into the cell so column widths stay aligned)."""
    return f"{session_id} {WINNER_GLYPH}" if winner else session_id


def format_branch(run_branch: str, base_branch: str, merged_into: str) -> str:
    """Where a run's work lives, one wording for every header: the run branch
    merged into its base, or the run branch and the base a merge lands on.
    "" for a session with no run branch (an ask, branch_per_run off)."""
    if not run_branch:
        return ""
    if merged_into:
        return f"{run_branch} (merged into {merged_into})"
    return f"{run_branch} → merges into {base_branch}" if base_branch else run_branch


def format_compare(compare: CompareStamp | None) -> tuple[str, str] | None:
    """A lane's fan-out compare outcome as `(headline, rationale)`, or None when
    the run carries no `compare` stamp. The headline reads e.g.
    `rank 1/2 · winner · judge ($0.0102)`; the parenthesised figure is the
    judge call's cost for the whole group, present whenever a judge call was
    made (a `~` marks an unpriced lower bound). The rationale is the judge's
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
    `status_word`), plus the reason with underscores spaced when there is one
    ("failed · provider error"). Shared by every hub listing, the run header, and
    the web wire form, so the same run reads the same on every surface."""
    return status if not reason else f"{status} · {reason.replace('_', ' ')}"


StatusLevel = Literal["ok", "info", "active", "warn", "error", "neutral"]

# How a status word (a run's from `listing.status_word`, a machine's from
# `machine_status_word`, the hub's pre-start words) reads: the level, decided
# once here; each surface maps a level to its own palette (Rich style, ANSI
# SGR, CSS class). A word not listed is neutral and renders plain: a clean
# finish carries no signal worth a colour, while a lost worker or a parked
# submission must never fade into the listing.
STATUS_LEVEL: dict[str, StatusLevel] = {
    "starting": "active",
    "running": "active",
    "waiting": "warn",  # blocked on the operator (approval / question)
    "parked": "warn",  # needs a resume to start
    "stopped": "warn",  # the operator's own act, not a failure
    "stale": "error",  # a lost worker: a crash is not neutral
    "failed": "error",
    "unreadable": "error",  # a corrupt machine source or journal
    "passed": "ok",
    "answered": "ok",  # an ask that answered is terminal success
    "ok": "ok",  # a machine's clean end
    "planned": "info",  # a completed plan verifies nothing: informational
}


def status_level(status: str) -> StatusLevel:
    return STATUS_LEVEL.get(status, "neutral")
