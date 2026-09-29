# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Word and mark run state the same way on the CLI, the TUI and the web.

The web client reads the rendered fields the view models carry (a task's glyph, a
transition's line, a cost cell) rather than keeping its own maps. `SPINNER_FRAMES` is
the one constant it copies, since it animates locally between polls; `tests/web`
pins the copy.
"""

from __future__ import annotations

import time
from typing import Literal

from agent6 import budget
from agent6.sessions import manifest

# Text characters, not graphics, so every terminal font renders them.
TASK_STATUS_GLYPH = {
    "passed": "✓",
    "failed": "✗",
    "in_progress": "▸",
    "pending": "·",
    "skipped": "–",  # noqa: RUF001
    "obsolete": "×",  # noqa: RUF001
}


def format_model_route(driver: manifest.ModelBrief | None) -> str:
    """Return a manifest driver as provider/model, or its model alone, or ""."""
    if driver is None or not driver.model:
        return ""
    return f"{driver.provider}/{driver.model}" if driver.provider else driver.model


SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def clip_cell(text: str, width: int) -> str:
    """Return the text as one line of at most `width` characters, an ellipsis marking a cut."""
    line = " ".join(text.split())
    return line if len(line) <= width else line[: max(1, width - 1)] + "\u2026"


def dead_run_note(word: str, detail: str) -> tuple[str, str]:
    """Word a run with no live worker for every surface.

    Args:
        word: The run's status word.
        detail: The status detail beside it.

    Returns:
        The state the run is in and the one action left (the TUI's composer line);
        `("", "")` for anything live or that ended normally.
    """
    if word == "parked":
        why = f" ({detail})" if detail else ""
        return f"parked at submission{why}", "type the go-ahead below (Enter resumes)"
    if word == "created":
        return "the run has not started", "type a follow-up below (Enter resumes)"
    if word == "stale":
        return (
            "worker exited without finishing (crashed or killed)",
            "type a follow-up below (Enter resumes)",
        )
    if word == "unreadable":
        # A corrupt manifest cannot be resumed, so there is no composer action.
        detail_s = f" ({detail})" if detail else ""
        return f"session state is unreadable{detail_s}", ""
    return "", ""


def spinner_frame(tick: int) -> str:
    """Return the braille spinner frame for a tick."""
    return SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]


def short_task_id(task_id: str) -> str:
    """Return a task id as an operator reads and types it.

    Args:
        task_id: The graph's id for the task.

    Returns:
        The run's own count with the padding zeros dropped, or the last six characters
        of an opaque id carried in from before.
    """
    return task_id.lstrip("0") or "0" if task_id.isdigit() else task_id[-6:]


def format_when(epoch: float, *, short: bool = False) -> str:
    """Return a listing's `when` cell in local time.

    Args:
        epoch: The time as epoch seconds.
        short: Keep only the time for today and only the date for older, for a
            narrow terminal.

    Returns:
        `MM-DD HH:MM`, or the short form.
    """
    if not short:
        return time.strftime("%m-%d %H:%M", time.localtime(epoch))
    today = time.localtime().tm_yday, time.localtime().tm_year
    then = time.localtime(epoch)
    return time.strftime("%H:%M" if (then.tm_yday, then.tm_year) == today else "%m-%d", then)


def format_age(seconds: float) -> str:
    """Return a compact age for status cells: `<1m`, `12m`, `3h`, `2d`, floored."""
    s = max(0.0, seconds)
    if s < 60:
        return "<1m"
    if s < 3600:
        return f"{int(s // 60)}m"
    if s < 172800:
        return f"{int(s // 3600)}h"
    return f"{int(s // 86400)}d"


def machine_state_mark(*, is_current: bool, is_visited: bool) -> str:
    """Return the mark before a machine state: current, visited or none."""
    return "▸" if is_current else ("·" if is_visited else " ")


def format_transition(seq: int, state: str, label: str, goto: str, detail: str = "") -> str:
    """Return a machine transition as every surface prints it.

    Args:
        seq: The transition's sequence number.
        state: The state left.
        label: The transition's label.
        goto: The state entered.
        detail: The failure evidence, appended when there is any.

    Returns:
        `[seq] state --label--> goto`, then ` -- detail` when given.
    """
    line = f"[{seq}] {state} --{label}--> {goto}"
    return f"{line} -- {detail}" if detail else line


def format_cost_cell(usd: float, *, partial: bool = False, plan_points: float | None = None) -> str:
    """Return a listing's cost cell.

    Args:
        usd: The dollar figure.
        partial: The figure is a lower bound.
        plan_points: The plan points consumed on a subscription-metered execution.

    Returns:
        The points as `Npt` when given, "" for a clean $0, else `format_usd`.
    """
    if plan_points is not None:
        return f"{plan_points:g}pt"
    if usd <= 0 and not partial:
        return ""
    return budget.format_usd(usd, partial=partial)


def budget_usd_text(
    usd_total: float, *, partial: bool, usd_cap: float, usd_prior_executions: float
) -> str:
    """Return the run view's cost line.

    Args:
        usd_total: The cumulative spend across executions.
        partial: The figure is a lower bound.
        usd_cap: This execution's cap; -1 for unlimited, 0 for none set.
        usd_prior_executions: The spend banked by earlier executions.

    Returns:
        The cumulative figure, then this execution's spend against its cap when earlier
        executions spent; `(unlimited)` for a cap of -1.
    """
    text = budget.format_usd(usd_total, partial=partial)
    if usd_cap > 0:
        cap = budget.format_usd(usd_cap)
        if usd_prior_executions > 0:
            execution = budget.format_usd(
                max(0.0, usd_total - usd_prior_executions), partial=partial
            )
            return f"{text} · execution {execution} / {cap}"
        return f"{text} / {cap}"
    return f"{text} (unlimited)" if usd_cap == -1 else text


# The mark on the lane a fan-out's compare ranked first.
WINNER_GLYPH = "★"


def winner_id(session_id: str, *, winner: bool) -> str:
    """Return a listing row's id cell, the winner glyph suffixed on a compare winner."""
    return f"{session_id} {WINNER_GLYPH}" if winner else session_id


def format_branch(run_branch: str, base_branch: str, merged_into: str) -> str:
    """Word where a run's work lives, for every header.

    Args:
        run_branch: The run's branch, "" for a session without one.
        base_branch: The branch a merge lands on.
        merged_into: The branch the run was merged into, "" while unmerged.

    Returns:
        The run branch merged into its base, or the run branch and the base a merge
        lands on; "" for a session with no run branch.
    """
    if not run_branch:
        return ""
    if merged_into:
        return f"{run_branch} (merged into {merged_into})"
    return f"{run_branch} → merges into {base_branch}" if base_branch else run_branch


def format_lineage(parent: str | None, turn: int | None, sha: str | None) -> str:
    """Word where a forked run came from, for every header.

    Args:
        parent: The parent session's id, None for a run that is not a fork.
        turn: The turn the fork left from.
        sha: The commit the fork left from.

    Returns:
        `<parent>@turn <n> (<sha12>)`, or "" for a run that is not a fork.
    """
    if not parent:
        return ""
    sha_note = f" ({sha[:12]})" if sha else ""
    return f"{parent}@turn {turn}{sha_note}"


def format_compare(compare: manifest.CompareStamp | None) -> tuple[str, str] | None:
    """Word a lane's fan-out compare outcome.

    Args:
        compare: The lane's compare stamp, None for a run outside a compared fan-out.

    Returns:
        The headline (`rank 1/2 · winner · judge ($0.0102)`, the figure being the judge
        call's cost for the whole group) and the judge's rationale (empty for a
        mechanical ranking), or None without a stamp.
    """
    if compare is None:
        return None
    parts = [f"rank {compare.rank}/{compare.of}"]
    if compare.winner:
        parts.append("winner")
    if compare.ranked_by:
        by = compare.ranked_by
        if compare.judge_cost_usd > 0 or compare.judge_cost_partial:
            cost = budget.format_usd(compare.judge_cost_usd, partial=compare.judge_cost_partial)
            by += f" ({cost})"
        parts.append(by)
    return " · ".join(parts), compare.rationale


def status_label(status: str, reason: str = "") -> str:
    """Return the label for a run outcome: the word, then the reason with underscores spaced."""
    return status if not reason else f"{status} · {reason.replace('_', ' ')}"


# Status words that already name the session's mode.
_MODE_IMPLIED: dict[str, str] = {"planned": "plan", "answered": "ask"}

# An unmerged mark means something only once the run is over.
_ENDED_WORDS = frozenset({"passed", "failed", "finished", "stopped", "undone"})


def lane_count(n: int) -> str:
    """Return the lane count as every listing words it: "3 lanes", "1 lane"."""
    return f"{n} lane{'' if n == 1 else 's'}"


def lane_id_cell(id_cell: str, depth: int = 1) -> str:
    """Return a lane's id cell nested under its fan-out's row, stepped in once per depth."""
    return f"{'  ' * depth}└ {id_cell}"


def listing_status_label(
    mode: str, status: str, reason: str = "", *, unmerged: bool = False
) -> str:
    """Return the listing status cell every hub shows.

    Args:
        mode: The session's mode.
        status: The status word.
        reason: The status detail.
        unmerged: The run's branch holds commits its base does not.

    Returns:
        The mode folded in when the word does not imply it ("plan · running", a bare
        "planned"), the reason, and the unmerged mark on an ended run.
    """
    label = status_label(status, reason)
    if mode not in ("run", "?", "") and _MODE_IMPLIED.get(status) != mode:
        label = f"{mode} · {label}"
    if unmerged and status in _ENDED_WORDS:
        label = f"{label} · unmerged"
    return label


StatusLevel = Literal["ok", "info", "active", "warn", "error", "neutral"]

# Each surface maps a level to its own palette; an unlisted word is neutral and renders plain.
STATUS_LEVEL: dict[str, StatusLevel] = {
    "starting": "active",
    "running": "active",
    "waiting": "warn",
    "parked": "warn",
    "created": "warn",
    "stopped": "warn",
    "stale": "error",
    "failed": "error",
    "unreadable": "error",
    "passed": "ok",
    "answered": "ok",
    "ok": "ok",
    "planned": "info",
}


def status_level(status: str) -> StatusLevel:
    """Return the level a status word renders at, neutral for an unlisted word."""
    return STATUS_LEVEL.get(status, "neutral")
