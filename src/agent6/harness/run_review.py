# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Review one finished session's record in one provider call, markdown out.

`agent6 sessions review` uses it. The journal folds into a `RunDigest` the reviewer reads;
nothing is written back, the operator turns a candidate into `agent6 memory add` or an
AGENTS.md line.
"""

from __future__ import annotations

import collections
import dataclasses
import json
from collections.abc import Sized
from typing import Literal

from agent6 import memory
from agent6.prompts import review
from agent6.providers import Provider, ProviderError, ProviderResponse
from agent6.sessions import layout as sessions_layout
from agent6.tools import sessions
from agent6.viewmodel import summarize_session_dir

# The harness's own interventions, named with their counts.
_NOTICE_EVENTS = (
    "loop.no_progress.nudge",
    "loop.tool_error.nudge",
    "loop.loop_guard.triggered",
    "loop.stagnation.nudged",
    "loop.review.rejected_finish",
    "loop.metric_plateau.nudge",
    "loop.metric_early_finish.rejected",
    "loop.verify_settled.nudge",
    "loop.memory_flip.nudged",
    "loop.memory_finish.gated",
    "loop.sandbox_tool_unreachable",
)
# The caps that keep the digest within any configured model's window.
_STEERS_MAX = 20
_STEER_CHARS = 500
_DECISIONS_MAX = 20
_VERIFY_SHOWN = 3
_VERIFY_TAIL_CHARS = 400
_TOOL_ERRORS_SHOWN = 5
_TOOL_ERROR_CHARS = 200
CONVERSATION_MAX_CHARS = 40_000


class RunReviewError(Exception):
    """The run-review call failed to produce a response."""


# `LogScan.verify_verdict`'s rule: ungated, a green final tree, this execution's own red, the rest.
VerifyWord = Literal["not gated", "passed", "failed", "unverified"]


@dataclasses.dataclass(frozen=True, slots=True)
class VerifyRun:
    """Hold one verify run as the digest shows it.

    Attributes:
        exit_code: The gate's exit code.
        duration_s: How long the gate ran.
        tail: The clipped stderr, else stdout.
    """

    exit_code: int
    duration_s: float
    tail: str


@dataclasses.dataclass(frozen=True, slots=True)
class RunDigest:
    """Hold one session's record as the reviewer reads it.

    Each list shows at most its cap; the `_total` counts are the journal's own.

    Attributes:
        session_id: The session id.
        mode: The session's mode.
        task: The task text.
        status: The session's status.
        end_reason: The session's end reason, "" when it has none.
        verify: The gate word.
        iterations: The iteration count, None when the journal has no end.
        tool_calls: The tool calls made.
        tool_errors: The tool calls that failed.
        cost_usd: The session's cost.
        steers: The operator's steers.
        decisions: The (question, answer) rulings recorded.
        verify_runs: The last verify runs.
        errors_by_tool: The (tool, count) failures, most common first.
        first_errors: The first failures as "[tool] summary".
        notices: The (notice, count) harness interventions.
        memory_wrote: The memory facts this session wrote.
        memory_index: The repo's memory index, shown to every run.
        conversation: The conversation's tail.
        steers_total: The steers the journal holds.
        decisions_total: The decisions the journal holds.
        verify_total: The verify runs the journal holds.
    """

    session_id: str
    mode: str
    task: str
    status: str
    end_reason: str
    verify: VerifyWord
    iterations: int | None
    tool_calls: int
    tool_errors: int
    cost_usd: float
    steers: tuple[str, ...] = ()
    decisions: tuple[tuple[str, str], ...] = ()
    verify_runs: tuple[VerifyRun, ...] = ()
    errors_by_tool: tuple[tuple[str, int], ...] = ()
    first_errors: tuple[str, ...] = ()
    notices: tuple[tuple[str, int], ...] = ()
    memory_wrote: tuple[str, ...] = ()
    memory_index: str = ""
    conversation: str = ""
    steers_total: int = 0
    decisions_total: int = 0
    verify_total: int = 0

    def render(self) -> str:
        """Return the digest rendered as the text the reviewer is given."""
        lines = [
            f"session {self.session_id} ({self.mode}): {self.task}",
            f"ended: {self.end_reason or self.status}; verify {self.verify};"
            f" {self.iterations if self.iterations is not None else '?'} iterations,"
            f" {self.tool_calls} tool calls ({self.tool_errors} failed), ${self.cost_usd:.2f}",
        ]
        if self.steers:
            lines.append(f"\noperator steers ({_shown(self.steers, self.steers_total)}):")
            lines.extend(f"- {s}" for s in self.steers)
        if self.decisions:
            lines.append(f"\nrulings recorded ({_shown(self.decisions, self.decisions_total)}):")
            lines.extend(f"- Q: {q}\n  A: {a}" for q, a in self.decisions)
        if self.verify_runs:
            lines.append(f"\nverify runs ({_shown(self.verify_runs, self.verify_total)}):")
            lines.extend(
                f"- exit {v.exit_code} ({v.duration_s:.1f}s): {v.tail}"
                if v.tail
                else f"- exit {v.exit_code} ({v.duration_s:.1f}s)"
                for v in self.verify_runs
            )
        if self.errors_by_tool:
            lines.append(f"\ntool errors ({self.tool_errors} of {self.tool_calls} calls):")
            lines.extend(f"- {name}: {n}" for name, n in self.errors_by_tool)
            lines.extend(f"- first: {e}" for e in self.first_errors)
        if self.notices:
            lines.append("\nharness notices: " + ", ".join(f"{k} x{n}" for k, n in self.notices))
        if self.memory_wrote:
            lines.append("\nmemory facts this session wrote: " + ", ".join(self.memory_wrote))
        if self.memory_index:
            lines.append(
                f"\nmemory index (every run on this repo is shown it):\n{self.memory_index}"
            )
        lines.append(f"\nconversation (tail):\n{self.conversation}")
        return "\n".join(lines)


def _shown(shown: Sized, total: int) -> str:
    """Return the count shown, naming how many more the journal holds."""
    left = total - len(shown)
    return f"{len(shown)}, {left} more not shown" if left > 0 else str(len(shown))


def run_digest(  # noqa: PLR0912, PLR0915 (linear fold, like scan_session_log)
    layout: sessions_layout.SessionLayout, *, max_chars: int = CONVERSATION_MAX_CHARS
) -> RunDigest:
    """Fold a session's journal into a digest.

    A torn line is skipped and a missing journal reads as no events.

    Args:
        layout: The session's layout.
        max_chars: The cap on the conversation tail.

    Returns:
        The digest.
    """
    summary = summarize_session_dir(layout.session_dir)
    steers: list[str] = []
    decisions: list[tuple[str, str]] = []
    verify_runs: list[VerifyRun] = []
    errors: collections.Counter[str] = collections.Counter()
    first_errors: list[str] = []
    notices: collections.Counter[str] = collections.Counter()
    tool_calls = 0
    end_reason = ""
    all_passed: bool | None = None
    execution_rc: int | None = None  # this execution's last verify exit, reset at a resume
    iterations: int | None = None
    try:
        raw = layout.logs_path.read_text(errors="replace")
    except OSError:
        raw = ""
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        etype = str(event.get("type", ""))
        if etype == "loop.steer.injected":
            steers.append(_clip(str(event.get("text", "")), _STEER_CHARS))
        elif etype == "loop.decision.recorded":
            decisions.append((str(event.get("question", "")), str(event.get("answer", ""))))
        elif etype == "verify.end":
            execution_rc = _int(event.get("exit_code"))
            tail = str(event.get("stderr_tail") or event.get("stdout_tail") or "")
            verify_runs.append(
                VerifyRun(
                    exit_code=execution_rc,
                    duration_s=float(event.get("duration_s") or 0.0),
                    tail=_clip(tail.strip(), _VERIFY_TAIL_CHARS),
                )
            )
        elif etype == "loop.resume.start":
            execution_rc = None
        elif etype == "tool.call":
            tool_calls += 1
        elif etype == "tool.result" and event.get("ok") is False:
            name = str(event.get("name", "?"))
            errors[name] += 1
            if len(first_errors) < _TOOL_ERRORS_SHOWN:
                first_errors.append(
                    f"[{name}] {_clip(str(event.get('summary', '')), _TOOL_ERROR_CHARS)}"
                )
        elif etype in _NOTICE_EVENTS:
            notices[etype.removeprefix("loop.")] += 1
        elif etype == "session.end":
            end_reason = str(event.get("reason", ""))
            passed = event.get("all_passed")
            all_passed = passed if isinstance(passed, bool) else None
            iterations = _int(event.get("iterations")) if "iterations" in event else iterations
    # `LogScan.verify_verdict`'s rule: "failed" only on this execution's own red verify.
    if summary.mode != "run" or (end_reason and all_passed is None):
        verify = "not gated"
    elif all_passed is True:
        verify = "passed"
    elif execution_rc not in (None, 0):
        verify = "failed"
    else:
        verify = "unverified"
    use = memory.read_use(layout.state_dir)
    wrote = tuple(sorted(n for n, u in use.items() if layout.session_id in u.writers))
    return RunDigest(
        session_id=layout.session_id,
        mode=summary.mode,
        task=summary.task,
        status=summary.status,
        end_reason=end_reason,
        verify=verify,
        iterations=iterations,
        tool_calls=tool_calls,
        tool_errors=sum(errors.values()),
        cost_usd=summary.cost_usd,
        steers=tuple(steers[:_STEERS_MAX]),
        decisions=tuple(decisions[:_DECISIONS_MAX]),
        verify_runs=tuple(verify_runs[-_VERIFY_SHOWN:]),
        errors_by_tool=tuple(errors.most_common()),
        first_errors=tuple(first_errors),
        notices=tuple(sorted(notices.items())),
        memory_wrote=wrote,
        memory_index=memory.clipped_index(memory.index_text(layout.state_dir)),
        conversation=sessions.conversation(layout, max_chars=max_chars),
        steers_total=len(steers),
        decisions_total=len(decisions),
        verify_total=len(verify_runs),
    )


def run_review(
    provider: Provider, *, digest: str, agents_md: str = "", max_tokens: int = 4096
) -> str:
    """Ask the reviewer model to review a run from its digest.

    Args:
        provider: The reviewer's provider.
        digest: The rendered digest.
        agents_md: The repo's AGENTS.md, given whole so a candidate line is judged against it.
        max_tokens: The output cap of the call.

    Returns:
        The review as markdown.

    Raises:
        RunReviewError: When the call fails or returns nothing.
    """
    parts: list[str] = []
    if agents_md.strip():
        parts.append(f"AGENTS.md:\n{agents_md.strip()}")
    parts.append(f"RUN RECORD:\n{digest}")
    try:
        resp: ProviderResponse = provider.call(
            system=review.RUN_REVIEW_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "\n\n".join(parts)}],
            max_tokens=max_tokens,
        )
    except ProviderError as exc:
        raise RunReviewError(f"provider call failed: {exc}") from exc
    text = resp.text.strip()
    if not text:
        raise RunReviewError("reviewer returned empty response")
    return text


def _clip(text: str, limit: int) -> str:
    """Return the text clipped to the limit, with the cut marked."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _int(value: object) -> int:
    """Return the value as an int, 0 when it is not one."""
    try:
        return int(value)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return 0
