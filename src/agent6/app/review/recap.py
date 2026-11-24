# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Review a finished session's record with the reviewer model, for `agent6 sessions review`.

The journal folds into a `RunDigest` the reviewer reads in one provider call, markdown out.
Nothing is written to the repo or the memory: the operator turns a candidate into
`agent6 memory add` or an AGENTS.md line.
`save_review` writes the rendered reviews in `<state-dir>/reviews/`, beside the provider
transcripts the review's calls leave there.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import pathlib
import shlex
import time
from collections.abc import Callable, Sized
from typing import Any, Literal

from agent6 import budget as agent6_budget
from agent6 import evidence, paths
from agent6.app import reporter as app_reporter
from agent6.app import role_providers
from agent6.app.session import setup as session_setup
from agent6.config import Config, ConfigError
from agent6.context import memory
from agent6.harness.prompt import repo_context
from agent6.prompts import review
from agent6.providers import provider_calls, provider_types
from agent6.sessions import journal
from agent6.sessions import layout as sessions_layout
from agent6.tools import sessions
from agent6.viewmodel import events as viewmodel_events
from agent6.viewmodel import listing
from agent6.viewmodel.session import log as session_log

# The harness's own interventions, named with their counts.
_NOTICE_EVENTS = (
    "nudge.no_progress",
    "nudge.tool_error",
    "nudge.loop_guard",
    "nudge.stagnation",
    "finish.refused.review",
    "nudge.metric_plateau",
    "finish.refused.metric",
    "nudge.verify_settled",
    "nudge.memory_flip",
    "jail.tool_unreachable",
)
# Each directive's words come from its accepted journal event.
_DIRECTIVES = {
    "task.queued": ("/task", "title"),
    "task.retired": ("/retire", "task_id"),
    "standing.set": ("/standing", "title"),
    "pin.added": ("/pin", "text"),
    "metric.set": ("/metric", "spec"),
}
# The caps that keep the digest within any configured model's window.
_STEERS_MAX = 20
_STEER_CHARS = 500
_DECISIONS_MAX = 20
_UNANSWERED_MAX = 20
_QUESTION_CHARS = 500
_VERIFY_SHOWN = 3
_VERIFY_TAIL_CHARS = 400
_TOOL_ERRORS_SHOWN = 5
_TOOL_ERROR_CHARS = 200
CONVERSATION_MAX_CHARS = 40_000


class RunReviewError(Exception):
    """The run-review call failed to produce a response."""


VerifyWord = Literal["not gated", "passed", "failed", "unverified"]


@dataclasses.dataclass(frozen=True, slots=True)
class VerifyRun:
    """Hold one verify run as the digest shows it.

    Attributes:
        exit_code: The gate's exit code.
        duration_s: How long the gate ran.
        tail: The clipped stderr, else stdout.
    """

    exit_code: int | None
    duration_s: float
    tail: str
    verify_id: str | None = None
    tree_before: str | None = None
    tree_after: str | None = None
    stdout_truncated: bool | None = None
    stderr_truncated: bool | None = None
    command: tuple[str, ...] = ()
    scoped: bool = False
    outcome: str = ""
    reason: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class RunDigest:
    """Hold one session's record as the reviewer reads it.

    Each list shows at most its cap; the `_total` counts are the journal's own.

    Attributes:
        session_id: The session id.
        mode: The session's mode.
        task: The task text.
        status: The session's status.
        stop_reason: The session's end reason, "" when it has none.
        verify: The gate word.
        turns: The turn count, None when the journal has no end.
        tool_calls: The tool calls made.
        tool_errors: The tool calls that failed.
        cost_usd: The session's cost.
        steers: The operator's instructions and task directives in journal order.
        decisions: The (question, answer) rulings recorded.
        unanswered_questions: The questions nobody saw, across executions.
        verify_runs: The last verify runs.
        errors_by_tool: The (tool, count) failures, most common first.
        first_errors: The first failures as "[tool] summary".
        notices: The (notice, count) harness interventions.
        memory_wrote: The memory facts this session wrote.
        memory_index: The repo's memory index, shown to every run.
        conversation: The conversation's tail.
        steers_total: The steers the journal holds.
        decisions_total: The decisions the journal holds.
        unanswered_total: The unseen questions the journal holds.
        verify_total: The verify runs the journal holds.
    """

    session_id: str
    mode: str
    task: str
    status: str
    stop_reason: str
    verify: VerifyWord
    turns: int | None
    tool_calls: int
    tool_errors: int
    cost_usd: float
    steers: tuple[str, ...] = ()
    decisions: tuple[tuple[str, str], ...] = ()
    unanswered_questions: tuple[str, ...] = ()
    verify_runs: tuple[VerifyRun, ...] = ()
    errors_by_tool: tuple[tuple[str, int], ...] = ()
    first_errors: tuple[str, ...] = ()
    notices: tuple[tuple[str, int], ...] = ()
    memory_wrote: tuple[str, ...] = ()
    memory_index: str = ""
    conversation: str = ""
    steers_total: int = 0
    decisions_total: int = 0
    unanswered_total: int = 0
    verify_total: int = 0

    def render(self) -> str:
        """Return the digest rendered as the text the reviewer is given."""
        lines = [
            f"session {self.session_id} ({self.mode}): {self.task}",
            f"ended: {self.stop_reason or self.status}; verify {self.verify};"
            f" {self.turns if self.turns is not None else '?'} turns,"
            f" {self.tool_calls} tool calls ({self.tool_errors} failed), ${self.cost_usd:.2f}",
        ]
        if self.steers:
            lines.append(f"\noperator steers ({_shown(self.steers, self.steers_total)}):")
            lines.extend(f"- {s}" for s in self.steers)
        if self.decisions:
            lines.append(f"\nrulings recorded ({_shown(self.decisions, self.decisions_total)}):")
            lines.extend(f"- Q: {q}\n  A: {a}" for q, a in self.decisions)
        if self.unanswered_questions:
            lines.append(
                "\nquestions nobody answered "
                f"({_shown(self.unanswered_questions, self.unanswered_total)}):"
            )
            lines.extend(f"- {question}" for question in self.unanswered_questions)
        if self.verify_runs:
            lines.append(f"\nverify runs ({_shown(self.verify_runs, self.verify_total)}):")
            for v in self.verify_runs:
                result = evidence.result_label(v.exit_code, v.outcome)
                tail = f": {v.tail}" if v.tail else ""
                lines.append(f"- {result} ({v.duration_s:.1f}s){tail}")
                lines.append(
                    f"  {v.verify_id or 'unrecorded id'}: {v.outcome}; "
                    f"{'Reason: ' + v.reason + '; ' if v.reason else ''}"
                    f"command {shlex.join(v.command)}; "
                    f"{'scoped' if v.scoped else 'full gate'}; "
                    f"tree {v.tree_before or 'unknown'} -> {v.tree_after or 'unknown'}; "
                    f"journal cap 2000 characters per stream; "
                    f"stdout {evidence.capture_label(v.stdout_truncated)}; "
                    f"stderr {evidence.capture_label(v.stderr_truncated)}"
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


def _unseen_questions(
    event: dict[str, Any], prompts: dict[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """Pair an unseen answer with its prompt; consumed ids may be reused on resume.

    Args:
        event: The journal event.
        prompts: The outstanding prompts by id, updated in place.

    Returns:
        The questions an unseen answer left unanswered, or none.
    """
    if event.get("type") not in ("question.prompt", "question.answer"):
        return ()
    match viewmodel_events.parse_event(event):
        case journal.QuestionPrompt(id=qid, questions=questions):
            prompts[qid] = tuple(q.question for q in viewmodel_events.question_rows(questions))
        case journal.QuestionAnswer(id=qid, unseen=unseen):
            texts = prompts.pop(qid, ())
            return texts if unseen else ()
        case _:
            pass
    return ()


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
    summary = listing.summarize_session_dir(layout.session_dir)
    steers: list[str] = []
    decisions: list[tuple[str, str]] = []
    prompts: dict[str, tuple[str, ...]] = {}
    unanswered: list[str] = []
    verify_runs: list[VerifyRun] = []
    errors: collections.Counter[str] = collections.Counter()
    first_errors: list[str] = []
    notices: collections.Counter[str] = collections.Counter()
    tool_calls = 0
    stop_reason = ""
    execution_rc: int | None = None  # this execution's last verify exit, reset at a resume
    turns: int | None = None
    try:
        raw = layout.logs_path.read_text(errors="replace")
    except OSError:
        raw = ""
    started = False
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        unanswered.extend(_clip(q, _QUESTION_CHARS) for q in _unseen_questions(event, prompts))
        etype = str(event.get("type", ""))
        if etype in journal.SESSION_START_EVENTS:
            started = True
        if directive := _DIRECTIVES.get(etype):
            verb, field = directive
            value = event.get(field)
            # A fork can carry its source's metric before its own first execution.
            if etype == "metric.set" and (not started or field not in event):
                continue
            if etype == "metric.set" and value is None:
                value = "off"
            if isinstance(value, str) and value:
                steers.append(_clip(f"{verb} {value}", _STEER_CHARS))
        elif etype == "steer.injected":
            steers.append(_clip(str(event.get("text", "")), _STEER_CHARS))
        elif etype == "decision.recorded":
            decisions.append((str(event.get("question", "")), str(event.get("answer", ""))))
        elif etype == "verify.end":
            execution_rc = _int(event["exit_code"]) if event.get("exit_code") is not None else None
            tail = str(event.get("stderr_tail") or event.get("stdout_tail") or "")
            verify_runs.append(
                VerifyRun(
                    exit_code=execution_rc,
                    duration_s=float(event.get("duration_s") or 0.0),
                    tail=_clip_tail(tail.strip(), _VERIFY_TAIL_CHARS),
                    verify_id=event.get("verify_id"),
                    tree_before=event.get("tree_before"),
                    tree_after=event.get("tree_after"),
                    stdout_truncated=event.get("stdout_truncated"),
                    stderr_truncated=event.get("stderr_truncated"),
                    command=tuple(event.get("command", ())),
                    scoped=bool(event.get("scoped", False)),
                    outcome=evidence.outcome(event),
                    reason=str(event.get("reason", "")),
                )
            )
        elif etype == "session.resume":
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
            notices[etype] += 1
        elif etype == "session.stop":
            stop_reason = str(event.get("reason", ""))
            turns = _int(event.get("turns")) if "turns" in event else turns
    verify: VerifyWord = (
        "not gated"
        if summary.verification == "not_applicable"
        else summary.verification or "unverified"
    )
    use = memory.read_use(layout.state_dir)
    wrote = tuple(sorted(n for n, u in use.items() if layout.session_id in u.writers))
    return RunDigest(
        session_id=layout.session_id,
        mode=summary.mode,
        task=summary.task_label.task,
        status=summary.status,
        stop_reason=stop_reason,
        verify=verify,
        turns=turns,
        tool_calls=tool_calls,
        tool_errors=sum(errors.values()),
        cost_usd=summary.cost_usd,
        steers=tuple(steers[:_STEERS_MAX]),
        decisions=tuple(decisions[:_DECISIONS_MAX]),
        unanswered_questions=tuple(unanswered[:_UNANSWERED_MAX]),
        verify_runs=tuple(verify_runs[-_VERIFY_SHOWN:]),
        errors_by_tool=tuple(errors.most_common()),
        first_errors=tuple(first_errors),
        notices=tuple(sorted(notices.items())),
        memory_wrote=wrote,
        memory_index=memory.clipped_index(memory.index_text(layout.state_dir)),
        conversation=sessions.conversation(layout, max_chars=max_chars),
        steers_total=len(steers),
        decisions_total=len(decisions),
        unanswered_total=len(unanswered),
        verify_total=len(verify_runs),
    )


def review_digest(
    provider: provider_calls.Provider, *, digest: str, agents_md: str = "", max_tokens: int = 4096
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
        resp: provider_types.ProviderResponse = provider.call(
            system=review.RUN_REVIEW_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "\n\n".join(parts)}],
            max_tokens=max_tokens,
        )
    except provider_types.ProviderError as exc:
        raise RunReviewError(f"provider call failed: {exc}") from exc
    text = resp.text.strip()
    if not text:
        raise RunReviewError("reviewer returned empty response")
    return text


def _clip(text: str, limit: int) -> str:
    """Return the text clipped to the limit, with the cut marked."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clip_tail(text: str, limit: int) -> str:
    """Keep the end, with a marked cut at a line boundary when one exists.

    Args:
        text: The existing output tail.
        limit: The character cap, including the cut mark.

    Returns:
        The marked trailing lines, or trailing characters for a single long line.
    """
    if len(text) <= limit:
        return text
    start = len(text) - limit + 2
    if text[start - 1] != "\n" and (boundary := text.find("\n", start)) >= 0:
        start = boundary + 1
    return "…\n" + text[start:]


def _int(value: object) -> int:
    """Return the value as an int, 0 when it is not one."""
    try:
        return int(value)  # pyrefly: ignore[bad-argument-type]
    except (TypeError, ValueError):
        return 0


def save_review(reviews_dir: pathlib.Path, *, label: str, body: str) -> pathlib.Path:
    """Write one rendered review under the reviews dir, beside the provider transcripts.

    The file is `<utc-stamp>-review.md`: a `# review: <label>` line, then the body.
    The operator finds a session's review by searching the directory.

    Args:
        reviews_dir: The `<state-dir>/reviews/` directory.
        label: What was reviewed.
        body: The review text.

    Returns:
        The file's path.
    """
    paths.mkdir_for_real_user(reviews_dir)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    content = f"# review: {label}\n\n{body.rstrip()}\n".encode()
    n = 1
    while True:
        path = reviews_dir / (f"{stamp}-review.md" if n == 1 else f"{stamp}-{n}-review.md")
        # An exclusive create claims the name, so two reviews in one second never collide.
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            n += 1
            continue
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        return path


def _not_reviewable(layout: sessions_layout.SessionLayout) -> str | None:
    """Return why the session's record cannot be reviewed yet, or None when it can."""
    if listing.session_is_live(layout.session_dir):
        return (
            f"{layout.session_id} is live; its record is not complete. Review it once it"
            f" has ended (`agent6 stop {layout.session_id}` ends it now)."
        )
    if not session_log.scan_session_log(layout.logs_path).saw_start:
        return f"{layout.session_id} never started; there is no record to review."
    return None


def _runnable_config(load: Callable[[], Config], reporter: app_reporter.Reporter) -> Config | None:
    """Return the config with a runnable reviewer, or None after reporting why there is none."""
    try:
        cfg = load()
    except ConfigError as exc:
        reporter.error(str(exc))
        return None
    cfg.require_runnable("reviewer")
    err = session_setup.check_provider_keys(cfg)
    if err is not None:
        reporter.error(f"{err}")
        return None
    return cfg


def recap(
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    *,
    config: Callable[[], Config],
    reporter: app_reporter.Reporter,
) -> int:
    """Review a session's record, print the review and save it under the state dir's reviews/.

    Read-only; no jail. The record is checked before the config loads, so a live or empty
    session is refused without a provider in hand.

    Args:
        cwd: The repository.
        layout: The session to review.
        config: Loads the effective config, the reviewer route already overridden.
        reporter: Receives the status lines, the refusals and the review text.

    Returns:
        The exit code: 0 reviewed, 2 refused, 3 budget exceeded.
    """
    if (why := _not_reviewable(layout)) is not None:
        reporter.error(why)
        return 2
    if (cfg := _runnable_config(config, reporter)) is None:
        return 2
    budget = session_setup.budget_tracker(cfg)
    reviews_dir = paths.state_dir(cwd) / "reviews"
    transcript_sink = provider_types.TranscriptSink(
        reviews_dir, objects_dir=reviews_dir / "objects"
    )
    try:
        reviewer = role_providers.build_role_provider(
            cfg, "reviewer", transcript_sink=transcript_sink, budget=budget, actor="reviewer:recap"
        )
    except provider_types.ProviderError as exc:
        reporter.error(f"provider init failed: {exc}")
        return 2
    digest = run_digest(layout)
    reporter.note(f"reviewing run: {layout.session_id}")
    try:
        text = review_digest(
            reviewer,
            digest=digest.render(),
            agents_md=repo_context.agents_md_text(cwd, cfg.sandbox.hide_paths),
        )
    except RunReviewError as exc:
        reporter.err(f"REVIEW FAILED: {exc}")
        return 2
    except agent6_budget.BudgetExceededError as exc:
        reporter.err(str(exc))  # the reason names the ledger that ran out
        return 3
    reporter.out(text)
    saved = save_review(reviews_dir, label=f"run {layout.session_id}", body=text)
    reporter.note(f"review saved: {saved}")
    budget.settle()
    reporter.err(budget.format_summary())
    return 0
