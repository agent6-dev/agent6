# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The context compaction driver.

Both strategies share the size, manual and idle restart rules and DAG check-off.
Tiered elides old results and restarts from prose. Endpoint adopts a signed block.
"""

from __future__ import annotations

import dataclasses
import functools
import time
import uuid
from collections.abc import Callable
from typing import ClassVar

import pydantic

from agent6 import budget
from agent6.graph import curator as graph_curator
from agent6.graph import models
from agent6.harness import advice
from agent6.harness.compaction import elision as compaction_elision
from agent6.harness.compaction import retention, strategy, writers
from agent6.harness.compaction import rules as compaction_rules
from agent6.harness.compaction import window as compaction_window
from agent6.prompts import compaction
from agent6.providers import (
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    provider_types,
)
from agent6.sessions import journal
from agent6.tools import waits


def _take_focus(context: strategy.Context) -> str | None:
    """Consume the pending manual summary request.

    Args:
        context: The prepared turn.

    Returns:
        The operator's focus, empty for an unfocused request, or None for no request.
    """
    focus = context.compact_requested()
    if focus is not None:
        context.compact_clear()
        detail = f" (focus: {focus[:80]})" if focus else ""
        context.log(f"LOOP: operator requested a manual compaction{detail}")
        context.emit(journal.CompactRequested(focus=focus))
    return focus


@dataclasses.dataclass(slots=True)
class _Scheduled:
    """Drop old results in steps and restart from one accepted summary.

    Attributes:
        tier2_floor_chars: Minimum size for the next automatic restart in this execution.
        cold_notice_execution: The execution that received the cold-cache notice.
    """

    endpoint: ClassVar[bool] = False
    tier2_floor_chars: int = 0
    cold_notice_execution: str | None = None

    def idle(self, context: strategy.Context, request_started: float) -> waits.Deadline | None:
        """Offer one summary during a wait above the idle and restart floors.

        Args:
            context: The conversation with the worker's reply.
            request_started: The monotonic start of the last worker attempt.

        Returns:
            The turn's deadline, or None when it has no idle compaction due.
        """
        idle = context.settings.idle
        if idle is None or idle.seconds == -1:
            return None
        total = context.prefix_chars + compaction_window.context_chars(context.conversation)
        if total <= idle.min_chars or total < self.tier2_floor_chars:
            return None
        # Gate notices can follow a prose reply while the run parks for the operator.
        sent_count = len(context.conversation.complete(keep_reply=True)) if self.endpoint else None
        return waits.Deadline(
            request_started + idle.seconds,
            lambda: self.summarize_and_restart(
                context, idle_gap_s=time.monotonic() - request_started, sent_count=sent_count
            ),
        )

    def _scheduled(
        self, context: strategy.Context, *, drop: Callable[[strategy.Context], None] | None = None
    ) -> bool:
        """Check for a restart before allowing tier 1 to rewrite the request prefix.

        Args:
            context: The prepared turn.
            drop: The local drop step, if it still needs to run.

        Returns:
            True if a summary replaced the history.
        """
        conversation = context.conversation
        forced = _take_focus(context)
        total = compaction_window.context_chars(conversation) + context.prefix_chars
        # Below four messages a restart loses more than it saves; a forced compaction skips
        # the growth floor.
        over = total > context.settings.summarize_at_chars and total >= self.tier2_floor_chars
        if (forced is not None or over) and len(conversation) > 3:
            return self.summarize_and_restart(context, focus=forced)
        if forced is not None:
            # The request was consumed above, so the refusal is said, not silent.
            context.log("LOOP: manual compaction skipped: too little history to summarize")
            context.emit(journal.CompactRefused(reason="too little history to summarize"))

        if drop is not None:
            drop(context)
        return False

    def summarize_and_restart(
        self,
        context: strategy.Context,
        *,
        focus: str | None = None,
        idle_gap_s: float | None = None,
        sent_count: int | None = None,
        cold: bool = False,
    ) -> bool:
        """Replace history only after either writer returns an accepted summary.

        Args:
            context: The prepared turn.
            focus: The operator's focus for the summary.
            idle_gap_s: Seconds since the last worker request began for an idle summary.
            sent_count: The endpoint request length at the start of the wait.
            cold: Use the helper directly for a cold local resume.

        Returns:
            True if the history was replaced.
        """
        conversation = context.conversation
        messages = conversation.messages
        reason = "idle" if idle_gap_s is not None else "size" if focus is None else "requested"
        if self.endpoint:
            sent = len(messages) if sent_count is None else sent_count
            replay = retention.Replay(messages[1:sent], (), messages[sent:])
        else:
            replay = retention.select(messages, context.settings, context.task.text, context.pins)
        open_tasks = (
            advice.open_subtasks(context.curator.nodes()) if context.curator is not None else []
        )
        written = write_summary(
            context,
            replay=replay,
            writer="endpoint"
            if self.endpoint
            else "helper"
            if cold
            else context.settings.summarizer,
            open_tasks=open_tasks,
            focus=focus or "",
        )
        if written is None:
            return False
        raw = written.raw
        summary = compaction_window.strip_checkoff(raw)
        # Only after the narrative passed: bookkeeping alone never mutates the DAG.
        if open_tasks:
            apply_checkoff(context, raw, valid_ids={tid for tid, _ in open_tasks})
        keep = replay.tail
        if written.block is not None:
            conversation.adopt(
                written.block,
                keep,
                context.endpoint_notice(),
                awaiting_reply=context.awaiting_reply,
            )
        else:
            conversation.restart(
                context.restart_notice(replay.quotes, task=replay.task, kept=keep) + summary,
                keep=keep,
                quotes=replay.quotes,
                endpoint_task=context.task if self.endpoint else None,
            )
        self._set_restart_floor(context)
        context.emit(
            journal.CompactSummarizeDone(
                reason="cold_start" if cold else reason,
                gap_s=idle_gap_s,
                writer=written.writer,
                fallback_reason=written.fallback_reason,
                summary_chars=len(summary),
                summary=summary,
                kept_messages=len(keep),
            )
        )
        if idle_gap_s is not None and written.writer != "helper":
            self._report_cold_cache(context, written.usage, idle_gap_s)
        return True

    def _set_restart_floor(self, context: strategy.Context) -> None:
        """Require growth after a restart, measured with the same prefix as its trigger."""
        self.tier2_floor_chars = int(
            (
                compaction_window.context_chars(context.conversation)
                + context.conversation.pending_notice_chars
                + context.prefix_chars
            )
            * 1.25
        )

    def _report_cold_cache(
        self, context: strategy.Context, usage: provider_types.ProviderUsage, gap_s: float
    ) -> None:
        """Report the first measured cold idle summary in this execution.

        Args:
            context: The turn that compacted.
            usage: The accepted worker summary's measured input.
            gap_s: Seconds since the last worker request began.
        """
        idle = context.settings.idle
        cached = usage.cache_read_tokens
        uncached = usage.input_tokens + (usage.cache_write_tokens or 0)
        if (
            idle is None
            or cached is None
            or cached >= uncached
            or self.cold_notice_execution == context.execution_id
        ):
            return
        self.cold_notice_execution = context.execution_id
        setting = f"providers.{idle.provider}.idle_compact_s"
        # One cold call does not prove the cause: a provider can miss its cache inside the lifetime.
        message = (
            f"Idle summary {gap_s:.1f}s after the last request began:"
            f" {uncached:,} uncached, {cached:,} cached input tokens."
            f" {setting} = {idle.seconds:g} may be longer than the cache lasts."
        )
        context.log(message)
        context.emit(
            journal.CompactIdleCacheCold(
                setting=setting,
                idle_compact_s=idle.seconds,
                gap_s=gap_s,
                uncached_tokens=uncached,
                cached_tokens=cached,
                message=message,
            )
        )


@dataclasses.dataclass(slots=True)
class Endpoint(_Scheduled):
    """Keep local tool results and adopt the endpoint's signed summary.

    Attributes:
        cleared: The tool-use ids of the results the endpoint clears on each request.
    """

    endpoint = True
    cleared: tuple[str, ...] = ()

    def compact(self, context: strategy.Context) -> bool:
        """Check for a summary without rewriting tool results locally.

        Args:
            context: The completed worker request.

        Returns:
            True if a summary replaced the history.
        """
        return self._scheduled(context)

    def clearing(self, context: strategy.Context) -> provider_types.ClearToolResults | None:
        """Return the worker request's clearing edit, moving its point past the drop threshold.

        Args:
            context: The prepared turn.

        Returns:
            The edit, or None when the endpoint clears nothing.
        """
        if not context.settings.endpoint.clears_tool_results:
            return None
        before = self.cleared
        self.cleared, edit = compaction_rules.clearing(
            context.conversation, context.settings.drop_at_chars, before
        )
        extends = self.cleared[: len(before)] == before
        added = self.cleared[len(before) :] if extends else self.cleared
        if added:
            context.log(
                f"LOOP: compaction asks the endpoint to clear {len(added)} more old"
                f" tool_result blocks ({len(self.cleared)} cleared on each request)"
            )
            context.emit(journal.CompactDropped(count=len(added), calls=list(added)))
        return edit


class Tiered(_Scheduled):
    """Drop local tool results before the next worker call."""

    def compact(self, context: strategy.Context) -> bool:
        """Run the ordinary size and manual checks, then the drop step.

        Args:
            context: The prepared worker request.

        Returns:
            True when the worker needs fresh focus notices after a restart.
        """
        return self._scheduled(context, drop=self.drop)

    def cold_start(self, context: strategy.Context) -> journal.ColdStartAction:
        """Apply `compaction.cold_start` to the first request of a cold resume.

        Args:
            context: The prepared worker request.

        Returns:
            What happened to the history; `summarize` and `truncate` restarted it, and
            `resend` also names a restart that had nothing to replace.

        Raises:
            BudgetExceededError: The budget refused the summary.
            ProviderError: The provider refused the summary.
            ProviderAborted: The operator stopped the run.
            ProviderInterrupted: A steer is pending.
        """
        conversation = context.conversation
        mode = context.settings.cold_start
        if mode == "resend":
            return mode
        if mode == "drop":
            self.drop(context, cold=True)
            return mode
        replay = retention.select(
            conversation.messages, context.settings, context.task.text, context.pins
        )
        if not replay.replaced:
            # The caps keep everything, so the history is sent as it is.
            return "resend"
        if mode == "truncate":
            conversation.restart(
                context.restart_notice(
                    replay.quotes, task=replay.task, kept=replay.tail, truncated=True
                ),
                keep=replay.tail,
                quotes=replay.quotes,
            )
            self._set_restart_floor(context)
        elif not self.summarize_and_restart(context, cold=True, focus=_take_focus(context)):
            return "summary_failed"
        elided, gists = compaction_rules.count_elisions(conversation)
        context.emit(journal.CompactRestored(elided=elided, gists=gists))
        return mode

    def drop(self, context: strategy.Context, *, cold: bool = False) -> None:
        """Apply the first tier in place without a restart."""
        conversation = context.conversation
        stats = compaction_rules.compact_old_tool_results(
            conversation,
            max_total_bytes=None if cold else context.settings.drop_at_chars,
            keep_recent=2,
            protect_paths=compaction_rules.recently_edited_paths(conversation),
            gister=functools.partial(distill_gists, context)
            if context.settings.elision_gists
            else None,
        )
        n_deduped = len(stats.deduped_calls)
        n_elided = len(stats.elided_calls)
        n_gisted = len(stats.gist_paths)
        n_demoted = len(stats.demoted_paths)
        rewrote = n_deduped or n_elided or n_demoted
        if context.settings.keep_thinking_turns > 0 and (cold or rewrote):
            # Warm, strip only alongside a result rewrite, so thinking adds no cache break.
            n_messages, n_chars = compaction_window.strip_old_thinking(
                conversation, keep_messages=context.settings.keep_thinking_turns
            )
            if n_messages:
                context.log(
                    f"LOOP: compaction dropped thinking from {n_messages} old messages"
                    f" ({n_chars} chars)"
                )
                context.emit(journal.CompactThinkingDropped(messages=n_messages, chars=n_chars))
        if n_deduped:
            context.log(f"LOOP: compaction deduplicated {n_deduped} identical tool results")
            context.emit(journal.CompactDeduped(count=n_deduped, calls=list(stats.deduped_calls)))
        if n_elided:
            detail = f", {n_gisted} kept as distilled gists" if n_gisted else ""
            context.log(f"LOOP: compaction elided {n_elided} old tool_result blocks{detail}")
            context.emit(journal.CompactDropped(count=n_elided, calls=list(stats.elided_calls)))
        if n_demoted:
            context.log(f"LOOP: compaction demoted {n_demoted} gists to bare placeholders")
        if n_gisted or n_demoted:
            context.emit(
                journal.CompactGistDone(
                    gisted=n_gisted,
                    demoted=n_demoted,
                    paths=list(stats.gist_paths),
                    demoted_paths=list(stats.demoted_paths),
                )
            )


def distill_gists(
    context: strategy.Context, requests: tuple[compaction_elision.GistRequest, ...]
) -> dict[str, str]:
    """Distill about-to-be-elided file reads into one-line gists.

    The summarizer makes the call. A provider error returns {} and every
    victim gets the bare placeholder, so gisting never breaks a drop.

    Args:
        context: The prepared turn.
        requests: The reads about to be elided.

    Returns:
        Path to gist for every path the model answered.
    """
    files = "\n\n".join(f"=== FILE {r.path} ===\n{r.content}" for r in requests)
    context.emit(journal.CompactGistCall(files=len(requests)))
    try:
        resp = context.call_helper(compaction.GIST_DISTILL_SYSTEM_PROMPT, files, uuid.uuid4().hex)
    except (ProviderError, budget.BudgetExceededError) as exc:
        context.log(f"  gist distillation failed: {exc}; eliding without gists")
        context.emit(journal.CompactGistFailed(error=str(exc)[:200]))
        return {}
    return compaction_elision.parse_gist_lines(resp.text or "", paths=[r.path for r in requests])


def write_summary(
    context: strategy.Context,
    *,
    replay: retention.Replay,
    writer: writers.Writer,
    open_tasks: list[tuple[str, str]],
    focus: str,
) -> writers.Summary | None:
    """Try the selected writer and at most one helper fallback.

    Args:
        context: The prepared turn.
        replay: History to replace and to retain.
        writer: The first summary writer to try.
        open_tasks: The tasks eligible for check-off.
        focus: The operator's summary focus.

    Returns:
        Accepted prose and provenance, or None when both writers fail.

    Raises:
        ProviderError: Authentication or another fatal provider failure.
        ProviderAborted: The operator stopped the run.
        ProviderInterrupted: A steer is pending.
    """
    messages = context.conversation.complete().messages
    local = writer != "endpoint"
    sent_count = len(context.conversation) - len(replay.tail)
    fallback_reason = ""
    if writer == "worker" and context.settings.endpoint.keeps_session:
        writer = "helper"
        fallback_reason = "endpoint owns its live session"
    resp: provider_types.ProviderResponse | None = None
    while True:
        context.check_budget()
        context.check_stop()
        call_id = uuid.uuid4().hex
        context.emit(
            journal.CompactSummarizeCall(
                messages=sent_count if writer == "endpoint" else len(messages),
                writer=writer,
                fallback_reason=fallback_reason,
                model_call_id=call_id,
            )
        )
        request = writers.instruction(
            writer=writer,
            task=context.task.text,
            open_tasks=open_tasks,
            focus=focus,
            pins=context.pins,
            messages=messages,
        )
        try:
            if writer == "endpoint":
                resp = context.call_endpoint(request, call_id, sent_count)
            elif writer == "worker":
                resp = context.call_worker(request, call_id)
            else:
                request = writers.helper_input(
                    replay.replaced,
                    request,
                    context_tokens=context.settings.helper_context_tokens,
                    include_answers=local,
                    max_tokens=context.settings.summary_max_tokens,
                )
                resp = context.call_helper(
                    compaction.CONTEXT_SUMMARY_SYSTEM_PROMPT, request, call_id
                )
            failure = (
                writers.endpoint_rejection(resp)
                if writer == "endpoint"
                else writers.rejection(resp)
            )
        except (ProviderAborted, ProviderInterrupted):
            raise
        except ProviderError as exc:
            if (exc.fatal and not isinstance(exc, provider_types.MediaInputError)) or (
                exc.status_code in (401, 402, 403, 429)
            ):
                raise
            failure = str(exc)
        if not failure:
            break
        context.log(f"LOOP: {writer} summary failed: {failure}")
        context.emit(journal.CompactSummarizeFailed(error=failure, writer=writer))
        if writer == "helper":
            return None
        writer = "helper"
        fallback_reason = failure
    context.check_stop()
    assert resp is not None
    return writers.Summary(
        resp.text.strip(),
        writer,
        fallback_reason,
        resp.usage,
        resp.summary_block if writer == "endpoint" else None,
    )


def apply_checkoff(context: strategy.Context, summary_text: str, *, valid_ids: set[str]) -> None:
    """Apply the summarizer's checkoff block to the curator, best-effort.

    Completed tasks are marked passed, discovered ones queued under the run's
    root; a curator refusal skips that item.

    Args:
        summary_text: The summarizer's raw reply.
        valid_ids: The open task ids the summarizer was shown.
        context: The prepared turn.
    """
    if context.curator is None:
        return
    completed, new_tasks = compaction_window.parse_checkoff(summary_text)
    completed = [cid for cid in completed if cid in valid_ids]  # ignore hallucinated ids
    if not completed and not new_tasks:
        return
    passed = queued = 0
    # One try per write: a refusal or a write error skips that item, never the rest.
    for cid in completed:
        try:
            context.curator.update_status(
                models.UpdateStatusIntent(id=cid, new_status="passed", note="compaction check-off")
            )
        except (graph_curator.CuratorError, OSError, pydantic.ValidationError) as exc:
            context.log(f"LOOP: compaction check-off skipped {cid} ({exc})")
            continue
        passed += 1
    for title in new_tasks[:8]:  # cap: a runaway summary can't flood the DAG
        try:
            context.curator.add_subtask(
                models.AddSubtaskIntent(
                    parent_id=context.root_task_id,
                    draft=models.TaskNodeDraft(title=title, created_by="planner"),
                )
            )
        except (graph_curator.CuratorError, OSError, pydantic.ValidationError) as exc:
            context.log(f"LOOP: compaction check-off could not queue {title[:60]!r} ({exc})")
            continue
        queued += 1
    if passed or queued:
        # Record successful writes, not requested writes.
        context.log(f"LOOP: compaction check-off -- passed {passed}, queued {queued}")
        context.emit_graph_snapshot()
