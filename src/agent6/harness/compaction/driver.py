# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The context compaction driver.

Tier 1 elides old tool results (deduped, gisted, the old thinking dropped)
at `CompactionSettings.drop_at_chars`; tier 2 selects the worker or helper
writer before tier 1 and restarts the conversation from the task plus
the summary at `summarise_at_chars`, or on the operator's request. The pure
rules live in `compaction_rules`; this object does the calls, the DAG check-off
and the events. The loop calls `compact` once per turn before the provider
call.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

import pydantic

from agent6 import budget, kinds
from agent6.graph import curator as graph_curator
from agent6.graph import models
from agent6.harness import advice, loop_conversation
from agent6.harness.compaction import elision as compaction_elision
from agent6.harness.compaction import rules as compaction_rules
from agent6.harness.compaction import window as compaction_window
from agent6.harness.compaction import writers
from agent6.prompts import compaction
from agent6.providers import (
    Provider,
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    ToolDefinition,
    provider_types,
    request_headers,
)
from agent6.sessions import journal

if TYPE_CHECKING:
    from agent6.harness import loop_state


@dataclasses.dataclass(frozen=True, slots=True)
class Compactor:
    """The compaction driver for one run.

    Attributes:
        settings: The compaction thresholds and the summariser seat.
        provider: The worker's provider, the summariser when none is configured.
        curator: The task DAG the check-off writes to; None when no DAG is wired.
        mode: The run mode.
        dag_available: The DAG tools are wired, so the restart notice says so.
        decisions: The operator's rulings block for the restart notice, read fresh.
        verify_fact: The verify fact for the restart notice, "" for a run without a gate.
        compact_requested: The operator's manual compaction request, or None.
        compact_clear: Consumes the manual request.
        log: The run log line sink.
        emit: The run event sink.
        emit_graph_snapshot: Re-snapshots the graph after a check-off.
        should_abort: Whether an operator stop is pending.
        should_interrupt: Whether an operator steer is pending.
        check_budget: Check the shared budget before either writer.
        temperature: The worker's sampling temperature.
    """

    settings: compaction_window.CompactionSettings
    provider: Provider
    curator: graph_curator.GraphCurator | None
    mode: kinds.LoopMode
    dag_available: bool
    decisions: Callable[[], str]
    verify_fact: Callable[[loop_state.LoopState], str]
    compact_requested: Callable[[], str | None]
    compact_clear: Callable[[], None]
    log: Callable[[str], None]
    emit: Callable[..., None]
    emit_graph_snapshot: Callable[[], None]
    should_abort: Callable[[], bool] = lambda: False
    should_interrupt: Callable[[], bool] = lambda: False
    check_budget: Callable[[], None] = lambda: None
    temperature: float | None = 0.0

    def compact(
        self,
        conversation: loop_conversation.Conversation,
        state: loop_state.LoopState,
        *,
        prefix_chars: int = 0,
        tools: list[ToolDefinition] | None = None,
    ) -> bool:
        """Run tiered compaction over the conversation, in place.

        Tier 2 selects a writer before tier 1 can rewrite the request prefix.
        A rejected summary leaves the history unchanged. Manual compaction
        bypasses thresholds but still needs four messages.

        Args:
            conversation: The loop's history.
            state: The run's loop state; the tier-2 floor lives on it.
            prefix_chars: The system prompt and tool definitions' size, counted
                with the conversation against the tier-2 threshold.
            tools: The tools the worker would receive this turn.

        Returns:
            True when a tier-2 restart replaced the history, so the caller
            re-surfaces the focus banner the restart wiped.
        """
        forced = self.compact_requested()
        if forced is not None:
            self.compact_clear()
            focus_note = f" (focus: {forced[:80]})" if forced else ""
            self.log(f"LOOP: operator requested a manual compaction{focus_note}")
            self.emit(journal.CompactRequested(focus=forced))
        total = compaction_window.context_chars(conversation) + prefix_chars
        # Below four messages a restart loses more than it saves; a forced compaction skips
        # the growth floor.
        over = total > self.settings.summarise_at_chars and total >= state.tier2_floor_chars
        if (forced is not None or over) and len(conversation) > 3:
            return self.summarise_and_restart(
                conversation, state, focus=forced or "", prefix_chars=prefix_chars, tools=tools
            )
        if forced is not None:
            # The request was consumed above, so the refusal is said, not silent.
            self.log("LOOP: manual compaction skipped: too little history to summarise")
            self.emit(journal.CompactRefused(reason="too little history to summarise"))

        stats = compaction_rules.compact_old_tool_results(
            conversation,
            max_total_bytes=self.settings.drop_at_chars,
            keep_recent=2,
            protect_paths=compaction_rules.recently_edited_paths(conversation),
            gister=self.distill_gists if self.settings.elision_gists else None,
        )
        n_deduped = len(stats.deduped_calls)
        n_elided = len(stats.elided_calls)
        n_gisted = len(stats.gist_paths)
        n_demoted = len(stats.demoted_paths)
        if self.settings.keep_thinking_turns > 0 and (n_deduped or n_elided or n_demoted):
            # Strip only alongside a result rewrite, so thinking adds no cache break.
            n_messages, n_chars = compaction_window.strip_old_thinking(
                conversation, keep_messages=self.settings.keep_thinking_turns
            )
            if n_messages:
                self.log(
                    f"LOOP: compaction dropped thinking from {n_messages} old messages"
                    f" ({n_chars} chars)"
                )
                self.emit(journal.CompactThinkingDropped(messages=n_messages, chars=n_chars))
        if n_deduped:
            self.log(f"LOOP: compaction deduplicated {n_deduped} identical tool results")
            self.emit(journal.CompactDeduped(count=n_deduped, calls=list(stats.deduped_calls)))
        if n_elided:
            detail = f", {n_gisted} kept as distilled gists" if n_gisted else ""
            self.log(f"LOOP: compaction elided {n_elided} old tool_result blocks{detail}")
            self.emit(journal.CompactDropped(count=n_elided, calls=list(stats.elided_calls)))
        if n_demoted:
            self.log(f"LOOP: compaction demoted {n_demoted} gists to bare placeholders")
        if n_gisted or n_demoted:
            self.emit(
                journal.CompactGistDone(
                    gisted=n_gisted,
                    demoted=n_demoted,
                    paths=list(stats.gist_paths),
                    demoted_paths=list(stats.demoted_paths),
                )
            )
        return False

    def distill_gists(self, requests: tuple[compaction_elision.GistRequest, ...]) -> dict[str, str]:
        """Distill about-to-be-elided file reads into one-line gists.

        The summariser seat does the call. A provider error returns {} and every
        victim gets the bare placeholder, so gisting never breaks a drop.

        Args:
            requests: The reads about to be elided.

        Returns:
            Path to gist for every path the model answered.
        """
        provider = self.settings.helper or self.provider
        files = "\n\n".join(f"=== FILE {r.path} ===\n{r.content}" for r in requests)
        self.emit(journal.CompactGistCall(files=len(requests)))
        try:
            resp = provider.call(
                system=compaction.GIST_DISTILL_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": files}],
                tools=[],
                max_tokens=self.settings.summary_max_tokens,
                temperature=0.0,
            )
        except (ProviderError, budget.BudgetExceededError) as exc:
            self.log(f"  gist distillation failed: {exc}; eliding without gists")
            self.emit(journal.CompactGistFailed(error=str(exc)[:200]))
            return {}
        return compaction_elision.parse_gist_lines(
            resp.text or "", paths=[r.path for r in requests]
        )

    def summarise_and_restart(
        self,
        conversation: loop_conversation.Conversation,
        state: loop_state.LoopState,
        *,
        focus: str = "",
        prefix_chars: int = 0,
        tools: list[ToolDefinition] | None = None,
    ) -> bool:
        """Replace the history with the task plus a model-written summary, in place.

        The loop calls this at the top of a turn, where every `tool_use`
        has its `tool_result`, so the restart drops the middle without orphaning
        a pairing.

        Args:
            conversation: The loop's history.
            state: The run's loop state.
            focus: The operator's focus text for the summary, "" for none.
            prefix_chars: The system prompt and tool definitions' size, counted
                into the next tier-2 floor.
            tools: The tools the worker would receive this turn.

        Returns:
            True when the history was replaced; False on every fail-safe path,
            where the current context is kept.
        """
        messages = conversation.messages
        tail_start = compaction_window.recent_tail_start(messages, self.settings.keep_recent_chars)
        if tail_start <= 1:
            tail_start = len(messages)
        open_tasks = advice.open_subtasks(self.curator.nodes()) if self.curator is not None else []
        written = self.write_summary(
            conversation,
            state,
            tail_start=tail_start,
            open_tasks=open_tasks,
            focus=focus,
            tools=tools,
        )
        if written is None:
            return False
        raw = written.raw
        summary = compaction_window.strip_checkoff(raw)
        # Only after the narrative passed: bookkeeping alone never mutates the DAG.
        if open_tasks:
            self.apply_checkoff(
                raw, valid_ids={tid for tid, _ in open_tasks}, root_id=state.root_task_id
            )
        conversation.restart(
            compaction.context_restart_notice(
                self.mode,
                pins=state.pins,
                decisions=self.decisions(),
                dag_available=self.dag_available,
                verification=self.verify_fact(state),
            )
            + summary,
            keep=messages[tail_start:],
        )
        # Measured as the trigger is, prefix included, or tier 2 re-fires next turn.
        state.tier2_floor_chars = int(
            (compaction_window.context_chars(conversation) + prefix_chars) * 1.25
        )
        self.emit(
            journal.CompactSummariseDone(
                writer=written.writer,
                fallback_reason=written.fallback_reason,
                summary_chars=len(summary),
                summary=summary,
                kept_messages=len(messages) - tail_start,
            )
        )
        return True

    def check_cancelled(self) -> None:
        """Stop a summary before another call or a history mutation.

        Raises:
            ProviderAborted: The operator stopped the run.
            ProviderInterrupted: A steer is pending.
        """
        if self.should_abort():
            raise ProviderAborted("operator stopped compaction")
        if self.should_interrupt():
            raise ProviderInterrupted("operator steered during compaction")

    def write_summary(
        self,
        conversation: loop_conversation.Conversation,
        state: loop_state.LoopState,
        *,
        tail_start: int,
        open_tasks: list[tuple[str, str]],
        focus: str,
        tools: list[ToolDefinition] | None,
    ) -> writers.Summary | None:
        """Try the selected writer and at most one helper fallback.

        Args:
            conversation: The prepared worker conversation, never mutated here.
            state: The run's task, pins and call identity.
            tail_start: The first message retained verbatim.
            open_tasks: The tasks eligible for check-off.
            focus: The operator's summary focus.
            tools: The worker's tools.

        Returns:
            Accepted prose and provenance, or None when both writers fail.

        Raises:
            ProviderError: Authentication or another fatal provider failure.
            ProviderAborted: The operator stopped the run.
            ProviderInterrupted: A steer is pending.
        """
        messages = conversation.messages
        writer = self.settings.summariser
        fallback_reason = ""
        if writer == "worker" and self.settings.worker_backend == "claude_code":
            writer = "helper"
            fallback_reason = "claude_code owns its live session"
        resp: provider_types.ProviderResponse | None = None
        while True:
            self.check_budget()
            self.check_cancelled()
            context = request_headers.RequestContext.initiated_by(
                conversation.initiator,
                call_context=kinds.CallContext(
                    execution_id=state.execution_id,
                    panel_id=state.call_context.panel_id
                    if writer == "worker" and state.call_context
                    else None,
                    reviewer_index=state.call_context.reviewer_index
                    if writer == "worker" and state.call_context
                    else None,
                ),
            )
            self.emit(
                journal.CompactSummariseCall(
                    messages=len(conversation),
                    writer=writer,
                    fallback_reason=fallback_reason,
                    model_call_id=context.call_context.model_call_id
                    if context.call_context
                    else "",
                )
            )
            request = writers.instruction(
                task=state.original_task,
                open_tasks=open_tasks,
                focus=focus,
                pins=state.pins,
                messages=messages if writer == "helper" else (),
            )
            try:
                if writer == "worker":
                    resp = self.provider.call(
                        system=state.system,
                        messages=[*conversation.to_wire(), {"role": "user", "content": request}],
                        tools=tools or [],
                        max_tokens=self.settings.summary_max_tokens,
                        temperature=self.temperature,
                        should_abort=self.should_abort,
                        should_interrupt=self.should_interrupt,
                        context=context,
                    )
                else:
                    request = writers.helper_input(
                        messages[1:tail_start],
                        request,
                        context_tokens=self.settings.helper_context_tokens,
                        max_tokens=self.settings.summary_max_tokens,
                    )
                    resp = (self.settings.helper or self.provider).call(
                        system=compaction.CONTEXT_SUMMARY_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": request}],
                        tools=[],
                        max_tokens=self.settings.summary_max_tokens,
                        temperature=0.0,
                        should_abort=self.should_abort,
                        should_interrupt=self.should_interrupt,
                        context=context,
                    )
                failure = writers.rejection(resp)
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
            self.log(f"LOOP: {writer} summary failed: {failure}")
            self.emit(journal.CompactSummariseFailed(error=failure, writer=writer))
            if writer == "helper":
                return None
            writer = "helper"
            fallback_reason = failure
        self.check_cancelled()
        assert resp is not None
        return writers.Summary(resp.text.strip(), writer, fallback_reason)

    def apply_checkoff(
        self, summary_text: str, *, valid_ids: set[str], root_id: str | None
    ) -> None:
        """Apply the summariser's checkoff block to the curator, best-effort.

        Completed tasks are marked passed, discovered ones queued under the run's
        root; a curator refusal skips that item.

        Args:
            summary_text: The summariser's raw reply.
            valid_ids: The open task ids the summariser was shown.
            root_id: The parent for discovered tasks.
        """
        if self.curator is None:
            return
        completed, new_tasks = compaction_window.parse_checkoff(summary_text)
        completed = [cid for cid in completed if cid in valid_ids]  # ignore hallucinated ids
        if not completed and not new_tasks:
            return
        passed = queued = 0
        # One try per write: a refusal or a write error skips that item, never the rest.
        for cid in completed:
            try:
                self.curator.update_status(
                    models.UpdateStatusIntent(
                        id=cid, new_status="passed", note="compaction check-off"
                    )
                )
            except (graph_curator.CuratorError, OSError, pydantic.ValidationError) as exc:
                self.log(f"LOOP: compaction check-off skipped {cid} ({exc})")
                continue
            passed += 1
        for title in new_tasks[:8]:  # cap: a runaway summary can't flood the DAG
            try:
                self.curator.add_subtask(
                    models.AddSubtaskIntent(
                        parent_id=root_id,
                        draft=models.TaskNodeDraft(title=title, created_by="planner"),
                    )
                )
            except (graph_curator.CuratorError, OSError, pydantic.ValidationError) as exc:
                self.log(f"LOOP: compaction check-off could not queue {title[:60]!r} ({exc})")
                continue
            queued += 1
        if passed or queued:
            # Record successful writes, not requested writes.
            self.log(f"LOOP: compaction check-off -- passed {passed}, queued {queued}")
            self.emit_graph_snapshot()
