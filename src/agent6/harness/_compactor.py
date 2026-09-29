# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The context compaction driver.

Tier 1 elides old tool results (deduped, gisted, the old thinking dropped)
at `CompactionSettings.drop_at_chars`; tier 2 summarises the elided history
with the summariser model and restarts the conversation from the task plus
the summary at `summarise_at_chars`, or on the operator's request. The pure
rules live in `_compaction`; this object does the calls, the DAG check-off
and the events. The loop calls `compact` once per turn before the provider
call.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import pydantic

from agent6 import budget
from agent6.graph import curator as graph_curator
from agent6.graph import models
from agent6.harness import _advice, _compaction, _conversation
from agent6.prompts import revision
from agent6.providers import Provider, ProviderError

if TYPE_CHECKING:
    from agent6.harness import _loop_state


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
        compact_requested: The operator's manual compaction request, or None.
        compact_clear: Consumes the manual request.
        log: The run log line sink.
        emit: The run event sink.
        emit_graph_snapshot: Re-snapshots the graph after a check-off.
    """

    settings: _compaction.CompactionSettings
    provider: Provider
    curator: graph_curator.GraphCurator | None
    mode: Literal["run", "plan", "ask", "agent"]
    dag_available: bool
    decisions: Callable[[], str]
    compact_requested: Callable[[], str | None]
    compact_clear: Callable[[], None]
    log: Callable[[str], None]
    emit: Callable[..., None]
    emit_graph_snapshot: Callable[[], None]

    def compact(
        self,
        conversation: _conversation.Conversation,
        state: _loop_state.LoopState,
        *,
        prefix_chars: int = 0,
    ) -> bool:
        """Run tiered compaction over the conversation, in place.

        Tier 1 drops old tool_result blocks once their content exceeds
        `drop_at_chars`. Tier 2 fires once the whole post-elision context (text,
        tool_use inputs, surviving results) crosses `summarise_at_chars`: the
        elided history is summarised and the conversation restarts from the task
        plus the summary. A summariser error or an empty summary leaves the
        tier-1-elided conversation as it is. An operator compact request forces
        tier 2 past the thresholds and is consumed here, so one request is one
        compaction.

        Args:
            conversation: The loop's history.
            state: The run's loop state; the tier-2 floor lives on it.
            prefix_chars: The system prompt and tool definitions' size, counted
                with the conversation against the tier-2 threshold.

        Returns:
            True when a tier-2 restart replaced the history, so the caller
            re-surfaces the focus banner the restart wiped.
        """
        forced = self.compact_requested()
        if forced is not None:
            self.compact_clear()
            focus_note = f" (focus: {forced[:80]})" if forced else ""
            self.log(f"LOOP: operator requested a manual compaction{focus_note}")
            self.emit("loop.compact.requested", focus=forced)
        stats = _compaction.compact_old_tool_results(
            conversation,
            max_total_bytes=self.settings.drop_at_chars,
            keep_recent=2,
            protect_paths=_compaction.recently_edited_paths(conversation),
            gister=self.distill_gists if self.settings.elision_gists else None,
        )
        n_deduped = len(stats.deduped_calls)
        n_elided = len(stats.elided_calls)
        n_gisted = len(stats.gist_paths)
        n_demoted = len(stats.demoted_paths)
        if self.settings.keep_thinking_turns > 0 and (
            n_deduped
            or n_elided
            or _compaction.context_chars(conversation) > self.settings.drop_at_chars
        ):
            # As with dedup: only at tier-1 pressure, never as a rolling per-turn rewrite.
            n_turns, n_chars = _compaction.strip_old_thinking(
                conversation, keep_turns=self.settings.keep_thinking_turns
            )
            if n_turns:
                self.log(
                    f"LOOP: compaction dropped thinking from {n_turns} old turns ({n_chars} chars)"
                )
                self.emit("loop.compact.thinking_dropped", turns=n_turns, chars=n_chars)
        if n_deduped:
            self.log(f"LOOP: compaction deduplicated {n_deduped} identical tool results")
            self.emit("loop.compact.deduped", n=n_deduped, calls=list(stats.deduped_calls))
        if n_elided:
            detail = f", {n_gisted} kept as distilled gists" if n_gisted else ""
            self.log(f"LOOP: compaction elided {n_elided} old tool_result blocks{detail}")
            self.emit("loop.compact.dropped", n=n_elided, calls=list(stats.elided_calls))
        if n_demoted:
            self.log(f"LOOP: compaction demoted {n_demoted} gists to bare placeholders")
        if n_gisted or n_demoted:
            self.emit(
                "loop.compact.gists",
                gisted=n_gisted,
                demoted=n_demoted,
                paths=list(stats.gist_paths),
                demoted_paths=list(stats.demoted_paths),
            )
        # The whole request: tier 1 bounded the results alone, and the window bounds the
        # prefix too.
        total = _compaction.context_chars(conversation) + prefix_chars
        # Below four turns a restart loses more than it saves; a forced compaction skips
        # the growth floor.
        over = total > self.settings.summarise_at_chars and total >= state.tier2_floor_chars
        if (forced is not None or over) and len(conversation) > 3:
            return self.summarise_and_restart(
                conversation, state, focus=forced or "", prefix_chars=prefix_chars
            )
        if forced is not None:
            # The request was consumed above, so the refusal is said, not silent.
            self.log("LOOP: manual compaction skipped: too little history to summarise")
            self.emit("loop.compact.refused", reason="too little history to summarise")
        return False

    def distill_gists(self, requests: tuple[_compaction.GistRequest, ...]) -> dict[str, str]:
        """Distill about-to-be-elided file reads into one-line gists.

        The summariser seat does the call. A provider error returns {} and every
        victim gets the bare placeholder, so gisting never breaks a drop.

        Args:
            requests: The reads about to be elided.

        Returns:
            Path to gist for every path the model answered.
        """
        provider = self.settings.summariser or self.provider
        files = "\n\n".join(f"=== FILE {r.path} ===\n{r.content}" for r in requests)
        self.emit("loop.compact.gist.call", files=len(requests))
        try:
            resp = provider.call(
                system=revision.GIST_DISTILL_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": files}],
                tools=[],
                max_tokens=self.settings.summary_max_tokens,
                temperature=0.0,
            )
        except (ProviderError, budget.BudgetExceededError) as exc:
            self.log(f"  gist distillation failed: {exc}; eliding without gists")
            self.emit("loop.compact.gist.failed", error=str(exc)[:200])
            return {}
        return _compaction.parse_gist_lines(resp.text or "", paths=[r.path for r in requests])

    def summarise_and_restart(
        self,
        conversation: _conversation.Conversation,
        state: _loop_state.LoopState,
        *,
        focus: str = "",
        prefix_chars: int = 0,
    ) -> bool:
        """Replace the history with the task plus a model-written summary, in place.

        The loop calls this at the top of an iteration, where every `tool_use`
        has its `tool_result`, so the restart drops the middle without orphaning
        a pairing.

        Args:
            conversation: The loop's history.
            state: The run's loop state.
            focus: The operator's focus text for the summary, "" for none.
            prefix_chars: The system prompt and tool definitions' size, counted
                into the next tier-2 floor.

        Returns:
            True when the history was replaced; False on every fail-safe path,
            where the tier-1-elided context is kept.
        """
        provider = self.settings.summariser or self.provider
        turns = conversation.turns
        # The verbatim tail survives the restart; the summary covers only what is dropped.
        tail_start = _compaction.recent_tail_start(turns, self.settings.keep_recent_chars)
        if tail_start <= 1:
            # A tail holding the whole history would grow the context; keep nothing.
            tail_start = len(turns)
        transcript = _conversation.format_transcript_tail(
            turns[1:tail_start], max_messages=len(conversation), max_chars=60_000
        )
        # The summariser checks off finished tasks and surfaces new ones, so task state
        # stays accurate without the worker calling update_task.
        open_tasks = _advice.open_subtasks(self.curator.nodes()) if self.curator is not None else []
        if open_tasks:
            task_lines = "\n".join(f"- {tid}: {title}" for tid, title in open_tasks)
            checkoff_req = (
                "\n\nThe worker is tracking these OPEN tasks:\n"
                f"{task_lines}\n\n"
                "After the summary, append a fenced block exactly like:\n"
                "```checkoff\n"
                '{"completed_ids": ["<ids the transcript clearly shows finished>"], '
                '"new_tasks": ["<short title of work discovered but not yet tracked>"]}\n'
                "```\n"
                "Mark a task completed ONLY if the transcript clearly shows it done;"
                " leave the rest open. Use [] when none apply."
            )
        else:
            checkoff_req = ""
        focus_req = (
            f"\n\nOperator focus for this summary, weigh these aspects heavily:\n{focus}"
            if focus
            else ""
        )
        pins_req = ""
        if state.pins:
            pin_lines = "\n".join(f"{i}. {p}" for i, p in enumerate(state.pins, start=1))
            pins_req = revision.PINS_NO_RESTATE_CLAUSE + pin_lines
        # The previous summary heads the history, which the clipped transcript drops
        # first, so it is carried out-of-band like pins.
        prior_req = ""
        for turn in conversation.turns:
            for item in getattr(turn, "items", ()):
                if isinstance(item, _conversation.Notice) and (
                    prior := revision.progress_summary_from_notice(item.text)
                ):
                    prior_req = (
                        "\n\nThis conversation was ALREADY compacted; the summary from"
                        " that restart follows, and the transcript below covers only what"
                        " happened SINCE. Carry anything still relevant into the new"
                        f" summary:\n{prior}"
                    )
        user_msg = (
            "Summarise the following agent transcript for a context restart."
            f"\n\nTASK (the goal, verbatim):\n{state.original_task}"
            f"{checkoff_req}{focus_req}{pins_req}{prior_req}"
            f"\n\nTRANSCRIPT (oldest first):\n{transcript}"
        )
        self.log(f"LOOP: tier-2 compaction summarise-and-restart ({len(conversation)} msgs)")
        self.emit("loop.compact.summarise.call", messages=len(conversation))
        try:
            resp = provider.call(
                system=revision.CONTEXT_SUMMARY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                tools=[],
                max_tokens=self.settings.summary_max_tokens,
                temperature=0.0,
            )
        except (ProviderError, budget.BudgetExceededError) as exc:
            # A real budget exhaustion is re-detected by the next provider call.
            self.log(f"  tier-2 summarise failed: {exc}; keeping current context")
            self.emit("loop.compact.summarise.failed", error=str(exc)[:200])
            return False
        raw = (resp.text or "").strip()
        summary = _compaction.strip_checkoff(raw) if open_tasks else raw
        if not summary:
            self.emit("loop.compact.summarise.failed", error="empty summary")
            return False
        # Only after the narrative passed: bookkeeping alone never mutates the DAG.
        if open_tasks:
            self.apply_checkoff(
                raw, valid_ids={tid for tid, _ in open_tasks}, root_id=state.root_task_id
            )
        conversation.restart(
            revision.context_restart_notice(
                self.mode,
                pins=state.pins,
                decisions=self.decisions(),
                dag_available=self.dag_available,
            )
            + summary,
            keep=turns[tail_start:],
        )
        # Measured as the trigger is, prefix included, or tier 2 re-fires next iteration.
        state.tier2_floor_chars = int(
            (_compaction.context_chars(conversation) + prefix_chars) * 1.25
        )
        self.emit(
            "loop.compact.summarise.done",
            summary_chars=len(summary),
            summary=summary,
            kept_turns=len(turns) - tail_start,
        )
        return True

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
        completed, new_tasks = _compaction.parse_checkoff(summary_text)
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
            # What landed, not what the summariser asked for.
            self.log(f"LOOP: compaction check-off -- passed {passed}, queued {queued}")
            self.emit_graph_snapshot()
