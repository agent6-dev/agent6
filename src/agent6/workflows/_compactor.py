# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The context compaction driver: tier 1 elides old tool results (deduped,
gisted, the old thinking dropped) at `CompactionSettings.drop_at_chars`;
tier 2 summarises the elided history with the summariser model and
restarts the conversation from the task plus the summary at
`summarise_at_chars`, or on the operator's request. The pure rules live in
`_compaction`; this object does the calls, the DAG check-off and the
events, and the loop calls `compact` once per turn before the provider
call."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from agent6.budget import BudgetExceeded
from agent6.graph.curator import CuratorError, GraphCurator
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft, UpdateStatusIntent
from agent6.graph.order import OPEN_STATUSES
from agent6.prompts.revision import (
    CONTEXT_SUMMARY_SYSTEM_PROMPT,
    GIST_DISTILL_SYSTEM_PROMPT,
    PINS_NO_RESTATE_CLAUSE,
    context_restart_notice,
    progress_summary_from_notice,
)
from agent6.providers import Provider, ProviderError
from agent6.workflows._compaction import (
    CompactionSettings,
    GistRequest,
    compact_old_tool_results,
    context_chars,
    parse_checkoff,
    parse_gist_lines,
    recent_tail_start,
    recently_edited_paths,
    strip_checkoff,
    strip_old_thinking,
)
from agent6.workflows._conversation import Conversation, Notice, format_transcript_tail

if TYPE_CHECKING:
    from agent6.workflows._loop_state import LoopState


@dataclass(frozen=True, slots=True)
class Compactor:
    """The compaction driver for one run: its settings, the worker's provider
    (the summariser when none is configured), the task DAG it checks off,
    and the run's bridge, log and event callables."""

    settings: CompactionSettings
    provider: Provider
    curator: GraphCurator | None
    mode: Literal["run", "plan", "ask", "agent"]
    # DAG tools are wired: the restart notice tells the model so.
    dag_available: bool
    # The operator's rulings block for the restart notice, read fresh.
    decisions: Callable[[], str]
    # The operator's manual compaction request and its consumption.
    compact_requested: Callable[[], str | None]
    compact_clear: Callable[[], None]
    log: Callable[[str], None]
    emit: Callable[..., None]
    emit_graph_snapshot: Callable[[], None]

    def compact(
        self, conversation: Conversation, state: LoopState, *, prefix_chars: int = 0
    ) -> bool:
        """Tiered compaction. Returns True iff a tier-2 summarise-and-restart
        actually replaced the history (so the caller can re-surface the
        current-task banner the restart wiped); False otherwise.

        Tier 1 (cheap): drop old tool_result blocks once cumulative content
        exceeds `CompactionSettings.drop_at_chars`.

        Tier 2 (expensive): once the WHOLE post-elision context (text +
        tool_use inputs + surviving tool_results, via `context_chars`)
        crosses `CompactionSettings.summarise_at_chars`, summarise the elided history
        into a compact progress block and restart the conversation from
        (original task + summary). Fail-safe: if
        summarisation errors or returns nothing, the conversation is left
        untouched (tier-1 elision already ran) and the run continues.

        An operator compact request (`compact_requested`, the TUI's
        "Compact now") forces tier 2 regardless of the size thresholds; the
        marker is consumed here so one request means one compaction.
        """
        forced = self.compact_requested()
        if forced is not None:
            self.compact_clear()
            focus_note = f" (focus: {forced[:80]})" if forced else ""
            self.log(f"LOOP: operator requested a manual compaction{focus_note}")
            self.emit("loop.compact.requested", focus=forced)
        stats = compact_old_tool_results(
            conversation,
            max_total_bytes=self.settings.drop_at_chars,
            keep_recent=2,
            protect_paths=recently_edited_paths(conversation),
            gister=self.distill_gists if self.settings.elision_gists else None,
        )
        n_deduped = len(stats.deduped_calls)
        n_elided = len(stats.elided_calls)
        n_gisted = len(stats.gist_paths)
        n_demoted = len(stats.demoted_paths)
        if self.settings.keep_thinking_turns > 0 and (
            n_deduped or n_elided or context_chars(conversation) > self.settings.drop_at_chars
        ):
            # Same cache-bundling rule as dedup: only at tier-1 pressure
            # moments, never as a rolling per-iteration rewrite.
            n_turns, n_chars = strip_old_thinking(
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
        # Measure the WHOLE post-elision request, not just tool_results: tier 1
        # already bounded those, so re-measuring them could never cross the
        # larger tier-2 threshold -- and the window bounds the request, so the
        # system prompt and the tool definitions count too
        # (`request_prefix_chars`).
        total = context_chars(conversation) + prefix_chars
        # Tier 2 needs at least an original-task turn plus enough history
        # to be worth summarising; below that a restart would lose more than
        # it saves. The growth floor (see LoopState.tier2_floor_chars) keeps
        # a restart that lands near the threshold from summarising every
        # other iteration; a forced (operator) compaction bypasses it.
        over = total > self.settings.summarise_at_chars and total >= state.tier2_floor_chars
        if (forced is not None or over) and len(conversation) > 3:
            return self.summarise_and_restart(
                conversation, state, focus=forced or "", prefix_chars=prefix_chars
            )
        if forced is not None:
            # The request was consumed above (one request, one compaction), so a
            # silent return would drop it: the front-end has already told the
            # operator it "applies before the next model call", and the focus
            # text is gone. Say the floor refused it.
            self.log("LOOP: manual compaction skipped: too little history to summarise")
            self.emit("loop.compact.refused", reason="too little history to summarise")
        return False

    def distill_gists(self, requests: tuple[GistRequest, ...]) -> dict[str, str]:
        """Distill about-to-be-elided file reads into one-line gists with the
        summariser model (same seat as tier-2). Fail-safe: any provider error
        returns {} and every victim gets the bare placeholder, so gisting can
        slow a drop event but never break one."""
        provider = self.settings.summariser or self.provider
        files = "\n\n".join(f"=== FILE {r.path} ===\n{r.content}" for r in requests)
        self.emit("loop.compact.gist.call", files=len(requests))
        try:
            resp = provider.call(
                system=GIST_DISTILL_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": files}],
                tools=[],
                max_tokens=self.settings.summary_max_tokens,
                temperature=0.0,
            )
        except (ProviderError, BudgetExceeded) as exc:
            self.log(f"  gist distillation failed: {exc}; eliding without gists")
            self.emit("loop.compact.gist.failed", error=str(exc)[:200])
            return {}
        return parse_gist_lines(resp.text or "", paths=[r.path for r in requests])

    def summarise_and_restart(
        self,
        conversation: Conversation,
        state: LoopState,
        *,
        focus: str = "",
        prefix_chars: int = 0,
    ) -> bool:
        """Replace the history with (original task + a model-written progress
        summary), in place. The loop only calls this at the top of an
        iteration, where the history is balanced (every `tool_use` already
        has its `tool_result`), so the restart can drop the middle without
        orphaning a tool-call pairing. Returns True iff the history was
        actually replaced; False on every fail-safe path (the tier-1-elided
        context is kept and the run continues).
        """
        provider = self.settings.summariser or self.provider
        turns = conversation.turns
        # The verbatim tail survives the restart, so the summary covers only
        # what is actually dropped (pi's keepRecentTokens shape).
        tail_start = recent_tail_start(turns, self.settings.keep_recent_chars)
        if tail_start <= 1:
            # A cap that swallows the whole history would make the restart
            # grow the context instead of shrinking it; keep nothing.
            tail_start = len(turns)
        transcript = format_transcript_tail(
            turns[1:tail_start], max_messages=len(conversation), max_chars=60_000
        )
        # The DAG is agent6's compaction memory: at each restart we ask the
        # summariser to check off finished tasks and surface newly-found ones, so
        # task state stays accurate across compaction without depending on the
        # worker calling update_task (which weak models rarely do).
        open_tasks = self.open_tasks_for_checkoff()
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
            pins_req = PINS_NO_RESTATE_CLAUSE + pin_lines
        # The previous restart's summary rides at the HEAD of the post-restart
        # history, which the tail-clipped transcript above drops first, so it
        # is carried out-of-band, like pins.
        prior_req = ""
        for turn in conversation.turns:
            for item in getattr(turn, "items", ()):
                if isinstance(item, Notice) and (prior := progress_summary_from_notice(item.text)):
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
                system=CONTEXT_SUMMARY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                tools=[],
                max_tokens=self.settings.summary_max_tokens,
                temperature=0.0,
            )
        except (ProviderError, BudgetExceeded) as exc:
            # Fail-safe: keep the current (tier-1-elided) context. A real
            # budget exhaustion is re-detected by the next provider call.
            self.log(f"  tier-2 summarise failed: {exc}; keeping current context")
            self.emit("loop.compact.summarise.failed", error=str(exc)[:200])
            return False
        raw = (resp.text or "").strip()
        summary = strip_checkoff(raw) if open_tasks else raw
        if not summary:
            self.emit("loop.compact.summarise.failed", error="empty summary")
            return False
        # Apply the check-off only after the stripped narrative passed the
        # fail-safe, so bookkeeping alone can neither mutate the DAG nor erase history.
        if open_tasks:
            self.apply_checkoff(raw, valid_ids={tid for tid, _ in open_tasks})
        conversation.restart(
            context_restart_notice(
                self.mode,
                pins=state.pins,
                decisions=self.decisions(),
                dag_available=self.dag_available,
            )
            + summary,
            keep=turns[tail_start:],
        )
        # The floor is measured the way the trigger is (the whole request, prefix
        # included): computed on the conversation alone it would sit below the
        # total from the moment the restart finishes, so tier 2 would re-fire on
        # the next iteration and paraphrase away the tail it just kept.
        state.tier2_floor_chars = int((context_chars(conversation) + prefix_chars) * 1.25)
        self.emit(
            "loop.compact.summarise.done",
            summary_chars=len(summary),
            summary=summary,
            kept_turns=len(turns) - tail_start,
        )
        return True

    def open_tasks_for_checkoff(self) -> list[tuple[str, str]]:
        """(id, title) of every pending/in_progress task in the DAG, for the
        tier-2 compaction check-off. Best-effort: no curator or a curator error
        yields an empty list, so compaction degrades to the plain summary."""
        if self.curator is None:
            return []
        out: list[tuple[str, str]] = []
        for nid, node in self.curator.nodes().items():
            # Subtasks only: never offer the auto-root (parent_id is None) for
            # check-off, mirroring the finish-gate and surface rules. The root is
            # the whole-run container, so a mid-run summary must not mark it
            # passed and end the run early.
            if node.parent_id is None or node.standing:
                continue
            if node.status in OPEN_STATUSES:
                out.append((nid, node.title[:120]))
        return out

    def apply_checkoff(self, summary_text: str, *, valid_ids: set[str]) -> None:
        """Parse the summariser's ```checkoff block and apply it to the curator:
        mark completed tasks passed, queue newly-discovered ones as children of
        the first root. Best-effort: a curator hiccup must never break the run."""
        if self.curator is None:
            return
        completed, new_tasks = parse_checkoff(summary_text)
        completed = [cid for cid in completed if cid in valid_ids]  # ignore hallucinated ids
        if not completed and not new_tasks:
            return
        passed = queued = 0
        # One try per write: a refusal (a container with unresolved children, a
        # retired task) or a write error skips that item, never the rest.
        for cid in completed:
            try:
                self.curator.update_status(
                    UpdateStatusIntent(id=cid, new_status="passed", note="compaction check-off")
                )
            except (CuratorError, OSError, ValidationError) as exc:
                self.log(f"LOOP: compaction check-off skipped {cid} ({exc})")
                continue
            passed += 1
        for title in new_tasks[:8]:  # cap: a runaway summary can't flood the DAG
            try:
                self.curator.add_subtask(
                    AddSubtaskIntent(
                        parent_id=self.first_root_id(),
                        draft=TaskNodeDraft(title=title, created_by="planner"),
                    )
                )
            except (CuratorError, OSError, ValidationError) as exc:
                self.log(f"LOOP: compaction check-off could not queue {title[:60]!r} ({exc})")
                continue
            queued += 1
        if passed or queued:
            # What LANDED, not what the summariser asked for: the cap above and
            # a refused status (a container with unresolved children) both make the
            # request bigger than the change.
            self.log(f"LOOP: compaction check-off -- passed {passed}, queued {queued}")
            self.emit_graph_snapshot()

    def first_root_id(self) -> str | None:
        """The first root task id (parent_id is None), or None. Best-effort."""
        if self.curator is None:
            return None
        for nid, node in self.curator.nodes().items():
            if node.parent_id is None:
                return nid
        return None
