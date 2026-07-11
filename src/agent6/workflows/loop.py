# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Minimal single-loop agent workflow.

See ``bench/improvement_plan.md`` for the full rationale. tl;dr: one
large system prompt, one LLM driving via tool calls, deterministic
harness (jail + budget + verify timeout + DAG curator for
persistence/resume). No planner/critic/triage subagent cascade.

The workflow also auto-commits on every verify-pass: the agent shouldn't
need to remember to ``git commit`` after a green verify, so the workflow
does it for them and score.sh sees the actual improvements that were made.

Not implemented yet:
- Alignment guard (no rigid plan to drift from)
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agent6.budget import BudgetExceeded, BudgetTracker
from agent6.config import Config
from agent6.git_ops import GitError, commit_all, commit_diff, diff_since
from agent6.git_ops import status as git_status
from agent6.graph.client import CuratorClientError, GraphClient
from agent6.graph.models import (
    AddSubtaskIntent,
    SetCursorIntent,
    TaskNodeDraft,
    UpdateStatusIntent,
)
from agent6.memory import MemoryEntry, MemoryStoreError
from agent6.memory import list_entries as memory_list_entries
from agent6.portable import atomic_write
from agent6.providers import (
    Provider,
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    ToolDefinition,
)
from agent6.tools.dispatch import OperatorCommandUnexecutable, ToolDispatcher, ToolError
from agent6.tools.schema import (
    FinishPlanningInput,
    FinishRunInput,
)
from agent6.types import RepoSummary
from agent6.workflows._cache import (
    roll_cache_breakpoints as _roll_cache_breakpoints,
)
from agent6.workflows._compaction import (
    DROP_BLOCKS_AT_CHARS as _DROP_BLOCKS_AT_CHARS,
)
from agent6.workflows._compaction import (
    SUMMARISE_AT_CHARS as _SUMMARISE_AT_CHARS,
)
from agent6.workflows._compaction import (
    GistRequest as _GistRequest,
)
from agent6.workflows._compaction import (
    cap_tool_result as _cap_tool_result,
)
from agent6.workflows._compaction import (
    compact_old_tool_results as _compact_old_tool_results,
)
from agent6.workflows._compaction import (
    context_chars as _context_chars,
)
from agent6.workflows._compaction import (
    parse_checkoff as _parse_checkoff,
)
from agent6.workflows._compaction import (
    parse_gist_lines as _parse_gist_lines,
)
from agent6.workflows._compaction import (
    recently_edited_paths as _recently_edited_paths,
)
from agent6.workflows._compaction import (
    strip_checkoff as _strip_checkoff,
)
from agent6.workflows._context import load_repo_summary
from agent6.workflows._critic import (
    CritiqueResult as _CritiqueResult,
)
from agent6.workflows._critic import (
    format_messages_tail_for_critic as _format_messages_tail_for_critic,
)
from agent6.workflows._critic import (
    parse_critic_verdict as _parse_critic_verdict,
)
from agent6.workflows._dag_focus import (
    DAG_MUTATING_TOOLS as _DAG_MUTATING_TOOLS,
)
from agent6.workflows._dag_focus import (
    STUCK_NUDGE_MAX as _STUCK_NUDGE_MAX,
)
from agent6.workflows._dag_focus import (
    STUCK_ON_TASK_AFTER as _STUCK_ON_TASK_AFTER,
)
from agent6.workflows._dag_focus import (
    current_task_banner as _current_task_banner,
)
from agent6.workflows._dag_focus import (
    current_task_id as _current_task_id,
)
from agent6.workflows._dag_focus import (
    initial_dag_hint as _initial_dag_hint,
)
from agent6.workflows._dag_focus import (
    stuck_on_task_nudge as _stuck_on_task_nudge,
)
from agent6.workflows._metric import (
    METRIC_EARLY_FINISH_PATIENCE as _METRIC_EARLY_FINISH_PATIENCE,
)
from agent6.workflows._metric import (
    METRIC_FINISH_NUDGE as _METRIC_FINISH_NUDGE,
)
from agent6.workflows._metric import (
    METRIC_PLATEAU_PATIENCE as _METRIC_PLATEAU_PATIENCE,
)
from agent6.workflows._metric import (
    METRIC_PLATEAU_STOP_BELOW_BUDGET as _METRIC_PLATEAU_STOP_BELOW_BUDGET,
)
from agent6.workflows._metric import (
    MetricSample as _MetricSample,
)
from agent6.workflows._metric import (
    best_metric_sample as _best_metric_sample,
)
from agent6.workflows._metric import (
    coerce_metric_score as _coerce_metric_score,
)
from agent6.workflows._metric import (
    extract_metric_targets as _extract_metric_targets,
)
from agent6.workflows._metric import (
    format_metric_feedback as _format_metric_feedback,
)
from agent6.workflows._metric import (
    metric_at_fraction_ceiling as _metric_at_fraction_ceiling,
)
from agent6.workflows._metric import (
    metric_goal as _metric_goal,
)
from agent6.workflows._metric import (
    metric_plateau_nudge as _metric_plateau_nudge,
)
from agent6.workflows._metric import (
    metric_plateau_summary as _metric_plateau_summary,
)
from agent6.workflows._nudges import (
    MEMORY_FINISH_NUDGE as _MEMORY_FINISH_NUDGE,
)
from agent6.workflows._nudges import (
    MEMORY_FLIP_NUDGE as _MEMORY_FLIP_NUDGE,
)
from agent6.workflows._nudges import (
    PLAN_BUDGET_NUDGE as _PLAN_BUDGET_NUDGE,
)
from agent6.workflows._nudges import (
    PLAN_BUDGET_NUDGE_BELOW as _PLAN_BUDGET_NUDGE_BELOW,
)
from agent6.workflows._nudges import (
    PLAN_NUDGE_AFTER_ITERS as _PLAN_NUDGE_AFTER_ITERS,
)
from agent6.workflows._nudges import (
    QUESTION_NUDGE as _QUESTION_NUDGE,
)
from agent6.workflows._nudges import (
    RUN_BUDGET_NUDGE as _RUN_BUDGET_NUDGE,
)
from agent6.workflows._nudges import (
    RUN_BUDGET_NUDGE_BELOW as _RUN_BUDGET_NUDGE_BELOW,
)
from agent6.workflows._nudges import (
    RUN_BUDGET_NUDGE_GATELESS as _RUN_BUDGET_NUDGE_GATELESS,
)
from agent6.workflows._nudges import (
    TASK_FINISH_PATIENCE as _TASK_FINISH_PATIENCE,
)
from agent6.workflows._nudges import (
    VERIFY_FINISH_GATE as _VERIFY_FINISH_GATE,
)
from agent6.workflows._nudges import (
    VERIFY_FINISH_PATIENCE as _VERIFY_FINISH_PATIENCE,
)
from agent6.workflows._nudges import (
    VERIFY_SETTLED_NUDGE as _VERIFY_SETTLED_NUDGE,
)
from agent6.workflows._nudges import (
    VERIFY_SETTLED_NUDGE_AFTER as _VERIFY_SETTLED_NUDGE_AFTER,
)
from agent6.workflows._nudges import (
    VERIFY_SETTLED_STOP_AFTER as _VERIFY_SETTLED_STOP_AFTER,
)
from agent6.workflows._nudges import (
    ends_with_question as _ends_with_question,
)
from agent6.workflows._panel import Decision as ReviewDecision
from agent6.workflows._panel import ReviewContext, render_findings
from agent6.workflows._prompt_revision import (
    PromptRevision as _PromptRevision,
)
from agent6.workflows._prompt_revision import (
    PromptRevisionError as _PromptRevisionError,
)
from agent6.workflows._prompt_revision import (
    clip_text as _clip_text,
)
from agent6.workflows._prompt_revision import (
    format_effective_task as _format_effective_task,
)
from agent6.workflows._prompt_revision import (
    format_prompt_revision_context as _format_prompt_revision_context,
)
from agent6.workflows._prompt_revision import (
    parse_prompt_revision as _parse_prompt_revision,
)
from agent6.workflows._prompts import (
    CONTEXT_SUMMARY_SYSTEM_PROMPT as _CONTEXT_SUMMARY_SYSTEM_PROMPT,
)
from agent6.workflows._prompts import (
    CRITIC_SYSTEM_PROMPT as _CRITIC_SYSTEM_PROMPT,
)
from agent6.workflows._prompts import (
    GIST_DISTILL_SYSTEM_PROMPT as _GIST_DISTILL_SYSTEM_PROMPT,
)
from agent6.workflows._prompts import (
    PROMPT_REVISION_SYSTEM_PROMPT as _PROMPT_REVISION_SYSTEM_PROMPT,
)
from agent6.workflows._prompts import build_system_prompt as _build_system_prompt
from agent6.workflows._prompts import (
    context_restart_notice as _context_restart_notice,
)
from agent6.workflows._review import ReviewDispatch, run_panel
from agent6.workflows._review import Seat as ReviewSeat
from agent6.workflows._run_state import (
    SNAPSHOT_VERSION as _SNAPSHOT_VERSION,
)
from agent6.workflows._run_state import (
    ResumeError,
    RunResult,
)
from agent6.workflows._run_state import (
    load_resume_snapshot as _load_resume_snapshot,
)
from agent6.workflows._toolset import build_readonly_review_tools
from agent6.workflows._toolset import (
    tool_definitions as _tool_definitions,
)

if TYPE_CHECKING:
    from agent6.events import EventSink


# HTTP statuses that will never succeed on a blind retry of the same request.
# 400 bad request, 401/403 auth, 402 insufficient credits, 404 bad
# model/endpoint, 422 malformed body. Retrying these only burns wall-time
# (observed live: a 402 "Insufficient credits" was retried on every turn for the
# rest of the run). 408/409/429 and all 5xx remain retryable and fall through to
# the normal backoff.
_NON_RETRYABLE_HTTP_STATUSES = frozenset({400, 401, 402, 403, 404, 422})

# Upper bound on how long we honor an upstream Retry-After hint. A 429/503 often
# carries Retry-After: <seconds>; we wait at least that long (the provider's own
# backoff is usually shorter and just exhausts the retries before the window
# clears), but never longer than this so a buggy/hostile header can't hang a run.
_RETRY_AFTER_CEILING_S = 120.0


def _provider_error_hint(status_code: int | None) -> str:
    """A short, actionable suffix for a fatal provider error, or "".

    The raw upstream body (e.g. a 401 JSON blob) tells a user nothing about how
    to fix it. Map the common credential/quota statuses to a next step.
    """
    if status_code in (401, 403):
        return (
            " Authentication failed: verify the provider key with `agent6 connect`"
            " or check [providers.<name>].api_key_env."
        )
    if status_code == 402:
        return " Insufficient credits/quota at the provider; top up or switch providers."
    return ""


# Finish/stop reasons that promise a tool call. A response carrying one of these
# but with NO tool_use and NO text is self-contradictory and gets retried (see
# _is_empty_tool_call_response).
_TOOL_CALL_STOP_REASONS = frozenset({"tool_calls", "tool_use"})

# Consecutive went-quiet turns after which a metric run drops the worker's
# per-call output cap from metric_task_max_tokens back to per_call_max_tokens
# (see Workflow._worker_max_tokens). 2 spares a one-off starvation its full
# recovery room while breaking a reasoning-binge spiral.
_STARVATION_BACKOFF_AFTER_QUIETS = 2


def _summarise_assistant_text_for_commit(
    text: str, iteration: int, *, fallback: str = "verify passed"
) -> str:
    """Build a one-line commit subject from the LLM's most recent prose.

    Replaces the constant "agent6 iter N: verify passed" subject
    line with the agent's own first non-empty sentence/line, prefixed
    with the iteration number for traceability. No extra LLM call -
    we already have ``resp.text`` from the same turn that produced the
    verify-passing edits, so the subject is free.

    Behaviour:
      - Strip leading XML/markdown noise (``<thinking>...</thinking>``,
        ``#`` headers, list bullets).
      - Take the first non-empty line, truncate to 72 chars (git's
        ``--oneline`` width), keep the rest as the body for ``git log``.
      - Fall back to "verify passed" when the assistant emitted no
        prose this turn (pure tool-call rounds).
    """
    cleaned = text
    # Drop any leading <thinking>...</thinking> block - common with reasoning models.
    while cleaned.lstrip().startswith("<thinking>"):
        end = cleaned.find("</thinking>")
        if end == -1:
            cleaned = ""
            break
        cleaned = cleaned[end + len("</thinking>") :]
    first_line = ""
    for raw_line in cleaned.splitlines():
        line = raw_line.strip().lstrip("#").lstrip("-*").strip()
        if line:
            first_line = line
            break
    if not first_line:
        first_line = fallback
    subject_body = first_line[:72]
    return f"agent6 iter {iteration}: {subject_body}"


def _is_empty_tool_call_response(resp: Any) -> bool:
    """A self-contradictory provider response: the finish/stop reason says the
    model stopped to make a tool call, but no tool_use and no text came back.

    Observed live with GLM via OpenRouter after a tier-2 context restart (~50% of
    turns): finish_reason=tool_calls with an empty payload (~20 reasoning tokens,
    no content, no tool_calls). A blind retry of the identical request recovers it
    about half the time; without one the loop counts it as went_quiet and the run
    dies at the first compaction. Excludes stop_reason=="length" (deterministic
    reasoning starvation, handled separately with its own nudge)."""
    return (
        str(getattr(resp, "stop_reason", "")) in _TOOL_CALL_STOP_REASONS
        and not resp.tool_uses
        and not (resp.text or "").strip()
    )


def _extract_initial_task(messages: list[dict[str, Any]]) -> str:
    """Recover the original task string from the first user message built
    by ``Workflow.run`` (``"TASK:\\n<task>\\n\\n..."``). Returns empty
    string if the shape is unexpected (resume from a different shape, or
    test fixtures that don't seed a task)."""
    if not messages:
        return ""
    first = messages[0]
    content = first.get("content", "")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", ""))
                break
    if not text.startswith("TASK:\n"):
        return text[:2000]
    body = text[len("TASK:\n") :]
    # The TASK section ends at the first blank line we added.
    end = body.find("\n\n")
    if end == -1:
        return body[:2000]
    return body[:end][:2000]


@dataclass(slots=True)
class _LoopState:
    """Mutable per-run bookkeeping threaded through the agent loop.

    The loop accumulates cross-iteration state: how often each intervention
    (critic rejection, went-quiet / plateau / early-finish nudge, plan/run
    budget nudge) has fired against its cap, the degenerate-repeat-call guard,
    and verify-settled completion tracking. Holding it in one object lets the
    loop's phases be methods that take ``state`` rather than a long parameter
    list, so adding an intervention is a one-field change, not another local
    threaded by hand.
    """

    original_task: str
    tool_calls: int
    metric_history: list[_MetricSample] = field(default_factory=list)
    # Consecutive before_finish critic rejections, so a stubborn worker can't
    # burn the budget bouncing off the critic.
    consecutive_critic_rejections: int = 0
    # Per-run TOTAL review-panel blocks (persisted across resume). Decays on a
    # pass; once it hits review_max_total_rejections the gate auto-disarms to
    # advisory for the rest of the run (oscillation can't burn the budget).
    review_rejections_total: int = 0
    # Last verify result the panel grounds against (None = no verify yet).
    last_verify_ok: bool | None = None
    last_verify_tail: str = ""
    # True once the tree has been edited since the last green verify (spans
    # iterations, unlike the per-iteration edit_since_verify_pass). Makes a
    # stale earlier pass not count as "currently green" for the finish gate.
    edited_since_verify: bool = False
    # Degenerate-loop guard: a back-to-back streak of the same (tool, args)
    # signature. Reset on any change so a normal re-read after edits is fine.
    last_tool_signature: str | None = None
    repeat_streak: int = 0
    repeat_warning_emitted_at: int = 0
    # Intervention nudge counters (each capped by a module-level patience const).
    went_quiet_nudges_used: int = 0
    plateau_nudges_used: int = 0
    metric_finish_nudges_used: int = 0
    task_finish_nudges_used: int = 0
    verify_finish_nudges_used: int = 0
    plan_finish_nudged: bool = False
    # A turn that ends in a prose question with no tool_use is nudged ONCE to
    # call ask_user (or finish_run) instead of narrating; then silent_finish is
    # accepted so a model that ignores the nudge cannot loop the run.
    question_nudged: bool = False
    # verify-settled completion (run mode): once verify has passed -- or, on a
    # gateless run, once an edit has been committed -- count no-progress
    # iterations; nudge then stop a worker that spins after success.
    verify_ever_passed: bool = False
    gateless_ever_committed: bool = False
    verify_settled_idle: int = 0
    verify_settled_nudged: bool = False
    run_budget_nudged: bool = False
    # Cross-run memory write nudges (run mode, memory store wired): one flip
    # advisory when verify first goes green after failing, one deferred
    # finish_run as the backstop. Both suppressed once the worker records
    # anything; a run whose verify never failed is never nudged.
    verify_ever_failed: bool = False
    memory_written: bool = False
    memory_flip_nudged: bool = False
    memory_finish_nudged: bool = False
    # surface-current-task: id of the subtask last injected as the focus banner.
    # Re-surface only on a focus change or after a tier-2 restart (reset to None
    # there) -- the banner survives tier-1 elision, so the worker keeps seeing it
    # between those events without re-appending it every turn.
    surfaced_task_id: str | None = None
    # anti-grind: the focus task being counted, how many consecutive turns it has
    # held (NOT reset by compaction -- only by forward motion), and how many stuck
    # nudges have fired for THIS focus task (reset on focus change; capped).
    last_focus_id: str | None = None
    turns_on_task: int = 0
    stuck_nudges_fired: int = 0


class _NextTurn:
    """Sentinel returned by ``_turn_provider_call``: the turn was discarded (a
    mid-stream steer chose continue, or injected an instruction) and the loop
    should start the next iteration immediately."""


_NEXT_TURN = _NextTurn()


@dataclass(slots=True)
class _TurnState:
    """Mutable bookkeeping for ONE assistant turn that dispatched tools.

    ``_drive_loop`` creates one per tool-use iteration and threads it through
    the turn phases; a field earns its place by being written in one phase and
    read in a later one, so each phase is a method taking ``(state, turn)``
    rather than a slice of ~15 hand-threaded locals. Cross-iteration state
    stays on ``_LoopState``.
    """

    iteration: int
    # The provider response driving this turn (duck-typed; see Provider.call).
    resp: Any
    # A finish_run/finish_planning call captured this turn; the finish gates
    # may revoke it (set back to None) before the stop checks honour it.
    finish_signal: str | None = None
    finish_payload: dict[str, Any] | None = None
    finish_kind: Literal["finish_run", "finish_planning"] = "finish_run"
    # The user-turn content accumulated for this turn: tool_result blocks
    # first, then any advisory text notices (critic, metric, nudges).
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    verify_just_passed: bool = False
    verify_just_failed: bool = False
    # Verify went green THIS turn after the run's last verify was red; feeds
    # the one-shot add_memory flip advisory in _turn_notices.
    verify_flipped_green: bool = False
    # An apply_edit/apply_patch AFTER a passing verify in the same turn changes
    # the tree that verify validated, so the green no longer applies. Tracked
    # separately from verify_just_passed (which the metric path also reads) so
    # only the auto-commit gate is affected.
    edit_since_verify_pass: bool = False
    edited: bool = False
    committed: bool = False
    dag_mutated: bool = False
    metric_after_verify_pass: bool = False
    metric_feedback: str | None = None
    metric_plateau_finish: str | None = None
    critic_text: str | None = None
    plateau_should_stop: bool = False
    verify_settled_stop: bool = False


@dataclass
class Workflow:
    """Single-loop agent workflow.

    The agent decides everything via tool calls in one large loop:
    when to read, when to plan (implicitly via subsequent tool calls),
    when to edit, when to verify, when to measure the metric, when to
    pivot, when to stop. The harness keeps the loop bounded
    (max_iterations, budget caps, verify_timeout) and observable
    (events).
    """

    root: Path
    config: Config
    provider: Provider
    dispatcher: ToolDispatcher
    logger: Callable[[str], None] = field(default=print)
    events: EventSink | None = None
    # GraphClient (connected to a running curator). When None,
    # DAG-as-tool handlers raise ToolError and the loop runs without DAG
    # persistence (still usable for bench / one-off tasks). When wired,
    # Workflow.run() seeds a root task and the agent can add subtasks
    # and update statuses; survives crashes via <run-dir>/graph.jsonl.
    graph_client: GraphClient | None = None
    # Per-invocation token budget tracker (the same instance wired into
    # the provider). When present the loop can read how much budget
    # remains and use it to decide whether a metric plateau is worth
    # quitting on. None in test / MCP paths; the loop degrades to fixed
    # count-based heuristics when it is unset.
    budget: BudgetTracker | None = None
    # Per-repo state dir holding the cross-run memory store
    # (<state_dir>/memories/). When set, active memories are injected into
    # the system prompt at run start; the CLI wires the same path into the
    # dispatcher so add_memory / invalidate_memory persist across runs.
    # None (bench / tests / one-off embedders) runs memory-less.
    state_dir: Path | None = None
    # Hard cap on assistant turns. Each turn = one provider.call. With the
    # default tool-use-loop pattern, agents take 30-100 turns on a non-
    # trivial task; 200 is well above that without being unbounded.
    max_iterations: int = 200
    # Per-call max_tokens for the LLM response. NOT the bench's total
    # output budget (that's BudgetTracker's job). Sized for ONE turn:
    # enough for reasoning + tool-call args + content on a reasoning
    # model, small enough to fit alongside the input in a 262k-context
    # model like Kimi 2.6. Sonnet (no reasoning) uses ~600 of this;
    # Kimi-k2.6 reasoning needs ~5-15k.
    per_call_max_tokens: int = 16384
    # Per-call output cap for the worker on metric-optimization runs (mode
    # "run" with a configured continuous metric). Those tasks reward large
    # single-turn edits, rewriting a hot function wholesale beats nibbling
    # at it across turns, and the worker routinely truncated mid-apply_patch
    # against the 16k default, wasting the whole turn. Lifting the ceiling
    # only when a metric goal is present keeps ordinary feature/bugfix runs
    # (where giant turns mostly mean a confused model) on the tighter cap.
    #
    # Bumped 32768 -> 65536: a heavy reasoner (Kimi K2.6, perf-takehome) was
    # *still* hitting the 32k cap with stop_reason="length", its reasoning ate
    # the whole budget and the turn ended before it could emit a tool call, so
    # ~30% of turns were pure waste and the run made no progress. The bigger
    # ceiling lets a reason-heavy turn finish and actually apply its edit.
    metric_task_max_tokens: int = 65536
    # Sampling temperature pinned for every provider
    # call (worker and critic). agent6 was previously passing
    # `temperature: None` through every call, meaning each provider
    # routed to its own default. OpenRouter's per-model defaults are
    # high enough that we observed Kimi K2.6 emitting 15997 literal
    # `\n` escapes inside a single `old_string` argument before hitting
    # the completion-tokens cap. Pinning 0.0 by default makes the
    # tool-use loop reproducible and removes one large degenerate-output
    # surface. CLI wires these from `cfg.models.<role>.temperature`.
    temperature: float | None = 0.0
    critic_temperature: float | None = 0.0
    # Tiered context compaction thresholds (chars).
    compact_drop_at_chars: int = _DROP_BLOCKS_AT_CHARS
    compact_summarise_at_chars: int = _SUMMARISE_AT_CHARS
    # Retry the provider call on transient ProviderError before aborting the
    # run. Common cases: Anthropic 529 overload, Anthropic "Server disconnected
    # without sending a response" (httpx2 RemoteProtocolError, no HTTP status),
    # OpenRouter 502, brief socket timeouts. Such a disconnect can flap for a
    # few seconds, so a single retry (the previous default) was too weak: one
    # bad blip aborted a long, expensive run that is otherwise fully
    # resumable. With exponential backoff (2s/4s/8s/16s, full-jittered, capped
    # at provider_retry_max_delay_s) four retries ride out a multi-second flap;
    # permanent statuses (401/402/403/404/422) and BudgetExceeded still fail
    # fast. Set to 0 to disable retrying.
    provider_retry_count: int = 4
    provider_retry_delay_s: float = 2.0
    provider_retry_max_delay_s: float = 30.0
    # Steering interrupt callbacks . Polled
    # between iterations; on request the workflow prompts the operator for an
    # instruction or "abort". When unset (the defaults) the loop runs without
    # operator interaction. audit finding wired this in.
    steer_requested: Callable[[], bool] = field(default=lambda: False)
    steer_clear: Callable[[], None] = field(default=lambda: None)
    steer_prompt: Callable[[], str | None] = field(default=lambda: None)
    # Polled DURING a streaming model call (not just between steps): True once the
    # operator has asked to stop, so a long reasoning turn aborts promptly.
    should_abort: Callable[[], bool] = field(default=lambda: False)
    # Polled DURING a streaming call: True once the operator has asked to STEER
    # (Ctrl-C / TUI `s`), so the watchdog ends the turn and the loop reaches its
    # steer boundary (the menu) at once instead of waiting the whole turn out.
    should_interrupt: Callable[[], bool] = field(default=lambda: False)
    # Hook invoked once per successful auto-commit (after the
    # commit lands). Returning "stop" exits the loop cleanly with
    # completed=True, reason="interactive_stop"; "continue" (the default)
    # lets the next iteration run. The CLI's `agent6 run -i` installs a
    # TTY prompt here for the REPL; default no-op preserves autonomous
    # behaviour for `agent6 run` and `agent6 resume`.
    after_auto_commit: Callable[[int, str], Literal["continue", "stop"]] = field(
        default=lambda _i, _sha: "continue"
    )
    # critic-in-loop. When `critic_provider` is set AND
    # `critic_mode != "off"`, the workflow invokes the critic at the
    # configured trigger (verify-failure / before finish_run / every
    # critic_period iters) and injects its critique back into the
    # conversation as a synthetic text block on the next user turn so
    # the worker sees it on the following iteration. Default off keeps
    # the single-provider behaviour intact.
    critic_provider: Provider | None = None
    critic_mode: Literal["off", "on_verify_fail", "before_finish", "periodic"] = "off"
    critic_period: int = 10
    # Optional one-shot prompt revision before the first worker call.
    # The CLI wires this to the reviewer model when prompt.revise_prompt !=
    # "off". It never receives tools and never iterates.
    prompt_reviser_provider: Provider | None = None
    revise_prompt: Literal["off", "auto", "interactive"] = "off"
    prompt_reviser_temperature: float | None = 0.0
    prompt_revision_max_tokens: int = 2048
    prompt_revision_selector: Callable[[str, str, tuple[str, ...]], str | None] | None = None
    # Tier-2 context compaction (summarise-and-restart). When the
    # cumulative tool_result size crosses ``compact_summarise_at_chars``,
    # the loop asks this provider to summarise the elided history into a
    # compact progress block and restarts the message list from (original
    # task + summary). Wired by the CLI to the reviewer role (cheaper than
    # the worker). When None the loop falls back to ``provider`` so the
    # feature still works without explicit wiring.
    summariser_provider: Provider | None = None
    context_summary_max_tokens: int = 2048
    # Tier-1 gist elision (``context.elision_gists``): large read_file results
    # decay to a distilled-gist placeholder (summariser model, one batched call
    # per drop event) before the bare marker. Off = pre-gist behavior.
    compact_elision_gists: bool = True
    # Cap on consecutive `before_finish` rejections.
    # When the worker repeatedly calls finish_run and the critic keeps
    # saying NEEDS_WORK, the loop would otherwise burn budget bouncing.
    # After this many back-to-back rejections, the next finish_run is
    # accepted (with a `[critic]` warning still injected so the
    # transcript records the disagreement). 0 disables the cap.
    max_consecutive_critic_rejections: int = 2
    # Adversarial review panel. When ``review_seats`` is non-empty the in-loop
    # critic triggers run the grounded PANEL (run_panel over the run diff +
    # verify result) instead of the single critic. ``review_decision`` gates only
    # for veto/quorum; "advisory" just injects findings as a [review] message.
    # The panel reviews ``git diff base_sha`` (the run's cumulative change). The
    # per-run rejection counter auto-disarms the gate after
    # ``review_max_total_rejections`` blocks so it can never stall the run.
    review_seats: list[ReviewSeat] = field(default_factory=list)
    review_decision: ReviewDecision = "advisory"
    review_quorum: int = 2
    review_max_total_rejections: int = 4
    review_budget_fraction: float = 0.25
    review_concurrency: int = 1
    base_sha: str = ""
    # When set, : Workflow writes a JSON snapshot of (system, messages,
    # tool_calls, next_iteration, root_task_id) before every LLM call. The
    # snapshot is provider-agnostic (it holds the anthropic-shaped message
    # list the loop maintains internally, not the on-the-wire OpenAI-shaped body
    # the openai provider sends) so `agent6 resume` works regardless of which
    # provider the prior run used. Atomic write (tmp + rename) so a crash
    # mid-write leaves the prior snapshot intact.
    resume_state_path: Path | None = None
    # Plan mode. When ``mode="plan"``, the workflow uses the
    # planning system prompt + plan-mode tool list (no apply_edit /
    # apply_patch; finish_planning replaces finish_run), skips auto-
    # commit-on-verify-pass, and on finish_planning writes the
    # ``plan_markdown`` argument to ``plan_output_path`` before exiting.
    # ``plan_output_path`` is required when ``mode="plan"``.
    mode: Literal["run", "plan", "ask", "machine", "agent"] = "run"
    plan_output_path: Path | None = None
    # weak-model resilience. Open-weights models (observed live
    # with Kimi K2.6) sometimes emit a single empty assistant turn
    # mid-run (no text, no tool_use, stop_reason="end_turn" or
    # equivalent) and would otherwise terminate the run immediately.
    # When `went_quiet_max_nudges > 0`, the loop instead injects a
    # short [harness] notice into the conversation and re-asks the
    # model, up to this many times PER RUN. Reset on any non-empty
    # turn. Set to 0 to restore the "fail fast on went_quiet"
    # behaviour. Raised from 2 to 4 after observing K2.6 perf
    # runs terminating after 5 tool calls because reasoning-starvation
    # bursts (32k tokens spent on reasoning, empty content + empty
    # tool_calls) count as went_quiet and exhausted the nudge budget
    # before the model had a chance to make real progress.
    went_quiet_max_nudges: int = 4
    # loop-guard escalation. The guard injects a one-shot
    # notice when the same (tool, args) signature streak hits
    # `repeat_threshold` (default 3). When the worker ignores it and the
    # streak reaches `loop_guard_kill_threshold`, the loop forcibly
    # terminates with reason="loop_guard_killed" rather than letting
    # the worker burn the rest of the budget circling the same call.
    # Set to 0 to disable forced termination (notice-only behaviour).
    loop_guard_kill_threshold: int = 10
    # One-shot guard so a persistently unwritable state dir (full disk, quota,
    # read-only mount) warns once instead of every turn. Snapshot persistence is
    # recovery state; a failure disables resume/fork but must not abort the run.
    _snapshot_write_failed: bool = field(default=False, init=False)

    def run(self, user_task: str) -> RunResult:
        """Drive the single-loop agent to completion."""
        if self.mode == "plan" and self.plan_output_path is None:
            raise ValueError("Workflow(mode='plan') requires plan_output_path to be set")
        self._emit("run.start", user_task=user_task[:200], mode=self.mode)
        self._log("LOOP: LOAD_CONTEXT")
        repo = self._load_repo_summary()
        system = _build_system_prompt(
            config=self.config, repo=repo, mode=self.mode, memories=self._load_memories()
        )

        try:
            effective_task = self._maybe_revise_prompt(user_task, repo)
        except _PromptRevisionError as exc:
            self._log(f"LOOP: prompt revision failed: {exc}")
            self._emit(
                "run.end",
                reason="prompt_revision_failed",
                iterations=0,
                all_passed=False,
            )
            return RunResult(
                completed=False,
                reason="prompt_revision_failed",
                summary=str(exc),
                iterations=0,
                tool_calls=0,
            )

        # Seed the run's root task and wire its id into the
        # dispatcher so add_task with parent_id=None has a parent. Skipped
        # gracefully if no graph_client is configured (DAG tools then
        # raise ToolError if called).
        root_id = self._seed_root_task(effective_task)
        if root_id is not None:
            self.dispatcher.set_run_root_node_id(root_id)
            self._log(f"LOOP: DAG root task seeded: {root_id}")
            self._emit_graph_snapshot()  # show the root in the live task view

        tools = _tool_definitions(self.dispatcher, mode=self.mode)
        self._log(
            f"LOOP: mode={self.mode} system={len(system)} chars,"
            f" tools={len(tools)}, task={len(effective_task)} chars"
        )

        # Initial user message - the task + a brief operational header.
        # Cache breakpoints are placed by _roll_cache_breakpoints each
        # iteration, so the growing history stays cached across turns.
        dag_hint = _initial_dag_hint(root_id, self.mode, self.config.prompt.decompose == "on")
        if self.mode == "plan":
            instructions = (
                "Begin planning. Use the read-only tools to gather what you"
                " need, then call `finish_planning` exactly once with the"
                " plan markdown."
            )
        elif self.mode == "machine":
            instructions = (
                "Author the machine now and return it via a single"
                " `finish_run` call (the complete `.asm.toml` in `result.toml`)."
                " Do not edit files or run anything."
            )
        elif self.mode == "agent":
            instructions = (
                "Do the task above, then call `finish_run` exactly once with a"
                " `result` object matching the schema named in the task. This is"
                " ONE step of a state machine, not a coding session — read only"
                " what the task needs and do NOT edit the repo or run verify."
            )
        else:
            instructions = (
                "Begin. Use the tools to read what you need, make edits,"
                " run verify, and call `finish_run` when done."
            )
        initial_user = f"TASK:\n{effective_task}\n\n{instructions}{dag_hint}"
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": initial_user}]}
        ]

        return self._drive_loop(
            system=system,
            messages=messages,
            tools=tools,
            tool_calls=0,
            start_iteration=1,
            root_task_id=root_id,
            original_task=effective_task,
        )

    def resume(self) -> RunResult:
        """Resume a paused/crashed run from its snapshot.

        Reads ``self.resume_state_path`` (the snapshot written by the
        loop before each LLM call), reattaches the DAG root task id to
        the dispatcher, and re-enters the loop at the saved iteration
        with the saved messages list. The budget tracker is fresh per
        invocation (by design - see ``agent6.budget`` docstring); the
        DAG state on disk is restored by spawning a curator against the
        same run layout in the CLI.
        """
        if self.resume_state_path is None:
            raise ResumeError("resume() called but resume_state_path is None")
        try:
            snapshot = _load_resume_snapshot(self.resume_state_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ResumeError(
                f"failed to load resume snapshot from {self.resume_state_path}: {exc}"
            ) from exc

        self._emit(
            "loop.resume.start",
            iteration=snapshot.next_iteration,
            messages=len(snapshot.messages),
        )
        self._log(
            f"LOOP: RESUME from {self.resume_state_path} "
            f"(iter={snapshot.next_iteration}, messages={len(snapshot.messages)}, "
            f"tool_calls={snapshot.tool_calls})"
        )

        if snapshot.root_task_id is not None:
            self.dispatcher.set_run_root_node_id(snapshot.root_task_id)
            self._log(f"LOOP: DAG root task restored: {snapshot.root_task_id}")

        # Honour self.mode: resuming a plan run must not hand the worker the
        # mutating run-mode tools (run() builds its list the same way).
        tools = _tool_definitions(self.dispatcher, mode=self.mode)
        return self._drive_loop(
            system=snapshot.system,
            messages=snapshot.messages,
            tools=tools,
            tool_calls=snapshot.tool_calls,
            start_iteration=snapshot.next_iteration,
            root_task_id=snapshot.root_task_id,
            review_rejections_total=snapshot.review_rejections_total,
            verify_ever_passed=snapshot.verify_ever_passed,
            gateless_ever_committed=snapshot.gateless_ever_committed,
            metric_best_score=snapshot.metric_best_score,
            metric_at_ceiling=snapshot.metric_at_ceiling,
        )

    def _drive_loop(  # noqa: PLR0911
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        tool_calls: int,
        start_iteration: int,
        root_task_id: str | None,
        original_task: str | None = None,
        review_rejections_total: int = 0,
        verify_ever_passed: bool = False,
        gateless_ever_committed: bool = False,
        metric_best_score: float | None = None,
        metric_at_ceiling: bool = False,
    ) -> RunResult:
        """Shared loop body for both fresh ``run()`` and ``resume()``: one
        ``_TurnState`` per tool-use iteration, driven through the turn phases
        in order. Any phase returning a RunResult ends the run.

        Before each provider call, writes a snapshot of the workflow's
        in-memory state to ``self.resume_state_path`` (if set) so a
        crash mid-call can be resumed from the same point.
        """
        # Cache the original task for in-loop critic calls.
        # Prefer the task passed straight from run() (exact, never truncated);
        # fall back to recovering it from messages[0] for resume(), where only
        # the snapshotted messages list is available. The fallback splits on the
        # first blank line, so a multi-paragraph task survives intact only via
        # the passed-in value -- which is why run() threads it explicitly.
        original_task = original_task or _extract_initial_task(messages)
        state = _LoopState(original_task=original_task, tool_calls=tool_calls)
        state.review_rejections_total = review_rejections_total  # survives resume
        # Restore completion-relevant state from the snapshot (all default to the
        # fresh-run values, so run() is unaffected). Without this the metric and
        # verify-settled stop logic regress to zero after a resume.
        state.verify_ever_passed = verify_ever_passed
        state.gateless_ever_committed = gateless_ever_committed
        if metric_at_ceiling or metric_best_score is not None:
            # Seed a single synthetic sample so `_metric_at_ceiling` and the
            # plateau guard see the prior best (we persist a compact summary, not
            # the full history, by design). `label` marks it as
            # resume-reconstructed. A consequence of that summary-only model:
            # `_metric_plateau_summary` needs several parsed samples to fire, so a
            # resumed already-plateaued run takes a few measurements to re-arm the
            # plateau-stop (it never stops early, and the ceiling-stop above is
            # immediate) -- the predictable trade for not carrying the whole
            # sample history across resume.
            state.metric_history.append(
                _MetricSample(
                    label="resumed",
                    score=metric_best_score,
                    returncode=0,
                    at_ceiling=metric_at_ceiling,
                )
            )
        for iteration in range(start_iteration, self.max_iterations + 1):
            self._turn_pre_call(
                system=system,
                messages=messages,
                state=state,
                iteration=iteration,
                start_iteration=start_iteration,
                root_task_id=root_task_id,
            )
            got = self._turn_provider_call(system, messages, tools, state, iteration=iteration)
            if isinstance(got, RunResult):
                return got
            if isinstance(got, _NextTurn):
                continue
            # Reconstruct the assistant message exactly from the response
            # content blocks so tool_use IDs round-trip cleanly.
            messages.append({"role": "assistant", "content": got.raw.get("content") or []})
            if not got.tool_uses:
                result = self._handle_no_tool_use(got, messages, state, iteration=iteration)
                if result is not None:
                    return result
                continue
            turn = _TurnState(iteration=iteration, resp=got)
            result = self._turn_dispatch_tools(state, turn)
            if result is not None:
                return result
            # One task-DAG snapshot per turn (not per mutation), so several
            # add_task/update_task calls in a turn collapse to a single event.
            if turn.dag_mutated:
                self._emit_graph_snapshot()
            result = self._turn_auto_commit_and_metric(state, turn)
            if result is not None:
                return result
            self._turn_critic_triggers(state, turn, messages)
            self._turn_finish_gates(state, turn, messages)
            self._turn_notices(state, turn)
            self._turn_metric_plateau(state, turn)
            self._turn_verify_settled(state, turn)
            messages.append({"role": "user", "content": turn.tool_results})
            # Snapshot AFTER the executed tools (assistant turn + tool_results
            # are now in `messages`) so a crash before iteration N+1's pre-call
            # snapshot resumes from AFTER the dispatched tools instead of
            # replaying them. Without this, a kill after a non-idempotent tool
            # (run_command `>>`, apply_patch, a migration) but before the next
            # iteration would re-issue iteration N's identical provider call and
            # re-dispatch the same tool_uses (temperature 0.0 makes re-emission
            # likely) -- double-applying edits / re-running commands.
            self._save_resume_snapshot(
                system=system,
                messages=messages,
                tool_calls=state.tool_calls,
                next_iteration=iteration + 1,
                root_task_id=root_task_id,
                state=state,
            )
            result = self._turn_stop_checks(state, turn)
            if result is not None:
                return result
            # Poll the steering flag between iterations. The operator can press
            # Ctrl-C once to drop a steering instruction into the conversation;
            # a second Ctrl-C within 2s raises KeyboardInterrupt and aborts.
            # The safe boundary is AFTER a complete iter so we never split a
            # tool_use / tool_result pair.
            outcome = self._steer_outcome(
                self._maybe_handle_steer(messages, iteration), iteration, state
            )
            if outcome is not None:
                return outcome

        self._log(f"LOOP: max_iterations={self.max_iterations} reached")
        self._final_checkpoint(self.max_iterations)
        self._emit(
            "run.end",
            reason="max_iterations",
            iterations=self.max_iterations,
            all_passed=False,
        )
        return RunResult(
            completed=False,
            reason="max_iterations",
            summary=f"max_iterations={self.max_iterations} reached without finish_run",
            iterations=self.max_iterations,
            tool_calls=state.tool_calls,
        )

    def _turn_pre_call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        state: _LoopState,
        iteration: int,
        start_iteration: int,
        root_task_id: str | None,
    ) -> None:
        """Prepare the context for this turn's provider call: budget heartbeat,
        tiered compaction, pre-call nudges, rolling cache breakpoints, then the
        pre-call resume snapshot.

        The cache breakpoints advance AFTER compaction + nudges (the tail must
        be final) and BEFORE the snapshot (markers persist across resume).
        After the snapshot write, a crash anywhere up to the next iteration's
        snapshot can be resumed by re-running this same call."""
        self._emit_budget(iteration)
        if self._maybe_compact(messages):
            # A tier-2 restart wiped the surfaced focus banner; let the next
            # nudge pass re-surface the current task into the fresh context.
            state.surfaced_task_id = None
        self._maybe_pre_call_nudges(
            messages, state, iteration=iteration, start_iteration=start_iteration
        )
        _roll_cache_breakpoints(messages)
        self._save_resume_snapshot(
            system=system,
            messages=messages,
            tool_calls=state.tool_calls,
            next_iteration=iteration,
            root_task_id=root_task_id,
            state=state,
        )

    def _turn_provider_call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        state: _LoopState,
        *,
        iteration: int,
    ) -> RunResult | _NextTurn | Any:
        """One worker call with terminal-error classification. Returns the
        provider response on success, a RunResult to end the run, or
        ``_NEXT_TURN`` when a mid-stream steer discarded the turn (the menu
        chose continue, or injected an instruction, so the turn is re-done)."""
        try:
            return self._call_with_retry(
                system,
                messages,
                tools,
                self._worker_max_tokens(state),
            )
        except BudgetExceeded as exc:
            self._log(f"LOOP: budget exhausted at iter {iteration} ({exc})")
            self._emit(
                "run.end",
                reason="budget_exhausted",
                iterations=iteration,
                all_passed=False,
            )
            return RunResult(
                completed=False,
                reason="budget_exhausted",
                summary=f"budget exhausted at iter {iteration}: {exc}",
                iterations=iteration,
                tool_calls=state.tool_calls,
            )
        except ProviderAborted:
            self.steer_clear()  # consume the stop; don't leave it on disk to re-read
            self._log(f"LOOP: operator stopped the run mid-turn at iter {iteration}")
            self._emit("run.end", reason="steer_abort", iterations=iteration, all_passed=False)
            return RunResult(
                completed=False,
                reason="steer_abort",
                summary=f"operator stopped the run at iter {iteration}",
                iterations=iteration,
                tool_calls=state.tool_calls,
            )
        except ProviderInterrupted:
            # A steer was requested mid-stream; the watchdog ended the (thinking)
            # turn so we handle it now rather than wait it out. The partial turn
            # is discarded; the menu decides continue / steer / stop / detach.
            self._log(f"LOOP: steer requested mid-turn at iter {iteration}")
            outcome = self._steer_outcome(
                self._maybe_handle_steer(messages, iteration), iteration, state
            )
            if outcome is not None:
                return outcome
            return _NEXT_TURN  # "continue" or an injected instruction -> re-do the turn
        except ProviderError as exc:
            hint = _provider_error_hint(exc.status_code)
            self._log(f"LOOP: provider error at iter {iteration}: {exc}{hint}")
            self._emit(
                "run.end",
                reason="provider_error",
                iterations=iteration,
                all_passed=False,
            )
            return RunResult(
                completed=False,
                reason="provider_error",
                summary=f"provider error at iter {iteration}: {exc}{hint}",
                iterations=iteration,
                tool_calls=state.tool_calls,
            )

    def _turn_dispatch_tools(self, state: _LoopState, turn: _TurnState) -> RunResult | None:
        """Dispatch each tool_use in the turn, appending one tool_result per
        call and noting effects (verify / metric / edits / DAG / finish) on
        ``turn``. Returns a RunResult only for the unexecutable-operator-
        command abort; tool errors become error tool_results instead."""
        # This iteration produced tool_uses, so the went_quiet
        # nudge budget refills (failures are per-streak, not per-run).
        state.went_quiet_nudges_used = 0
        for tu in turn.resp.tool_uses:
            state.tool_calls += 1
            name = tu.get("name", "")
            tool_input = tu.get("input", {})
            tu_id = tu.get("id", "")
            # degenerate-loop signature tracking. Stable
            # JSON so dict key order does not break equality. Same
            # (name, args) back-to-back across iterations increments
            # `state.repeat_streak`; anything else resets it.
            try:
                sig = f"{name}:{json.dumps(tool_input, sort_keys=True, ensure_ascii=False)}"
            except (TypeError, ValueError):
                sig = f"{name}:<unhashable>"
            if sig == state.last_tool_signature:
                state.repeat_streak += 1
            else:
                state.last_tool_signature = sig
                state.repeat_streak = 1
            self._emit("loop.tool.call", name=name, iteration=turn.iteration)
            try:
                result = self.dispatcher.dispatch(name, tool_input)
                content = json.dumps(result, ensure_ascii=False)
                self._note_tool_effects(state, turn, name, result)
            except ToolError as exc:
                content = json.dumps({"error": str(exc)})
                self._log(f"  tool_error: {name}: {exc}")
            except OperatorCommandUnexecutable as exc:
                return self._unexecutable_abort(
                    exc, iteration=turn.iteration, tool_calls=state.tool_calls
                )
            turn.tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": _cap_tool_result(content, tool_name=name),
                }
            )
            self._capture_finish(turn, name, tool_input)
        return None

    def _note_tool_effects(
        self, state: _LoopState, turn: _TurnState, name: str, result: dict[str, Any]
    ) -> None:
        """Record a dispatched tool's side effects on the turn: verify results
        (they feed auto-commit-on-verify-pass and ground the review panel:
        verify-pass presumes correctness, verify-red is the hard signal),
        manual metric samples, tree edits, and DAG mutations."""
        if name == "run_verify_command":
            rc = result.get("returncode")
            if rc == 0:
                turn.verify_just_passed = True
                if state.last_verify_ok is False:
                    turn.verify_flipped_green = True
                # This verify validated the current tree; any earlier
                # edit is now covered.
                turn.edit_since_verify_pass = False
                state.edited_since_verify = False
            elif rc is not None:
                turn.verify_just_failed = True
                state.verify_ever_failed = True
            if rc is not None:
                state.last_verify_ok = rc == 0
                tail = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
                state.last_verify_tail = tail.strip()[-2000:]
        elif name == "run_metric_command":
            if turn.verify_just_passed:
                turn.metric_after_verify_pass = True
            turn.metric_feedback = self._record_metric_result(
                state.metric_history,
                result,
                iteration=turn.iteration,
                label=f"manual iter {turn.iteration}",
                sha="",
            )
            if turn.verify_just_passed:
                turn.metric_plateau_finish = self._metric_plateau_summary(state.metric_history)
        elif name == "add_memory":
            # Only successful dispatches reach here, so the write persisted;
            # both memory nudges stay quiet for the rest of the run.
            state.memory_written = True
        if name in ("apply_edit", "apply_patch"):
            turn.edited = True
            # Invalidate a same-turn earlier verify pass: the commit
            # gate must not label this edited tree "verify passed".
            turn.edit_since_verify_pass = True
            state.edited_since_verify = True
        if name in _DAG_MUTATING_TOOLS:
            turn.dag_mutated = True  # snapshot once after the turn

    def _capture_finish(self, turn: _TurnState, name: str, tool_input: Any) -> None:
        """Capture a finish_run / finish_planning call's summary + payload on
        the turn (the finish gates may still revoke it). finish_planning also
        persists the plan markdown: schema validation already guaranteed the
        field when the dispatcher dispatched it, but the raw tool_input is what
        the model sent us, so stay defensive against a malformed call."""
        if name == FinishRunInput.TOOL_NAME:
            turn.finish_kind = "finish_run"
            turn.finish_signal = (
                tool_input.get("summary", "(no summary)")
                if isinstance(tool_input, dict)
                else "(no summary)"
            )
            raw_result = tool_input.get("result") if isinstance(tool_input, dict) else None
            turn.finish_payload = raw_result if isinstance(raw_result, dict) else None
        elif name == FinishPlanningInput.TOOL_NAME:
            turn.finish_kind = "finish_planning"
            turn.finish_signal = (
                tool_input.get("summary", "(no summary)")
                if isinstance(tool_input, dict)
                else "(no summary)"
            )
            plan_md = ""
            if isinstance(tool_input, dict):
                plan_md = str(tool_input.get("plan_markdown", ""))
            if self.plan_output_path is not None and plan_md:
                try:
                    self.plan_output_path.parent.mkdir(parents=True, exist_ok=True)
                    self.plan_output_path.write_text(plan_md, encoding="utf-8")
                    self._log(f"  plan written: {self.plan_output_path} ({len(plan_md)} chars)")
                    self._emit(
                        "loop.plan_written",
                        path=str(self.plan_output_path),
                        bytes=len(plan_md.encode("utf-8")),
                    )
                except OSError as exc:
                    self._log(f"  plan write failed: {exc}")
                    self._emit(
                        "loop.plan_write.failed",
                        path=str(self.plan_output_path),
                        error=str(exc),
                    )

    def _turn_auto_commit_and_metric(self, state: _LoopState, turn: _TurnState) -> RunResult | None:
        """Auto-commit the turn's work, then take the automatic metric sample.

        With a verify command, commits are gated on a green verify (the agent
        shouldn't need to remember to ``git commit`` after a green verify);
        with none configured (a gateless run), each editing step is committed
        as an un-gated checkpoint so resume + the audit trail still work.
        ``turn.edited`` (apply_edit/apply_patch) is the cheap fast-path; the
        worktree-dirty fallback catches run_command-authored edits (else they'd
        never be committed gateless). Plan mode is read-only and never commits.
        Best-effort: commit failures (e.g. nothing to commit) are logged but
        don't abort the run; the catch includes OSError so a transient FS
        hiccup doesn't kill an otherwise-fine run.

        Returns a RunResult for the REPL hook's "stop" directive or an
        unexecutable operator metric command; None otherwise."""
        gateless = not self.config.workflow.verify_command
        gateless_changed = gateless and (turn.edited or self._worktree_dirty())
        verified_commit = turn.verify_just_passed and not turn.edit_since_verify_pass
        if self.mode != "run" or not (verified_commit or gateless_changed):
            return None
        commit_subject = _summarise_assistant_text_for_commit(
            turn.resp.text or "",
            turn.iteration,
            fallback="checkpoint" if gateless else "verify passed",
        )
        sha = ""
        try:
            sha = commit_all(self.root, commit_subject)
            self._log(f"  auto-commit: {sha[:12]}")
            self._emit("loop.auto_commit", iteration=turn.iteration, sha=sha)
            turn.committed = bool(sha)
            if gateless and sha:
                # Seed the idle-stop net for gateless runs (no green verify
                # ever fires); see the verify-settled bookkeeping.
                state.gateless_ever_committed = True
            if sha:
                # Surface "what the worker just changed" to a live viewer
                # (the TUI diff panel). Capped; best-effort.
                self._emit(
                    "diff.updated",
                    sha=sha,
                    patch=commit_diff(self.root, sha, max_bytes=8000),
                )
        except (GitError, OSError) as exc:
            self._report_auto_commit_failure(exc, commit_subject, iteration=turn.iteration)
        # REPL hook. Default no-op returns "continue".
        if sha:
            directive = self.after_auto_commit(turn.iteration, sha)
            if directive == "stop":
                self._log(f"LOOP: interactive stop at iter {turn.iteration}")
                self._emit_run_end_passed(reason="interactive_stop", iterations=turn.iteration)
                return RunResult(
                    completed=True,
                    reason="interactive_stop",
                    summary=f"stopped interactively after iter {turn.iteration}",
                    iterations=turn.iteration,
                    tool_calls=state.tool_calls,
                )
        if not turn.metric_after_verify_pass:
            # The auto path raises OperatorCommandUnexecutable just like a
            # manual run_metric_command would; abort the same graceful way
            # the per-tool handler does (it is a distinct exception, NOT a
            # ToolError, so _auto_metric_feedback does not swallow it).
            try:
                turn.metric_feedback = self._auto_metric_feedback(
                    state.metric_history,
                    iteration=turn.iteration,
                    sha=sha,
                )
            except OperatorCommandUnexecutable as exc:
                return self._unexecutable_abort(
                    exc, iteration=turn.iteration, tool_calls=state.tool_calls
                )
            turn.metric_plateau_finish = self._metric_plateau_summary(state.metric_history)
        return None

    def _report_auto_commit_failure(
        self, exc: GitError | OSError, commit_subject: str, *, iteration: int
    ) -> None:
        """Log + emit a non-benign auto-commit failure with a worktree status
        snapshot, so the event payload tells the operator what was in the tree
        at the failure point. "nothing to commit" variants are benign and stay
        silent: the phrase can arrive in either the stdout or the stderr half
        of the detail string (see git_ops._run); "no changes added" covers the
        variant when only paths outside the worktree (or .gitignore'd) changed;
        "working tree clean" covers a verify pass without any file mutation."""
        msg = str(exc).lower()
        benign = (
            "nothing to commit" in msg or "no changes added" in msg or "working tree clean" in msg
        )
        if benign:
            return
        self._log(f"  auto-commit failed: {exc}")
        # Best-effort: if status itself raises (rare; the outside-a-repo case
        # is already gone by this point in the loop), omit the snapshot.
        worktree_status = ""
        try:
            st = git_status(self.root)
            worktree_status = (
                f"branch={st.branch}"
                f" head={st.head_sha[:12]}"
                f" clean={st.is_clean}"
                f" modified={st.modified_count}"
                f" untracked={st.untracked_count}"
            )
        except (GitError, OSError):
            pass
        self._emit(
            "loop.auto_commit.failed",
            iteration=iteration,
            error=str(exc)[:2000],
            worktree_status=worktree_status,
            commit_subject=commit_subject[:200],
        )

    def _turn_critic_triggers(
        self, state: _LoopState, turn: _TurnState, messages: list[dict[str, Any]]
    ) -> None:
        """Observe-only critic triggers (before_finish, which can revoke a
        finish, lives in the finish gates):

          on_verify_fail - the verify just failed; surface a critique
                           alongside the failure so the worker has a second
                           opinion before its next edit.
          periodic       - every critic_period iterations.
        """
        if (
            self.critic_mode == "on_verify_fail"
            and turn.verify_just_failed
            and self._has_reviewer()
        ):
            critique = self._review_or_critic(
                state=state,
                messages=messages,
                trigger="verify_failed",
                iteration=turn.iteration,
            )
            if critique is not None:
                turn.critic_text = critique.text
        elif (
            self.critic_mode == "periodic"
            and self._has_reviewer()
            and turn.iteration % max(1, self.critic_period) == 0
        ):
            critique = self._review_or_critic(
                state=state,
                messages=messages,
                trigger="periodic",
                iteration=turn.iteration,
            )
            if critique is not None:
                turn.critic_text = critique.text

    def _turn_finish_gates(
        self, state: _LoopState, turn: _TurnState, messages: list[dict[str, Any]]
    ) -> None:
        """Gates that can revoke this turn's finish_run, in precedence order:
        critic (before_finish), metric early-finish, open subtasks, verify
        green, memory backstop. Each clears ``turn.finish_signal`` and appends
        its nudge; later gates then see the finish as already revoked and stay
        quiet."""
        self._gate_before_finish_critic(state, turn, messages)
        self._gate_metric_early_finish(state, turn)
        self._gate_task_finish(state, turn)
        self._gate_verify_green(state, turn)
        self._gate_memory_finish(state, turn)

    def _gate_before_finish_critic(
        self, state: _LoopState, turn: _TurnState, messages: list[dict[str, Any]]
    ) -> None:
        """Gate the agent's finish_run on critic approval. If the critic says
        NEEDS_WORK, suppress the finish (the tool_result still goes back so the
        call isn't half-applied) and inject the critique - the loop carries on
        with the critique visible. After ``max_consecutive_critic_rejections``
        back-to-back rejections the finish goes through (critique still
        injected) so the worker can't bounce indefinitely."""
        if not (
            turn.finish_signal is not None
            and turn.finish_kind == "finish_run"
            and self.critic_mode == "before_finish"
            and self._has_reviewer()
        ):
            return
        critique = self._review_or_critic(
            state=state,
            messages=messages,
            trigger="before_finish",
            iteration=turn.iteration,
        )
        cap = self.max_consecutive_critic_rejections
        cap_reached = cap > 0 and state.consecutive_critic_rejections >= cap
        if critique is not None and not critique.satisfied and not cap_reached:
            self._log(f"  critic rejected finish_run at iter {turn.iteration}")
            self._emit("loop.critic.rejected_finish", iteration=turn.iteration)
            turn.finish_signal = None
            turn.finish_payload = None
            state.consecutive_critic_rejections += 1
            turn.critic_text = (
                "The critic rejected your finish_run call. Address the"
                " issues below before calling finish_run again.\n\n" + critique.text
            )
        elif critique is not None and not critique.satisfied and cap_reached:
            self._log(
                f"  critic rejected finish_run at iter {turn.iteration} but"
                f" rejection cap ({cap}) reached - letting finish through"
            )
            self._emit(
                "loop.critic.rejection_cap_reached",
                iteration=turn.iteration,
                rejections=state.consecutive_critic_rejections,
            )
            turn.critic_text = (
                "The critic flagged issues but the rejection cap was"
                " reached; finish_run will be accepted. Critique:\n\n" + critique.text
            )
            state.consecutive_critic_rejections = 0
        elif critique is not None:
            self._log("  critic approved finish_run")
            state.consecutive_critic_rejections = 0

    def _gate_metric_early_finish(self, state: _LoopState, turn: _TurnState) -> None:
        """Metric-run early-finish guard. On optimisation runs the worker often
        calls finish_run with most of its budget unspent, even though the task
        asks it to keep optimising up to the cap. Mirror the plateau policy:
        while the run still has runway above the final budget slice, reject an
        early finish_run a few times and nudge the worker to keep going; only
        honour it once we are in the final budget slice or patience is
        exhausted. Requires a real budget signal - with none (tests / MCP) we
        defer to the worker's own judgement so a finish can never deadlock."""
        if not (
            turn.finish_signal is not None
            and turn.finish_kind == "finish_run"
            and self.mode == "run"
            and _metric_goal(self.config.workflow.metric) is not None
            and not self._metric_at_ceiling(state.metric_history)
        ):
            return
        finish_budget_remaining = self._budget_fraction_remaining()
        has_runway = (
            finish_budget_remaining is not None
            and finish_budget_remaining > _METRIC_PLATEAU_STOP_BELOW_BUDGET
        )
        if has_runway and state.metric_finish_nudges_used < _METRIC_EARLY_FINISH_PATIENCE:
            assert finish_budget_remaining is not None
            state.metric_finish_nudges_used += 1
            turn.finish_signal = None
            turn.finish_payload = None
            turn.tool_results.append({"type": "text", "text": _METRIC_FINISH_NUDGE})
            self._log(
                f"  metric early-finish rejected #{state.metric_finish_nudges_used}"
                f" at iter {turn.iteration} (budget {finish_budget_remaining:.0%} left)"
            )
            self._emit(
                "loop.metric_early_finish.rejected",
                iteration=turn.iteration,
                nudges_used=state.metric_finish_nudges_used,
                budget_remaining=finish_budget_remaining,
            )

    def _gate_task_finish(self, state: _LoopState, turn: _TurnState) -> None:
        """Task finish-gate: don't let finish_run through while the worker's
        own subtasks are still open (capped; see _task_finish_gate_nudge)."""
        if not (
            turn.finish_signal is not None
            and turn.finish_kind == "finish_run"
            and self.mode == "run"
        ):
            return
        task_nudge = self._task_finish_gate_nudge(state)
        if task_nudge is None:
            return
        turn.finish_signal = None
        turn.finish_payload = None
        turn.tool_results.append({"type": "text", "text": task_nudge})
        self._log(
            f"  finish_run gated: open subtasks remain (nudge"
            f" #{state.task_finish_nudges_used}) at iter {turn.iteration}"
        )
        self._emit(
            "loop.task_finish.gated",
            iteration=turn.iteration,
            nudges_used=state.task_finish_nudges_used,
        )

    def _gate_verify_green(self, state: _LoopState, turn: _TurnState) -> None:
        """Opt-in hard finish gate: refuse finish_run while verify is red or
        stale (bounded, so a genuinely-unpassable task can't pin the loop). The
        honest all_passed=False signal in the stop checks applies whether or
        not this is on; this just gives the worker a few pushes to get green
        first."""
        if not (
            turn.finish_signal is not None
            and turn.finish_kind == "finish_run"
            and self.mode == "run"
            and self._tree_is_verify_green(state) is False
            and self.config.workflow.require_verify_to_finish
            and state.verify_finish_nudges_used < _VERIFY_FINISH_PATIENCE
        ):
            return
        state.verify_finish_nudges_used += 1
        turn.finish_signal = None
        turn.finish_payload = None
        turn.tool_results.append({"type": "text", "text": _VERIFY_FINISH_GATE})
        self._log(
            f"  finish_run gated: verify not green (nudge"
            f" #{state.verify_finish_nudges_used}) at iter {turn.iteration}"
        )
        self._emit(
            "loop.verify_finish.gated",
            iteration=turn.iteration,
            nudges_used=state.verify_finish_nudges_used,
        )

    def _gate_memory_finish(self, state: _LoopState, turn: _TurnState) -> None:
        """Memory write-side backstop: defer the first finish_run ONCE when the
        run recovered from a red verify to green and recorded nothing via
        add_memory - the nudge asks for the root cause or an immediate re-finish
        (see _nudges for the measurement behind it). Explicit finish_run only: a
        went-quiet worker is never bounced here."""
        if not (
            turn.finish_signal is not None
            and turn.finish_kind == "finish_run"
            and self.mode == "run"
            and self.state_dir is not None
            and state.verify_ever_failed
            and state.last_verify_ok is True
            and not state.memory_written
            and not state.memory_finish_nudged
        ):
            return
        state.memory_finish_nudged = True
        turn.finish_signal = None
        turn.finish_payload = None
        turn.tool_results.append({"type": "text", "text": _MEMORY_FINISH_NUDGE})
        self._log(f"  finish_run deferred once: memory backstop at iter {turn.iteration}")
        self._emit("loop.memory_finish.gated", iteration=turn.iteration)

    def _turn_notices(self, state: _LoopState, turn: _TurnState) -> None:
        """Append the turn's advisory texts to the tool_results block: critic
        critique, metric feedback, the memory flip advisory, then the
        degenerate-loop notice.

        The memory flip advisory fires once per run, at the first verify that
        goes green after a red one, while nothing has been recorded via
        add_memory: that is the moment a hard-won root cause is in hand (see
        _nudges for the measurement behind it).

        The loop-guard notice fires when the same (tool, args) signature has
        been called >= 3 times in a row, re-emitted once per "fresh" streak
        (when a new streak crosses the threshold) so spamming the same call
        only triggers once per latch episode. The repeat counter resets on any
        new signature, so a normal re-read after an edit does not trigger."""
        if turn.critic_text:
            turn.tool_results.append({"type": "text", "text": f"[critic]\n{turn.critic_text}"})
        if turn.metric_feedback:
            turn.tool_results.append({"type": "text", "text": turn.metric_feedback})
        if (
            turn.verify_flipped_green
            and self.mode == "run"
            and self.state_dir is not None
            and not state.memory_written
            and not state.memory_flip_nudged
        ):
            state.memory_flip_nudged = True
            turn.tool_results.append({"type": "text", "text": _MEMORY_FLIP_NUDGE})
            self._log("  memory: verify flipped green - injecting add_memory advisory")
            self._emit("loop.memory_flip.nudged", iteration=turn.iteration)
        repeat_threshold = 3
        if (
            state.repeat_streak >= repeat_threshold
            and state.repeat_warning_emitted_at < turn.iteration - 1
        ):
            # Strip the args-JSON suffix for the user-facing text.
            latched_name = (state.last_tool_signature or "").split(":", 1)[0] or "<unknown>"
            notice = (
                f"[loop-guard] You have called `{latched_name}` with"
                f" identical arguments {state.repeat_streak} times in a row."
                " The tool result has not changed. Re-issuing the same"
                " call again will not yield new information. Change"
                " your approach: try different arguments, a different"
                " tool, commit to an edit, or call `finish_run` if"
                " you have already done what the task requires."
            )
            turn.tool_results.append({"type": "text", "text": notice})
            self._emit(
                "loop.loop_guard.triggered",
                iteration=turn.iteration,
                tool=latched_name,
                streak=state.repeat_streak,
            )
            self._log(
                f"  loop-guard: {latched_name} called"
                f" {state.repeat_streak}x in a row - injecting notice"
            )
            state.repeat_warning_emitted_at = turn.iteration

    def _turn_metric_plateau(self, state: _LoopState, turn: _TurnState) -> None:
        """Metric-plateau handling. When a verified metric merely ties the
        prior best, the plateau detector fires. Rather than quit at the first
        stall (often with most of the budget unspent), nudge the worker to
        pivot to a different approach; only stop once we are in the final
        budget slice and have still failed to beat the best after a few pivot
        nudges. With no budget signal (tests / MCP) the fixed
        ``_METRIC_PLATEAU_PATIENCE`` bounds the nudging. Sets
        ``turn.plateau_should_stop``; the stop itself happens in the stop
        checks, after the post-tools snapshot."""
        if turn.metric_plateau_finish is None:
            return
        budget_remaining = self._budget_fraction_remaining()
        in_final_slice = (
            budget_remaining is None or budget_remaining <= _METRIC_PLATEAU_STOP_BELOW_BUDGET
        )
        if self._metric_at_ceiling(state.metric_history):
            # A metric at its provable ceiling (e.g. SCORE: 27/27) cannot
            # improve: stop now rather than nudge the worker to "pivot" toward
            # a number that does not exist. This is the dominant cause of weak
            # reasoning models burning their whole budget (and wall-clock)
            # re-deriving a solved task.
            turn.plateau_should_stop = True
            self._emit("loop.metric_ceiling.stop", iteration=turn.iteration)
        elif in_final_slice and state.plateau_nudges_used >= _METRIC_PLATEAU_PATIENCE:
            turn.plateau_should_stop = True
        else:
            # Count patience only against final-slice nudges. While the run
            # still has runway (in_final_slice False), keep nudging the
            # worker to explore without consuming the budget, exactly as the
            # early-finish guard only counts rejections while it has runway.
            # Counting runway ties here would exhaust _METRIC_PLATEAU_PATIENCE
            # before the final slice, so the run would stop the instant the
            # budget crossed the threshold and the escalating FINAL
            # ("make your one best bet") nudge would never fire.
            if in_final_slice:
                state.plateau_nudges_used += 1
            nudge_text = _metric_plateau_nudge(budget_remaining)
            turn.tool_results.append({"type": "text", "text": nudge_text})
            budget_note = "n/a" if budget_remaining is None else f"{budget_remaining:.0%} left"
            self._log(
                f"  metric_plateau pivot-nudge at iter {turn.iteration} (budget"
                f" {budget_note}; final-slice patience"
                f" {state.plateau_nudges_used}/{_METRIC_PLATEAU_PATIENCE})"
            )
            self._emit(
                "loop.metric_plateau.nudge",
                iteration=turn.iteration,
                nudges_used=state.plateau_nudges_used,
                budget_remaining=budget_remaining,
            )

    def _turn_verify_settled(self, state: _LoopState, turn: _TurnState) -> None:
        """Verify-settled completion bookkeeping (run mode): count no-progress
        iterations after the first green verify; nudge once, then stop (the
        stop happens in the stop checks, via ``turn.verify_settled_stop``).

        "Progress" is any forward motion the prompt encourages, so a
        legitimately-working run is never truncated: an apply_edit/apply_patch,
        a new commit, or an uncommitted worktree change (an edit made via
        run_command). A verify RUN itself (re-verifying between reads is active
        work, not idle) is held neutral so it neither resets nor accrues. Only
        the pathology, spinning on read-only commands with a clean,
        already-committed tree, accrues idle.

        Only governs PLAIN runs. A metric/optimisation run is also mode=="run"
        but its completion is owned by the metric early-finish guard +
        plateau/ceiling logic (which deliberately keep going while budget
        remains); measure/analyse/read iterations there legitimately make no
        commit, so the settled detector must defer to them. (Gating the
        bookkeeping here also keeps the worktree-dirty git check off the
        metric hot path.)"""
        if turn.verify_just_passed:
            state.verify_ever_passed = True
        non_metric_run = self.mode == "run" and _metric_goal(self.config.workflow.metric) is None
        # "Settled" once the run reached a good state: a green verify, or (on a
        # gateless run, where verify never fires) a committed edit.
        settled_seeded = state.verify_ever_passed or state.gateless_ever_committed
        if non_metric_run and settled_seeded:
            made_progress = turn.committed or turn.edited or self._worktree_dirty()
            if made_progress:
                state.verify_settled_idle = 0
                state.verify_settled_nudged = False  # a fresh idle streak may re-nudge
            elif not (turn.verify_just_passed or turn.verify_just_failed):
                state.verify_settled_idle += 1
        turn.verify_settled_stop = (
            non_metric_run
            and turn.finish_signal is None
            and settled_seeded
            and state.verify_settled_idle >= _VERIFY_SETTLED_STOP_AFTER
        )
        if (
            non_metric_run
            and turn.finish_signal is None
            and not turn.verify_settled_stop
            and settled_seeded
            and state.verify_settled_idle >= _VERIFY_SETTLED_NUDGE_AFTER
            and not state.verify_settled_nudged
        ):
            state.verify_settled_nudged = True
            turn.tool_results.append({"type": "text", "text": _VERIFY_SETTLED_NUDGE})
            self._emit(
                "loop.verify_settled.nudge",
                iteration=turn.iteration,
                idle=state.verify_settled_idle,
            )

    def _turn_stop_checks(self, state: _LoopState, turn: _TurnState) -> RunResult | None:
        """Terminal checks, run after the turn's tool_results are in
        ``messages`` and the post-tools snapshot is written, in precedence
        order: verify-settled stop, metric-plateau stop, loop-guard kill, then
        honouring a finish call that survived the gates."""
        if turn.verify_settled_stop:
            self._log(
                f"LOOP: verify_settled at iter {turn.iteration} (idle {state.verify_settled_idle})"
            )
            self._final_checkpoint(turn.iteration)
            self._emit_run_end_passed(reason="verify_settled", iterations=turn.iteration)
            return RunResult(
                completed=True,
                reason="verify_settled",
                summary="verify passed and the worker stopped making changes",
                iterations=turn.iteration,
                tool_calls=state.tool_calls,
            )
        if turn.plateau_should_stop:
            assert turn.metric_plateau_finish is not None
            self._log(f"LOOP: metric_plateau at iter {turn.iteration}")
            self._final_checkpoint(turn.iteration)
            self._emit_run_end_passed(reason="metric_plateau", iterations=turn.iteration)
            return RunResult(
                completed=True,
                reason="metric_plateau",
                summary=turn.metric_plateau_finish,
                iterations=turn.iteration,
                tool_calls=state.tool_calls,
            )
        # loop-guard escalation. The notice in _turn_notices is advisory; if
        # the worker keeps issuing the same call past loop_guard_kill_threshold,
        # terminate the run before it burns the rest of the budget circling.
        # Threshold of 0 disables (notice-only behaviour). The kill happens
        # AFTER the tool_results were appended so the transcript on disk
        # reflects exactly what the model produced up to the kill, which is
        # essential when triaging "why did my run die at iter N".
        if (
            self.loop_guard_kill_threshold > 0
            and state.repeat_streak >= self.loop_guard_kill_threshold
        ):
            latched_name = (state.last_tool_signature or "").split(":", 1)[0] or "<unknown>"
            self._log(
                f"LOOP: loop_guard_killed at iter {turn.iteration} -"
                f" {latched_name} called {state.repeat_streak}x in a row"
                f" (threshold={self.loop_guard_kill_threshold})"
            )
            self._emit(
                "run.end",
                reason="loop_guard_killed",
                iterations=turn.iteration,
                all_passed=False,
                tool=latched_name,
                streak=state.repeat_streak,
            )
            return RunResult(
                completed=False,
                reason="loop_guard_killed",
                summary=(
                    f"loop-guard killed run: `{latched_name}`"
                    f" called {state.repeat_streak}x in a row with"
                    f" identical arguments (threshold"
                    f" {self.loop_guard_kill_threshold})"
                ),
                iterations=turn.iteration,
                tool_calls=state.tool_calls,
            )
        if turn.finish_signal is not None:
            self._log(f"LOOP: {turn.finish_kind} called at iter {turn.iteration}")
            self._final_checkpoint(turn.iteration)
            # Honest finish: finish_planning is always a clean finish, but a
            # finish_run over a red/stale verify is "finished", not "passed"
            # -- all_passed reflects the actual verify state, never just "the
            # model called finish_run".
            if turn.finish_kind == "finish_run" and self._tree_is_verify_green(state) is False:
                self._emit(
                    "run.end",
                    reason=turn.finish_kind,
                    iterations=turn.iteration,
                    all_passed=False,
                )
            else:
                self._emit_run_end_passed(reason=turn.finish_kind, iterations=turn.iteration)
            return RunResult(
                completed=True,
                reason=turn.finish_kind,
                summary=turn.finish_signal,
                iterations=turn.iteration,
                tool_calls=state.tool_calls,
                finish_payload=turn.finish_payload,
            )
        return None

    def _maybe_pre_call_nudges(
        self,
        messages: list[dict[str, Any]],
        state: _LoopState,
        *,
        iteration: int,
        start_iteration: int,
    ) -> None:
        """Before the LLM call, surface the current task for one-task focus, and
        inject a one-shot finish directive when a verbose planner or a non-metric
        run is reading forever without landing a plan / verify+finish before the
        budget dies."""
        # Surface-current-task first, so when a low budget ALSO fires a finish
        # directive below, that finish nudge is the most-recent (strongest)
        # message rather than the focus banner.
        self._maybe_surface_current_task(messages, state)
        # Force a verbose planner to land a plan. Trigger on EITHER a low
        # token budget OR too many planning turns, with prompt caching a
        # planner can take many cheap turns, so an iteration cap is the
        # reliable lever for the "reads forever" failure mode. A rough
        # delivered plan beats an exhaustive one that never gets emitted.
        if self.mode == "plan" and not state.plan_finish_nudged:
            remaining = self._budget_fraction_remaining()
            low_budget = remaining is not None and remaining <= _PLAN_BUDGET_NUDGE_BELOW
            too_many_turns = iteration - start_iteration + 1 >= _PLAN_NUDGE_AFTER_ITERS
            if low_budget or too_many_turns:
                state.plan_finish_nudged = True
                messages.append(
                    {"role": "user", "content": [{"type": "text", "text": _PLAN_BUDGET_NUDGE}]}
                )
                self._log(
                    f"LOOP: plan finish-nudge at iter {iteration}"
                    f" (turns={too_many_turns}, low_budget={low_budget})"
                )
                self._emit(
                    "loop.plan_finish.nudge", iteration=iteration, budget_remaining=remaining
                )

        # Same lever for a non-metric coding run: force a verify + finish
        # before the budget dies (metric runs have their own end-game).
        if (
            self.mode == "run"
            and not state.run_budget_nudged
            and _metric_goal(self.config.workflow.metric) is None
        ):
            remaining = self._budget_fraction_remaining()
            if remaining is not None and remaining <= _RUN_BUDGET_NUDGE_BELOW:
                state.run_budget_nudged = True
                nudge = (
                    _RUN_BUDGET_NUDGE
                    if self.config.workflow.verify_command
                    else _RUN_BUDGET_NUDGE_GATELESS
                )
                messages.append({"role": "user", "content": [{"type": "text", "text": nudge}]})
                self._log(f"LOOP: run budget-nudge at iter {iteration}")
                self._emit("loop.run_budget.nudge", iteration=iteration, budget_remaining=remaining)

    def _maybe_surface_current_task(
        self, messages: list[dict[str, Any]], state: _LoopState
    ) -> None:
        """Surface-current-task: keep the worker on ONE task at a time.

        Compute the current task (the cursor if it still points at an open
        subtask, else the first dependency-satisfied open subtask), advance the
        cursor to it, and inject a focus banner when the focus first appears,
        changes, or was wiped by a tier-2 restart (``surfaced_task_id`` reset to
        None there). Advancing the cursor each turn means that once the worker
        marks the current task passed, the next turn's frontier recompute moves
        focus to the next ready task -- the cursor walks the frontier on its own.

        Also runs the anti-grind counter: when the focus task holds for
        ``_STUCK_ON_TASK_AFTER`` turns with no forward motion, fire one nudge
        offering to split / pass / skip it.

        Run mode only; no curator or no open subtask is a no-op (the finish-gate
        covers the empty-frontier finish). Best-effort throughout: a curator
        hiccup logs and returns rather than breaking the loop.
        """
        if self.mode != "run" or self.graph_client is None:
            return
        try:
            gstate = self.graph_client.get_state()
        except Exception as exc:  # dead socket / IPC error must not break the loop
            self._log(f"LOOP: surface-current-task skipped: {exc}")
            return
        nodes = gstate.get("nodes", {})
        if not isinstance(nodes, dict):
            return
        cursor = gstate.get("cursor")
        current_id = _current_task_id(nodes, cursor)
        if current_id is None:
            state.turns_on_task = 0  # frontier empty: nothing to grind on
            state.last_focus_id = None
            return  # nothing decomposed yet, or the frontier is empty
        if cursor != current_id:
            # Advance the cursor onto the frontier task (auto-advance: a passed
            # cursor task drops out of the frontier, so this moves forward).
            try:
                self.graph_client.set_cursor(SetCursorIntent(id=current_id))
            except Exception as exc:  # cursor advance is advisory; never fatal
                self._log(f"LOOP: cursor advance skipped: {exc}")
        # Anti-grind: count consecutive turns on the same focus task. Any forward
        # motion (cursor advance, a task marked done, or a decompose that moves the
        # cursor to a new subtask) changes current_id and resets the count; survives
        # compaction (last_focus_id is not reset there). Re-fire every
        # _STUCK_ON_TASK_AFTER turns, capped at _STUCK_NUDGE_MAX per task.
        if current_id != state.last_focus_id:
            state.turns_on_task = 0
            state.last_focus_id = current_id
            state.stuck_nudges_fired = 0
        else:
            state.turns_on_task += 1
            if (
                state.turns_on_task % _STUCK_ON_TASK_AFTER == 0
                and state.stuck_nudges_fired < _STUCK_NUDGE_MAX
            ):
                state.stuck_nudges_fired += 1
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _stuck_on_task_nudge(
                                    current_id, nodes[current_id], state.turns_on_task
                                ),
                            }
                        ],
                    }
                )
                self._log(
                    f"LOOP: stuck-on-task nudge #{state.stuck_nudges_fired} for"
                    f" {current_id} after {state.turns_on_task} turns"
                )
                self._emit(
                    "loop.task.stuck_nudge",
                    task_id=current_id,
                    turns=state.turns_on_task,
                    n=state.stuck_nudges_fired,
                )
        if current_id == state.surfaced_task_id:
            return  # already surfaced; the banner survives tier-1 elision
        node = nodes[current_id]
        if node.get("status") == "pending":
            # Reflect that this task is now being worked, keeping the DAG honest
            # for the TUI and the check-off / finish-gate "open" set. Best-effort.
            try:
                self.graph_client.update_status(
                    UpdateStatusIntent(id=current_id, new_status="in_progress")
                )
            except Exception as exc:
                self._log(f"LOOP: mark in_progress skipped: {exc}")
        banner = _current_task_banner(
            current_id, node, decompose=self.config.prompt.decompose == "on"
        )
        messages.append({"role": "user", "content": [{"type": "text", "text": banner}]})
        state.surfaced_task_id = current_id
        self._log(f"LOOP: surfaced current task {current_id}")
        self._emit("loop.task.surfaced", task_id=current_id)
        # The harness-driven cursor/status writes bypass the tool-dispatch path
        # that emits graph.update, so refresh the live view here.
        self._emit_graph_snapshot()

    def _handle_no_tool_use(
        self, resp: Any, messages: list[dict[str, Any]], state: _LoopState, *, iteration: int
    ) -> RunResult | None:
        """Handle a turn with no tool_use. Either a silent finish (the agent
        emitted text; gated like an explicit finish_run) or went-quiet (an
        empty turn; nudged up to a cap). Returns a terminal RunResult, or None
        to continue the loop after appending a nudge.

        Distinguishing the two matters: "agent talked then stopped" is likely
        an implicit finish (the user gets the text as summary), while "agent
        emitted nothing" is a went-quiet failure (an empty provider response,
        or a confused agent) that bench scoring must NOT treat as success."""
        text = resp.text.strip() if resp.text else ""
        if text:
            return self._handle_silent_finish(text, messages, state, iteration=iteration)
        return self._handle_went_quiet(resp, messages, state, iteration=iteration)

    def _handle_silent_finish(
        self, text: str, messages: list[dict[str, Any]], state: _LoopState, *, iteration: int
    ) -> RunResult | None:
        """A no-tool_use turn WITH text: treat it as an implicit finish and run
        it through the same gates as an explicit finish_run. Returns None (with
        a nudge appended to ``messages``) when a gate sends the worker back to
        work; the silent_finish RunResult once every gate lets it through."""
        # Same before_finish critic gate as an explicit finish_run tool_use.
        # Without this, an agent that stops emitting tool calls bypasses
        # critic review entirely. The rejection cap is shared with the
        # tool_use path so a stubborn worker can't bounce the loop forever.
        if (
            self.critic_mode == "before_finish"
            and self._has_reviewer()
            and self._silent_finish_critic_rejects(state, messages, iteration=iteration)
        ):
            return None
        # metric-run early-finish guard, mirroring the finish_run path: a
        # silent finish on an optimisation run with budget to spare should be
        # nudged to keep optimising rather than accepted. Without this,
        # dropping tool_use was a way to skip the plateau/early-finish policy
        # entirely.
        if (
            self.mode == "run"
            and _metric_goal(self.config.workflow.metric) is not None
            and not self._metric_at_ceiling(state.metric_history)
        ):
            finish_budget_remaining = self._budget_fraction_remaining()
            has_runway = (
                finish_budget_remaining is not None
                and finish_budget_remaining > _METRIC_PLATEAU_STOP_BELOW_BUDGET
            )
            if has_runway and state.metric_finish_nudges_used < _METRIC_EARLY_FINISH_PATIENCE:
                assert finish_budget_remaining is not None
                state.metric_finish_nudges_used += 1
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": _METRIC_FINISH_NUDGE}],
                    }
                )
                self._log(
                    f"  metric early-finish (silent) rejected"
                    f" #{state.metric_finish_nudges_used} at iter {iteration}"
                    f" (budget {finish_budget_remaining:.0%} left)"
                )
                self._emit(
                    "loop.metric_early_finish.rejected",
                    iteration=iteration,
                    nudges_used=state.metric_finish_nudges_used,
                    budget_remaining=finish_budget_remaining,
                    trigger="silent_finish",
                )
                return None
        # Task finish-gate (silent path): a worker that stops emitting tool
        # calls with its own subtasks still open is steered back to the list
        # rather than silently finished (shares the cap with the finish_run
        # path). This is the premature-stop case observed live.
        task_nudge = self._task_finish_gate_nudge(state)
        if task_nudge is not None:
            self._log(
                f"  silent_finish gated: open subtasks remain (nudge"
                f" #{state.task_finish_nudges_used}) at iter {iteration}"
            )
            self._emit(
                "loop.task_finish.gated",
                iteration=iteration,
                nudges_used=state.task_finish_nudges_used,
                trigger="silent_finish",
            )
            messages.append({"role": "user", "content": [{"type": "text", "text": task_nudge}]})
            return None
        # Question-nudge (run mode, once): the model ended by asking the
        # operator something in prose without calling ask_user, so the run
        # would silently finish with an unanswered question. Nudge once to
        # call ask_user / finish_run; if it asks again, accept the finish
        # (bounded, so a stubborn model cannot loop the run).
        if self.mode == "run" and not state.question_nudged and _ends_with_question(text):
            state.question_nudged = True
            self._log(f"  silent_finish nudged: ended on a question at iter {iteration}")
            self._emit("loop.question_nudge", iteration=iteration)
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": _QUESTION_NUDGE}]}
            )
            return None
        self._log(f"LOOP: silent_finish at iter {iteration} - agent emitted text but no tool_use")
        self._final_checkpoint(iteration)
        self._emit_run_end_passed(reason="silent_finish", iterations=iteration)
        return RunResult(
            completed=True,
            reason="silent_finish",
            # In ask mode the final prose IS the answer the caller
            # prints, so keep it whole; run/plan only need a short
            # summary line.
            summary=text if self.mode == "ask" else text[:1000],
            iterations=iteration,
            tool_calls=state.tool_calls,
        )

    def _silent_finish_critic_rejects(
        self, state: _LoopState, messages: list[dict[str, Any]], *, iteration: int
    ) -> bool:
        """Run the before_finish critic against a silent finish. True = the
        finish was rejected (critique appended to ``messages``; the loop
        continues). A cap-reached rejection or an approval resets the
        rejection counter and lets the finish proceed."""
        critique = self._review_or_critic(
            state=state,
            messages=messages,
            trigger="before_finish",
            iteration=iteration,
        )
        cap = self.max_consecutive_critic_rejections
        cap_reached = cap > 0 and state.consecutive_critic_rejections >= cap
        if critique is not None and not critique.satisfied and not cap_reached:
            self._log(f"  critic rejected silent_finish at iter {iteration}")
            self._emit(
                "loop.critic.rejected_silent_finish",
                iteration=iteration,
            )
            state.consecutive_critic_rejections += 1
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "[critic]\nThe critic"
                                " rejected your"
                                " silent finish (no"
                                " tool_use, just"
                                " text). Address the"
                                " issues below and"
                                " continue the task.\n\n" + critique.text
                            ),
                        }
                    ],
                }
            )
            return True
        if critique is not None and not critique.satisfied and cap_reached:
            self._log(
                f"  critic rejected silent_finish at"
                f" iter {iteration} but rejection cap"
                f" ({cap}) reached - accepting finish"
            )
            self._emit(
                "loop.critic.rejection_cap_reached",
                iteration=iteration,
                rejections=state.consecutive_critic_rejections,
            )
            state.consecutive_critic_rejections = 0
        elif critique is not None:
            self._log("  critic approved silent_finish")
            state.consecutive_critic_rejections = 0
        return False

    def _handle_went_quiet(
        self, resp: Any, messages: list[dict[str, Any]], state: _LoopState, *, iteration: int
    ) -> RunResult | None:
        """A fully-empty turn (no text, no tool_use): surface reasoning
        starvation explicitly, then nudge-and-retry up to the per-streak cap
        before ending the run as went_quiet.

        The nudge is cheap (~50 input tokens vs aborting the entire run) and
        almost always gets a weak open-weights model back on track. The empty
        assistant turn is dropped from ``messages`` first: Anthropic rejects an
        assistant message with empty content, a THINKING-ONLY turn (reasoning
        starvation: blocks but no text/tool_use) translates to one with no
        content and no tool_calls that strict OpenAI-compatible backends reject
        with a non-retryable 400, and either way it is dead context.
        AGENT6_WENT_QUIET_MAX_NUDGES overrides the cap."""
        # reasoning-starvation trip-wire. When a model spends its entire output
        # budget on reasoning_content and emits nothing user-visible, the
        # provider returns stop_reason="length" with empty text + no tool_uses.
        # Without this breadcrumb the failure mode is indistinguishable from a
        # model that genuinely gave up, and the only way to diagnose it is to
        # read raw transcripts (took ~7 minutes per case forensics). Surface it
        # explicitly so the next undetected reasoning model is one log line
        # away.
        reasoning_chars = 0
        raw_content = (resp.raw or {}).get("content") or []
        if isinstance(raw_content, list):
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    reasoning_chars = len(str(block.get("thinking") or ""))
                    break
        starved = resp.stop_reason == "length" and reasoning_chars > 0 and resp.output_tokens > 0
        if starved:
            self._log(
                f"LOOP: reasoning_starvation at iter {iteration}"
                f" - stop_reason=length, reasoning_chars={reasoning_chars},"
                f" output_tokens={resp.output_tokens}; the model spent"
                f" its entire output budget on reasoning_content."
                f" Add this model to _REASONING_MODEL_HINTS in"
                f" providers/openai.py if it isn't already."
            )
            self._emit(
                "loop.reasoning_starvation",
                iteration=iteration,
                reasoning_chars=reasoning_chars,
                output_tokens=resp.output_tokens,
                stop_reason=resp.stop_reason,
            )
        self._log(f"LOOP: went_quiet at iter {iteration} - agent emitted no text and no tool_use")
        env_max = os.environ.get("AGENT6_WENT_QUIET_MAX_NUDGES", "").strip()
        effective_max_nudges = int(env_max) if env_max.isdigit() else self.went_quiet_max_nudges
        if state.went_quiet_nudges_used < effective_max_nudges:
            state.went_quiet_nudges_used += 1
            if messages and messages[-1].get("role") == "assistant":
                last_content = messages[-1].get("content") or []
                substantive = isinstance(last_content, list) and any(
                    isinstance(b, dict)
                    and (
                        (b.get("type") == "text" and str(b.get("text", "")).strip())
                        or b.get("type") == "tool_use"
                    )
                    for b in last_content
                )
                if not substantive:
                    messages.pop()
            # starvation-specific nudge. When the previous turn ended with
            # stop_reason=length AND reasoning_content ate the entire budget,
            # the generic "your turn was empty" message gives the model no
            # actionable feedback and it repeats the same reasoning loop next
            # turn. Tell it explicitly to stop thinking and commit to a tool
            # call.
            if starved:
                nudge_text = (
                    "[harness] Your previous turn spent its entire"
                    f" output budget ({resp.output_tokens} tokens) on"
                    " reasoning_content with no visible content and"
                    " no tool_use. STOP REASONING. On this next turn,"
                    " emit a tool_use IMMEDIATELY — do not think"
                    " further. If you genuinely don't know what to do"
                    " next, call `read_file` on the most relevant"
                    " source file to ground your next decision, or"
                    " call `finish_run` if the task is complete. Any"
                    " response that is not a tool_use will waste the"
                    " entire run."
                )
            else:
                nudge_text = (
                    "[harness] Your previous turn was empty: no text"
                    " content and no tool_use. This is a synthetic"
                    " prompt from the agent6 harness. Either call a"
                    " tool to make progress, or call `finish_run`"
                    " with a summary if the task is complete. Do"
                    " not reply with another empty turn."
                )
            messages.append({"role": "user", "content": [{"type": "text", "text": nudge_text}]})
            self._emit(
                "loop.went_quiet.nudge",
                iteration=iteration,
                nudges_used=state.went_quiet_nudges_used,
                nudges_max=effective_max_nudges,
            )
            return None
        self._emit(
            "run.end",
            reason="went_quiet",
            iterations=iteration,
            all_passed=False,
        )
        return RunResult(
            completed=False,
            reason="went_quiet",
            summary="(agent emitted no text and no tool_use)",
            iterations=iteration,
            tool_calls=state.tool_calls,
        )

    def _seed_root_task(self, user_task: str) -> str | None:
        """Create the run's root task in the DAG when the curator
        is wired. Returns the new node id, or None if no graph_client.

        The root is the user's task itself. Subsequent agent ``add_task``
        calls with ``parent_id=None`` attach under this root."""
        if self.graph_client is None:
            return None
        # audit: TaskNodeDraft.title has min_length=1. The previous
        # `user_task.splitlines()[0]` crashed when user_task started with `\n`
        # (the first line was the empty string before the newline). Pick the
        # first NON-EMPTY line; fall back to "(run)" if the whole task is
        # blank.
        first_nonempty = next(
            (line.strip() for line in user_task.splitlines() if line.strip()),
            "",
        )
        title = first_nonempty[:200] if first_nonempty else "(run)"
        try:
            draft = TaskNodeDraft(
                title=title,
                rationale="single-loop run; root task seeded by Workflow",
                acceptance="",
                relevant_paths=(),
                created_by="user",
            )
            node = self.graph_client.add_subtask(AddSubtaskIntent(parent_id=None, draft=draft))
            return node.id
        except CuratorClientError as exc:
            self._log(f"LOOP: failed to seed root task: {exc}")
            return None

    def _worktree_dirty(self) -> bool:
        """True if the repo has uncommitted changes, e.g. an edit a worker made
        via run_command that the verify-pass auto-commit hasn't captured yet. The
        verify-settled detector treats that as in-progress work. Best-effort:
        any git error reports clean, so a hiccup can't wedge the detector."""
        try:
            return not git_status(self.root).is_clean
        except (GitError, OSError):
            return False

    def _final_checkpoint(self, iteration: int) -> None:
        """Best-effort commit of any dirty worktree on a successful exit so
        run_command-authored edits on a gated run aren't lost from git history.

        On a gated run (verify_command set) the in-loop auto-commit only fires
        on a green verify; an edit made via run_command after a prior green
        verify, never re-verified, is left only in the working tree and is
        silently lost when the run ends (score.sh, resume, and the diff viewer
        all read git history). Capturing it here closes that gap."""
        if self.mode != "run" or not self._worktree_dirty():
            return
        try:
            sha = commit_all(self.root, f"checkpoint (iter {iteration})")
            if sha:
                self._log(f"  final checkpoint: {sha[:12]}")
                self._emit("loop.auto_commit", iteration=iteration, sha=sha)
        except (GitError, OSError) as exc:
            msg = str(exc).lower()
            benign = (
                "nothing to commit" in msg
                or "no changes added" in msg
                or "working tree clean" in msg
            )
            if not benign:
                self._log(f"  final checkpoint commit failed: {exc}")

    def _pass_pending_root_tasks(self) -> None:
        """On successful completion, mark still-pending root task(s) as passed.

        The loop seeds one root task per ``run()`` (each ask REPL follow-up seeds
        another), but the worker finishes via ``finish_run`` without ever
        touching it -- so a completed ask/run otherwise reads ``tasks 0/1``. Pass
        any root (``parent_id is None``) still pending/in-progress so the DAG --
        and every viewer + resume -- agrees the run completed. Subtasks the
        worker deliberately left unfinished are untouched (kept honest).
        Best-effort: a curator hiccup must never break completion."""
        if self.graph_client is None:
            return
        try:
            state = self.graph_client.get_state()
        except Exception as exc:  # dead socket / IPC error must not break finish
            self._log(f"LOOP: auto-pass root skipped: {exc}")
            return
        nodes = state.get("nodes", {})
        if not isinstance(nodes, dict):
            return
        changed = False
        for nid, node in nodes.items():
            if node.get("parent_id") is None and node.get("status") in ("pending", "in_progress"):
                try:
                    self.graph_client.update_status(UpdateStatusIntent(id=nid, new_status="passed"))
                    changed = True
                except Exception as exc:  # IPC/validation glitch must not break finish
                    self._log(f"LOOP: auto-pass root {nid} failed: {exc}")
                    break  # a dead socket fails for every remaining node too
        if changed:
            self._emit_graph_snapshot()

    def _emit_run_end_passed(self, *, reason: str, iterations: int) -> None:
        """Emit a successful ``run.end``, first auto-passing any still-pending
        root task so the DAG (and every viewer + resume) agrees the run
        completed -- otherwise a finish_run-only ask/run reads ``tasks 0/1``."""
        self._pass_pending_root_tasks()
        self._emit("run.end", reason=reason, iterations=iterations, all_passed=True)

    def _tree_is_verify_green(self, state: _LoopState) -> bool | None:
        """Is the current tree in a verified-green state? None when no verify
        command is configured (nothing to gate on); else True iff the last verify
        was green AND nothing has been edited since. Grounds both the honest
        finish signal and the opt-in hard finish gate, so 'passed' can never mean
        'finished over a red or stale verify'."""
        if not self.config.workflow.verify_command:
            return None
        return state.last_verify_ok is True and not state.edited_since_verify

    def _emit_graph_snapshot(self) -> None:
        """Emit the current task DAG so a live viewer (the TUI) can render it.
        The worker's add_task/update_task tree lives in the curator, not the
        event log, so we snapshot it (once per turn, see the call site).

        Project to ONLY the fields the viewer renders, a full node dump carries
        unbounded model-authored text (rationale/acceptance/notes/paths) that
        bloats the fsync'd event log for no benefit. Best-effort: a curator
        hiccup must never break the run."""
        if self.graph_client is None:
            return
        try:
            state = self.graph_client.get_state()
        except Exception as exc:
            # (CuratorClientError, IpcError, OSError/BrokenPipeError on a dead
            # socket) must never break an otherwise-healthy run. Matches the
            # convention at tools/dispatch.py's get_state() call site.
            self._log(f"LOOP: graph snapshot skipped: {exc}")
            return
        raw = state.get("nodes", {})
        if not isinstance(raw, dict):
            return
        nodes = {
            nid: {
                "title": n.get("title", ""),
                "status": n.get("status", "pending"),
                "parent_id": n.get("parent_id"),
                "children": n.get("children", ()),
            }
            for nid, n in raw.items()
            if isinstance(n, dict)
        }
        self._emit("graph.update", nodes=nodes, cursor=state.get("cursor"))

    def _load_repo_summary(self) -> RepoSummary:
        """Reuse the shared `load_repo_summary` and extend with structural priors
        (co-change, hot symbols) - structural priors
        delivered directly into the loop's system prompt.

        Hot-symbols / co-change calls are best-effort: a missing git history
        or a tree-sitter parser hiccup shouldn't block the run. -era
        audit: re-raise BudgetExceeded and KeyboardInterrupt so the loop's
        budget guarantee and operator-abort path stay intact.
        """
        # prompt.structural_priors=false -> base summary only (no hot symbols /
        # co-change / symbol outline), a leaner prompt that leans on on-demand tools.
        disp = self.dispatcher if self.config.prompt.structural_priors else None
        return load_repo_summary(self.root, dispatcher=disp)

    def _load_memories(self) -> tuple[MemoryEntry, ...]:
        """Active cross-run memories for the system prompt.

        () when no state_dir is wired, and for machine/agent modes (their
        prompt assembly drops repo context, memories included). An unreadable
        store logs loudly and returns () rather than aborting the run,
        mirroring the snapshot-write policy: memory is context, not
        correctness.
        """
        if self.state_dir is None or self.mode in ("machine", "agent"):
            return ()
        try:
            entries = memory_list_entries(self.state_dir)
        except (MemoryStoreError, OSError) as exc:
            self._log(f"LOOP: WARNING: cross-run memories unavailable: {exc}")
            return ()
        active = tuple(e for e in entries if e.is_active)
        if active:
            self._log(f"LOOP: memories: {len(active)} active")
        return active

    def _save_resume_snapshot(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tool_calls: int,
        next_iteration: int,
        root_task_id: str | None,
        state: _LoopState,
    ) -> None:
        """Write loop state to disk for resume.

        Called before each LLM call and again at the end of each iteration
        (after the executed tool_results are appended) so a crash after a
        non-idempotent tool dispatch resumes from AFTER the executed tools
        rather than replaying them. Atomic via tmp-file + replace so a crash
        mid-write leaves the prior snapshot intact. No-op if
        ``resume_state_path`` is None (e.g. unit tests).
        """
        if self.resume_state_path is None:
            return
        goal = _metric_goal(self.config.workflow.metric)
        best = _best_metric_sample(state.metric_history, goal=goal) if goal is not None else None
        payload = {
            "version": _SNAPSHOT_VERSION,
            "system": system,
            "messages": messages,
            "tool_calls": tool_calls,
            "next_iteration": next_iteration,
            "root_task_id": root_task_id,
            "review_rejections_total": state.review_rejections_total,
            # So resume reuses the exact verify resolution (gated argv or [] for
            # gateless) instead of re-inferring and possibly diverging from the
            # frozen `system` prompt's verify/no-verify block.
            "verify_command": list(self.config.workflow.verify_command),
            # Completion-relevant scalars: without these the metric / verify-
            # settled stop logic restarts from zero on resume (re-rejecting a
            # correct finish_run, re-counting idle from scratch). Compact metric
            # summary only -- enough to seed `_metric_at_ceiling` + the plateau.
            "verify_ever_passed": state.verify_ever_passed,
            "gateless_ever_committed": state.gateless_ever_committed,
            "metric_best_score": best.score if best is not None else None,
            "metric_at_ceiling": self._metric_at_ceiling(state.metric_history),
            # Fork checkpoint extras: the workspace HEAD + curator DAG version at
            # this turn, so `agent6 fork --at-turn N` can cut the new run's branch
            # at the right sha and clone the DAG as of that version. Plain resume
            # reads head_sha (its divergence guard) but ignores graph_version.
            # Best-effort: a checkpoint is recovery state, so an
            # unreadable git/curator degrades to "" / 0 rather than crashing.
            "head_sha": self._checkpoint_head_sha(),
            "graph_version": self._checkpoint_graph_version(),
        }
        blob = json.dumps(payload)
        # The snapshot is recovery state, not run output: an unwritable state dir
        # (full disk, quota, read-only mount) disables resume/fork but must not
        # abort an otherwise-healthy run whose edits + commits are already on disk
        # independently. Warn once, then continue.
        try:
            # Write the append-only checkpoint first, then advance loop_state.json
            # as the latest pointer. If the second write fails, default fork still
            # follows loop_state.json, while explicit --at-turn can use the durable
            # checkpoint.
            cp_dir = self.resume_state_path.parent / "checkpoints"
            atomic_write(cp_dir / f"{next_iteration:04d}.json", blob)
            atomic_write(self.resume_state_path, blob)
        except OSError as exc:
            if not self._snapshot_write_failed:
                self._snapshot_write_failed = True
                self._log(
                    f"LOOP: WARNING could not persist resume snapshot ({exc}); "
                    "resume/fork are unavailable for this run, continuing anyway"
                )

    def _checkpoint_head_sha(self) -> str:
        """Workspace HEAD for the per-turn checkpoint; "" if it can't be read.

        A checkpoint is best-effort recovery state -- a missing sha must not
        crash the snapshot. fork degrades gracefully when it is empty."""
        try:
            return git_status(self.root).head_sha
        except (GitError, OSError):
            return ""

    def _checkpoint_graph_version(self) -> int:
        """Curator DAG version for the per-turn checkpoint; 0 if no curator."""
        if self.graph_client is None:
            return 0
        try:
            gv = self.graph_client.get_state().get("graph_version", 0)
        except (CuratorClientError, OSError):
            return 0
        return gv if isinstance(gv, int) else 0

    def _record_metric_result(
        self,
        history: list[_MetricSample],
        result: dict[str, Any],
        *,
        iteration: int,
        label: str,
        sha: str,
    ) -> str | None:
        metric_cfg = self.config.workflow.metric
        goal = _metric_goal(metric_cfg)
        if goal is None:
            return None
        assert metric_cfg is not None  # goal is None otherwise
        score = _coerce_metric_score(result.get("score"))
        raw_returncode = result.get("returncode")
        returncode = raw_returncode if isinstance(raw_returncode, int) else None
        stdout = str(result.get("stdout", ""))
        stderr = str(result.get("stderr", ""))
        combined = f"{stdout}\n{stderr}"
        targets = _extract_metric_targets(combined, goal=goal)
        at_ceiling = (
            goal == "maximize"
            and score is not None
            # Only count an X/Y ceiling reported on the score-match line, so an
            # incidental "100/100" progress bar elsewhere cannot latch it.
            and _metric_at_fraction_ceiling(combined, score, pattern=metric_cfg.pattern)
        )
        sample = _MetricSample(
            label=label,
            score=score,
            returncode=returncode,
            sha=sha,
            stdout_tail=stdout[-500:],
            stderr_tail=stderr[-500:],
            targets=targets,
            at_ceiling=at_ceiling,
        )
        history.append(sample)
        self._emit(
            "loop.metric.sample",
            iteration=iteration,
            label=label,
            score=score,
            returncode=returncode,
            sha=sha[:12],
        )
        return _format_metric_feedback(history, goal=goal)

    def _auto_metric_feedback(
        self,
        history: list[_MetricSample],
        *,
        iteration: int,
        sha: str,
    ) -> str | None:
        metric_cfg = self.config.workflow.metric
        goal = _metric_goal(metric_cfg)
        if self.mode != "run" or goal is None:
            return None
        self._log(f"LOOP: auto metric after verify-pass at iter {iteration}")
        self._emit("loop.metric.auto_call", iteration=iteration, sha=sha[:12])
        try:
            result = self.dispatcher.dispatch("run_metric_command", {})
        except ToolError as exc:
            sample = _MetricSample(
                label=f"auto iter {iteration}",
                score=None,
                returncode=None,
                sha=sha,
                error=str(exc),
            )
            history.append(sample)
            self._emit(
                "loop.metric.auto_failed",
                iteration=iteration,
                error=str(exc)[:200],
            )
            return _format_metric_feedback(history, goal=goal)
        return self._record_metric_result(
            history,
            result,
            iteration=iteration,
            label=f"auto iter {iteration}",
            sha=sha,
        )

    def _metric_plateau_summary(self, history: list[_MetricSample]) -> str | None:
        metric_cfg = self.config.workflow.metric
        goal = _metric_goal(metric_cfg)
        if self.mode != "run" or goal is None:
            return None
        return _metric_plateau_summary(history, goal=goal)

    def _metric_at_ceiling(self, history: list[_MetricSample]) -> bool:
        """True once any verified sample reached the metric's provable
        ceiling (e.g. ``SCORE: 27/27``). Such a metric cannot be improved, so
        the loop honours an early ``finish_run`` and stops nudging instead of
        spending the rest of the budget chasing an unbeatable number."""
        return any(sample.at_ceiling for sample in history)

    def _budget_fraction_remaining(self) -> float | None:
        """Fraction of the token budget still available, or None when no
        BudgetTracker is wired in (tests / MCP path)."""
        if self.budget is None:
            return None
        return self.budget.fraction_remaining()

    def _unexecutable_abort(
        self, exc: OperatorCommandUnexecutable, *, iteration: int, tool_calls: int
    ) -> RunResult:
        """Graceful abort when an operator verify/metric command cannot run in
        the jail (e.g. its binary is not on the jail PATH). The model cannot fix
        operator config, so stop loudly rather than flail against a gate that
        never executes or silently report success. Shared by the manual per-tool
        path and the auto-metric-after-verify path so the same misconfiguration
        ends the same way regardless of who triggered the command."""
        self._log(f"LOOP: aborting -- {exc}")
        self._emit(
            "run.end",
            reason="verify_command_unexecutable",
            iterations=iteration,
            all_passed=False,
        )
        return RunResult(
            completed=False,
            reason="verify_command_unexecutable",
            summary=str(exc),
            iterations=iteration,
            tool_calls=tool_calls,
        )

    def _worker_max_tokens(self, state: _LoopState) -> int:
        """Per-call output cap for the worker turn.

        Metric-optimization runs (mode "run" with a configured continuous
        metric) lift the ceiling to ``metric_task_max_tokens`` so a single turn
        can rewrite a hot function wholesale without truncating mid-apply_patch.
        Every other run keeps ``per_call_max_tokens``.

        Starvation backoff: once the worker has gone quiet (no text + no
        tool_use -- typically a reasoning model that spent its whole output
        budget on reasoning_content) on >= 2 CONSECUTIVE turns, drop back to
        ``per_call_max_tokens`` even on a metric run. A spiraling over-reasoner
        (observed: GLM 5.2) otherwise burns a fresh ~65k-token reasoning binge
        every nudged turn until it exhausts ``went_quiet_max_nudges`` and the run
        dies with zero progress. A tight cap plus the forceful "emit a tool_use
        now" nudge pressures it to ACT; ``went_quiet_nudges_used`` resets to 0 on
        the first productive turn, so the very next turn gets the full ceiling
        back for the real edit (the recovery edit itself is never truncated).
        The 2-quiet threshold spares the model the high ceiling was raised FOR
        (Kimi K2.x finishes its reasoning within 65k and rarely goes quiet, let
        alone twice in a row), so the backoff targets the spiral, not the model.
        """
        metric_run = self.mode == "run" and _metric_goal(self.config.workflow.metric) is not None
        if metric_run and state.went_quiet_nudges_used < _STARVATION_BACKOFF_AFTER_QUIETS:
            return max(self.per_call_max_tokens, self.metric_task_max_tokens)
        return self.per_call_max_tokens

    def _maybe_compact(self, messages: list[dict[str, Any]]) -> bool:
        """Tiered compaction. Returns True iff a tier-2 summarise-and-restart
        actually replaced the history (so the caller can re-surface the
        current-task banner the restart wiped); False otherwise.

        Tier 1 (cheap): drop old tool_result blocks once cumulative content
        exceeds ``compact_drop_at_chars``.

        Tier 2 (expensive): once the WHOLE post-elision context (text +
        tool_use inputs + surviving tool_results, via ``_context_chars``)
        crosses ``compact_summarise_at_chars``, summarise the elided history
        into a compact progress block and restart the message list from
        (original task + summary). Measuring only tool_results here -- which
        tier 1 just capped -- left tier 2 unreachable. Fail-safe: if
        summarisation errors or returns nothing, the message list is left
        untouched (tier-1 elision already ran) and the run continues.
        """
        stats = _compact_old_tool_results(
            messages,
            max_total_bytes=self.compact_drop_at_chars,
            keep_recent=2,
            protect_paths=_recently_edited_paths(messages),
            gister=self._distill_gists if self.compact_elision_gists else None,
        )
        if stats.elided:
            detail = f", {stats.gisted} kept as distilled gists" if stats.gisted else ""
            self._log(f"LOOP: compaction elided {stats.elided} old tool_result blocks{detail}")
            self._emit("loop.compact.dropped", n=stats.elided)
        if stats.demoted:
            self._log(f"LOOP: compaction demoted {stats.demoted} gists to bare placeholders")
        if stats.gisted or stats.demoted:
            self._emit("loop.compact.gists", gisted=stats.gisted, demoted=stats.demoted)
        # Tier 2 must measure something tier 1 does NOT already bound. Tier 1
        # just capped tool_result bytes to ``compact_drop_at_chars``, so
        # re-measuring only tool_results here could never exceed the (larger)
        # tier-2 threshold -- tier 2 was unreachable. Measure the WHOLE post-
        # elision context (text + tool_use inputs + surviving tool_results),
        # which keeps growing across a long run from assistant prose and
        # tool-call args even after old tool_results are dropped.
        total = _context_chars(messages)
        # Tier 2 needs at least an original-task message plus enough history
        # to be worth summarising; below that a restart would lose more than
        # it saves.
        if total > self.compact_summarise_at_chars and len(messages) > 3:
            return self._summarise_and_restart(messages)
        return False

    def _distill_gists(self, requests: tuple[_GistRequest, ...]) -> dict[str, str]:
        """Distill about-to-be-elided file reads into one-line gists with the
        summariser model (same seat as tier-2). Fail-safe: any provider error
        returns {} and every victim gets the bare placeholder, so gisting can
        slow a drop event but never break one."""
        provider = self.summariser_provider or self.provider
        files = "\n\n".join(f"=== FILE {r.path} ===\n{r.content}" for r in requests)
        self._emit("loop.compact.gist.call", files=len(requests))
        try:
            resp = provider.call(
                system=_GIST_DISTILL_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": files}],
                tools=[],
                max_tokens=self.context_summary_max_tokens,
                temperature=0.0,
            )
        except (ProviderError, BudgetExceeded) as exc:
            self._log(f"  gist distillation failed: {exc}; eliding without gists")
            self._emit("loop.compact.gist.failed", error=str(exc)[:200])
            return {}
        return _parse_gist_lines(resp.text or "", paths=[r.path for r in requests])

    def _summarise_and_restart(self, messages: list[dict[str, Any]]) -> bool:
        """Replace the message history with (original task + a model-written
        progress summary). Mutates ``messages`` in place. The loop only calls
        this at the top of an iteration, where the history is balanced (every
        ``tool_use`` already has its ``tool_result``), so the restart can drop
        the middle without orphaning a tool-call pairing. Returns True iff the
        history was actually replaced; False on every fail-safe path (the
        tier-1-elided context is kept and the run continues).
        """
        provider = self.summariser_provider or self.provider
        original = messages[0]
        transcript = _format_messages_tail_for_critic(
            messages[1:], max_messages=len(messages), max_chars=60_000
        )
        # The DAG is agent6's compaction memory: at each restart we ask the
        # summariser to check off finished tasks and surface newly-found ones, so
        # task state stays accurate across compaction without depending on the
        # worker calling update_task (which weak models rarely do -- observed live).
        open_tasks = self._open_tasks_for_checkoff()
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
        user_msg = (
            "Summarise the following agent transcript for a context restart."
            f"{checkoff_req}\n\nTRANSCRIPT (oldest first):\n{transcript}"
        )
        self._log(f"LOOP: tier-2 compaction summarise-and-restart ({len(messages)} msgs)")
        self._emit("loop.compact.summarise.call", messages=len(messages))
        try:
            resp = provider.call(
                system=_CONTEXT_SUMMARY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                tools=[],
                max_tokens=self.context_summary_max_tokens,
                temperature=0.0,
            )
        except (ProviderError, BudgetExceeded) as exc:
            # Fail-safe: keep the current (tier-1-elided) context. A real
            # budget exhaustion is re-detected by the next provider call.
            self._log(f"  tier-2 summarise failed: {exc}; keeping current context")
            self._emit("loop.compact.summarise.failed", error=str(exc)[:200])
            return False
        raw = (resp.text or "").strip()
        if not raw:
            self._emit("loop.compact.summarise.failed", error="empty summary")
            return False
        # Apply the check-off to the curator (best-effort) and strip the block
        # from the summary so the restarted worker sees narrative, not bookkeeping.
        if open_tasks:
            self._apply_compaction_checkoff(raw, valid_ids={tid for tid, _ in open_tasks})
        summary = _strip_checkoff(raw) if open_tasks else raw
        restart = {
            "role": "user",
            "content": [{"type": "text", "text": _context_restart_notice(self.mode) + summary}],
        }
        messages[:] = [original, restart]
        self._emit("loop.compact.summarise.done", summary_chars=len(summary))
        return True

    def _open_tasks_for_checkoff(self) -> list[tuple[str, str]]:
        """(id, title) of every pending/in_progress task in the DAG, for the
        tier-2 compaction check-off. Best-effort: no curator or an IPC hiccup
        yields an empty list, so compaction degrades to the plain summary."""
        if self.graph_client is None:
            return []
        try:
            state = self.graph_client.get_state()
        except Exception as exc:  # dead socket / IPC error must not break compaction
            self._log(f"LOOP: checkoff task list skipped: {exc}")
            return []
        nodes = state.get("nodes", {})
        if not isinstance(nodes, dict):
            return []
        out: list[tuple[str, str]] = []
        for nid, node in nodes.items():
            # Subtasks only: never offer the auto-root (parent_id is None) for
            # check-off, mirroring the finish-gate and surface rules. The root is
            # the whole-run container, so a mid-run summary must not mark it
            # passed and end the run early.
            if node.get("parent_id") is None:
                continue
            if node.get("status") in ("pending", "in_progress"):
                out.append((nid, str(node.get("title", ""))[:120]))
        return out

    def _apply_compaction_checkoff(self, summary_text: str, *, valid_ids: set[str]) -> None:
        """Parse the summariser's ```checkoff block and apply it to the curator:
        mark completed tasks passed, queue newly-discovered ones as children of
        the first root. Best-effort: a curator hiccup must never break the run."""
        if self.graph_client is None:
            return
        completed, new_tasks = _parse_checkoff(summary_text)
        completed = [cid for cid in completed if cid in valid_ids]  # ignore hallucinated ids
        if not completed and not new_tasks:
            return
        changed = False
        try:
            for cid in completed:
                self.graph_client.update_status(
                    UpdateStatusIntent(id=cid, new_status="passed", note="compaction check-off")
                )
                changed = True
            if new_tasks:
                root_id = self._first_root_id()
                for title in new_tasks[:8]:  # cap: a runaway summary can't flood the DAG
                    self.graph_client.add_subtask(
                        AddSubtaskIntent(
                            parent_id=root_id,
                            draft=TaskNodeDraft(title=title, created_by="planner"),
                        )
                    )
                    changed = True
        except Exception as exc:  # IPC/validation glitch must not break the run
            self._log(f"LOOP: compaction check-off partial ({exc})")
        if changed:
            self._log(
                f"LOOP: compaction check-off -- passed {len(completed)}, queued {len(new_tasks)}"
            )
            self._emit_graph_snapshot()

    def _first_root_id(self) -> str | None:
        """The first root task id (parent_id is None), or None. Best-effort."""
        if self.graph_client is None:
            return None
        try:
            state = self.graph_client.get_state()
        except Exception:
            return None
        nodes = state.get("nodes", {})
        if isinstance(nodes, dict):
            for nid, node in nodes.items():
                if node.get("parent_id") is None:
                    return nid
        return None

    def _task_finish_gate_nudge(self, state: _LoopState) -> str | None:
        """If the worker created subtasks and any are still open, return a nudge
        message to re-prompt with instead of finishing; else None (finish OK).

        Only SUBTASKS (parent_id is not None) gate -- the auto-root is pending
        until the run ends, so gating on it would deadlock. Capped by
        ``_TASK_FINISH_PATIENCE``: after that many blocked finishes the finish is
        honoured (a task the worker can't close, and won't mark obsolete/skipped,
        must not bounce the loop forever). Best-effort: no curator -> no gate."""
        if self.graph_client is None:
            return None
        try:
            nodes = self.graph_client.get_state().get("nodes", {})
        except Exception as exc:  # dead socket / IPC error must not block finishing
            self._log(f"LOOP: task finish-gate skipped: {exc}")
            return None
        if not isinstance(nodes, dict):
            return None
        open_subtasks = [
            (nid, str(node.get("title", ""))[:120])
            for nid, node in nodes.items()
            if node.get("parent_id") is not None
            and node.get("status") in ("pending", "in_progress")
        ]
        if not open_subtasks:
            return None
        if state.task_finish_nudges_used >= _TASK_FINISH_PATIENCE:
            return None  # cap reached: stop bouncing, honour the finish
        state.task_finish_nudges_used += 1
        listing = "\n".join(f"- {tid}: {title}" for tid, title in open_subtasks)
        return (
            "[harness] You still have open tasks; finish the work before stopping. "
            f"{len(open_subtasks)} task(s) are pending/in_progress:\n{listing}\n"
            "Continue with the next one. If a task is genuinely not needed or you"
            " cannot do it, call update_task to mark it skipped or obsolete -- do"
            " not just abandon it. Then finish_run once the list is clear."
        )

    def _maybe_revise_prompt(self, user_task: str, repo: RepoSummary) -> str:
        if self.revise_prompt == "off":
            return user_task
        if self.prompt_reviser_provider is None:
            raise _PromptRevisionError(
                "prompt.revise_prompt is enabled but no reviser provider is wired"
            )

        context = _format_prompt_revision_context(repo)
        user_msg = (
            f"RAW_TASK:\n{user_task}\n\nREPO_CONTEXT:\n{context}\n\nRewrite the raw task now."
        )
        self._log(f"LOOP: prompt revision ({self.revise_prompt})")
        self._emit("loop.prompt_revision.call", mode=self.revise_prompt)
        try:
            resp = self.prompt_reviser_provider.call(
                system=_PROMPT_REVISION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                tools=[],
                max_tokens=self.prompt_revision_max_tokens,
                temperature=self.prompt_reviser_temperature,
            )
        except (ProviderError, BudgetExceeded) as exc:
            self._emit("loop.prompt_revision.failed", error=str(exc)[:200])
            raise _PromptRevisionError(str(exc)) from exc

        revision = _parse_prompt_revision(resp.text or "")
        if not revision.revised_task:
            self._emit("loop.prompt_revision.failed", error="empty revised task")
            raise _PromptRevisionError("reviser returned an empty task")

        self._emit(
            "loop.prompt_revision.result",
            raw_chars=len(user_task),
            revised_chars=len(revision.revised_task),
            questions=len(revision.clarifying_questions),
        )
        self._log(
            "PROMPT REVISION\n"
            "--- original ---\n"
            f"{_clip_text(user_task, 4000)}\n"
            "--- revised ---\n"
            f"{_clip_text(revision.revised_task, 6000)}"
        )
        if revision.clarifying_questions:
            self._log(
                "PROMPT REVISION QUESTIONS\n"
                + "\n".join(f"- {q}" for q in revision.clarifying_questions)
            )

        if self.revise_prompt == "interactive":
            if self.prompt_revision_selector is None:
                raise _PromptRevisionError(
                    "prompt.revise_prompt='interactive' needs an interactive selector"
                )
            selected = self.prompt_revision_selector(
                user_task,
                revision.revised_task,
                revision.clarifying_questions,
            )
            if selected is None or not selected.strip():
                raise _PromptRevisionError("operator aborted prompt revision")
            selected_task = selected.strip()
            if selected_task == user_task.strip():
                return user_task
            return _format_effective_task(
                user_task,
                _PromptRevision(
                    revised_task=selected_task,
                    clarifying_questions=revision.clarifying_questions,
                ),
            )

        return _format_effective_task(user_task, revision)

    def _call_with_retry(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int,
    ) -> Any:
        """Bounded-retry wrapper around ``provider.call``: up to
        ``provider_retry_count + 1`` attempts. Two retry paths share that budget:

        - Transient ``ProviderError`` (Anthropic 529, OpenRouter 502, brief socket
          timeout): retried with exponential backoff + full jitter so one flap
          doesn't abort the run. ``BudgetExceeded`` is never retried (hard stop).
          Permanent client errors (``ProviderError.status_code`` in
          ``_NON_RETRYABLE_HTTP_STATUSES``: 400/401/402/403/404/422) re-raise
          immediately without consuming a retry -- a second identical request
          cannot succeed (observed live: a 402 "Insufficient credits" was
          otherwise retried every remaining turn).
        - A self-contradictory empty tool-call response
          (``_is_empty_tool_call_response``: stop_reason promises a tool call but
          none and no text came back -- GLM via OpenRouter, ~50% post-restart):
          retried with a short fixed delay (model flakiness, not rate-limiting),
          excluding ``stop_reason=length`` starvation. If every attempt is empty
          the last is returned and the loop's went_quiet handler takes over.
        """
        attempts = max(1, self.provider_retry_count + 1)
        last_exc: ProviderError | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = self.provider.call(
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    should_abort=self.should_abort,
                    should_interrupt=self.should_interrupt,
                )
            except (ProviderAborted, ProviderInterrupted):
                raise  # operator stop/steer: handle it, never retry as a fault
            except ProviderError as exc:
                last_exc = exc
                non_retryable = exc.status_code in _NON_RETRYABLE_HTTP_STATUSES
                if attempt < attempts and not non_retryable:
                    base_delay = self.provider_retry_delay_s * (2 ** (attempt - 1))
                    capped_delay = min(base_delay, self.provider_retry_max_delay_s)
                    # jitter (full jitter, lower-bounded at 0.5) decorrelates
                    # concurrent retriers; non-crypto randomness is fine here.
                    delay = capped_delay * random.uniform(0.5, 1.0)  # noqa: S311
                    # Honor an upstream Retry-After (429/503): wait at least the
                    # advertised window (bounded), since our own backoff is
                    # usually shorter and would just burn the retries before the
                    # rate-limit clears.
                    if exc.retry_after_s is not None:
                        delay = max(delay, min(exc.retry_after_s, _RETRY_AFTER_CEILING_S))
                    self._log(
                        f"LOOP: provider error attempt {attempt}/{attempts}: "
                        f"{exc} - retrying in {delay:.2f}s"
                    )
                    self._emit(
                        "loop.provider.retry",
                        attempt=attempt,
                        error=str(exc)[:200],
                    )
                    time.sleep(delay)
                    continue
                if non_retryable:
                    self._log(f"LOOP: provider error {exc.status_code} is permanent; not retrying")
                    self._emit(
                        "loop.provider.fatal",
                        status_code=exc.status_code,
                        error=str(exc)[:200],
                    )
                raise
            # A self-contradictory empty tool-call response (GLM via OpenRouter,
            # ~50% after a context restart): retry the identical request, which
            # recovers it about half the time. Bounded by the same attempt budget;
            # if every attempt comes back empty the loop's went_quiet handler takes
            # over. A short delay (no exponential growth) -- this is model
            # flakiness, not rate-limiting.
            if _is_empty_tool_call_response(resp) and attempt < attempts:
                delay = min(self.provider_retry_delay_s, 1.0) * random.uniform(0.5, 1.0)  # noqa: S311
                self._log(
                    f"LOOP: empty tool-call response attempt {attempt}/{attempts}"
                    f" (stop_reason={resp.stop_reason!r}, no tool_use/text);"
                    f" retrying in {delay:.2f}s"
                )
                self._emit(
                    "loop.provider.empty_tool_call_retry",
                    attempt=attempt,
                    stop_reason=str(resp.stop_reason),
                )
                time.sleep(delay)
                continue
            return resp
        # Defensive: loop above either returns or raises; this is unreachable.
        # Kept for type-checker exhaustiveness in case the loop body changes.
        assert last_exc is not None
        raise last_exc

    def _has_reviewer(self) -> bool:
        """A second opinion is available: the review panel (seats) or the legacy
        single critic. Gates the in-loop critic triggers."""
        return bool(self.review_seats) or self.critic_provider is not None

    def _review_or_critic(
        self,
        *,
        state: _LoopState,
        messages: list[dict[str, Any]],
        trigger: str,
        iteration: int,
    ) -> _CritiqueResult | None:
        """Dispatch the in-loop second-opinion: the grounded review PANEL when
        ``review_seats`` is configured, else the legacy single critic. Both
        return a ``_CritiqueResult`` the trigger logic consumes identically."""
        if self.review_seats:
            return self._run_review_panel(state, trigger=trigger, iteration=iteration)
        return self._run_critic(
            task=state.original_task, messages=messages, trigger=trigger, iteration=iteration
        )

    def _run_diff(self) -> str:
        """The run's cumulative change: base commit vs the working tree, so it
        includes committed AND uncommitted edits, with untracked files as
        additions. Empty if no base is known or git fails. Routed through
        git_ops so the repo-controlled fsmonitor/diff.external/hooks keys stay
        neutralized (a raw `git diff` here would run a poisoned `.git/config`
        payload on the host)."""
        if not self.base_sha:
            return ""
        return diff_since(self.root, self.base_sha)

    def _read_agents_md(self) -> str:
        path = self.root / "AGENTS.md"
        try:
            return path.read_text(encoding="utf-8") if path.is_file() else ""
        except OSError:
            return ""

    def _readonly_review_tools(self) -> tuple[list[ToolDefinition], ReviewDispatch]:
        return build_readonly_review_tools(self.dispatcher)

    def _run_review_panel(
        self, state: _LoopState, *, trigger: str, iteration: int
    ) -> _CritiqueResult | None:
        """Run the grounded review panel over the run diff. Returns a
        ``_CritiqueResult`` (``satisfied=False`` only when the panel BLOCKS and
        the gate is still armed). Per-seat + panel events are emitted in seat
        order; the per-run rejection counter decays on a pass and disarms the gate
        once it hits the cap so a gating panel can never stall the run."""
        diff = self._run_diff()
        if not diff.strip():
            # No diff to ground against (nothing changed, or base_sha missing on a
            # pre-field resume). Can't review -> approve, but make the skip visible
            # so a "gate didn't run" is never silent.
            self._emit(
                "loop.review.skipped", iteration=iteration, trigger=trigger, reason="no_diff"
            )
            return None
        # Skip the panel once the run's remaining token budget falls below
        # review_budget_fraction: reviewing is most expensive (esp. explore-tier
        # seats) exactly when budget is scarcest, and a skipped panel is
        # approve-and-proceed (the before_finish gate only blocks on an explicit
        # unsatisfied critique, so returning None here lets finish through). This
        # is the sole read site for review_budget_fraction.
        remaining = self._budget_fraction_remaining()
        if remaining is not None and remaining < self.review_budget_fraction:
            self._emit(
                "loop.review.skipped",
                iteration=iteration,
                trigger=trigger,
                reason="budget_fraction",
                remaining=round(remaining, 3),
            )
            return None
        # on_verify_fail/periodic never gate (advisory text only); only
        # before_finish consumes .satisfied + the rejection counter.
        decision: ReviewDecision = (
            self.review_decision if trigger == "before_finish" else "advisory"
        )
        ctx = ReviewContext(
            task=state.original_task,
            agents_md=self._read_agents_md(),
            diff=diff,
            verify_ok=state.last_verify_ok,
            verify_output=state.last_verify_tail,
        )
        self._emit(
            "loop.review.start", iteration=iteration, trigger=trigger, seats=len(self.review_seats)
        )
        tools: list[ToolDefinition] | None = None
        dispatch: ReviewDispatch | None = None
        if any(s.tier == "explore" for s in self.review_seats):
            tools, dispatch = self._readonly_review_tools()
        try:
            result = run_panel(
                self.review_seats,
                ctx,
                decision=decision,
                quorum=self.review_quorum,
                panel_id=f"{trigger}-{iteration}",
                concurrency=self.review_concurrency,
                tools=tools,
                dispatch=dispatch,
            )
        except BudgetExceeded:
            self._emit("loop.review.skipped", iteration=iteration, reason="budget")
            return None
        for v in result.per_seat:
            self._emit(
                "loop.review.seat",
                iteration=iteration,
                seat=v.seat,
                model=v.model,
                verdict="abstain" if v.error else v.verdict,
                findings=len(v.findings),
            )
        disarmed = state.review_rejections_total >= self.review_max_total_rejections
        effective_blocked = result.blocked and not disarmed
        self._emit(
            "loop.review.panel",
            iteration=iteration,
            trigger=trigger,
            decision=decision,
            blocked=effective_blocked,
            raw_blocked=result.blocked,
            disarmed=disarmed,
            n_block=result.n_block,
            n_abstain=result.n_abstain,
        )
        if trigger == "before_finish":
            if effective_blocked:
                state.review_rejections_total += 1
            else:
                state.review_rejections_total = max(0, state.review_rejections_total - 1)
        text = render_findings(result.merged_findings) or "No blocking findings."
        return _CritiqueResult(text=text, satisfied=not effective_blocked)

    def _run_critic(
        self,
        *,
        task: str,
        messages: list[dict[str, Any]],
        trigger: str,
        iteration: int,
    ) -> _CritiqueResult | None:
        """Invoke the reviewer model as an in-loop critic.

        Returns None when no critic provider is configured (caller treats
        as "no critique, proceed"). Provider/budget errors are caught and
        logged so a flaky critic never aborts an otherwise-working run.
        """
        if self.critic_provider is None:
            return None
        transcript = _format_messages_tail_for_critic(messages)
        user_msg = (
            f"TASK:\n{task}\n\nTRIGGER: {trigger}\n\n"
            f"RECENT WORKER ACTIVITY (most recent last):\n{transcript}\n\n"
            "Critique. End with VERDICT: SATISFIED or VERDICT: NEEDS_WORK."
        )
        self._emit("loop.critic.call", iteration=iteration, trigger=trigger)
        try:
            resp = self.critic_provider.call(
                system=_CRITIC_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                tools=[],
                max_tokens=self.per_call_max_tokens,
                temperature=self.critic_temperature,
            )
        except (ProviderError, BudgetExceeded) as exc:
            self._log(f"  critic call failed: {exc}")
            self._emit("loop.critic.failed", iteration=iteration, error=str(exc)[:200])
            return None
        text = (resp.text or "").strip()
        satisfied = _parse_critic_verdict(text)
        self._emit(
            "loop.critic.verdict",
            iteration=iteration,
            trigger=trigger,
            satisfied=satisfied,
        )
        return _CritiqueResult(text=text, satisfied=satisfied)

    def _steer_outcome(
        self, steer_result: str | None, iteration: int, state: _LoopState
    ) -> RunResult | None:
        """Map a _maybe_handle_steer result to a terminal RunResult, or None to keep
        going (empty steer, or an instruction injected into messages)."""
        if steer_result == "abort":
            self._emit("run.end", reason="steer_abort", iterations=iteration, all_passed=False)
            return RunResult(
                completed=False,
                reason="steer_abort",
                summary=f"operator aborted at iter {iteration} via steering prompt",
                iterations=iteration,
                tool_calls=state.tool_calls,
            )
        if steer_result == "detach":
            # Not an end: the caller respawns a detached `resume` that appends to this
            # same log, so a persistent viewer follows straight through (no run.end).
            # The per-iteration snapshot is the resume point.
            return RunResult(
                completed=False,
                reason="detached",
                summary=f"operator detached at iter {iteration}; resuming in the background",
                iterations=iteration,
                tool_calls=state.tool_calls,
            )
        return None

    def _maybe_handle_steer(
        self,
        messages: list[dict[str, Any]],
        iteration: int,
    ) -> str | None:
        """Operator steering between iterations.

        Returns ``"abort"`` if the operator typed "abort" at the prompt;
        the loop should then return a steer_abort result. Returns ``None``
        in all other cases (no request, empty steer, or instruction
        injected into messages).

        Polls steer_requested() and, on a positive, calls steer_prompt()
        to capture operator text. Empty / None / KeyboardInterrupt aborts;
        boundary is between completed iters so a tool_use / tool_result pair
        is never split.
        """
        if not self.steer_requested():
            return None
        self._emit("loop.steer.requested", iteration=iteration)
        self._log(f"STEER: operator requested mid-run steering at iter {iteration}")
        try:
            text = self.steer_prompt()
        finally:
            self.steer_clear()
        if text is None or not text.strip():
            self._log("  (empty - continuing)")
            return None
        steer_text = text.strip()
        if steer_text.lower() == "abort":
            self._emit("loop.steer.aborted")
            self._log("  abort - halting the run")
            return "abort"
        if steer_text.lower() == "detach":
            self._emit("loop.steer.detached")
            self._log("  detach - stopping to resume in the background")
            return "detach"
        self._log(f"  injecting steering instruction ({len(steer_text)} chars)")
        self._emit("loop.steer.injected", chars=len(steer_text))
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "OPERATOR STEERING (mid-run instruction; "
                            "incorporate this into your next step):\n"
                            f"{steer_text}"
                        ),
                    }
                ],
            }
        )
        return None

    def _log(self, msg: str) -> None:
        self.logger(f"[agent6] {msg}")

    def _emit(self, event_type: str, **fields: Any) -> None:
        if self.events is not None:
            self.events.emit(event_type, **fields)

    def _emit_budget(self, iteration: int) -> None:
        """Per-iteration usage heartbeat: running token + cost totals. Lets
        `agent6 runs show` / the TUI show live spend, and leaves a recent event at
        the start of each iteration so a long provider call is still
        distinguishable from a stall."""
        if self.budget is None:
            return
        snap = self.budget.snapshot()
        cost, _ = self.budget.estimate_usd()
        self._emit(
            "loop.budget",
            iteration=iteration,
            input_tokens=snap.input_total,
            output_tokens=snap.output_total,
            cache_read_tokens=snap.cache_read_total,
            cost_usd=round(cost, 6),
        )
