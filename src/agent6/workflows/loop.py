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
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agent6.budget import BudgetExceeded, BudgetTracker
from agent6.config import Config
from agent6.git_ops import GitError, commit_all, commit_diff
from agent6.git_ops import status as git_status
from agent6.graph.client import CuratorClientError, GraphClient
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft, UpdateStatusIntent
from agent6.providers import Provider, ProviderError, ToolDefinition
from agent6.tools.dispatch import OperatorCommandUnexecutable, ToolDispatcher, ToolError
from agent6.tools.schema import (
    ALL_TOOLS,
    ASK_EXTRA_TOOLS,
    LOOP_EXTRA_TOOLS,
    MACHINE_EXTRA_TOOLS,
    PLAN_EXTRA_TOOLS,
    ApplyEditInput,
    ApplyPatchInput,
    FinishPlanningInput,
    FinishRunInput,
    RunCommandInput,
    RunVerifyInput,
)
from agent6.types import RepoSummary
from agent6.workflows._compaction import (
    DROP_BLOCKS_AT_CHARS as _DROP_BLOCKS_AT_CHARS,
)
from agent6.workflows._compaction import (
    SUMMARISE_AT_CHARS as _SUMMARISE_AT_CHARS,
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
    CONTEXT_RESTART_NOTICE as _CONTEXT_RESTART_NOTICE,
)
from agent6.workflows._prompts import (
    CONTEXT_SUMMARY_SYSTEM_PROMPT as _CONTEXT_SUMMARY_SYSTEM_PROMPT,
)
from agent6.workflows._prompts import (
    CRITIC_SYSTEM_PROMPT as _CRITIC_SYSTEM_PROMPT,
)
from agent6.workflows._prompts import (
    PROMPT_REVISION_SYSTEM_PROMPT as _PROMPT_REVISION_SYSTEM_PROMPT,
)
from agent6.workflows._prompts import build_system_prompt as _build_system_prompt
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

if TYPE_CHECKING:
    from agent6.events import EventSink


# HTTP statuses that will never succeed on a blind retry of the same request.
# 400 bad request, 401/403 auth, 402 insufficient credits, 404 bad
# model/endpoint, 422 malformed body. Retrying these only burns wall-time
# (observed live: a 402 "Insufficient credits" was retried on every turn for the
# rest of the run). 408/409/429 and all 5xx remain retryable and fall through to
# the normal backoff.
_NON_RETRYABLE_HTTP_STATUSES = frozenset({400, 401, 402, 403, 404, 422})

# cache_control marker on the initial user message
# so the system + initial context get cached across the loop's turns.
_CACHE_CONTROL_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}


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


# plan at all. A plan rarely needs more than a handful of reads.
_PLAN_BUDGET_NUDGE_BELOW = 0.35
_PLAN_NUDGE_AFTER_ITERS = 12

# Tool names that mutate the task DAG; after one runs we re-snapshot the graph
# (graph.update event) so a live viewer can render the worker's task breakdown.
_DAG_MUTATING_TOOLS = frozenset({"add_task", "update_task", "set_cursor"})

# verify-settled completion (run mode). A non-metric run has no positive "done"
# signal, clean exit depends on the worker volunteering finish_run, and a weak
# worker keeps re-running read-only commands after success (Kimi K2.6 observed:
# 128 iters when done at ~45). Once verify has passed, count iterations that
# make no progress (no new commit + no edit): nudge to finish at the first
# threshold, hard-stop at the second. NOT "green verify = instant stop", verify
# fires per-edit and is often lenient, so green-but-still-editing must continue.
# Thresholds are deliberately generous: the failure mode is only a little wasted
# budget on an already-done run, whereas a too-tight window could cut off a
# worker still reading toward its next edit in a big multi-file change.
_VERIFY_SETTLED_NUDGE_AFTER = 3
_VERIFY_SETTLED_STOP_AFTER = 6

_VERIFY_SETTLED_NUDGE = (
    "[harness settled] Your recent changes are committed and your last turns"
    " made no new changes (no commit, no edit). If the task is complete, call"
    " finish_run now with a short summary. If not, make a concrete edit toward"
    " what remains — do not keep re-running read-only commands."
)

# A non-metric `run` injects a one-shot wrap-up directive when the budget gets
# low. Observed live (Kimi K2.6): the worker solves the task, never re-runs
# verify, never calls finish_run, and burns the remaining budget on read-only
# commands; the verify-settled detector cannot engage without a green verify.
_RUN_BUDGET_NUDGE_BELOW = 0.25

_RUN_BUDGET_NUDGE = (
    "[harness budget] You are running low on budget. Run `run_verify_command`"
    " NOW. If it passes, call finish_run immediately with a short summary. If"
    " it fails, fix ONLY the smallest blocking issue, re-verify, and finish."
    " Do not run any other commands."
)

# Gateless variant (no verify command this run): there is nothing to verify, so
# steer straight to finish_run.
_RUN_BUDGET_NUDGE_GATELESS = (
    "[harness budget] You are running low on budget. Call finish_run NOW with a"
    " short summary of what you changed. Do not run any other commands."
)

_PLAN_BUDGET_NUDGE = (
    "[harness budget] You are running low on token budget and have NOT yet"
    " called finish_planning. Stop reading and reasoning now and call"
    " finish_planning immediately with the best plan you have — a concise, even"
    " rough, plan that is actually delivered is far more useful than an"
    " exhaustive one you never emit. Do not call any other tool first."
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


# The ONLY tools an explore-tier reviewer may use: read-only navigation, no
# edits/commits/run_command/dag/finish. Enforced both by what we expose AND by
# the dispatch wrapper (defense in depth).
_READONLY_REVIEW_TOOLS = frozenset(
    {
        "read_file",
        "list_dir",
        "grep",
        "outline",
        "find_definition",
        "find_references",
        "agent6_docs",
    }
)


def _tool_definitions(
    dispatcher: ToolDispatcher,
    *,
    mode: Literal["run", "plan", "ask", "machine", "agent"] = "run",
) -> list[ToolDefinition]:
    """Build the tool list exposed to the loop. Filters by what the
    dispatcher actually allows (e.g. run_command may be disabled).

    ``mode="plan"`` filters mutating tools
    (``apply_edit``/``apply_patch``) out of ``ALL_TOOLS`` and swaps
    ``LOOP_EXTRA_TOOLS`` for ``PLAN_EXTRA_TOOLS`` (drops
    ``finish_run``/``run_metric_command``, adds ``finish_planning``).
    ``mode="machine"`` (machine authoring) keeps only read-only navigation +
    ``finish_run`` so the agent's one job is to emit a `.asm.toml`.
    """
    available = set(dispatcher.available_tool_names())
    extras: tuple[type[Any], ...]
    if mode == "plan":
        extras = PLAN_EXTRA_TOOLS
    elif mode == "ask":
        extras = ASK_EXTRA_TOOLS
    elif mode in ("machine", "agent"):
        extras = MACHINE_EXTRA_TOOLS
    else:
        extras = LOOP_EXTRA_TOOLS
    base_tools: tuple[type[Any], ...] = ALL_TOOLS
    if mode in ("plan", "ask"):
        # Plan and ask are read-only; filter mutating tools even if the
        # dispatcher would otherwise allow them (the dispatcher's own
        # mode guard is the second line of defence).
        blocked = {ApplyEditInput.TOOL_NAME, ApplyPatchInput.TOOL_NAME}
        base_tools = tuple(cls for cls in ALL_TOOLS if cls.TOOL_NAME not in blocked)
    elif mode in ("machine", "agent"):
        # Authoring / machine agent-state: read-only navigation + finish_run
        # only, no edit/patch/verify/run_command. The deliverable is the
        # finish_run payload, not a file edit or a command run, and weak models
        # otherwise wander off editing the repo (observed live on Kimi K2.6).
        blocked = {
            ApplyEditInput.TOOL_NAME,
            ApplyPatchInput.TOOL_NAME,
            RunVerifyInput.TOOL_NAME,
            RunCommandInput.TOOL_NAME,
        }
        base_tools = tuple(cls for cls in ALL_TOOLS if cls.TOOL_NAME not in blocked)
    out: list[ToolDefinition] = []
    for cls in (*base_tools, *extras):
        if cls.TOOL_NAME not in available and cls not in extras:
            # Extras (finish_run / finish_planning / run_metric / dag_*) are
            # always exposed even though they're not in ALL_TOOLS.
            continue
        schema = cls.model_json_schema()
        schema.setdefault("type", "object")
        out.append(
            ToolDefinition(
                name=cls.TOOL_NAME,
                description=cls.TOOL_DESCRIPTION,
                input_schema=schema,
            )
        )
    # Any MCP tools the dispatcher's manager discovered get
    # appended verbatim. Names already carry the `mcp__<server>__`
    # prefix so they can never collide with built-in tool names.
    mgr = getattr(dispatcher, "_mcp_manager", None)
    if mgr is not None:
        for desc in mgr.descriptors():
            schema = dict(desc.input_schema)
            schema.setdefault("type", "object")
            out.append(
                ToolDefinition(
                    name=desc.qualified_name,
                    description=desc.description or f"MCP tool {desc.tool_name!r}",
                    input_schema=schema,
                )
            )
    return out


def build_readonly_review_tools(
    dispatcher: ToolDispatcher,
) -> tuple[list[ToolDefinition], ReviewDispatch]:
    """Read-only tool surface for explore-tier review seats: the navigation tools
    *dispatcher* exposes filtered to ``_READONLY_REVIEW_TOOLS``, plus a dispatch
    wrapper that REFUSES anything outside the allowlist (so a reviewer can never
    edit, commit, run a command, or mutate the task graph). Shared by the in-loop
    panel and the post-hoc ``agent6 review`` path."""
    tools = [
        t for t in _tool_definitions(dispatcher, mode="run") if t.name in _READONLY_REVIEW_TOOLS
    ]

    def dispatch(name: str, tool_input: dict[str, Any]) -> Any:
        if name not in _READONLY_REVIEW_TOOLS:
            raise ToolError(f"review reviewer may not call {name!r} (read-only)")
        return dispatcher.dispatch(name, tool_input)

    return tools, dispatch


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
    # Degenerate-loop guard: a back-to-back streak of the same (tool, args)
    # signature. Reset on any change so a normal re-read after edits is fine.
    last_tool_signature: str | None = None
    repeat_streak: int = 0
    repeat_warning_emitted_at: int = 0
    # Intervention nudge counters (each capped by a module-level patience const).
    went_quiet_nudges_used: int = 0
    plateau_nudges_used: int = 0
    metric_finish_nudges_used: int = 0
    plan_finish_nudged: bool = False
    # verify-settled completion (run mode): once verify has passed -- or, on a
    # gateless run, once an edit has been committed -- count no-progress
    # iterations; nudge then stop a worker that spins after success.
    verify_ever_passed: bool = False
    gateless_ever_committed: bool = False
    verify_settled_idle: int = 0
    verify_settled_nudged: bool = False
    run_budget_nudged: bool = False


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
    # without sending a response" (httpx RemoteProtocolError, no HTTP status),
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
    # The CLI wires this to the reviewer model when workflow.revise_prompt !=
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

    def run(self, user_task: str) -> RunResult:
        """Drive the single-loop agent to completion."""
        if self.mode == "plan" and self.plan_output_path is None:
            raise ValueError("Workflow(mode='plan') requires plan_output_path to be set")
        self._emit("run.start", user_task=user_task[:200], mode=self.mode)
        self._log("LOOP: LOAD_CONTEXT")
        repo = self._load_repo_summary()
        system = _build_system_prompt(config=self.config, repo=repo, mode=self.mode)

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
        # cache_control marker on the user message so the prefix stays
        # cached across the loop's turns (lineage).
        dag_hint = ""
        if root_id is not None:
            dag_hint = (
                "\n\nThe DAG-as-tool surface is wired. Root task id is"
                f" `{root_id}`. Use `add_task` to break this into trackable"
                " subtasks (or skip the DAG entirely - it's optional)."
            )
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
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": initial_user,
                        "cache_control": _CACHE_CONTROL_EPHEMERAL,
                    }
                ],
            }
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

    def _drive_loop(  # noqa: PLR0911, PLR0912, PLR0915
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
        """Shared loop body for both fresh ``run()`` and ``resume()``.

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
            # plateau guard see the prior best (we persist a summary, not the
            # full history). `label` marks it as resume-reconstructed.
            state.metric_history.append(
                _MetricSample(
                    label="resumed",
                    score=metric_best_score,
                    returncode=0,
                    at_ceiling=metric_at_ceiling,
                )
            )
        for iteration in range(start_iteration, self.max_iterations + 1):
            self._emit_budget(iteration)
            self._maybe_compact(messages)
            self._maybe_pre_call_nudges(
                messages, state, iteration=iteration, start_iteration=start_iteration
            )
            # Snapshot BEFORE the LLM call. After this write, a
            # crash anywhere up to the next iteration's snapshot can be
            # resumed by re-running this same call.
            self._save_resume_snapshot(
                system=system,
                messages=messages,
                tool_calls=state.tool_calls,
                next_iteration=iteration,
                root_task_id=root_task_id,
                state=state,
            )

            try:
                resp = self._call_with_retry(
                    system,
                    messages,
                    tools,
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
            except ProviderError as exc:
                self._log(f"LOOP: provider error at iter {iteration}: {exc}")
                self._emit(
                    "run.end",
                    reason="provider_error",
                    iterations=iteration,
                    all_passed=False,
                )
                return RunResult(
                    completed=False,
                    reason="provider_error",
                    summary=f"provider error at iter {iteration}: {exc}",
                    iterations=iteration,
                    tool_calls=state.tool_calls,
                )

            # Reconstruct the assistant message exactly from the response
            # content blocks so tool_use IDs round-trip cleanly.
            assistant_blocks = resp.raw.get("content") or []
            messages.append({"role": "assistant", "content": assistant_blocks})

            if not resp.tool_uses:
                result = self._handle_no_tool_use(resp, messages, state, iteration=iteration)
                if result is not None:
                    return result
                continue
            # Dispatch each tool_use, append tool_result to the user message.
            finish_signal: str | None = None
            finish_payload: dict[str, Any] | None = None
            finish_kind: Literal["finish_run", "finish_planning"] = "finish_run"
            tool_results: list[dict[str, Any]] = []
            verify_just_passed = False
            verify_just_failed = False
            edited_this_iter = False
            committed_this_iter = False
            dag_mutated_this_iter = False
            metric_called_after_verify_pass = False
            metric_feedback_text: str | None = None
            metric_plateau_finish: str | None = None
            # This iteration produced tool_uses, so the went_quiet
            # nudge budget refills (failures are per-streak, not per-run).
            state.went_quiet_nudges_used = 0
            for tu in resp.tool_uses:
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
                self._emit("loop.tool.call", name=name, iteration=iteration)
                try:
                    result = self.dispatcher.dispatch(name, tool_input)
                    content = json.dumps(result, ensure_ascii=False)
                    # auto-commit-on-verify-pass. Whenever the
                    # agent's verify_command returns exit 0, the workflow
                    # immediately commits any uncommitted changes so
                    # score.sh's git-history rescue can pick the best
                    # commit.
                    # commit strategy without forcing the agent to think
                    # about git. Agent can still commit explicitly via
                    # run_command if it wants a specific message.
                    if name == "run_verify_command":
                        rc = result.get("returncode")
                        if rc == 0:
                            verify_just_passed = True
                        elif rc is not None:
                            verify_just_failed = True
                        if rc is not None:
                            # Remember the latest verify result so the review
                            # panel can ground findings against it (verify-pass
                            # presumes correctness; verify-red is the hard signal).
                            state.last_verify_ok = rc == 0
                            tail = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
                            state.last_verify_tail = tail.strip()[-2000:]
                    elif name == "run_metric_command":
                        if verify_just_passed:
                            metric_called_after_verify_pass = True
                        metric_feedback_text = self._record_metric_result(
                            state.metric_history,
                            result,
                            iteration=iteration,
                            label=f"manual iter {iteration}",
                            sha="",
                        )
                        if verify_just_passed:
                            metric_plateau_finish = self._metric_plateau_summary(
                                state.metric_history
                            )
                    if name in ("apply_edit", "apply_patch"):
                        edited_this_iter = True
                    if name in _DAG_MUTATING_TOOLS:
                        dag_mutated_this_iter = True  # snapshot once after the turn
                except ToolError as exc:
                    content = json.dumps({"error": str(exc)})
                    self._log(f"  tool_error: {name}: {exc}")
                except OperatorCommandUnexecutable as exc:
                    # The operator's verify/metric command cannot run in the jail.
                    # The model can't fix operator config, so abort loudly instead
                    # of letting it flail against a verify that never executes (and
                    # never silently report all_passed on an un-run gate).
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
                        tool_calls=state.tool_calls,
                    )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu_id,
                        "content": _cap_tool_result(content, tool_name=name),
                    }
                )
                if name == FinishRunInput.TOOL_NAME:
                    finish_kind = "finish_run"
                    finish_signal = (
                        tool_input.get("summary", "(no summary)")
                        if isinstance(tool_input, dict)
                        else "(no summary)"
                    )
                    raw_result = tool_input.get("result") if isinstance(tool_input, dict) else None
                    finish_payload = raw_result if isinstance(raw_result, dict) else None
                elif name == FinishPlanningInput.TOOL_NAME:
                    finish_kind = "finish_planning"
                    finish_signal = (
                        tool_input.get("summary", "(no summary)")
                        if isinstance(tool_input, dict)
                        else "(no summary)"
                    )
                    # Persist the plan markdown. Schema validation
                    # already guaranteed `plan_markdown` is a non-empty
                    # string when the dispatcher dispatched it, but the
                    # raw tool_input is what the model sent us; be defensive
                    # so a malformed call cannot crash the loop.
                    plan_md = ""
                    if isinstance(tool_input, dict):
                        plan_md = str(tool_input.get("plan_markdown", ""))
                    if self.plan_output_path is not None and plan_md:
                        try:
                            self.plan_output_path.parent.mkdir(parents=True, exist_ok=True)
                            self.plan_output_path.write_text(plan_md, encoding="utf-8")
                            self._log(
                                f"  plan written: {self.plan_output_path} ({len(plan_md)} chars)"
                            )
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

            # One task-DAG snapshot per turn (not per mutation), so several
            # add_task/update_task calls in a turn collapse to a single event.
            if dag_mutated_this_iter:
                self._emit_graph_snapshot()

            # Auto-commit, AFTER the tool_results so the commit message
            # reflects the iteration number. Best-effort: commit failures (e.g.
            # nothing to commit) are logged but don't abort the run. The catch
            # also handles OSError (subprocess failures) so a transient FS
            # hiccup doesn't kill an otherwise-fine run.
            # Plan mode is read-only; never auto-commit.
            # With a verify command, commits are gated on a green verify; with
            # none configured (a gateless run), each editing step is committed
            # as an un-gated checkpoint so resume + the audit trail still work.
            # `edited_this_iter` (apply_edit/apply_patch) is the cheap fast-path;
            # fall back to a worktree-dirty check so run_command-authored edits
            # are also checkpointed (else they'd never be committed gateless).
            gateless = not self.config.workflow.verify_command
            gateless_changed = gateless and (edited_this_iter or self._worktree_dirty())
            if self.mode == "run" and (verify_just_passed or gateless_changed):
                commit_subject = _summarise_assistant_text_for_commit(
                    resp.text or "",
                    iteration,
                    fallback="checkpoint" if gateless else "verify passed",
                )
                sha = ""
                try:
                    sha = commit_all(
                        self.root,
                        commit_subject,
                    )
                    self._log(f"  auto-commit: {sha[:12]}")
                    self._emit("loop.auto_commit", iteration=iteration, sha=sha)
                    committed_this_iter = bool(sha)
                    if gateless and sha:
                        # Seed the idle-stop net for gateless runs (no green
                        # verify ever fires); see the verify-settled logic below.
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
                    msg = str(exc).lower()
                    # "nothing to commit" can arrive in either
                    # the stdout half or the stderr half of the detail
                    # string (see git_ops._run). "no changes added"
                    # covers the variant when files were edited but
                    # only paths outside the worktree (or .gitignore'd)
                    # changed. "working tree clean" covers the case
                    # where verify passed without any file mutation.
                    benign = (
                        "nothing to commit" in msg
                        or "no changes added" in msg
                        or "working tree clean" in msg
                    )
                    if not benign:
                        self._log(f"  auto-commit failed: {exc}")
                        # Capture a status snapshot so the event payload
                        # tells the operator what was in the worktree
                        # at the failure point. Best-effort: if status
                        # itself raises (rare; outside-a-repo case is
                        # already gone by this point in the loop), omit.
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
                # REPL hook. Default no-op returns "continue".
                if sha:
                    directive = self.after_auto_commit(iteration, sha)
                    if directive == "stop":
                        self._log(f"LOOP: interactive stop at iter {iteration}")
                        self._emit_run_end_passed(reason="interactive_stop", iterations=iteration)
                        return RunResult(
                            completed=True,
                            reason="interactive_stop",
                            summary=f"stopped interactively after iter {iteration}",
                            iterations=iteration,
                            tool_calls=state.tool_calls,
                        )
                if not metric_called_after_verify_pass:
                    metric_feedback_text = self._auto_metric_feedback(
                        state.metric_history,
                        iteration=iteration,
                        sha=sha,
                    )
                    metric_plateau_finish = self._metric_plateau_summary(state.metric_history)

            # critic-in-loop triggers.
            #   on_verify_fail - the verify just failed; surface a
            #                    critique alongside the failure so the
            #                    worker has a second opinion before its
            #                    next edit.
            #   periodic       - every critic_period iterations.
            #   before_finish  - handled below, after finish_signal is
            #                    inspected, because it can revoke finish.
            critic_text: str | None = None
            if self.critic_mode == "on_verify_fail" and verify_just_failed and self._has_reviewer():
                critique = self._review_or_critic(
                    state=state,
                    messages=messages,
                    trigger="verify_failed",
                    iteration=iteration,
                )
                if critique is not None:
                    critic_text = critique.text
            elif (
                self.critic_mode == "periodic"
                and self._has_reviewer()
                and iteration % max(1, self.critic_period) == 0
            ):
                critique = self._review_or_critic(
                    state=state,
                    messages=messages,
                    trigger="periodic",
                    iteration=iteration,
                )
                if critique is not None:
                    critic_text = critique.text

            # before_finish: gate the agent's finish_run on critic approval.
            # If the critic says NEEDS_WORK, suppress the finish (the
            # tool_result still goes back so the call isn't half-applied)
            # and inject the critique - the loop carries on for another
            # iteration with the critique visible. After
            # `max_consecutive_critic_rejections` back-to-back rejections
            # we let the finish through (with the critique still
            # injected) so the worker can't bounce indefinitely.
            if (
                finish_signal is not None
                and finish_kind == "finish_run"
                and self.critic_mode == "before_finish"
                and self._has_reviewer()
            ):
                critique = self._review_or_critic(
                    state=state,
                    messages=messages,
                    trigger="before_finish",
                    iteration=iteration,
                )
                cap = self.max_consecutive_critic_rejections
                cap_reached = cap > 0 and state.consecutive_critic_rejections >= cap
                if critique is not None and not critique.satisfied and not cap_reached:
                    self._log(f"  critic rejected finish_run at iter {iteration}")
                    self._emit("loop.critic.rejected_finish", iteration=iteration)
                    finish_signal = None
                    finish_payload = None
                    state.consecutive_critic_rejections += 1
                    critic_text = (
                        "The critic rejected your finish_run call. Address the"
                        " issues below before calling finish_run again.\n\n" + critique.text
                    )
                elif critique is not None and not critique.satisfied and cap_reached:
                    self._log(
                        f"  critic rejected finish_run at iter {iteration} but"
                        f" rejection cap ({cap}) reached - letting finish through"
                    )
                    self._emit(
                        "loop.critic.rejection_cap_reached",
                        iteration=iteration,
                        rejections=state.consecutive_critic_rejections,
                    )
                    critic_text = (
                        "The critic flagged issues but the rejection cap was"
                        " reached; finish_run will be accepted. Critique:\n\n" + critique.text
                    )
                    state.consecutive_critic_rejections = 0
                elif critique is not None:
                    self._log("  critic approved finish_run")
                    state.consecutive_critic_rejections = 0

            # metric-run early-finish guard. On optimisation runs the worker
            # often calls finish_run with most of its budget unspent, even
            # though the task asks it to keep optimising up to the cap. Mirror
            # the plateau policy: while the run still has runway above the final
            # budget slice, reject an early finish_run a few times and nudge the
            # worker to keep going; only honour it once we are in the final
            # budget slice or patience is exhausted. Requires a real budget
            # signal - with none (tests / MCP) we defer to the worker's own
            # judgement so a finish can never deadlock.
            if (
                finish_signal is not None
                and finish_kind == "finish_run"
                and self.mode == "run"
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
                    finish_signal = None
                    finish_payload = None
                    tool_results.append({"type": "text", "text": _METRIC_FINISH_NUDGE})
                    self._log(
                        f"  metric early-finish rejected #{state.metric_finish_nudges_used}"
                        f" at iter {iteration} (budget {finish_budget_remaining:.0%} left)"
                    )
                    self._emit(
                        "loop.metric_early_finish.rejected",
                        iteration=iteration,
                        nudges_used=state.metric_finish_nudges_used,
                        budget_remaining=finish_budget_remaining,
                    )

            if critic_text:
                tool_results.append(
                    {
                        "type": "text",
                        "text": f"[critic]\n{critic_text}",
                    }
                )

            if metric_feedback_text:
                tool_results.append(
                    {
                        "type": "text",
                        "text": metric_feedback_text,
                    }
                )

            # degenerate-loop intervention. When the same
            # (tool, args) signature has been called >=3 times in a row,
            # append a one-shot system-style notice to the user turn so
            # the worker sees on its next call that re-issuing the same
            # request will not yield new information. We re-emit once
            # per "fresh" streak (when a new streak crosses the
            # threshold) so spamming the same call only triggers once
            # per latch episode. The repeat counter resets on any new
            # signature, so a normal re-read after an edit does not
            # trigger.
            repeat_threshold = 3
            if (
                state.repeat_streak >= repeat_threshold
                and state.repeat_warning_emitted_at < iteration - 1
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
                tool_results.append({"type": "text", "text": notice})
                self._emit(
                    "loop.loop_guard.triggered",
                    iteration=iteration,
                    tool=latched_name,
                    streak=state.repeat_streak,
                )
                self._log(
                    f"  loop-guard: {latched_name} called"
                    f" {state.repeat_streak}x in a row - injecting notice"
                )
                state.repeat_warning_emitted_at = iteration

            # metric-plateau handling. When a verified metric merely ties the
            # prior best, the plateau detector fires. Rather than quit at the
            # first stall (often with most of the budget unspent), nudge the
            # worker to pivot to a different approach; only stop once we are in
            # the final budget slice and have still failed to beat the best
            # after a few pivot nudges. With no budget signal (tests / MCP)
            # the fixed `_METRIC_PLATEAU_PATIENCE` bounds the nudging.
            plateau_should_stop = False
            if metric_plateau_finish is not None:
                budget_remaining = self._budget_fraction_remaining()
                # A metric at its provable ceiling (e.g. SCORE: 27/27) cannot
                # improve: stop now rather than nudge the worker to "pivot"
                # toward a number that does not exist. This is the dominant
                # cause of weak reasoning models burning their whole budget
                # (and wall-clock) re-deriving a solved task.
                in_final_slice = (
                    budget_remaining is None
                    or budget_remaining <= _METRIC_PLATEAU_STOP_BELOW_BUDGET
                )
                if self._metric_at_ceiling(state.metric_history):
                    plateau_should_stop = True
                    self._emit("loop.metric_ceiling.stop", iteration=iteration)
                elif in_final_slice and state.plateau_nudges_used >= _METRIC_PLATEAU_PATIENCE:
                    plateau_should_stop = True
                else:
                    state.plateau_nudges_used += 1
                    nudge_text = _metric_plateau_nudge(budget_remaining)
                    tool_results.append({"type": "text", "text": nudge_text})
                    budget_note = (
                        "n/a" if budget_remaining is None else f"{budget_remaining:.0%} left"
                    )
                    self._log(
                        "  metric_plateau pivot-nudge"
                        f" #{state.plateau_nudges_used} at iter {iteration} (budget {budget_note})"
                    )
                    self._emit(
                        "loop.metric_plateau.nudge",
                        iteration=iteration,
                        nudges_used=state.plateau_nudges_used,
                        budget_remaining=budget_remaining,
                    )

            # verify-settled completion bookkeeping (run mode). Track no-progress
            # iterations after the first green verify; nudge once, then stop.
            # "progress" is any forward motion the prompt encourages, so a
            # legitimately-working run is never truncated:
            #   - an apply_edit/apply_patch, or a new commit, or
            #   - an uncommitted worktree change (an edit made via run_command),
            #   - a verify RUN itself (re-verifying between reads is active work,
            #     not idle), held neutral so it neither resets nor accrues.
            # Only the pathology, spinning on read-only commands with a clean,
            # already-committed tree, accrues idle.
            if verify_just_passed:
                state.verify_ever_passed = True
            # Only governs PLAIN runs. A metric/optimisation run is also
            # mode=="run" but its completion is owned by the metric early-finish
            # guard + plateau/ceiling logic (which deliberately keep going while
            # budget remains); measure/analyse/read iterations there legitimately
            # make no commit, so the settled detector must defer to them. (Gating
            # the bookkeeping here also keeps the worktree-dirty git check off the
            # metric hot path.)
            non_metric_run = (
                self.mode == "run" and _metric_goal(self.config.workflow.metric) is None
            )
            # "Settled" once the run reached a good state: a green verify, or (on
            # a gateless run, where verify never fires) a committed edit.
            settled_seeded = state.verify_ever_passed or state.gateless_ever_committed
            if non_metric_run and settled_seeded:
                made_progress = committed_this_iter or edited_this_iter or self._worktree_dirty()
                if made_progress:
                    state.verify_settled_idle = 0
                    state.verify_settled_nudged = False  # a fresh idle streak may re-nudge
                elif not (verify_just_passed or verify_just_failed):
                    state.verify_settled_idle += 1
            verify_settled_stop = (
                non_metric_run
                and finish_signal is None
                and settled_seeded
                and state.verify_settled_idle >= _VERIFY_SETTLED_STOP_AFTER
            )
            if (
                non_metric_run
                and finish_signal is None
                and not verify_settled_stop
                and settled_seeded
                and state.verify_settled_idle >= _VERIFY_SETTLED_NUDGE_AFTER
                and not state.verify_settled_nudged
            ):
                state.verify_settled_nudged = True
                tool_results.append({"type": "text", "text": _VERIFY_SETTLED_NUDGE})
                self._emit(
                    "loop.verify_settled.nudge", iteration=iteration, idle=state.verify_settled_idle
                )

            messages.append({"role": "user", "content": tool_results})

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

            if verify_settled_stop:
                self._log(
                    f"LOOP: verify_settled at iter {iteration} (idle {state.verify_settled_idle})"
                )
                self._final_checkpoint(iteration)
                self._emit_run_end_passed(reason="verify_settled", iterations=iteration)
                return RunResult(
                    completed=True,
                    reason="verify_settled",
                    summary="verify passed and the worker stopped making changes",
                    iterations=iteration,
                    tool_calls=state.tool_calls,
                )

            if plateau_should_stop:
                assert metric_plateau_finish is not None
                self._log(f"LOOP: metric_plateau at iter {iteration}")
                self._final_checkpoint(iteration)
                self._emit_run_end_passed(reason="metric_plateau", iterations=iteration)
                return RunResult(
                    completed=True,
                    reason="metric_plateau",
                    summary=metric_plateau_finish,
                    iterations=iteration,
                    tool_calls=state.tool_calls,
                )

            # loop-guard escalation. The notice above is
            # advisory; if the worker keeps issuing the same call past
            # `loop_guard_kill_threshold`, terminate the run before it
            # burns the rest of the budget circling. Threshold of 0
            # disables (notice-only behaviour). The kill happens AFTER
            # appending tool_results so the transcript on disk reflects
            # exactly what the model produced up to the kill, which is
            # essential when triaging "why did my run die at iter N".
            if (
                self.loop_guard_kill_threshold > 0
                and state.repeat_streak >= self.loop_guard_kill_threshold
            ):
                latched_name = (state.last_tool_signature or "").split(":", 1)[0] or "<unknown>"
                self._log(
                    f"LOOP: loop_guard_killed at iter {iteration} -"
                    f" {latched_name} called {state.repeat_streak}x in a row"
                    f" (threshold={self.loop_guard_kill_threshold})"
                )
                self._emit(
                    "run.end",
                    reason="loop_guard_killed",
                    iterations=iteration,
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
                    iterations=iteration,
                    tool_calls=state.tool_calls,
                )

            if finish_signal is not None:
                self._log(f"LOOP: {finish_kind} called at iter {iteration}")
                self._final_checkpoint(iteration)
                self._emit_run_end_passed(reason=finish_kind, iterations=iteration)
                return RunResult(
                    completed=True,
                    reason=finish_kind,
                    summary=finish_signal,
                    iterations=iteration,
                    tool_calls=state.tool_calls,
                    finish_payload=finish_payload,
                )

            # audit finding: poll the steering flag between
            # iterations. The operator can press Ctrl-C once to drop a
            # steering instruction into the conversation; a second Ctrl-C
            # within 2s raises KeyboardInterrupt and aborts. Same shape as
            # the steering hook; safe boundary is
            # AFTER a complete iter so we never split a tool_use / tool_result
            # pair.
            steer_result = self._maybe_handle_steer(messages, iteration)
            if steer_result == "abort":
                self._emit(
                    "run.end",
                    reason="steer_abort",
                    iterations=iteration,
                    all_passed=False,
                )
                return RunResult(
                    completed=False,
                    reason="steer_abort",
                    summary=f"operator aborted at iter {iteration} via steering prompt",
                    iterations=iteration,
                    tool_calls=state.tool_calls,
                )

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

    def _maybe_pre_call_nudges(
        self,
        messages: list[dict[str, Any]],
        state: _LoopState,
        *,
        iteration: int,
        start_iteration: int,
    ) -> None:
        """Before the LLM call, inject a one-shot finish directive when a
        verbose planner or a non-metric run is reading forever without
        landing a plan / verify+finish before the budget dies."""
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

    def _handle_no_tool_use(  # noqa: PLR0912, PLR0915
        self, resp: Any, messages: list[dict[str, Any]], state: _LoopState, *, iteration: int
    ) -> RunResult | None:
        """Handle a turn with no tool_use. Either a silent finish (the agent
        emitted text; gated by the before_finish critic and the metric
        early-finish guard) or went-quiet (an empty turn; nudged up to a cap).
        Returns a terminal RunResult, or None to continue the loop after
        appending a nudge."""
        # Agent emitted no tool_use. audit finding:
        # distinguish "agent talked then stopped" (likely an
        # implicit finish - user gets the text as summary) from
        # "agent emitted nothing" (likely went-quiet failure -
        # provider returned an empty response, or the agent got
        # confused; bench scoring should NOT treat this as
        # success).
        text = resp.text.strip() if resp.text else ""
        if text:
            # silent_finish goes through the same
            # before_finish critic gate as an explicit
            # finish_run tool_use. Without this, an agent that
            # stops emitting tool calls bypasses critic review
            # entirely. Rejection cap is shared with the
            # tool_use path so a stubborn worker can't bounce
            # the loop forever.
            if self.critic_mode == "before_finish" and self._has_reviewer():
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
                    return None
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
            # metric-run early-finish guard, mirroring the finish_run
            # path: a silent finish on an optimisation run with budget to
            # spare should be nudged to keep optimising rather than
            # accepted. Without this, dropping tool_use was a way to skip
            # the plateau/early-finish policy entirely.
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
            self._log(
                f"LOOP: silent_finish at iter {iteration} - agent emitted text but no tool_use"
            )
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
        # reasoning-starvation trip-wire. When a model
        # spends its entire output budget on reasoning_content
        # and emits nothing user-visible, the provider returns
        # stop_reason="length" with empty text + no tool_uses.
        # Without this breadcrumb the failure mode is
        # indistinguishable from a model that genuinely gave up,
        # and the only way to diagnose it is to read raw
        # transcripts (took ~7 minutes per case
        # forensics). Surface it explicitly so the next
        # undetected reasoning model is one log line away.
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
        # nudge-and-retry instead of immediate exit.
        # Weak open-weights models occasionally emit a single
        # empty assistant turn mid-run; a one-line synthetic
        # user prompt almost always gets them back on track,
        # and costs ~50 input tokens vs aborting the entire run.
        # AGENT6_WENT_QUIET_MAX_NUDGES env override.
        env_max = os.environ.get("AGENT6_WENT_QUIET_MAX_NUDGES", "").strip()
        effective_max_nudges = int(env_max) if env_max.isdigit() else self.went_quiet_max_nudges
        if state.went_quiet_nudges_used < effective_max_nudges:
            state.went_quiet_nudges_used += 1
            # Drop the empty assistant turn we appended at the
            # top of this iteration. An assistant message with
            # empty content is rejected by Anthropic
            # ("messages.N.content: Input should be a valid
            # list") and is wasted context everywhere else.
            # The nudge becomes a normal user turn appended to
            # the prior assistant turn.
            if messages and messages[-1].get("role") == "assistant":
                last_content = messages[-1].get("content") or []
                if not last_content:
                    messages.pop()
            # starvation-specific nudge. When the previous
            # turn ended with stop_reason=length AND reasoning_content
            # ate the entire budget, the generic "your turn was empty"
            # message gives the model no actionable feedback and it
            # repeats the same reasoning loop next turn. Tell it
            # explicitly to stop thinking and commit to a tool call.
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
        # workflow.structural_priors=false -> base summary only (no hot symbols /
        # co-change / symbol outline), a leaner prompt that leans on on-demand tools.
        disp = self.dispatcher if self.config.workflow.structural_priors else None
        return load_repo_summary(self.root, dispatcher=disp)

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
        }
        self.resume_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.resume_state_path.with_suffix(self.resume_state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.resume_state_path)

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
            and _metric_at_fraction_ceiling(combined, score)
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

    def _worker_max_tokens(self) -> int:
        """Per-call output cap for the worker turn.

        Metric-optimization runs (mode "run" with a configured continuous
        metric) lift the ceiling to ``metric_task_max_tokens`` so a single
        turn can rewrite a hot function wholesale without truncating
        mid-apply_patch. Every other run keeps ``per_call_max_tokens``.
        """
        if self.mode == "run" and _metric_goal(self.config.workflow.metric) is not None:
            return max(self.per_call_max_tokens, self.metric_task_max_tokens)
        return self.per_call_max_tokens

    def _maybe_compact(self, messages: list[dict[str, Any]]) -> None:
        """Tiered compaction

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
        n_dropped = _compact_old_tool_results(
            messages, max_total_bytes=self.compact_drop_at_chars, keep_recent=2
        )
        if n_dropped:
            self._log(f"LOOP: compaction elided {n_dropped} old tool_result blocks")
            self._emit("loop.compact.dropped", n=n_dropped)
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
            self._summarise_and_restart(messages)

    def _summarise_and_restart(self, messages: list[dict[str, Any]]) -> None:
        """Replace the message history with (original task + a model-written
        progress summary). Mutates ``messages`` in place. The loop only calls
        this at the top of an iteration, where the history is balanced (every
        ``tool_use`` already has its ``tool_result``), so the restart can drop
        the middle without orphaning a tool-call pairing.
        """
        provider = self.summariser_provider or self.provider
        original = messages[0]
        transcript = _format_messages_tail_for_critic(
            messages[1:], max_messages=len(messages), max_chars=60_000
        )
        user_msg = (
            "Summarise the following agent transcript for a context restart.\n\n"
            f"TRANSCRIPT (oldest first):\n{transcript}"
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
            return
        summary = (resp.text or "").strip()
        if not summary:
            self._emit("loop.compact.summarise.failed", error="empty summary")
            return
        restart = {
            "role": "user",
            "content": [{"type": "text", "text": _CONTEXT_RESTART_NOTICE + summary}],
        }
        messages[:] = [original, restart]
        self._emit("loop.compact.summarise.done", summary_chars=len(summary))

    def _maybe_revise_prompt(self, user_task: str, repo: RepoSummary) -> str:
        if self.revise_prompt == "off":
            return user_task
        if self.prompt_reviser_provider is None:
            raise _PromptRevisionError(
                "workflow.revise_prompt is enabled but no reviser provider is wired"
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
                    "workflow.revise_prompt='interactive' needs an interactive selector"
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
    ) -> Any:
        """Single-retry wrapper around ``provider.call``.

        audit finding: a single transient ProviderError
        (Anthropic 529, OpenRouter 502, brief socket timeout) shouldn't
        abort the run. Retry once with a short fixed backoff; on the second
        failure re-raise and let the loop handle it as before. BudgetExceeded
        is never retried - it's a hard stop signal.

        Permanent client errors (``ProviderError.status_code`` in
        ``_NON_RETRYABLE_HTTP_STATUSES``: 401/402/403/404/422) are re-raised
        immediately without consuming a retry - a second identical request
        cannot succeed (observed live: a 402 "Insufficient credits" was
        otherwise retried on every remaining turn).
        """
        attempts = max(1, self.provider_retry_count + 1)
        last_exc: ProviderError | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self.provider.call(
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=self._worker_max_tokens(),
                    temperature=self.temperature,
                )
            except ProviderError as exc:
                last_exc = exc
                non_retryable = exc.status_code in _NON_RETRYABLE_HTTP_STATUSES
                if attempt < attempts and not non_retryable:
                    base_delay = self.provider_retry_delay_s * (2 ** (attempt - 1))
                    capped_delay = min(base_delay, self.provider_retry_max_delay_s)
                    # jitter (full jitter, lower-bounded at 0.5) decorrelates
                    # concurrent retriers; non-crypto randomness is fine here.
                    delay = capped_delay * random.uniform(0.5, 1.0)  # noqa: S311
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
        """The run's cumulative change (``git diff base_sha``: base commit vs the
        working tree, so it includes committed AND uncommitted edits). Empty if
        no base is known or git fails."""
        if not self.base_sha:
            return ""
        proc = subprocess.run(
            ["git", "diff", self.base_sha],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout if proc.returncode == 0 else ""

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
        `agent6 status` / the TUI show live spend, and leaves a recent event at
        the start of each iteration so a long provider call is still
        distinguishable from a stall."""
        if self.budget is None:
            return
        snap = self.budget.snapshot()
        cost, _ = self.budget.estimate_usd()
        self._emit(
            "loop.budget",
            iteration=iteration,
            input_tokens=snap.get("input_total"),
            output_tokens=snap.get("output_total"),
            cache_read_tokens=snap.get("cache_read_total"),
            cost_usd=round(cost, 6),
        )
