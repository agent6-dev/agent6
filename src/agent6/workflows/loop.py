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
from agent6.git_ops import GitError, commit_all, commit_diff
from agent6.git_ops import co_change_pairs as git_co_change_pairs
from agent6.git_ops import status as git_status
from agent6.graph.client import CuratorClientError, GraphClient
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft
from agent6.providers import Provider, ProviderError, ToolDefinition
from agent6.tools.dispatch import ToolDispatcher, ToolError
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
from agent6.workflows._prompts import (
    AGENT_SYSTEM_PROMPT_BASE as _AGENT_SYSTEM_PROMPT_BASE,
)
from agent6.workflows._prompts import (
    ASK_SYSTEM_PROMPT_BASE as _ASK_SYSTEM_PROMPT_BASE,
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
    MACHINE_SYSTEM_PROMPT_BASE as _MACHINE_SYSTEM_PROMPT_BASE,
)
from agent6.workflows._prompts import (
    PLAN_SYSTEM_PROMPT_BASE as _PLAN_SYSTEM_PROMPT_BASE,
)
from agent6.workflows._prompts import (
    PROMPT_REVISION_SYSTEM_PROMPT as _PROMPT_REVISION_SYSTEM_PROMPT,
)
from agent6.workflows._prompts import (
    SYSTEM_PROMPT_BASE as _SYSTEM_PROMPT_BASE,
)
from agent6.workflows._prompts import (
    V2_BUDGET_BLOCK_TEMPLATE as _V2_BUDGET_BLOCK_TEMPLATE,
)
from agent6.workflows._prompts import (
    V2_METRIC_BLOCK_TEMPLATE as _V2_METRIC_BLOCK_TEMPLATE,
)
from agent6.workflows._prompts import (
    V2_REPO_BLOCK_TEMPLATE as _V2_REPO_BLOCK_TEMPLATE,
)
from agent6.workflows._prompts import (
    V2_VERIFY_BLOCK_TEMPLATE as _V2_VERIFY_BLOCK_TEMPLATE,
)
from agent6.workflows._symbol_outline import (
    build_symbol_outline_block as _build_symbol_outline_block,
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


@dataclass(frozen=True, slots=True)
class RunResult:
    """Final state of a run.

    ``reason`` values:
      finish_run       - agent called the finish_run tool explicitly.
      silent_finish    - agent emitted text but no tool_use (talking).
      went_quiet       - agent emitted neither text nor tool_use.
      budget_exhausted - BudgetTracker raised; partial progress kept.
      provider_error   - ProviderError after retry; loop aborted.
    metric_plateau   - metric run tied prior best after enough samples.
            prompt_revision_failed - revise_prompt failed before the worker loop.
      max_iterations   - hit max_iterations cap without finish.
      steer_abort      - operator typed "abort" at a steering prompt.
    """

    completed: bool
    reason: str
    summary: str
    iterations: int
    tool_calls: int
    finish_payload: dict[str, Any] | None = None


class ResumeError(Exception):
    """Raised when resume cannot proceed (missing/corrupt snapshot)."""


@dataclass(frozen=True, slots=True)
class _ResumeSnapshot:
    """provider-agnostic in-memory snapshot of loop state.

    Written before each LLM call so a crash mid-call can be resumed from
    the same point. Provider-agnostic because the OpenAI provider
    translates anthropic-shaped messages before its transcript sink runs
    - we cannot reuse provider transcripts for cross-provider resume.
    """

    system: str
    messages: list[dict[str, Any]]
    tool_calls: int
    next_iteration: int
    root_task_id: str | None


_SNAPSHOT_VERSION = 1


def _load_resume_snapshot(path: Path) -> _ResumeSnapshot:
    """Load and validate a resume snapshot. Raises on bad shape."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.get("version")
    if version != _SNAPSHOT_VERSION:
        raise ValueError(f"snapshot version mismatch at {path}: {version!r} != {_SNAPSHOT_VERSION}")
    return _ResumeSnapshot(
        system=raw["system"],
        messages=raw["messages"],
        tool_calls=int(raw["tool_calls"]),
        next_iteration=int(raw["next_iteration"]),
        root_task_id=raw.get("root_task_id"),
    )


def _summarise_assistant_text_for_commit(text: str, iteration: int) -> str:
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
        first_line = "verify passed"
    subject_body = first_line[:72]
    return f"agent6 iter {iteration}: {subject_body}"


@dataclass(frozen=True, slots=True)
class _PromptRevision:
    revised_task: str
    clarifying_questions: tuple[str, ...] = ()


class _PromptRevisionError(Exception):
    """Raised when the optional prompt-revision pass cannot produce a task."""


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 40)].rstrip() + "\n...[truncated for prompt revision]"


def _tag_body(text: str, tag: str) -> str:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    if start == -1:
        return ""
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        return ""
    return text[start:end].strip()


def _parse_prompt_revision(text: str) -> _PromptRevision:
    revised = _tag_body(text, "revised_task") if "<revised_task>" in text else text.strip()
    questions_raw = _tag_body(text, "clarifying_questions")
    questions: list[str] = []
    for raw_line in questions_raw.splitlines():
        line = raw_line.strip().lstrip("-*0123456789. ").strip()
        if not line or line.lower() in {"none", "n/a", "no questions"}:
            continue
        questions.append(line)
    return _PromptRevision(revised_task=revised.strip(), clarifying_questions=tuple(questions[:3]))


def _format_prompt_revision_context(repo: RepoSummary) -> str:
    parts = [
        f"Repository: branch={repo.branch}, head={repo.head_sha[:12]}, files={repo.file_count}",
        f"Top-level: {', '.join(repo.top_level)}",
    ]
    if repo.agents_md:
        parts.append("AGENTS.md:\n" + _clip_text(repo.agents_md, 5000))
    if repo.repo_map:
        parts.append("Repo map:\n" + _clip_text(repo.repo_map, 4000))
    if repo.symbol_outline:
        parts.append("Symbol outline:\n" + _clip_text(repo.symbol_outline, 5000))
    if repo.co_change_pairs:
        lines = "\n".join(f"  {a} <-> {b} ({count})" for a, b, count in repo.co_change_pairs[:15])
        parts.append("Git co-change pairs:\n" + lines)
    if repo.hot_symbols:
        lines = "\n".join(
            f"  {name} ({kind}) at {path}:{line + 1}, {n_files} files"
            for name, kind, path, line, n_files in repo.hot_symbols[:12]
        )
        parts.append("Hot symbols:\n" + lines)
    if repo.recent_log:
        parts.append("Recent commits:\n" + _clip_text(repo.recent_log, 2000))
    return _clip_text("\n\n".join(parts), 20_000)


def _format_effective_task(raw_task: str, revision: _PromptRevision) -> str:
    pieces = [
        "Revised task prompt:",
        revision.revised_task,
        "Original user task (authoritative if anything conflicts):",
        raw_task,
    ]
    if revision.clarifying_questions:
        pieces.extend(
            [
                "Clarifying questions raised by the revision pass:",
                "\n".join(f"- {q}" for q in revision.clarifying_questions),
                (
                    "Proceed under conservative assumptions if these cannot be answered from"
                    " repository context; do not stop solely because questions exist."
                ),
            ]
        )
    return "\n\n".join(pieces)


# A `plan` run injects a one-shot "finish now" directive once its token budget
# drops below this fraction OR it has taken `_PLAN_NUDGE_AFTER_ITERS` turns.
# Verbose reasoning models (Kimi K2.6 observed live) otherwise read forever,
# cheaply, under prompt caching, and never call finish_planning, yielding NO
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
    "[harness verify-settled] Your verify command is passing and your last turns"
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


def _build_system_prompt(
    *,
    config: Config,
    repo: RepoSummary,
    mode: Literal["run", "plan", "ask", "machine", "agent"] = "run",
) -> str:
    """Assemble the system prompt from static blocks + run-specific context.

    The whole system prompt is sent on every turn but gets cached by the
    Anthropic prompt-caching machinery (lineage). Per-turn cost
    after the first call is ~10% of full input rate for the cached prefix.

    ``mode="plan"`` swaps the base block for the planning-mode
    prompt; the verify/metric/repo/co-change/hot-symbols blocks below
    are appended unchanged so the planner sees the same project context
    an executor would.
    """
    base = (
        _ASK_SYSTEM_PROMPT_BASE
        if mode == "ask"
        else _MACHINE_SYSTEM_PROMPT_BASE
        if mode == "machine"
        else _AGENT_SYSTEM_PROMPT_BASE
        if mode == "agent"
        else _PLAN_SYSTEM_PROMPT_BASE
        if mode == "plan"
        else _SYSTEM_PROMPT_BASE
    )
    parts = [base]

    # When the bench harness sets
    # `AGENT6_DISABLE_APPLY_EDIT=1`, apply_edit is filtered out of the
    # tool list. Tell the model so it doesn't try to call a tool that's
    # been removed and waste turns on the resulting `Unknown tool` errors.
    # Plan mode already filters both apply_edit and apply_patch, so the
    # patch-only banner does not apply.
    if mode == "run" and os.environ.get("AGENT6_DISABLE_APPLY_EDIT") == "1":
        parts.append(
            "<patch-only-mode>\n"
            "`apply_edit` has been disabled for this run. The only edit\n"
            "primitive available is `apply_patch` (unified diff). Use it\n"
            "for every change, including file creation (emit a diff with\n"
            "`--- /dev/null` as the source side).\n"
            "</patch-only-mode>\n"
        )

    # Machine-authoring and machine `agent`-state modes have no verify/metric/
    # repo context: those blocks reference tools they aren't given (run_verify /
    # run_metric) and the repo prior only tempts them to spelunk. They just need
    # the budget cap + their base prompt.
    if mode in ("machine", "agent"):
        parts.append(
            _V2_BUDGET_BLOCK_TEMPLATE.format(
                in_cap=config.budget.max_input_tokens,
                out_cap=config.budget.max_output_tokens,
            )
        )
        return "\n".join(parts)

    verify_argv = list(config.workflow.verify_command)
    parts.append(
        _V2_VERIFY_BLOCK_TEMPLATE.format(
            argv=json.dumps(verify_argv),
            timeout_s=config.workflow.verify_timeout_s,
        )
    )

    if config.workflow.metric is not None:
        m = config.workflow.metric
        parts.append(
            _V2_METRIC_BLOCK_TEMPLATE.format(
                argv=json.dumps(list(m.command)),
                pattern=m.pattern,
                goal=m.goal,
            )
        )

    parts.append(
        _V2_BUDGET_BLOCK_TEMPLATE.format(
            in_cap=config.budget.max_input_tokens,
            out_cap=config.budget.max_output_tokens,
        )
    )

    # Structural priors injected directly.
    co_change_block = ""
    if repo.co_change_pairs:
        lines = "\n".join(
            f"  {a} <-> {b}  (changed together {c} times)" for a, b, c in repo.co_change_pairs[:20]
        )
        co_change_block = (
            "Git co-change pairs (files that historically change together;"
            " consider when editing one of these):\n"
            f"{lines}\n\n"
        )

    hot_symbols_block = ""
    if repo.hot_symbols:
        lines = "\n".join(
            f"  {name} ({kind}) at {path}:{line + 1}, referenced across {n_files} files"
            for name, kind, path, line, n_files in repo.hot_symbols[:15]
        )
        hot_symbols_block = (
            "Hot symbols (cross-file reference hot spots from static analysis;"
            " changing one of these forces edits across the listed file count):\n"
            f"{lines}\n\n"
        )

    repo_map_block = ""
    if repo.repo_map:
        repo_map_block = f"Repo map (tracked files grouped by directory):\n{repo.repo_map}\n\n"

    symbol_outline_block = ""
    if repo.symbol_outline:
        symbol_outline_block = (
            "Symbol outline (top-level defs per file from the tree-sitter index;"
            " line numbers are 1-based):\n"
            f"{repo.symbol_outline}\n\n"
        )

    parts.append(
        _V2_REPO_BLOCK_TEMPLATE.format(
            branch=repo.branch,
            head_sha=repo.head_sha[:12] or "(no commits yet)",
            file_count=repo.file_count,
            top_level=", ".join(repo.top_level),
            agents_md=repo.agents_md or "(empty)",
            repo_map_block=repo_map_block,
            symbol_outline_block=symbol_outline_block,
            co_change_block=co_change_block,
            hot_symbols_block=hot_symbols_block,
            recent_log=repo.recent_log or "(none)",
        )
    )

    return "\n".join(parts)


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
        # Track consecutive before_finish rejections so a
        # stubborn worker can't burn the budget bouncing off the critic.
        consecutive_critic_rejections = 0
        metric_history: list[_MetricSample] = []
        # degenerate-loop guard. Some open-weights models (observed
        # with Kimi K2.6 on the perf takehome) latch onto the same tool
        # call with identical arguments and re-issue it dozens of times
        # in a row, getting byte-identical results and emitting only
        # whitespace deltas between turns. We detect a back-to-back
        # streak of the same (tool_name, args) signature and inject a
        # one-shot text block into the next user turn telling the worker
        # the result has not changed and to pivot. Reset on any signature
        # change, so a normal re-read after edits does not trigger.
        last_tool_signature: str | None = None
        repeat_streak: int = 0
        repeat_warning_emitted_at: int = 0
        # went_quiet nudge counter. See `went_quiet_max_nudges`.
        went_quiet_nudges_used: int = 0
        # metric-plateau pivot-nudge counter. See `_METRIC_PLATEAU_PATIENCE`.
        plateau_nudges_used: int = 0
        # metric-run early-finish rejection counter. See
        # `_METRIC_EARLY_FINISH_PATIENCE`.
        metric_finish_nudges_used: int = 0
        # plan-mode finish nudge: fire once per loop segment when a verbose
        # planner (Kimi K2.6 observed live) keeps reading without ever calling
        # finish_planning. Like the sibling nudge counters this is loop-local, so
        # a resume starts a fresh segment and may nudge again, benign (one extra
        # harmless directive); the turn count is measured from start_iteration.
        plan_finish_nudged = False
        # verify-settled completion (run mode). Once verify has passed at least
        # once, count consecutive iterations that make no progress (no new
        # commit + no edit). After a couple, nudge the worker to finish_run;
        # if it keeps spinning, stop. This is the positive completion signal a
        # non-metric run otherwise lacks, a weak worker (Kimi K2.6 observed:
        # 128 iters when done at ~45) keeps re-running commands after success.
        verify_ever_passed = False
        verify_settled_idle = 0
        verify_settled_nudged = False
        run_budget_nudged = False
        for iteration in range(start_iteration, self.max_iterations + 1):
            self._maybe_compact(messages)

            # Force a verbose planner to land a plan. Trigger on EITHER a low
            # token budget OR too many planning turns, with prompt caching a
            # planner can take many cheap turns, so an iteration cap is the
            # reliable lever for the "reads forever" failure mode. A rough
            # delivered plan beats an exhaustive one that never gets emitted.
            if self.mode == "plan" and not plan_finish_nudged:
                remaining = self._budget_fraction_remaining()
                low_budget = remaining is not None and remaining <= _PLAN_BUDGET_NUDGE_BELOW
                too_many_turns = iteration - start_iteration + 1 >= _PLAN_NUDGE_AFTER_ITERS
                if low_budget or too_many_turns:
                    plan_finish_nudged = True
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
                and not run_budget_nudged
                and _metric_goal(self.config.workflow.metric) is None
            ):
                remaining = self._budget_fraction_remaining()
                if remaining is not None and remaining <= _RUN_BUDGET_NUDGE_BELOW:
                    run_budget_nudged = True
                    messages.append(
                        {"role": "user", "content": [{"type": "text", "text": _RUN_BUDGET_NUDGE}]}
                    )
                    self._log(f"LOOP: run budget-nudge at iter {iteration}")
                    self._emit(
                        "loop.run_budget.nudge", iteration=iteration, budget_remaining=remaining
                    )

            # Snapshot BEFORE the LLM call. After this write, a
            # crash anywhere up to the next iteration's snapshot can be
            # resumed by re-running this same call.
            self._save_resume_snapshot(
                system=system,
                messages=messages,
                tool_calls=tool_calls,
                next_iteration=iteration,
                root_task_id=root_task_id,
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
                    tool_calls=tool_calls,
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
                    tool_calls=tool_calls,
                )

            # Reconstruct the assistant message exactly from the response
            # content blocks so tool_use IDs round-trip cleanly.
            assistant_blocks = resp.raw.get("content") or []
            messages.append({"role": "assistant", "content": assistant_blocks})

            if not resp.tool_uses:
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
                    if self.critic_mode == "before_finish" and self.critic_provider is not None:
                        critique = self._run_critic(
                            task=original_task,
                            messages=messages,
                            trigger="before_finish",
                            iteration=iteration,
                        )
                        cap = self.max_consecutive_critic_rejections
                        cap_reached = cap > 0 and consecutive_critic_rejections >= cap
                        if critique is not None and not critique.satisfied and not cap_reached:
                            self._log(f"  critic rejected silent_finish at iter {iteration}")
                            self._emit(
                                "loop.critic.rejected_silent_finish",
                                iteration=iteration,
                            )
                            consecutive_critic_rejections += 1
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
                            continue
                        if critique is not None and not critique.satisfied and cap_reached:
                            self._log(
                                f"  critic rejected silent_finish at"
                                f" iter {iteration} but rejection cap"
                                f" ({cap}) reached - accepting finish"
                            )
                            self._emit(
                                "loop.critic.rejection_cap_reached",
                                iteration=iteration,
                                rejections=consecutive_critic_rejections,
                            )
                            consecutive_critic_rejections = 0
                        elif critique is not None:
                            self._log("  critic approved silent_finish")
                            consecutive_critic_rejections = 0
                    # metric-run early-finish guard, mirroring the finish_run
                    # path: a silent finish on an optimisation run with budget to
                    # spare should be nudged to keep optimising rather than
                    # accepted. Without this, dropping tool_use was a way to skip
                    # the plateau/early-finish policy entirely.
                    if (
                        self.mode == "run"
                        and _metric_goal(self.config.workflow.metric) is not None
                        and not self._metric_at_ceiling(metric_history)
                    ):
                        finish_budget_remaining = self._budget_fraction_remaining()
                        has_runway = (
                            finish_budget_remaining is not None
                            and finish_budget_remaining > _METRIC_PLATEAU_STOP_BELOW_BUDGET
                        )
                        if has_runway and metric_finish_nudges_used < _METRIC_EARLY_FINISH_PATIENCE:
                            assert finish_budget_remaining is not None
                            metric_finish_nudges_used += 1
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": _METRIC_FINISH_NUDGE}],
                                }
                            )
                            self._log(
                                f"  metric early-finish (silent) rejected"
                                f" #{metric_finish_nudges_used} at iter {iteration}"
                                f" (budget {finish_budget_remaining:.0%} left)"
                            )
                            self._emit(
                                "loop.metric_early_finish.rejected",
                                iteration=iteration,
                                nudges_used=metric_finish_nudges_used,
                                budget_remaining=finish_budget_remaining,
                                trigger="silent_finish",
                            )
                            continue
                    self._log(
                        f"LOOP: silent_finish at iter {iteration} - "
                        f"agent emitted text but no tool_use"
                    )
                    self._emit(
                        "run.end",
                        reason="silent_finish",
                        iterations=iteration,
                        all_passed=True,
                    )
                    return RunResult(
                        completed=True,
                        reason="silent_finish",
                        # In ask mode the final prose IS the answer the caller
                        # prints, so keep it whole; run/plan only need a short
                        # summary line.
                        summary=text if self.mode == "ask" else text[:1000],
                        iterations=iteration,
                        tool_calls=tool_calls,
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
                starved = (
                    resp.stop_reason == "length" and reasoning_chars > 0 and resp.output_tokens > 0
                )
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
                self._log(
                    f"LOOP: went_quiet at iter {iteration} - agent emitted no text and no tool_use"
                )
                # nudge-and-retry instead of immediate exit.
                # Weak open-weights models occasionally emit a single
                # empty assistant turn mid-run; a one-line synthetic
                # user prompt almost always gets them back on track,
                # and costs ~50 input tokens vs aborting the entire run.
                # AGENT6_WENT_QUIET_MAX_NUDGES env override.
                env_max = os.environ.get("AGENT6_WENT_QUIET_MAX_NUDGES", "").strip()
                if env_max.isdigit():
                    effective_max_nudges = int(env_max)
                else:
                    effective_max_nudges = self.went_quiet_max_nudges
                if went_quiet_nudges_used < effective_max_nudges:
                    went_quiet_nudges_used += 1
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
                    messages.append(
                        {"role": "user", "content": [{"type": "text", "text": nudge_text}]}
                    )
                    self._emit(
                        "loop.went_quiet.nudge",
                        iteration=iteration,
                        nudges_used=went_quiet_nudges_used,
                        nudges_max=effective_max_nudges,
                    )
                    continue
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
                    tool_calls=tool_calls,
                )

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
            went_quiet_nudges_used = 0
            for tu in resp.tool_uses:
                tool_calls += 1
                name = tu.get("name", "")
                tool_input = tu.get("input", {})
                tu_id = tu.get("id", "")
                # degenerate-loop signature tracking. Stable
                # JSON so dict key order does not break equality. Same
                # (name, args) back-to-back across iterations increments
                # `repeat_streak`; anything else resets it.
                try:
                    sig = f"{name}:{json.dumps(tool_input, sort_keys=True, ensure_ascii=False)}"
                except (TypeError, ValueError):
                    sig = f"{name}:<unhashable>"
                if sig == last_tool_signature:
                    repeat_streak += 1
                else:
                    last_tool_signature = sig
                    repeat_streak = 1
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
                    elif name == "run_metric_command":
                        if verify_just_passed:
                            metric_called_after_verify_pass = True
                        metric_feedback_text = self._record_metric_result(
                            metric_history,
                            result,
                            iteration=iteration,
                            label=f"manual iter {iteration}",
                            sha="",
                        )
                        if verify_just_passed:
                            metric_plateau_finish = self._metric_plateau_summary(metric_history)
                    if name in ("apply_edit", "apply_patch"):
                        edited_this_iter = True
                    if name in _DAG_MUTATING_TOOLS:
                        dag_mutated_this_iter = True  # snapshot once after the turn
                except ToolError as exc:
                    content = json.dumps({"error": str(exc)})
                    self._log(f"  tool_error: {name}: {exc}")
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

            # Auto-commit on verify pass, AFTER the tool_results so the
            # commit message reflects the iteration number. Best-effort:
            # commit failures (e.g. nothing to commit) are logged but
            # don't abort the run. The catch also handles OSError
            # (subprocess failures) so a transient FS hiccup doesn't kill
            # an otherwise-fine run.
            # Plan mode is read-only; never auto-commit.
            if verify_just_passed and self.mode == "run":
                commit_subject = _summarise_assistant_text_for_commit(resp.text or "", iteration)
                sha = ""
                try:
                    sha = commit_all(
                        self.root,
                        commit_subject,
                    )
                    self._log(f"  auto-commit: {sha[:12]}")
                    self._emit("loop.auto_commit", iteration=iteration, sha=sha)
                    committed_this_iter = bool(sha)
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
                        self._emit(
                            "run.end",
                            reason="interactive_stop",
                            iterations=iteration,
                            all_passed=True,
                        )
                        return RunResult(
                            completed=True,
                            reason="interactive_stop",
                            summary=f"stopped interactively after iter {iteration}",
                            iterations=iteration,
                            tool_calls=tool_calls,
                        )
                if not metric_called_after_verify_pass:
                    metric_feedback_text = self._auto_metric_feedback(
                        metric_history,
                        iteration=iteration,
                        sha=sha,
                    )
                    metric_plateau_finish = self._metric_plateau_summary(metric_history)

            # critic-in-loop triggers.
            #   on_verify_fail - the verify just failed; surface a
            #                    critique alongside the failure so the
            #                    worker has a second opinion before its
            #                    next edit.
            #   periodic       - every critic_period iterations.
            #   before_finish  - handled below, after finish_signal is
            #                    inspected, because it can revoke finish.
            critic_text: str | None = None
            if (
                self.critic_mode == "on_verify_fail"
                and verify_just_failed
                and self.critic_provider is not None
            ):
                critique = self._run_critic(
                    task=original_task,
                    messages=messages,
                    trigger="verify_failed",
                    iteration=iteration,
                )
                if critique is not None:
                    critic_text = critique.text
            elif (
                self.critic_mode == "periodic"
                and self.critic_provider is not None
                and iteration % max(1, self.critic_period) == 0
            ):
                critique = self._run_critic(
                    task=original_task,
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
                and self.critic_provider is not None
            ):
                critique = self._run_critic(
                    task=original_task,
                    messages=messages,
                    trigger="before_finish",
                    iteration=iteration,
                )
                cap = self.max_consecutive_critic_rejections
                cap_reached = cap > 0 and consecutive_critic_rejections >= cap
                if critique is not None and not critique.satisfied and not cap_reached:
                    self._log(f"  critic rejected finish_run at iter {iteration}")
                    self._emit("loop.critic.rejected_finish", iteration=iteration)
                    finish_signal = None
                    finish_payload = None
                    consecutive_critic_rejections += 1
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
                        rejections=consecutive_critic_rejections,
                    )
                    critic_text = (
                        "The critic flagged issues but the rejection cap was"
                        " reached; finish_run will be accepted. Critique:\n\n" + critique.text
                    )
                    consecutive_critic_rejections = 0
                elif critique is not None:
                    self._log("  critic approved finish_run")
                    consecutive_critic_rejections = 0

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
                and not self._metric_at_ceiling(metric_history)
            ):
                finish_budget_remaining = self._budget_fraction_remaining()
                has_runway = (
                    finish_budget_remaining is not None
                    and finish_budget_remaining > _METRIC_PLATEAU_STOP_BELOW_BUDGET
                )
                if has_runway and metric_finish_nudges_used < _METRIC_EARLY_FINISH_PATIENCE:
                    assert finish_budget_remaining is not None
                    metric_finish_nudges_used += 1
                    finish_signal = None
                    finish_payload = None
                    tool_results.append({"type": "text", "text": _METRIC_FINISH_NUDGE})
                    self._log(
                        f"  metric early-finish rejected #{metric_finish_nudges_used}"
                        f" at iter {iteration} (budget {finish_budget_remaining:.0%} left)"
                    )
                    self._emit(
                        "loop.metric_early_finish.rejected",
                        iteration=iteration,
                        nudges_used=metric_finish_nudges_used,
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
            if repeat_streak >= repeat_threshold and repeat_warning_emitted_at < iteration - 1:
                # Strip the args-JSON suffix for the user-facing text.
                latched_name = (last_tool_signature or "").split(":", 1)[0] or "<unknown>"
                notice = (
                    f"[loop-guard] You have called `{latched_name}` with"
                    f" identical arguments {repeat_streak} times in a row."
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
                    streak=repeat_streak,
                )
                self._log(
                    f"  loop-guard: {latched_name} called"
                    f" {repeat_streak}x in a row - injecting notice"
                )
                repeat_warning_emitted_at = iteration

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
                if self._metric_at_ceiling(metric_history):
                    plateau_should_stop = True
                    self._emit("loop.metric_ceiling.stop", iteration=iteration)
                elif in_final_slice and plateau_nudges_used >= _METRIC_PLATEAU_PATIENCE:
                    plateau_should_stop = True
                else:
                    plateau_nudges_used += 1
                    nudge_text = _metric_plateau_nudge(budget_remaining)
                    tool_results.append({"type": "text", "text": nudge_text})
                    budget_note = (
                        "n/a" if budget_remaining is None else f"{budget_remaining:.0%} left"
                    )
                    self._log(
                        "  metric_plateau pivot-nudge"
                        f" #{plateau_nudges_used} at iter {iteration} (budget {budget_note})"
                    )
                    self._emit(
                        "loop.metric_plateau.nudge",
                        iteration=iteration,
                        nudges_used=plateau_nudges_used,
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
                verify_ever_passed = True
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
            if non_metric_run and verify_ever_passed:
                made_progress = committed_this_iter or edited_this_iter or self._worktree_dirty()
                if made_progress:
                    verify_settled_idle = 0
                    verify_settled_nudged = False  # a fresh idle streak may re-nudge
                elif not (verify_just_passed or verify_just_failed):
                    verify_settled_idle += 1
            verify_settled_stop = (
                non_metric_run
                and finish_signal is None
                and verify_ever_passed
                and verify_settled_idle >= _VERIFY_SETTLED_STOP_AFTER
            )
            if (
                non_metric_run
                and finish_signal is None
                and not verify_settled_stop
                and verify_ever_passed
                and verify_settled_idle >= _VERIFY_SETTLED_NUDGE_AFTER
                and not verify_settled_nudged
            ):
                verify_settled_nudged = True
                tool_results.append({"type": "text", "text": _VERIFY_SETTLED_NUDGE})
                self._emit(
                    "loop.verify_settled.nudge", iteration=iteration, idle=verify_settled_idle
                )

            messages.append({"role": "user", "content": tool_results})

            if verify_settled_stop:
                self._log(f"LOOP: verify_settled at iter {iteration} (idle {verify_settled_idle})")
                self._emit(
                    "run.end", reason="verify_settled", iterations=iteration, all_passed=True
                )
                return RunResult(
                    completed=True,
                    reason="verify_settled",
                    summary="verify passed and the worker stopped making changes",
                    iterations=iteration,
                    tool_calls=tool_calls,
                )

            if plateau_should_stop:
                assert metric_plateau_finish is not None
                self._log(f"LOOP: metric_plateau at iter {iteration}")
                self._emit(
                    "run.end",
                    reason="metric_plateau",
                    iterations=iteration,
                    all_passed=True,
                )
                return RunResult(
                    completed=True,
                    reason="metric_plateau",
                    summary=metric_plateau_finish,
                    iterations=iteration,
                    tool_calls=tool_calls,
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
                and repeat_streak >= self.loop_guard_kill_threshold
            ):
                latched_name = (last_tool_signature or "").split(":", 1)[0] or "<unknown>"
                self._log(
                    f"LOOP: loop_guard_killed at iter {iteration} -"
                    f" {latched_name} called {repeat_streak}x in a row"
                    f" (threshold={self.loop_guard_kill_threshold})"
                )
                self._emit(
                    "run.end",
                    reason="loop_guard_killed",
                    iterations=iteration,
                    all_passed=False,
                    tool=latched_name,
                    streak=repeat_streak,
                )
                return RunResult(
                    completed=False,
                    reason="loop_guard_killed",
                    summary=(
                        f"loop-guard killed run: `{latched_name}`"
                        f" called {repeat_streak}x in a row with"
                        f" identical arguments (threshold"
                        f" {self.loop_guard_kill_threshold})"
                    ),
                    iterations=iteration,
                    tool_calls=tool_calls,
                )

            if finish_signal is not None:
                self._log(f"LOOP: {finish_kind} called at iter {iteration}")
                self._emit(
                    "run.end",
                    reason=finish_kind,
                    iterations=iteration,
                    all_passed=True,
                )
                return RunResult(
                    completed=True,
                    reason=finish_kind,
                    summary=finish_signal,
                    iterations=iteration,
                    tool_calls=tool_calls,
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
                    tool_calls=tool_calls,
                )

        self._log(f"LOOP: max_iterations={self.max_iterations} reached")
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
            tool_calls=tool_calls,
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
        base = load_repo_summary(self.root)
        try:
            hot = tuple(self.dispatcher.hot_symbols(max_symbols=20, min_files_referenced=2))
        except (BudgetExceeded, KeyboardInterrupt):
            raise
        except Exception:
            hot = ()
        try:
            co_change = tuple(git_co_change_pairs(self.root, n_commits=200))
        except (BudgetExceeded, KeyboardInterrupt):
            raise
        except Exception:
            co_change = ()
        try:
            symbol_outline = _build_symbol_outline_block(
                self.dispatcher.file_outlines(),
                root=self.root,
            )
        except (BudgetExceeded, KeyboardInterrupt):
            raise
        except Exception:
            symbol_outline = ""
        return RepoSummary(
            root=base.root,
            branch=base.branch,
            head_sha=base.head_sha,
            file_count=base.file_count,
            top_level=base.top_level,
            agents_md=base.agents_md,
            recent_log=base.recent_log,
            co_change_pairs=co_change,
            hot_symbols=hot,
            repo_map=base.repo_map,
            symbol_outline=symbol_outline,
        )

    def _save_resume_snapshot(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tool_calls: int,
        next_iteration: int,
        root_task_id: str | None,
    ) -> None:
        """Write loop state to disk for resume.

        Called before each LLM call. Atomic via tmp-file + replace so a
        crash mid-write leaves the prior snapshot intact. No-op if
        ``resume_state_path`` is None (e.g. unit tests).
        """
        if self.resume_state_path is None:
            return
        payload = {
            "version": _SNAPSHOT_VERSION,
            "system": system,
            "messages": messages,
            "tool_calls": tool_calls,
            "next_iteration": next_iteration,
            "root_task_id": root_task_id,
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
