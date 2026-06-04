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
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agent6.budget import BudgetExceeded, BudgetTracker
from agent6.config import Config
from agent6.git_ops import GitError, commit_all
from agent6.git_ops import co_change_pairs as git_co_change_pairs
from agent6.git_ops import status as git_status
from agent6.graph.client import CuratorClientError, GraphClient
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft
from agent6.providers import Provider, ProviderError, ToolDefinition
from agent6.tools.dispatch import ToolDispatcher, ToolError
from agent6.tools.index import Symbol
from agent6.tools.schema import (
    ALL_TOOLS,
    LOOP_EXTRA_TOOLS,
    PLAN_EXTRA_TOOLS,
    ApplyEditInput,
    ApplyPatchInput,
    FinishPlanningInput,
    FinishRunInput,
)
from agent6.types import RepoSummary
from agent6.workflows._context import load_repo_summary

if TYPE_CHECKING:
    from agent6.events import EventSink


_ELISION_PLACEHOLDER = (
    "<elided by context compaction: this tool_result has been replaced "
    "with this short marker to keep the loop's cumulative input bounded. "
    "Re-call the tool with the same args if you still need the content.>"
)

# per-tool-result cap. was a hard 20_000 char slice
# applied mid-JSON, which produced a malformed result the model could
# not parse. Weak models (Kimi K2.6 observed live) then concluded the
# tool result was "cut off" and re-called `read_file` repeatedly trying
# to see the rest, latching the loop-guard. The fix: lift the cap to
# 60_000 chars (~15k tokens, comfortably fits most source files) AND
# when truncation is unavoidable, wrap the result in a fresh,
# well-formed JSON object that explicitly tells the model what
# happened and how to get the rest.
_TOOL_RESULT_CHAR_CAP = 60_000

# HTTP statuses that will never succeed on a blind retry of the same request.
# 401/403 auth, 402 insufficient credits, 404 bad model/endpoint, 422 malformed
# body. Retrying these only burns wall-time (observed live: a 402 "Insufficient
# credits" was retried on every turn for the rest of the run). 408/409/429 and
# all 5xx remain retryable and fall through to the normal backoff.
_NON_RETRYABLE_HTTP_STATUSES = frozenset({400, 401, 402, 403, 404, 422})


def _cap_tool_result(content: str, *, tool_name: str) -> str:
    """Cap a serialized tool_result payload at ``_TOOL_RESULT_CHAR_CAP``
    chars without producing malformed JSON. If the payload is over the
    cap, wrap it in a new JSON envelope that tells the model:
    (a) the result was truncated, (b) how many chars were shown vs
    total, (c) the head of the original content, (d) actionable next
    steps. This prevents weak models from inferring "the tool itself
    returned a partial result, let me call it again"."""
    if len(content) <= _TOOL_RESULT_CHAR_CAP:
        return content
    head_budget = _TOOL_RESULT_CHAR_CAP - 1_000  # reserve room for envelope
    head = content[:head_budget]
    if tool_name == "read_file":
        guidance = (
            "Use `read_file` again with `offset` and `limit` to read the rest"
            " of the file in chunks. Do NOT re-call with identical arguments"
            " expecting a different result - you will get the same truncated"
            " head and waste budget."
        )
    elif tool_name in ("run_command", "run_verify_command"):
        guidance = (
            "Re-run with a narrower scope (e.g. a single test, smaller grep"
            " pattern, head/tail) to get a result that fits. Do NOT re-call"
            " with identical arguments expecting different output."
        )
    else:
        guidance = (
            "Re-call with arguments that produce less output. Do NOT re-call"
            " with identical arguments expecting different output."
        )
    return json.dumps(
        {
            "_tool_result_truncated": True,
            "tool": tool_name,
            "shown_chars": len(head),
            "total_chars": len(content),
            "head": head,
            "guidance": guidance,
        },
        ensure_ascii=False,
    )


def _compact_old_tool_results(
    messages: list[dict[str, Any]],
    *,
    max_total_bytes: int,
    keep_recent: int = 2,
) -> int:
    """Elide old tool_result blocks once cumulative content exceeds the
    threshold. Walks messages oldest-first, replaces each tool_result's
    ``content`` with a short placeholder, stops once total size is back
    under ``max_total_bytes``. The most recent ``keep_recent`` are always
    preserved. Idempotent on already-elided entries. Returns the number
    of entries elided (for telemetry).
    """
    pointers: list[tuple[int, int, int]] = []  # (msg_idx, item_idx, size)
    total = 0
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item_idx, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            if item.get("type") != "tool_result":
                continue
            raw_content = item.get("content")
            size = len(raw_content) if isinstance(raw_content, str) else len(str(raw_content))
            pointers.append((msg_idx, item_idx, size))
            total += size

    if total <= max_total_bytes or len(pointers) <= keep_recent:
        return 0
    elided_count = 0
    for msg_idx, item_idx, size in pointers[:-keep_recent]:
        if total <= max_total_bytes:
            break
        item = messages[msg_idx]["content"][item_idx]
        current = item.get("content")
        if isinstance(current, str) and current.startswith("<elided by context compaction"):
            continue
        item["content"] = _ELISION_PLACEHOLDER
        total -= size - len(_ELISION_PLACEHOLDER)
        elided_count += 1
    return elided_count


# cache_control marker on the initial user message
# so the system + initial context get cached across the loop's turns.
_CACHE_CONTROL_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}

# compaction thresholds (chars, not tokens - approximate; tokens
# are roughly chars/4 for English-shaped content).
_DROP_BLOCKS_AT_CHARS = 256_000  # ~64k tokens of tool_result content
_SUMMARISE_AT_CHARS = 768_000  # ~192k tokens: full context restart


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


_SYSTEM_PROMPT_BASE = """<role>
You are agent6, a sandboxed coding agent. You receive a task in the
first user message, plan and execute changes in this repository, verify
them, and finish when done or when your compute budget runs out.

Your harness gives you tools to read, search, edit, run commands, run
the verify command, and (if configured) measure a continuous-score
metric. The harness is also tracking your spend against a hard budget;
the loop will halt if you exceed it.
</role>

<edit-rules>
- `apply_edit`: each edit's `old_string` MUST occur EXACTLY ONCE in the
  file (whitespace, indentation byte-for-byte). Use `kind="create"` for
  new files (empty `old_string`, full content in `new_string`).
- `apply_patch`: standard unified-diff (`--- a/PATH`, `+++ b/PATH`,
  `@@ -L,N +L,N @@` hunks). Use this for multi-hunk edits to one file -
  cheaper than several `apply_edit` calls.
- Stay inside the files the task asks you to change. Drive-by refactors
  and "while I'm here" cleanups produce review failures and waste budget.
- NEVER leave TODOs, "implement this later" comments, ellipses, or stub
  bodies (`pass`, `raise NotImplementedError`) in place of real code.
</edit-rules>

<tool-use-rules>
- Anchor reads: prefer `outline` to see file shape before `read_file`.
- For symbol queries prefer `find_definition` / `find_references` over
  plain `grep` (those exclude strings/comments).
- After every meaningful edit run `run_verify_command` to check
  correctness. Don't chain many edits without a verify pass; each
  uncommitted-but-broken edit cost compounds.
- If an edit fails verify and you need to revert it, do NOT call
    `git checkout`, `git reset`, or other history-mutating git commands
    through `run_command`: `.git/` is protected inside the jail and those
    calls will fail. Instead read the previous content with a read-only
    command such as `git show HEAD:path/to/file` and use `apply_patch` /
    `apply_edit` to restore the file, or manually undo the bad hunk.
- The harness AUTO-COMMITS after every verify-pass. You don't need to
  `git commit` manually - score is computed against the latest commit
  on this branch and the workflow's git-history rescue picks the
  best-scoring commit at the end. If you DO want a specific commit
  message you can still call `run_command` with `git commit`, but
  it's optional.
- `finish_run` is the only way to terminate cleanly. Call it when the
  task is done, when the metric plateaued, or when you are blocked.
</tool-use-rules>

<dag-rules>
The DAG-as-tool surface (`add_task`, `update_task`, `set_cursor`,
`list_tasks`) maintains a persistent task breakdown. OPTIONAL - skip
it entirely for one-shot fixes, single-file edits, or perf-takehome-
style "make this number smaller" runs. Use it ONLY when the task
naturally decomposes into 3+ subtasks worth tracking and humans
watching the TUI benefit from seeing the breakdown.

When you do use it: `add_task(title, parent_id?)` returns an id;
`update_task(id, status="in_progress")` when you start a subtask;
`update_task(id, status="passed")` only after verify confirms it.
`set_cursor(id)` is cosmetic - it updates the TUI's "current task"
pointer; it is NOT the resume mechanism (the workflow snapshots its
own state independently before each LLM call).
</dag-rules>

<scope-and-style>
Project conventions live in AGENTS.md (read it if present). Defaults:
minimum-necessary edits matching the file's existing style. Tests are
the authoritative behavioural specification - if a test says X must
happen, match that behaviour even if a docstring says otherwise.

When the task is to ADD behaviour (not fix a regression in code that
already had a test), prefer the TDD loop: write or extend a test that
encodes the desired behaviour FIRST, run `run_verify_command` to
confirm it fails for the right reason, THEN implement the change and
re-run verify. This catches "fixed the symptom but not the bug" and
gives the harness a concrete signal to commit against. Skip this only
when the existing test suite already exercises the change point or
when no test framework is in scope (one-shot script edits, perf
takehomes that already ship a metric).
</scope-and-style>
"""

# Alternate base system prompt used by `agent6 plan`. Replaces
# the edit-/verify-/dag-/style-rules blocks with planning-mode rules.
# The verify and metric blocks below are still appended unchanged so the
# planner can call `run_verify_command` to confirm the verify chain is
# wired and `run_metric_command` (when configured) to baseline a score.
_PLAN_SYSTEM_PROMPT_BASE = """<role>
You are agent6 in PLAN mode, a sandboxed planning agent. You receive a
task in the first user message; your job is to PLAN how to execute it,
not to execute it. You will read what you need, optionally run commands
to confirm assumptions (verify chain, dependency probes, etc.), and
then emit a written plan via `finish_planning`.

You CANNOT edit files in this mode: `apply_edit`, `apply_patch`, and
any commit-related tools are not exposed. If the planning task seems
to require a small write to confirm an assumption, note the assumption
in the plan and leave verification for the execution pass.
</role>

<tool-use-rules>
- Anchor reads: prefer `outline` to see file shape before `read_file`.
- For symbol queries prefer `find_definition` / `find_references` over
  plain `grep` (those exclude strings/comments).
- `run_verify_command` is allowed and encouraged: a baseline verify run
  proves the chain works and surfaces existing failures the executor
  should not be blamed for.
- `run_command` is allowed for read-only probes (`ls`, `cat`, `git log`,
  dependency-version checks, etc.). Do not invoke anything that mutates
  the working tree.
- The DAG-as-tool surface (`add_task`, `update_task`, `set_cursor`,
  `list_tasks`) is exposed and useful as a scratchpad while you plan,
  but the FINAL deliverable is the markdown you pass to
  `finish_planning` - not the DAG. The execution run started later via
  `agent6 run --from-plan` will build its own DAG from the plan text.
</tool-use-rules>

<plan-output>
The plan you pass to `finish_planning(plan_markdown=...)` is the single
artefact this whole pass produces. It is written to
`.agent6/runs/<run-id>/plan.md` and consumed verbatim by
`agent6 run --from-plan <run-id>` (which feeds it as the new run's
task description). Suggested skeleton:

```
# Plan: <one-line title>

## Original task
<the user's task verbatim>

## Context discovered
<short prose: relevant files, existing patterns, constraints>

## Tasks
1. <imperative title>
   - Files: <paths>
   - Acceptance: <verifiable condition>
2. <imperative title>
   - ...

## Open questions
> **Q:** <question for the operator>
> **A:**

## Verification approach
<which verify commands / metric calls confirm success>
```

Include `## Open questions` only when there are real ambiguities the
operator must resolve before execution. Leave the `**A:**` lines blank
- the operator fills them in via `agent6 plan --edit <run-id>`.

Call `finish_planning` exactly once when the plan is complete. Do not
call any other tools after `finish_planning`.
</plan-output>
"""

_V2_VERIFY_BLOCK_TEMPLATE = """<verify-command>
This run's verify_command (call via `run_verify_command`):
  argv: {argv}
  timeout: {timeout_s}s

Returncode 0 means the change passes verify. Non-zero means the change
broke something - either revert it or fix the regression before
proceeding. The timeout is set to catch infinite-loop / quadratic edits
early; if verify legitimately needs longer, the operator misconfigured
the timeout.
</verify-command>
"""

_V2_METRIC_BLOCK_TEMPLATE = """<metric-command>
This run has a continuous-score metric (call via `run_metric_command`):
  argv: {argv}
  pattern: {pattern}
  goal: {goal}

After every verify-passing edit, the harness automatically runs this
metric command and injects a compact `[harness metric]` block into the
next turn with latest score, best score, trajectory, and a verdict. You
may also call `run_metric_command` manually when probing a specific idea.
After enough metric samples, a verified edit that only ties the existing
best may finish the run automatically to preserve performance per dollar.

Metric work discipline: keep changes that improve the score AND preserve
correctness; revert anything that doesn't. Prefer cheap local experiments
and measured edits over long speculation. When the `[harness metric]`
verdict says the latest edit is flat/worse, restore the prior best or
pivot to a different bottleneck instead of polishing the same approach.
When the score plateaus despite several distinct edits, call `finish_run`.
</metric-command>
"""

_V2_BUDGET_BLOCK_TEMPLATE = """<budget-awareness>
Hard caps: max_input_tokens={in_cap}, max_output_tokens={out_cap}.
The loop will halt if either is exceeded. Track your spend - tool
results contribute to input on every subsequent turn (they get
re-sent in the conversation), so prefer narrow `read_file` ranges
and specific `grep` patterns over broad reads.
</budget-awareness>
"""

_V2_REPO_BLOCK_TEMPLATE = """<repo-priors>
Repository: branch={branch}, head={head_sha}, files={file_count}
Top-level: {top_level}

{repo_map_block}{symbol_outline_block}AGENTS.md (project conventions):
{agents_md}

{co_change_block}{hot_symbols_block}Recent commits:
{recent_log}
</repo-priors>
"""


_SYMBOL_OUTLINE_MAX_CHARS = 8000
_SYMBOL_OUTLINE_MAX_FILES = 120
_SYMBOL_OUTLINE_MAX_PER_FILE = 12
_SYMBOL_OUTLINE_KIND_PRIORITY: tuple[str, ...] = (
    "class",
    "struct",
    "enum",
    "trait",
    "interface",
    "function",
    "method",
)


def _build_symbol_outline_block(
    outlines: dict[Path, list[Symbol]],
    *,
    root: Path,
) -> str:
    """Format the per-file symbol outline into a compact prompt block.

    Layout::

        path/to/file.py:
          class Foo:12
          function bar:30
          ...
        path/to/other.rs:
          struct Bar:5

    Hard caps keep the block bounded:
      - At most ``_SYMBOL_OUTLINE_MAX_PER_FILE`` rows per file (truncated
        with a ``... (+N more)`` line).
      - At most ``_SYMBOL_OUTLINE_MAX_FILES`` files (overflow summarised).
      - At most ``_SYMBOL_OUTLINE_MAX_CHARS`` characters total; we stop
        emitting files as soon as the budget would be exceeded.

    Returns an empty string when ``outlines`` is empty.
    """
    if not outlines:
        return ""
    root_resolved = root.resolve()
    rel_entries: list[tuple[str, list[Symbol]]] = []
    for path, syms in outlines.items():
        if not syms:
            continue
        try:
            rel = path.resolve().relative_to(root_resolved)
            rel_str = str(rel)
        except ValueError:
            continue
        rel_entries.append((rel_str, syms))
    rel_entries.sort(key=lambda t: t[0])

    rows: list[str] = []
    total = 0
    files_emitted = 0
    for files_emitted, (rel_str, syms) in enumerate(rel_entries):
        if files_emitted >= _SYMBOL_OUTLINE_MAX_FILES:
            remaining = len(rel_entries) - files_emitted
            rows.append(f"... ({remaining} more files)")
            break
        kept = sorted(
            syms,
            key=lambda s: (
                _SYMBOL_OUTLINE_KIND_PRIORITY.index(s.kind)
                if s.kind in _SYMBOL_OUTLINE_KIND_PRIORITY
                else len(_SYMBOL_OUTLINE_KIND_PRIORITY),
                s.line,
            ),
        )[:_SYMBOL_OUTLINE_MAX_PER_FILE]
        kept.sort(key=lambda s: s.line)
        header = f"{rel_str}:"
        body_lines = [f"  {s.kind} {s.name}:{s.line + 1}" for s in kept]
        if len(syms) > _SYMBOL_OUTLINE_MAX_PER_FILE:
            body_lines.append(f"  ... (+{len(syms) - _SYMBOL_OUTLINE_MAX_PER_FILE} more)")
        chunk = "\n".join([header, *body_lines])
        added = len(chunk) + 1
        if total + added > _SYMBOL_OUTLINE_MAX_CHARS and rows:
            remaining = len(rel_entries) - files_emitted
            rows.append(f"... ({remaining} more files; outline budget exhausted)")
            break
        rows.append(chunk)
        total += added
    return "\n".join(rows)


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


_CRITIC_SYSTEM_PROMPT = (
    "You are a strict reviewing critic embedded inside an autonomous coding"
    " agent's loop. The worker agent is editing a real repository to satisfy"
    " a user task. You see (a) the task, (b) a short tail of the worker's"
    " recent assistant messages and tool calls, and (c) the trigger that"
    " summoned you.\n\n"
    "Your job is to point out concrete problems the worker is likely to miss:"
    " mis-stated requirements, off-by-one logic, missing edge cases, broken"
    " invariants, security regressions, test coverage gaps, anything that"
    " suggests the work is not actually done.\n\n"
    "Be terse. Bullet points. If everything looks fine, say so. End your"
    " response with exactly one of these verdict lines on its own line:\n"
    "    VERDICT: SATISFIED\n"
    "    VERDICT: NEEDS_WORK\n"
    "Anything else in the last line is treated as NEEDS_WORK."
)

_PROMPT_REVISION_SYSTEM_PROMPT = """\
You revise raw coding-agent tasks before the main worker loop starts.

Goal: transform a terse, vague, or under-specified task into a clear task
specification the worker can act on immediately. Preserve every explicit
constraint from the raw task. Do not invent requirements. Use repo context only
to name likely files, conventions, verification commands, and success criteria.

If the raw task is already crisp, still restate it compactly rather than adding
new scope. If important ambiguity remains, list at most 3 clarifying questions;
the downstream worker may have to proceed under conservative assumptions, so the
revised task must remain actionable without answers.

Output exactly this shape, with no preamble:
<revised_task>
...plain text revised task...
</revised_task>
<clarifying_questions>
- question, or "none"
</clarifying_questions>
"""


_CONTEXT_SUMMARY_SYSTEM_PROMPT = (
    "You are compacting a long autonomous-coding-agent transcript so the agent"
    " can keep working with a smaller context window. Produce a dense, factual"
    " progress summary that lets the agent resume WITHOUT re-reading the"
    " elided history. Cover, in order:\n"
    "1. The goal, in one line.\n"
    "2. What has been tried and the outcome of each attempt — which edits were"
    " kept, which were reverted, and which directions turned out to be dead"
    " ends (so the agent does not repeat them).\n"
    "3. The current state: files changed so far, the best result/score"
    " achieved, and the latest verified commit sha.\n"
    "4. The concrete next steps the agent intended to take.\n"
    "Be specific about file paths, function names, numbers, and commit shas."
    " Do not include pleasantries or meta-commentary. Output only the summary."
)

# Prepended to the post-compaction restart message so the worker knows the
# history was summarised rather than lost, and continues rather than restarting.
_CONTEXT_RESTART_NOTICE = (
    "[harness context restart] The earlier conversation was compacted to free"
    " up context. Everything you had done up to this point is captured in the"
    " progress summary below — trust it for prior results and continue the task"
    " from here. Do NOT start over.\n\nPROGRESS SUMMARY:\n"
)


@dataclass(frozen=True, slots=True)
class _CritiqueResult:
    text: str
    satisfied: bool


@dataclass(frozen=True, slots=True)
class _PromptRevision:
    revised_task: str
    clarifying_questions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _MetricSample:
    label: str
    score: float | None
    returncode: int | None
    sha: str = ""
    error: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    # Comparison thresholds parsed from the metric command output (e.g.
    # ``assert cycles() < 1487`` lines). Used to point the worker at the
    # next unmet target rather than a vague "go faster". See
    # ``_extract_metric_targets``.
    targets: tuple[float, ...] = ()
    # True when the grader reported the score as a maxed-out fraction
    # (``SCORE: 27/27``): the metric is at its provable ceiling and cannot
    # be improved. See ``_metric_at_fraction_ceiling``.
    at_ceiling: bool = False


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


def _coerce_metric_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


# A comparison operator followed by a numeric literal, e.g. the
# ``< 1487`` in ``assert cycles() < 1487``. Underscores in the literal
# (Python int separators) are tolerated and stripped.
_METRIC_TARGET_RE = re.compile(r"(<=|>=|<|>)\s*([0-9][0-9_]*(?:\.[0-9]+)?)")


def _extract_metric_targets(
    text: str,
    *,
    goal: Literal["minimize", "maximize"],
) -> tuple[float, ...]:
    """Pull threshold numbers out of metric-command output.

    For ``goal="minimize"`` we want upper bounds the score must get
    *under* (``<`` / ``<=`` thresholds); for ``"maximize"`` we want lower
    bounds it must get *over* (``>`` / ``>=``). Benchmarks commonly print
    these as ``assert <expr> < N`` lines (one per unmet speed tier), so
    extracting them turns "go faster" into a concrete next target.
    Order-preserving and de-duplicated.
    """
    wanted = {"<", "<="} if goal == "minimize" else {">", ">="}
    seen: set[float] = set()
    out: list[float] = []
    for op, num in _METRIC_TARGET_RE.findall(text):
        if op not in wanted:
            continue
        try:
            value = float(num.replace("_", ""))
        except ValueError:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def _next_metric_target(
    targets: tuple[float, ...],
    current: float | None,
    goal: Literal["minimize", "maximize"],
) -> float | None:
    """The nearest threshold the current score has not yet met: the
    largest ``<`` bound still above the score (minimize) or the smallest
    ``>`` bound still below it (maximize). None when all are met or there
    is nothing to aim at."""
    if not targets or current is None:
        return None
    if goal == "minimize":
        unmet = [t for t in targets if t < current]
        return max(unmet) if unmet else None
    unmet = [t for t in targets if t > current]
    return min(unmet) if unmet else None


# A fraction in metric output, e.g. the ``27/27`` in ``SCORE: 27/27``. A
# maxed-out fraction means the metric is at its provable ceiling. See
# ``_metric_at_fraction_ceiling``.
_METRIC_FRACTION_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)")


def _metric_at_fraction_ceiling(text: str, score: float) -> bool:
    """True if ``text`` reports ``score`` as a maxed-out ``X/Y`` fraction.

    Many graders print a bounded score as ``X/Y`` (``SCORE: 27/27``,
    ``passed 27/27``). When the numerator equals both the parsed score and
    the denominator, the metric is provably at its ceiling: no further edit
    can push it higher. Detecting this lets a ``maximize`` run stop cleanly
    instead of treating the unbeatable plateau as a local optimum worth
    spending the rest of the budget pivoting away from. Conservative: only
    fires on an exact ``score/score`` match, so partial scores (``26/27``)
    and unbounded metrics (raw cycle counts, which never print a
    denominator) are unaffected.
    """
    for num_s, den_s in _METRIC_FRACTION_RE.findall(text):
        try:
            num = float(num_s)
            den = float(den_s)
        except ValueError:  # pragma: no cover - regex already constrains digits
            continue
        if num == score and num == den:
            return True
    return False


def _metric_is_better(
    candidate: float,
    incumbent: float,
    goal: Literal["minimize", "maximize"],
) -> bool:
    if goal == "minimize":
        return candidate < incumbent
    return candidate > incumbent


def _best_metric_sample(
    samples: list[_MetricSample],
    *,
    goal: Literal["minimize", "maximize"],
) -> _MetricSample | None:
    parsed = [sample for sample in samples if sample.score is not None]
    if not parsed:
        return None
    best = parsed[0]
    for sample in parsed[1:]:
        assert sample.score is not None
        assert best.score is not None
        if _metric_is_better(sample.score, best.score, goal):
            best = sample
    return best


def _format_score(score: float | None) -> str:
    if score is None:
        return "unparsed"
    return f"{score:g}"


def _format_metric_sample(sample: _MetricSample) -> str:
    parts = [f"{sample.label}: score={_format_score(sample.score)}"]
    if sample.returncode is not None:
        parts.append(f"exit={sample.returncode}")
    if sample.sha:
        parts.append(f"sha={sample.sha[:12]}")
    if sample.error:
        parts.append(f"error={sample.error[:200]}")
    return ", ".join(parts)


def _metric_goal(metric_cfg: Any) -> Literal["minimize", "maximize"] | None:
    goal = getattr(metric_cfg, "goal", None)
    if goal in ("minimize", "maximize"):
        return goal
    return None


def _format_metric_feedback(
    history: list[_MetricSample],
    *,
    goal: Literal["minimize", "maximize"],
) -> str:
    latest = history[-1]
    best = _best_metric_sample(history, goal=goal)
    previous_best = _best_metric_sample(history[:-1], goal=goal)
    best_line = _format_metric_sample(best) if best is not None else "none parsed yet"

    if latest.score is None:
        verdict = "latest metric score was not parsed; inspect output before trusting this edit"
    elif previous_best is None:
        verdict = "first parsed metric sample"
    else:
        assert previous_best.score is not None
        verdict = (
            "new best; continue from this commit"
            if _metric_is_better(latest.score, previous_best.score, goal)
            else "not a new best; revert this edit or pivot unless it was purely enabling"
        )

    lines = [
        "[harness metric]",
        f"goal: {goal} ({'lower' if goal == 'minimize' else 'higher'} is better)",
        f"latest: {_format_metric_sample(latest)}",
        f"best: {best_line}",
        f"verdict: {verdict}",
        "trajectory (last 5):",
    ]
    lines.extend(f"- {_format_metric_sample(sample)}" for sample in history[-5:])
    next_target = _next_metric_target(latest.targets, latest.score, goal)
    if next_target is not None and latest.score is not None:
        direction = "below" if goal == "minimize" else "above"
        lines.append(
            f"next target: drive the metric {direction} {next_target:g}"
            f" (current {latest.score:g}) — the nearest threshold you have not"
            f" cleared yet; aim edits at crossing it."
        )
    if latest.score is None:
        if latest.stdout_tail:
            lines.append(f"stdout tail: {latest.stdout_tail[-500:]}")
        if latest.stderr_tail:
            lines.append(f"stderr tail: {latest.stderr_tail[-500:]}")
    lines.append(
        "next: keep verify-passing edits that improve the metric; for flat/worse results, "
        "restore the prior best or change strategy instead of polishing the same approach."
    )
    return "\n".join(lines)


# How many times a detected metric plateau is met with a "pivot strategy"
# nudge before the loop actually stops. The plateau detector is eager (it
# fires the first time a verified metric merely ties the prior best), and on
# optimisation tasks the remaining budget often still hides large gains that
# only a fundamentally different approach unlocks. Rather than quit at the
# first stall, nudge the worker to change strategy a few times; only stop if
# it still cannot beat its best after that.
_METRIC_PLATEAU_PATIENCE = 3

# A metric plateau only becomes a terminal condition once the run has
# entered its final budget slice. While more than this fraction of the
# token budget remains, a plateau is treated as a local optimum worth
# pivoting away from rather than a reason to quit: stopping with most of
# the budget unspent leaves measurable gains (and money) on the table.
# Only consulted when a real BudgetTracker is wired in; with no budget
# signal the loop falls back to the fixed `_METRIC_PLATEAU_PATIENCE`.
_METRIC_PLATEAU_STOP_BELOW_BUDGET = 0.25

# Plateau nudges escalate with budget pressure. A stall means the worker has
# hit a local optimum; how aggressively we push it off that optimum scales
# with how much runway is left. With most of the budget intact a plateau is
# cheap to explore around, so we invite a bold experiment we can afford to
# throw away. As the budget drains the ask narrows from "try another angle"
# to "spend your remaining budget on the single highest-value structural bet
# you can make". Selected by `_metric_plateau_nudge`; the shared
# "[harness plateau]" prefix keeps the signal greppable across tiers.
_METRIC_PLATEAU_NUDGE_EXPLORE = (
    "[harness plateau] Your recent verified edits have stopped improving the"
    " metric \u2014 you have hit a local optimum. You still have most of your"
    " budget left, so you can afford to explore boldly. Do NOT call finish_run"
    " yet. Keep the current best commit, then run an experiment you have not"
    " tried: a structurally different algorithm, a different data layout, or a"
    " property of the problem you have not exploited. A failed experiment is"
    " cheap right now \u2014 a wasted budget is not. Be ambitious."
)
_METRIC_PLATEAU_NUDGE_PIVOT = (
    "[harness plateau] Your recent verified edits have stopped improving the"
    " metric \u2014 you are polishing the same approach and have hit a local"
    " optimum. About half your budget is gone and micro-tuning is no longer"
    " paying off. Do NOT call finish_run yet. Pivot decisively to a"
    " fundamentally different strategy: re-read the problem for a structurally"
    " better algorithm (vectorise/batch the hot loop, change the data layout,"
    " eliminate redundant work) rather than nibbling at what you already have."
    " Keep the current best commit, then commit to a genuinely new direction."
)
_METRIC_PLATEAU_NUDGE_FINAL = (
    "[harness plateau] Your recent verified edits have stopped improving the"
    " metric and your budget is nearly spent \u2014 this is your last chance to"
    " move the number. Do NOT fritter the remainder on micro-tuning. Identify"
    " the single change with the highest expected payoff (the biggest"
    " structural rewrite you are confident you can land and verify) and spend"
    " what is left on landing it. Keep the current best commit as a floor, then"
    " make your one best bet count."
)

# Budget fraction above which a plateau is treated as cheap to explore.
_METRIC_PLATEAU_NUDGE_EXPLORE_ABOVE = 0.5

# Nudge injected when the worker calls finish_run on an optimisation run while
# real budget still remains. On metric runs the task explicitly asks the worker
# to keep optimising up to the cap, but workers routinely call finish_run with
# most of the budget unspent \u2014 leaving measurable gains (and money) on the
# table. This is a worker-initiated early stop, distinct from a metric plateau,
# so it carries its own "[harness budget]" prefix to stay greppable.
_METRIC_FINISH_NUDGE = (
    "[harness budget] You called finish_run, but this is an optimisation run"
    " and a large share of your budget is still unspent. Stopping now leaves"
    " measurable gains on the table \u2014 the task asks you to keep optimising"
    " right up to the budget cap. Do NOT finish yet. Keep your current best"
    " commit as a floor, then make another concrete attempt to move the metric:"
    " profile the hot path again, try a structurally different approach, or"
    " exploit a property of the problem you have not used. You may call"
    " finish_run once your budget is nearly spent."
)

# How many times an early finish_run on a metric run is rejected (with a
# keep-going nudge) before the loop honours it. Bounds the nudging so a worker
# that genuinely has nothing left to try can still stop cleanly.
_METRIC_EARLY_FINISH_PATIENCE = 3


def _metric_plateau_nudge(budget_remaining: float | None) -> str:
    """Select a plateau nudge whose intensity scales with budget pressure.

    With no budget signal (tests / MCP) we default to the explore tier so the
    worker is encouraged to keep trying new directions rather than quit.
    """
    if budget_remaining is None or budget_remaining > _METRIC_PLATEAU_NUDGE_EXPLORE_ABOVE:
        return _METRIC_PLATEAU_NUDGE_EXPLORE
    if budget_remaining > _METRIC_PLATEAU_STOP_BELOW_BUDGET:
        return _METRIC_PLATEAU_NUDGE_PIVOT
    return _METRIC_PLATEAU_NUDGE_FINAL


def _metric_plateau_summary(
    history: list[_MetricSample],
    *,
    goal: Literal["minimize", "maximize"],
    min_parsed_samples: int = 5,
) -> str | None:
    parsed = [sample for sample in history if sample.score is not None]
    if len(parsed) < min_parsed_samples:
        return None
    latest = parsed[-1]
    previous_best = _best_metric_sample(parsed[:-1], goal=goal)
    if previous_best is None or latest.score is None or previous_best.score is None:
        return None
    if latest.score != previous_best.score:
        return None
    best = _format_metric_sample(previous_best)
    latest_text = _format_metric_sample(latest)
    return (
        "metric plateau: latest verified metric tied the prior best after "
        f"{len(parsed)} parsed samples; stopping to preserve performance per dollar. "
        f"latest={latest_text}; best={best}"
    )


def _format_messages_tail_for_critic(
    messages: list[dict[str, Any]], *, max_messages: int = 6, max_chars: int = 6000
) -> str:
    """Render the last few messages as a plain-text transcript for the
    critic. Tool calls / results are shown as compact summaries; long
    payloads are truncated so the critic call stays cheap.
    """
    tail = messages[-max_messages:]
    parts: list[str] = []
    for msg in tail:
        role = str(msg.get("role", "?"))
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}] {content[:1500]}")
            continue
        if not isinstance(content, list):
            parts.append(f"[{role}] {str(content)[:1500]}")
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(f"[{role}:text] {str(block.get('text', ''))[:1500]}")
            elif btype == "tool_use":
                inp = json.dumps(block.get("input") or {}, ensure_ascii=False)
                parts.append(f"[{role}:tool_use {block.get('name', '')}] {inp[:800]}")
            elif btype == "tool_result":
                body = block.get("content", "")
                if not isinstance(body, str):
                    body = json.dumps(body, ensure_ascii=False)
                parts.append(f"[{role}:tool_result] {body[:800]}")
            elif btype == "thinking":
                # Skip reasoning blocks - the critic doesn't need them.
                continue
    joined = "\n".join(parts)
    if len(joined) > max_chars:
        joined = joined[-max_chars:]
    return joined


def _parse_critic_verdict(text: str) -> bool:
    """Return True iff the critic's last non-empty line is ``VERDICT:
    SATISFIED``. Anything else is treated as NEEDS_WORK."""
    last = ""
    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if line:
            last = line
            break
    return last.upper() == "VERDICT: SATISFIED"


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
    mode: Literal["run", "plan"] = "run",
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
    base = _PLAN_SYSTEM_PROMPT_BASE if mode == "plan" else _SYSTEM_PROMPT_BASE
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
    mode: Literal["run", "plan"] = "run",
) -> list[ToolDefinition]:
    """Build the tool list exposed to the loop. Filters by what the
    dispatcher actually allows (e.g. run_command may be disabled).

    ``mode="plan"`` filters mutating tools
    (``apply_edit``/``apply_patch``) out of ``ALL_TOOLS`` and swaps
    ``LOOP_EXTRA_TOOLS`` for ``PLAN_EXTRA_TOOLS`` (drops
    ``finish_run``/``run_metric_command``, adds ``finish_planning``).
    """
    available = set(dispatcher.available_tool_names())
    extras: tuple[type[Any], ...] = PLAN_EXTRA_TOOLS if mode == "plan" else LOOP_EXTRA_TOOLS
    base_tools: tuple[type[Any], ...] = ALL_TOOLS
    if mode == "plan":
        # Plan mode is read-only; filter mutating tools even if the
        # dispatcher would otherwise allow them.
        plan_blocked = {ApplyEditInput.TOOL_NAME, ApplyPatchInput.TOOL_NAME}
        base_tools = tuple(cls for cls in ALL_TOOLS if cls.TOOL_NAME not in plan_blocked)
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
    # and update statuses; survives crashes via .agent6/runs/<id>/graph.jsonl.
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
    # single-turn edits — rewriting a hot function wholesale beats nibbling
    # at it across turns — and the worker routinely truncated mid-apply_patch
    # against the 16k default, wasting the whole turn. Lifting the ceiling
    # only when a metric goal is present keeps ordinary feature/bugfix runs
    # (where giant turns mostly mean a confused model) on the tighter cap.
    metric_task_max_tokens: int = 32768
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
    # Retry the provider call once on transient ProviderError before aborting
    # the run. Common cases: Anthropic 529 overload, OpenRouter 502, brief
    # socket timeouts. Off by default (0) is no-retry. -era
    # audit finding #5 - cheap insurance against single-flake aborts.
    provider_retry_count: int = 1
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
    mode: Literal["run", "plan"] = "run"
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

        tools = _tool_definitions(self.dispatcher)
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
    ) -> RunResult:
        """Shared loop body for both fresh ``run()`` and ``resume()``.

        Before each provider call, writes a snapshot of the workflow's
        in-memory state to ``self.resume_state_path`` (if set) so a
        crash mid-call can be resumed from the same point.
        """
        # Cache the original task for in-loop critic calls.
        # Works for both fresh run() (messages[0] has the task) and
        # resume() (snapshot persists the same messages list).
        original_task = _extract_initial_task(messages)
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
        for iteration in range(start_iteration, self.max_iterations + 1):
            self._maybe_compact(messages)

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
                        summary=text[:1000],
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

            messages.append({"role": "user", "content": tool_results})

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

        Tier 2 (expensive): once cumulative tool_result content crosses
        ``compact_summarise_at_chars``, summarise the elided history into a
        compact progress block and restart the message list from (original
        task + summary). Fail-safe: if summarisation errors or returns
        nothing, the message list is left untouched (tier-1 elision already
        ran) and the run continues.
        """
        n_dropped = _compact_old_tool_results(
            messages, max_total_bytes=self.compact_drop_at_chars, keep_recent=2
        )
        if n_dropped:
            self._log(f"LOOP: compaction elided {n_dropped} old tool_result blocks")
            self._emit("loop.compact.dropped", n=n_dropped)
        total = sum(
            len(item.get("content", "") or "")
            for msg in messages
            if isinstance(msg.get("content"), list)
            for item in msg["content"]
            if isinstance(item, dict) and item.get("type") == "tool_result"
        )
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
