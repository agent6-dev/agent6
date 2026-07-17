# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Typed assembly of the agent-loop system prompt.

The helpers that fill the pure `agent6.prompts.loop` block templates with a
run's config + repo summary + memories + skills. These stay in the workflow
layer because their signatures need agent6 types (`Config`, `RepoSummary`,
`MemoryEntry`, `ResolvedSkills`); the leaf `agent6.prompts` package holds only
the dependency-free text they render.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from agent6.config import Config
from agent6.memory import MemoryEntry
from agent6.prompts.loop import (
    AGENT_SYSTEM_PROMPT_BASE,
    ASK_SYSTEM_PROMPT_BASE,
    MACHINE_SYSTEM_PROMPT_BASE,
    MEMORIES_HEADER_READONLY,
    MEMORIES_HEADER_RUN,
    PLAN_SYSTEM_PROMPT_BASE,
    SKILLS_HEADER,
    SYSTEM_PROMPT_BASE,
    V2_BUDGET_BLOCK_TEMPLATE,
    V2_METRIC_BLOCK_TEMPLATE,
    V2_REPO_BLOCK_TEMPLATE,
    V2_VERIFY_BLOCK_TEMPLATE,
    dag_rules_block,
    no_verify_block,
)
from agent6.skills import ResolvedSkills
from agent6.types import RepoSummary

# Cross-run memories injected into the system prompt. Bounded so the block
# can never crowd out the task: one entry is clipped at MEMORY_ENTRY_MAX_CHARS
# and the whole block at MEMORIES_MAX_CHARS (newest entries win; the count of
# elided older ones is shown).
MEMORY_ENTRY_MAX_CHARS = 1200
MEMORIES_MAX_CHARS = 12000


def memories_block(
    entries: tuple[MemoryEntry, ...],
    *,
    mode: Literal["run", "plan", "ask", "machine", "agent"],
) -> str:
    """Render the <memories> system-prompt block from ACTIVE entries.

    Run mode always renders it (the header doubles as the add_memory usage
    guidance); plan/ask render it only when there is something to read.
    Machine/agent assembly returns before this block, so those modes never
    see it. Callers pass active entries only; invalidated ones are filtered
    at load time.
    """
    if mode != "run" and not entries:
        return ""
    # Newest win under the total cap: rank by (created_at, id) descending and
    # keep the contiguous newest window that fits; render keepers in original
    # (chronological, per-scope) order. The +48 approximates the id/date line
    # overhead per entry.
    ranked = sorted(entries, key=lambda e: (e.created_at, e.id), reverse=True)
    kept: set[str] = set()
    used = 0
    for e in ranked:
        cost = min(len(e.body), MEMORY_ENTRY_MAX_CHARS) + 48
        if kept and used + cost > MEMORIES_MAX_CHARS:
            break
        kept.add(e.id)
        used += cost
    lines: list[str] = [MEMORIES_HEADER_RUN if mode == "run" else MEMORIES_HEADER_READONLY, ""]
    elided = len(entries) - len(kept)
    if elided:
        lines.append(f"({elided} older memories elided)")
        lines.append("")
    rendered_any = False
    for scope in ("facts", "decisions", "preferences"):
        scoped = [e for e in entries if e.scope == scope and e.id in kept]
        if not scoped:
            continue
        rendered_any = True
        lines.append(f"[{scope}]")
        for e in scoped:
            body = e.body
            if len(body) > MEMORY_ENTRY_MAX_CHARS:
                body = body[:MEMORY_ENTRY_MAX_CHARS] + " [clipped]"
            first, *rest = body.splitlines() or [""]
            lines.append(f"- {e.id} ({e.created_at[:10]}): {first}")
            lines.extend(f"  {ln}" for ln in rest)
        lines.append("")
    if not rendered_any:
        lines.append("(none recorded yet)")
        lines.append("")
    if lines[-1] == "":
        lines.pop()
    lines.append("</memories>")
    return "\n".join(lines) + "\n"


SKILL_INDEX_LINE_MAX_CHARS = 200
SKILLS_INDEX_MAX_CHARS = 8000
SKILL_ALWAYS_MAX_CHARS = 24000


def skills_block(resolved: ResolvedSkills) -> str:
    """Render the skills system-prompt parts: full text for ``always`` skills,
    a bounded one-line-per-skill index for the rest. Empty when no skills."""
    if not resolved.enabled and not resolved.always:
        return ""
    parts: list[str] = []
    for sk in resolved.always:
        text = sk.text
        if len(text) > SKILL_ALWAYS_MAX_CHARS:
            text = text[:SKILL_ALWAYS_MAX_CHARS] + "\n[clipped]"
        parts.append(f'<skill name="{sk.name}">\n{text.rstrip()}\n</skill>\n')
    if resolved.enabled:
        lines = [SKILLS_HEADER, ""]
        used = 0
        shown = 0
        for sk in resolved.enabled:
            line = f"- {sk.name} — {sk.description}"
            if len(line) > SKILL_INDEX_LINE_MAX_CHARS:
                line = line[: SKILL_INDEX_LINE_MAX_CHARS - 10] + " [clipped]"
            if used + len(line) > SKILLS_INDEX_MAX_CHARS:
                break
            lines.append(line)
            used += len(line) + 1
            shown += 1
        if shown < len(resolved.enabled):
            lines.append(
                f"({len(resolved.enabled) - shown} skills elided; `agent6 skills list` shows all)"
            )
        lines.append("</skills>")
        parts.append("\n".join(lines) + "\n")
    return "\n".join(parts)


def repo_priors_block(repo: RepoSummary) -> str:
    """Render the <repo-priors> block: the repo header line plus the structural
    priors (co-change pairs, hot symbols, repo map, symbol outline) that are
    present on this summary. Outside a git repository (`agent6 ask` runs
    anywhere) the header names the situation so the model doesn't reach for
    git history or a tracked-file map that isn't there."""
    co_change_block = ""
    if repo.co_change_pairs:
        lines = "\n".join(
            f"  {p.file_a} <-> {p.file_b}  (changed together {p.count} times)"
            for p in repo.co_change_pairs[:20]
        )
        co_change_block = (
            "Git co-change pairs (files that historically change together;"
            " consider when editing one of these):\n"
            f"{lines}\n\n"
        )

    hot_symbols_block = ""
    if repo.hot_symbols:
        lines = "\n".join(
            f"  {s.name} ({s.kind}) at {s.def_path}:{s.def_line + 1},"
            f" referenced across {s.files_referenced} files"
            for s in repo.hot_symbols[:15]
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

    if repo.is_git:
        repo_line = (
            f"Repository: branch={repo.branch},"
            f" head={repo.head_sha[:12] or '(no commits yet)'}, files={repo.file_count}"
        )
    else:
        repo_line = "Directory (not a git repository; no branch, history, or tracked-file map)."
    return V2_REPO_BLOCK_TEMPLATE.format(
        repo_line=repo_line,
        top_level=", ".join(repo.top_level),
        agents_md=repo.agents_md or "(empty)",
        repo_map_block=repo_map_block,
        symbol_outline_block=symbol_outline_block,
        co_change_block=co_change_block,
        hot_symbols_block=hot_symbols_block,
        recent_log=repo.recent_log or "(none)",
    )


def build_system_prompt(
    *,
    config: Config,
    repo: RepoSummary,
    mode: Literal["run", "plan", "ask", "machine", "agent"] = "run",
    memories: tuple[MemoryEntry, ...] = (),
    skills: ResolvedSkills | None = None,
) -> str:
    """Assemble the system prompt from static blocks + run-specific context.

    The whole system prompt is sent on every turn but gets cached by the
    Anthropic prompt-caching machinery (lineage). Per-turn cost
    after the first call is ~10% of full input rate for the cached prefix.

    ``mode="plan"`` swaps the base block for the planning-mode
    prompt; the verify/repo/co-change/hot-symbols blocks below are
    appended unchanged so the planner sees the same project context an
    executor would. The metric block is run-mode only (the other modes
    do not expose `run_metric_command`).
    """
    base = (
        ASK_SYSTEM_PROMPT_BASE
        if mode == "ask"
        else MACHINE_SYSTEM_PROMPT_BASE
        if mode == "machine"
        else AGENT_SYSTEM_PROMPT_BASE
        if mode == "agent"
        else PLAN_SYSTEM_PROMPT_BASE
        if mode == "plan"
        else SYSTEM_PROMPT_BASE
    )
    # ADVANCED override: replace run-mode's static base with an operator-supplied
    # file. The dynamic blocks below (verify/metric/budget/repo-priors) still
    # append, so repo context + budget awareness are preserved. The file is
    # validated to exist at config-load time; run startup warns if it omits the
    # core tool names. Scoped to run mode -- the worker is what operators tune.
    override = config.prompt.system_prompt_file
    if mode == "run" and override:
        base = Path(override).expanduser().read_text(encoding="utf-8")
    # Fill the DAG-rules sentinel (present only in the run-mode default base).
    # On an override file the sentinel is absent, so this is a no-op there.
    # "auto" is pinned to on/off by the CLI (resolve_decompose) before the
    # workflow starts; an unresolved "auto" reaching here (bench/embedders)
    # conservatively renders the optional block.
    base = base.replace("__DAG_RULES_BLOCK__", dag_rules_block(config.prompt.decompose == "on"))
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
            V2_BUDGET_BLOCK_TEMPLATE.format(
                in_cap=config.budget.max_input_tokens,
                out_cap=config.budget.max_output_tokens,
            )
        )
        return "\n".join(parts)

    verify_argv = list(config.workflow.verify_command)
    if verify_argv:
        parts.append(
            V2_VERIFY_BLOCK_TEMPLATE.format(
                argv=json.dumps(verify_argv),
                timeout_s=config.workflow.verify_timeout_s,
            )
        )
    else:
        parts.append(no_verify_block(mode))

    # Run mode only: plan/ask do not expose `run_metric_command`, and the
    # "harness automatically runs this metric" behaviour is the run loop's.
    if mode == "run" and config.workflow.metric is not None:
        m = config.workflow.metric
        parts.append(
            V2_METRIC_BLOCK_TEMPLATE.format(
                argv=json.dumps(list(m.command)),
                pattern=m.pattern,
                goal=m.goal,
            )
        )

    parts.append(
        V2_BUDGET_BLOCK_TEMPLATE.format(
            in_cap=config.budget.max_input_tokens,
            out_cap=config.budget.max_output_tokens,
        )
    )

    parts.append(repo_priors_block(repo))

    # Cross-run memories, after the repo priors. Empty for machine/agent
    # (returned above) and for plan/ask with nothing recorded.
    memories_part = memories_block(memories, mode=mode)
    if memories_part:
        parts.append(memories_part)

    # Operator-installed skills, last: `always` full texts + the on-demand
    # index. The caller resolves discovery + [skills.state]; None or an empty
    # resolution renders nothing.
    if skills is not None and (skills_part := skills_block(skills)):
        parts.append(skills_part)

    return "\n".join(parts)
