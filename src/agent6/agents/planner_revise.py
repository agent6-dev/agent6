# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Planner-revise sub-agent.

Given an existing `Plan` plus free-text user feedback (and the repo summary
that produced it), ask the planner LLM to emit a revised `Plan` in the same
schema. Used by `agent6.workflows.plan_mode` during the interactive loop.
"""

from __future__ import annotations

from agent6.agents._common import call_for_model
from agent6.models import Plan
from agent6.providers import Provider
from agent6.types import RepoSummary

_SYSTEM = """You are the planner for a coding agent.

You are revising an existing Plan based on user feedback. Emit a complete
new Plan (same schema as the input). Preserve unchanged steps verbatim, drop
steps the user no longer wants, add new ones if the feedback asks for them,
and reorder where requested.

Rules (unchanged from initial planning):
- Each step must end at a state where verify_command can pass.
- relevant_paths must be real repo-relative paths.
- Never instruct the Worker to push, force, rebase, or rewrite history.
- Never include arbitrary commands; the Worker has a fixed tool set.
"""


def planner_revise(
    provider: Provider,
    *,
    previous_plan: Plan,
    user_feedback: str,
    repo: RepoSummary,
    steer_instruction: str = "",
) -> Plan:
    steps_str = "\n".join(
        f"  {i + 1}. {s.title}\n     rationale: {s.rationale}\n"
        f"     relevant_paths: {list(s.relevant_paths)}\n"
        f"     acceptance: {s.acceptance}"
        for i, s in enumerate(previous_plan.steps)
    )
    steer_block = (
        f"\nSTEERING INSTRUCTION (mid-run; overrides feedback below if conflicting):\n"
        f"{steer_instruction}\n"
        if steer_instruction
        else ""
    )
    user = (
        f"CURRENT PLAN:\nsummary: {previous_plan.summary}\nsteps:\n{steps_str}\n"
        f"{steer_block}\n"
        f"USER FEEDBACK:\n{user_feedback}\n\n"
        f"REPO SUMMARY:\n"
        f"  branch: {repo.branch}\n"
        f"  head: {repo.head_sha[:12]}\n"
        f"  files: {repo.file_count}\n"
        f"  top-level: {', '.join(repo.top_level)}\n\n"
        f"AGENTS.md:\n{repo.agents_md or '(empty)'}\n\n"
        f"Emit the revised Plan."
    )
    return call_for_model(provider, system=_SYSTEM, user=user, output_model=Plan, max_tokens=4096)
