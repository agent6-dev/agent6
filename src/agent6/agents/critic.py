# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Critic-of-prompt sub-agent: refine a vague task into a precise spec."""

from __future__ import annotations

from agent6.agents._common import call_for_model
from agent6.models import OpenQuestion, RefinedSpec
from agent6.providers import Provider

_SYSTEM = """You are the critic-of-prompt for a coding agent.

Your only job: take the user's raw task plus the project's AGENTS.md and produce
a refined task statement that is precise enough for a planner to act on.

Rules:
- Do NOT propose an implementation. That is the planner's job.
- Refined task should be one short paragraph, imperative, concrete.
- Never invent constraints not present in the input.
- If AGENTS.md is empty or absent, raise an open question recommending that the
  user run `agent6 init` (or hand-write an AGENTS.md) before serious work.

Open-questions discipline — ONLY raise an open question when ALL of:
  (a) the answer would materially change the resulting diff, AND
  (b) the answer cannot be inferred from the task text, existing code,
      existing tests, AGENTS.md, or standard project conventions, AND
  (c) picking the most plausible interpretation would risk producing wrong
      code (not just sub-optimal code).

If any of (a)/(b)/(c) is false, DO NOT raise the question — pick the most
plausible interpretation, encode it in the refined task, and proceed.
Rhetorical questions, metacognitive questions, style preferences, and
"should I prioritize X over Y" questions almost always fail (b) or (c) and
should NOT be surfaced. The default is to proceed, not to ask.

Each open question MUST be an object with a `question` string and a
`suggestions` array of 2-4 short candidate answers (the most-plausible
answer FIRST). The user picks one by index or types their own. Empty
suggestions are allowed only when no candidate answers make sense.

Leave open_questions empty unless the bar above is met.
"""

_NO_AGENTS_MD_QUESTION = OpenQuestion(
    question=(
        "No AGENTS.md found in the repository root. agent6 works best when the "
        "project has an AGENTS.md describing conventions, the verify command, and "
        "security invariants. How should we proceed?"
    ),
    suggestions=(
        "Run `agent6 init` now to scaffold one",
        "Proceed without AGENTS.md (not recommended)",
    ),
)


def critic_refine(
    provider: Provider,
    *,
    user_task: str,
    agents_md: str,
) -> RefinedSpec:
    user = f"USER TASK:\n{user_task}\n\nAGENTS.md:\n{agents_md or '(empty)'}\n"
    refined = call_for_model(
        provider, system=_SYSTEM, user=user, output_model=RefinedSpec, max_tokens=2048
    )
    # Defensively guarantee the missing-AGENTS.md question is surfaced even if
    # the model forgot — agents downstream depend on it appearing.
    if not agents_md.strip() and not any("AGENTS.md" in q.question for q in refined.open_questions):
        return RefinedSpec(
            refined_task=refined.refined_task,
            open_questions=(*refined.open_questions, _NO_AGENTS_MD_QUESTION),
        )
    return refined
