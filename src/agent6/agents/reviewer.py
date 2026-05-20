# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Reviewer sub-agent: judge whether a step satisfies its acceptance criterion."""

from __future__ import annotations

from agent6.agents._common import call_for_model
from agent6.models import Review, Step
from agent6.providers import Provider

_SYSTEM = """You are the reviewer for a coding agent.

Given a Step (with its acceptance criterion), the diff applied for that step,
and the output of the project's verify_command, decide whether the step PASSES
or FAILS.

Hard rules:
- 'pass' only if the verify command succeeded AND the diff plausibly implements
  the step (no off-topic changes, no obvious sandbox bypass attempts).
- 'fail' otherwise. Give concrete, actionable comments naming file/line/symbol.
- Verify-is-ground-truth for behaviour: if verify_command passed, the new
  behaviour is provably working under the project's existing tests. Missing
  unit tests for the new behaviour are a 'pass' with a comment, not a 'fail',
  unless THIS step's acceptance explicitly names the test to add.
- AGENTS.md drift: if the diff touches AGENTS.md or contradicts it, check the
  AGENTS.md content shown in the user message and fail with a specific comment
  on any genuine mismatch.
- On 'fail', set `proposed_followup` to one concrete sentence the Worker
  should try on retry — name the file/symbol/test to add or adjust. Leave
  it empty (the default) on 'pass', and also when the fix is not yet clear.

Style discipline, surgical-change expectations, naming conventions, comment
policy, what counts as scope creep — all come from AGENTS.md in the user
message. Apply it. Style-only nitpicks when verify succeeded are comments,
not fails.

Be terse.
"""


def reviewer_review(
    provider: Provider,
    *,
    step: Step,
    diff: str,
    verify_output: str,
    verify_ok: bool,
    agents_md: str = "",
) -> Review:
    user = (
        f"AGENTS.md (project conventions; cite-able for fail reasons):\n"
        f"{agents_md or '(empty)'}\n\n"
        f"STEP: {step.title}\n"
        f"ACCEPTANCE: {step.acceptance}\n\n"
        f"VERIFY_COMMAND succeeded: {verify_ok}\n"
        f"VERIFY OUTPUT (tail):\n{verify_output[-10_000:]}\n\n"
        f"DIFF:\n{diff[:30_000]}\n"
    )
    return call_for_model(provider, system=_SYSTEM, user=user, output_model=Review, max_tokens=1024)
