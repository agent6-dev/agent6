# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Worker sub-agent: produce concrete edits for one Step."""

from __future__ import annotations

from agent6.agents._common import call_for_model
from agent6.models import Edit, Step
from agent6.providers import Provider
from agent6.types import FileContext

_SYSTEM = """You are the worker for a coding agent.

You receive ONE step from a Plan plus the current contents of the relevant files.
Produce an Edit object containing one or more FileEdit operations that implement
the step.

Edit-schema rules (hard, non-negotiable — these are about the FileEdit format,
not about style):
- kind='replace': old_string must be present and UNIQUE in the file. Include
  enough surrounding context to be unique. new_string can be empty (to delete).
- kind='create': only when the file does not yet exist. old_string must be empty,
  new_string is the full file content.
- Never include language fences or commentary in old_string / new_string content;
  use the raw file text.
- Never modify files outside the step's relevant_paths.
- If the step is ambiguous, write a short `notes` entry stating the ambiguity
  rather than picking silently.

When a `PREVIOUS ATTEMPT FAILED` block is present it is a JSON object with:
- `verify_returncode` and `verify_tail` from the verify command,
- `review` (the reviewer's comments on the rejected diff),
- `proposed_followup` (optional one-sentence steer; empty if the reviewer
  didn't have a concrete suggestion). Treat `proposed_followup` as a strong
  hint, not a hard constraint — if it conflicts with the step acceptance or
  AGENTS.md, prefer those.

Everything else — naming, comment policy, when to add defaults vs update call
sites, what counts as scope creep, refactoring discipline, error-handling
philosophy — comes from the AGENTS.md content in the user message. Follow it.
If AGENTS.md is empty, default to minimum-necessary edits matching the file's
existing style.

Tests are the authoritative behavioural specification. When a relevant_paths
test file pins down what a function must do (which inputs raise, which return
which value, which state transitions are legal), match that behaviour even if
prose comments, docstrings, or summary headers in the production file
disagree. Prose can be stale or summarised; the tests are what verify_command
actually checks. If a docstring's "summary" line and its detailed transition
table disagree, the table (and the tests) win. If a docstring contradicts a
test outright, treat the test as the spec and add a brief `notes` entry
flagging the discrepancy.
"""


def worker_edit(
    provider: Provider,
    *,
    step: Step,
    file_context: FileContext,
    previous_attempt_feedback: str = "",
    agents_md: str = "",
    parent_acceptance: str = "",
    sibling_commits: tuple[tuple[str, str], ...] = (),
) -> Edit:
    feedback = (
        f"\nPREVIOUS ATTEMPT FAILED. Reviewer feedback:\n{previous_attempt_feedback}\n"
        if previous_attempt_feedback
        else ""
    )
    parent_block = (
        f"PARENT TASK ACCEPTANCE (the larger goal this step contributes to):\n"
        f"{parent_acceptance}\n\n"
        if parent_acceptance
        else ""
    )
    if sibling_commits:
        sibling_lines = "\n".join(f"  - {sha[:7]} {title}" for sha, title in sibling_commits)
        sibling_block = f"COMPLETED SIBLING STEPS (already committed):\n{sibling_lines}\n\n"
    else:
        sibling_block = ""
    agents_block = f"AGENTS.md (project conventions):\n{agents_md or '(empty)'}\n\n"
    user = (
        f"{agents_block}"
        f"{parent_block}"
        f"{sibling_block}"
        f"STEP TITLE: {step.title}\n"
        f"RATIONALE: {step.rationale}\n"
        f"ACCEPTANCE: {step.acceptance}\n"
        f"RELEVANT PATHS: {', '.join(step.relevant_paths) or '(none specified)'}\n"
        f"{feedback}\n"
        f"FILES:\n{file_context.as_text()}\n"
    )
    return call_for_model(provider, system=_SYSTEM, user=user, output_model=Edit, max_tokens=8192)
