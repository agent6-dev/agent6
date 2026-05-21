# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Triage sub-agent: classify a task into a workflow ``Profile``.

The triage agent runs once at the top of the workflow on a cheap
(haiku-class) model. Its only job is to choose a ``TaskClass`` so the
state machine can short-circuit the critic / planner / reviewer pipeline
on simple work. See ``agent6.workflows.profiles`` for the dispatch table.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from agent6.agents._common import call_for_model
from agent6.providers import Provider
from agent6.types import RepoSummary

_LLM_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class TaskClass(StrEnum):
    """Outcome of the triage classifier — drives :class:`Profile` selection
    in :mod:`agent6.workflows.profiles`."""

    TRIVIAL = "trivial"
    """One-line bug fix, comment / type-hint touch-up, AGENTS.md edit."""

    SINGLE_STEP = "single"
    """One logical change that may touch a few files but has no internal
    decomposition the planner could meaningfully expose."""

    MULTI_STEP = "multi"
    """Two or more logical steps that benefit from a typed Plan."""

    EXPLORATION = "exploration"
    """Codebase is unfamiliar and the change needs reading before planning."""


class TaskClassification(BaseModel):
    """Output of the triage sub-agent."""

    model_config = _LLM_MODEL_CONFIG

    task_class: TaskClass
    reasoning: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


_SYSTEM = """You are the triage agent for a coding agent.

Classify the user's task into ONE of four classes. Return JSON only.

CLASSES (most-specific first; pick the FIRST that fits):

- trivial: one-line bug fix, typo, comment / docstring tweak, type-hint
  addition, a single import change, an AGENTS.md edit. The change is
  contained in one file and the operator could point at the exact lines.
  Examples: "fix off-by-one in factorial", "add return type to render()",
  "rename FOO_BAR to foo_bar in constants.py".

- single: one logical change that may touch a few files but has no
  internal decomposition worth surfacing as separate steps. The
  acceptance criterion is one sentence.
  Examples: "add --reverse flag to cat command", "make Calculator.divide
  raise on zero", "add a sleep_ms helper to utils.py and use it in retry".

- multi: two or more logical steps that benefit from a typed Plan with
  per-step verify between them. Cross-file refactors, feature additions
  with new tests, anything where a planner would normally produce >=2
  steps.
  Examples: "add a new --json output mode to all subcommands", "extract
  the storage layer into its own module", "implement the FooBar
  protocol across parser/lexer/codegen".

- exploration: the operator hasn't pinned down the change. The agent
  must read the codebase to find the right files, the right bug, or
  the right approach BEFORE planning.
  Examples: "find and fix the race in the curator", "audit our use of
  subprocess for shell injection", "make the test suite faster".

CONFIDENCE: 0.0 - 1.0. When the task could plausibly be either class,
pick the larger (more conservative) class and lower confidence. False
positives at trivial / single are expensive (the worker takes the whole
task in one shot, may fail verify); false positives at multi cost cheap
tokens (an opus planner on a one-line fix produces a one-step plan).

Output ONLY a JSON object matching the schema. Reasoning must be one
short sentence naming the deciding signal.
"""


def triage_classify(
    provider: Provider,
    *,
    user_task: str,
    agents_md: str,
    repo: RepoSummary,
) -> TaskClassification:
    """Classify ``user_task`` into a ``TaskClass``.

    The repo summary is included to give the model a sense of project
    size — a "refactor" in a 5-file repo is different from a "refactor"
    in a 500-file repo. AGENTS.md is included because it often pins down
    project conventions that disambiguate the class (e.g. an AGENTS.md
    that says "every PR is one logical change" pushes towards 'single').
    """
    user = (
        f"USER TASK:\n{user_task}\n\n"
        f"REPO:\n"
        f"  branch: {repo.branch}\n"
        f"  files: {repo.file_count}\n"
        f"  top-level: {', '.join(repo.top_level) or '(empty)'}\n\n"
        f"AGENTS.md:\n{agents_md or '(empty)'}\n"
    )
    return call_for_model(
        provider,
        system=_SYSTEM,
        user=user,
        output_model=TaskClassification,
        max_tokens=512,
    )
