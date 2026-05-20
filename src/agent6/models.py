# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pydantic models for LLM I/O — JSON-Schema source of truth.

These cross a trust boundary (LLM outputs), so pydantic is appropriate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_LLM_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class OpenQuestion(BaseModel):
    """One clarifying question the critic wants the user to answer.

    ``suggestions`` is a list of short candidate answers the user can
    pick from (by index) or override with free-form text. The list may
    be empty, in which case the user must type a free-form answer.
    """

    model_config = _LLM_MODEL_CONFIG

    question: str = Field(min_length=1, description="The clarifying question to ask the user.")
    suggestions: tuple[str, ...] = Field(
        default=(),
        description="Short candidate answers the user can select by index.",
    )


class RefinedSpec(BaseModel):
    """Output of the critic-of-prompt sub-agent."""

    model_config = _LLM_MODEL_CONFIG

    refined_task: str = Field(min_length=1, description="Restated, precise task.")
    open_questions: tuple[OpenQuestion, ...] = Field(
        default=(),
        description="Unresolved ambiguities the user must answer before planning.",
    )


class Step(BaseModel):
    """One unit of work in a Plan."""

    model_config = _LLM_MODEL_CONFIG

    title: str = Field(min_length=1, description="Imperative title used as commit subject.")
    rationale: str = Field(default="", description="Why this step is needed.")
    relevant_paths: tuple[str, ...] = Field(
        default=(),
        description=(
            "Repo-relative paths the worker will need to read. The workflow gathers these"
            " deterministically before the worker is invoked."
        ),
    )
    acceptance: str = Field(
        default="",
        description="Plain-language criteria the verify_command should make true.",
    )


class Plan(BaseModel):
    """Output of the planner sub-agent."""

    model_config = _LLM_MODEL_CONFIG

    summary: str = Field(min_length=1)
    steps: tuple[Step, ...] = Field(min_length=1, max_length=50)


class RunManifest(BaseModel):
    """Top-level metadata written into ``.agent6/runs/<id>/manifest.json``.

    Not LLM-sourced: agent6 writes this itself when a plan or run is created
    so downstream commands (``agent6 plan show``, ``revise``, ``edit``) can
    reconstruct the original Plan without rebuilding it from the task graph.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    kind: Literal["plan", "run"]
    created_at: str = Field(min_length=1, description="ISO-8601 UTC timestamp.")
    task: str = Field(min_length=1, description="Original user task string.")
    refined_task: str = Field(default="", description="Critic-refined task; empty for raw runs.")
    plan: Plan | None = None
    parent_run_id: str = Field(
        default="",
        description="Run id this one was derived from (revise/edit). Empty for fresh plans.",
    )


class FileEdit(BaseModel):
    """A single old-string → new-string replacement, or full-file create."""

    model_config = _LLM_MODEL_CONFIG

    path: str = Field(min_length=1, description="Repo-relative path.")
    kind: Literal["replace", "create"]
    old_string: str = Field(
        default="",
        description="Must be present and unique in the file when kind == 'replace'.",
    )
    new_string: str = Field(
        default="",
        description="Replacement (or full file content when kind == 'create').",
    )


class Edit(BaseModel):
    """Output of the worker sub-agent for one step."""

    model_config = _LLM_MODEL_CONFIG

    notes: str = Field(default="", description="Worker's notes on what it did.")
    edits: tuple[FileEdit, ...]


class Review(BaseModel):
    """Output of the reviewer sub-agent."""

    model_config = _LLM_MODEL_CONFIG

    verdict: Literal["pass", "fail"]
    comments: str = Field(default="")
    proposed_followup: str = Field(
        default="",
        description=(
            "Optional. When verdict == 'fail', a one-sentence concrete next "
            "action the Worker should try on retry (e.g. 'also update "
            "tests/foo.py to cover the new branch'). Empty on pass."
        ),
    )


class Summary(BaseModel):
    """Output of the summarizer sub-agent."""

    model_config = _LLM_MODEL_CONFIG

    summary: str = Field(min_length=1)


AlignmentAction = Literal["expand", "execute", "add_subtask", "skip", "resume"]
AlignmentVerdictName = Literal[
    "proceed",
    "reorder",
    "reject",
    "re-plan-subtree",
    "re-plan-root",
    "ask",
]


class AlignmentVerdict(BaseModel):
    """Output of the alignment-guard sub-agent.

    Verdicts:
      - proceed: no problem; carry on with the proposed_action.
      - reorder: keep the node but execute it in a different order;
        `suggested_reorder` lists sibling node ids in the desired new order.
      - reject: mark the node obsolete; do not execute.
      - re-plan-subtree: this subtree is invalid; planner should revise it.
      - re-plan-root: the entire plan is invalid; back to plan mode.
      - ask: surface to the user; do not act unilaterally.
    """

    model_config = _LLM_MODEL_CONFIG

    verdict: AlignmentVerdictName
    reasoning: str = Field(min_length=1)
    suggested_reorder: tuple[str, ...] = Field(default=())
