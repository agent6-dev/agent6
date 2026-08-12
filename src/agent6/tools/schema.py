# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool input schemas, pydantic models converted to JSON Schema for Anthropic."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Annotated, Any, ClassVar, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from agent6.graph.models import NodeStatus
from agent6.types import session_kind

# Derived from the NodeStatus Literal so the task-status vocabulary has ONE
# owner (a new status can't silently drift the tool schema). Same order, so
# the LLM-facing pattern bytes are unchanged; pinned in
# tests/unit/test_tool_schema_wire.py.
_STATUS_PATTERN = f"^({'|'.join(get_args(NodeStatus))})$"

# A task id as the DAG tools accept it: ULIDs are exactly 26 chars.
Ulid = Annotated[str, StringConstraints(min_length=26, max_length=26)]


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    TOOL_NAME: ClassVar[str] = ""
    TOOL_DESCRIPTION: ClassVar[str] = ""


class ReadFileInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "read_file"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Read a UTF-8 text file. `path` is repo-root-relative (absolute only"
        " inside granted directories). start_line (1-based) and limit select a"
        " range. Very large files truncate (truncated: true); narrow the range"
        " to reach the rest. outline shows structure without content."
    )

    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    limit: int | None = Field(default=None, gt=0)


class Agent6DocsInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "agent6_docs"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Read agent6's OWN documentation to answer questions about how to USE "
        "agent6 (configuring providers/models, sandbox isolation, machines, the "
        "CLI, budgets, etc.). Call with an empty `name` to list the available "
        "docs, or set `name` to one of them (e.g. README, CONFIG, SECURITY, "
        "AGENTS, ARCHITECTURE) to read its markdown."
    )

    name: str = Field(default="")


class ListDirInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "list_dir"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "List immediate entries in a directory (non-recursive). `path` is "
        "repo-root-relative; defaults to '.'. Hidden entries (starting with "
        "'.') are included. Returns names with a trailing '/' for directories. "
        "For a recursive view, use `run_command` (e.g. `rg --files`, `find`)."
    )

    path: str = Field(default=".")


class ApplyEditInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "apply_edit"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Edit one file. `edits` is an array of {old_string, new_string, kind?}."
        " Each old_string must occur exactly once in the file, byte for byte;"
        " expand it with surrounding context if not unique, and re-read the"
        ' file if not found. kind="create" makes a new file: empty old_string,'
        " full content in new_string, the only edit in the array. preview=true"
        " returns the would-be diff without touching disk."
    )

    path: str = Field(min_length=1)
    edits: tuple[EditPair, ...] = Field(min_length=1)
    preview: bool = False

    @model_validator(mode="after")
    def _check_create_is_sole(self) -> ApplyEditInput:
        # `create` writes the whole file from `new_string`, so combining it
        # with other edits is nonsensical: the dispatcher's create branch only
        # guards "file already exists" for the FIRST edit, so a `create` placed
        # after a `replace` would skip that guard and silently overwrite the
        # file (discarding the prior edits). Require create to be the sole edit
        # and fail loud at the trust boundary instead.
        if len(self.edits) > 1 and any(e.kind == "create" for e in self.edits):
            raise ValueError(
                "kind='create' must be the only edit (it writes the entire file);"
                " do not combine it with other edits"
            )
        return self


class ApplyPatchInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "apply_patch"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Patch one file per call. Accepts a standard unified diff (`--- a/PATH`,"
        " `+++ b/PATH`, @@ hunks; `--- /dev/null` creates) or OpenAI's"
        " *** Begin/Update File/End Patch format. Context lines must match the"
        " file exactly; no deletion. `path` optional (taken from headers)."
        " preview=true echoes the diff without writing."
    )

    path: str = ""
    patch: str = Field(min_length=1)
    preview: bool = False


class EditPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Default to "replace": small models routinely omit the discriminator and
    # send a bare {old_string, new_string}, which pydantic otherwise rejects
    # with "Field required: kind". Replace
    # is the overwhelming-majority case; `create` must still be set explicitly
    # and `_check_shape` enforces its empty-old_string contract.
    kind: str = Field(default="replace", pattern="^(replace|create)$")
    old_string: str = ""
    new_string: str

    @model_validator(mode="after")
    def _check_shape(self) -> EditPair:
        # kind="replace" with an empty old_string would match anywhere (or
        # nowhere depending on str.count semantics); reject it loud so the
        # model gets a clear error instead of a silent corruption.
        if self.kind == "replace" and self.old_string == "":
            raise ValueError("old_string must be non-empty for kind='replace'")
        # kind="create" ignores old_string; reject a non-empty value to catch
        # the common LLM mistake of pasting context into the wrong field.
        if self.kind == "create" and self.old_string != "":
            raise ValueError("old_string must be empty for kind='create'")
        return self


class RunVerifyInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "run_verify_command"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Run the user-declared verify command in the sandbox. No arguments."
    )


class RunCommandInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "run_command"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Run a command in the sandbox. argv is an array of strings, no shell."
        " Requires the run_commands capability; 'ask' prompts the operator."
        " Under jailed isolation PATH is minimal; prefer absolute paths like"
        " /usr/bin/python3. A command still running at the check-in is handed"
        " back with returncode null, still_running true, and a background_id,"
        " and keeps running: poll with read_background, stop with"
        " stop_background, or continue working; output printed so far comes"
        " with the hand-back. background=true returns the handle at once."
        " Jailed commands in one run share the run's private network, so a"
        " server started here answers later commands. All background commands"
        " die when the run ends."
    )

    argv: tuple[str, ...] = Field(min_length=1)
    background: bool = False


class FetchInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "fetch"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Fetch an http(s) URL (GET). Returns status, headers, and body text"
        " (truncated at a cap). Requires network reach from this run."
    )

    url: str = Field(min_length=1)


class ReadSessionInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "read_session"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Read another session's transcript summary by id (or the latest when"
        " omitted). Use to continue or review earlier work; read-only."
    )

    id: str = ""
    query: str = ""
    max_chars: int = Field(default=20_000, ge=500, le=200_000)


class ReadBackgroundInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "read_background"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Read a background command's output. `background_id` from run_command;"
        " `offset` skips bytes already seen. Returns output, running state,"
        " and returncode when finished."
    )

    id: str = ""
    tail_lines: int = Field(default=200, ge=1, le=2000)
    # None = the operator's configured check-in; 0 = look without waiting. The
    # interval has ONE owner ([workflow].command_checkin_s), so the default is
    # resolved by the dispatcher rather than duplicated here.
    wait_s: float | None = Field(default=None, ge=0.0)


class StopBackgroundInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "stop_background"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Stop a background command by background_id. Returns its final output and returncode."
    )

    id: str


class RunMetricInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "run_metric_command"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Run the configured metric command. Returns the parsed score. The"
        " harness also runs it automatically after each verify-passing edit."
    )


class FinishSessionInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "finish_session"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "End the run cleanly. Call when the task is done and verify passes,"
        " the metric has plateaued, or you are blocked. summary: one"
        " paragraph for the operator on what was done and left undone. Call"
        " no tools after it."
    )

    summary: str = Field(min_length=1)
    result: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional JSON object. When the task names a result schema,"
            " return the matching object here; validated at the trust"
            " boundary."
        ),
    )
    stale_gate: str = Field(
        default="",
        description=(
            "Set only when the verify command no longer matches the task: it"
            " pins behaviour this run deliberately changed, or cannot run."
            " Give the command you believe is right; it records a proposal"
            " and never changes the gate or passes the run. A merely failing"
            " gate means fix the work."
        ),
    )


class FinishPlanningInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "finish_planning"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Signal that the planning pass is complete and the workflow should "
        "exit. Available ONLY in plan mode (`agent6 plan`); in execution "
        "mode use `finish_session` instead. `plan_markdown` is the full plan "
        "document (markdown) that gets saved to the run directory as "
        "`plan.md`. Include: a one-line `# Plan: <title>`, the original "
        "task, context discovered, an ordered task list with acceptance "
        "criteria, any open questions for the user as `**Q:** ...` blocks "
        "with blank `**A:**` lines, and the verification approach. The "
        "operator can edit this file (`agent6 plan edit <run-id>`) to "
        "fill in answers, then hand it to `agent6 run --from-plan "
        "<run-id>` to start execution. `summary` is a one-paragraph "
        "description surfaced to the operator at exit. Do not call any "
        "other tools after finish_planning."
    )

    # Per-field descriptions so the disambiguation lives IN the JSON schema the
    # model fills, not only in the prose above. finish_planning is the one finish
    # tool whose fields were self-undocumented, and models put the whole plan
    # into `summary` (listed first, and a natural sink for "primary output"),
    # leaving a degenerate plan.md that still passed min_length=1. finish_session's
    # `result` already carries a field description; this matches it.
    summary: str = Field(
        min_length=1,
        description=(
            "A one-paragraph description of the plan, surfaced to the operator at "
            "exit. This is NOT the plan itself -- the full plan goes in plan_markdown."
        ),
    )
    plan_markdown: str = Field(
        min_length=1,
        description=(
            "The FULL plan document in markdown, saved verbatim to plan.md and fed "
            "to `agent6 run --from-plan`. This is the deliverable: put the entire "
            "plan here (title, task, context, ordered task list with acceptance "
            "criteria, open questions, verification) -- not a short blurb, and not "
            "in summary."
        ),
    )


# DAG-as-tool surface. Lets the agent maintain its own task
# breakdown in the persistent curator-backed graph. Survives crashes via
# <run-dir>/graph.jsonl; operator can inspect via `agent6 attach`.


class DagAddTaskInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "add_task"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Add a subtask to the persistent task graph; skip for one-shot work."
        " parent_id attaches under an existing task (default the root). title"
        " is a short imperative; acceptance the verifiable condition. after"
        " inserts directly after that sibling. depends_on lists task ULIDs"
        " that must pass first. standing=true marks the run's never-finishing"
        " fallback goal: worked when nothing else is ready, never passes,"
        " retired with skipped/obsolete. Returns the new task's ULID."
    )

    title: str = Field(min_length=1)
    # ULID is exactly 26 chars, like update_task; None still means
    # "under the run root". "" silently attached to root before the constraint.
    parent_id: str | None = Field(default=None, min_length=26, max_length=26)
    # A sibling under the same parent; the task lands right after it.
    after: str | None = Field(default=None, min_length=26, max_length=26)
    rationale: str = ""
    acceptance: str = ""
    relevant_paths: tuple[str, ...] = ()
    depends_on: tuple[Ulid, ...] = ()
    standing: bool = False


class DagUpdateTaskInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "update_task"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Update a task: status (in_progress moves focus; passed only after"
        " verify confirms), title, acceptance, or depends_on (task ULIDs that"
        " must pass first). Fields omitted stay unchanged."
    )

    id: str = Field(min_length=26, max_length=26)
    status: str | None = Field(default=None, pattern=_STATUS_PATTERN)
    note: str = ""
    depends_on: tuple[Ulid, ...] = ()


class DagListTasksInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "list_tasks"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "List the task graph: ids, titles, statuses, dependencies, and the current focus."
    )

    # The same status enum update_task uses, so a typo is a schema rejection
    # rather than an empty result.
    status: str | None = Field(default=None, pattern=_STATUS_PATTERN)


# Cross-run memory surface. Lets the agent persist repo-scoped notes
# (agent6.memory store under <state_dir>/memories/) that future runs see in
# the <memories> system-prompt block. Run mode only (LOOP_EXTRA_TOOLS).


class UseSkillInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "use_skill"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Load an installed skill's full instructions by name (from the"
        " <skills> index) and follow them."
    )

    name: str = Field(min_length=1, max_length=100)
    file: str | None = Field(default=None, min_length=1, max_length=300)


class OutlineInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "outline"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Structural outline of a source file: top-level and nested defs,"
        " classes, and their line ranges. Cheaper than reading the file when"
        " you need shape, not content."
    )

    path: str = Field(min_length=1)


class FindDefinitionInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "find_definition"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Find where a symbol is defined (tree-sitter; excludes strings and"
        " comments). Returns file:line with a snippet. Cheaper than grep for"
        " symbols."
    )

    symbol: str = Field(min_length=1)


class FindReferencesInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "find_references"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "List references to a symbol across the repo (tree-sitter; excludes"
        " strings and comments). Returns file:line rows with a snippet."
    )

    symbol: str = Field(min_length=1)


class UserQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1)
    options: tuple[str, ...] = ()


class AskUserInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "ask_user"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Ask the operator and wait. Use for decisions the repo and task cannot"
        " settle, or when the task says to check with the operator; a question"
        " written as plain text is never seen. `questions` is an array of"
        " {question, options?}; give 2-4 options for a choice (free text is"
        " always allowed); batch related questions into one call. Returns"
        " {answers: [...]} aligned to questions. Headless runs with nobody"
        " watching return empty answers."
    )

    questions: tuple[UserQuestion, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_single_question(cls, data: Any) -> Any:
        # A model that sends a lone question flat (question=..., options=...) rather
        # than wrapping it in `questions` still works -- fold it into the list.
        if isinstance(data, dict) and "questions" not in data and "question" in data:
            q: dict[str, Any] = {"question": data.get("question")}
            if "options" in data:
                q["options"] = data.get("options")
            data = {k: v for k, v in data.items() if k not in ("question", "options")}
            data["questions"] = [q]
        return data


ApplyEditInput.model_rebuild()

ALL_TOOLS: tuple[type[_ToolInput], ...] = (
    ReadFileInput,
    ListDirInput,
    OutlineInput,
    FindDefinitionInput,
    FindReferencesInput,
    ApplyEditInput,
    ApplyPatchInput,
    RunVerifyInput,
    RunCommandInput,
    ReadSessionInput,
    FetchInput,
    ReadBackgroundInput,
    StopBackgroundInput,
)

# Extra tools exposed only to the single-loop workflow (run_metric,
# finish_session, dag_*, memory). Kept separate from ALL_TOOLS so the read-only
# ToolDispatcher surface used by tests and external callers does not
# advertise loop-only control tools.
LOOP_EXTRA_TOOLS: tuple[type[_ToolInput], ...] = (
    RunMetricInput,
    FinishSessionInput,
    AskUserInput,
    DagAddTaskInput,
    DagUpdateTaskInput,
    DagListTasksInput,
    # Operator-installed skills (hidden by the dispatcher when none are
    # installed or [skills].enabled is off).
    UseSkillInput,
)

# Tool list for plan mode (`agent6 plan`). Excludes the
# execution-mode terminal tool (`finish_session`) and the metric tool
# (planning never iterates a metric); adds `finish_planning` instead.
# Plan-mode also filters `apply_edit` / `apply_patch` out of `ALL_TOOLS`
# at the workflow layer so a planner cannot accidentally mutate source.
PLAN_EXTRA_TOOLS: tuple[type[_ToolInput], ...] = (
    DagAddTaskInput,
    DagUpdateTaskInput,
    DagListTasksInput,
    FinishPlanningInput,
)

# Tool list for ask mode (`agent6 ask`). Edit-free Q&A: like plan it filters
# `apply_edit`/`apply_patch` out of `ALL_TOOLS` at the workflow layer, and it
# exposes NO control tools (no DAG, no finish_planning, no finish_session -- the
# agent answers by emitting its final message as prose, a "silent finish"). It
# DOES add `agent6_docs` so it can answer "how do I use agent6" questions.
ASK_EXTRA_TOOLS: tuple[type[_ToolInput], ...] = (Agent6DocsInput,)

# Tool list for machine-authoring mode (`agent6 machine create`). The agent's
# only deliverable is a `.asm.toml` returned via `finish_session`'s `result.toml`,
# so it gets read-only navigation (in case the task references existing files)
# plus `finish_session`, no edit/patch/verify/run_command/DAG/metric tools, which
# only tempt a weak model into writing the file or spelunking the repo.
MACHINE_EXTRA_TOOLS: tuple[type[_ToolInput], ...] = (FinishSessionInput,)


@dataclass(frozen=True, slots=True)
class ModeTools:
    """One mode's LLM tool surface: `base` (ALL_TOOLS minus the mode's blocked
    mutators) plus `extras` (its control tools). `tool_definitions` exposes
    exactly `base + extras`; the dispatcher refuses names outside `permitted`
    as its backstop, so exposure and enforcement cannot drift apart.
    `permitted` is `names` plus agent6_docs, which is exposed only in ask
    (elsewhere it is tool-list noise) but safe to execute anywhere -- a
    read-only doc fetch the in-loop review seat uses in every mode."""

    base: tuple[type[_ToolInput], ...]
    extras: tuple[type[_ToolInput], ...]
    names: frozenset[str]
    permitted: frozenset[str]


# The mode-specific additions. Everything else about a mode is read off its
# `SessionKind`; these are the one thing a record cannot carry, being tool
# classes this module defines.
_EXTRA_TOOLS: dict[str, tuple[type[_ToolInput], ...]] = {
    "plan": PLAN_EXTRA_TOOLS,
    "ask": ASK_EXTRA_TOOLS,
    "machine": MACHINE_EXTRA_TOOLS,
    "agent": MACHINE_EXTRA_TOOLS,
}


@cache
def mode_tools(mode: str) -> ModeTools:
    kind = session_kind(mode)
    extras = _EXTRA_TOOLS.get(mode, LOOP_EXTRA_TOOLS)
    blocked: set[str] = set()
    if not kind.edits:
        # Read-only modes: no in-process file mutation.
        blocked = {ApplyEditInput.TOOL_NAME, ApplyPatchInput.TOOL_NAME}
    if not kind.runs_commands:
        # Machine authoring / agent states additionally never run commands:
        # the deliverable is the finish_session payload, and command tools only
        # tempt a weak model into spelunking.
        # `ask` keeps run_command for read-only, approval-gated investigation.
        # read_session and fetch go with them: a machine state answers about
        # ITS input, so this project's run history is not its business, and
        # neither is the network -- a deliverable assembled from a page the
        # state fetched is not the deliverable the operator asked for.
        blocked |= {
            RunVerifyInput.TOOL_NAME,
            RunCommandInput.TOOL_NAME,
            ReadSessionInput.TOOL_NAME,
            FetchInput.TOOL_NAME,
        }
    if not kind.edits:
        # Only a session that edits owns a background command's lifetime: every
        # other mode is a short read-only pass, and a command killed at its end
        # would be started for nothing.
        blocked |= {
            ReadBackgroundInput.TOOL_NAME,
            StopBackgroundInput.TOOL_NAME,
        }
    base = tuple(cls for cls in ALL_TOOLS if cls.TOOL_NAME not in blocked)
    names = frozenset(cls.TOOL_NAME for cls in (*base, *extras))
    return ModeTools(
        base=base,
        extras=extras,
        names=names,
        permitted=names | {Agent6DocsInput.TOOL_NAME},
    )


def _strip_titles(node: Any) -> Any:
    """Drop pydantic's auto "title" keys: they duplicate the field name in
    Title Case and carry no signal on the wire (~1.6k chars across the
    surface)."""
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def schemas_as_provider_tools() -> list[dict[str, Any]]:
    """Emit Anthropic-API-shape tool descriptors. (kept dict-typed to avoid circular import)"""
    out: list[dict[str, Any]] = []
    for cls in ALL_TOOLS:
        schema = _strip_titles(cls.model_json_schema())
        # Anthropic wants the schema directly, not wrapped, with "type" present.
        schema.setdefault("type", "object")
        out.append(
            {
                "name": cls.TOOL_NAME,
                "description": cls.TOOL_DESCRIPTION,
                "input_schema": schema,
            }
        )
    return out
