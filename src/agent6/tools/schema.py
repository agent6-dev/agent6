# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool input schemas — pydantic models converted to JSON Schema for Anthropic."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    TOOL_NAME: ClassVar[str] = ""
    TOOL_DESCRIPTION: ClassVar[str] = ""


class ReadFileInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "read_file"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Read a text file from the repository. `path` is repo-root-relative "
        "(e.g. 'src/foo.py', NOT '/abs/...' or './src/foo.py'). Returns the "
        "UTF-8 decoded contents. Optional `offset` (0-based line number to"
        " start at, default 0) and `limit` (max lines to return, default"
        " all). Fails when: path is outside the repo, file does not exist,"
        " file is not UTF-8 decodable, or file is binary. Use `outline`"
        " instead when you only need a file's structure, not every line."
    )

    path: str = Field(min_length=1)
    offset: int = Field(default=0, ge=0)
    limit: int | None = Field(default=None, gt=0)


class ListDirInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "list_dir"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "List immediate entries in a directory (non-recursive). `path` is "
        "repo-root-relative; defaults to '.'. Hidden entries (starting with "
        "'.') are excluded. Returns names with a trailing '/' for directories. "
        "For recursive listing, use `grep` with a permissive pattern instead."
    )

    path: str = Field(default=".")


class GrepInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "grep"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Search for a Python-flavor regex `pattern` in files under `path` "
        "(repo-root-relative, defaults to '.'). Recursive. Returns matching "
        "lines prefixed by file:line. The pattern is matched per line; use "
        "`^` / `$` for line anchors. Common usage: `pattern='def foo'` to "
        "find function definitions across the repo; `pattern='import "
        "requests'` for imports. For semantic identifier matching that "
        "excludes string/comment occurrences, prefer `find_definition` or "
        "`find_references`."
    )

    pattern: str = Field(min_length=1)
    path: str = Field(default=".")
    case_insensitive: bool = False


class ApplyEditInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "apply_edit"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Apply substring edits to one file. `edits` MUST be an ARRAY of"
        " objects (NOT a JSON-encoded string), each with three string"
        ' fields: {"kind": "replace"|"create", "old_string": "...",'
        ' "new_string": "..."}. Each edit\'s `old_string` MUST occur'
        " EXACTLY ONCE in the file (whitespace, indentation, and line"
        " endings must match byte-for-byte). If `old_string` is not"
        " unique, expand it with more surrounding context. If `old_string`"
        " is not found, the on-disk file content likely differs from what"
        " you expect - re-read with `read_file` before retrying. Set"
        " kind='create' (with empty `old_string`) to create a new file;"
        " `new_string` is then the full file content."
        " Pass `preview=true` for a dry-run: the unified diff and hunk"
        " count of the would-be change are returned WITHOUT touching disk."
        " Use this to sanity-check large or risky multi-edit calls before"
        " committing to them; defaults to false (apply directly)."
    )

    path: str = Field(min_length=1)
    edits: tuple[EditPair, ...] = Field(min_length=1)
    preview: bool = False


class ApplyPatchInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "apply_patch"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Apply a unified-diff patch to one file. Format: standard `diff -u` output"
        " beginning with `--- a/PATH` and `+++ b/PATH` headers followed by"
        " `@@ -L,N +L,N @@` hunks with ` `, `-`, `+` line prefixes. Context lines"
        " must match the on-disk file exactly (no fuzzy match). For multi-hunk"
        " changes to one file this is cheaper than several `apply_edit` calls"
        " because hunks are anchored by line number, not by unique substrings."
        " To create a new file use `--- /dev/null` as the source header."
        " File deletion is not supported. One file per call."
        " Pass `preview=true` for a dry-run: the diff and hunk count are"
        " echoed back WITHOUT writing to disk; defaults to false."
    )

    path: str = Field(min_length=1)
    patch: str = Field(min_length=1)
    preview: bool = False


class EditPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(pattern="^(replace|create)$")
    old_string: str
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
        "Run a command in the sandbox. argv must be an array of strings (no shell)."
        " Requires `run_commands` capability != 'no' in config; if 'ask', the user is prompted."
        " The jail's PATH is `/usr/bin:/bin` and only `/usr` is bind-mounted from the host —"
        " prefer absolute paths like `/usr/bin/python3` or `/usr/bin/pytest`. Bare `python`"
        " will fail on Debian/Ubuntu-style hosts where only `python3` exists."
        " Mutating git subcommands such as `checkout`, `reset`, `restore`, `clean`, `stash`,"
        " and `commit` are refused because `.git/` is protected; to undo a bad edit, read"
        " prior content with `git show HEAD:path/to/file` and restore it with `apply_patch`"
        " or `apply_edit`."
    )

    argv: tuple[str, ...] = Field(min_length=1)


class RunMetricInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "run_metric_command"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Run the user-declared metric command in the sandbox. Returns the "
        "command output plus the parsed score (the first capture group of "
        "the configured `pattern` regex, parsed as float). For metric-"
        "driven optimization runs: call this after each successful verify "
        "to see whether the change improved the score. Returns "
        "{returncode, stdout, stderr, duration_s, score}. `score` is null "
        "if the pattern didn't match or the capture group wasn't numeric. "
        "No arguments. Exposed to the agent loop so the agent itself "
        "decides when to verify."
    )


class FinishRunInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "finish_run"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Signal that the agent has completed its work and the workflow "
        "should exit cleanly. Call this when (a) the task is done and "
        "verify passes, or (b) the metric has plateaued and further work "
        "is unlikely to improve it, or (c) you are blocked and cannot "
        "make progress. `summary` is a one-paragraph description of what "
        "was done / left undone, surfaced to the operator. Do not call any "
        "other tools after finish_run."
    )

    summary: str = Field(min_length=1)
    result: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional structured JSON object. When the task instructs you to "
            "return data matching a named schema, put that object here; it is "
            "validated against the schema at the trust boundary."
        ),
    )


class FinishPlanningInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "finish_planning"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Signal that the planning pass is complete and the workflow should "
        "exit. Available ONLY in plan mode (`agent6 plan`); in execution "
        "mode use `finish_run` instead. `plan_markdown` is the full plan "
        "document (markdown) that gets saved to the run directory as "
        "`plan.md`. Include: a one-line `# Plan: <title>`, the original "
        "task, context discovered, an ordered task list with acceptance "
        "criteria, any open questions for the user as `**Q:** ...` blocks "
        "with blank `**A:**` lines, and the verification approach. The "
        "operator can edit this file (`agent6 plan --edit <run-id>`) to "
        "fill in answers, then hand it to `agent6 run --from-plan "
        "<run-id>` to start execution. `summary` is a one-paragraph "
        "description surfaced to the operator at exit. Do not call any "
        "other tools after finish_planning."
    )

    summary: str = Field(min_length=1)
    plan_markdown: str = Field(min_length=1)


# DAG-as-tool surface. Lets the agent maintain its own task
# breakdown in the persistent curator-backed graph. Survives crashes via
# .agent6/runs/<id>/graph.jsonl; operator can inspect via `agent6 watch`.
# DAG manipulation tools.
# directly through its planner/worker/critic pipeline.


class DagAddTaskInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "add_task"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Add a subtask to the persistent task graph. Use only when the task"
        " naturally decomposes into 3+ trackable steps; skip for one-shot"
        " or single-file work. `parent_id` attaches under an existing task"
        " (omit to attach under the run's root). `title` is a short"
        " imperative; `acceptance` is the verify-able condition."
        " Returns the new task's 26-char ULID."
    )

    title: str = Field(min_length=1)
    parent_id: str | None = None
    rationale: str = ""
    acceptance: str = ""
    relevant_paths: tuple[str, ...] = ()


class DagUpdateTaskInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "update_task"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Change a task's status. `id` is the ULID returned from add_task."
        " `status` is one of pending | in_progress | passed | failed |"
        " skipped | obsolete. Mark in_progress when starting a subtask and"
        " passed (only) after verify confirms it."
    )

    id: str = Field(min_length=26, max_length=26)
    status: str = Field(pattern="^(pending|in_progress|passed|failed|skipped|obsolete)$")
    note: str = ""


class DagSetCursorInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "set_cursor"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Move the DAG's 'current task' pointer to `id` (or null to clear)."
        " Purely organizational - shows up in the TUI and lets humans see"
        " what you're working on. Not required for resume (the workflow"
        " snapshots loop state separately)."
    )

    # audit finding: ULID is exactly 26 chars; match
    # update_task's enforcement. None still clears the cursor.
    id: str | None = Field(default=None, min_length=26, max_length=26)


class DagListTasksInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "list_tasks"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "List tasks in the DAG. Optional `status` filter (pending |"
        " in_progress | passed | failed | skipped | obsolete). Use to find"
        " a parent_id before add_task, or to check what's still pending."
        " Returns {id, parent_id, title, status, acceptance, relevant_paths}"
        " per task."
    )

    # audit finding: enforce the same status enum here that
    # update_task uses, so an agent typo gets a clear schema rejection
    # rather than a silently-empty result.
    status: str | None = Field(
        default=None,
        pattern="^(pending|in_progress|passed|failed|skipped|obsolete)$",
    )


class OutlineInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "outline"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "List top-level and nested definitions (functions / classes / structs "
        "/ enums) in ONE source file with their start line numbers. Tree-"
        "sitter backed, deterministic, cheap. Use this instead of `read_file` "
        "when you only need a file's shape (e.g. 'what classes are in "
        "core.py?'). Supported extensions: .py .rs .ts .tsx - other files "
        "return an empty list. Returns names + line numbers only; for "
        "function bodies use `read_file`."
    )

    path: str = Field(min_length=1)


class FindDefinitionInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "find_definition"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Find every declaration site of an identifier (function / class / "
        "struct / enum / type alias) across the project. Returns matches as "
        "(file path, line, kind). Tree-sitter backed; matches by exact name "
        "(not by type or scope). Common usage: locate where a symbol is "
        "defined before reading its file. Cheaper than `grep 'def foo'` "
        "because it excludes occurrences in strings, comments, and other "
        "identifier-shaped tokens that happen to share the name."
    )

    name: str = Field(min_length=1)


class FindReferencesInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "find_references"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Find every identifier occurrence of `name` across the project, "
        "including the definition site itself. Tree-sitter backed: matches "
        "inside strings and comments are excluded - vastly cleaner than "
        "plain `grep <name>`. Use this to enumerate all call sites of a "
        "function before renaming it or changing its signature. Caveat: "
        "this is text-level identifier matching, NOT semantic resolution. "
        "Unrelated `foo`s in unrelated scopes (e.g. a local var `foo` in "
        "one function and a top-level function `foo` in another) will all "
        "be returned; disambiguate by inspecting the surrounding context."
    )

    name: str = Field(min_length=1)


class FindDefinitionLspInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "find_definition_lsp"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Semantic 'go to definition' via a real Language Server "
        "(Astral's `ty`, Python-only for now). Pass `path` to the file "
        "where the symbol is referenced and `symbol` to the identifier "
        "name; the tool resolves the first whole-word occurrence and "
        "asks the LSP for its definition site(s). Use this over "
        "`find_definition` when you need scope-aware resolution (e.g. "
        "to disambiguate two `foo`s, or to follow `from x import foo` "
        "across modules). Falls back with an error if `ty` is "
        "unavailable; the tree-sitter `find_definition` is always "
        "available as a backup."
    )

    path: str = Field(min_length=1)
    symbol: str = Field(min_length=1)


class FindReferencesLspInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "find_references_lsp"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Semantic 'find all references' via a real Language Server "
        "(Astral's `ty`, Python-only for now). Pass `path` to a file "
        "where the symbol is defined or used and `symbol` to the "
        "identifier name; the tool resolves the first whole-word "
        "occurrence and asks the LSP for every reference, including "
        "cross-module ones discovered through import resolution. Use "
        "this over `find_references` when cross-file rename correctness "
        "matters: it ignores unrelated identifiers that share the same "
        "name. Falls back with an error if `ty` is unavailable."
    )

    path: str = Field(min_length=1)
    symbol: str = Field(min_length=1)


ApplyEditInput.model_rebuild()

ALL_TOOLS: tuple[type[_ToolInput], ...] = (
    ReadFileInput,
    ListDirInput,
    GrepInput,
    OutlineInput,
    FindDefinitionInput,
    FindReferencesInput,
    FindDefinitionLspInput,
    FindReferencesLspInput,
    ApplyEditInput,
    ApplyPatchInput,
    RunVerifyInput,
    RunCommandInput,
)

# Extra tools exposed only to the single-loop workflow (run_metric,
# finish_run, dag_*). Kept separate from ALL_TOOLS so the read-only
# ToolDispatcher surface used by tests and external callers does not
# advertise loop-only control tools.
# adds the DAG tools (add_task / update_task / set_cursor /
# list_tasks).
LOOP_EXTRA_TOOLS: tuple[type[_ToolInput], ...] = (
    RunMetricInput,
    FinishRunInput,
    DagAddTaskInput,
    DagUpdateTaskInput,
    DagSetCursorInput,
    DagListTasksInput,
)

# Tool list for plan mode (`agent6 plan`). Excludes the
# execution-mode terminal tool (`finish_run`) and the metric tool
# (planning never iterates a metric); adds `finish_planning` instead.
# Plan-mode also filters `apply_edit` / `apply_patch` out of `ALL_TOOLS`
# at the workflow layer so a planner cannot accidentally mutate source.
PLAN_EXTRA_TOOLS: tuple[type[_ToolInput], ...] = (
    DagAddTaskInput,
    DagUpdateTaskInput,
    DagSetCursorInput,
    DagListTasksInput,
    FinishPlanningInput,
)


def schemas_as_provider_tools() -> list[dict[str, Any]]:
    """Emit Anthropic-API-shape tool descriptors. (kept dict-typed to avoid circular import)"""
    out: list[dict[str, Any]] = []
    for cls in ALL_TOOLS:
        schema = cls.model_json_schema()
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
