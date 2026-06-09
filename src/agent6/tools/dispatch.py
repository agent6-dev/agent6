# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool dispatch: validates incoming LLM tool calls and executes them.

All filesystem reads/writes are clamped to *root* (the repo cwd). All command
execution goes through agent6.sandbox.jail.run_in_jail. Capability gating
(`run_commands = "no" | "ask" | "yes"`) is enforced here.
"""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agent6.config import Config
from agent6.events import EventSink
from agent6.graph.models import (
    AddSubtaskIntent,
    SetCursorIntent,
    TaskNodeDraft,
    UpdateStatusIntent,
)
from agent6.paths import agent6_dir as _agent6_dir_path
from agent6.sandbox.jail import JailUnavailableError, run_in_jail
from agent6.tools.index import Symbol, SymbolIndex
from agent6.tools.lsp import LspClient, LspError
from agent6.tools.mcp_client import MCP_TOOL_PREFIX, MCPError, MCPManager
from agent6.tools.patch_apply import PatchError, apply_patch_text
from agent6.tools.schema import (
    ALL_TOOLS,
    ApplyEditInput,
    ApplyPatchInput,
    DagAddTaskInput,
    DagListTasksInput,
    DagSetCursorInput,
    DagUpdateTaskInput,
    FindDefinitionInput,
    FindDefinitionLspInput,
    FindReferencesInput,
    FindReferencesLspInput,
    FinishPlanningInput,
    FinishRunInput,
    GrepInput,
    ListDirInput,
    OutlineInput,
    ReadFileInput,
    RunCommandInput,
    RunMetricInput,
    RunVerifyInput,
)
from agent6.types import CommandResult, JailPolicy, SandboxProfile


class ToolError(Exception):
    """The LLM tried something the tool layer refused."""


_GIT_MUTATING_SUBCOMMANDS = frozenset(
    {
        "add",
        "am",
        "checkout",
        "cherry-pick",
        "clean",
        "commit",
        "merge",
        "mv",
        "pull",
        "push",
        "rebase",
        "reset",
        "restore",
        "revert",
        "rm",
        "stash",
        "switch",
    }
)
_GIT_GLOBAL_OPTIONS_WITH_VALUE = frozenset(
    {
        "-C",
        "-c",
        "--exec-path",
        "--git-dir",
        "--namespace",
        "--super-prefix",
        "--work-tree",
    }
)
_GIT_GLOBAL_OPTIONS_WITH_VALUE_PREFIXES = tuple(
    f"{opt}=" for opt in _GIT_GLOBAL_OPTIONS_WITH_VALUE if opt.startswith("--")
)


def _git_subcommand(argv: tuple[str, ...]) -> str | None:
    if not argv or Path(argv[0]).name != "git":
        return None
    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--":
            return None
        if arg in _GIT_GLOBAL_OPTIONS_WITH_VALUE:
            idx += 2
            continue
        if arg.startswith(_GIT_GLOBAL_OPTIONS_WITH_VALUE_PREFIXES):
            idx += 1
            continue
        if arg.startswith("-"):
            idx += 1
            continue
        return arg
    return None


def _refuse_mutating_git_command(argv: tuple[str, ...]) -> None:
    subcommand = _git_subcommand(argv)
    if subcommand not in _GIT_MUTATING_SUBCOMMANDS:
        return
    raise ToolError(
        f"run_command refuses mutating git subcommand `git {subcommand}` because "
        ".git/ is protected inside the jail. For revert/recovery, inspect prior "
        "content with `git show HEAD:path/to/file`, then restore it with "
        "apply_patch or apply_edit. Read-only git commands such as status, diff, "
        "show, and log are still allowed."
    )


class _Approver(Protocol):
    def __call__(self, prompt: str, /) -> bool: ...


def _default_approver(prompt: str) -> bool:  # pragma: no cover — interactive
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


@dataclass(frozen=True, slots=True)
class _SafePath:
    abs_path: Path
    rel_path: Path


def _resolve_in_root(root: Path, candidate: str) -> _SafePath:
    """Resolve *candidate* relative to *root* and ensure it stays inside *root*."""
    if candidate.startswith("/"):
        raise ToolError(f"Absolute paths not allowed: {candidate!r}")
    parts = Path(candidate).parts
    if ".." in parts:
        raise ToolError(f"Path contains '..': {candidate!r}")
    abs_path = (root / candidate).resolve()
    try:
        rel = abs_path.relative_to(root.resolve())
    except ValueError as exc:
        raise ToolError(f"Path escapes repo root: {candidate!r}") from exc
    return _SafePath(abs_path=abs_path, rel_path=rel)


def _refuse_protected_write(
    candidate: str, dir_name: str, *, why: str, resolved: _SafePath | None = None
) -> None:
    """Refuse an in-process ``apply_edit`` / ``apply_patch`` into a protected
    top-level directory.

    Two dirs are protected. ``.agent6`` (or whatever ``[agent6].workspace_subdir``
    renamed it to): the DAG curator writes ``graph.jsonl`` there, the event sink
    ``logs.jsonl``, the transcript sink ``transcripts/*`` -- a stray write would
    break the resumable-DAG moat. ``.git`` (when ``protect_git``): the edit tools
    write **in-process, outside the jail**, so without this an LLM could create
    or rewrite ``.git/hooks/*`` or ``.git/config`` (e.g. ``core.fsmonitor``) and
    get code executed outside the sandbox on the next ``git`` invocation, or
    corrupt git history -- defeating ``protect_git`` entirely (the jail's RO bind
    of ``.git`` never covers these in-process writes). Reads stay allowed.

    Checks both the raw candidate string AND the post-symlink-resolution relative
    path, so a symlink ``./decoy -> .git`` can't launder a write past the prefix
    check.
    """
    parts = Path(candidate).parts
    if parts and parts[0] == dir_name:
        raise ToolError(f"Refusing to write under {dir_name}/ ({why}): {candidate!r}")
    if resolved is not None:
        rel_parts = resolved.rel_path.parts
        if rel_parts and rel_parts[0] == dir_name:
            raise ToolError(
                f"Refusing to write under {dir_name}/ ({why}) via symlink: {candidate!r} "
                f"resolves to {resolved.rel_path!s}"
            )


def _preview_result(
    path: str,
    old_text: str | None,
    new_text: str,
    *,
    applied: list[str] | None = None,
) -> dict[str, Any]:
    """Build the dry-run response for ``apply_edit``/``apply_patch`` with
    ``preview=true``. Returns the unified diff (old vs new) and a hunk
    count, but does NOT write anything to disk.

    Lets the agent sanity-check a complex multi-edit call
    before committing to it. Diff is bounded so a preview of a 100k-line
    rewrite doesn't dump the whole file back into the conversation.
    """
    old_lines = (old_text or "").splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    label_a = "/dev/null" if old_text is None else f"a/{path}"
    label_b = f"b/{path}"
    diff_iter = difflib.unified_diff(old_lines, new_lines, fromfile=label_a, tofile=label_b, n=3)
    diff = "".join(diff_iter)
    hunks = sum(1 for line in diff.splitlines() if line.startswith("@@ "))
    truncated = False
    _MAX_DIFF_CHARS = 8000
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + f"\n... <truncated {len(diff) - _MAX_DIFF_CHARS} chars>\n"
        truncated = True
    result: dict[str, Any] = {
        "preview": True,
        "path": path,
        "diff": diff or "(no changes)",
        "hunks": hunks,
        "bytes_before": len(old_text or ""),
        "bytes_after": len(new_text),
        "truncated": truncated,
    }
    if applied is not None:
        result["would_apply"] = applied
    return result


class ToolDispatcher:
    """Runtime tool dispatcher. Constructed once per workflow run."""

    def __init__(
        self,
        *,
        root: Path,
        config: Config,
        sandbox_profile: SandboxProfile = "strict",
        approver: _Approver | None = None,
        events: EventSink | None = None,
        graph_client: object | None = None,
        run_root_node_id: str | None = None,
        mcp_manager: MCPManager | None = None,
        extra_protect_paths: tuple[Path, ...] = (),
    ) -> None:
        self._root = root.resolve()
        self._config = config
        self._sandbox_profile: SandboxProfile = sandbox_profile
        # Extra read-only paths layered into every run_command jail on top of
        # protect_git/protect_agent6 (e.g. a running machine's own .asm.toml +
        # scripts bundle, so an agent state can't rewrite them mid-run).
        self._extra_protect_paths = extra_protect_paths
        self._approver: _Approver = approver or _default_approver
        self._events = events
        # Optional GraphClient + root-task id for the DAG-as-tool
        # surface. When wired, the dispatcher exposes add_task /
        # update_task / set_cursor / list_tasks. Typed as `object` to
        # avoid a circular import (agent6.graph.client depends on
        # agent6.graph.models which is upstream of dispatch in the tach
        # graph).
        self._graph_client = graph_client
        self._run_root_node_id = run_root_node_id
        # Optional MCP (Model Context Protocol) manager. When
        # set, ``dispatch`` routes any tool name starting with the MCP
        # prefix to the manager. Discovered tool names are also added
        # to ``available_tool_names()`` so the workflow exposes them.
        self._mcp_manager = mcp_manager
        # Name of the in-repo agent6 dir (``.agent6`` or the
        # ``[agent6].workspace_subdir`` rename). Used by the write-refusal
        # guard and the jail protect-paths so both track the configured name.
        self._agent6_dir_name = _agent6_dir_path(self._root, config.agent6.workspace_subdir).name
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            ReadFileInput.TOOL_NAME: self._read_file,
            ListDirInput.TOOL_NAME: self._list_dir,
            GrepInput.TOOL_NAME: self._grep,
            OutlineInput.TOOL_NAME: self._outline,
            FindDefinitionInput.TOOL_NAME: self._find_definition,
            FindReferencesInput.TOOL_NAME: self._find_references,
            FindDefinitionLspInput.TOOL_NAME: self._find_definition_lsp,
            FindReferencesLspInput.TOOL_NAME: self._find_references_lsp,
            ApplyEditInput.TOOL_NAME: self._apply_edit,
            ApplyPatchInput.TOOL_NAME: self._apply_patch,
            RunVerifyInput.TOOL_NAME: self._run_verify,
            RunCommandInput.TOOL_NAME: self._run_command,
            # run_metric: LLM-exposed via LOOP_EXTRA_TOOLS so the
            # loop can call it after a successful verify when
            # [workflow.metric] is configured.
            RunMetricInput.TOOL_NAME: self._run_metric,
            # finish_run signals the loop should exit. Handler
            # just echoes the summary; the workflow checks for this tool name
            # in resp.tool_uses and terminates after dispatching it.
            FinishRunInput.TOOL_NAME: self._finish_run,
            FinishPlanningInput.TOOL_NAME: self._finish_planning,
            # DAG-as-tool. Handlers raise ToolError if no graph_client was
            # wired (so standalone tests can omit it).
            DagAddTaskInput.TOOL_NAME: self._dag_add_task,
            DagUpdateTaskInput.TOOL_NAME: self._dag_update_task,
            DagSetCursorInput.TOOL_NAME: self._dag_set_cursor,
            DagListTasksInput.TOOL_NAME: self._dag_list_tasks,
        }
        self._available = {cls.TOOL_NAME for cls in ALL_TOOLS}
        self._index: SymbolIndex | None = None
        # Lazy LSP client for find_*_lsp tools. Spawned on
        # first use, killed by close(). Outside the jail, same trust
        # boundary as the tree-sitter index.
        self._lsp: LspClient | None = None

    @property
    def root(self) -> Path:
        return self._root

    def set_run_root_node_id(self, node_id: str | None) -> None:
        """Workflow sets this after seeding the run's root task.
        ``add_task`` with parent_id=None falls back to this as the parent."""
        self._run_root_node_id = node_id

    def available_tool_names(self) -> tuple[str, ...]:
        # run_command is filtered out if disabled.
        names = list(self._available)
        if self._config.sandbox.run_commands == "no":
            names = [n for n in names if n != RunCommandInput.TOOL_NAME]
        # Bench / A-B harness: hide the tree-sitter index tools when this env
        # var is set so we can compare cost/quality with and without them
        # without rebuilding agent6.
        if os.environ.get("AGENT6_DISABLE_INDEX_TOOLS") == "1":
            hidden = {
                OutlineInput.TOOL_NAME,
                FindDefinitionInput.TOOL_NAME,
                FindReferencesInput.TOOL_NAME,
            }
            names = [n for n in names if n not in hidden]
        # Bench probe for the "tool-surface fit"
        # hypothesis. Hide `apply_edit` so the only edit primitive is
        # `apply_patch` (unified-diff). Lets us measure whether models
        # that look weak on agent6's diff-style search-and-replace
        # surface improve when handed a patch tool instead. No-op when
        # unset (default keeps both tools available).
        if os.environ.get("AGENT6_DISABLE_APPLY_EDIT") == "1":
            names = [n for n in names if n != ApplyEditInput.TOOL_NAME]
        if self._mcp_manager is not None:
            names.extend(d.qualified_name for d in self._mcp_manager.descriptors())
        return tuple(sorted(names))

    def dispatch(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
        # MCP routing happens BEFORE the built-in handler
        # check so mcp__* names don't collide with the built-in
        # "Unknown tool" error path.
        if name.startswith(MCP_TOOL_PREFIX):
            if self._mcp_manager is None:
                raise ToolError(f"{name}: MCP is not configured")
            self._emit("tool.call", name=name, args=_truncate_args(raw_input))
            try:
                result = self._mcp_manager.call(name, raw_input)
            except MCPError as exc:
                self._emit("tool.result", name=name, ok=False, summary=str(exc))
                raise ToolError(str(exc)) from exc
            self._emit(
                "tool.result",
                name=name,
                ok=True,
                summary=_summarize_result(name, result),
            )
            return result
        if name not in self._handlers:
            raise ToolError(f"Unknown tool: {name}")
        if name == RunCommandInput.TOOL_NAME and self._config.sandbox.run_commands == "no":
            raise ToolError("run_command is disabled by config (run_commands = 'no')")
        if os.environ.get("AGENT6_DISABLE_INDEX_TOOLS") == "1" and name in {
            OutlineInput.TOOL_NAME,
            FindDefinitionInput.TOOL_NAME,
            FindReferencesInput.TOOL_NAME,
        }:
            raise ToolError(f"{name} is disabled (AGENT6_DISABLE_INDEX_TOOLS=1)")
        if os.environ.get("AGENT6_DISABLE_APPLY_EDIT") == "1" and name == ApplyEditInput.TOOL_NAME:
            raise ToolError(
                f"{name} is disabled (AGENT6_DISABLE_APPLY_EDIT=1); use apply_patch instead"
            )
        self._emit("tool.call", name=name, args=_truncate_args(raw_input))
        try:
            result = self._handlers[name](raw_input)
        except ToolError as exc:
            self._emit("tool.result", name=name, ok=False, summary=str(exc))
            raise
        except Exception as exc:
            self._emit("tool.result", name=name, ok=False, summary=str(exc))
            raise ToolError(f"{name} failed: {exc}") from exc
        self._emit("tool.result", name=name, ok=True, summary=_summarize_result(name, result))
        return result

    def _emit(self, event_type: str, /, **fields: Any) -> None:
        if self._events is not None:
            self._events.emit(event_type, **fields)

    # ----- handlers -----

    def _read_file(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = ReadFileInput.model_validate(raw)
        sp = _resolve_in_root(self._root, args.path)
        if not sp.abs_path.is_file():
            raise ToolError(f"Not a file: {args.path}")
        try:
            full = sp.abs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"File is not UTF-8 text: {args.path}") from exc
        if args.offset == 0 and args.limit is None:
            return {"content": full, "size": len(full), "lines_total": full.count("\n") + 1}
        lines = full.splitlines(keepends=True)
        end = len(lines) if args.limit is None else min(len(lines), args.offset + args.limit)
        slice_text = "".join(lines[args.offset : end])
        return {
            "content": slice_text,
            "size": len(slice_text),
            "lines_total": len(lines),
            "offset": args.offset,
            "lines_returned": end - args.offset,
        }

    def _list_dir(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = ListDirInput.model_validate(raw)
        sp = _resolve_in_root(self._root, args.path)
        if not sp.abs_path.is_dir():
            raise ToolError(f"Not a directory: {args.path}")
        entries: list[str] = []
        for entry in sorted(sp.abs_path.iterdir()):
            suffix = "/" if entry.is_dir() else ""
            entries.append(entry.name + suffix)
        return {"entries": entries}

    def _grep(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = GrepInput.model_validate(raw)
        sp = _resolve_in_root(self._root, args.path)
        try:
            pat = re.compile(args.pattern, re.IGNORECASE if args.case_insensitive else 0)
        except re.error as exc:
            raise ToolError(f"Invalid regex: {exc}") from exc
        hits: list[dict[str, Any]] = []
        targets: list[Path]
        # Skip hidden files/dirs (.git, .agent6, ...) only when they are BELOW
        # the requested path, so an explicit `grep <pat> .github/` (or a hidden
        # file named directly) is still searched. `skip_base` is the requested
        # directory; for an explicitly-named file there is nothing to skip.
        skip_base: Path | None
        if sp.abs_path.is_file():
            targets = [sp.abs_path]
            skip_base = None
        else:
            targets = [p for p in sp.abs_path.rglob("*") if p.is_file()]
            skip_base = sp.abs_path
        for path in targets:
            if skip_base is not None and any(
                part.startswith(".") for part in path.relative_to(skip_base).parts
            ):
                continue
            try:
                for lineno, line in enumerate(
                    path.read_text(encoding="utf-8", errors="ignore").splitlines(),
                    start=1,
                ):
                    if pat.search(line):
                        hits.append(
                            {
                                "path": str(path.relative_to(self._root)),
                                "line": lineno,
                                "text": line[:500],
                            }
                        )
                        if len(hits) >= 500:
                            return {"hits": hits, "truncated": True}
            except OSError:
                continue
        return {"hits": hits, "truncated": False}

    def _refuse_protected_writes(self, path: str, resolved: _SafePath | None = None) -> None:
        """Apply every protected-dir write guard for an in-process edit."""
        _refuse_protected_write(
            path, self._agent6_dir_name, why="agent6 run state", resolved=resolved
        )
        if self._config.sandbox.protect_git:
            _refuse_protected_write(path, ".git", why="git history/metadata", resolved=resolved)

    def _apply_edit(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = ApplyEditInput.model_validate(raw)
        self._refuse_protected_writes(args.path)
        sp = _resolve_in_root(self._root, args.path)
        self._refuse_protected_writes(args.path, sp)
        # Write-outside-cwd is enforced by _resolve_in_root already (root == cwd).
        applied: list[str] = []
        existing = sp.abs_path.read_text(encoding="utf-8") if sp.abs_path.exists() else None
        new_content = existing
        for i, edit in enumerate(args.edits):
            if edit.kind == "create":
                if existing is not None and i == 0:
                    raise ToolError(f"create requested but file already exists: {args.path}")
                new_content = edit.new_string
                applied.append("create")
            else:
                if new_content is None:
                    raise ToolError(f"replace requested but file does not exist: {args.path}")
                occurrences = new_content.count(edit.old_string)
                if occurrences == 0:
                    # The previous error format wrapped the
                    # full file body in literal `---BEGIN <path>---` /
                    # `---END <path>---` markers. Models that degenerate on
                    # repetitive content (observed live with Kimi K2.6) end
                    # up copying those scaffolding markers verbatim into the
                    # next `old_string` payload, which then guarantees another
                    # mismatch and feeds an infinite hallucination loop. The
                    # new format is "shape only": file size + line count +
                    # head/tail snippet, with NO wrapping markers and an
                    # explicit instruction to re-read for the full body. Keeps
                    # the worker honest about what's actually on disk without
                    # giving it text it can plagiarise.
                    lines = new_content.splitlines()
                    head = "\n".join(lines[:5])
                    tail = "\n".join(lines[-5:]) if len(lines) > 10 else ""
                    snippet = (
                        f"file size: {len(new_content)} bytes, "
                        f"{len(lines)} lines\n"
                        f"first 5 lines:\n{head}"
                    )
                    if tail:
                        snippet += f"\n...\nlast 5 lines:\n{tail}"
                    raise ToolError(
                        f"old_string not found in {args.path} (edit #{i}). "
                        f"Your old_string does not match the actual file "
                        f"content byte-for-byte. Re-read the file with "
                        f"read_file to get the current full content, then "
                        f"retry with a shorter, uniquely-anchored old_string. "
                        f"File shape (for orientation only — do NOT use as "
                        f"old_string verbatim):\n{snippet}"
                    )
                if occurrences > 1:
                    raise ToolError(
                        f"old_string is not unique in {args.path} "
                        f"(edit #{i}, {occurrences} matches); add more surrounding "
                        f"context to make it unique"
                    )
                new_content = new_content.replace(edit.old_string, edit.new_string, 1)
                applied.append("replace")
        if new_content is None:
            raise ToolError("No content to write")
        if args.preview:
            return _preview_result(args.path, existing, new_content, applied=applied)
        sp.abs_path.parent.mkdir(parents=True, exist_ok=True)
        sp.abs_path.write_text(new_content, encoding="utf-8")
        if self._index is not None:
            self._index.mark_changed(sp.abs_path)
        return {"applied": applied, "path": str(sp.rel_path)}

    def _apply_patch(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = ApplyPatchInput.model_validate(raw)
        self._refuse_protected_writes(args.path)
        sp = _resolve_in_root(self._root, args.path)
        self._refuse_protected_writes(args.path, sp)
        existing = sp.abs_path.read_text(encoding="utf-8") if sp.abs_path.exists() else None
        try:
            target_path, new_content = apply_patch_text(args.patch, existing)
        except PatchError as exc:
            raise ToolError(f"apply_patch failed for {args.path}: {exc}") from exc
        # The patch's `+++` header path must agree with the explicit `path` arg.
        # We trust `path` (the caller-supplied, schema-validated value) for the
        # write location; the header path is advisory but mismatches are a sign
        # the model is confused, so we surface them as an error.
        if target_path and target_path != args.path:
            raise ToolError(
                f"apply_patch: `path` argument {args.path!r} disagrees with patch "
                f"`+++` header {target_path!r}; emit them consistently"
            )
        if args.preview:
            return _preview_result(args.path, existing, new_content)
        sp.abs_path.parent.mkdir(parents=True, exist_ok=True)
        sp.abs_path.write_text(new_content, encoding="utf-8")
        if self._index is not None:
            self._index.mark_changed(sp.abs_path)
        return {"path": str(sp.rel_path), "bytes_written": len(new_content)}

    # ----- tree-sitter index handlers -----

    _INDEX_RESULT_CAP = 500

    def _ensure_index(self) -> SymbolIndex:
        if self._index is None:
            self._index = SymbolIndex(self._root)
        return self._index

    def hot_symbols(
        self,
        *,
        max_symbols: int = 20,
        min_files_referenced: int = 2,
    ) -> list[tuple[str, str, str, int, int]]:
        """Public passthrough to ``SymbolIndex.hot_symbols``.

        Called by ``ImplementWorkflow._load_context`` to populate
        ``RepoSummary.hot_symbols``. Shares the dispatcher's index so
        a workflow that has already paid for the scan doesn't re-scan
        on cold-plan.
        """
        idx = self._ensure_index()
        return idx.hot_symbols(
            max_symbols=max_symbols,
            min_files_referenced=min_files_referenced,
        )

    def file_outlines(self) -> dict[Path, list[Symbol]]:
        """Public passthrough to ``SymbolIndex.file_outlines``.

        Used by ``Workflow._load_repo_summary`` to build the
        per-file symbol outline injected into the system prompt.
        """
        idx = self._ensure_index()
        return idx.file_outlines()

    def _outline(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = OutlineInput.model_validate(raw)
        sp = _resolve_in_root(self._root, args.path)
        if not sp.abs_path.is_file():
            raise ToolError(f"Not a file: {args.path}")
        idx = self._ensure_index()
        syms = idx.outline(sp.abs_path)
        out = [{"name": s.name, "kind": s.kind, "line": s.line, "col": s.col} for s in syms]
        truncated = len(out) > self._INDEX_RESULT_CAP
        return {"symbols": out[: self._INDEX_RESULT_CAP], "truncated": truncated}

    def _find_definition(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = FindDefinitionInput.model_validate(raw)
        idx = self._ensure_index()
        defs = idx.find_definition(args.name)
        out: list[dict[str, Any]] = []
        for s in defs:
            try:
                rel = s.path.relative_to(self._root)
            except ValueError:
                continue
            out.append(
                {"name": s.name, "kind": s.kind, "path": str(rel), "line": s.line, "col": s.col}
            )
        truncated = len(out) > self._INDEX_RESULT_CAP
        return {"definitions": out[: self._INDEX_RESULT_CAP], "truncated": truncated}

    def _find_references(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = FindReferencesInput.model_validate(raw)
        idx = self._ensure_index()
        refs = idx.find_references(args.name)
        out: list[dict[str, Any]] = []
        for r in refs:
            try:
                rel = r.path.relative_to(self._root)
            except ValueError:
                continue
            out.append({"name": r.name, "path": str(rel), "line": r.line, "col": r.col})
        truncated = len(out) > self._INDEX_RESULT_CAP
        return {"references": out[: self._INDEX_RESULT_CAP], "truncated": truncated}

    # LSP-backed navigation. Lazy spawn so runs that never
    # call a *_lsp tool don't pay the server-startup tax.
    def _ensure_lsp(self) -> LspClient:
        if self._lsp is None:
            client = LspClient(self._root)
            try:
                client.start()
            except LspError as exc:
                raise ToolError(str(exc)) from exc
            self._lsp = client
        return self._lsp

    def _find_definition_lsp(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = FindDefinitionLspInput.model_validate(raw)
        sp = _resolve_in_root(self._root, args.path)
        if not sp.abs_path.is_file():
            raise ToolError(f"Not a file: {args.path}")
        client = self._ensure_lsp()
        try:
            locs = client.find_definition(sp.abs_path, args.symbol)
        except LspError as exc:
            raise ToolError(str(exc)) from exc
        out: list[dict[str, Any]] = []
        for loc in locs:
            try:
                rel = loc.path.resolve().relative_to(self._root)
            except ValueError:
                continue
            out.append({"path": str(rel), "line": loc.line, "col": loc.col})
        truncated = len(out) > self._INDEX_RESULT_CAP
        return {"definitions": out[: self._INDEX_RESULT_CAP], "truncated": truncated}

    def _find_references_lsp(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = FindReferencesLspInput.model_validate(raw)
        sp = _resolve_in_root(self._root, args.path)
        if not sp.abs_path.is_file():
            raise ToolError(f"Not a file: {args.path}")
        client = self._ensure_lsp()
        try:
            locs = client.find_references(sp.abs_path, args.symbol)
        except LspError as exc:
            raise ToolError(str(exc)) from exc
        out: list[dict[str, Any]] = []
        for loc in locs:
            try:
                rel = loc.path.resolve().relative_to(self._root)
            except ValueError:
                continue
            out.append({"path": str(rel), "line": loc.line, "col": loc.col})
        truncated = len(out) > self._INDEX_RESULT_CAP
        return {"references": out[: self._INDEX_RESULT_CAP], "truncated": truncated}

    def close(self) -> None:
        """Release subprocess resources (LSP server).

        Idempotent. Safe to call from CLI teardown alongside
        ``mcp_manager.close()``.
        """
        if self._lsp is not None:
            self._lsp.close()
            self._lsp = None

    def _run_verify(self, _raw: dict[str, Any]) -> dict[str, Any]:
        argv = tuple(self._config.workflow.verify_command)
        # per-call timeout from config. Defaults to the jail's
        # general 600s but bench configs crank it down so infinite-loop
        # edits fail fast instead of burning ~10 min of wall per attempt.
        timeout_s = self._config.workflow.verify_timeout_s
        self._emit("verify.start", cmd=list(argv), timeout_s=timeout_s)
        res = self._run_argv_in_jail(argv, label="verify_command", timeout_s=timeout_s)
        self._emit(
            "verify.end",
            cmd=list(argv),
            exit_code=res["returncode"],
            duration_s=res["duration_s"],
            timeout_s=timeout_s,
            stdout_tail=str(res["stdout"])[-2000:],
            stderr_tail=str(res["stderr"])[-2000:],
        )
        return res

    def _run_command(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = RunCommandInput.model_validate(raw)
        _refuse_mutating_git_command(args.argv)
        if self._config.sandbox.run_commands == "ask":
            ok = self._approver(f"Allow run_command {args.argv!r}?")
            if not ok:
                raise ToolError("run_command denied by user")
        return self._run_argv_in_jail(args.argv, label="run_command")

    def _finish_run(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Signal the workflow to terminate. The workflow checks
        for this tool name in the response's tool_uses and exits after
        dispatching it. Handler echoes the validated summary (and any
        structured ``result`` payload, used by state-machine agent states)."""
        args = FinishRunInput.model_validate(raw)
        return {"acknowledged": True, "summary": args.summary, "result": args.result}

    def _finish_planning(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Signal the planning pass is done. Plan-mode counterpart
        of finish_run; the workflow writes ``plan_markdown`` to disk and
        exits after dispatching it. Handler echoes the validated summary."""
        args = FinishPlanningInput.model_validate(raw)
        return {
            "acknowledged": True,
            "summary": args.summary,
            "plan_bytes": len(args.plan_markdown.encode("utf-8")),
        }

    # DAG-as-tool handlers. All raise ToolError if no graph_client
    # was wired so standalone test instantiation works unchanged.

    def _dag_add_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        if self._graph_client is None:
            raise ToolError("add_task: DAG curator not available in this run")
        args = DagAddTaskInput.model_validate(raw)
        parent_id = args.parent_id or self._run_root_node_id
        draft = TaskNodeDraft(
            title=args.title,
            rationale=args.rationale,
            acceptance=args.acceptance,
            relevant_paths=args.relevant_paths,
            created_by="worker",
        )
        intent = AddSubtaskIntent(parent_id=parent_id, draft=draft)
        node = self._graph_client.add_subtask(intent)  # type: ignore[attr-defined]
        return {
            "id": node.id,
            "parent_id": node.parent_id,
            "title": node.title,
            "status": node.status,
        }

    def _dag_update_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        if self._graph_client is None:
            raise ToolError("update_task: DAG curator not available in this run")
        args = DagUpdateTaskInput.model_validate(raw)
        intent = UpdateStatusIntent(
            id=args.id,
            new_status=args.status,  # type: ignore[arg-type]  # pydantic validates the literal
            note=args.note,
        )
        node = self._graph_client.update_status(intent)  # type: ignore[attr-defined]
        return {"id": node.id, "status": node.status, "title": node.title}

    def _dag_set_cursor(self, raw: dict[str, Any]) -> dict[str, Any]:
        if self._graph_client is None:
            raise ToolError("set_cursor: DAG curator not available in this run")
        args = DagSetCursorInput.model_validate(raw)
        self._graph_client.set_cursor(SetCursorIntent(id=args.id))  # type: ignore[attr-defined]
        return {"acknowledged": True, "cursor": args.id}

    def _dag_list_tasks(self, raw: dict[str, Any]) -> dict[str, Any]:
        if self._graph_client is None:
            raise ToolError("list_tasks: DAG curator not available in this run")
        args = DagListTasksInput.model_validate(raw)
        state = self._graph_client.get_state()  # type: ignore[attr-defined]
        nodes = state.get("nodes", {}) if isinstance(state, dict) else {}
        out: list[dict[str, Any]] = []
        for node_id, raw_node in nodes.items():
            if not isinstance(raw_node, dict):
                continue
            if args.status and raw_node.get("status") != args.status:
                continue
            out.append(
                {
                    "id": node_id,
                    "parent_id": raw_node.get("parent_id"),
                    "title": raw_node.get("title", ""),
                    "status": raw_node.get("status", "pending"),
                    "acceptance": raw_node.get("acceptance", ""),
                    "relevant_paths": list(raw_node.get("relevant_paths", ())),
                }
            )
        return {"tasks": out, "count": len(out)}

    def _run_metric(self, _raw: dict[str, Any]) -> dict[str, Any]:
        """Run ``cfg.workflow.metric.command`` in the jail.

        Exposed to the agent loop so the LLM can call it directly between
        edits to check its optimisation progress.
        Raises ToolError if no metric is configured.

        Return shape mirrors `_run_argv_in_jail` (returncode / stdout /
        stderr / duration_s) plus a parsed ``score`` field (audit
        finding: the schema description had always promised this, but the
        old handler only forwarded the raw command output. Now the
        ``pattern`` regex's first capture group is parsed to a float; if
        the pattern doesn't match or doesn't parse, ``score`` is null and
        the agent can fall back to grepping stdout itself).
        """
        metric_cfg = self._config.workflow.metric
        if metric_cfg is None:
            raise ToolError("run_metric_command: no [workflow.metric] configured")
        argv = tuple(metric_cfg.command)
        self._emit("metric.start", cmd=list(argv))
        res = self._run_argv_in_jail(argv, label="metric_command")
        score = _parse_metric_score(res, pattern=metric_cfg.pattern)
        res["score"] = score
        self._emit(
            "metric.end",
            cmd=list(argv),
            exit_code=res["returncode"],
            duration_s=res["duration_s"],
            stdout_tail=str(res["stdout"])[-2000:],
            stderr_tail=str(res["stderr"])[-2000:],
            score=score,
        )
        return res

    def _run_argv_in_jail(
        self,
        argv: tuple[str, ...],
        *,
        label: str,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        # run_command reaches the network only under tool_network = "allow"
        # (which the config validator pins to agent_network = "open", so this
        # process is in the host netns and the child inherits it).
        allow_network = self._config.sandbox.tool_network == "allow"
        # Resolve symlinks so the launcher's strip_prefix(cwd) check sees
        # canonical paths; the Rust side canonicalizes too as a backstop.
        protect_paths: list[Path] = []
        if self._config.sandbox.protect_git:
            protect_paths.append((self._root / ".git").resolve())
        if self._config.sandbox.protect_agent6:
            protect_paths.append((self._root / self._agent6_dir_name).resolve())
        protect_paths.extend(self._extra_protect_paths)
        # caller-provided timeout overrides the JailPolicy default
        # (600s). Used by verify_command + metric_command for fast failure
        # detection on pathological edits.
        policy_kwargs: dict[str, Any] = {}
        if timeout_s is not None:
            policy_kwargs["timeout_s"] = timeout_s
        policy = JailPolicy(
            cwd=self._root,
            argv=argv,
            profile=self._sandbox_profile,
            env=tuple(sorted(_passthrough_env().items())),
            allow_network=allow_network,
            extra_protect_paths=tuple(protect_paths),
            **policy_kwargs,
        )
        try:
            res: CommandResult = run_in_jail(policy)
        except JailUnavailableError as exc:
            raise ToolError(f"{label}: jail unavailable: {exc}") from exc
        return {
            "returncode": res.returncode,
            "stdout": res.stdout[-20_000:],
            "stderr": res.stderr[-20_000:],
            "duration_s": res.duration_s,
        }


_PASSTHROUGH_ENV_KEYS = ("LANG", "LC_ALL", "TERM", "CI")


def _passthrough_env() -> dict[str, str]:
    return {k: os.environ[k] for k in _PASSTHROUGH_ENV_KEYS if k in os.environ}


def _parse_metric_score(res: dict[str, Any], *, pattern: str) -> float | None:
    """Apply the metric ``pattern`` regex to combined stdout+stderr.

    Shared metric parser; centralised so the workflow and tool handler
    scores from the same command output. Returns ``None`` on regex compile
    failure, no-match, or non-numeric capture group - the caller treats
    that as "no score this turn" and falls back to raw stdout inspection.
    """
    combined = f"{res.get('stdout', '')}\n{res.get('stderr', '')}"
    try:
        m = re.search(pattern, combined)
    except re.error:
        return None
    if m is None:
        return None
    try:
        return float(m.group(1))
    except (ValueError, IndexError):
        return None


def _truncate_args(raw: dict[str, Any], *, max_value_chars: int = 200) -> dict[str, Any]:
    """Cheap argument preview for telemetry; truncates strings longer than
    *max_value_chars* and lists longer than 10 items."""
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, str) and len(v) > max_value_chars:
            out[k] = v[:max_value_chars] + f"… ({len(v)} chars)"
        elif isinstance(v, list | tuple) and len(v) > 10:
            out[k] = [*list(v[:10]), f"… ({len(v)} items)"]
        else:
            out[k] = v
    return out


def _summarize_result(name: str, result: dict[str, Any]) -> str:  # noqa: PLR0911
    """One-line human-readable summary for the TUI / log tail."""
    if "size" in result:
        return f"{result['size']} bytes"
    if "entries" in result and isinstance(result["entries"], list):
        return f"{len(result['entries'])} entries"
    if "hits" in result and isinstance(result["hits"], list):
        more = " (truncated)" if result.get("truncated") else ""
        return f"{len(result['hits'])} matches{more}"
    if "symbols" in result and isinstance(result["symbols"], list):
        more = " (truncated)" if result.get("truncated") else ""
        return f"{len(result['symbols'])} symbols{more}"
    if "definitions" in result and isinstance(result["definitions"], list):
        more = " (truncated)" if result.get("truncated") else ""
        return f"{len(result['definitions'])} definitions{more}"
    if "references" in result and isinstance(result["references"], list):
        more = " (truncated)" if result.get("truncated") else ""
        return f"{len(result['references'])} references{more}"
    if "applied" in result:
        return f"applied={result['applied']} path={result.get('path')}"
    if "bytes_written" in result:
        return f"patched path={result.get('path')} bytes={result['bytes_written']}"
    if "returncode" in result:
        return f"exit={result['returncode']} in {result.get('duration_s', 0):.1f}s"
    return name
