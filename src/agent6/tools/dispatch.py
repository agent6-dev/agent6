# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool dispatch: validates incoming LLM tool calls and executes them.

All filesystem reads/writes are clamped to *root* (the repo cwd). All command
execution goes through agent6.sandbox.jail.run_in_jail. Capability gating
(`run_commands = "no" | "ask" | "yes"`) is enforced here.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from agent6.config import Config
from agent6.events import EventSink
from agent6.graph.client import GraphClient
from agent6.graph.models import (
    AddSubtaskIntent,
    SetCursorIntent,
    TaskNodeDraft,
    UpdateStatusIntent,
)
from agent6.sandbox.jail import JailUnavailableError, run_in_jail
from agent6.tools._agent6_docs import (
    list_agent6_docs as _list_agent6_docs,
)
from agent6.tools._agent6_docs import (
    read_agent6_doc as _read_agent6_doc,
)
from agent6.tools._edit_diag import (
    edit_mismatch_error as _edit_mismatch_error,
)
from agent6.tools._edit_diag import (
    preview_result as _preview_result,
)
from agent6.tools._result_format import (
    parse_metric_score as _parse_metric_score,
)
from agent6.tools._result_format import (
    passthrough_env as _passthrough_env,
)
from agent6.tools._result_format import (
    summarize_result as _summarize_result,
)
from agent6.tools._result_format import (
    truncate_args as _truncate_args,
)
from agent6.tools.index import Symbol, SymbolIndex
from agent6.tools.lsp import LspClient, LspError
from agent6.tools.mcp_client import MCP_TOOL_PREFIX, MCPError, MCPManager
from agent6.tools.patch_apply import (
    PatchError,
    apply_patch_text,
    apply_v4a_text,
    is_v4a_patch,
    patch_target_path,
)
from agent6.tools.schema import (
    ALL_TOOLS,
    Agent6DocsInput,
    ApplyEditInput,
    ApplyPatchInput,
    AskUserInput,
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


# agent6's own docs, for the `agent6_docs` ask tool. Bundled into the wheel at
# agent6/_docs/ (hatch_build copies them); in a source checkout they're read
# straight from the repo root.

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


def _strip_env_wrapper(argv: tuple[str, ...]) -> tuple[str, ...]:
    """Peel a leading ``env [-i] [-u NAME] [NAME=VALUE...]`` wrapper.

    Best-effort: ``env git clean -fdx`` is the common way to slip a mutating git
    command past the argv[0]=="git" check; strip the wrapper so the refusal
    still applies. Does not (and cannot) catch every wrapper (sh -c, sudo, ...);
    the real protection is the jail RO-binding .git -- this just closes the
    obvious hole.
    """
    if not argv or Path(argv[0]).name != "env":
        return argv
    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        if arg in ("-i", "--ignore-environment"):
            idx += 1
        elif arg in ("-u", "--unset"):
            idx += 2  # takes a NAME argument
        elif "=" in arg and not arg.startswith("-"):
            idx += 1  # NAME=VALUE assignment
        else:
            break
    return argv[idx:]


def _git_subcommand(argv: tuple[str, ...]) -> str | None:
    argv = _strip_env_wrapper(argv)
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


class _Questioner(Protocol):
    def __call__(self, question: str, options: tuple[str, ...], /) -> str: ...


def _default_questioner(  # pragma: no cover — interactive
    question: str, options: tuple[str, ...]
) -> str:
    """Fallback for `ask_user` when no TUI/CLI bridge is wired: a numbered stdin
    prompt. A non-TTY/headless stdin returns "" immediately so a run never hangs
    (mirrors run.py's _default_stdin_questioner)."""
    if not sys.stdin.isatty():
        return ""
    lines = [question, *(f"  {i}) {opt}" for i, opt in enumerate(options, start=1))]
    try:
        ans = input("\n".join(lines) + "\n> ").strip()
    except EOFError:
        return ""
    if ans.isdigit() and 1 <= int(ans) <= len(options):
        return options[int(ans) - 1]
    return ans


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

    ``.git`` (when ``protect_git``): the edit tools write **in-process, outside
    the jail**, so without this an LLM could create or rewrite ``.git/hooks/*``
    or ``.git/config`` (e.g. ``core.fsmonitor``) and get code executed outside
    the sandbox on the next ``git`` invocation, or corrupt git history --
    defeating ``protect_git`` entirely (the strict jail's RO bind of ``.git``
    never covers these in-process writes). Reads stay allowed. (Run state lives
    out of the workspace, so it is unreachable by edits and needs no guard.)

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


class ToolDispatcher:
    """Runtime tool dispatcher. Constructed once per workflow run."""

    def __init__(
        self,
        *,
        root: Path,
        config: Config,
        sandbox_profile: SandboxProfile = "strict",
        approver: _Approver | None = None,
        questioner: _Questioner | None = None,
        events: EventSink | None = None,
        graph_client: GraphClient | None = None,
        run_root_node_id: str | None = None,
        mcp_manager: MCPManager | None = None,
        extra_protect_paths: tuple[Path, ...] = (),
        mode: Literal["run", "plan", "ask", "machine"] = "run",
    ) -> None:
        self._root = root.resolve()
        self._config = config
        self._sandbox_profile: SandboxProfile = sandbox_profile
        # In plan mode the LLM's tool list already omits apply_edit/apply_patch;
        # this is the defense-in-depth backstop so the dispatcher itself refuses
        # a source mutation even if something dispatched one directly.
        self._mode: Literal["run", "plan", "ask", "machine"] = mode
        # Extra read-only paths layered into every run_command jail on top of
        # the strict-profile protect_git bind (e.g. a running machine's own
        # .asm.toml + scripts bundle, so an agent state can't rewrite them
        # mid-run).
        self._extra_protect_paths = extra_protect_paths
        self._approver: _Approver = approver or _default_approver
        self._questioner: _Questioner = questioner or _default_questioner
        self._events = events
        # Optional GraphClient + root-task id for the DAG-as-tool
        # surface. When wired, the dispatcher exposes add_task /
        # update_task / set_cursor / list_tasks.
        self._graph_client = graph_client
        self._run_root_node_id = run_root_node_id
        # Optional MCP (Model Context Protocol) manager. When
        # set, ``dispatch`` routes any tool name starting with the MCP
        # prefix to the manager. Discovered tool names are also added
        # to ``available_tool_names()`` so the workflow exposes them.
        self._mcp_manager = mcp_manager
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            Agent6DocsInput.TOOL_NAME: self._agent6_docs,
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
            AskUserInput.TOOL_NAME: self._ask_user,
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
        # No verify_command (and none inferred) -> a gateless run: hide
        # run_verify_command rather than offer a tool that would error.
        if not self._config.workflow.verify_command:
            names = [n for n in names if n != RunVerifyInput.TOOL_NAME]
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
        if self._mode in ("plan", "ask", "machine") and name in {
            ApplyEditInput.TOOL_NAME,
            ApplyPatchInput.TOOL_NAME,
        }:
            # Backstop the read-only guarantee at the dispatcher, not just by
            # omitting these from the LLM's tool list.
            raise ToolError(f"{name} is not available in {self._mode} mode (read-only)")
        if self._mode == "machine" and name in {
            RunCommandInput.TOOL_NAME,
            RunVerifyInput.TOOL_NAME,
        }:
            # machine-authoring + machine agent-state loops never run commands
            # (unlike `ask`, which allows read-only run_command investigation).
            raise ToolError(f"{name} is not available in {self._mode} mode (read-only)")
        if self._mode != "run" and name == AskUserInput.TOOL_NAME:
            # ask_user is a run-mode tool (LOOP_EXTRA_TOOLS only); backstop it so
            # a future tool-list regression can't pause a plan/ask/machine loop.
            raise ToolError(f"{name} is not available in {self._mode} mode")
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

    def _agent6_docs(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = Agent6DocsInput.model_validate(raw)
        available = _list_agent6_docs()
        if not args.name:
            return {"available": available}
        content = _read_agent6_doc(args.name)
        if content is None:
            raise ToolError(
                f"unknown agent6 doc {args.name!r}; available: {', '.join(available) or '(none)'}"
            )
        cap = 60_000
        return {
            "name": args.name,
            "content": content[:cap],
            "truncated": len(content) > cap,
        }

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
        # Skip hidden files/dirs (.git, ...) only when they are BELOW
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
        """Refuse an in-process edit into ``.git`` (it bypasses the jail entirely)."""
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
                    # Prefer handing the model the exact closest on-disk text so
                    # it retries directly. Re-reading the whole file is the
                    # dominant small-model time sink on a botched edit; the
                    # shape-only fallback (no copyable body) is used only when
                    # nothing on disk is similar enough to anchor on.
                    raise ToolError(
                        _edit_mismatch_error(args.path, i, new_content, edit.old_string)
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
        v4a = is_v4a_patch(args.patch)
        # The write location: the explicit `path` arg if given, else derived from
        # the patch headers (V4A always embeds it; GPT-family models omit `path`).
        # Either way it is resolved + protected-path-checked below, so deriving it
        # from the patch never widens where a write can land.
        try:
            derived_path = patch_target_path(args.patch)
        except PatchError as exc:
            raise ToolError(f"apply_patch failed for {args.path or '<unknown>'}: {exc}") from exc
        target = args.path or derived_path
        # Security checks on the write location come first (absolute path, repo
        # escape, protected dirs), before the lower-priority model-confusion
        # check that an explicit `path` matches the patch header.
        self._refuse_protected_writes(target)
        sp = _resolve_in_root(self._root, target)
        self._refuse_protected_writes(target, sp)
        if args.path and args.path != derived_path:
            raise ToolError(
                f"apply_patch: `path` argument {args.path!r} disagrees with the patch "
                f"header path {derived_path!r}; emit them consistently or omit `path`"
            )
        existing = sp.abs_path.read_text(encoding="utf-8") if sp.abs_path.exists() else None
        try:
            if v4a:
                _, new_content = apply_v4a_text(args.patch, existing)
            else:
                _, new_content = apply_patch_text(args.patch, existing)
        except PatchError as exc:
            raise ToolError(f"apply_patch failed for {target}: {exc}") from exc
        if args.preview:
            return _preview_result(target, existing, new_content)
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

    def _ask_user(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Pose a question to the operator and return their answer. The injected
        questioner does the actual prompting (TUI modal / stdin / headless skip)
        and owns the question.prompt/answer events; this handler just validates."""
        args = AskUserInput.model_validate(raw)
        answer = self._questioner(args.question, args.options)
        return {"answer": answer}

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
        node = self._graph_client.add_subtask(intent)
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
        node = self._graph_client.update_status(intent)
        return {"id": node.id, "status": node.status, "title": node.title}

    def _dag_set_cursor(self, raw: dict[str, Any]) -> dict[str, Any]:
        if self._graph_client is None:
            raise ToolError("set_cursor: DAG curator not available in this run")
        args = DagSetCursorInput.model_validate(raw)
        self._graph_client.set_cursor(SetCursorIntent(id=args.id))
        return {"acknowledged": True, "cursor": args.id}

    def _dag_list_tasks(self, raw: dict[str, Any]) -> dict[str, Any]:
        if self._graph_client is None:
            raise ToolError("list_tasks: DAG curator not available in this run")
        args = DagListTasksInput.model_validate(raw)
        state = self._graph_client.get_state()
        nodes = state.get("nodes", {})
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
        # protect_paths are read-only bind-remounts, which only the strict
        # profile (mount namespace) can apply. On hardened the cwd is blanket
        # read-write -- there is no way to carve .git read-only without also
        # denying new top-level entries (breaking toolchains), so .git is
        # writable there. It is recoverable, gated by run_commands, and run
        # state lives out of the workspace, so nothing sensitive is exposed.
        if self._sandbox_profile == "strict" and self._config.sandbox.protect_git:
            protect_paths.append((self._root / ".git").resolve())
        protect_paths.extend(self._extra_protect_paths)
        # caller-provided timeout overrides the JailPolicy default
        # (600s). Used by verify_command + metric_command for fast failure
        # detection on pathological edits.
        policy_kwargs: dict[str, Any] = {}
        if timeout_s is not None:
            policy_kwargs["timeout_s"] = timeout_s
        env = _passthrough_env()
        # Toolchains need a writable cache root (go test -> $HOME/.cache/go-build,
        # cargo -> $CARGO_HOME or $HOME/.cargo, pip/uv likewise). The jail's /tmp
        # is writable on both profiles (fresh tmpfs on strict, Landlock rw grant
        # on hardened), so point HOME there. Without it `go test` fails outright
        # and models burn whole budgets probing the sandbox for a writable spot.
        env.setdefault("HOME", "/tmp/agent6-home")  # noqa: S108 - resolved inside the jail
        policy = JailPolicy(
            cwd=self._root,
            argv=argv,
            profile=self._sandbox_profile,
            env=tuple(sorted(env.items())),
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
