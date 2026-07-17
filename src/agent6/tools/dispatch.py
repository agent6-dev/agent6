# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool dispatch: validates incoming LLM tool calls and executes them.

All filesystem reads/writes are clamped to *root* (the repo cwd). All command
execution goes through agent6.sandbox.jail.run_in_jail. Capability gating
(`run_commands = "no" | "ask" | "yes"`) is enforced here.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from agent6.config import Config
from agent6.events import EventSink
from agent6.graph.client import GraphClient
from agent6.paths import data_dir
from agent6.sandbox.jail import JailUnavailableError, run_in_jail
from agent6.skills import (
    ResolvedSkills,
    discover_skills,
    resolve_states,
    skill_search_dirs,
)
from agent6.tools._dag_tools import add_dependency as _add_dependency
from agent6.tools._dag_tools import add_task as _add_task
from agent6.tools._dag_tools import list_tasks as _list_tasks
from agent6.tools._dag_tools import set_cursor as _set_cursor
from agent6.tools._dag_tools import update_task as _update_task
from agent6.tools._fs_tools import agent6_docs as _fs_agent6_docs
from agent6.tools._fs_tools import apply_edit as _fs_apply_edit
from agent6.tools._fs_tools import apply_patch as _fs_apply_patch
from agent6.tools._fs_tools import grep as _fs_grep
from agent6.tools._fs_tools import list_dir as _fs_list_dir
from agent6.tools._fs_tools import read_file as _fs_read_file
from agent6.tools._git_guard import refuse_mutating_git_command
from agent6.tools._memory_tools import add_memory as _add_memory_impl
from agent6.tools._memory_tools import invalidate_memory as _invalidate_memory_impl
from agent6.tools._memory_tools import use_skill as _use_skill_impl
from agent6.tools._nav_tools import find_definition as _nav_find_definition
from agent6.tools._nav_tools import find_definition_lsp as _nav_find_definition_lsp
from agent6.tools._nav_tools import find_references as _nav_find_references
from agent6.tools._nav_tools import find_references_lsp as _nav_find_references_lsp
from agent6.tools._nav_tools import outline as _nav_outline
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
from agent6.tools.errors import OperatorCommandUnexecutable, ToolError
from agent6.tools.index import Symbol, SymbolIndex
from agent6.tools.lsp import LspClient, LspError, lsp_tools_useful
from agent6.tools.mcp_client import MCP_TOOL_PREFIX, MCPError, MCPManager
from agent6.tools.schema import (
    ALL_TOOLS,
    AddMemoryInput,
    Agent6DocsInput,
    ApplyEditInput,
    ApplyPatchInput,
    AskUserInput,
    DagAddDependencyInput,
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
    InvalidateMemoryInput,
    ListDirInput,
    OutlineInput,
    ReadFileInput,
    RunCommandInput,
    RunMetricInput,
    RunVerifyInput,
    UserQuestion,
    UseSkillInput,
)
from agent6.types import CommandResult, JailPolicy, SandboxProfile


def _coerce_stringified_args(
    raw_input: dict[str, Any], exc: ValidationError
) -> dict[str, Any] | None:
    """Recover a tool call whose structured argument arrived as a JSON string.

    Weak models occasionally serialize an array/object argument to a string
    (observed live: haiku 4.5 sending apply_edit ``edits`` as
    ``'[{...}]\\n</invoke>'``), wasting a full round-trip on a validation
    error the model must repair. For each top-level field named in the
    validation error whose provided value is a str, parse the string's head
    as JSON (``raw_decode`` tolerates trailing junk like a leaked closing
    tag) and substitute the parsed value when it is a container. Fields the
    schema really declares as strings are unaffected: a wrong substitution
    fails re-validation and the caller re-raises the original error. Returns
    the coerced copy of ``raw_input``, or None when nothing was coercible.
    """
    decoder = json.JSONDecoder()
    coerced: dict[str, Any] | None = None
    for err in exc.errors():
        loc = err.get("loc") or ()
        key = loc[0] if loc else None
        if not isinstance(key, str):
            continue
        val = raw_input.get(key)
        if not isinstance(val, str):
            continue
        try:
            parsed, _ = decoder.raw_decode(val.strip())
        except ValueError:
            continue
        if not isinstance(parsed, dict | list):
            continue
        if coerced is None:
            coerced = dict(raw_input)
        coerced[key] = parsed
    return coerced


# --- operator tool reachability ----------------------------------------------
# The jail's baseline PATH is /usr/bin:/bin and it bind-mounts only the system
# roots below. Operator tools (uv, node, ...) installed elsewhere are otherwise
# unreachable, so a verify/run command dies 127. We add the standard bin dirs that
# exist to PATH, and for those outside the system roots -- or whose symlinks
# resolve out to one (a pipx `uv` at /usr/local/bin -> /opt/pipx/...) -- pass the
# real dirs as tool_paths for a real-location RO+exec mount. Read+exec only; the
# jail still confines writes and network, so containment is unchanged.
_JAIL_BASE_PATH_DIRS = ("/usr/bin", "/bin")
_SYSTEM_ROOTS = (
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/etc"),
    Path("/dev"),
)


def _under_system_root(p: Path) -> bool:
    return any(p.is_relative_to(r) for r in _SYSTEM_ROOTS)


def _operator_tool_paths() -> tuple[str, tuple[Path, ...]]:
    """Return (PATH string, real-location mount dirs) so operator-installed tools
    resolve in the jail. Recomputed per call so a tool the model just installed is
    picked up (dirs under a mounted system root only join PATH; dirs outside it, and
    the real dirs symlinks resolve out to, also need the RO+exec mount)."""
    home = Path.home()
    candidates = (
        Path("/usr/local/bin"),
        Path("/usr/local/sbin"),
        home / ".local/bin",
        home / ".cargo/bin",
        Path("/opt/homebrew/bin"),
        Path("/snap/bin"),
    )
    path_dirs: list[str] = list(_JAIL_BASE_PATH_DIRS)
    mounts: set[Path] = set()
    for d in candidates:
        if not d.is_dir():
            continue
        path_dirs.append(str(d))
        if not _under_system_root(d):
            mounts.add(d)  # real binaries in a non-system dir need the dir itself
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_symlink():
                continue  # real files are covered by the dir / the /usr mount
            try:
                real = entry.resolve()
            except OSError:
                continue
            if real.is_file() and not _under_system_root(real):
                mounts.add(real.parent)  # e.g. /opt/pipx/venvs/uv/bin
    # Interpreter toolchains a repo venv's python may symlink to: uv-managed
    # CPython lives under XDG data, not any bin dir. Without this mount the
    # jail sees such a venv "linked to a non-existent interpreter" and an
    # in-jail `uv run` DELETES and recreates the operator's .venv (observed
    # live: the verify tail read "Removed virtual environment at: .venv").
    # Mount-only, never a PATH entry.
    data_home = Path(os.environ.get("XDG_DATA_HOME") or home / ".local/share")
    uv_pythons = data_home / "uv" / "python"
    if uv_pythons.is_dir():
        mounts.add(uv_pythons)
    return ":".join(path_dirs), tuple(sorted(mounts))


# Execution tools whose stdout/stderr IS the diagnostic signal. Their tool.result
# event carries a capped output tail (like verify.end) so logs.jsonl shows
# the command's output for quick observability -- not just a one-line summary --
# without opening the transcripts (where the full, uncapped output always lives).
_EXEC_OUTPUT_TOOLS = frozenset({RunCommandInput.TOOL_NAME, RunMetricInput.TOOL_NAME})
_TOOL_OUTPUT_TAIL = 2000  # chars, matching verify.end's stdout_tail/stderr_tail


def _output_tails(name: str, result: Any) -> dict[str, str]:
    """Capped stdout/stderr tails for an execution tool's result, else {}."""
    if name not in _EXEC_OUTPUT_TOOLS or not isinstance(result, dict):
        return {}
    return {
        "stdout_tail": str(result.get("stdout", ""))[-_TOOL_OUTPUT_TAIL:],
        "stderr_tail": str(result.get("stderr", ""))[-_TOOL_OUTPUT_TAIL:],
    }


class _Approver(Protocol):
    def __call__(self, prompt: str, /) -> bool: ...


def _default_approver(prompt: str) -> bool:  # pragma: no cover — interactive
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


class _Questioner(Protocol):
    def __call__(self, questions: tuple[UserQuestion, ...], /) -> tuple[str, ...]: ...


def _default_questioner(  # pragma: no cover — interactive
    questions: tuple[UserQuestion, ...],
) -> tuple[str, ...]:
    """Fallback for `ask_user` when no TUI/CLI bridge is wired: numbered stdin
    prompts, one per question. A non-TTY/headless stdin returns "" for each so a run
    never hangs (mirrors run.py's _default_stdin_questioner)."""
    if not sys.stdin.isatty():
        return tuple("" for _ in questions)
    answers: list[str] = []
    for q in questions:
        lines = [q.question, *(f"  {i}) {opt}" for i, opt in enumerate(q.options, start=1))]
        try:
            ans = input("\n".join(lines) + "\n> ").strip()
        except EOFError:
            ans = ""
        if ans.isdigit() and 1 <= int(ans) <= len(q.options):
            ans = q.options[int(ans) - 1]
        answers.append(ans)
    return tuple(answers)


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
        state_dir: Path | None = None,
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
        # Per-repo state dir holding the cross-run memory store
        # (<state_dir>/memories/). None (tests, review/one-off dispatchers)
        # leaves add_memory / invalidate_memory unwired: they raise ToolError,
        # like the DAG tools without a curator.
        self._state_dir = state_dir
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
            DagAddDependencyInput.TOOL_NAME: self._dag_add_dependency,
            # Cross-run memory. Handlers raise ToolError if no
            # state_dir was wired.
            AddMemoryInput.TOOL_NAME: self._add_memory,
            InvalidateMemoryInput.TOOL_NAME: self._invalidate_memory,
            # Operator-installed skills; resolved lazily from config + the
            # data dir on first use (see _resolved_skills).
            UseSkillInput.TOOL_NAME: self._use_skill,
        }
        self._available = {cls.TOOL_NAME for cls in ALL_TOOLS}
        self._index: SymbolIndex | None = None
        # Guards the lazy build of self._index so concurrent explore-review
        # seats (sharing one dispatcher across ThreadPoolExecutor threads)
        # can't double-build it.
        self._index_lock = threading.Lock()
        # Lazy LSP client for find_*_lsp tools. Spawned on
        # first use, killed by close(). Outside the jail, same trust
        # boundary as the tree-sitter index.
        self._lsp: LspClient | None = None
        # The ty LSP server is Python-only; hide the two find_*_lsp tools when
        # they can't help (no ty/uvx, or a non-Python repo) so they don't waste
        # schema tokens or confuse the model with dead near-duplicate tools.
        self._lsp_tools_useful = lsp_tools_useful(self._root)
        # Operator-installed skills, resolved once on first use (a disk scan
        # of the configured skill dirs). None = not yet resolved.
        self._skills_cache: ResolvedSkills | None = None

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
        # Python-only LSP tools are dead weight on a non-Python repo or with no
        # ty/uvx installed: hide them rather than offer tools that only error.
        if not self._lsp_tools_useful:
            lsp_names = {FindDefinitionLspInput.TOOL_NAME, FindReferencesLspInput.TOOL_NAME}
            names = [n for n in names if n not in lsp_names]
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
        # Emit `tool.call` UP FRONT, before any guard, so EVERY dispatched tool
        # -- including ones a guard rejects (unknown name, disabled, wrong mode)
        # -- produces a matching `tool.result(ok=...)` pair. Otherwise a reader
        # sees a `loop.tool.call` with no result and has to guess what happened.
        # The emit + the ok flag live here in the dispatcher (not gated on the
        # model), so a prompt injection cannot suppress the event or fake
        # success; rejection reasons come from these hardcoded guards, not from
        # model-supplied content.
        # The finish tools' `summary` is the human end-of-run statement (shown on
        # the done line + in `watch`); keep it whole. Generic args stay clipped.
        max_chars = 2000 if name in ("finish_run", "finish_planning") else 200
        preview = _truncate_args(raw_input, max_value_chars=max_chars)
        self._emit("tool.call", name=name, args=preview)
        try:
            result = self._dispatch_inner(name, raw_input)
        except ToolError as exc:
            self._emit("tool.result", name=name, ok=False, summary=str(exc))
            raise
        except OperatorCommandUnexecutable as exc:
            # Not a model-fixable tool error: an operator verify/metric command
            # that cannot execute in the jail. Record the failed result for the
            # audit trail, then propagate (NOT wrapped as ToolError) so the loop
            # aborts the run loudly instead of surfacing it as a normal failure.
            self._emit("tool.result", name=name, ok=False, summary=str(exc))
            raise
        except Exception as exc:
            self._emit("tool.result", name=name, ok=False, summary=str(exc))
            raise ToolError(f"{name} failed: {exc}") from exc
        self._emit(
            "tool.result",
            name=name,
            ok=True,
            summary=_summarize_result(name, result),
            **_output_tails(name, result),
        )
        return result

    def _dispatch_inner(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
        """Resolve + execute a tool. Raises ToolError on a rejected/failed call;
        the caller (`dispatch`) owns the tool.call/tool.result events."""
        # MCP routing happens BEFORE the built-in handler check so mcp__* names
        # don't collide with the built-in "Unknown tool" error path.
        if name.startswith(MCP_TOOL_PREFIX):
            if self._mcp_manager is None:
                raise ToolError(f"{name}: MCP is not configured")
            try:
                return self._mcp_manager.call(name, raw_input)
            except MCPError as exc:
                raise ToolError(str(exc)) from exc
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
        if self._mode != "run" and name in {
            AskUserInput.TOOL_NAME,
            AddMemoryInput.TOOL_NAME,
            InvalidateMemoryInput.TOOL_NAME,
            UseSkillInput.TOOL_NAME,
        }:
            # Run-mode tools (LOOP_EXTRA_TOOLS only); backstop them so a future
            # tool-list regression can't pause a plan/ask/machine loop
            # (ask_user) or let it write cross-run memories.
            raise ToolError(f"{name} is not available in {self._mode} mode")
        return self._run_handler(name, raw_input)

    def _run_handler(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
        """Execute the handler, retrying once with stringified-JSON args coerced."""
        # The provider couldn't parse the tool-call arguments as JSON and left the
        # `_raw_arguments` sentinel (after a lenient re-parse already failed). A
        # schema error about "_raw_arguments extra fields" would misdirect the
        # model; tell it plainly the JSON was malformed so it resends in one shot.
        if set(raw_input) == {"_raw_arguments"}:
            raw = raw_input.get("_raw_arguments")
            raw_len = len(raw) if isinstance(raw, str) else 0
            if raw_len > 20_000:
                # Not a formatting slip: the arguments ran away (observed:
                # kimi-k2.7 emitting a 117KB grep pattern of one alternation
                # repeated until the output-token ceiling cut the JSON string
                # mid-way). "Resend" feedback makes such a model regenerate
                # the same runaway; name the actual problem instead.
                raise ToolError(
                    f"{name}: the arguments were cut off mid-generation"
                    f" ({raw_len // 1000} KB, truncated before the JSON closed)."
                    " Do NOT resend the same call. Emit a much smaller call:"
                    " short literal values only (e.g. a grep pattern under 200"
                    " characters, one or two alternations), and split broad"
                    " searches into several small ones."
                )
            raise ToolError(
                f"{name}: the arguments were not valid JSON. Resend the call with a"
                " single valid JSON object of arguments."
            )
        try:
            return self._handlers[name](raw_input)
        except ValidationError as exc:
            coerced = _coerce_stringified_args(raw_input, exc)
            if coerced is None:
                raise
            try:
                return self._handlers[name](coerced)
            except ValidationError:
                # The coercion guessed wrong; the original shape error is the
                # honest one to surface.
                raise exc from None

    def _emit(self, event_type: str, /, **fields: Any) -> None:
        if self._events is not None:
            self._events.emit(event_type, **fields)

    # ----- handlers -----

    def _agent6_docs(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _fs_agent6_docs(raw)

    def _read_file(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _fs_read_file(self._root, raw)

    def _list_dir(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _fs_list_dir(self._root, raw)

    def _grep(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _fs_grep(self._root, raw)

    def _apply_edit(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _fs_apply_edit(self._root, self._config, self._extra_protect_paths, self._index, raw)

    def _apply_patch(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _fs_apply_patch(
            self._root, self._config, self._extra_protect_paths, self._index, raw
        )

    # ----- tree-sitter index handlers -----

    def _ensure_index(self) -> SymbolIndex:
        if self._index is None:
            with self._index_lock:
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
        return _nav_outline(self._root, self._ensure_index, raw)

    def _find_definition(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _nav_find_definition(self._root, self._ensure_index, raw)

    def _find_references(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _nav_find_references(self._root, self._ensure_index, raw)

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
        return _nav_find_definition_lsp(self._root, self._ensure_lsp, raw)

    def _find_references_lsp(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _nav_find_references_lsp(self._root, self._ensure_lsp, raw)

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
        if res.get("exec_failed"):
            raise OperatorCommandUnexecutable(
                f"verify_command {list(argv)} could not be executed in the sandbox: "
                f"{res['stderr']}. The jail PATH is /usr/bin:/bin plus the standard bin "
                "dirs that exist (/usr/local/bin, ~/.local/bin, ~/.cargo/bin, "
                "/opt/homebrew/bin, /snap/bin), each mounted read-only; the command is on "
                "none of them. Install the tool into one of those on the host, or grant "
                "its real path via sandbox.extra_read_paths."
            )
        return res

    def _run_command(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = RunCommandInput.model_validate(raw)
        refuse_mutating_git_command(args.argv)
        if self._config.sandbox.run_commands == "ask":
            # A shell-style command line, not a Python tuple repr: the operator
            # is approving a command, so show it the way they would type it.
            ok = self._approver(f"Allow run_command: {shlex.join(args.argv)}")
            if not ok:
                raise ToolError("run_command denied by user")
        return self._run_argv_in_jail(args.argv, label="run_command")

    def _ask_user(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Pose one or more questions to the operator and return the answers. The
        injected questioner does the actual prompting (TUI modal / stdin / headless
        skip) and owns the question.prompt/answer events; this handler just
        validates. Answers align to `questions` by index."""
        args = AskUserInput.model_validate(raw)
        answers = self._questioner(args.questions)
        return {"answers": list(answers)}

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
        return _add_task(self._graph_client, self._run_root_node_id, raw)

    def _dag_update_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _update_task(self._graph_client, raw)

    def _dag_set_cursor(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _set_cursor(self._graph_client, raw)

    def _dag_add_dependency(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _add_dependency(self._graph_client, raw)

    def _dag_list_tasks(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _list_tasks(self._graph_client, raw)

    # Cross-run memory handlers. Writes go through trusted code
    # (agent6.memory) to fixed markdown files under <state_dir>/memories/,
    # outside the workspace and the jail; the LLM controls only the scope
    # (schema-validated literal) and the note text, which is inert data.

    def _add_memory(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _add_memory_impl(self._state_dir, raw)

    def _invalidate_memory(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _invalidate_memory_impl(self._state_dir, raw)

    def resolved_skills(self) -> ResolvedSkills:
        """Discover + state-resolve operator skills, once per dispatcher.

        Same source of truth as the loop's system-prompt index:
        ``[skills].extra_dirs`` first, then the installed dir under the user
        data dir. An off switch resolves to nothing.
        """
        if self._skills_cache is None:
            if not self._config.skills.enabled:
                self._skills_cache = ResolvedSkills(enabled=(), always=(), warnings=())
            else:
                dirs = skill_search_dirs(self._config.skills.extra_dirs, data_dir() / "skills")
                found, warns = discover_skills(dirs)
                resolved = resolve_states(found, self._config.skills.state)
                self._skills_cache = ResolvedSkills(
                    enabled=resolved.enabled,
                    always=resolved.always,
                    warnings=(*warns, *resolved.warnings),
                )
        return self._skills_cache

    def skills_available(self) -> bool:
        """True when at least one enabled/always skill exists; gates whether
        ``use_skill`` is exposed in the loop's tool list."""
        resolved = self.resolved_skills()
        return bool(resolved.enabled or resolved.always)

    def _use_skill(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _use_skill_impl(self.resolved_skills, raw)

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
        if res.get("exec_failed"):
            raise OperatorCommandUnexecutable(
                f"metric_command {list(argv)} could not be executed in the sandbox: "
                f"{res['stderr']}. See run_verify_command's note: PATH is /usr/bin:/bin "
                "plus the standard bin dirs; install the tool into one of those on the "
                "host, or grant its real path via sandbox.extra_read_paths."
            )
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
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        # `uv run` inside the jail must use the venv the operator already synced: the
        # jail is offline (network is brokered to providers only) and HOME is a fresh
        # tmpfs, so a sync/build would re-resolve against an empty cache and fail. Run
        # against the existing env instead (a verify's `uv run ruff` then works).
        env.setdefault("UV_NO_SYNC", "1")
        # Make operator-installed tools (uv, ...) reachable: a controlled PATH that
        # extends /usr/bin:/bin with the standard bin dirs, plus their real dirs as
        # RO+exec mounts. Without this a `uv run` verify dies 127.
        tool_path, tool_mounts = _operator_tool_paths()
        env["PATH"] = tool_path
        policy = JailPolicy(
            cwd=self._root,
            argv=argv,
            profile=self._sandbox_profile,
            env=tuple(sorted(env.items())),
            allow_network=allow_network,
            extra_protect_paths=tuple(protect_paths),
            extra_ro_paths=tuple(Path(p) for p in self._config.sandbox.extra_read_paths),
            tool_paths=tool_mounts,
            memory_limit_mb=self._config.sandbox.memory_limit_mb,
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
            "exec_failed": res.exec_failed,
        }
