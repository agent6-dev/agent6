# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool dispatch: validates incoming LLM tool calls and executes them.

All filesystem reads/writes are clamped to *root* (the repo cwd). All command
execution goes through agent6.sandbox.jail.run_in_jail. Capability gating
(`run_commands = "no" | "ask" | "yes"`) is enforced here.
"""

from __future__ import annotations

import itertools
import json
import os
import shlex
import shutil
import sys
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from agent6.config import Config
from agent6.events import EventSink
from agent6.graph.curator import GraphCurator
from agent6.paths import data_dir
from agent6.sandbox.jail import (
    JailSession,
    JailUnavailableError,
    jail_search_path,
    operator_tool_paths,
    run_in_jail,
)
from agent6.sessions.ipc import effective_run_commands
from agent6.sessions.layout import session_layout
from agent6.skills import (
    ResolvedSkills,
    discover_skills,
    resolve_states,
    skill_search_dirs,
)
from agent6.tools._control_tools import ask_user, finish_planning, finish_session
from agent6.tools._dag_tools import add_dependency, add_task, list_tasks, set_cursor, update_task
from agent6.tools._fs_tools import agent6_docs, apply_edit, apply_patch, grep, list_dir, read_file
from agent6.tools._memory_tools import (
    add_memory,
    invalidate_memory,
    read_notes,
    use_skill,
    write_notes,
)
from agent6.tools._nav_tools import (
    find_definition,
    find_definition_lsp,
    find_references,
    find_references_lsp,
    outline,
)
from agent6.tools._result_format import (
    parse_metric_score,
    passthrough_env,
    truncate_args,
)
from agent6.tools.background import BackgroundError, BackgroundShells
from agent6.tools.errors import OperatorCommandUnexecutable, ToolDenied, ToolError
from agent6.tools.fetch import FetchRefused, check_url, fetch, host_allowed
from agent6.tools.index import Symbol, SymbolIndex
from agent6.tools.lsp import LspClient, LspError, lsp_tools_useful
from agent6.tools.mcp_client import MCP_TOOL_PREFIX, MCPError, MCPManager
from agent6.tools.results import (
    BackgroundResult,
    ExecResult,
    FetchResult,
    MetricResult,
    RawResult,
    SessionsResult,
    ToolResult,
)
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
    FetchInput,
    FindDefinitionInput,
    FindDefinitionLspInput,
    FindReferencesInput,
    FindReferencesLspInput,
    FinishPlanningInput,
    FinishSessionInput,
    GrepInput,
    InvalidateMemoryInput,
    ListDirInput,
    OutlineInput,
    ReadBackgroundInput,
    ReadFileInput,
    ReadNotesInput,
    ReadSessionInput,
    RunBackgroundInput,
    RunCommandInput,
    RunMetricInput,
    RunVerifyInput,
    StopBackgroundInput,
    UserQuestion,
    UseSkillInput,
    WriteNotesInput,
    mode_tools,
)
from agent6.tools.sessions import conversation, roster
from agent6.types import CommandResult, IsolationLevel, JailPolicy, session_kind


def _coerce_stringified_args(
    raw_input: dict[str, Any], exc: ValidationError
) -> dict[str, Any] | None:
    """Recover a tool call whose structured argument arrived as a JSON string.

    Weak models occasionally serialize an array/object argument to a string
    (e.g. apply_edit ``edits`` arriving as ``'[{...}]'``), wasting a
    round-trip on a validation error the model must repair. For each top-level field named in the
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


# Execution tools whose stdout/stderr IS the diagnostic signal. Their tool.result
# event carries a capped output tail (like verify.end) so logs.jsonl shows
# the command's output for quick observability -- not just a one-line summary --
# without opening the transcripts (where the full, uncapped output always lives).
_EXEC_OUTPUT_TOOLS = frozenset({RunCommandInput.TOOL_NAME, RunMetricInput.TOOL_NAME})
_TOOL_OUTPUT_TAIL = 2000  # chars, matching verify.end's stdout_tail/stderr_tail


def _output_tails(name: str, result: ToolResult) -> dict[str, str]:
    """Capped stdout/stderr tails for an execution tool's result, else {}."""
    if name not in _EXEC_OUTPUT_TOOLS or not isinstance(result, ExecResult | MetricResult):
        return {}
    return {
        "stdout_tail": result.stdout[-_TOOL_OUTPUT_TAIL:],
        "stderr_tail": result.stderr[-_TOOL_OUTPUT_TAIL:],
    }


class Approver(Protocol):
    def __call__(self, prompt: str, /, *, standing: bool = True) -> bool: ...


def _default_approver(prompt: str, /, *, standing: bool = True) -> bool:  # pragma: no cover
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


# Every tool that runs a command in the jail. They all execute model-influenced
# argv with the same reach, so one knob governs them: `run_commands = "no"`
# hides them, "ask" prompts (the session-allow marker keeps that to one prompt
# per run), "yes" runs. run_verify_command is here too -- its argv is the
# operator's when configured, but INFERRED from a file the model can edit when
# it is not, and either way it is a command in the same sandbox.
_COMMAND_TOOLS = frozenset(
    {
        RunCommandInput.TOOL_NAME,
        RunVerifyInput.TOOL_NAME,
        RunBackgroundInput.TOOL_NAME,
        StopBackgroundInput.TOOL_NAME,
    }
)


def jail_policy(
    root: Path,
    config: Config,
    isolation: IsolationLevel,
    argv: tuple[str, ...],
    *,
    timeout_s: float | None = None,
    extra_rw_paths: tuple[Path, ...] = (),
    extra_protect_paths: tuple[Path, ...] = (),
) -> JailPolicy:
    """The sandbox policy every LLM-influenced argv runs under.

    One owner, so every caller is confined identically: a foreground command, a
    detached one (`run_background`), and the baseline gate re-run all get the
    same protect paths, env, tool mounts and memory cap. The baseline once built
    its own and inherited no PATH, so every real gate exited 127 and the run was
    told its failure pre-existed.
    """
    # run_command reaches the network only under tool_network = "allow" (the
    # jailed child then shares the host network instead of an empty namespace).
    allow_network = config.sandbox.tool_network == "allow"
    protect_paths: list[Path] = []
    # STRICT only. A writable `.git` is not merely "recoverable": a jailed
    # command can plant a `filter.<n>.clean` in `.git/config` plus a
    # `.gitattributes`, and agent6's own auto-commit then executes it on the
    # HOST, outside the jail. Strict re-binds `.git` read-only, which needs a
    # mount namespace.
    #
    # Hardened has none, so the only tool is Landlock -- which has no deny
    # rules. Protecting `.git` there meant not granting the workspace ROOT,
    # because a Landlock grant is recursive and granting the root its own
    # create/remove rights grants them over `.git` too. That cost every
    # top-level write (`touch newfile`, `mkdir build`), which is too much to
    # pay for a protection the operator can have properly by using strict.
    # The in-process edit tools refuse `.git` writes on both isolation levels.
    if config.sandbox.protect_git and isolation == "strict":
        protect_paths.append((root / ".git").resolve())
    protect_paths.extend(extra_protect_paths)
    policy_kwargs: dict[str, Any] = {}
    if timeout_s is not None:
        policy_kwargs["timeout_s"] = timeout_s
    env = passthrough_env()
    # Toolchains need a writable cache root (go test -> $HOME/.cache/go-build,
    # cargo -> $CARGO_HOME, pip/uv likewise). The jail's /tmp is writable on both
    # isolation levels, so point HOME there.
    env.setdefault("HOME", "/tmp/agent6-home")  # noqa: S108 - resolved inside the jail
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # `uv run` inside the jail must use the venv the operator already synced: the
    # jail is offline and HOME is a fresh tmpfs, so a sync would re-resolve
    # against an empty cache and fail.
    env.setdefault("UV_NO_SYNC", "1")
    # Make operator-installed tools reachable: a controlled PATH extending
    # /usr/bin:/bin with the standard bin dirs, plus their real dirs as RO+exec
    # mounts. Without this a `uv run` verify dies 127.
    tool_path, tool_mounts = operator_tool_paths()
    env["PATH"] = tool_path
    return JailPolicy(
        cwd=root,
        argv=argv,
        isolation=isolation,
        env=tuple(sorted(env.items())),
        allow_network=allow_network,
        extra_protect_paths=tuple(protect_paths),
        extra_ro_paths=tuple(Path(p) for p in config.sandbox.extra_read_paths),
        extra_rw_paths=extra_rw_paths,
        tool_paths=tool_mounts,
        memory_limit_mb=config.sandbox.memory_limit_mb,
        **policy_kwargs,
    )


def _roster(shells: BackgroundShells) -> tuple[str, ...]:
    return tuple(v.line() for v in shells.roster())


class ToolDispatcher:
    """Runtime tool dispatcher. Constructed once per workflow run."""

    def __init__(
        self,
        *,
        root: Path,
        config: Config,
        isolation: IsolationLevel = "strict",
        approver: Approver | None = None,
        questioner: _Questioner | None = None,
        events: EventSink | None = None,
        curator: GraphCurator | None = None,
        run_root_node_id: str | None = None,
        mcp_manager: MCPManager | None = None,
        extra_protect_paths: tuple[Path, ...] = (),
        mode: Literal["run", "plan", "ask", "machine"] = "run",
        state_dir: Path | None = None,
        session_dir: Path | None = None,
        use_jail_session: bool = False,
    ) -> None:
        self._root = root.resolve()
        self._config = config
        self._isolation: IsolationLevel = isolation
        # In plan mode the LLM's tool list already omits apply_edit/apply_patch;
        # this is the defense-in-depth backstop so the dispatcher itself refuses
        # a source mutation even if something dispatched one directly.
        self._mode: Literal["run", "plan", "ask", "machine"] = mode
        # next() is atomic under the GIL; seats on the shared dispatcher get
        # distinct ids without a lock.
        self._call_seq = itertools.count(1)
        # Extra read-only paths layered into every run_command jail on top of
        # the strict-isolation protect_git bind (e.g. a running machine's own
        # .asm.toml + scripts bundle, so an agent state can't rewrite them
        # mid-run).
        self._extra_protect_paths = extra_protect_paths
        self._approver: Approver = approver or _default_approver
        self._questioner: _Questioner = questioner or _default_questioner
        self._events = events
        # Optional in-process GraphCurator + root-task id for the DAG-as-tool
        # surface. When wired, the dispatcher exposes add_task /
        # update_task / set_cursor / list_tasks.
        self._curator = curator
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
        # Background commands live under the run dir so they die with the run
        # and `sessions rm` clears them. None (tests, review dispatchers) leaves
        # them unwired: the tools raise ToolError, like the DAG tools.
        self._shells = BackgroundShells(session_dir / "shells") if session_dir is not None else None
        # One jail process for the whole run, opened on the first jailed
        # command and closed at teardown, so a run's commands share a netns, a
        # PID namespace and a /tmp and pay the setup once. RUN-SCOPED: a bare
        # dispatcher (a one-off tool, an embedder) has no run to scope it to
        # and keeps the per-command launcher.
        self._use_session = use_jail_session
        self._session: JailSession | None = None
        self._session_failed = False
        # The run's dir, for the effective command policy: the operator's
        # session choice and away-mode live there and can change mid-run.
        self._session_dir = session_dir
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
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
            ReadSessionInput.TOOL_NAME: self._read_session,
            FetchInput.TOOL_NAME: self._fetch,
            RunBackgroundInput.TOOL_NAME: self._run_background,
            ReadBackgroundInput.TOOL_NAME: self._read_background,
            StopBackgroundInput.TOOL_NAME: self._stop_background,
            # run_metric: LLM-exposed via LOOP_EXTRA_TOOLS so the
            # loop can call it after a successful verify when
            # [workflow.metric] is configured.
            RunMetricInput.TOOL_NAME: self._run_metric,
            # finish_session signals the loop should exit. Handler
            # just echoes the summary; the workflow checks for this tool name
            # in resp.tool_uses and terminates after dispatching it.
            FinishSessionInput.TOOL_NAME: self._finish_session,
            FinishPlanningInput.TOOL_NAME: self._finish_planning,
            AskUserInput.TOOL_NAME: self._ask_user,
            # DAG-as-tool. Handlers raise ToolError if no curator was
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
            ReadNotesInput.TOOL_NAME: self._read_notes,
            WriteNotesInput.TOOL_NAME: self._write_notes,
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

    def command_policy(self) -> str:
        """ "no" | "ask" | "yes" for this run, right now.

        Re-read rather than cached: an operator who denies for the session
        mid-run withdraws the tools from the next turn, and one who allows for
        the session stops being prompted from the next call.
        """
        configured = self._config.sandbox.run_commands
        if self._session_dir is None:
            return configured
        return effective_run_commands(configured, self._session_dir)

    def available_tool_names(self) -> tuple[str, ...]:
        names = list(self._available)
        # `no` withholds every command tool, run_verify_command included.
        if self.command_policy() == "no":
            names = [n for n in names if n not in _COMMAND_TOOLS]
        # No verify_command (and none inferred) -> a gateless run: hide
        # run_verify_command rather than offer a tool that would error.
        if not self._config.workflow.verify_command:
            names = [n for n in names if n != RunVerifyInput.TOOL_NAME]
        # `fetch` exists because a jailed command has no network. Where one
        # DOES (`tool_network = "allow"`), the worker can already run curl, and
        # two ways to do one thing is the thing we do not do.
        if self._config.sandbox.tool_network == "allow":
            names = [n for n in names if n != FetchInput.TOOL_NAME]
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

    def dispatch(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
        # Returns the typed result; the caller serializes it with to_wire() at
        # the single wire boundary (the loop / review seat / mcp server).
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
        max_chars = 2000 if name in ("finish_session", "finish_planning") else 200
        preview = truncate_args(raw_input, max_value_chars=max_chars)
        # Correlation id shared by this dispatch's call/result pair: concurrent
        # review seats interleave events through the one shared sink, and
        # name-based pairing cross-stamps same-name calls.
        cid = next(self._call_seq)
        self._emit("tool.call", name=name, args=preview, call_id=cid)
        try:
            result = self._dispatch_inner(name, raw_input)
        except ToolError as exc:
            self._emit("tool.result", name=name, ok=False, summary=str(exc), call_id=cid)
            raise
        except OperatorCommandUnexecutable as exc:
            # Not a model-fixable tool error: an operator verify/metric command
            # that cannot execute in the jail. Record the failed result for the
            # audit trail, then propagate (NOT wrapped as ToolError) so the loop
            # aborts the run loudly instead of surfacing it as a normal failure.
            self._emit("tool.result", name=name, ok=False, summary=str(exc), call_id=cid)
            raise
        except Exception as exc:
            self._emit("tool.result", name=name, ok=False, summary=str(exc), call_id=cid)
            raise ToolError(f"{name} failed: {exc}") from exc
        self._emit(
            "tool.result",
            name=name,
            ok=True,
            summary=result.summary(),
            call_id=cid,
            **_output_tails(name, result),
        )
        return result

    def _dispatch_inner(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
        """Resolve + execute a tool. Raises ToolError on a rejected/failed call;
        the caller (`dispatch`) owns the tool.call/tool.result events."""
        # MCP routing happens BEFORE the built-in handler check so mcp__* names
        # don't collide with the built-in "Unknown tool" error path.
        if name.startswith(MCP_TOOL_PREFIX):
            if not session_kind(self._mode).edits:
                # MCP tools are arbitrary external capabilities agent6 cannot
                # classify as read-only, so every non-run mode refuses them --
                # the same dispatcher backstop the built-in mutating tools get,
                # covering the documented read-only guarantee of plan/ask and
                # the machine-authoring "do not edit or run anything" contract.
                raise ToolError(f"{name} is not available in {self._mode} mode (read-only)")
            if self._mcp_manager is None:
                raise ToolError(f"{name}: MCP is not configured")
            try:
                return RawResult(self._mcp_manager.call(name, raw_input))
            except MCPError as exc:
                raise ToolError(str(exc)) from exc
        if name not in self._handlers:
            raise ToolError(f"Unknown tool: {name}")
        if name in _COMMAND_TOOLS and self.command_policy() == "no":
            raise ToolError(f"{name} is not available (run_commands = 'no')")
        if name == FetchInput.TOOL_NAME and self._config.sandbox.tool_network == "allow":
            raise ToolError(f"{name} is not available (a jailed command has the network)")
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
        if name not in mode_tools(self._mode).permitted:
            # Backstop the mode's tool surface at the dispatcher, not just by
            # omitting tools from the LLM's list: a tool-list regression or a
            # hallucinated name must not mutate the repo or run commands
            # (including the approval-gate-free metric command) from a
            # read-only mode, pause a non-run loop (ask_user), or write
            # cross-run memories. Enforcing membership in the same surface
            # `tool_definitions` exposes means the two cannot drift.
            raise ToolError(f"{name} is not available in {self._mode} mode")
        return self._run_handler(name, raw_input)

    def _run_handler(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
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

    def _agent6_docs(self, raw: dict[str, Any]) -> ToolResult:
        return agent6_docs(raw)

    def _read_file(self, raw: dict[str, Any]) -> ToolResult:
        return read_file(self._root, raw)

    def _list_dir(self, raw: dict[str, Any]) -> ToolResult:
        return list_dir(self._root, raw)

    def _grep(self, raw: dict[str, Any]) -> ToolResult:
        return grep(self._root, raw)

    def _apply_edit(self, raw: dict[str, Any]) -> ToolResult:
        return apply_edit(self._root, self._config, self._extra_protect_paths, self._index, raw)

    def _apply_patch(self, raw: dict[str, Any]) -> ToolResult:
        return apply_patch(self._root, self._config, self._extra_protect_paths, self._index, raw)

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
        """Public passthrough to ``SymbolIndex.hot_symbols``, sharing the
        dispatcher's index so an already-paid scan is not repeated."""
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

    def _outline(self, raw: dict[str, Any]) -> ToolResult:
        return outline(self._root, self._ensure_index, raw)

    def _find_definition(self, raw: dict[str, Any]) -> ToolResult:
        return find_definition(self._root, self._ensure_index, raw)

    def _find_references(self, raw: dict[str, Any]) -> ToolResult:
        return find_references(self._root, self._ensure_index, raw)

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

    def _find_definition_lsp(self, raw: dict[str, Any]) -> ToolResult:
        return find_definition_lsp(self._root, self._ensure_lsp, raw)

    def _find_references_lsp(self, raw: dict[str, Any]) -> ToolResult:
        return find_references_lsp(self._root, self._ensure_lsp, raw)

    def close(self) -> None:
        """Release subprocess resources (LSP server).

        Idempotent. Safe to call from CLI teardown alongside
        ``mcp_manager.close()``.
        """
        if self._shells is not None:
            self._shells.stop_all()
        self.close_jail_session()
        if self._lsp is not None:
            self._lsp.close()
            self._lsp = None

    def adopt_verify_command(self, argv: tuple[str, ...]) -> bool:
        """Adopt a verify command mid-run: the loop's gateless adoption after
        the tree materializes (see Workflow._maybe_adopt_verify). Same trust
        as preflight's in-memory injection: derived from the repo's own
        AGENTS.md fence or project signals, operator-origin, never persisted.

        False (nothing adopted) when a bare argv[0] does not resolve on the
        jail PATH: adopting a gate the sandbox cannot execute would turn a
        would-be honest settle into an unexecutable-verify abort. Path-form
        commands are accepted as-is (they resolve against the mounted cwd)."""
        if self.command_policy() == "no":
            # Every command tool is withheld, the gate included. Adopting one
            # would gate the run on something it can never run.
            return False
        exe = argv[0]
        if "/" not in exe and shutil.which(exe, path=jail_search_path()) is None:
            return False
        self._config = self._config.with_verify_command(argv)
        return True

    def _run_verify(self, _raw: dict[str, Any]) -> ExecResult:
        argv = tuple(self._config.workflow.verify_command)
        if self.command_policy() == "ask" and not self._approver(
            f"Allow run_verify_command: {shlex.join(argv)}"
        ):
            raise ToolDenied("run_verify_command not approved (sandbox.run_commands='ask')")
        # per-call timeout from config. Defaults to the jail's
        # general 600s but bench configs crank it down so infinite-loop
        # edits fail fast instead of burning ~10 min of wall per attempt.
        timeout_s = self._config.workflow.verify_timeout_s
        self._emit("verify.start", cmd=list(argv), timeout_s=timeout_s)
        res = self._run_argv_in_jail(argv, label="verify_command", timeout_s=timeout_s)
        # Name the gate in the result: it is the operator's command, or one
        # inferred from the repo, so the worker cannot otherwise tell WHICH
        # thing judged it -- or that it is judging the wrong thing (stale_gate).
        res = replace(res, command=argv)
        self._emit(
            "verify.end",
            cmd=list(argv),
            exit_code=res.returncode,
            duration_s=res.duration_s,
            timeout_s=timeout_s,
            stdout_tail=res.stdout[-2000:],
            stderr_tail=res.stderr[-2000:],
        )
        if res.exec_failed:
            raise OperatorCommandUnexecutable(
                f"verify_command {list(argv)} could not be executed in the sandbox: "
                f"{res.stderr}. The jail PATH is /usr/bin:/bin plus the standard bin "
                "dirs that exist (/usr/local/bin, ~/.local/bin, ~/.cargo/bin, "
                "/opt/homebrew/bin, /snap/bin), each mounted read-only; the command is on "
                "none of them. Install the tool into one of those on the host, use a "
                "path inside the workspace (e.g. .venv/bin/pytest), or grant its real "
                "directory via sandbox.extra_read_paths."
            )
        return res

    def _run_command(self, raw: dict[str, Any]) -> ExecResult:
        args = RunCommandInput.model_validate(raw)
        if self.command_policy() == "ask":
            # A shell-style command line, not a Python tuple repr: the operator
            # is approving a command, so show it the way they would type it.
            ok = self._approver(f"Allow run_command: {shlex.join(args.argv)}")
            if not ok:
                # The gate can't tell a human "no" from the policy auto-deny of
                # an unattended run, so the message blames neither and names
                # the knob.
                raise ToolDenied("run_command not approved (sandbox.run_commands='ask')")
        return self._run_argv_in_jail(args.argv, label="run_command")

    def _background(self) -> BackgroundShells:
        if self._shells is None:
            raise ToolError("background commands need a run directory; none was wired")
        return self._shells

    def _fetch(self, raw: dict[str, Any]) -> FetchResult:
        args = FetchInput.model_validate(raw)
        try:
            target = check_url(args.url)
        except FetchRefused as exc:
            raise ToolError(str(exc)) from exc
        # On the list: read it. Off the list: ask. The list IS the standing
        # approval, and a prompt per doc read only trains a reflexive yes --
        # but a GET can carry data out in its path, so a host the operator
        # never named is their call, and an absent one is a no (the away-mode
        # approver refuses without waiting).
        if not host_allowed(target.host, self._config.sandbox.fetch_hosts) and not self._approver(
            f"Allow fetch: {target.prompt()}", standing=False
        ):
            raise ToolDenied(
                f"fetch not approved for {target.host} (add it to sandbox.fetch_hosts to allow it)"
            )
        try:
            got = fetch(target)
        except FetchRefused as exc:
            raise ToolError(str(exc)) from exc
        return FetchResult(
            url=got.url,
            status=got.status,
            content_type=got.content_type,
            body=got.body,
            location=got.location,
        )

    def _read_session(self, raw: dict[str, Any]) -> SessionsResult:
        args = ReadSessionInput.model_validate(raw)
        if self._state_dir is None:
            raise ToolError("read_session needs the project state dir; none was wired")
        lines = roster(self._state_dir, args.query).lines()
        if not args.id:
            return SessionsResult(sessions=lines)
        layout = session_layout(self._state_dir, args.id)
        if layout is None:
            raise ToolError(f"no session {args.id!r} in this project")
        return SessionsResult(
            sessions=lines, conversation=conversation(layout, max_chars=args.max_chars)
        )

    def _run_background(self, raw: dict[str, Any]) -> BackgroundResult:
        args = RunBackgroundInput.model_validate(raw)
        shells = self._background()
        if self.command_policy() == "ask" and not self._approver(
            f"Allow run_background: {shlex.join(args.argv)}"
        ):
            raise ToolDenied("run_background not approved (sandbox.run_commands='ask')")
        try:
            shells.start(
                args.argv,
                lambda argv, rw: self._jail_policy(argv, extra_rw_paths=rw),
                session=self._run_session(),
            )
        except BackgroundError as exc:
            raise ToolError(str(exc)) from exc
        return BackgroundResult(shells=_roster(shells))

    def _read_background(self, raw: dict[str, Any]) -> BackgroundResult:
        args = ReadBackgroundInput.model_validate(raw)
        shells = self._background()
        if not args.id:
            return BackgroundResult(shells=_roster(shells))
        try:
            _view, output = shells.read(args.id, tail_lines=args.tail_lines)
        except BackgroundError as exc:
            raise ToolError(str(exc)) from exc
        return BackgroundResult(shells=_roster(shells), output=output)

    def _stop_background(self, raw: dict[str, Any]) -> BackgroundResult:
        args = StopBackgroundInput.model_validate(raw)
        shells = self._background()
        try:
            shells.stop(args.id)
        except BackgroundError as exc:
            raise ToolError(str(exc)) from exc
        return BackgroundResult(shells=_roster(shells))

    def _ask_user(self, raw: dict[str, Any]) -> ToolResult:
        return ask_user(self._questioner, raw)

    def _finish_session(self, raw: dict[str, Any]) -> ToolResult:
        return finish_session(raw)

    def _finish_planning(self, raw: dict[str, Any]) -> ToolResult:
        return finish_planning(raw)

    # DAG-as-tool handlers. All raise ToolError if no curator
    # was wired so standalone test instantiation works unchanged.

    def _dag_add_task(self, raw: dict[str, Any]) -> ToolResult:
        return add_task(self._curator, self._run_root_node_id, raw)

    def _dag_update_task(self, raw: dict[str, Any]) -> ToolResult:
        return update_task(self._curator, raw)

    def _dag_set_cursor(self, raw: dict[str, Any]) -> ToolResult:
        return set_cursor(self._curator, raw)

    def _dag_add_dependency(self, raw: dict[str, Any]) -> ToolResult:
        return add_dependency(self._curator, raw)

    def _dag_list_tasks(self, raw: dict[str, Any]) -> ToolResult:
        return list_tasks(self._curator, raw)

    # Cross-run memory handlers. Writes go through trusted code
    # (agent6.memory) to fixed markdown files under <state_dir>/memories/,
    # outside the workspace and the jail; the LLM controls only the scope
    # (schema-validated literal) and the note text, which is inert data.

    def _read_notes(self, raw: dict[str, Any]) -> ToolResult:
        return read_notes(self._state_dir, raw)

    def _write_notes(self, raw: dict[str, Any]) -> ToolResult:
        return write_notes(self._state_dir, raw)

    def _add_memory(self, raw: dict[str, Any]) -> ToolResult:
        return add_memory(self._state_dir, raw)

    def _invalidate_memory(self, raw: dict[str, Any]) -> ToolResult:
        return invalidate_memory(self._state_dir, raw)

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

    def _use_skill(self, raw: dict[str, Any]) -> ToolResult:
        return use_skill(self.resolved_skills, raw)

    def _run_metric(self, _raw: dict[str, Any]) -> MetricResult:
        """Run ``cfg.workflow.metric.command`` in the jail.

        Return shape mirrors `_run_argv_in_jail` (returncode / stdout /
        stderr / duration_s) plus ``score``: the ``pattern`` regex's first
        capture group as a float, or null when it does not match or parse (the
        agent can then grep stdout itself). Raises ToolError when no metric is
        configured.
        """
        metric_cfg = self._config.workflow.metric
        if metric_cfg is None:
            raise ToolError("run_metric_command: no [workflow.metric] configured")
        argv = tuple(metric_cfg.command)
        self._emit("metric.start", cmd=list(argv))
        res = self._run_argv_in_jail(
            argv, label="metric_command", timeout_s=self._config.workflow.verify_timeout_s
        )
        if res.exec_failed:
            raise OperatorCommandUnexecutable(
                f"metric_command {list(argv)} could not be executed in the sandbox: "
                f"{res.stderr}. See run_verify_command's note: PATH is /usr/bin:/bin "
                "plus the standard bin dirs; install the tool into one of those on the "
                "host, use a path inside the workspace, or grant its real directory "
                "via sandbox.extra_read_paths."
            )
        score = parse_metric_score(res.stdout, res.stderr, pattern=metric_cfg.pattern)
        self._emit(
            "metric.end",
            cmd=list(argv),
            exit_code=res.returncode,
            duration_s=res.duration_s,
            stdout_tail=res.stdout[-2000:],
            stderr_tail=res.stderr[-2000:],
            score=score,
        )
        return MetricResult.from_exec(res, score)

    def _jail_policy(
        self,
        argv: tuple[str, ...],
        *,
        timeout_s: float | None = None,
        extra_rw_paths: tuple[Path, ...] = (),
    ) -> JailPolicy:
        return jail_policy(
            self._root,
            self._config,
            self._isolation,
            argv,
            timeout_s=timeout_s,
            extra_rw_paths=extra_rw_paths,
            extra_protect_paths=self._extra_protect_paths,
        )

    def _run_session(self) -> JailSession | None:
        """The run's jail process, or None to give each command its own.

        STRICT only: the other levels have no PID namespace to bound what a
        command leaves running. A session that cannot start (an older bundled
        launcher, a host without namespaces) answers None once and is not
        retried, so the per-command path is the fallback rather than the run
        failing.

        Its confinement is fixed when it opens, so the policy is the run's, not
        the first command's: every command in the run gets the same one, and
        the background log root is granted before any command asks for it.
        """
        if not self._use_session or self._isolation != "strict":
            return None
        if self._session is None and not self._session_failed:
            rw = () if self._shells is None else (self._shells.log_root,)
            try:
                self._session = JailSession.open(self._jail_policy(("true",), extra_rw_paths=rw))
            except (JailUnavailableError, OSError):
                self._session_failed = True
        return self._session

    def close_jail_session(self) -> None:
        """End the run's jail process; its PID namespace takes the rest down."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def _run_argv_in_jail(
        self,
        argv: tuple[str, ...],
        *,
        label: str,
        timeout_s: float | None = None,
    ) -> ExecResult:
        policy = self._jail_policy(argv, timeout_s=timeout_s)
        try:
            session = self._run_session()
            res: CommandResult = (
                session.run(argv, env=policy.env, timeout_s=policy.timeout_s)
                if session is not None
                else run_in_jail(policy)
            )
        except JailUnavailableError as exc:
            raise ToolError(f"{label}: jail unavailable: {exc}") from exc
        return ExecResult(
            returncode=res.returncode,
            stdout=res.stdout[-20_000:],
            stderr=res.stderr[-20_000:],
            duration_s=res.duration_s,
            exec_failed=res.exec_failed,
        )
