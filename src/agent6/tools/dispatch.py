# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool dispatch: validate each LLM tool call and execute it.

File reads and writes resolve through the workspace boundary: the repo root plus the operator's
extra grants and the per-repo memory dir. Commands run jailed, in the run's `JailSession` or a
per-command `run_in_jail`. The `run_commands` gate ("deny", "ask", "allow") is enforced here.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import pathlib
import shlex
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any, Literal

import pydantic

from agent6 import errors, event_log, evidence, kinds, memory, paths, skills
from agent6.config import Config
from agent6.graph import curator as graph_curator
from agent6.sandbox import jail, jail_session, tool_paths
from agent6.sessions import ipc
from agent6.sessions import layout as sessions_layout
from agent6.tools import (
    _arguments,
    _command_tools,
    _control_tools,
    _dag_tools,
    _fs_tools,
    _nav_tools,
    _result_format,
    _skill_tools,
    background,
    fetch,
    index,
    mcp_client,
    operator_prompts,
    results,
    schema,
    sessions,
)
from agent6.tools import errors as tools_errors
from agent6.tools import policy as tools_policy
from agent6.vcs import git_read

# An MCP approval prompt carries the full arguments up to this bound, then a session-dir file.
_PAYLOAD_IDS = itertools.count(1)
_APPROVAL_PROMPT_MAX_CHARS = 4096


# Every tool that runs a command in the jail, governed by the one `run_commands` knob: the
# verify and metric commands are inferred from or run at the model's asking in the same sandbox.
_COMMAND_TOOLS = frozenset(
    {
        schema.RunCommandInput.TOOL_NAME,
        schema.RunVerifyInput.TOOL_NAME,
        schema.RunMetricInput.TOOL_NAME,
        schema.StopBackgroundInput.TOOL_NAME,
    }
)

# Bench arms keyed by AGENT6_SYMBOL_TOOLS: "none" hides the symbol tools; unset or unknown hides
# nothing, and the bench harness validates the arm name.
_SYMBOL_TOOL_ARMS: dict[str, frozenset[str]] = {
    "none": frozenset(
        {
            schema.OutlineInput.TOOL_NAME,
            schema.FindDefinitionInput.TOOL_NAME,
            schema.FindReferencesInput.TOOL_NAME,
        }
    ),
}


class ToolDispatcher:
    """The tool dispatcher of one execution, or of a bare caller such as a review seat.

    Args:
        root: The workspace root.
        config: The run's config.
        isolation: The resolved isolation level; read by the prompt builder too.
        prompts: The operator gate; a bare dispatcher gets one over its own journal.
        events: The run's event sink; None emits nothing.
        curator: The task graph; None answers "no curator" from the DAG tools.
        run_root_node_id: The parent `add_task` falls back to.
        mcp_manager: The MCP servers; None routes no `mcp__` name.
        extra_protect_paths: Paths every jail re-binds read-only and the edit tools refuse,
            such as a running machine's own bundle; read by the prompt assembly too.
        worktree_git_dir: The repository git dir recorded for a fork's linked worktree.
        mode: The mode whose tool surface the dispatcher enforces as its backstop.
        state_dir: The per-repo state dir, for the sessions roster and the memory grant.
        session_dir: The run's dir, for background commands, the file bridge and the
            effective command policy; None leaves them unwired.
        use_jail_session: Whether one jail process serves every command of the run.
        session_net: The run's session network; None makes the dispatcher's own.
    """

    def __init__(
        self,
        *,
        root: pathlib.Path,
        config: Config,
        isolation: kinds.IsolationLevel = "full",
        prompts: operator_prompts.OperatorPrompts | None = None,
        events: event_log.EventSink | None = None,
        curator: graph_curator.GraphCurator | None = None,
        run_root_node_id: str | None = None,
        mcp_manager: mcp_client.MCPManager | None = None,
        extra_protect_paths: tuple[pathlib.Path, ...] = (),
        worktree_git_dir: pathlib.Path | None = None,
        mode: Literal["run", "plan", "ask", "machine"] = "run",
        state_dir: pathlib.Path | None = None,
        session_dir: pathlib.Path | None = None,
        use_jail_session: bool = False,
        session_net: jail.SessionNetwork | None = None,
        shared_writes: tuple[pathlib.Path, ...] = (),
    ) -> None:
        self._root = root.resolve()
        self.verify_tree: Callable[[], str] = self._verification_tree
        self.execution_id: str | None = None
        self.last_verification: evidence.VerificationEvidence | None = None
        self._config = config
        # Memory files are read and edited with the ordinary tools, in-process only.
        mem = memory.memory_dir(state_dir) if state_dir is not None else None
        if mem is not None:
            # The grant's target must exist: the model cannot mkdir outside the jail.
            paths.mkdir_for_real_user(mem)
        # Resolved once: every jail launch and the file tools hold the same directories. A
        # machine state's grants are judged against the regions its other states write too.
        self._grants = tools_policy.resolve_grants(config, self._root, shared_writes=shared_writes)
        self._ws = tools_policy.workspace_for(
            config, self._root, memory_dir=mem, grants=self._grants
        )
        self.isolation: kinds.IsolationLevel = isolation
        self._mode: Literal["run", "plan", "ask", "machine"] = mode
        # next() is atomic under the GIL, so seats on a shared dispatcher need no lock.
        self._call_seq = itertools.count(1)
        self.extra_protect_paths = extra_protect_paths
        self._worktree_git_dir = worktree_git_dir
        self._events = events
        self._prompts = prompts or operator_prompts.OperatorPrompts(
            journal=self._emit, session_dir=session_dir
        )
        # The call being dispatched, per thread: concurrent review seats share one dispatcher.
        self._gating = threading.local()
        # Seconds blocked on the operator; the loop subtracts them from its wall clock.
        self.operator_wait_s = 0.0
        self._curator = curator
        # Read by the tool list: the DAG tools answer "no curator" for a run built without one.
        self.dag_available = curator is not None
        self._run_root_node_id = run_root_node_id
        self._mcp_manager = mcp_manager
        self._state_dir = state_dir
        # Background commands live under the run dir so they die with the run.
        self._shells = (
            background.BackgroundShells(session_dir / background.SHELLS_DIR)
            if session_dir is not None
            else None
        )
        # One jail process per run, so its commands share a netns, a PID namespace and a /tmp.
        self._use_session = use_jail_session
        self._session_net = session_net
        self._own_session_net: jail.SessionNetwork | None = None
        self._session: jail_session.JailSession | None = None
        self._session_failed = False
        # Two threads racing the lazy open would leak a jail process and its namespaces.
        self._session_lock = threading.Lock()
        self._session_dir = session_dir
        self._jail_access = _command_tools.JailAccess(
            get_policy=self._jail_policy, get_session=self._run_session, net_for=self._net_for
        )
        self._handlers: dict[str, Callable[[dict[str, Any]], results.ToolResult]] = {
            schema.Agent6DocsInput.TOOL_NAME: _fs_tools.agent6_docs,
            schema.ReadFileInput.TOOL_NAME: lambda raw: _fs_tools.read_file(self._ws, raw),
            schema.ListDirInput.TOOL_NAME: lambda raw: _fs_tools.list_dir(self._ws, raw),
            schema.OutlineInput.TOOL_NAME: lambda raw: _nav_tools.outline(
                self._ws, self.symbol_index, raw
            ),
            schema.FindDefinitionInput.TOOL_NAME: lambda raw: _nav_tools.find_definition(
                self._ws, self.symbol_index, raw
            ),
            schema.FindReferencesInput.TOOL_NAME: lambda raw: _nav_tools.find_references(
                self._ws, self.symbol_index, raw
            ),
            schema.ApplyEditInput.TOOL_NAME: lambda raw: _fs_tools.apply_edit(
                self._ws, self._config, self.extra_protect_paths, self._index, raw
            ),
            schema.ApplyPatchInput.TOOL_NAME: lambda raw: _fs_tools.apply_patch(
                self._ws, self._config, self.extra_protect_paths, self._index, raw
            ),
            schema.RunVerifyInput.TOOL_NAME: self._run_verify,
            schema.RunCommandInput.TOOL_NAME: self._run_command,
            schema.ReadSessionInput.TOOL_NAME: self._read_session,
            schema.FetchInput.TOOL_NAME: self._fetch,
            schema.ReadBackgroundInput.TOOL_NAME: self._read_background,
            schema.StopBackgroundInput.TOOL_NAME: self._stop_background,
            schema.RunMetricInput.TOOL_NAME: self._run_metric,
            schema.FinishSessionInput.TOOL_NAME: _control_tools.finish_session,
            schema.FinishPlanningInput.TOOL_NAME: _control_tools.finish_planning,
            schema.AskUserInput.TOOL_NAME: self._ask_user,
            schema.DagAddTaskInput.TOOL_NAME: lambda raw: _dag_tools.add_task(
                self._curator, self._run_root_node_id, raw
            ),
            schema.DagUpdateTaskInput.TOOL_NAME: lambda raw: _dag_tools.update_task(
                self._curator, raw
            ),
            schema.DagListTasksInput.TOOL_NAME: lambda raw: _dag_tools.list_tasks(
                self._curator, raw
            ),
            schema.UseSkillInput.TOOL_NAME: lambda raw: _skill_tools.use_skill(
                self._ws, self.resolved_skills, raw
            ),
        }
        self._index: index.SymbolIndex | None = None
        # Concurrent review seats must not double-build the index.
        self._index_lock = threading.Lock()
        # Resolved once on first use, a disk scan of the configured skill dirs.
        self._skills_cache: skills.ResolvedSkills | None = None

    def set_run_root_node_id(self, node_id: str | None) -> None:
        """Set the parent `add_task` falls back to, once the harness has seeded the root task."""
        self._run_root_node_id = node_id

    def command_policy(self) -> str:
        """Return the run's command policy right now: "deny", "ask" or "allow".

        Re-read rather than cached: an operator who denies for the session mid-run withdraws
        the tools from the next turn.

        Returns:
            `[sandbox].run_commands` as `ipc.effective_run_commands` resolves it for the
            session, or as configured when no run dir is wired.
        """
        configured = self._config.sandbox.run_commands
        if self._session_dir is None:
            return configured
        return ipc.effective_run_commands(configured, self._session_dir)

    def metric_configured(self) -> bool:
        """Return whether `[harness.metric]` gives `run_metric_command` anything to run."""
        return self._config.harness.metric is not None

    def tool_is_withheld(self, name: str) -> bool:
        """Return whether the model is denied the tool, extras included."""
        return name in _COMMAND_TOOLS and self.command_policy() == "deny"

    def _tool_refusal(self, name: str) -> str | None:
        """Return why a built-in tool is withheld right now, or None when it is offered."""
        if self.tool_is_withheld(name):
            return "not available (run_commands = 'deny')"
        # `fetch` is redundant when a jailed command already has the network.
        if (
            name == schema.FetchInput.TOOL_NAME
            and tools_policy.resolve_network(self._config, self.isolation) == "host"
        ):
            return "not available (a jailed command has the network)"
        if name in _SYMBOL_TOOL_ARMS.get(os.environ.get("AGENT6_SYMBOL_TOOLS", ""), frozenset()):
            return "not available (AGENT6_SYMBOL_TOOLS)"
        if (
            os.environ.get("AGENT6_DISABLE_APPLY_EDIT") == "1"
            and name == schema.ApplyEditInput.TOOL_NAME
        ):
            return f"{name} is disabled (AGENT6_DISABLE_APPLY_EDIT=1); use apply_patch instead"
        return None

    def available_tool_names(self) -> tuple[str, ...]:
        """Return the base tools on offer right now, MCP tools included, sorted."""
        names = [
            cls.TOOL_NAME for cls in schema.ALL_TOOLS if self._tool_refusal(cls.TOOL_NAME) is None
        ]
        # A gateless run hides run_verify_command rather than offer a tool that would error.
        if not self._config.harness.verify_command:
            names = [n for n in names if n != schema.RunVerifyInput.TOOL_NAME]
        names.extend(d.qualified_name for d in self.mcp_descriptors())
        return tuple(sorted(names))

    def mcp_denied(self, server: str) -> bool:
        """Return whether the operator answered "deny all" for the server this session.

        Read at both gates: the tool list drops the server, and the call gate refuses it, since
        the model still has the previous turn's list in context.

        Args:
            server: The server's name in config.

        Returns:
            Whether its session deny marker is set; False when no run dir is wired.
        """
        if self._session_dir is None:
            return False
        return ipc.session_deny_set(self._session_dir, f"{ipc.MCP_SCOPE_PREFIX}{server}")

    def mcp_descriptors(self) -> tuple[mcp_client.MCPToolDescriptor, ...]:
        """Return the MCP tools on offer right now, minus any server denied for the session."""
        if self._mcp_manager is None:
            return ()
        return tuple(
            d for d in self._mcp_manager.descriptors() if not self.mcp_denied(d.server_name)
        )

    def dispatch(self, name: str, raw_input: dict[str, Any]) -> results.ToolResult:
        """Execute one tool call, journaling a `tool.call` and `tool.result` pair around it.

        The pair is emitted here, before any guard and outside the model's reach, so a
        rejected call still produces its result and a prompt injection cannot fake success.

        Args:
            name: The tool's name.
            raw_input: The model's arguments.

        Returns:
            The typed result; the caller serializes it at the wire boundary.

        Raises:
            ToolError: The call was refused, its arguments invalid, or the handler failed.
            OperatorCommandUnexecutableError: An operator verify or metric command cannot run in
                the jail; the loop aborts rather than surface it to the model.
        """
        # The finish tools' `summary` is the human end-of-run statement: kept whole.
        max_chars = 2000 if name in ("finish_session", "finish_planning") else 200
        preview = _result_format.truncate_args(raw_input, max_value_chars=max_chars)
        # Concurrent review seats interleave events, and name-based pairing cross-stamps calls.
        cid = next(self._call_seq)
        self._emit("tool.call", name=name, args=preview, call_id=cid)
        outer = self._gating_call_id()
        self._gating.call_id = cid
        try:
            result = self._dispatch_inner(name, raw_input)
        except tools_errors.ToolError as exc:
            self._emit("tool.result", name=name, ok=False, summary=str(exc), call_id=cid)
            raise
        except tools_errors.OperatorCommandUnexecutableError as exc:
            self._emit("tool.result", name=name, ok=False, summary=str(exc), call_id=cid)
            raise
        except pydantic.ValidationError as exc:
            message = _arguments.invalid_arguments(exc)
            self._emit("tool.result", name=name, ok=False, summary=message, call_id=cid)
            raise tools_errors.ToolError(message) from exc
        except Exception as exc:
            self._emit("tool.result", name=name, ok=False, summary=str(exc), call_id=cid)
            raise tools_errors.ToolError(f"failed: {exc}") from exc
        finally:
            self._gating.call_id = outer
        self._emit(
            "tool.result",
            name=name,
            ok=True,
            summary=result.summary(),
            call_id=cid,
            **_result_format.output_tails(name, result),
        )
        return result

    def _dispatch_inner(self, name: str, raw_input: dict[str, Any]) -> results.ToolResult:
        """Resolve and execute a tool; `dispatch` owns the events around it.

        Args:
            name: The tool's name; an `mcp__` name routes to its MCP server.
            raw_input: The model's arguments.

        Returns:
            The handler's result.

        Raises:
            ToolError: The tool is unknown, withheld, outside the mode's surface, or failed.
        """
        if name.startswith(mcp_client.MCP_TOOL_PREFIX):
            if not kinds.session_kind(self._mode).edits:
                # An MCP tool cannot be classified as read-only, so every non-run mode refuses it.
                raise tools_errors.ToolError(f"not available in {self._mode} mode (run mode only)")
            if self._mcp_manager is None:
                raise tools_errors.ToolError("MCP is not configured")
            self._approve_mcp_call(name, raw_input)
            try:
                return results.RawResult(self._mcp_manager.call(name, raw_input))
            except mcp_client.MCPError as exc:
                raise tools_errors.ToolError(str(exc)) from exc
        if name not in self._handlers:
            raise tools_errors.ToolError(f"Unknown tool: {name}")
        if name == schema.RunVerifyInput.TOOL_NAME and self.tool_is_withheld(name):
            return self.run_verify()  # records the denied attempt as evidence, then refuses
        if (refusal := self._tool_refusal(name)) is not None:
            raise tools_errors.ToolError(refusal)
        if name not in schema.mode_tools(self._mode).permitted:
            # The backstop: a hallucinated name must not mutate the repo from a read-only mode.
            raise tools_errors.ToolError(f"not available in {self._mode} mode")
        return self._run_handler(name, raw_input)

    def _run_handler(self, name: str, raw_input: dict[str, Any]) -> results.ToolResult:
        """Execute the handler, retrying once with stringified JSON arguments coerced.

        Args:
            name: The built-in tool's name, a key of `_handlers`.
            raw_input: The model's arguments, or `{"_raw_arguments": text}` when the provider
                could not parse them as a JSON object.

        Returns:
            The handler's result.

        Raises:
            ToolError: The provider could not parse the arguments as JSON.
            ValidationError: The arguments do not fit the schema, after the coercion too.
        """
        # The provider left the `_raw_arguments` sentinel; a schema error about it would misdirect.
        if set(raw_input) == {"_raw_arguments"}:
            raw = raw_input.get("_raw_arguments")
            raw_len = len(raw) if isinstance(raw, str) else 0
            if raw_len > 20_000:
                # The arguments ran away to the output-token ceiling; "resend" would repeat that.
                raise tools_errors.ToolError(
                    "the arguments were cut off mid-generation"
                    f" ({raw_len // 1000} KB, truncated before the JSON closed)."
                    " Do NOT resend the same call. Emit a much smaller call:"
                    " short literal values only (keep any pattern or argument"
                    " under a couple hundred characters), and split broad work"
                    " into several small calls."
                )
            raise tools_errors.ToolError(
                "the arguments were not a JSON object. Resend the call with a"
                " single valid JSON object of arguments."
            )
        try:
            return self._handlers[name](raw_input)
        except pydantic.ValidationError as exc:
            coerced = _arguments.coerce_stringified_args(raw_input, exc)
            if coerced is None:
                raise
            try:
                return self._handlers[name](coerced)
            except pydantic.ValidationError:
                # The coercion guessed wrong; the original error is the honest one.
                raise exc from None

    def _emit(self, event_type: str, /, **fields: Any) -> None:
        """Write one event to the sink, when there is one."""
        if self._events is not None:
            self._events.emit(event_type, **fields)

    def _gating_call_id(self) -> int | None:
        """Return the call this thread is dispatching, or None outside a dispatch."""
        return getattr(self._gating, "call_id", None)

    def symbol_index(self) -> index.SymbolIndex:
        """Return the dispatcher's shared symbol index, built once."""
        if self._index is None:
            with self._index_lock:
                if self._index is None:
                    self._index = index.SymbolIndex(self._ws)
        return self._index

    def settle_background(self) -> None:
        """Write down the ending of any background command that has finished.

        Called at the turn boundary: only an observed exit reaches disk.
        """
        if self._shells is not None:
            self._shells.settle()

    def close(self) -> None:
        """Stop the background commands and close the jail session; idempotent."""
        if self._shells is not None:
            self._shells.stop_all()
        with self._session_lock:
            if self._session is not None:
                survivors = self._session.close()
                self._session = None
                if survivors:
                    self._emit("jail.degraded", detail=jail.survivors_message(survivors))
        if self._own_session_net is not None:  # never the run's; that is its own to close
            self._own_session_net.close()
            self._own_session_net = None

    def adopt_verify_command(self, argv: tuple[str, ...]) -> bool:
        """Adopt a verify command mid-run, once a gateless run's tree has materialized.

        The same trust as preflight's injection: derived from the repo's own AGENTS.md fence or
        project signals, never persisted.

        Args:
            argv: The command.

        Returns:
            Whether it was adopted; False when commands are withheld or a bare argv[0] does not
            resolve on the jail PATH, since a gate the sandbox cannot run would abort the run.
        """
        if self.command_policy() == "deny":
            return False
        if not tool_paths.jail_resolves(argv[0]):
            return False
        self._config = self._config.with_verify_command(argv)
        return True

    def drop_verify_command(self) -> None:
        """Drop the verify command: the gate proved unrunnable, so the run is gateless again."""
        self._config = self._config.with_verify_command(())

    def _approve_mcp_call(self, name: str, raw_input: dict[str, Any]) -> None:
        """Gate one MCP tool call on its server's `approve`.

        A server's tools are asked about like a command, on their own scope: "allow all" for one
        server grants that server alone. The prompt holds the arguments whole. In a run, past
        `_APPROVAL_PROMPT_MAX_CHARS`, it shows their start and names the run-dir file that holds
        them all: a clipped argument is consent to an operation the operator never saw.

        Args:
            name: The `mcp__<server>__<tool>` name the model called.
            raw_input: The model's arguments.

        Raises:
            ToolError: The server is not configured (the scope becomes a filename, and the LLM
                chooses tool names), or was denied for the session.
            ToolDeniedError: The operator did not approve.
        """
        server, _tool = mcp_client.split_tool_name(name)
        entry = self._config.mcp.servers.get(server)
        if entry is None:
            raise tools_errors.ToolError(f"unknown MCP server in {name!r}")
        if self.mcp_denied(server):
            raise tools_errors.ToolError(f"not available ({server!r} was denied for this session)")
        if entry.approve == "allow":
            return
        args = json.dumps(raw_input, ensure_ascii=False, sort_keys=True)
        if len(args) > _APPROVAL_PROMPT_MAX_CHARS and self._session_dir is not None:
            # A wall of text is as unread as a clip: the complete payload goes to a file a jailed
            # command cannot reach, one per prompt since review seats share a dispatcher.
            full = self._session_dir / f"approval_payload-{next(_PAYLOAD_IDS)}.json"
            full.write_text(args, encoding="utf-8")
            args = (
                f"{args[:_APPROVAL_PROMPT_MAX_CHARS]}"
                f" ...[{len(args)} chars total; full payload: {full}]"
            )
        if not self._approve(f"Allow {name}: {args}", scope=f"{ipc.MCP_SCOPE_PREFIX}{server}"):
            raise tools_errors.ToolDeniedError(
                f"{name} not approved (set [mcp.servers.{server}].approve = 'allow' to stop asking)"
            )

    def _approve(self, prompt: str, *, scope: str | None = None) -> bool:
        """Return the operator's answer to an approval, counting the wait as operator time."""
        started = time.monotonic()
        try:
            return self._prompts.approve(prompt, scope=scope, call_id=self._gating_call_id())
        finally:
            self.operator_wait_s += time.monotonic() - started

    def _not_approved(self, name: str) -> tools_errors.ToolDeniedError:
        """Return the refusal for an unapproved command.

        The gate cannot tell a human "no" from an unattended run's auto-deny, so the message
        names the knob; a stop requested while the approval waited is the one cause it can name.

        Args:
            name: What the refusal calls the command, such as `run_command`.

        Returns:
            The error to raise.
        """
        if self._session_dir is not None and ipc.stop_request_pending(self._session_dir):
            return tools_errors.ToolDeniedError(
                f"{name} not run: the run was asked to stop while awaiting approval"
            )
        return tools_errors.ToolDeniedError(f"{name} not approved (sandbox.run_commands='ask')")

    def _run_verify(self, raw: dict[str, Any]) -> results.ExecResult:
        """Return the gate's outcome, run at the model's asking."""
        schema.RunVerifyInput.model_validate(raw)
        return self.run_verify()

    def run_verify(self, extra_argv: tuple[str, ...] = ()) -> results.ExecResult:
        """Run the gate; the model's `run_verify_command` and the harness share this path.

        Args:
            extra_argv: Appended to the configured command (the harness's scoped fallback
                passes the selected test paths).

        Returns:
            The outcome, its `command` naming the argv that ran so the worker can tell which
            gate judged it.

        Raises:
            KeyboardInterrupt: The command was interrupted.
            ToolError: No command is configured or the jail is unavailable.
            ToolDeniedError: The operator did not approve.
            OperatorCommandUnexecutableError: The command cannot run in the jail.
        """
        argv = self._config.harness.verify_command + extra_argv
        timeout_s = self._config.harness.verify_timeout_s
        record = evidence.VerificationEvidence(
            verify_id=uuid.uuid4().hex,
            execution_id=self.execution_id,
            cmd=argv,
            tree_before=self.verify_tree() or None,
            tree_after=None,
            scoped=bool(extra_argv),
            selected_paths=extra_argv,
            outcome="unavailable",
            exit_code=None,
            duration_s=0.0,
            timeout_s=timeout_s,
            stdout_tail="",
            stderr_tail="",
            stdout_truncated=False,
            stderr_truncated=False,
        )
        self._emit(
            "verify.start",
            verify_id=record.verify_id,
            execution_id=record.execution_id,
            tree_before=record.tree_before,
            cmd=list(argv),
            scoped=record.scoped,
            selected_paths=list(extra_argv),
            timeout_s=timeout_s,
        )
        started = time.monotonic()
        try:
            if self.command_policy() == "deny":
                raise tools_errors.ToolDeniedError("not available (run_commands = 'deny')")
            if not self._config.harness.verify_command:
                raise tools_errors.ToolError("no verify command configured")
            self._approve_command("run_verify_command", argv)
            res = _command_tools.run_argv_in_jail(
                argv,
                label="verify_command",
                timeout_s=timeout_s,
                access=self._jail_access,
            )
        except (tools_errors.ToolError, KeyboardInterrupt) as exc:
            outcome = "interrupted" if isinstance(exc, KeyboardInterrupt) else "unavailable"
            if isinstance(exc, tools_errors.ToolDeniedError):
                outcome = "denied"
            record = dataclasses.replace(
                record,
                outcome=outcome,
                tree_after=self.verify_tree() or None,
                duration_s=time.monotonic() - started,
                reason=str(exc)[:2000] + (" (truncated)" if len(str(exc)) > 2000 else ""),
                stdout_truncated=False if outcome == "denied" else None,
                stderr_truncated=False if outcome == "denied" else None,
            )
            self.last_verification = record
            self._emit("verify.end", **dataclasses.asdict(record))
            raise
        outcome = "passed" if res.returncode == 0 else "failed"
        if res.exec_failed:
            outcome = "unavailable"
        elif res.stopped_by_operator or res.returncode is None or res.returncode < 0:
            outcome = "interrupted"
        elif res.returncode == 124:
            outcome = "timed_out"
        record = dataclasses.replace(
            record,
            outcome=outcome,
            tree_after=self.verify_tree() or None,
            exit_code=res.returncode,
            duration_s=res.duration_s,
            stdout_tail=res.stdout[-2000:],
            stderr_tail="" if res.exec_failed else res.stderr[-2000:],
            reason=res.stderr if res.exec_failed else "",
            stdout_truncated=True if len(res.stdout) > 2000 else res.stdout_truncated,
            stderr_truncated=True if len(res.stderr) > 2000 else res.stderr_truncated,
        )
        self.last_verification = record
        res = dataclasses.replace(res, command=argv, verification=record)
        self._emit("verify.end", **dataclasses.asdict(record))
        if res.exec_failed:
            raise tools_errors.OperatorCommandUnexecutableError(
                f"verify_command {list(argv)} could not be executed in the sandbox: "
                f"{res.stderr}. The jail PATH is /usr/bin:/bin plus the standard bin "
                "dirs that exist (/usr/local/bin, /usr/local/sbin, ~/.local/bin, "
                "~/.cargo/bin, /opt/homebrew/bin, /snap/bin), each mounted read-only; "
                "the command is on "
                "none of them. Install the tool into one of those on the host, use a "
                "path inside the workspace (e.g. .venv/bin/pytest), or grant its real "
                "directory via sandbox.extra_read_paths."
            )
        return res

    def _verification_tree(self) -> str:
        """Return a bare dispatcher's content identity; a harness supplies its chain reader."""
        try:
            return git_read.worktree_tree(
                self._root, git_read.status(self._root).head_sha or None, ()
            )
        except (errors.GitError, OSError):
            return ""

    def run_setup(self, argv: tuple[str, ...]) -> kinds.CommandResult:
        """Run an isolated workspace's setup command under the policy a run command gets.

        Args:
            argv: The setup command.

        Returns:
            The unclipped outcome, its `argv` naming the command that ran.

        Raises:
            ToolDeniedError: The operator did not approve.
            ToolError: The jail is unavailable.
        """
        self._approve_command("setup_command", argv)
        res, _timeout = _command_tools.run_argv_raw(
            argv,
            label="setup_command",
            access=self._jail_access,
        )
        self._emit("workspace.setup", **kinds.workspace_setup_fields(res))
        return res

    def _approve_command(self, name: str, argv: tuple[str, ...]) -> None:
        """Gate a command on the run's policy.

        Args:
            name: What the prompt and the refusal call the command, such as `run_command`.
            argv: The command.

        Raises:
            ToolDeniedError: The policy is "ask" and the operator did not approve.
        """
        if self.command_policy() == "ask" and not self._approve(
            f"Allow {name}: {shlex.join(argv)}", scope=ipc.COMMAND_SCOPE
        ):
            raise self._not_approved(name)

    def _run_command(self, raw: dict[str, Any]) -> results.ExecResult:
        """Return the outcome of a command the model chose, or its handle when detached."""
        args = schema.RunCommandInput.model_validate(raw)
        self._approve_command("run_command", args.argv)
        if args.background:
            return _command_tools.start_detached(
                args.argv,
                mode=self._mode,
                get_background=self._background,
                access=self._jail_access,
            )
        return _command_tools.run_model_command(
            args.argv,
            mode=self._mode,
            shells=self._shells,
            checkin=self._config.harness.command_checkin_s,
            interrupted=self._operator_wants_out_now,
            emit=self._emit,
            access=self._jail_access,
        )

    def _operator_wants_out(self) -> bool:
        """Return whether the operator has asked this run to stop or abort.

        Consulted only to cut a wait short; the run still ends at its own boundary.

        Returns:
            True while a stop request or an `abort` steer answer is pending; False when no run
            dir is wired.
        """
        if self._session_dir is None:
            return False
        return ipc.stop_request_pending(self._session_dir) or ipc.steer_answer_is_abort(
            self._session_dir
        )

    def _operator_wants_out_now(self) -> bool:
        """Return whether the operator's stop is immediate: the abort, or a stop marked `now`.

        A stop after the step lets the step's command finish.

        Returns:
            True while an `abort` steer answer or a `now` stop is pending; False when no run
            dir is wired.
        """
        if self._session_dir is None:
            return False
        return ipc.stop_request_is_now(self._session_dir) or ipc.steer_answer_is_abort(
            self._session_dir
        )

    def _read_background(self, raw: dict[str, Any]) -> results.BackgroundResult:
        """Return background output with the run's check-in and stop policy.

        Args:
            raw: The tool call's arguments.

        Returns:
            The roster, with the command's output when the call gives an `id`.
        """
        return _command_tools.read_background(
            raw,
            get_background=self._background,
            checkin=self._config.harness.command_checkin_s,
            interrupted=self._operator_wants_out,
        )

    def _stop_background(self, raw: dict[str, Any]) -> results.BackgroundResult:
        """Return the background roster after stopping a command.

        Args:
            raw: The tool call's arguments, naming the command by its `id`.

        Returns:
            The roster after the stop.
        """
        return _command_tools.stop_background(raw, get_background=self._background)

    def _background(self) -> background.BackgroundShells:
        """Return the background roster.

        Returns:
            The roster.

        Raises:
            ToolError: No run directory was wired.
        """
        if self._shells is None:
            raise tools_errors.ToolError("background commands need a run directory; none was wired")
        return self._shells

    def _fetch(self, raw: dict[str, Any]) -> results.FetchResult:
        """Fetch one URL: an allow-listed host reads, any other asks.

        Args:
            raw: The tool call's arguments.

        Returns:
            The response.

        Raises:
            ToolError: The URL or the response was refused.
            ToolDeniedError: The host is off the list and the operator did not approve.
        """
        args = schema.FetchInput.model_validate(raw)
        try:
            checked = fetch.check_url(args.url)
        except fetch.FetchRefusedError as exc:
            raise tools_errors.ToolError(str(exc)) from exc
        # The list is the standing approval; a GET can carry data out, so an unnamed host is the
        # operator's call. Nothing has resolved yet: the DNS query itself carries the name out.
        if not fetch.host_allowed(
            checked.host, self._config.sandbox.fetch_approved_hosts
        ) and not self._approve(f"Allow fetch: {checked.prompt()}"):
            raise tools_errors.ToolDeniedError(
                f"fetch not approved for {checked.host}"
                " (add it to sandbox.fetch_approved_hosts to allow it)"
            )
        try:
            got = fetch.fetch_url(checked)
        except fetch.FetchRefusedError as exc:
            raise tools_errors.ToolError(str(exc)) from exc
        return results.FetchResult(
            url=got.url,
            status=got.status,
            content_type=got.content_type,
            body=got.body,
            location=got.location,
        )

    def _read_session(self, raw: dict[str, Any]) -> results.SessionsResult:
        """List the project's sessions, and read one's conversation when asked.

        Args:
            raw: The tool call's arguments.

        Returns:
            The roster, with the conversation when an id was given.

        Raises:
            ToolError: No state dir was wired, or the session does not exist.
        """
        args = schema.ReadSessionInput.model_validate(raw)
        if self._state_dir is None:
            raise tools_errors.ToolError("read_session needs the project state dir; none was wired")
        lines = sessions.roster(self._state_dir, args.query).lines()
        if not args.id:
            return results.SessionsResult(sessions=lines)
        layout = sessions_layout.session_layout(self._state_dir, args.id)
        if layout is None:
            raise tools_errors.ToolError(f"no session {args.id!r} in this project")
        return results.SessionsResult(
            sessions=lines, conversation=sessions.conversation(layout, max_chars=args.max_chars)
        )

    def _ask_user(self, raw: dict[str, Any]) -> results.ToolResult:
        """Return the operator's answers to the model's questions, counting the wait as theirs."""
        args = schema.AskUserInput.model_validate(raw)
        started = time.monotonic()
        try:
            answer = self._prompts.ask(args.questions, call_id=self._gating_call_id())
        finally:
            self.operator_wait_s += time.monotonic() - started
        return results.AnswersResult(
            answers=answer.answers,
            note=operator_prompts.unanswered_note(answer),
            asked=tuple(q.question for q in args.questions),
        )

    def resolved_skills(self) -> skills.ResolvedSkills:
        """Return the operator's skills, resolved once per dispatcher.

        The same source as the system prompt's index: `[skills].extra_dirs`, then the installed
        dir under the user data dir.

        Returns:
            The skills; empty when `[skills].enabled` is false.
        """
        if self._skills_cache is None:
            self._skills_cache = skills.operator_skills(
                self._config.skills.enabled,
                self._config.skills.extra_dirs,
                self._config.skills.state,
                paths.data_dir() / "skills",
                hidden=paths.hidden_paths(pathlib.Path(p) for p in self._config.sandbox.hide_paths),
            )
        return self._skills_cache

    def skills_available(self) -> bool:
        """Return whether at least one enabled or always skill exists."""
        resolved = self.resolved_skills()
        return bool(resolved.enabled or resolved.always)

    def _run_metric(self, raw: dict[str, Any]) -> results.MetricResult:
        """Run the configured metric command in the jail and parse its score.

        Args:
            raw: The tool call's arguments; the tool takes none.

        Returns:
            The outcome plus the score: the pattern's first capture group as a float, or None
            when it does not match or parse.

        Raises:
            ToolError: No metric is configured, or the jail is unavailable.
            ToolDeniedError: The operator did not approve.
            OperatorCommandUnexecutableError: The command cannot run in the jail.
        """
        schema.RunMetricInput.model_validate(raw)
        metric_cfg = self._config.harness.metric
        if metric_cfg is None:
            raise tools_errors.ToolError("no [harness.metric] configured")
        argv = metric_cfg.command
        self._approve_command("run_metric_command", argv)
        self._emit("metric.start", cmd=list(argv))
        outcome, timeout_s = _command_tools.run_argv_raw(
            argv,
            label="metric_command",
            timeout_s=self._config.harness.verify_timeout_s,
            access=self._jail_access,
        )
        if outcome.exec_failed:
            raise tools_errors.OperatorCommandUnexecutableError(
                f"metric_command {list(argv)} could not be executed in the sandbox: "
                f"{outcome.stderr}. See run_verify_command's note: PATH is /usr/bin:/bin "
                "plus the standard bin dirs; install the tool into one of those on the "
                "host, use a path inside the workspace, or grant its real directory "
                "via sandbox.extra_read_paths."
            )
        # Scored from the unclipped outcome: the display clip's marker matches a loose pattern.
        score = _result_format.parse_metric_score(
            outcome.stdout, outcome.stderr, pattern=metric_cfg.pattern
        )
        res = _result_format.exec_result(outcome, timeout_s=timeout_s)
        self._emit(
            "metric.end",
            cmd=list(argv),
            exit_code=res.returncode,
            duration_s=res.duration_s,
            stdout_tail=res.stdout[-2000:],
            stderr_tail=res.stderr[-2000:],
            score=score,
        )
        return results.MetricResult.from_exec(res, score)

    def _jail_policy(
        self,
        argv: tuple[str, ...],
        *,
        timeout_s: float | None = None,
        extra_rw_paths: tuple[pathlib.Path, ...] = (),
    ) -> kinds.JailPolicy:
        """Return the jail policy for an argv, with this dispatcher's protect paths and git dir."""
        return tools_policy.jail_policy(
            self._root,
            self._config,
            self.isolation,
            argv,
            timeout_s=timeout_s,
            extra_rw_paths=extra_rw_paths,
            extra_protect_paths=self.extra_protect_paths,
            worktree_git_dir=self._worktree_git_dir,
            grants=self._grants,
        )

    def _net(self) -> jail.SessionNetwork:
        """Return the session network this dispatcher's commands join.

        Returns:
            The run's network when there is a run, shared with its MCP servers; a dispatcher
            built without one makes its own, so its commands still reach each other.
        """
        if self._session_net is not None:
            return self._session_net
        if self._own_session_net is None:
            self._own_session_net = jail.SessionNetwork.open()
        return self._own_session_net

    def _net_for(self, policy: kinds.JailPolicy) -> jail.SessionNetwork | None:
        """Return the network a command under the policy joins: the one owner of that decision.

        Args:
            policy: The command's jail policy.

        Returns:
            The session network for a `session` policy; None for every other network mode,
            which opens nothing.
        """
        return self._net() if policy.network == "session" else None

    def _run_session(self) -> jail_session.JailSession | None:
        """Return the run's jail process, or None to give each command its own.

        Every isolation level uses it, `none` included: the launcher owns output capture and the
        background lifecycle. Its confinement is fixed when it opens, so the policy is the run's,
        not the first command's, and the background log root is granted before any command asks
        for it.

        Returns:
            The session; None when `use_jail_session` is off or once it has failed to start.
        """
        if not self._use_session:
            return None
        with self._session_lock:
            if self._session is None and not self._session_failed:
                rw = () if self._shells is None else (self._shells.log_root,)
                try:
                    policy = self._jail_policy(("true",), extra_rw_paths=rw)
                    net = self._net_for(policy)
                    self._session = jail_session.JailSession.open(policy, session_net=net)
                    if self._session.startup_stderr:
                        # A degraded jail (rootless podman refusing /proc) is said once, here.
                        self._emit("jail.degraded", detail=self._session.startup_stderr)
                except (jail.JailUnavailableError, OSError):
                    self._session_failed = True
            return self._session
