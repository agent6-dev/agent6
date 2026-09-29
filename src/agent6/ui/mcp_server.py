# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Serve agent6 as an MCP server over stdio.

Exposes the workspace's verify command, jail, patch tool and task graph to an
external MCP client, as line-delimited JSON-RPC 2.0. The trust posture is the
agent's own tools': every command routes through the jail by way of a
`ToolDispatcher` built on the loaded config, and `[sandbox].run_commands =
"ask"` is a hard deny, since nobody answers at the MCP boundary.

The tools: `run_verify`, `run_in_sandbox`, `apply_patch_in_sandbox`,
`query_dag`, `list_sessions`.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from agent6 import __version__
from agent6.config import Config
from agent6.config.layer import load_effective
from agent6.graph.storage import load_graph
from agent6.paths import state_dir
from agent6.sessions.id import SessionIdError, resolve_session
from agent6.sessions.layout import (
    SESSION_BUCKETS,
    is_safe_session_id,
)
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.tools.dispatch import ToolDispatcher, ToolError
from agent6.tools.errors import OperatorCommandUnexecutableError
from agent6.viewmodel import session_dirs
from agent6.viewmodel.listing import ListingRow, nested_rows, summarize_session_dir, summary_row

_PROTOCOL_VERSION = "2024-11-05"
_SERVER_NAME = "agent6"
_MAX_LINE_BYTES = 1 << 22  # 4 MiB per JSON-RPC line


class _RpcError(Exception):
    """A JSON-RPC level failure: a bad method or bad params, unlike a tool's isError result."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class _ToolSpec:
    """One published tool.

    Attributes:
        name: The tool's name.
        description: What the client is told.
        input_schema: The published JSON schema.
        handler: Runs the call.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


# The command-spawning tools, withdrawn together when commands are off.
_COMMAND_TOOLS = frozenset({"run_verify", "run_in_sandbox", "apply_patch_in_sandbox"})

# Withdrawn without a verify command: `apply_patch_in_sandbox` must refuse before applying.
_GATE_TOOLS = frozenset({"run_verify", "apply_patch_in_sandbox"})


def _schema_violation(value: Any, schema: dict[str, Any], where: str) -> str | None:
    """Return the first way the value fails the schema, or None.

    A client that ignores the published schema is held to it here, or a wrong-typed
    field reaches the jail. Covers the subset the tool table uses.

    Args:
        value: The value.
        schema: The JSON schema.
        where: The value's path, for the message.

    Returns:
        The violation, or None.
    """
    typ = schema.get("type")
    if typ == "object":
        return _object_violation(value, schema, where)
    if typ == "array":
        return _array_violation(value, schema, where)
    if typ == "string":
        return _string_violation(value, schema, where)
    return None


def _object_violation(value: Any, schema: dict[str, Any], where: str) -> str | None:
    """Return the first way the value fails an object schema, or None."""
    if not isinstance(value, dict):
        return f"{where} must be an object"
    props: dict[str, Any] = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        unknown = sorted(k for k in value if k not in props)
        if unknown:
            return f"{where} has unknown field(s): {', '.join(unknown)}"
    for req in schema.get("required", ()):
        if req not in value:
            return f"{where} is missing required field {req!r}"
    for key, sub in props.items():
        if key in value and (nested := _schema_violation(value[key], sub, f"{where}.{key}")):
            return nested
    return None


def _array_violation(value: Any, schema: dict[str, Any], where: str) -> str | None:
    """Return the first way the value fails an array schema, or None."""
    if not isinstance(value, list):
        return f"{where} must be an array"
    if len(value) < schema.get("minItems", 0):
        return f"{where} must have at least {schema['minItems']} item(s)"
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for i, item in enumerate(value):
            if nested := _schema_violation(item, item_schema, f"{where}[{i}]"):
                return nested
    return None


def _string_violation(value: Any, schema: dict[str, Any], where: str) -> str | None:
    """Return the first way the value fails a string schema, or None."""
    if not isinstance(value, str):
        return f"{where} must be a string"
    if len(value) < schema.get("minLength", 0):
        return f"{where} must be at least {schema['minLength']} character(s)"
    return None


def _no_one_to_ask(config: Config) -> Config:
    """Return the config with commands off when it would prompt for them.

    Nobody answers at the MCP boundary, so `"ask"` cannot be answered: the same
    rule as a detached run with an away mode of "deny".
    """
    if config.sandbox.run_commands != "ask":
        return config
    return config.with_sandbox_overrides(no_commands=True)


def _listable_sessions(agent6_dir: Path) -> list[Path]:
    """Return every session dir, newest first: the sessions the CLI and the web hub list."""
    return session_dirs(agent6_dir, SESSION_BUCKETS)


def _most_recent_session_id(agent6_dir: Path) -> str | None:
    """Return the newest session's id, or None when there is none."""
    candidates = _listable_sessions(agent6_dir)
    return candidates[0].name if candidates else None


class MCPServer:
    """One serving session: a `ToolDispatcher` behind line-delimited JSON-RPC over stdio."""

    def __init__(
        self,
        *,
        root: Path,
        config: Config,
        stdin: IO[bytes],
        stdout: IO[bytes],
    ) -> None:
        self._root = root.resolve()
        self._config = config
        self._agent6_dir = state_dir(self._root)
        self._stdin = stdin
        self._stdout = stdout
        self._dispatcher = ToolDispatcher(root=self._root, config=_no_one_to_ask(config))
        # Absent from tools/list either way; `_call_tool` names the reason to a client anyway.
        self._commands_withdrawn = config.sandbox.run_commands in ("ask", "no")
        self._gate_missing = not config.harness.verify_command
        specs = self._build_tools()
        if self._commands_withdrawn:
            specs = [t for t in specs if t.name not in _COMMAND_TOOLS]
        if self._gate_missing:
            specs = [t for t in specs if t.name not in _GATE_TOOLS]
        self._tools: dict[str, _ToolSpec] = {t.name: t for t in specs}

    def serve(self) -> None:
        """Answer JSON-RPC requests from stdin on stdout until EOF; notifications are ignored."""
        try:
            while True:
                # An unbounded readline buffers the whole line before any size check.
                line = self._stdin.readline(_MAX_LINE_BYTES + 1)
                if not line:
                    return
                if len(line) > _MAX_LINE_BYTES:
                    # Drain the rest of the oversized line in bounded chunks, then drop it.
                    while line and not line.endswith(b"\n"):
                        line = self._stdin.readline(_MAX_LINE_BYTES + 1)
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # JSON-RPC answers an unparseable request with a parse error under a null id.
                    self._reply(None, error={"code": -32700, "message": "parse error"})
                    continue
                if not isinstance(msg, dict):
                    continue
                self._handle(msg)
        finally:
            self._dispatcher.close()

    def _build_tools(self) -> list[_ToolSpec]:
        """Return the published tools, before the withdrawals."""
        return [
            _ToolSpec(
                name="run_verify",
                description=(
                    "Run the workspace's configured verify command inside the agent6"
                    " jail. Returns {command, returncode, stdout, stderr,"
                    " duration_s, exec_failed}."
                ),
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._h_run_verify,
            ),
            _ToolSpec(
                name="run_in_sandbox",
                description=(
                    "Run an arbitrary argv inside the agent6 jail (Landlock + seccomp"
                    " + user namespace). Requires [sandbox].run_commands = 'yes' in"
                    " your config; 'ask' and 'no' are refused at the MCP boundary"
                    " because there is no operator to prompt."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "argv": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    },
                    "required": ["argv"],
                    "additionalProperties": False,
                },
                handler=self._h_run_in_sandbox,
            ),
            _ToolSpec(
                name="apply_patch_in_sandbox",
                description=(
                    "Apply a unified-diff patch to a single file under the workspace"
                    " root, then re-run the verify command. Returns {apply: {...},"
                    " verify: {...}}. The caller is responsible for reverting on"
                    " verify failure; agent6 does not auto-revert."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "patch": {"type": "string", "minLength": 1},
                    },
                    "required": ["path", "patch"],
                    "additionalProperties": False,
                },
                handler=self._h_apply_patch_in_sandbox,
            ),
            _ToolSpec(
                name="query_dag",
                description=(
                    "Load the task graph for a given run id (default: most recent)."
                    " Returns {session_id, nodes: {id: {title, status, parent_id, ...}}}."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "additionalProperties": False,
                },
                handler=self._h_query_dag,
            ),
            _ToolSpec(
                name="list_sessions",
                description=(
                    "Enumerate sessions under the per-repo state dir (most-recent first): the"
                    " listing row every hub shares (status, label, level, reason, mode, task,"
                    " cost) plus the manifest (lineage: parent, branch, worktree, base_sha)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._h_list_sessions,
            ),
        ]

    def _handle(self, msg: dict[str, Any]) -> None:
        """Answer one request; a notification, which has no id, gets nothing."""
        method = msg.get("method")
        req_id = msg.get("id")
        raw_params = msg.get("params")
        params = raw_params if isinstance(raw_params, dict) else {}
        if not isinstance(method, str):
            return
        if req_id is None:
            return
        try:
            result = self._route(method, params)
            self._reply(req_id, result=result)
        except _RpcError as exc:
            self._reply(req_id, error={"code": exc.code, "message": exc.message})

    def _route(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Return the result for one method.

        Raises:
            _RpcError: The method is unknown, or the call is malformed.
        """
        if method == "initialize":
            return {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": _SERVER_NAME, "version": __version__},
            }
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                    }
                    for t in self._tools.values()
                ],
            }
        if method == "tools/call":
            return self._call_tool(params)
        raise _RpcError(-32601, f"unknown method: {method!r}")

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call.

        Returns:
            The tool's result, or an isError result for a tool-level failure.

        Raises:
            _RpcError: The tool is unknown or withdrawn, or the arguments fail its schema.
        """
        name = params.get("name")
        raw_args = params.get("arguments")
        args = raw_args if isinstance(raw_args, dict) else {}
        if not isinstance(name, str) or name not in self._tools:
            if isinstance(name, str) and name in _COMMAND_TOOLS and self._commands_withdrawn:
                mode = self._config.sandbox.run_commands
                detail = (
                    "no operator answers the MCP boundary"
                    if mode == "ask"
                    else "the operator disabled commands"
                )
                raise _RpcError(
                    -32601,
                    f"{name} is withdrawn: [sandbox].run_commands = {mode!r} ({detail})",
                )
            if isinstance(name, str) and name in _GATE_TOOLS and self._gate_missing:
                raise _RpcError(
                    -32601,
                    f"{name} is withdrawn: this workspace has no [harness]"
                    " verify_command, so there is no gate to run",
                )
            raise _RpcError(-32601, f"unknown tool: {name!r}")
        if raw_args is not None and not isinstance(raw_args, dict):
            raise _RpcError(-32602, "arguments must be an object")
        violation = _schema_violation(args, self._tools[name].input_schema, "arguments")
        if violation is not None:
            raise _RpcError(-32602, violation)
        try:
            payload = self._tools[name].handler(args)
        except (ToolError, OperatorCommandUnexecutableError) as exc:
            # An escaping error would end the serve process and break the pipe for every client.
            return {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            }
        return {
            "content": [{"type": "text", "text": json.dumps(payload, separators=(",", ":"))}],
            "structuredContent": payload,
        }

    def _reply(
        self,
        req_id: Any,
        *,
        result: Any = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Write one JSON-RPC response."""
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        self._stdout.write(json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n")
        self._stdout.flush()

    def _h_run_verify(self, _args: dict[str, Any]) -> dict[str, Any]:
        """Return the verify command's result."""
        return self._dispatcher.dispatch("run_verify_command", {}).to_wire()

    # The schema is checked at the call boundary, so a handler reads its arguments as typed.
    def _h_run_in_sandbox(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return the result of running an argv in the jail."""
        return self._dispatcher.dispatch("run_command", {"argv": list(args["argv"])}).to_wire()

    def _h_apply_patch_in_sandbox(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return the results of applying a patch, then running the verify command."""
        apply_result = self._dispatcher.dispatch(
            "apply_patch", {"path": args["path"], "patch": args["patch"]}
        )
        verify_result = self._dispatcher.dispatch("run_verify_command", {})
        return {"apply": apply_result.to_wire(), "verify": verify_result.to_wire()}

    def _h_query_dag(self, args: dict[str, Any]) -> dict[str, Any]:
        """Load a session's task graph.

        Returns:
            The session id and its nodes.

        Raises:
            ToolError: The session id is unsafe or unknown, or there is no session.
        """
        session_id_arg = args.get("session_id")
        if isinstance(session_id_arg, str) and session_id_arg:
            # A traversing or absolute id is refused before it builds a path.
            if not is_safe_session_id(session_id_arg):
                raise ToolError(f"invalid session_id: {session_id_arg!r}")
            session_id = session_id_arg
        else:
            resolved = _most_recent_session_id(self._agent6_dir)
            if resolved is None:
                raise ToolError("no sessions found under the agent6 state dir")
            session_id = resolved
        try:
            layout = resolve_session(self._agent6_dir, session_id)
        except SessionIdError as exc:
            raise ToolError(str(exc)) from exc
        nodes = load_graph(layout)
        return {
            "session_id": session_id,
            "nodes": {nid: node.model_dump(mode="json") for nid, node in nodes.items()},
        }

    def _h_list_sessions(self, _args: dict[str, Any]) -> dict[str, Any]:
        """Return every session: the shared listing row plus its manifest."""
        dirs = {d.name: d for d in _listable_sessions(self._agent6_dir)}

        def entry(row: ListingRow) -> dict[str, Any]:
            # The row every hub shows, plus the manifest for the lineage fields it omits.
            out: dict[str, Any] = summary_row(
                row.summary, lanes=[entry(lane) for lane in row.lanes]
            )
            out["mtime"] = row.mtime
            with contextlib.suppress(ManifestError):
                out["manifest"] = read_manifest(dirs[row.summary.session_id]).model_dump(
                    mode="json"
                )
            return out

        rows = nested_rows(summarize_session_dir(d) for d in dirs.values())
        return {"sessions": [entry(row) for row in rows]}


def run_server(config_path: Path | None) -> int:
    """Serve from the cwd under the effective config until stdin EOF, for `agent6 mcp serve`.

    Returns:
        0 on a clean exit.
    """
    root = Path.cwd()
    cfg = load_effective(root, config_path).config
    server = MCPServer(
        root=root,
        config=cfg,
        stdin=sys.stdin.buffer,
        stdout=sys.stdout.buffer,
    )
    server.serve()
    return 0
