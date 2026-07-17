# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent6 as an MCP (Model Context Protocol) server.

Exposes the workspace's verify command, jail, patch tool, and DAG
storage to an external MCP client (e.g. VS Code Copilot's hand-off
menu, Claude Desktop). Speaks line-delimited JSON-RPC 2.0 over stdio,
the same framing the embedded client in ``tools/mcp_client.py``
consumes.

Trust posture: identical to the agent's own tools. Every command-
spawning handler routes through ``agent6.sandbox.jail.run_in_jail``
via a ``ToolDispatcher`` constructed against the loaded config, so
the same Landlock + seccomp + namespace policy applies to anything
the MCP client asks us to run. ``run_in_sandbox`` honours the
existing ``[sandbox].run_commands`` gate; ``"ask"`` mode is treated
as a hard deny because the MCP boundary is non-interactive.

Tool surface:
    run_verify              - run the configured verify command in jail.
    run_in_sandbox          - run arbitrary argv in jail (gated).
    apply_patch_in_sandbox  - apply a unified diff + re-run verify.
    query_dag               - load <run-dir>/graph/*.md as nodes.
    list_runs               - enumerate runs (per-repo run-state dir) with manifest summary.
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
from agent6.config.layer import load_effective, resolved_state_dir
from agent6.graph.storage import load_graph
from agent6.runs.layout import RunLayout
from agent6.runs.manifest import ManifestError, read_manifest
from agent6.tools.dispatch import ToolDispatcher, ToolError
from agent6.viewmodel import run_mtime

_PROTOCOL_VERSION = "2024-11-05"
_SERVER_NAME = "agent6"
_MAX_LINE_BYTES = 1 << 22  # 4 MiB; mirrors the client-side cap.


# ---------------------------------------------------------------------------
# JSON-RPC error sentinel.
# ---------------------------------------------------------------------------


class _RpcError(Exception):
    """A JSON-RPC level failure (bad method, bad params). Distinct from
    ``ToolError``, which is surfaced as a tool-level isError result."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Tool spec table.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _deny_approver(_prompt: str) -> bool:
    """Approver used by the MCP-boundary dispatcher. The MCP transport
    has no human at the other end, so any tool that asks for approval
    (``run_commands = "ask"``) is denied cleanly instead of hanging on
    ``input()`` for stdin we already own."""
    return False


def _runs_root(agent6_dir: Path) -> Path:
    return agent6_dir / "runs"


def _run_dirs_newest_first(runs: Path) -> list[Path]:
    """Run dirs sorted newest-first by run activity.

    Run ids are NOT chronologically sortable -- they start with a random
    ``<adjective>-<noun>`` and the embedded ms timestamp rolls over -- so a
    name sort picks the alphabetically-last run, not the latest. Sort by
    logs.jsonl activity instead of directory mtime so a front-end writing
    frontend.pid into an older run does not make it look newest.
    """
    return sorted(
        (d for d in runs.iterdir() if d.is_dir()),
        key=run_mtime,
        reverse=True,
    )


def _most_recent_run_id(agent6_dir: Path) -> str | None:
    runs = _runs_root(agent6_dir)
    if not runs.is_dir():
        return None
    candidates = _run_dirs_newest_first(runs)
    return candidates[0].name if candidates else None


# ---------------------------------------------------------------------------
# Server.
# ---------------------------------------------------------------------------


class MCPServer:
    """One serve() session. Owns a ``ToolDispatcher`` and reads/writes
    line-delimited JSON-RPC over the supplied stdio handles."""

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
        self._agent6_dir = resolved_state_dir(self._root)
        self._stdin = stdin
        self._stdout = stdout
        self._dispatcher = ToolDispatcher(
            root=self._root,
            config=config,
            approver=_deny_approver,
        )
        self._tools: dict[str, _ToolSpec] = {t.name: t for t in self._build_tools()}

    # ---- public entry point -----

    def serve(self) -> None:
        """Read JSON-RPC messages from stdin until EOF. Each request is
        answered on stdout. Notifications (no ``id``) are ignored."""
        try:
            while True:
                line = self._stdin.readline()
                if not line:
                    return
                if len(line) > _MAX_LINE_BYTES:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                self._handle(msg)
        finally:
            self._dispatcher.close()

    # ---- tool catalog -----

    def _build_tools(self) -> list[_ToolSpec]:
        return [
            _ToolSpec(
                name="run_verify",
                description=(
                    "Run the workspace's configured verify command inside the agent6"
                    " jail. Returns {returncode, stdout, stderr, duration_s}."
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
                    " + user namespace). Requires [sandbox].run_commands = 'auto' or"
                    " 'yes' in your config; 'ask' and 'no' modes are refused at the"
                    " MCP boundary because there is no operator to prompt."
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
                    " Returns {run_id, nodes: {id: {title, status, parent_id, ...}}}."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "additionalProperties": False,
                },
                handler=self._h_query_dag,
            ),
            _ToolSpec(
                name="list_runs",
                description=(
                    "Enumerate runs under the per-repo run-state dir (most-recent first) with"
                    " their manifest summary (task, base_sha, models, ...)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._h_list_runs,
            ),
        ]

    # ---- request routing -----

    def _handle(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        req_id = msg.get("id")
        raw_params = msg.get("params")
        params = raw_params if isinstance(raw_params, dict) else {}
        if not isinstance(method, str):
            return
        # Notifications carry no id and expect no response.
        if req_id is None:
            return
        try:
            result = self._route(method, params)
            self._reply(req_id, result=result)
        except _RpcError as exc:
            self._reply(req_id, error={"code": exc.code, "message": exc.message})

    def _route(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
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
        name = params.get("name")
        raw_args = params.get("arguments")
        args = raw_args if isinstance(raw_args, dict) else {}
        if not isinstance(name, str) or name not in self._tools:
            raise _RpcError(-32601, f"unknown tool: {name!r}")
        if raw_args is not None and not isinstance(raw_args, dict):
            raise _RpcError(-32602, "arguments must be an object")
        try:
            payload = self._tools[name].handler(args)
        except ToolError as exc:
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
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        self._stdout.write(json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n")
        self._stdout.flush()

    # ---- tool handlers -----

    def _h_run_verify(self, _args: dict[str, Any]) -> dict[str, Any]:
        return self._dispatcher.dispatch("run_verify_command", {})

    def _h_run_in_sandbox(self, args: dict[str, Any]) -> dict[str, Any]:
        argv = args.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(s, str) for s in argv):
            raise ToolError("argv must be a non-empty list of strings")
        return self._dispatcher.dispatch("run_command", {"argv": list(argv)})

    def _h_apply_patch_in_sandbox(self, args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path")
        patch = args.get("patch")
        if not isinstance(path, str) or not isinstance(patch, str):
            raise ToolError("path and patch must be strings")
        apply_result = self._dispatcher.dispatch("apply_patch", {"path": path, "patch": patch})
        verify_result = self._dispatcher.dispatch("run_verify_command", {})
        return {"apply": apply_result, "verify": verify_result}

    def _h_query_dag(self, args: dict[str, Any]) -> dict[str, Any]:
        run_id_arg = args.get("run_id")
        if isinstance(run_id_arg, str) and run_id_arg:
            run_id = run_id_arg
        else:
            resolved = _most_recent_run_id(self._agent6_dir)
            if resolved is None:
                raise ToolError("no runs found under the agent6 runs dir")
            run_id = resolved
        layout = RunLayout(state_dir=self._agent6_dir, run_id=run_id)
        if not layout.run_dir.is_dir():
            raise ToolError(f"run not found: {run_id}")
        nodes = load_graph(layout)
        return {
            "run_id": run_id,
            "nodes": {nid: node.model_dump(mode="json") for nid, node in nodes.items()},
        }

    def _h_list_runs(self, _args: dict[str, Any]) -> dict[str, Any]:
        runs = _runs_root(self._agent6_dir)
        if not runs.is_dir():
            return {"runs": []}
        entries: list[dict[str, Any]] = []
        for d in _run_dirs_newest_first(runs):
            summary: dict[str, Any] = {"run_id": d.name}
            # A missing/corrupt manifest lists the run without one.
            with contextlib.suppress(ManifestError):
                summary["manifest"] = read_manifest(d).model_dump(mode="json")
            entries.append(summary)
        return {"runs": entries}


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def run_server(config_path: Path | None) -> int:
    """``agent6 mcp serve`` body. Loads the layered effective config
    (global + repo, plus an optional explicit ``config_path``), spawns an
    :class:`MCPServer` against cwd, and serves until stdin EOF. Returns 0
    on clean exit."""
    root = Path.cwd()
    try:
        cfg = load_effective(root, config_path).config
    except Exception as exc:
        print(f"ERROR: failed to load config: {exc}", file=sys.stderr)
        return 2
    server = MCPServer(
        root=root,
        config=cfg,
        stdin=sys.stdin.buffer,
        stdout=sys.stdout.buffer,
    )
    server.serve()
    return 0
