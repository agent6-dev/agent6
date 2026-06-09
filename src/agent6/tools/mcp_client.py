# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Minimal stdio MCP (Model Context Protocol) client.

agent6 spawns each configured MCP server as a long-lived subprocess and
talks JSON-RPC 2.0 over stdin/stdout. We implement the small subset of
the protocol we actually need:

* ``initialize`` (handshake).
* ``notifications/initialized`` (we send it; we ignore incoming
  notifications).
* ``tools/list`` (discover tools at startup).
* ``tools/call`` (dispatch one tool call).

Anything else the server might send (``logging/*``, ``prompts/*``,
``resources/*``, server-side ``ping``) is silently dropped on the
client side — we do not advertise the corresponding capabilities.

Threat model
============

Each MCP server runs as the *operator's* user, OUTSIDE the agent6 jail,
with whatever environment the operator's shell has. The argv comes
exclusively from your config (``[[mcp.servers]] command = [...]``);
the LLM cannot influence it. This is the same trust model as the
``[notify].on_complete`` hook: operator-controlled argv, full user
authority, no sandboxing.

What the LLM *can* influence is the *arguments* to ``tools/call`` once
a server is connected. The MCP server is responsible for validating
those — agent6 forwards them verbatim. Operators should treat each MCP
server as a tool surface as serious as any agent6 built-in tool.

A misbehaving server (crash, hang, malformed JSON, oversized reply)
must not take the agent down. Each ``call_tool`` is wrapped in a
timeout and a try/except; the manager surfaces a clean ``MCPError`` to
the dispatcher, which converts it to a ``tool.result ok=false`` event.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

# MCP protocol version we speak. The spec is versioned by date string;
# we negotiate this in `initialize` and accept whatever the server says
# back (we don't validate compatibility beyond "we got a result").
_MCP_PROTOCOL_VERSION = "2024-11-05"

# Anything longer than this on a single line is treated as a protocol
# error and the line is dropped. 8 MiB is generous for a tools/list
# response on a server with a few dozen tools.
_MAX_LINE_BYTES = 8 * 1024 * 1024

# Prefix every MCP tool name with this + the server name so collisions
# with built-in tools (and across servers) are structurally impossible.
# Sonnet / GPT-4o / Kimi all accept ``[A-Za-z0-9_]+`` tool names of
# 64-128 chars; double-underscore segmentation keeps the prefix human-
# parseable in transcripts.
MCP_TOOL_PREFIX = "mcp__"


class MCPError(RuntimeError):
    """Anything the MCP client refuses to do or could not complete."""


@dataclass(frozen=True, slots=True)
class MCPToolDescriptor:
    """One tool advertised by one MCP server. ``qualified_name`` is what
    the LLM sees and what the dispatcher routes on."""

    server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified_name(self) -> str:
        return f"{MCP_TOOL_PREFIX}{self.server_name}__{self.tool_name}"


@dataclass
class _MCPServer:
    """One running MCP server. Owns its subprocess + an id counter +
    a stdout-reader thread that publishes responses into ``_pending``
    keyed by request id."""

    name: str
    command: tuple[str, ...]
    startup_timeout_s: float
    call_timeout_s: float
    _proc: subprocess.Popen[bytes] | None = None
    _next_id: int = 1
    _id_lock: threading.Lock = field(default_factory=threading.Lock)
    _pending: dict[int, dict[str, Any]] = field(default_factory=dict)
    _pending_cv: threading.Condition = field(default_factory=threading.Condition)
    _reader: threading.Thread | None = None
    _reader_stop: threading.Event = field(default_factory=threading.Event)
    _tools: tuple[MCPToolDescriptor, ...] = ()

    def start(self) -> None:
        """Spawn the subprocess and pump it through ``initialize`` +
        ``tools/list``. Raises ``MCPError`` if anything in the handshake
        fails, leaving the subprocess terminated."""
        if self._proc is not None:
            raise MCPError(f"server {self.name!r} already started")
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except (OSError, FileNotFoundError) as exc:
            raise MCPError(f"could not spawn MCP server {self.name!r}: {exc}") from exc
        # Start the reader before issuing the first request so the
        # initialize response can't race the reader thread.
        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"mcp-reader[{self.name}]",
            daemon=True,
        )
        self._reader.start()
        try:
            init_result = self._request(
                "initialize",
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "agent6", "version": "0"},
                },
                timeout_s=self.startup_timeout_s,
            )
        except MCPError:
            self.close()
            raise
        if not isinstance(init_result, dict):
            self.close()
            raise MCPError(f"server {self.name!r} returned non-dict initialize result")
        self._notify("notifications/initialized", {})
        try:
            listed = self._request("tools/list", {}, timeout_s=self.startup_timeout_s)
        except MCPError:
            self.close()
            raise
        tools_raw = listed.get("tools") if isinstance(listed, dict) else None
        if not isinstance(tools_raw, list):
            self.close()
            raise MCPError(f"server {self.name!r} tools/list returned no tools array")
        descs: list[MCPToolDescriptor] = []
        for entry in tools_raw:
            if not isinstance(entry, dict):
                continue
            tname = entry.get("name")
            if not isinstance(tname, str) or not tname:
                continue
            desc = entry.get("description")
            schema = entry.get("inputSchema")
            if not isinstance(schema, dict):
                schema = {"type": "object"}
            descs.append(
                MCPToolDescriptor(
                    server_name=self.name,
                    tool_name=tname,
                    description=str(desc) if desc is not None else "",
                    input_schema=schema,
                )
            )
        self._tools = tuple(descs)

    @property
    def tools(self) -> tuple[MCPToolDescriptor, ...]:
        return self._tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None:
            raise MCPError(f"server {self.name!r} is not running")
        result = self._request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout_s=self.call_timeout_s,
        )
        if not isinstance(result, dict):
            raise MCPError(f"server {self.name!r} tools/call returned non-dict result")
        return result

    def close(self) -> None:
        """Best-effort shutdown. Idempotent. Never raises."""
        self._reader_stop.set()
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                with contextlib.suppress(OSError):
                    proc.stdin.close()
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, OSError):
                    proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=1.0)
        finally:
            # Wake any thread blocked on _pending_cv so it can exit
            # cleanly instead of hanging on a server we just killed.
            with self._pending_cv:
                self._pending_cv.notify_all()

    # ----- internals -----

    def _allocate_id(self) -> int:
        with self._id_lock:
            req_id = self._next_id
            self._next_id += 1
            return req_id

    def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float,
    ) -> Any:
        req_id = self._allocate_id()
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        self._write_line(payload)
        deadline = time.monotonic() + timeout_s
        with self._pending_cv:
            while req_id not in self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPError(
                        f"server {self.name!r} timed out after {timeout_s:.1f}s on {method}"
                    )
                # If the reader thread died (server crashed mid-call)
                # we'd otherwise wait the full timeout for nothing.
                if self._reader is not None and not self._reader.is_alive():
                    raise MCPError(f"server {self.name!r} died before responding to {method}")
                self._pending_cv.wait(timeout=min(remaining, 0.25))
            response = self._pending.pop(req_id)
        if "error" in response:
            err = response["error"]
            msg = err.get("message", "(no message)") if isinstance(err, dict) else str(err)
            raise MCPError(f"server {self.name!r} {method} returned error: {msg}")
        return response.get("result")

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        # JSON-RPC notifications have no id and expect no response.
        self._write_line({"jsonrpc": "2.0", "method": method, "params": params})

    def _write_line(self, obj: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise MCPError(f"server {self.name!r} is not writable (process gone)")
        line = json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n"
        try:
            proc.stdin.write(line)
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPError(f"server {self.name!r} stdin closed: {exc}") from exc

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        stream = proc.stdout
        while not self._reader_stop.is_set():
            try:
                # Bound the read: an unbounded readline() would buffer an entire
                # multi-GiB line from a runaway/malicious server into memory
                # BEFORE any size check, OOM'ing the agent. Cap at the limit + 1
                # so we can detect (and drain) an oversized line.
                raw = stream.readline(_MAX_LINE_BYTES + 1)
            except (OSError, ValueError):
                break
            if not raw:
                break  # EOF
            if len(raw) > _MAX_LINE_BYTES:
                # Oversized: drain the rest of this line (up to its newline) in
                # bounded chunks, discarding, then drop the whole payload.
                # Refusing to parse is safer than OOM on a runaway server.
                while raw and not raw.endswith(b"\n"):
                    raw = stream.readline(_MAX_LINE_BYTES + 1)
                continue
            try:
                msg = json.loads(raw.decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(msg, dict):
                continue
            req_id = msg.get("id")
            # We only consume responses (have an id we sent). Server-
            # initiated requests / notifications are silently dropped.
            if isinstance(req_id, int):
                with self._pending_cv:
                    self._pending[req_id] = msg
                    self._pending_cv.notify_all()


@dataclass
class MCPManager:
    """Owns N MCP server subprocesses for one agent6 run. Constructed
    once at the top of ``_cmd_run``, closed in the finally block.

    The ``configs`` arg is an iterable of (name, command, startup_timeout_s,
    call_timeout_s) tuples; we keep this constructor decoupled from the
    ``Config`` types so tests can pass plain tuples without booting
    the whole config validator.
    """

    _servers: dict[str, _MCPServer] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        configs: Iterable[tuple[str, tuple[str, ...], float, float]],
        *,
        logger: Callable[[str], None] | None = None,
    ) -> MCPManager:
        mgr = cls()
        for name, command, startup_s, call_s in configs:
            if name in mgr._servers:
                raise MCPError(f"duplicate MCP server name {name!r}")
            srv = _MCPServer(
                name=name,
                command=command,
                startup_timeout_s=startup_s,
                call_timeout_s=call_s,
            )
            try:
                srv.start()
            except MCPError as exc:
                # One bad server shouldn't take the whole agent down;
                # we log and skip. The agent will simply not see this
                # server's tools.
                if logger is not None:
                    logger(f"[mcp] failed to start {name!r}: {exc}")
                srv.close()
                continue
            mgr._servers[name] = srv
            if logger is not None:
                logger(f"[mcp] started {name!r} ({len(srv.tools)} tools)")
        return mgr

    def descriptors(self) -> tuple[MCPToolDescriptor, ...]:
        out: list[MCPToolDescriptor] = []
        for srv in self._servers.values():
            out.extend(srv.tools)
        return tuple(out)

    def call(self, qualified_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not qualified_name.startswith(MCP_TOOL_PREFIX):
            raise MCPError(f"not an MCP tool name: {qualified_name!r}")
        suffix = qualified_name[len(MCP_TOOL_PREFIX) :]
        # Split on the FIRST double-underscore so tool names that
        # themselves contain "__" survive intact.
        try:
            server_name, tool_name = suffix.split("__", 1)
        except ValueError as exc:
            raise MCPError(f"malformed MCP tool name: {qualified_name!r}") from exc
        srv = self._servers.get(server_name)
        if srv is None:
            raise MCPError(f"unknown MCP server: {server_name!r}")
        return srv.call_tool(tool_name, arguments)

    def close(self) -> None:
        for srv in self._servers.values():
            srv.close()
        self._servers.clear()
