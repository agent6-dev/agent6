# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Minimal MCP (Model Context Protocol) client over stdio or HTTP.

agent6 spawns each configured stdio server as a long-lived subprocess and speaks JSON-RPC 2.0
over its pipes: `initialize`, `notifications/initialized`, `tools/list` and `tools/call`.
Incoming notifications and server-initiated requests are dropped, and no other capability is
advertised.

Threat model: a server is a jailed child by default (its own `[mcp.servers.<name>.sandbox]`
policy; `unconfined = true` opts out) under a curated env that never carries the provider API
keys; `pass_env` adds named vars, and config refuses one naming a provider key. The argv comes
from the operator's config alone. The LLM influences only the arguments of `tools/call`, which
the server validates and agent6 forwards verbatim. A crashing, hanging, malformed or oversized
server never takes the agent down: every call has a timeout and surfaces an `MCPError`, which
the dispatcher turns into a failed tool result.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent6 import __version__
from agent6.child_env import curated_env
from agent6.kinds import JailPolicy
from agent6.portable import drain_stderr, stderr_tail
from agent6.sandbox.jail import (
    JailedProcess,
    JailUnavailableError,
    SessionNetwork,
    spawn_in_jail,
)
from agent6.tools.mcp_http import HttpTransport, MCPHttpError, MCPSessionExpiredError

# Negotiated in `initialize`; whatever the server answers is accepted.
_MCP_PROTOCOL_VERSION = "2024-11-05"

# A longer line is a protocol error and is dropped; 8 MiB covers a tools/list of dozens of tools.
_MAX_LINE_BYTES = 8 * 1024 * 1024

# A description rides in every provider request and a result floods the context, so both are
# bounded at this trust boundary; the result cap matches fetch.MAX_BYTES.
_MAX_INLINE_TEXT_CHARS = 2048  # tool descriptions and error detail
_MAX_RESULT_CHARS = 1 << 20


def _bounded_inline_text(text: str) -> str:
    """Return the text cut at the inline cap, with a marker saying so."""
    if len(text) <= _MAX_INLINE_TEXT_CHARS:
        return text
    return text[:_MAX_INLINE_TEXT_CHARS] + " …[agent6: truncated]"


def _bounded_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return an oversized tools/call result cut to its text content up to the cap, marked."""
    blob = json.dumps(result, ensure_ascii=False, default=str)
    if len(blob) <= _MAX_RESULT_CHARS:
        return result
    content = result.get("content")
    texts = [
        c["text"]
        for c in (content if isinstance(content, list) else ())
        if isinstance(c, dict) and isinstance(c.get("text"), str)
    ]
    note = (
        f"[agent6: this result was {len(blob)} chars serialized; text content kept up"
        f" to {_MAX_RESULT_CHARS} chars, everything else dropped]"
    )
    kept = "\n".join(texts)[:_MAX_RESULT_CHARS]
    return {"content": [{"type": "text", "text": f"{note}\n{kept}".rstrip()}]}


# The prefix plus the server name makes a collision with a built-in or another server impossible.
MCP_TOOL_PREFIX = "mcp__"


def tool_count(n: int) -> str:
    """Return "N tool" or "N tools"."""
    return f"{n} tool{'' if n == 1 else 's'}"


def split_tool_name(qualified_name: str) -> tuple[str, str]:
    """Return the (server, tool) of a `mcp__<server>__<tool>` name.

    The split is on the first double underscore after the prefix, so a tool name containing
    "__" survives; a server name cannot contain one (`mcp_server_name_refusal`).

    Raises:
        MCPError: The name lacks the prefix or the separator.
    """
    if not qualified_name.startswith(MCP_TOOL_PREFIX):
        raise MCPError(f"not an MCP tool name: {qualified_name!r}")
    suffix = qualified_name[len(MCP_TOOL_PREFIX) :]
    try:
        server_name, tool_name = suffix.split("__", 1)
    except ValueError as exc:
        raise MCPError(f"malformed MCP tool name: {qualified_name!r}") from exc
    return server_name, tool_name


# The provider tool-name grammar; a tool outside it is skipped at registration. Matched with
# fullmatch, since `$` also matches before a terminal newline.
_VALID_MCP_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]+")

# The cross-provider bound on the whole qualified name; one over-limit entry breaks the tools array.
_MAX_QUALIFIED_TOOL_NAME_LEN = 64
# A server minting fresh cursors forever would otherwise hold the handshake.
_MAX_TOOL_PAGES = 64


class MCPError(RuntimeError):
    """Anything the MCP client refuses to do or could not complete."""


class MCPTimeoutError(MCPError):
    """A request the server did not answer within its timeout."""


class MCPRestarted(MCPError):  # noqa: N818  # a signal, not an error
    """A request cut short because another caller's timeout replaced the server."""


@dataclass(frozen=True, slots=True)
class MCPToolDescriptor:
    """One tool advertised by one MCP server.

    Attributes:
        server_name: The server's name in config.
        tool_name: The name the server advertised.
        description: The server's description, bounded.
        input_schema: The JSON schema of the arguments.
    """

    server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified_name(self) -> str:
        """The `mcp__<server>__<tool>` name the LLM sees and the dispatcher routes on."""
        return f"{MCP_TOOL_PREFIX}{self.server_name}__{self.tool_name}"


@dataclass(frozen=True, slots=True)
class MCPStartFailure:
    """A configured server that did not start.

    Recorded rather than only logged: under an editor, stderr is a pane nobody watches.
    """

    name: str
    error: str


def _spawn_server(
    command: tuple[str, ...],
    policy: JailPolicy | None,
    pass_env: tuple[str, ...],
    session_net: SessionNetwork | None = None,
) -> JailedProcess:
    """Start one stdio server through the jail, or at the `none` level when opted out.

    Both paths are `spawn_in_jail`, the launcher and policy a jailed command gets; a second
    confinement stack would drift. Stderr is a pipe the caller drains, capped: everything that
    goes wrong before the handshake says so there and nowhere else, and an undrained pipe blocks
    the writer at 64 KB.

    Args:
        command: The server's argv from config.
        policy: The sandbox policy, or None for an unconfined server.
        pass_env: The env vars an unconfined server keeps by name.
        session_net: The run's session network, for a policy that joins it.

    Returns:
        The running process.

    Raises:
        JailUnavailableError: The jail cannot confine the server.
        OSError: The command could not be started.
    """
    if policy is not None:
        return spawn_in_jail(
            policy,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            session_net=session_net,
        )
    # The opt-out is the same spawner's `none` level: its own session, tied to the agent by
    # PDEATHSIG, registered so a sibling's sweep spares it. A curated env keeps provider keys out.
    unconfined = JailPolicy(
        cwd=Path.cwd(),
        argv=command,
        isolation="none",
        env=tuple(sorted(curated_env(passthrough=pass_env, desktop=True).items())),
    )
    return spawn_in_jail(
        unconfined, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def _result_of(
    response: dict[str, Any],
    *,
    name: str,
    method: str,
    redact: Callable[[str], str] | None = None,
) -> Any:
    """Return the `result` of a JSON-RPC response.

    Args:
        response: The message.
        name: The server's name, for the error.
        method: The method called, for the error.
        redact: Strips credential values from the error text.

    Returns:
        The result, or None when the response carries none.

    Raises:
        MCPError: The response carries an `error`.
    """
    if "error" in response:
        err = response["error"]
        detail = err.get("message", "(no message)") if isinstance(err, dict) else err
        text = str(detail)
        if redact is not None:
            text = redact(text)
        text = _bounded_inline_text(text)
        raise MCPError(f"server {name!r} {method} returned error: {text}")
    return response.get("result")


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    """What starting one MCP server needs: the config's shape, at the boundary.

    Attributes:
        name: The server's name in config.
        command: The argv to spawn.
        startup_timeout_s: The handshake's timeout.
        call_timeout_s: Each tool call's timeout.
        pass_env: The env vars the server needs by name; naming each keeps a provider key out.
        http: The transport for a server the operator runs, in place of `command`.
        policy: The sandbox policy, built by the caller from the same `jail_policy` a command
            uses; None for `[mcp.servers.<n>.sandbox].unconfined`.
    """

    name: str
    command: tuple[str, ...]
    startup_timeout_s: float
    call_timeout_s: float
    pass_env: tuple[str, ...] = ()
    http: HttpTransport | None = None
    policy: JailPolicy | None = None


@dataclass
class _MCPServer:
    """One running MCP server: its process, an id counter and a reader thread.

    Attributes:
        name: The server's name in config.
        command: The argv to spawn.
        startup_timeout_s: The handshake's timeout.
        call_timeout_s: Each tool call's timeout.
        pass_env: The env vars the server keeps by name.
        policy: The sandbox policy; None is the operator's explicit `unconfined = true`.
        session_net: The run's session network, for a policy that joins it.
        http: The transport for a server the operator runs; agent6 then owns none of its
            environment, lifetime or confinement.
    """

    name: str
    command: tuple[str, ...]
    startup_timeout_s: float
    call_timeout_s: float
    pass_env: tuple[str, ...] = ()
    policy: JailPolicy | None = None
    session_net: SessionNetwork | None = None
    http: HttpTransport | None = None
    _proc: JailedProcess | None = None
    # The tail of stderr, drained by a thread and read only to explain a failure.
    _errors: list[bytes] = field(default_factory=list)
    # Pids the sweep of each close could not kill, handed to the manager with the last close's.
    _survivors: set[int] = field(default_factory=set)
    _next_id: int = 1
    _id_lock: threading.Lock = field(default_factory=threading.Lock)
    # Concurrent tools/call threads would interleave pipe writes larger than PIPE_BUF.
    _stdin_lock: threading.Lock = field(default_factory=threading.Lock)
    # One slot per in-flight request, registered before the write and filled only while
    # registered, so a late or unsolicited reply is dropped and the map stays bounded.
    _pending: dict[int, dict[str, Any] | None] = field(default_factory=dict)
    _pending_cv: threading.Condition = field(default_factory=threading.Condition)
    _reader: threading.Thread | None = None
    _stderr_reader: threading.Thread | None = None
    _reader_stop: threading.Event = field(default_factory=threading.Event)
    _tools: tuple[MCPToolDescriptor, ...] = ()
    # Bumped under `_restart_lock` by the caller whose timed-out call replaces the process: a
    # request in flight under another caller ends as MCPRestarted the moment it changes.
    _generation: int = 0
    _restart_lock: threading.Lock = field(default_factory=threading.Lock)
    # Releases the keeper thread a restart spawned the process from; set by `close`.
    _keeper_release: threading.Event | None = None

    def _redact_secrets(self, text: str) -> str:
        """Return a diagnostic with the passed credential values stripped.

        A third-party server may echo one through stderr or a protocol error, both of which
        can reach the transcript and the journal.
        """
        names = self.pass_env
        if self.http is not None and self.http.token_env:
            names = (*names, self.http.token_env)
        for name in names:
            value = os.environ.get(name, "")
            if value:
                text = text.replace(value, "<REDACTED>")
        return text

    def _stderr_tail(self, *, settle: bool = False) -> str:
        """Return the redacted tail of stderr, after a short join of its drainer when asked."""
        if settle and self._stderr_reader is not None:
            self._stderr_reader.join(timeout=0.1)
        return self._redact_secrets(stderr_tail(self._errors))

    def start(self) -> None:
        """Spawn the process, or connect over HTTP, and run the handshake.

        Raises:
            MCPError: The server is already started, could not be spawned, or failed the
                handshake; the process is terminated.
        """
        if self._proc is not None:
            raise MCPError(f"server {self.name!r} already started")
        if self.http is not None:
            self._handshake()
            return
        try:
            self._proc = _spawn_server(
                self.command, self.policy, self.pass_env, session_net=self.session_net
            )
        except (OSError, FileNotFoundError, JailUnavailableError) as exc:
            raise MCPError(f"could not spawn MCP server {self.name!r}: {exc}") from exc
        if self._proc.stderr is not None:
            self._stderr_reader = threading.Thread(
                target=drain_stderr,
                args=(self._proc.stderr, self._errors),
                name=f"mcp-stderr[{self.name}]",
                daemon=True,
            )
            self._stderr_reader.start()
        # The reader starts before the first request so the initialize response cannot race it.
        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"mcp-reader[{self.name}]",
            daemon=True,
        )
        self._reader.start()
        self._handshake()

    def _list_tools(self) -> list[Any]:
        """Return every page of `tools/list`, followed through `nextCursor`.

        Raises:
            MCPError: A page has no tools array, an invalid cursor, or the pages exceed the cap.
        """
        tools_raw: list[Any] = []
        cursor = ""
        seen_cursors: set[str] = set()
        while True:
            params = {"cursor": cursor} if cursor else {}
            listed = self._request("tools/list", params, timeout_s=self.startup_timeout_s)
            page = listed.get("tools") if isinstance(listed, dict) else None
            if not isinstance(page, list):
                raise MCPError(f"server {self.name!r} tools/list returned no tools array")
            tools_raw.extend(page)
            next_cursor = listed.get("nextCursor")
            if next_cursor is None:
                return tools_raw
            if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen_cursors:
                raise MCPError(f"server {self.name!r} tools/list returned an invalid nextCursor")
            if len(seen_cursors) + 1 >= _MAX_TOOL_PAGES:
                raise MCPError(
                    f"server {self.name!r} tools/list paged past {_MAX_TOOL_PAGES} pages"
                )
            seen_cursors.add(next_cursor)
            cursor = next_cursor

    def _handshake(self) -> None:
        """Run `initialize` then `tools/list` and register the valid tools, over either transport.

        Raises:
            MCPError: A request failed; the server is closed first.
        """
        try:
            init_result = self._request(
                "initialize",
                {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "agent6", "version": __version__},
                },
                timeout_s=self.startup_timeout_s,
            )
            if not isinstance(init_result, dict):
                raise MCPError(f"server {self.name!r} returned non-dict initialize result")
            self._notify("notifications/initialized", {})
            tools_raw = self._list_tools()
        except MCPError:
            self.close()
            raise
        descs: list[MCPToolDescriptor] = []
        seen: set[str] = set()
        for entry in tools_raw:
            if not isinstance(entry, dict):
                continue
            tname = entry.get("name")
            if not isinstance(tname, str) or not tname:
                continue
            if not _VALID_MCP_TOOL_NAME.fullmatch(tname) or tname in seen:
                # An invalid or duplicate name would break the whole tools array; first wins.
                continue
            desc = entry.get("description")
            schema = entry.get("inputSchema")
            if not isinstance(schema, dict):
                schema = {"type": "object"}
            qualified = MCPToolDescriptor(
                server_name=self.name,
                tool_name=tname,
                description=_bounded_inline_text(str(desc) if desc is not None else ""),
                input_schema=schema,
            )
            if len(qualified.qualified_name) > _MAX_QUALIFIED_TOOL_NAME_LEN:
                continue
            seen.add(tname)
            descs.append(qualified)
        self._tools = tuple(descs)

    @property
    def tools(self) -> tuple[MCPToolDescriptor, ...]:
        """The tools registered at the handshake."""
        return self._tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one advertised tool, restarting a stdio server that times out.

        Args:
            tool_name: The name the server advertised; one outside the registered set is
                refused before any request leaves agent6.
            arguments: The LLM's arguments, forwarded verbatim.

        Returns:
            The tools/call result, bounded.

        Raises:
            MCPError: The tool is not advertised, the server is not running, the call timed
                out, the tool reported an error, or the server was restarted under the call
                twice.
        """
        if tool_name not in {d.tool_name for d in self._tools}:
            raise MCPError(f"server {self.name!r} did not advertise tool {tool_name!r}")
        # Once more when another caller's timeout replaced the server under this call.
        for _ in range(2):
            with self._restart_lock:
                if self._proc is None and self.http is None:
                    raise MCPError(f"server {self.name!r} is not running")
                generation = self._generation
            try:
                return self._call(tool_name, arguments)
            except MCPTimeoutError as exc:
                raise MCPError(f"{exc}; {self._restart(generation)}") from exc
            except MCPRestarted:
                continue
        raise MCPError(f"server {self.name!r} was restarted under tools/call twice; giving up")

    def _restart(self, generation: int) -> str:
        """Replace the process after a timed-out call, once per generation.

        A stdio server still busy with the call it never answered cannot take the next one.

        Returns:
            What happened, for the call's error.
        """
        with self._restart_lock:
            if self._generation != generation:
                return "the server was already restarted by another call"
            self._generation += 1
            self.close()
            self._reader_stop.clear()
            self._errors = []
            # PDEATHSIG ties a child to the thread that forked it, and this caller may be a
            # pool worker about to end: the spawn runs on a keeper thread that outlives it.
            release = threading.Event()
            spawned = threading.Event()
            failure: list[MCPError] = []

            def keep() -> None:
                try:
                    self.start()
                except MCPError as exc:
                    failure.append(exc)
                spawned.set()
                release.wait()

            self._keeper_release = release
            threading.Thread(target=keep, name=f"mcp-keeper[{self.name}]", daemon=True).start()
            spawned.wait()
            if failure:
                return f"restarting it failed ({failure[0]})"
            return "the server was restarted"

    def _call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Send one tools/call.

        Returns:
            The result, bounded.

        Raises:
            MCPError: The result is not a dict, or the tool reported an error.
        """
        result = self._request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout_s=self.call_timeout_s,
        )
        if not isinstance(result, dict):
            raise MCPError(f"server {self.name!r} tools/call returned non-dict result")
        if result.get("isError") is True:
            # The spec reports a tool failure as a successful result with isError=true.
            content = result.get("content")
            text = ""
            if isinstance(content, list):
                text = " ".join(
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and isinstance(c.get("text"), str)
                ).strip()
            detail = _bounded_inline_text(self._redact_secrets(text)) or "(no detail)"
            raise MCPError(f"server {self.name!r} tool {tool_name!r} reported error: {detail}")
        return _bounded_result(result)

    def close(self) -> frozenset[int]:
        """Shut the server down; idempotent, never raises.

        An HTTP server is the operator's and is not stopped; each request already closed its
        connection.

        Returns:
            The pids the escapee sweep of this and every earlier close could not kill.
        """
        self._reader_stop.set()
        if self._keeper_release is not None:
            self._keeper_release.set()
            self._keeper_release = None
        proc = self._proc
        self._proc = None
        if proc is None:
            return frozenset(self._survivors)
        try:
            # The handle takes the process group down and sweeps the setsid escapees.
            self._survivors |= proc.close()
            return frozenset(self._survivors)
        finally:
            # A thread blocked on a request must not hang on a server this teardown killed.
            with self._pending_cv:
                self._pending_cv.notify_all()

    def _allocate_id(self) -> int:
        """Return the next request id."""
        with self._id_lock:
            req_id = self._next_id
            self._next_id += 1
            return req_id

    def _reinitialize(self) -> None:
        """Re-run `initialize` after an HTTP session expiry; the tool list does not change.

        Raises:
            MCPError: The server returned a non-dict result.
        """
        init = self._request(
            "initialize",
            {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agent6", "version": __version__},
            },
            timeout_s=self.startup_timeout_s,
        )
        if not isinstance(init, dict):
            raise MCPError(f"server {self.name!r} returned non-dict re-initialize result")
        self._notify("notifications/initialized", {})

    def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float,
    ) -> Any:
        """Send one request and wait for its response.

        Args:
            method: The JSON-RPC method.
            params: The parameters.
            timeout_s: How long to wait.

        Returns:
            The response's `result`.

        Raises:
            MCPTimeoutError: No response arrived in time.
            MCPRestarted: Another caller replaced the server while this request was in flight.
            MCPError: The server died, sent no or someone else's response, or returned an error.
        """
        req_id = self._allocate_id()
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        if self.http is not None:
            # HTTP pairs request and response itself: no pending slot, no reader thread.
            try:
                response = self.http.send(payload, timeout_s=timeout_s)
            except MCPSessionExpiredError:
                # Retried once; the re-initialize carries no session id, so it cannot loop back.
                self._reinitialize()
                try:
                    response = self.http.send(payload, timeout_s=timeout_s)
                except MCPHttpError as exc:
                    raise self._http_error(exc) from exc
            except MCPHttpError as exc:
                raise self._http_error(exc) from exc
            if response is None:
                raise MCPError(f"server {self.name!r} sent no response to {method}")
            # The stdio reader's checks: a server-initiated request or a gateway can put someone
            # else's message first, and taking it hands the model another call's answer.
            if "method" in response:
                raise MCPError(
                    f"server {self.name!r} answered {method} with its own"
                    f" {response['method']!r} request, not a response"
                )
            if response.get("id") != req_id:
                raise MCPError(
                    f"server {self.name!r} answered {method} with a response to"
                    f" id {response.get('id')!r}, not to {req_id}"
                )
            return _result_of(response, name=self.name, method=method, redact=self._redact_secrets)
        generation = self._generation
        with self._pending_cv:
            self._pending[req_id] = None
        try:
            self._write_line(payload)
            deadline = time.monotonic() + timeout_s
            with self._pending_cv:
                while (response := self._pending[req_id]) is None:
                    if self._generation != generation:
                        raise MCPRestarted(f"server {self.name!r} was restarted under {method}")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        # A server that logs its reason then waits on stdin would read as a timeout.
                        said = self._stderr_tail()
                        detail = f": {said}" if said else ""
                        raise MCPTimeoutError(
                            f"server {self.name!r} timed out after"
                            f" {timeout_s:.1f}s on {method}{detail}"
                        )
                    # A server that crashed mid-call would otherwise cost the full timeout.
                    if self._reader is not None and not self._reader.is_alive():
                        said = self._stderr_tail(settle=True)
                        detail = f": {said}" if said else ""
                        raise MCPError(
                            f"server {self.name!r} died before responding to {method}{detail}"
                        )
                    self._pending_cv.wait(timeout=min(remaining, 0.25))
        finally:
            with self._pending_cv:
                self._pending.pop(req_id, None)
        return _result_of(response, name=self.name, method=method, redact=self._redact_secrets)

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a notification, which has no id and expects no response.

        Raises:
            MCPError: The transport failed.
        """
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        if self.http is not None:
            try:
                self.http.send(payload, timeout_s=self.startup_timeout_s)
            except MCPHttpError as exc:
                raise self._http_error(exc) from exc
            return
        self._write_line(payload)

    def _http_error(self, exc: MCPHttpError) -> MCPError:
        """Return a transport failure as the MCPError every caller handles, its text redacted."""
        return MCPError(self._redact_secrets(str(exc)))

    def _write_line(self, obj: dict[str, Any]) -> None:
        """Write one JSON-RPC message to the server's stdin.

        Raises:
            MCPError: The server is HTTP, gone, or its stdin closed.
        """
        proc = self._proc
        if self.http is not None:
            raise MCPError(f"server {self.name!r} is HTTP; _write_line is the stdio path")
        if proc is None or proc.stdin is None:
            raise MCPError(f"server {self.name!r} is not writable (process gone)")
        line = json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n"
        try:
            with self._stdin_lock:
                proc.stdin.write(line)
                proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            said = self._stderr_tail(settle=True)
            detail = f": {said}" if said else ""
            raise MCPError(f"server {self.name!r} stdin closed: {exc}{detail}") from exc

    def _read_loop(self) -> None:
        """Publish each response line into its pending slot until EOF or stop."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        stream = proc.stdout
        while not self._reader_stop.is_set():
            try:
                # Bounded: an unbounded readline() would buffer a multi-GiB line before any check.
                raw = stream.readline(_MAX_LINE_BYTES + 1)
            except (OSError, ValueError):
                break
            if not raw:
                break  # EOF
            if len(raw) > _MAX_LINE_BYTES:
                # The rest of the line is drained in bounded chunks and the payload dropped.
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
            # A message with an id and a "method" is a server-initiated request whose id can
            # collide with one of ours; it and every notification are ignored.
            if isinstance(req_id, int) and "method" not in msg:
                with self._pending_cv:
                    if req_id in self._pending:
                        self._pending[req_id] = msg
                        self._pending_cv.notify_all()


@dataclass
class MCPManager:
    """The MCP servers of one run, closed by the lifecycle that built it.

    Attributes:
        failures: The configured servers that did not start, in configuration order.
        networks: The network each started server got, by name: the resolved word,
            `unconfined`, or `remote (not jailed)` for a `url` server.
    """

    _servers: dict[str, _MCPServer] = field(default_factory=dict)
    # A failed start's survivors, handed back by the next close.
    _survivors: set[int] = field(default_factory=set)
    failures: tuple[MCPStartFailure, ...] = ()
    networks: dict[str, str] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        configs: Iterable[MCPServerSpec],
        *,
        logger: Callable[[str], None] | None = None,
        session_net: SessionNetwork | None = None,
    ) -> MCPManager:
        """Start every configured server, recording the ones that fail.

        Args:
            configs: One spec per server, so a caller can build one without the config
                validator.
            logger: Takes one line per server started or failed.
            session_net: The run's session network, for a policy that joins it.

        Returns:
            The manager.

        Raises:
            MCPError: Two servers share a name.
        """
        mgr = cls()
        failures: list[MCPStartFailure] = []
        for spec in configs:
            name = spec.name
            if name in mgr._servers:
                raise MCPError(f"duplicate MCP server name {name!r}")
            srv = _MCPServer(
                name=name,
                command=spec.command,
                startup_timeout_s=spec.startup_timeout_s,
                call_timeout_s=spec.call_timeout_s,
                pass_env=spec.pass_env,
                http=spec.http,
                policy=spec.policy,
                session_net=session_net,
            )
            try:
                srv.start()
            except MCPError as exc:
                # Recorded and skipped; the caller turns the record into a journal event.
                failures.append(MCPStartFailure(name=name, error=str(exc)))
                if logger is not None:
                    logger(f"[mcp] failed to start {name!r}: {exc}")
                mgr._survivors |= srv.close()
                continue
            mgr._servers[name] = srv
            mgr.networks[name] = (
                "remote (not jailed)"
                if srv.http is not None
                else "unconfined"
                if srv.policy is None
                else srv.policy.network
            )
            if logger is not None:
                logger(
                    f"[mcp] started {name!r} ({tool_count(len(srv.tools))},"
                    f" network: {mgr.networks[name]})"
                )
        mgr.failures = tuple(failures)
        return mgr

    def descriptors(self) -> tuple[MCPToolDescriptor, ...]:
        """Return every started server's tools."""
        out: list[MCPToolDescriptor] = []
        for srv in self._servers.values():
            out.extend(srv.tools)
        return tuple(out)

    def call(self, qualified_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Route a qualified tool name to its server and call the tool.

        Args:
            qualified_name: The `mcp__<server>__<tool>` name.
            arguments: The LLM's arguments.

        Returns:
            The tools/call result, bounded.

        Raises:
            MCPError: The name is malformed, the server is unknown, or the call failed.
        """
        server_name, tool_name = split_tool_name(qualified_name)
        srv = self._servers.get(server_name)
        if srv is None:
            raise MCPError(f"unknown MCP server: {server_name!r}")
        return srv.call_tool(tool_name, arguments)

    def close(self) -> frozenset[int]:
        """Close every server.

        Returns:
            The pids the sweeps could not kill, a failed start's included, reported once.
        """
        for srv in self._servers.values():
            self._survivors |= srv.close()
        self._servers.clear()
        out = frozenset(self._survivors)
        self._survivors.clear()
        return out
