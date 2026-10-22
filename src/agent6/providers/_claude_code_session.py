# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Claude Code session state, continuation checks and pipe I/O."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
import queue
import shutil
import signal
import subprocess
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import agent6
from agent6 import budget as agent6_budget
from agent6.providers import _claude_code_wire, _stream, types

_MAX_LINE_BYTES = 8 * 1024 * 1024
KILL_GRACE_S = 1.0


def end_child(proc: subprocess.Popen[bytes], private_dir: pathlib.Path) -> None:
    """End the child and remove its private directory.

    SIGTERM to its process group (the CLI exits at once; on stdin EOF alone it
    stays up while a tools/call awaits its answer), SIGKILL after the grace. The
    reader threads close stdout and stderr at their EOF.

    Args:
        proc: The child, leader of its own process group.
        private_dir: The child's cwd.
    """
    with contextlib.suppress(OSError, ValueError):
        if proc.stdin is not None:
            proc.stdin.close()
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    shutil.rmtree(private_dir, ignore_errors=True)


@dataclasses.dataclass(slots=True)
class ToolCall:
    """One pending `tools/call`: the control request to answer and its JSON-RPC id."""

    request_id: str
    rpc_id: Any


@dataclasses.dataclass(slots=True, weakref_slot=True)
class Session:
    """One spawned child and everything the continuation rule needs.

    Attributes:
        proc: The child.
        private_dir: The child's cwd, holding the system prompt file.
        system: The system prompt the child was spawned with.
        tools: The tools the child was spawned with.
        lines: The reader thread's queue; None marks stdout's EOF.
        stderr_drain: Fills `stderr_tail`; joined before an exit is reported.
        pushback: A line read past a round's end, returned by the next read.
        reap: Ends the child once, on close or at interpreter exit.
        stdin_lock: Serialises writes to the child.
        stderr_tail: What the child wrote to stderr before dying.
        stdin_log: The lines written since the last transcript record.
        consumed: The caller's history as this session consumed it, one skeleton per message.
        pending: The tool_use ids the last round produced, answered in order by the next call.
        calls: The `tools/call` requests received, by tool_use id.
        refused: The CLI's own error per tool_use id it refused to forward.
        plan: The latest plan reading.
        resolved_model: The model the CLI reported at init.
        session_id: The CLI's session id.
        account_email: The login's email, kept only to scrub it from returned text.
        restart_next: The next call respawns rather than continues.
    """

    proc: subprocess.Popen[bytes]
    private_dir: pathlib.Path
    system: str
    tools: list[types.ToolDefinition]
    lines: queue.Queue[dict[str, Any] | None]
    stderr_drain: threading.Thread
    pushback: list[dict[str, Any] | None] = dataclasses.field(default_factory=list)
    reap: weakref.finalize[[subprocess.Popen[bytes], pathlib.Path], Session] = dataclasses.field(
        init=False
    )
    stdin_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    stderr_tail: list[bytes] = dataclasses.field(default_factory=list)
    stdin_log: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    consumed: tuple[_claude_code_wire.Skeleton, ...] = ()
    pending: tuple[str, ...] = ()
    calls: dict[str, ToolCall] = dataclasses.field(default_factory=dict)
    refused: dict[str, str] = dataclasses.field(default_factory=dict)
    plan: agent6_budget.PlanUsage | None = None
    resolved_model: str = ""
    session_id: str = ""
    account_email: str = ""
    restart_next: bool = False

    def __post_init__(self) -> None:
        """Arm the reaper."""
        self.reap = weakref.finalize(self, end_child, self.proc, self.private_dir)


@dataclasses.dataclass(frozen=True, slots=True)
class Tail:
    """What a continuing call sends.

    Attributes:
        results: The pending tool results by tool_use id.
        texts: The text blocks that follow them (notices, nudges, steer text).
    """

    results: dict[str, str]
    texts: tuple[str, ...]


def continuation(s: Session, messages: Sequence[Mapping[str, Any]]) -> Tail | None:
    """Apply the one rule for keeping the process: the history extends what it produced.

    The consumed prefix keeps its skeleton; after it comes the last round's
    assistant message (or nothing, when the loop popped a quiet one), then user messages
    only: the pending tool results first, text blocks after.

    Args:
        s: The session.
        messages: The new history.

    Returns:
        What to send, or None to restart.
    """
    n = len(s.consumed)
    if _claude_code_wire.history_skeleton(messages[:n]) != s.consumed:
        return None
    tail = list(messages[n:])
    if (
        tail
        and tail[0].get("role") == "assistant"
        and _claude_code_wire.tool_use_ids(tail[0]) == s.pending
    ):
        tail = tail[1:]
    elif s.pending:
        return None
    if not tail or any(m.get("role") != "user" for m in tail):
        return None
    results: dict[str, str] = {}
    texts: list[str] = []
    if s.pending:
        results = _claude_code_wire.tool_results(tail[0])
        if tuple(results) != s.pending:
            return None
        texts.extend(_claude_code_wire.message_texts(tail[0]))
        tail = tail[1:]
    if any(_claude_code_wire.tool_results(m) for m in tail):
        return None
    texts.extend(text for m in tail for text in _claude_code_wire.message_texts(m))
    return Tail(results=results, texts=tuple(texts)) if results or texts else None


def serve_inline(s: Session, msg: dict[str, Any]) -> bool:
    """Answer, from the reader thread, what needs no loop state.

    The MCP handshake, `tools/list`, `ping`, an unknown method, and the CLI's own
    initialize response, whose account block is dropped except the email kept for
    the scrub.

    Args:
        s: The session.
        msg: The line read.

    Returns:
        True when consumed; a `tools/call` or any other line is queued for the call.
    """
    kind = msg.get("type")
    if kind == "control_response":
        response = msg.get("response") or {}
        if response.get("request_id") == "init-1":
            account = (response.get("response") or {}).get("account") or {}
            s.account_email = str(account.get("email") or "")
            s.lines.put({"type": "_agent6_handshake_done"})
        return True
    if kind != "control_request":
        return False
    request = msg.get("request") or {}
    if request.get("subtype") != "mcp_message":
        return False
    rpc = request.get("message") or {}
    method, rpc_id = rpc.get("method"), rpc.get("id")
    if method == "tools/call":
        return False
    rid = str(msg.get("request_id", ""))
    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": (rpc.get("params") or {}).get("protocolVersion", ""),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": _claude_code_wire.MCP_SERVER, "version": agent6.__version__},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {"name": td.name, "description": td.description, "inputSchema": td.input_schema}
                for td in s.tools
            ]
        }
    elif method == "ping" or rpc_id is None:
        result = {}  # a ping, or a notification acknowledged with an empty result
    else:
        write(
            s,
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": rid,
                    "response": {
                        "mcp_response": {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "error": {"code": -32601, "message": "unknown method"},
                        }
                    },
                },
            },
            log=False,
        )
        return True
    write(s, _claude_code_wire.mcp_answer(rid, rpc_id, result), log=False)
    return True


def read_stdout(s: Session) -> None:
    """Run the reader thread, which owns the pipe.

    A close from another thread while a read is in flight would let that read
    land on the next child's descriptor.

    Args:
        s: The session whose `lines` queue this thread fills, ending with None.
    """
    stream = s.proc.stdout
    assert stream is not None
    while True:
        try:
            raw = stream.readline(_MAX_LINE_BYTES + 1)
        except (OSError, ValueError):
            break
        if not raw:
            break
        if len(raw) > _MAX_LINE_BYTES:
            while raw and not raw.endswith(b"\n"):
                raw = stream.readline(_MAX_LINE_BYTES + 1)
            s.lines.put({"type": "_agent6_error", "text": "claude wrote a line over 8 MiB"})
            continue
        try:
            msg = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            s.lines.put(
                {"type": "_agent6_error", "text": f"claude wrote a non-JSON line: {raw[:200]!r}"}
            )
            continue
        if not isinstance(msg, dict):
            continue
        try:
            consumed = serve_inline(s, msg)
        except types.ProviderError as exc:
            s.lines.put({"type": "_agent6_error", "text": str(exc)})
            break
        except Exception as exc:
            # The thread's boundary: a frame of any shape reaches the round as an error.
            s.lines.put(
                {
                    "type": "_agent6_error",
                    "text": f"claude wrote an invalid stream-json message: {exc}",
                }
            )
            break
        if not consumed:
            s.lines.put(msg)
    stream.close()
    s.lines.put(None)


def write(s: Session, obj: dict[str, Any], *, log: bool = True) -> None:
    """Write one stdin line.

    Args:
        s: The session.
        obj: The line's object.
        log: Whether the line joins the transcript's stdin log.

    Raises:
        ProviderError: The child stopped reading.
    """
    data = json.dumps(obj, separators=(",", ":")).encode() + b"\n"
    with s.stdin_lock:
        stdin = s.proc.stdin
        assert stdin is not None
        try:
            stdin.write(data)
            stdin.flush()
        except (OSError, ValueError) as exc:
            raise types.ProviderError(f"claude stopped reading stdin: {exc}") from exc
    if log:
        s.stdin_log.append(obj)


class Watch:
    """The wait loop's per-tick checks: operator stop and steer, and the idle clock.

    Attributes:
        limit: The active idle budget, one of the SSE providers' phase thresholds.
        last_at: The monotonic time of the last meaningful line.
    """

    def __init__(
        self,
        should_abort: Callable[[], bool] | None,
        should_interrupt: Callable[[], bool] | None,
    ) -> None:
        """Start the clock in the prefill phase."""
        self._abort = should_abort
        self._interrupt = should_interrupt
        self.limit = _stream.STREAM_FIRST_DATA_TIMEOUT_S
        self.last_at = time.monotonic()

    def mark(self) -> None:
        """Mark a meaningful line."""
        self.last_at = time.monotonic()

    def tick(self) -> None:
        """Run the per-tick checks.

        Raises:
            ProviderAborted: The operator stopped the run.
            ProviderInterrupted: The operator asked to steer.
            ProviderError: The idle budget ran out.
        """
        if _stream.safe_poll(self._abort):
            raise types.ProviderAborted("run stopped by operator")
        if _stream.safe_poll(self._interrupt):
            raise types.ProviderInterrupted("steer requested mid-turn")
        if time.monotonic() - self.last_at > self.limit:
            raise types.ProviderError(f"claude produced no output for {self.limit:.0f}s")


def note_refusals(s: Session, line: Mapping[str, Any]) -> None:
    """Note the tool calls the CLI refused itself.

    A `user` echo carrying an `is_error` tool_result the CLI wrote means it refused
    that call's input and never sent the `tools/call`; the next round follows.

    Args:
        s: The session; each refusal's reason lands in its `refused` map.
        line: A `user` line the CLI echoed.
    """
    content = (line.get("message") or {}).get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, Mapping) or block.get("type") != "tool_result":
            continue
        if block.get("is_error"):
            s.refused[str(block.get("tool_use_id", ""))] = echo_reason(block.get("content"))


def echo_reason(content: object) -> str:
    """Return the first line of an echoed tool_result, its `<tool_use_error>` tag off."""
    if isinstance(content, list):
        text = "\n".join(str(b.get("text", "")) for b in content if isinstance(b, Mapping))
    else:
        text = str(content or "")
    text = text.strip().removeprefix("<tool_use_error>").removesuffix("</tool_use_error>").strip()
    return text.splitlines()[0][:200] if text else "no reason given"
