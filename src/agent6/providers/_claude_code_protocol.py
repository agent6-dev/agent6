# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Handle Claude Code control messages, tool continuations and watched line reads."""

from __future__ import annotations

import queue
import subprocess
from collections.abc import Mapping
from typing import Any

from agent6 import portable
from agent6.providers import (
    _claude_code_session,
    _claude_code_stream,
    _claude_code_wire,
    _stream,
    types,
)


def audit_init(s: _claude_code_session.Session, line: Mapping[str, Any], *, model: str) -> None:
    """Check every `system/init`: the tool set is exactly what agent6 offered.

    Args:
        s: The session; the line's model and session id are stored on it.
        line: The `system/init` line.
        model: The configured model, used when init reports none.

    Raises:
        ProviderError: The child exposed another tool set, or resolved an API key
            instead of the login (fatal).
    """
    offered = {_claude_code_wire.TOOL_PREFIX + td.name for td in s.tools}
    exposed = {str(t) for t in (line.get("tools") or ())}
    if exposed != offered:
        extra = ", ".join(sorted(exposed - offered)) or "none"
        missing = ", ".join(sorted(offered - exposed)) or "none"
        raise types.ProviderError(
            f"Claude Code exposed a tool set agent6 did not offer (extra: {extra};"
            f" missing: {missing})",
            fatal=True,
        )
    source = line.get("apiKeySource")
    if source != "none":
        raise types.ProviderError(
            f"Claude Code resolved an API key source ({source}) instead of the subscription"
            " login; agent6 passes no ANTHROPIC_* variable, so the key comes from the Claude"
            " config dir. Remove it there, or sign in with `claude auth login`.",
            fatal=True,
        )
    s.resolved_model = str(line.get("model") or model)
    s.session_id = str(line.get("session_id") or "")


def continue_session(
    s: _claude_code_session.Session,
    tail: _claude_code_session.Tail,
    watch: _claude_code_session.Watch,
    *,
    model: str,
) -> None:
    """Send the continuing call: each pending result as its answer, the texts on the last.

    Args:
        s: The live session.
        tail: The results and texts the new history adds.
        watch: The wait loop's checks.
        model: The configured model, used when init reports none.

    Raises:
        ProviderError: A result with its notices is over the CLI's persist threshold
            (fatal), or the child moved on.
    """
    if not tail.results:
        _claude_code_session.write(s, _claude_code_wire.user_line("\n\n".join(tail.texts)))
        return
    # With every call refused the CLI's next round already runs; the texts go in as a user line.
    answered = [i for i in s.pending if i not in s.refused]
    if not answered:
        s.refused.clear()
        s.pending = ()
        if tail.texts:
            _claude_code_session.write(s, _claude_code_wire.user_line("\n\n".join(tail.texts)))
        return
    for tool_use_id in s.pending:
        if tool_use_id in s.refused:
            # The CLI answered it with its own error and the loop recorded the same.
            del s.refused[tool_use_id]
            continue
        call = await_call(s, tool_use_id, watch, model=model)
        content = [{"type": "text", "text": tail.results[tool_use_id]}]
        if tool_use_id == answered[-1]:
            content.extend({"type": "text", "text": text} for text in tail.texts)
        size = sum(len(str(item["text"]).encode()) for item in content)
        if size > _claude_code_wire.CLAUDE_CODE_PERSIST_BYTES:
            raise types.ProviderError(
                f"a {size}-byte tool result, with its turn's notices, is over Claude"
                f" Code's {_claude_code_wire.CLAUDE_CODE_PERSIST_BYTES}-byte "
                "threshold: it would be written"
                " under ~/.claude/projects and reach the model as a 2 KB preview. The"
                " loop's result cap for this provider leaves room for a turn's notices.",
                fatal=True,
            )
        _claude_code_session.write(
            s, _claude_code_wire.mcp_answer(call.request_id, call.rpc_id, {"content": content})
        )
    s.pending = ()


def await_call(
    s: _claude_code_session.Session,
    tool_use_id: str,
    watch: _claude_code_session.Watch,
    *,
    model: str,
) -> _claude_code_session.ToolCall:
    """Return the `tools/call` for a tool_use id, stashed or the next one the CLI sends.

    Args:
        s: The session.
        tool_use_id: The id of one of the last round's pending calls.
        watch: The wait loop's checks.
        model: The configured model, used when init reports none.

    Returns:
        The `tools/call`, taken out of the session's stash.

    Raises:
        ProviderError: The CLI refused the call itself or moved on without it.
    """
    watch.limit = _stream.STREAM_FIRST_DATA_TIMEOUT_S
    while tool_use_id not in s.calls:
        if tool_use_id in s.refused:
            raise types.ProviderError(
                f"claude refused tool call {tool_use_id} itself and moved on"
                f" ({s.refused[tool_use_id]}); agent6's result for it is undeliverable"
            )
        line = next_line(s, watch, model=model)
        if line.get("type") in ("stream_event", "assistant", "result"):
            raise types.ProviderError(
                f"claude moved on while tool call {tool_use_id} was unanswered"
                f" ({line.get('type')} arrived)"
            )
        absorb(s, line, model=model)
    return s.calls.pop(tool_use_id)


def absorb(s: _claude_code_session.Session, line: Mapping[str, Any], *, model: str) -> None:
    """Handle a line outside a round's stream.

    A tools/call is stashed, a stray can_use_tool allowed, any other control
    request refused, a plan reading taken, a refused call noted, and the CLI's
    other echoes and progress lines dropped.

    Args:
        s: The session.
        line: A stdout line of any `type`.
        model: The configured model, used when init reports none.
    """
    kind = line.get("type")
    if kind == "control_request":
        control(s, line)
    elif kind == "user":
        _claude_code_session.note_refusals(s, line)
    elif kind == "rate_limit_event":
        s.plan = (
            _claude_code_wire.plan_usage_from_rate_limit(line.get("rate_limit_info") or {})
            or s.plan
        )
    elif kind == "system" and line.get("subtype") == "init":
        audit_init(s, line, model=model)
    elif kind == "result":
        check_result(s, line)


def control(s: _claude_code_session.Session, line: Mapping[str, Any]) -> None:
    """Handle a control request.

    Args:
        s: The session; a tools/call is stashed on it.
        line: The `control_request` line.

    Raises:
        ProviderError: An MCP request other than a tools/call.
    """
    request = line.get("request") or {}
    rid = str(line.get("request_id", ""))
    subtype = request.get("subtype")
    if subtype == "mcp_message":
        rpc = request.get("message") or {}
        params = rpc.get("params") or {}
        meta = params.get("_meta") or {}
        tool_use_id = str(meta.get("claudecode/toolUseId") or "")
        if rpc.get("method") != "tools/call" or not tool_use_id:
            raise types.ProviderError(
                f"claude sent an unexpected MCP request: {rpc.get('method')!r}"
            )
        s.calls[tool_use_id] = _claude_code_session.ToolCall(rid, rpc.get("id"))
    elif subtype == "can_use_tool":
        # Never sent under the allowed-tools argv; agent6's approval gate already ran.
        _claude_code_session.write(
            s,
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": rid,
                    "response": {"behavior": "allow", "updatedInput": request.get("input")},
                },
            },
        )
    else:
        _claude_code_session.write(
            s,
            {
                "type": "control_response",
                "response": {
                    "subtype": "error",
                    "request_id": rid,
                    "error": "unsupported control request",
                },
            },
        )


def check_result(s: _claude_code_session.Session, line: Mapping[str, Any]) -> None:
    """Raise a `result` line's error, if it carries one.

    Args:
        s: The session, whose login email is scrubbed from the error.
        line: The `result` line.

    Raises:
        ProviderError: The turn failed; a signed-out login is fatal, and an API
            status carried by the line keeps its retry class.
    """
    subtype = str(line.get("subtype") or "")
    if not line.get("is_error") and not subtype.startswith("error"):
        return
    text = _claude_code_stream.scrub(s, str(line.get("result") or subtype))
    if "not logged in" in text.lower():
        raise types.ProviderError(
            "Claude Code is not signed in; run `claude auth login` as this user", fatal=True
        )
    status = line.get("api_error_status")
    status = status if isinstance(status, int) else None
    where = f" (HTTP {status})" if status else ""
    raise types.ProviderError(f"claude result {subtype}{where}: {text[:500]}", status_code=status)


def next_line(
    s: _claude_code_session.Session,
    watch: _claude_code_session.Watch,
    *,
    model: str,
) -> dict[str, Any]:
    """Return the next meaningful line, ticking the watch while waiting.

    Args:
        s: The session to read; a pushed-back line comes first.
        watch: The wait loop's checks; a returned line resets its idle clock.
        model: The configured model, used when init reports none.

    Returns:
        The line; a plan reading is absorbed and a ping skipped, so neither comes
        back.

    Raises:
        ProviderError: The child exited or wrote an unreadable line.
    """
    while True:
        if s.pushback:
            line = s.pushback.pop()
        else:
            try:
                line = s.lines.get(timeout=_stream.STREAM_WATCHDOG_TICK_S)
            except queue.Empty:
                watch.tick()
                continue
        if line is None:
            # stdout closes a moment before the exit status lands.
            try:
                rc: int | None = s.proc.wait(timeout=_claude_code_session.KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                rc = None
            # A tail read before the drain's EOF misses the child's last words.
            s.stderr_drain.join(timeout=_claude_code_session.KILL_GRACE_S)
            tail = _claude_code_stream.scrub(s, portable.stderr_tail(s.stderr_tail)) or "no stderr"
            raise types.ProviderError(f"claude exited {rc}: {tail}")
        kind = line.get("type")
        if kind == "_agent6_error":
            raise types.ProviderError(str(line.get("text")))
        if kind == "rate_limit_event":
            # The CLI repeats a reading while it waits out a window; it is not progress.
            absorb(s, line, model=model)
            watch.tick()
            continue
        event = line.get("event")
        if kind == "stream_event" and isinstance(event, Mapping) and event.get("type") == "ping":
            watch.tick()
            continue
        watch.mark()
        return line
