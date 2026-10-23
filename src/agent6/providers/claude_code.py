# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `claude_code` provider: the worker on the operator's Claude Code login.

One long-lived `claude -p` process per worker execution, driven over stream-json.
agent6 is its only tool source: it serves the sdk-type MCP server `agent6` on the
same pipes and answers each `tools/call` with the result the loop dispatched. A
tool round returns while the CLI is blocked on its first `tools/call`; the next
call answers every pending request in order, with later notice text riding on the
last answer. A call that passes no tools is one process per call.

The process continues only when the new history extends what this session
produced; anything else (resume, fork, a compaction restart, a retried or
interrupted call, a tool-list or system change, a live context near the window)
respawns it on the history rendered as one user message.

Spend is plan-metered: every round records $0 with the plan windows from the CLI's
`rate_limit_event`, and a round without a reading fails closed. The child runs
unjailed as the operator with every Claude Code capability off: argv is operator
config and literals, the system prompt travels in a 0600 file inside a private
empty cwd, prompts and tool results travel on stdin, and the environment is
curated. Of the handshake's account block only the email is kept, in memory, to
scrub it from returned text (docs/security.md).
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent6 import budget as agent6_budget
from agent6 import portable
from agent6.providers import (
    _claude_code_protocol,
    _claude_code_session,
    _claude_code_stream,
    _claude_code_wire,
    _stream,
    types,
)
from agent6.sandbox import jail

# A reading follows message_stop within milliseconds (4 ms measured on CLI 2.1.251), so one
# tick covers it; a process's first round always gets one, inside the grace or not meterable.
_PLAN_READING_DRAIN_S = _stream.STREAM_WATCHDOG_TICK_S
_PLAN_READING_GRACE_S = 3.0
# The reserve the compaction thresholds keep below the window; past it the session restarts.
_CONTEXT_RESERVE_TOKENS = 16_384


def _missing_binary(binary: str) -> str:
    """Return the remedy for a binary not on PATH."""
    return (
        f"Claude Code binary {binary!r} not found on PATH; install Claude Code and sign in"
        " (`claude auth login`), or set [providers.<name>].binary"
    )


def login_status(binary: str, *, timeout_s: float = 20.0) -> str | None:
    """Check the binary's login.

    Only `loggedIn` is read; the body's email and org are never returned or logged.

    Args:
        binary: The Claude Code binary.
        timeout_s: How long the status command may take.

    Returns:
        None when signed in, else the remedy.
    """
    argv = [binary, "auth", "status", "--json"]
    try:
        proc = subprocess.run(  # noqa: PLW1510 - a signed-out login exits 1 with a JSON body
            argv, env=_claude_code_wire.child_env(), capture_output=True, timeout=timeout_s
        )
    except FileNotFoundError:
        return _missing_binary(binary)
    except subprocess.TimeoutExpired:
        return f"`{binary} auth status --json` did not answer within {timeout_s:.0f}s"
    except OSError as exc:
        return f"`{binary} auth status --json` failed: {exc}"
    try:
        data = json.loads(proc.stdout.decode(errors="replace"))
    except ValueError:
        tail = proc.stderr.decode(errors="replace").strip()[-400:]
        return (
            f"`{binary} auth status --json` returned no JSON (exit {proc.returncode}):"
            f" {tail or 'no output'}"
        )
    if isinstance(data, dict) and data.get("loggedIn") is True:
        return None
    return (
        f"Claude Code is not signed in for uid {os.getuid()} (HOME={os.environ.get('HOME', '')}):"
        " run `claude auth login` as that user. agent6 stores no Claude credentials and this"
        " provider uses no API key."
    )


def _int(value: Any) -> int:
    """Return a usage value as an int; 0 when absent or not one."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


@dataclasses.dataclass(frozen=True, slots=True)
class ClaudeCodeProvider:
    """The worker on the operator's Claude Code login.

    Attributes:
        model: The model id.
        binary: The Claude Code binary.
        effort: The reasoning effort passed on argv.
        transcript_sink: Where each round is recorded.
        budget: The run's tracker; None skips metering.
        context_tokens: The model's window; a live context near it restarts the session.
    """

    model: str
    binary: str = "claude"
    effort: str | None = None
    transcript_sink: types.TranscriptRecorder | None = None
    budget: agent6_budget.BudgetTracker | None = None
    context_tokens: int | None = None
    # The live session; a list, since the dataclass is frozen.
    _cell: list[_claude_code_session.Session | None] = dataclasses.field(
        default_factory=lambda: [None]
    )

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[types.ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        text_delta_callback: Callable[[str], None] | None = None,
        thinking_delta_callback: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> types.ProviderResponse:
        """Run one round on the worker's process, or one process for a tool-less call.

        Args:
            system: The system prompt.
            messages: The conversation in Anthropic shape.
            tools: The tools the model may call; none means a one-process side call.
            max_tokens: Ignored; the binary owns sampling.
            temperature: Ignored; the binary owns sampling.
            reasoning_effort: Ignored; effort is passed on argv at spawn.
            text_delta_callback: Receives visible text as it streams.
            thinking_delta_callback: Receives reasoning text as it streams.
            should_abort: Polled each tick; True abandons the turn.
            should_interrupt: Polled each tick; True ends the turn to steer.

        Returns:
            The round's response.

        Raises:
            ProviderError: The run is over budget, the child failed, or the round
                could not be metered.
        """
        del max_tokens, temperature, reasoning_effort
        if self.budget is not None:
            self.budget.check()
        tool_list = list(tools or ())
        watch = _claude_code_session.Watch(should_abort, should_interrupt)
        s = self._cell[0] if tool_list else None
        try:
            if tool_list:
                s = self._worker_session(s, system, tool_list, messages, watch)
            else:
                # The worker's session, blocked on its tools/call, is untouched.
                s = self._spawn(system, [], messages, watch)
            s.consumed = _claude_code_wire.history_skeleton(messages)
            resp = self._read_round(s, watch, text_delta_callback, thinking_delta_callback)
            self._record_transcript(s, system, messages, resp)
        except BaseException:
            if s is not None:
                self._close(s)  # no half-consumed process survives an exceptional exit
            raise
        if not tool_list:
            s.reap()
        return resp

    def close(self) -> None:
        """Kill the worker's child and remove its private directory."""
        if (s := self._cell[0]) is not None:
            self._close(s)

    def _close(self, s: _claude_code_session.Session) -> None:
        """Reap a session and forget it when it is the live one."""
        s.reap()
        if self._cell[0] is s:
            self._cell[0] = None

    # ---- process lifecycle -------------------------------------------------

    def _worker_session(
        self,
        s: _claude_code_session.Session | None,
        system: str,
        tools: list[types.ToolDefinition],
        messages: Sequence[Mapping[str, Any]],
        watch: _claude_code_session.Watch,
    ) -> _claude_code_session.Session:
        """Continue the worker's process when the history extends what it produced, else respawn.

        Args:
            s: The live session, or None.
            system: The system prompt.
            tools: The tools offered.
            messages: The new history.
            watch: The wait loop's checks.

        Returns:
            The session to read the round from.
        """
        if (
            s is not None
            and s.proc.poll() is None
            and not s.restart_next
            and s.system == system
            and s.tools == tools
        ):
            tail = _claude_code_session.continuation(s, messages)
            if tail is not None:
                _claude_code_protocol.continue_session(s, tail, watch, model=self.model)
                return s
        self.close()
        s = self._cell[0] = self._spawn(system, tools, messages, watch)
        return s

    def _spawn(
        self,
        system: str,
        tools: list[types.ToolDefinition],
        messages: Sequence[Mapping[str, Any]],
        watch: _claude_code_session.Watch,
    ) -> _claude_code_session.Session:
        """Spawn one child through its handshake and the history's `system/init`.

        Args:
            system: The system prompt, written to a 0600 file in the private cwd.
            tools: The tools offered.
            messages: The history, rendered as one user message.
            watch: The wait loop's checks.

        Returns:
            The session.

        Raises:
            ProviderError: The binary is missing or cannot start, or the child failed
                on the way; a spawned child is reaped.
        """
        private_dir = pathlib.Path(tempfile.mkdtemp(prefix="agent6-claude-"))
        prompt_file = private_dir / "system_prompt.txt"
        portable.atomic_write(prompt_file, system)  # a new file lands 0600
        try:
            proc = subprocess.Popen(
                _claude_code_wire.claude_argv(self.binary, self.model, self.effort, prompt_file),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=private_dir,
                env=_claude_code_wire.child_env(),
                start_new_session=True,
                preexec_fn=jail.die_with_parent(os.getpid(), sig=signal.SIGKILL),  # noqa: PLW1509
            )
        except FileNotFoundError as exc:
            shutil.rmtree(private_dir, ignore_errors=True)
            raise types.ProviderError(_missing_binary(self.binary), fatal=True) from exc
        except OSError as exc:
            shutil.rmtree(private_dir, ignore_errors=True)
            raise types.ProviderError(f"cannot start {self.binary!r}: {exc}") from exc
        # Unregistered, the child is exactly what the escapee sweep would kill.
        jail.keep_out_of_the_sweep(proc.pid)
        assert proc.stderr is not None
        tail: list[bytes] = []
        drain = threading.Thread(
            target=portable.drain_stderr,
            args=(proc.stderr, tail),
            kwargs={"close": True},
            name="agent6-claude-stderr",
            daemon=True,
        )
        s = _claude_code_session.Session(
            proc=proc,
            private_dir=private_dir,
            system=system,
            tools=tools,
            lines=queue.Queue(),
            stderr_drain=drain,
            stderr_tail=tail,
        )
        threading.Thread(
            target=_claude_code_session.read_stdout,
            args=(s,),
            name="agent6-claude-stdout",
            daemon=True,
        ).start()
        drain.start()
        try:
            _claude_code_session.write(
                s,
                {
                    "type": "control_request",
                    "request_id": "init-1",
                    "request": {"subtype": "initialize", "hooks": None},
                },
                log=False,
            )
            while (
                _claude_code_protocol.next_line(s, watch, model=self.model).get("type")
                != "_agent6_handshake_done"
            ):
                pass
            _claude_code_session.write(
                s, _claude_code_wire.user_line(_claude_code_wire.render_history(messages))
            )
            while True:
                line = _claude_code_protocol.next_line(s, watch, model=self.model)
                if line.get("type") == "system" and line.get("subtype") == "init":
                    _claude_code_protocol.audit_init(s, line, model=self.model)
                    return s
                _claude_code_protocol.absorb(s, line, model=self.model)
        except BaseException:
            s.reap()
            raise

    # ---- one API round --------------------------------------------------------

    def _read_round(
        self,
        s: _claude_code_session.Session,
        watch: _claude_code_session.Watch,
        text_cb: Callable[[str], None] | None,
        thinking_cb: Callable[[str], None] | None,
    ) -> types.ProviderResponse:
        """Read one round to its end and meter it.

        Args:
            s: The session.
            watch: The wait loop's checks.
            text_cb: Receives visible text as it streams.
            thinking_cb: Receives reasoning text as it streams.

        Returns:
            The round's response.
        """
        r = _claude_code_stream.Round()
        plan_before = s.plan
        watch.limit = _stream.STREAM_FIRST_DATA_TIMEOUT_S
        while not r.ended:
            line = _claude_code_protocol.next_line(s, watch, model=self.model)
            kind = line.get("type")
            if kind == "stream_event":
                _claude_code_stream.stream_event(
                    s, r, line.get("event") or {}, watch, text_cb, thinking_cb
                )
            elif kind == "assistant":
                r.blocks.extend(
                    _claude_code_stream.block(s, b)
                    for b in _claude_code_wire.message_blocks(line.get("message") or {})
                )
            elif kind == "result":
                # A turn that ended without a stream: the signed-out synthetic reply.
                _claude_code_protocol.check_result(s, line)
                r.stop_reason = r.stop_reason or str(line.get("stop_reason") or "end_turn")
                r.ended = True
            else:
                _claude_code_protocol.absorb(s, line, model=self.model)
        ids = tuple(str(b.get("id", "")) for b in r.blocks if b.get("type") == "tool_use")
        if not ids:
            self._read_to_result(s, watch)
        self._drain_after_round(s, ids, plan_before)
        s.pending = ids
        return self._finish_round(s, r, ids)

    def _finish_round(
        self, s: _claude_code_session.Session, r: _claude_code_stream.Round, ids: tuple[str, ...]
    ) -> types.ProviderResponse:
        """Meter the round and build its response.

        Args:
            s: The session.
            r: The round as read.
            ids: The round's tool_use ids.

        Returns:
            The response, with the CLI's own refusals under `refused`.
        """
        input_tokens = _int(r.usage.get("input_tokens"))
        cache_read = _int(r.usage.get("cache_read_input_tokens"))
        cache_creation = _int(r.usage.get("cache_creation_input_tokens"))
        output_tokens = _int(r.usage.get("output_tokens"))
        self._record(s, input_tokens, output_tokens, cache_read, cache_creation)
        live_context = input_tokens + cache_read + cache_creation
        if self.context_tokens and live_context > self.context_tokens - _CONTEXT_RESERVE_TOKENS:
            s.restart_next = True  # the next call replays the compacted mirror
        return types.ProviderResponse(
            text="\n\n".join(str(b.get("text", "")) for b in r.blocks if b.get("type") == "text"),
            tool_uses=tuple(
                {"id": b.get("id"), "name": b.get("name"), "input": b.get("input")}
                for b in r.blocks
                if b.get("type") == "tool_use"
            ),
            stop_reason=r.stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            cost_usd=0.0,
            raw={
                "content": r.blocks,
                "model": s.resolved_model,
                "id": r.message_id,
                "usage": r.usage,
            },
            refused={i: s.refused[i] for i in ids if i in s.refused},
        )

    def _read_to_result(
        self, s: _claude_code_session.Session, watch: _claude_code_session.Watch
    ) -> None:
        """Read a prose round's `result` line, so a turn error is raised where it happened.

        Args:
            s: The session.
            watch: The wait loop's checks.

        Raises:
            ProviderError: The turn failed, or another round started first.
        """
        watch.limit = _stream.STREAM_IDLE_TIMEOUT_S
        while True:
            line = _claude_code_protocol.next_line(s, watch, model=self.model)
            if line.get("type") == "result":
                _claude_code_protocol.check_result(s, line)
                return
            if line.get("type") in ("stream_event", "assistant"):
                raise types.ProviderError(
                    "claude started another round before reporting the turn's result"
                )
            _claude_code_protocol.absorb(s, line, model=self.model)

    def _drain_after_round(
        self,
        s: _claude_code_session.Session,
        ids: tuple[str, ...],
        plan_before: agent6_budget.PlanUsage | None,
    ) -> None:
        """Wait one drain window for the round's plan reading and the CLI's verdict on its calls.

        The verdict is the `tools/call` of the first accepted call (the rest wait on
        its answer) or the refusal echoes of every call. A line of the next round is
        pushed back for the next read.

        Args:
            s: The session.
            ids: The round's tool_use ids.
            plan_before: The plan reading before the round, to tell a new one.
        """
        deadline = time.monotonic() + (
            _PLAN_READING_GRACE_S if s.plan is None else _PLAN_READING_DRAIN_S
        )
        while True:
            judged = not ids or any(i in s.calls for i in ids) or all(i in s.refused for i in ids)
            if s.plan is not plan_before and judged:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                line = s.lines.get(timeout=min(remaining, _stream.STREAM_WATCHDOG_TICK_S))
            except queue.Empty:
                continue
            if line is None or line.get("type") in (
                "_agent6_error",
                "stream_event",
                "assistant",
                "result",
            ):
                s.pushback.append(line)
                return
            _claude_code_protocol.absorb(s, line, model=self.model)

    def _record(
        self,
        s: _claude_code_session.Session,
        inp: int,
        out: int,
        cache_read: int,
        cache_creation: int,
    ) -> None:
        """Record the round's spend against the plan window.

        Args:
            s: The session.
            inp: Input tokens.
            out: Output tokens.
            cache_read: Cache-read tokens.
            cache_creation: Cache-write tokens.

        Raises:
            ProviderError: No plan reading (fatal: a retry would bill another round
                into the same absence), or no input usage after the record.
        """
        if self.budget is None:
            return
        if s.plan is None:
            raise types.ProviderError(
                "claude reported no plan window (rate_limit_event) for this round; agent6 meters"
                " this provider by plan window only",
                fatal=True,
            )
        # The round was generated and the plan window moved, whatever the input side reported.
        self.budget.record(
            model=s.resolved_model or self.model,
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            cost_usd=0.0,
            plan_usage=s.plan,
        )
        if inp + cache_read + cache_creation <= 0:
            raise types.ProviderError(
                "claude reported no usage input tokens for this round; budgeted runs require"
                " provider usage accounting"
            )

    def _record_transcript(
        self,
        s: _claude_code_session.Session,
        system: str,
        messages: list[dict[str, Any]],
        resp: types.ProviderResponse,
    ) -> None:
        """Record the round as one transcript entry, then clear the stdin log."""
        if self.transcript_sink is None:
            s.stdin_log.clear()
            return
        self.transcript_sink.record(
            url=f"claude-code://{s.session_id}",
            request_headers={},
            request_body={"system": system, "messages": messages, "stdin": list(s.stdin_log)},
            response_status=200,
            response_body={
                "id": resp.raw.get("id", ""),
                "role": "assistant",
                "model": s.resolved_model,
                "content": resp.raw.get("content", []),
                "stop_reason": resp.stop_reason,
                "usage": resp.raw.get("usage", {}),
            },
        )
        s.stdin_log.clear()
