# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The ChatGPT subscription provider, over the Codex Responses backend.

Authorised by the OAuth credential from `agent6 connect` plus the account id
header. The backend is stream-only, so every call runs over SSE. With
`store=false` the model's encrypted reasoning items replay verbatim, so its chain
of thought survives across tool calls. Usage draws on the plan's limits, so
`cost_usd` stays 0 while token counts are metered. The rating endpoints are never
called: a rating would opt the turn into provider-side training.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import httpx2

from agent6 import budget as agent6_budget
from agent6.providers import (
    _openai_messages,
    _openai_recovery,
    _stream,
    _transport,
    chatgpt_oauth,
    types,
)
from agent6.providers import wire as providers_wire

DEFAULT_MAX_TOKENS = 8192

# Each carries the final response object; `response.failed` carries an envelope instead.
_TERMINAL_EVENTS = frozenset({"response.completed", "response.done", "response.incomplete"})

# The plan's window is exhausted; carried as a 429 so the loop backs off rather than failing.
_USAGE_LIMIT_CODES = frozenset({"usage_limit_reached", "usage_not_included", "rate_limit_exceeded"})


def responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Anthropic-shaped messages into Responses `input` items.

    A text run is one message item, a `tool_use` a `function_call`, a `tool_result`
    a `function_call_output` with its content as a string. A `thinking` block
    carrying a `chatgpt_reasoning` item replays it in place, only with its
    following kept item; any other `thinking` block is display-only and dropped.

    Args:
        messages: The conversation in Anthropic shape.

    Returns:
        The input items, in wire order.
    """
    items: list[dict[str, Any]] = []
    # A blank-name or id-less tool_use is dropped, and its paired tool_result with it.
    dropped_ids: set[str] = set()
    for msg in messages:
        role = str(msg.get("role", "user"))
        if role not in ("user", "assistant"):
            role = "user"
        content = msg.get("content", "")
        if isinstance(content, list):
            items.extend(_content_items(role, content, dropped_ids))
        elif content:
            items.append(_message_item(role, str(content)))
    return items


def _content_items(role: str, blocks: list[Any], dropped_ids: set[str]) -> list[dict[str, Any]]:
    """Return one message's content blocks as input items, text runs batched."""
    items: list[dict[str, Any]] = []
    text_run: list[str] = []
    # An orphaned reasoning item violates the wire's pairing rules and 400s the request.
    pending_reasoning: list[dict[str, Any]] = []

    def flush() -> None:
        """Emit the pending text run as one message item."""
        if text_run:
            items.append(_message_item(role, "\n\n".join(text_run)))
            text_run.clear()

    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = str(block.get("text", ""))
            if not text:
                continue
            items.extend(pending_reasoning)
            pending_reasoning.clear()
            text_run.append(text)
        elif btype == "thinking" and role == "assistant":
            item = block.get("chatgpt_reasoning")
            if isinstance(item, dict):
                flush()
                pending_reasoning.append(item)
        elif btype == "tool_use" and role == "assistant":
            call_id = str(block.get("id") or "")
            if not str(block.get("name") or "").strip() or not call_id.strip():
                dropped_ids.add(call_id)
                pending_reasoning.clear()
                continue
            flush()
            items.extend(pending_reasoning)
            pending_reasoning.clear()
            items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": str(block.get("name", "")),
                    "arguments": json.dumps(block.get("input") or {}),
                }
            )
        elif btype == "tool_result":
            if str(block.get("tool_use_id", "")) in dropped_ids:
                continue
            flush()
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(block.get("tool_use_id", "")),
                    "output": _openai_messages.tool_result_text(block.get("content", "")),
                }
            )
    flush()
    return items


def _message_item(role: str, text: str) -> dict[str, Any]:
    """Return one message item of the role's text kind."""
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def tools_to_responses(tools: list[types.ToolDefinition]) -> list[dict[str, Any]]:
    """Return the tools as Responses function tools (flat, not nested)."""
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
            "strict": False,
        }
        for t in tools
    ]


def _tool_use_of(item: dict[str, Any], *, n: int) -> dict[str, Any] | None:
    """Translate a `function_call` item into a tool_use.

    Args:
        item: The output item.
        n: The call's ordinal, for a synthesised id.

    Returns:
        The tool_use, or None for a blank name, which never enters history.
        Unparseable arguments keep the lenient repair or the `_raw_arguments`
        sentinel so dispatch can ask for a resend.
    """
    name = str(item.get("name", "")).strip()
    if not name:
        return None
    args_raw = item.get("arguments", "")
    try:
        parsed = json.loads(args_raw) if args_raw else {}
        if not isinstance(parsed, dict):
            parsed = {"_value": parsed}
    except (json.JSONDecodeError, TypeError):
        repaired = _openai_recovery.lenient_json_object(str(args_raw))
        parsed = repaired if repaired is not None else {"_raw_arguments": str(args_raw)[:500]}
    return {
        "id": str(item.get("call_id") or item.get("id") or f"call_{n}"),
        "name": name,
        "input": parsed,
    }


def _usage_count(value: Any, field_name: str) -> int:
    """Return one usage count; 0 when absent.

    Raises:
        ProviderError: The value is not a non-negative integer.
    """  # noqa: DOC501  # the TypeError is raised and caught in the same try
    if value is None:
        return 0
    try:
        if isinstance(value, bool):
            raise TypeError
        count = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise types.ProviderError(
            f"ChatGPT response usage.{field_name} was not a non-negative integer"
        ) from exc
    if count < 0 or (isinstance(value, float) and not value.is_integer()):
        raise types.ProviderError(
            f"ChatGPT response usage.{field_name} was not a non-negative integer"
        )
    return count


def parse_output_items(
    items: list[Any], *, usage: Mapping[str, Any], stop_reason: str
) -> types.ProviderResponse:
    """Parse the final output items into a response.

    `raw["content"]` holds one block per item in wire order: a message's text, a
    reasoning item as a `thinking` block (its summary for display, the raw item
    under `chatgpt_reasoning` for replay), a function_call as `tool_use`.

    Args:
        items: The output items.
        usage: The terminal usage object; `input_tokens` is the cached plus fresh
            total, normalised to fresh-only as the OpenAI parser does.
        stop_reason: The stop reason read off the terminal event.

    Returns:
        The response in agent6's canonical shape.
    """
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            text = "".join(
                str(part.get("text", ""))
                for part in item.get("content") or []
                if isinstance(part, dict) and part.get("type") in ("output_text", "text")
            )
            if text:
                text_parts.append(text)
                blocks.append({"type": "text", "text": text})
        elif itype == "reasoning":
            summary = "\n\n".join(
                str(part["text"])
                for part in item.get("summary") or []
                if isinstance(part, dict) and str(part.get("text", ""))
            )
            blocks.append({"type": "thinking", "thinking": summary, "chatgpt_reasoning": item})
        elif itype == "function_call":
            tool_use = _tool_use_of(item, n=len(tool_uses))
            if tool_use is not None:
                tool_uses.append(tool_use)
                blocks.append({"type": "tool_use", **tool_use})
    text = "\n\n".join(text_parts)
    if tool_uses and stop_reason == "end_turn":
        # A turn that stopped to call tools says so, which arms the loop's empty-call detector.
        stop_reason = "tool_use"
    details = usage.get("input_tokens_details")
    cached = (
        _usage_count(details.get("cached_tokens"), "input_tokens_details.cached_tokens")
        if isinstance(details, Mapping)
        else 0
    )
    prompt_total = _usage_count(usage.get("input_tokens"), "input_tokens")
    cached = min(cached, prompt_total)
    return types.ProviderResponse(
        text=text,
        tool_uses=tuple(tool_uses),
        stop_reason=stop_reason,
        input_tokens=prompt_total - cached,
        output_tokens=_usage_count(usage.get("output_tokens"), "output_tokens"),
        cache_read_tokens=cached,
        cache_creation_tokens=0,
        cost_usd=0.0,
        raw={"content": blocks, "usage": dict(usage), "output": items},
    )


_EFFORT_UNSUPPORTED = re.compile(r"Supported values are:\s*((?:'[a-z]+'(?:,\s*(?:and\s*)?)?)+)")


def _accepted_efforts(status: int | None, text: str) -> tuple[str, ...]:
    """Return the effort levels a 400 lists for `reasoning.effort`, in the order given."""
    if status != 400 or '"reasoning.effort"' not in text or "unsupported_value" not in text:
        return ()
    m = _EFFORT_UNSUPPORTED.search(text)
    return tuple(re.findall(r"'([a-z]+)'", m.group(1))) if m else ()


def _unreachable_hook(data: dict[str, Any]) -> Any:
    """Refuse the non-streaming hooks; this wire is stream-only.

    Raises:
        ProviderError: Always.
    """
    raise types.ProviderError("chatgpt provider is stream-only")  # pragma: no cover


_WINDOW_HEADER = re.compile(r"^x-codex-(?P<name>.+)-used-percent$")


def _num(value: Any) -> float:
    """Return a header or body value as a finite float; 0.0 when it is not one."""
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _window_order(name: str) -> tuple[int, str]:
    """Return a window's sort key: primary, secondary, then every other family by name."""
    return ({"primary": 0, "secondary": 1}.get(name, 2), name)


def _plan_usage_of(headers: Mapping[str, str]) -> agent6_budget.PlanUsage | None:
    """Read the plan windows off a response's `x-codex-*` headers.

    Every `x-codex-<name>-used-percent` family is one window, with its window
    minutes and reset siblings; every plan-usage surface reads this one parse.

    Args:
        headers: The response headers.

    Returns:
        The plan usage, or None when the primary reading is absent or malformed.
    """
    lowered = {k.lower(): v for k, v in headers.items()}
    windows: list[agent6_budget.PlanWindow] = []
    for key in sorted(lowered):
        m = _WINDOW_HEADER.match(key)
        if m is None:
            continue
        name = m.group("name")
        try:
            used = float(lowered[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(used) or used < 0:
            continue
        resets_at = _num(lowered.get(f"x-codex-{name}-reset-at"))
        if not resets_at:
            resets_at = time.time() + _num(lowered.get(f"x-codex-{name}-reset-after-seconds"))
        windows.append(
            agent6_budget.PlanWindow(
                name=name,
                used_percent=used,
                window_minutes=int(_num(lowered.get(f"x-codex-{name}-window-minutes"))),
                resets_at=resets_at,
            )
        )
    if not any(w.name == "primary" for w in windows):
        return None

    def _flag(name: str) -> bool:
        """Return whether a header reads true."""
        return (lowered.get(name) or "").strip().lower() == "true"

    return agent6_budget.PlanUsage(
        windows=tuple(sorted(windows, key=lambda w: _window_order(w.name))),
        has_credits=_flag("x-codex-credits-has-credits"),
        credits_unlimited=_flag("x-codex-credits-unlimited"),
        credits_balance=(lowered.get("x-codex-credits-balance") or "").strip(),
    )


def plan_usage_from_usage_body(body: Mapping[str, Any]) -> agent6_budget.PlanUsage | None:
    """Read the account's plan state off the backend's `/usage` body.

    Args:
        body: The response body.

    Returns:
        Every `<name>_window` under `rate_limit` as a window, the backend's
        limit-reached verdict and the credit family; None without a primary window.
    """
    limits = body.get("rate_limit")
    if not isinstance(limits, Mapping):
        return None
    windows: list[agent6_budget.PlanWindow] = []
    for key, raw in limits.items():
        if not (isinstance(key, str) and key.endswith("_window") and isinstance(raw, Mapping)):
            continue
        name = key.removesuffix("_window")
        try:
            used = float(raw.get("used_percent", ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(used) or used < 0:
            continue
        resets_at = _num(raw.get("reset_at"))
        if not resets_at:
            resets_at = time.time() + _num(raw.get("reset_after_seconds"))
        windows.append(
            agent6_budget.PlanWindow(
                name=name,
                used_percent=used,
                window_minutes=int(_num(raw.get("limit_window_seconds")) / 60),
                resets_at=resets_at,
            )
        )
    if not any(w.name == "primary" for w in windows):
        return None
    credits = body.get("credits")
    credits = credits if isinstance(credits, Mapping) else {}
    return agent6_budget.PlanUsage(
        windows=tuple(sorted(windows, key=lambda w: _window_order(w.name))),
        has_credits=credits.get("has_credits") is True,
        credits_unlimited=credits.get("unlimited") is True,
        credits_balance=str(credits.get("balance") or "").strip(),
        limit_reached=limits.get("limit_reached") is True,
    )


def _stream_error(evt: dict[str, Any]) -> types.ProviderError:
    """Return the error a `response.failed` or `error` frame carries, classified."""
    response = evt.get("response")
    err = response.get("error") if isinstance(response, dict) else evt.get("error")
    if not isinstance(err, dict):
        err = {"message": str(err or evt.get("message") or "unknown stream error")}
    code = str(err.get("code") or "")
    status = 429 if code in _USAGE_LIMIT_CODES else None
    detail = str(err.get("message") or code or "unknown error")
    if code in _USAGE_LIMIT_CODES:
        plan = str(err.get("plan_type") or "")
        detail += f" (ChatGPT {plan} plan usage limit)" if plan else " (ChatGPT usage limit)"
    return types.ProviderError(
        f"ChatGPT stream error: {code or 'error'}: {detail}", status_code=status
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ChatGPTProvider:
    """The ChatGPT provider, constructed once per run.

    Attributes:
        model: The model id.
        credential: The OAuth credential; a 401 refreshes it once.
        account_id: The `chatgpt-account-id` header.
        base_url: The backend's base URL.
        extra_headers: Operator headers merged over the built ones.
        extra_body: Operator body keys merged last; the structural keys are reserved.
        extra_query: Operator query parameters.
        timeout_s: The read budget in seconds.
        transcript_sink: Where each round-trip is recorded.
        budget: The run's tracker; None skips metering and the preflight.
        reasoning_effort: The role's effort; unset takes the model's default, and
            "off" sends the wire's explicit "none".
        session_id: The `prompt_cache_key` and `session-id` header, so caching
            keys to this run's conversation.
    """

    model: str
    credential: chatgpt_oauth.ChatGPTCredential
    account_id: str
    base_url: str
    extra_headers: tuple[tuple[str, str], ...] = ()
    extra_body: dict[str, Any] = dataclasses.field(default_factory=dict)
    extra_query: dict[str, str] = dataclasses.field(default_factory=dict)
    timeout_s: float = 600.0
    transcript_sink: types.TranscriptRecorder | None = None
    budget: agent6_budget.BudgetTracker | None = None
    reasoning_effort: str | None = None
    session_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    # One usage preflight per provider; a list, since the dataclass is frozen.
    _preflighted: list[bool] = dataclasses.field(default_factory=lambda: [False])
    # The effort levels the served model accepts, learned from its 400; empty until then.
    _efforts_accepted: list[tuple[str, ...]] = dataclasses.field(default_factory=lambda: [()])

    def preflight(self) -> agent6_budget.PlanUsage | None:
        """Read the account's plan state off the backend's `/usage` before any call.

        Best effort: any failure reads as no reading, never as a block. The body
        carries the account's email, so it is parsed and dropped, never recorded.

        Returns:
            The plan usage, or None when it could not be read.
        """
        try:
            token = self.credential.token()
            resp = httpx2.get(
                f"{self.base_url.rstrip('/')}/usage",
                headers=self._build_headers(token),
                timeout=20.0,
            )
            if resp.status_code != 200:
                return None
            body = resp.json()
        except (types.ProviderError, httpx2.HTTPError, ValueError, OSError):
            return None
        return plan_usage_from_usage_body(body) if isinstance(body, dict) else None

    def _adapt_effort_400(self, status: int | None, text: str, body: dict[str, Any]) -> bool:
        """Resend at the lowest effort the served model accepts, and keep that floor.

        The plan serves whichever model it serves; one refuses `none`, the next may not. The
        400 names the accepted levels, lowest first.

        Args:
            status: The HTTP status.
            text: The error text.
            body: The request body, rewritten in place.

        Returns:
            Whether the body was adapted, so the transport retries once.
        """
        accepted = _accepted_efforts(status, text)
        if not accepted:
            return False
        reasoning = body.get("reasoning")
        if not isinstance(reasoning, dict) or reasoning.get("effort") in accepted:
            return False
        self._efforts_accepted[0] = accepted
        reasoning["effort"] = accepted[0]
        return True

    def _build_headers(self, token: str) -> dict[str, str]:
        """Return one attempt's request headers, built from its token."""
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
            "chatgpt-account-id": self.account_id,
            "openai-beta": "responses=experimental",
            "originator": "agent6",
            "session-id": self.session_id,
        }
        for k, v in self.extra_headers:
            headers[k.lower()] = v
        return headers

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[types.ToolDefinition] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        extended_thinking: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        text_delta_callback: Callable[[str], None] | None = None,
        thinking_delta_callback: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> types.ProviderResponse:
        """Make one Responses call over SSE.

        Args:
            system: The system prompt, sent as `instructions`.
            messages: The conversation in Anthropic shape.
            tools: The tools the model may call.
            max_tokens: Ignored; the backend sizes its output.
            temperature: Ignored; the backend rejects one, and effort is the only knob.
            extended_thinking: Ignored; this wire has no equivalent.
            reasoning_effort: A per-call effort over the provider's own.
            text_delta_callback: Receives visible text as it streams.
            thinking_delta_callback: Receives reasoning summaries as they stream.
            should_abort: Polled during the stream; True abandons the turn.
            should_interrupt: Polled during the stream; True ends the turn to steer.

        Returns:
            The parsed response.

        Raises:
            ProviderError: The run is over budget, or the call failed.
        """
        del max_tokens, temperature, extended_thinking
        if self.budget is not None:
            if not self._preflighted[0]:
                self._preflighted[0] = True
                plan = self.preflight()
                if plan is not None:
                    self.budget.record_plan_preflight(self.model, plan)
            self.budget.check()
        url, _ = providers_wire.request_url(
            api_format="chatgpt",
            deployment="direct",
            base_url=self.base_url,
            model=self.model,
            streaming=True,
            extra_query=self.extra_query,
        )
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": responses_input(messages),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "store": False,
            "stream": True,
            # The encrypted reasoning items replayed for continuity across tool calls.
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": self.session_id,
        }
        if tools:
            body["tools"] = tools_to_responses(tools)
        effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        if effort:
            # Omitting the field leaves the model's default on; "off" needs the explicit "none".
            wire = "none" if effort == "off" else effort
            accepted = self._efforts_accepted[0]
            if accepted and wire not in accepted:
                wire = accepted[0]  # the served model refused the level once; its floor stands in
            body["reasoning"] = {"effort": wire, "summary": "auto"}
        if self.extra_body:
            reserved = {
                "model",
                "instructions",
                "input",
                "tools",
                "tool_choice",
                "store",
                "stream",
                "include",
                "prompt_cache_key",
            }
            body.update({k: v for k, v in self.extra_body.items() if k not in reserved})

        return _transport.ProviderCall(
            api_label="ChatGPT",
            api_format="chatgpt",
            url=url,
            body=body,
            timeout_s=self.timeout_s,
            api_key="",
            credential=self.credential,
            transcript_sink=self.transcript_sink,
            budget=self.budget,
            model=self.model,
            build_headers=self._build_headers,
            adapt_400=self._adapt_effort_400,
            adapt_attempts=1,
            # Stream-only, so the non-streaming hooks are unreachable; metering happens in-stream.
            require_metered=_unreachable_hook,
            parse=_unreachable_hook,
            stream=lambda attempt_headers: self._call_streaming(
                url=url,
                headers=attempt_headers,
                body=body,
                text_delta_callback=text_delta_callback,
                thinking_delta_callback=thinking_delta_callback,
                should_abort=should_abort,
                should_interrupt=should_interrupt,
            ),
        ).run()

    def _call_streaming(  # noqa: C901, PLR0915  # one streaming state machine; a split hides the event order
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        text_delta_callback: Callable[[str], None] | None,
        thinking_delta_callback: Callable[[str], None] | None,
        should_abort: Callable[[], bool] | None,
        should_interrupt: Callable[[], bool] | None,
    ) -> types.ProviderResponse:
        """Make the call over SSE; this method owns the Responses event shape.

        Each frame is a JSON object whose `type` names the event. Deltas feed the
        callbacks only; the content comes from the `output_item.done` items,
        reconciled against the terminal response object, where usage lives.

        Args:
            url: The URL dialled.
            headers: The attempt's request headers.
            body: The request body.
            text_delta_callback: Receives visible text as it streams.
            thinking_delta_callback: Receives reasoning summaries as they stream.
            should_abort: Polled each watchdog tick; True abandons the turn.
            should_interrupt: Polled each watchdog tick; True ends the turn to steer.

        Returns:
            The parsed response.

        Raises:
            ProviderError: A failed response, a stream cut before its terminal
                event, or missing usage on a budgeted run; what the cut turn
                already cost is recorded first.
        """
        stream_headers = dict(headers)
        stream_headers["accept"] = "text/event-stream"

        items: list[Any] = []
        delta_text: list[str] = []
        usage: dict[str, Any] = {}
        plan_usage: agent6_budget.PlanUsage | None = None
        stop_reason = ""
        done = False

        def observe_headers(response_headers: Mapping[str, str]) -> None:
            """Read the plan windows off the response headers."""
            nonlocal plan_usage
            plan_usage = _plan_usage_of(response_headers)

        call = _stream.SseCall(
            api_label="ChatGPT",
            api_format="chatgpt",
            url=url,
            headers=stream_headers,
            body=body,
            timeout_s=self.timeout_s,
            transcript_sink=self.transcript_sink,
            should_abort=should_abort,
            should_interrupt=should_interrupt,
            response_headers=observe_headers,
        )

        def consume(  # noqa: PLR0912, PLR0915
            resp: httpx2.Response, clock: _stream.StreamClock
        ) -> None:
            """Read the stream's events into the accumulators.

            Raises:
                ProviderError: A failed response or an unknown terminal status.
            """  # noqa: DOC501  # `_stream_error` builds the ProviderError named above
            nonlocal usage, stop_reason, done
            for _event, data in _stream.sse_events(resp):
                clock.mark_data()
                data_str = data.strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    evt: dict[str, Any] = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                kind = str(evt.get("type", ""))
                if kind == "response.output_text.delta":
                    piece = str(evt.get("delta", ""))
                    if piece:
                        clock.mark_output()
                        delta_text.append(piece)
                        if text_delta_callback is not None:
                            with contextlib.suppress(Exception):
                                text_delta_callback(piece)
                elif kind in (
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                ):
                    piece = str(evt.get("delta", ""))
                    if piece:
                        clock.mark_output()
                        if thinking_delta_callback is not None:
                            with contextlib.suppress(Exception):
                                thinking_delta_callback(piece)
                elif kind == "response.output_item.done":
                    clock.mark_output()
                    items.append(evt.get("item"))
                elif kind in ("response.failed", "error"):
                    failed = evt.get("response")
                    failed_usage = failed.get("usage") if isinstance(failed, dict) else None
                    if isinstance(failed_usage, dict):
                        usage = failed_usage
                    call.record(status=0, response=data_str[:8192])
                    raise _stream_error(evt)
                elif kind in _TERMINAL_EVENTS:
                    response = evt.get("response") or {}
                    evt_usage = response.get("usage")
                    if isinstance(evt_usage, dict):
                        usage = evt_usage
                    status = str(response.get("status") or kind.removeprefix("response."))
                    if status == "failed":
                        call.record(status=0, response=data_str[:8192])
                        raise _stream_error(evt)
                    if status not in ("completed", "incomplete", "done"):
                        call.record(status=0, response=data_str[:8192])
                        raise types.ProviderError(f"ChatGPT response ended with status {status}")
                    final_items = response.get("output")
                    if isinstance(final_items, list) and final_items:
                        # The terminal response is the whole output; item.done events may be absent.
                        items[:] = final_items
                    if status == "incomplete":
                        reason = str((response.get("incomplete_details") or {}).get("reason") or "")
                        stop_reason = (
                            "max_tokens"
                            if reason == "max_output_tokens"
                            else (reason or "incomplete")
                        )
                    else:
                        stop_reason = "end_turn"
                    done = True
                    return

        def _record_billed() -> None:
            """Record what the turn cost so far, and the plan window it moved."""
            if not usage and plan_usage is None:
                return
            billed = parse_output_items([], usage=usage, stop_reason="")
            _stream.record_billed_usage(
                self.budget,
                self.model,
                input_tokens=billed.input_tokens,
                output_tokens=billed.output_tokens,
                cache_read_tokens=billed.cache_read_tokens,
                cache_creation_tokens=0,
                cost_usd=0.0,
                plan_usage=plan_usage,
            )

        try:
            call.run(consume)
        except BaseException:
            _record_billed()
            raise

        if not done:
            _record_billed()
            call.record(status=0, response="stream ended without a terminal response event")
            raise types.ProviderError(
                f"ChatGPT stream from {url} ended without response.completed;"
                " upstream appears cut off."
            )

        parsed = parse_output_items(items, usage=usage, stop_reason=stop_reason)
        if not parsed.text and delta_text:
            # Text deltas without a final message item keep what the operator watched arrive.
            text = "".join(delta_text)
            parsed = dataclasses.replace(
                parsed,
                text=text,
                raw={
                    **parsed.raw,
                    "content": [{"type": "text", "text": text}, *parsed.raw["content"]],
                },
            )
        call.record(
            status=200,
            response={
                "output": parsed.raw.get("output", []),
                "usage": usage,
                "status": stop_reason,
            },
        )
        if self.budget is not None:
            if int(usage.get("input_tokens") or 0) <= 0:
                _record_billed()
                raise types.ProviderError(
                    "ChatGPT stream reported no usage input tokens;"
                    " budgeted runs require provider usage accounting"
                )
            self.budget.record(
                model=self.model,
                input_tokens=parsed.input_tokens,
                output_tokens=parsed.output_tokens,
                cache_read_tokens=parsed.cache_read_tokens,
                cache_creation_tokens=0,
                cost_usd=0.0,
                plan_usage=plan_usage,
            )
        return parsed
