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

import dataclasses
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import httpx2

from agent6 import budget as agent6_budget
from agent6.providers import (
    _headers,
    _responses,
    _stream,
    _transport,
    chatgpt_oauth,
    types,
)
from agent6.providers import wire as providers_wire

DEFAULT_MAX_TOKENS = 8192

# The plan's window is exhausted; carried as a 429 so the loop backs off rather than failing.
_USAGE_LIMIT_CODES = frozenset({"usage_limit_reached", "usage_not_included", "rate_limit_exceeded"})


_EFFORT_UNSUPPORTED = re.compile(r"Supported values are:\s*((?:'[a-z]+'(?:,\s*(?:and\s*)?)?)+)")


def _accepted_efforts(status: int | None, text: str) -> tuple[str, ...]:
    """Return the effort levels a 400 lists for `reasoning.effort`, in the order given."""
    if status != 400 or '"reasoning.effort"' not in text or "unsupported_value" not in text:
        return ()
    m = _EFFORT_UNSUPPORTED.search(text)
    return tuple(re.findall(r"'([a-z]+)'", m.group(1))) if m else ()


def _unreachable_hook(data: dict[str, Any]) -> Any:
    """Refuse the non-streaming hooks; this wire is stream-only.

    Args:
        data: The response body, unused.

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
    return _responses.stream_error(evt, label="ChatGPT", usage_limit_codes=_USAGE_LIMIT_CODES)


@dataclasses.dataclass(frozen=True, slots=True)
class ChatGPTProvider:
    """The ChatGPT provider, constructed once per run.

    Attributes:
        model: The model id.
        credential: The OAuth credential; a 401 refreshes it once.
        account_id: The `chatgpt-account-id` header.
        base_url: The backend's base URL.
        extra_headers: The entry's extra headers, rendered per request.
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
    extra_headers: _headers.HeaderSet = dataclasses.field(default_factory=_headers.HeaderSet)
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
            url = f"{self.base_url.rstrip('/')}/usage"
            resp = httpx2.get(
                url,
                headers=self._build_headers(token, _headers.plain_facts("usage")),
                timeout=20.0,
                follow_redirects=False,
                trust_env=_transport.trust_env_for(url),
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

    def _build_headers(self, token: str, facts: _headers.RequestFacts) -> dict[str, str]:
        """Return one attempt's request headers, built from its token and the request's facts."""
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
            "chatgpt-account-id": self.account_id,
            "openai-beta": "responses=experimental",
            "originator": "agent6",
            "session-id": self.session_id,
        }
        headers.update(self.extra_headers.render(facts))
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
        context: _headers.RequestContext | None = None,
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
            context: The application's context for the call; None outside a conversation.

        Returns:
            The parsed response.

        Raises:
            ProviderError: The run is over budget, or the call failed.
        """
        del max_tokens, temperature, extended_thinking
        if self.budget is not None:
            if not self._preflighted[0]:
                self._preflighted[0] = True
                self.budget.register_plan_reader(self.model, self.preflight)
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
            "input": _responses.responses_input(messages, route=self._route),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "store": False,
            "stream": True,
            # The encrypted reasoning items replayed for continuity across tool calls.
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": self.session_id,
        }
        if tools:
            body["tools"] = _responses.tools_to_responses(tools)
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
            facts=_headers.call_facts(self.model, _headers.responses_body_facts(body), context),
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

    @property
    def _route(self) -> str:
        """The route a reasoning item may be replayed to."""
        return _responses.replay_route("chatgpt", self.base_url, self.model)

    def _call_streaming(
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
        """Make the call over SSE, metering the plan windows the response headers carry.

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
        state = _responses.ResponsesStreamState()
        plan_usage: agent6_budget.PlanUsage | None = None

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

        def consume(resp: httpx2.Response, clock: _stream.StreamClock) -> None:
            """Read the stream's events into the state."""
            _responses.consume_stream(
                call,
                resp,
                clock,
                state,
                label="ChatGPT",
                error_of=_stream_error,
                text_delta_callback=text_delta_callback,
                thinking_delta_callback=thinking_delta_callback,
            )

        def _record_billed() -> None:
            """Record what the turn cost so far, and the plan window it moved."""
            if not state.usage and plan_usage is None:
                return
            billed = _responses.parse_output_items(
                [], usage=state.usage, stop_reason="", label="ChatGPT", route=self._route
            )
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
        try:
            parsed = _responses.finish_stream(
                state, call=call, url=url, label="ChatGPT", route=self._route
            )
        except types.ProviderError:
            _record_billed()
            raise
        if self.budget is not None:
            if int(state.usage.get("input_tokens") or 0) <= 0:
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
