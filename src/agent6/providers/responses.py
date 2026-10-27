# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The generic Responses API adapter: `api_format = "responses"` on a configured endpoint.

Ordinary configured authentication (a key, a token command or a configured issuer), JSON
or SSE, ordinary token accounting. The subscription profile lives in `chatgpt.py`; both
share `_responses` for the codec and the stream machine.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import httpx2

from agent6 import budget as agent6_budget
from agent6.providers import _headers, _responses, _stream, _transport, types, wire

DEFAULT_MAX_TOKENS = 8192
_LABEL = "Responses"
# Body keys the adapter or the harness owns; `extra_body` never reaches them: retention,
# remote continuation, the output cap and the output format stay agent6's.
RESERVED_BODY = frozenset(
    {
        "background",
        "conversation",
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "model",
        "parallel_tool_calls",
        "previous_response_id",
        "response_format",
        "store",
        "stream",
        "text",
        "tool_choice",
        "tools",
    }
)


def _require_usage(usage: object) -> None:
    """Refuse a budgeted call whose usage cannot be metered.

    Args:
        usage: The response's `usage` value.

    Raises:
        ProviderError: No positive `input_tokens` (retryable: a usage-less reply is a
            gateway integrity failure, and a permanent class would end a budgeted run).
    """
    total = usage.get("input_tokens") if isinstance(usage, Mapping) else None
    try:
        metered = int(total or 0) > 0
    except (TypeError, ValueError):
        metered = False
    if not metered:
        raise types.ProviderError(
            f"{_LABEL} response carried no usage.input_tokens; budgeted runs require provider"
            " usage accounting"
        )


def _stream_error(evt: dict[str, Any]) -> types.ProviderError:
    """Return the error a failure frame carries, classified."""
    return _responses.stream_error(evt, label=_LABEL)


@dataclasses.dataclass(frozen=True, slots=True)
class ResponsesProvider:
    """A provider speaking the Responses API to a configured endpoint.

    Attributes:
        api_key: The static credential; "" sends no auth header (a local endpoint).
        model: The model id, sent in the body.
        base_url: The endpoint's base URL; `/responses` is appended.
        auth_style: The auth header style: `bearer`, `api_key_header` or `none`.
        extra_headers: The entry's extra headers, rendered per request.
        extra_body: Operator body keys merged last; `RESERVED_BODY` is filtered out.
        extra_query: Operator query parameters.
        timeout_s: The read budget in seconds.
        transcript_sink: Where each round-trip is recorded; None records nothing.
        budget: The run's tracker; None skips metering.
        reasoning_effort: The default effort; `off` sends `none`, None sends nothing.
        credential: A short-lived bearer source; a 401 re-mints it once.
    """

    api_key: str
    model: str
    base_url: str
    auth_style: wire.AuthStyle = "bearer"
    extra_headers: _headers.HeaderSet = dataclasses.field(default_factory=_headers.HeaderSet)
    extra_body: dict[str, Any] = dataclasses.field(default_factory=dict)
    extra_query: dict[str, str] = dataclasses.field(default_factory=dict)
    timeout_s: float = 600.0
    transcript_sink: types.TranscriptRecorder | None = None
    budget: agent6_budget.BudgetTracker | None = None
    reasoning_effort: str | None = None
    credential: types.BearerCredential | None = None
    # Latched on a 400 naming `temperature` so the rest of the run omits it; a list, since
    # the dataclass is frozen.
    _omit_temperature: list[bool] = dataclasses.field(default_factory=lambda: [False])

    @property
    def _route(self) -> str:
        """The route a reasoning item may be replayed to."""
        return _responses.replay_route("responses", self.base_url, self.model)

    def _build_headers(self, token: str, facts: _headers.RequestFacts) -> dict[str, str]:
        """Return one attempt's request headers, built from its token and the request's facts."""
        headers: dict[str, str] = {"content-type": "application/json"}
        authed = wire.auth_header(self.auth_style, token)
        if authed is not None:
            headers[authed[0]] = authed[1]
        headers.update(self.extra_headers.render(facts))
        return headers

    def _adapt_body_for_400(self, status: int | None, text: str, body: dict[str, Any]) -> bool:
        """Drop `temperature` after a 400 naming it (a reasoning model); latch for the run.

        Args:
            status: The HTTP status.
            text: The error text.
            body: The request body, rewritten in place.

        Returns:
            Whether the body changed, so the call retries once.
        """
        if status == 400 and "temperature" in text and "temperature" in body:
            self._omit_temperature[0] = True
            body.pop("temperature", None)
            return True
        return False

    def _parse_response(self, data: dict[str, Any]) -> types.ProviderResponse:
        """Parse a non-streaming response object.

        Args:
            data: The decoded 2xx body.

        Returns:
            The parsed response.

        Raises:
            ProviderError: The response failed or ended with an unknown status.
        """  # noqa: DOC501  # `_stream_error` builds the ProviderError named above
        status = str(data.get("status") or "completed")
        if status == "failed":
            raise _stream_error({"response": data})
        if status not in ("completed", "incomplete", "done"):
            raise types.ProviderError(f"{_LABEL} response ended with status {status}")
        usage = data.get("usage")
        output = data.get("output")
        return _responses.parse_output_items(
            output if isinstance(output, list) else [],
            usage=usage if isinstance(usage, dict) else {},
            stop_reason=_responses.stop_reason_of(data, status),
            label=_LABEL,
            route=self._route,
        )

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
        """Make one Responses call, streaming when a delta callback is set.

        Args:
            system: The system prompt, sent as `instructions`.
            messages: The conversation in Anthropic shape.
            tools: The tools the model may call.
            max_tokens: The output cap, sent as `max_output_tokens`.
            temperature: The sampling temperature; dropped after the endpoint refuses it.
            extended_thinking: Ignored; this wire has no equivalent.
            reasoning_effort: A per-call effort over the provider's own.
            text_delta_callback: Receives visible text as it streams.
            thinking_delta_callback: Receives reasoning summaries as they stream.
            should_abort: Polled during a stream; True abandons the turn.
            should_interrupt: Polled during a stream; True ends the turn to steer.
            context: The application's context for the call; None outside a conversation.

        Returns:
            The parsed response.

        Raises:
            ProviderError: The run is over budget, or the call failed.
        """
        del extended_thinking
        if self.budget is not None:
            self.budget.check()
        url, _ = wire.request_url(
            api_format="responses",
            deployment="direct",
            base_url=self.base_url,
            model=self.model,
            streaming=False,
            extra_query=self.extra_query,
        )
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": _responses.responses_input(messages, route=self._route),
            "tool_choice": "auto",
            "store": False,
            "max_output_tokens": max_tokens,
            # The encrypted reasoning items replayed for continuity across tool calls.
            "include": ["reasoning.encrypted_content"],
        }
        if tools:
            body["tools"] = _responses.tools_to_responses(tools)
        effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        if effort:
            # Omitting the field leaves the model's default on; "off" needs the explicit "none".
            body["reasoning"] = {"effort": "none" if effort == "off" else effort, "summary": "auto"}
        if temperature is not None and not self._omit_temperature[0]:
            body["temperature"] = temperature
        if self.extra_body:
            body.update({k: v for k, v in self.extra_body.items() if k not in RESERVED_BODY})
        streaming = text_delta_callback is not None or thinking_delta_callback is not None
        return _transport.ProviderCall(
            api_label=_LABEL,
            api_format="responses",
            url=url,
            body=body,
            timeout_s=self.timeout_s,
            api_key=self.api_key,
            credential=self.credential,
            transcript_sink=self.transcript_sink,
            budget=self.budget,
            model=self.model,
            facts=_headers.call_facts(self.model, _headers.responses_body_facts(body), context),
            build_headers=self._build_headers,
            adapt_400=self._adapt_body_for_400,
            adapt_attempts=int("temperature" in body),
            require_metered=lambda data: _require_usage(data.get("usage")),
            parse=self._parse_response,
            stream=(
                lambda attempt_headers: self._call_streaming(
                    url=url,
                    headers=attempt_headers,
                    body=body,
                    text_delta_callback=text_delta_callback,
                    thinking_delta_callback=thinking_delta_callback,
                    should_abort=should_abort,
                    should_interrupt=should_interrupt,
                )
            )
            if streaming
            else None,
        ).run()

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
        """Make the call over SSE and book what it cost, finished or cut.

        Args:
            url: The URL dialled.
            headers: The attempt's request headers.
            body: The request body; `stream` is set on a copy.
            text_delta_callback: Receives visible text as it streams.
            thinking_delta_callback: Receives reasoning summaries as they stream.
            should_abort: Polled each watchdog tick; True abandons the turn.
            should_interrupt: Polled each watchdog tick; True ends the turn to steer.

        Returns:
            The parsed response.

        Raises:
            ProviderError: A failed response, a stream cut before its terminal event, or
                missing usage on a budgeted run; what the cut turn already cost is
                recorded first.
        """
        body = {**body, "stream": True}
        stream_headers = {**headers, "accept": "text/event-stream"}
        state = _responses.ResponsesStreamState()
        call = _stream.SseCall(
            api_label=_LABEL,
            api_format="responses",
            url=url,
            headers=stream_headers,
            body=body,
            timeout_s=self.timeout_s,
            transcript_sink=self.transcript_sink,
            should_abort=should_abort,
            should_interrupt=should_interrupt,
        )

        def consume(resp: httpx2.Response, clock: _stream.StreamClock) -> None:
            """Read the stream's events into the state."""
            _responses.consume_stream(
                call,
                resp,
                clock,
                state,
                label=_LABEL,
                error_of=_stream_error,
                text_delta_callback=text_delta_callback,
                thinking_delta_callback=thinking_delta_callback,
            )

        def record_billed() -> None:
            """Record what the cut turn already cost."""
            if not state.usage:
                return
            billed = _responses.parse_output_items(
                [], usage=state.usage, stop_reason="", label=_LABEL, route=self._route
            )
            _stream.record_billed_usage(
                self.budget,
                self.model,
                input_tokens=billed.input_tokens,
                output_tokens=billed.output_tokens,
                cache_read_tokens=billed.cache_read_tokens,
            )

        try:
            call.run(consume)
            parsed = _responses.finish_stream(
                state, call=call, url=url, label=_LABEL, route=self._route
            )
            if self.budget is not None:
                _require_usage(state.usage)
        except BaseException:
            record_billed()
            raise
        _transport.meter_completion(self.budget, self.model, parsed, _LABEL)
        return parsed
