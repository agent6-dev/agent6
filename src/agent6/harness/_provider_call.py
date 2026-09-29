# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Call the provider under a bounded retry.

`ProviderCaller` runs `provider.call`; the predicates and constants classify one response or
status on their own, so they test without a `Harness`.
"""

from __future__ import annotations

import dataclasses
import random
import time
from collections.abc import Callable
from typing import Any

from agent6.providers import (
    Provider,
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    ProviderResponse,
    ToolDefinition,
    output_cap_truncated,
)

# A timeout (408), a conflict (409), too early (425), a rate limit (429) and every 5xx retry.
NON_RETRYABLE_HTTP_STATUSES = frozenset(set(range(300, 500)) - {408, 409, 425, 429})

# The longest wait an upstream Retry-After header earns, so a hostile header cannot hang a run.
RETRY_AFTER_CEILING_S = 120.0

# The stop reasons that promise a tool call.
TOOL_CALL_STOP_REASONS = frozenset({"tool_calls", "tool_use"})


def provider_error_hint(status_code: int | None, provider: str = "") -> str:
    """Return the next step for a credential or quota status, "" for any other.

    Args:
        status_code: The HTTP status of the fatal error.
        provider: The failing provider's name, for its config key.

    Returns:
        A sentence to append to the error, or "".
    """
    if status_code in (401, 403):
        return (
            " Authentication failed: verify the provider key with `agent6 connect`"
            f" or check [providers.{provider or '<name>'}].api_key_env."
        )
    if status_code == 402:
        return " Insufficient credits/quota at the provider; top up or switch providers."
    return ""


def is_empty_tool_call_response(resp: Any) -> bool:
    """Return whether the stop reason promises a tool call that no tool_use or text delivers.

    A blind retry recovers such a response about half the time; a `length` stop is reasoning
    starvation instead, with its own nudge.

    Args:
        resp: The provider response.

    Returns:
        True for the self-contradictory shape.
    """
    return (
        str(getattr(resp, "stop_reason", "")) in TOOL_CALL_STOP_REASONS
        and not resp.tool_uses
        and not (resp.text or "").strip()
    )


def reasoning_starvation(resp: ProviderResponse) -> int:
    """Return the reasoning characters of a turn the output cap cut, 0 for any other turn.

    Args:
        resp: The provider response.

    Returns:
        The count, which tells a starved reasoner from a model that gave up.
    """
    if not output_cap_truncated(resp) or resp.output_tokens <= 0:
        return 0
    reasoning_chars = 0
    raw_content = (resp.raw or {}).get("content") or []
    if isinstance(raw_content, list):
        for block in raw_content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                reasoning_chars += len(str(block.get("thinking") or ""))
    return reasoning_chars


@dataclasses.dataclass(frozen=True, slots=True)
class CallSettings:
    """Hold the worker call's knobs.

    Attributes:
        retry_count: Retries of a transient provider error; 0 disables retrying.
        retry_delay_s: The first backoff delay, doubled per attempt with full jitter.
        retry_max_delay_s: The backoff cap.
        temperature: The sampling temperature for every call; None leaves each provider its own.
        per_call_max_tokens: One turn's output cap, sized for reasoning plus a tool call.
        metric_task_max_tokens: The output cap on a metric run, whose single-turn edits are large.
    """

    retry_count: int = 4
    retry_delay_s: float = 2.0
    retry_max_delay_s: float = 30.0
    temperature: float | None = 0.0
    per_call_max_tokens: int = 16384
    metric_task_max_tokens: int = 65536


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderCaller:
    """Call the provider with at most `retry_count + 1` attempts.

    A transient `ProviderError` backs off with full jitter, waiting at least the upstream
    Retry-After; a permanent one re-raises at once. An empty tool-call response is re-asked
    after a short delay, and when every attempt is empty the last is returned for the
    went-quiet handler. An abort, an interrupt and `BudgetExceededError` are never retried.

    Attributes:
        provider: The provider called.
        retry_count: Retries of a transient error.
        retry_delay_s: The first backoff delay.
        retry_max_delay_s: The backoff cap.
        temperature: The sampling temperature, None for the provider's default.
        should_abort: Whether an operator stop is pending.
        should_interrupt: Whether an operator steer is pending.
        log: The run's text logger.
        emit: The run's event emitter.
    """

    provider: Provider
    retry_count: int
    retry_delay_s: float
    retry_max_delay_s: float
    temperature: float | None
    should_abort: Callable[[], bool]
    should_interrupt: Callable[[], bool]
    log: Callable[[str], None]
    emit: Callable[..., None]

    def call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int,
    ) -> ProviderResponse:
        """Call the provider under the retry.

        Args:
            system: The system prompt.
            messages: The conversation.
            tools: The tool definitions.
            max_tokens: The output cap.

        Returns:
            The first usable response, or the last empty one.

        Raises:
            ProviderError: A permanent error, or a transient one after the last attempt.
            ProviderAborted: An operator stop.
            ProviderInterrupted: An operator steer.
        """
        attempts = max(1, self.retry_count + 1)
        attempt = 1
        while True:
            try:
                resp = self.provider.call(
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    should_abort=self.should_abort,
                    should_interrupt=self.should_interrupt,
                )
            except (ProviderAborted, ProviderInterrupted):
                raise  # an operator stop or steer is handled, never retried
            except ProviderError as exc:
                if exc.fatal or exc.status_code in NON_RETRYABLE_HTTP_STATUSES:
                    self.log(
                        f"LOOP: provider error {exc.status_code or 'fatal'} is permanent;"
                        " not retrying"
                    )
                    self.emit(
                        "loop.provider.fatal", status_code=exc.status_code, error=str(exc)[:200]
                    )
                    raise
                if attempt == attempts:
                    exc.attempts = attempt
                    raise
                delay = self._backoff(attempt, exc.retry_after_s)
                self.log(
                    f"LOOP: provider error attempt {attempt}/{attempts}: {exc}"
                    f" - retrying in {delay:.2f}s"
                )
                self.emit("loop.provider.retry", attempt=attempt, error=str(exc)[:200])
            else:
                if attempt == attempts or not is_empty_tool_call_response(resp):
                    return resp
                # Model flakiness, not rate limiting: a short fixed delay.
                delay = min(self.retry_delay_s, 1.0) * random.uniform(0.5, 1.0)  # noqa: S311
                self.log(
                    f"LOOP: empty tool-call response attempt {attempt}/{attempts}"
                    f" (stop_reason={resp.stop_reason!r}, no tool_use/text);"
                    f" retrying in {delay:.2f}s"
                )
                self.emit(
                    "loop.provider.empty_tool_call_retry",
                    attempt=attempt,
                    stop_reason=str(resp.stop_reason),
                )
            time.sleep(delay)
            attempt += 1

    def _backoff(self, attempt: int, retry_after_s: float | None) -> float:
        """Return the delay before the next attempt.

        Exponential with full jitter floored at half, capped at `retry_max_delay_s`, and never
        shorter than the upstream Retry-After capped at `RETRY_AFTER_CEILING_S`.

        Args:
            attempt: The attempt that failed, from 1.
            retry_after_s: The upstream Retry-After hint, when any.

        Returns:
            The delay in seconds.
        """
        capped = min(self.retry_delay_s * 2 ** (attempt - 1), self.retry_max_delay_s)
        delay = capped * random.uniform(0.5, 1.0)  # noqa: S311
        if retry_after_s is not None:
            delay = max(delay, min(retry_after_s, RETRY_AFTER_CEILING_S))
        return delay


__all__ = [
    "NON_RETRYABLE_HTTP_STATUSES",
    "RETRY_AFTER_CEILING_S",
    "TOOL_CALL_STOP_REASONS",
    "ProviderCaller",
    "is_empty_tool_call_response",
    "provider_error_hint",
    "reasoning_starvation",
]
