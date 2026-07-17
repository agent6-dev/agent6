# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Provider-response classification for the loop's call wrapper.

Pure predicates + policy constants the retry wrapper (``_call_with_retry``) and
the per-turn error handler (``_turn_provider_call``) consume: which HTTP statuses
are permanent, how long an upstream Retry-After is honored, an actionable hint
for a fatal error, and the self-contradictory empty-tool-call detection that
drives the blind retry. No loop state here -- each classifies one response or
status in isolation, so they stay unit-testable without a Workflow.
"""

from __future__ import annotations

from typing import Any

# HTTP statuses that will never succeed on a blind retry of the same request.
# 400 bad request, 401/403 auth, 402 insufficient credits, 404 bad
# model/endpoint, 422 malformed body. Retrying these only burns wall-time
# (observed live: a 402 "Insufficient credits" was retried on every turn for the
# rest of the run). 408/409/429 and all 5xx remain retryable and fall through to
# the normal backoff.
NON_RETRYABLE_HTTP_STATUSES = frozenset({400, 401, 402, 403, 404, 422})

# Upper bound on how long we honor an upstream Retry-After hint. A 429/503 often
# carries Retry-After: <seconds>; we wait at least that long (the provider's own
# backoff is usually shorter and just exhausts the retries before the window
# clears), but never longer than this so a buggy/hostile header can't hang a run.
RETRY_AFTER_CEILING_S = 120.0

# Finish/stop reasons that promise a tool call. A response carrying one of these
# but with NO tool_use and NO text is self-contradictory and gets retried (see
# is_empty_tool_call_response).
TOOL_CALL_STOP_REASONS = frozenset({"tool_calls", "tool_use"})


def provider_error_hint(status_code: int | None) -> str:
    """A short, actionable suffix for a fatal provider error, or "".

    The raw upstream body (e.g. a 401 JSON blob) tells a user nothing about how
    to fix it. Map the common credential/quota statuses to a next step.
    """
    if status_code in (401, 403):
        return (
            " Authentication failed: verify the provider key with `agent6 connect`"
            " or check [providers.<name>].api_key_env."
        )
    if status_code == 402:
        return " Insufficient credits/quota at the provider; top up or switch providers."
    return ""


def is_empty_tool_call_response(resp: Any) -> bool:
    """A self-contradictory provider response: the finish/stop reason says the
    model stopped to make a tool call, but no tool_use and no text came back.

    Observed live with GLM via OpenRouter after a tier-2 context restart (~50% of
    turns): finish_reason=tool_calls with an empty payload (~20 reasoning tokens,
    no content, no tool_calls). A blind retry of the identical request recovers it
    about half the time; without one the loop counts it as went_quiet and the run
    dies at the first compaction. Excludes stop_reason=="length" (deterministic
    reasoning starvation, handled separately with its own nudge)."""
    return (
        str(getattr(resp, "stop_reason", "")) in TOOL_CALL_STOP_REASONS
        and not resp.tool_uses
        and not (resp.text or "").strip()
    )


__all__ = [
    "NON_RETRYABLE_HTTP_STATUSES",
    "RETRY_AFTER_CEILING_S",
    "TOOL_CALL_STOP_REASONS",
    "is_empty_tool_call_response",
    "provider_error_hint",
]
