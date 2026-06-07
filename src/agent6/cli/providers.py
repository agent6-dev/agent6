# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Construct role/critic/reviser/summariser providers for CLI commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent6.budget import BudgetTracker
from agent6.config import (
    AnthropicProviderEntry,
    Config,
    RoleName,
)
from agent6.events import EventSink
from agent6.providers import (
    AnthropicProvider,
    OpenAIProvider,
    Provider,
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    TranscriptSink,
)
from agent6.secrets import resolve_api_key


def _build_role_provider(
    cfg: Config,
    role: RoleName,
    *,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
    model_override: str = "",
) -> Provider:
    """Construct the configured provider for `role`.

    Resolves the API key via `agent6.secrets.resolve_api_key` (env var named
    by `api_key_env` first, then `secrets.toml`). `model_override` (if
    truthy) replaces the model string; provider routing is unchanged. The
    role's `thinking` level is wired to the provider's default reasoning
    effort. Callers should have validated routing via
    `cfg.require_runnable(role)` first.
    """
    rm = cfg.models.resolve(role)
    if rm is None:  # pragma: no cover - blocked by require_runnable
        raise ProviderError(f"no model configured for role {role!r}")
    model = model_override or rm.model
    entry = cfg.providers.get(rm.provider)
    if entry is None:  # pragma: no cover - blocked by config validation
        raise ProviderError(
            f"models.{role}.provider = {rm.provider!r} but [providers.{rm.provider}] missing"
        )
    key = resolve_api_key(rm.provider, entry.api_key_env)
    if isinstance(entry, AnthropicProviderEntry):
        if not key:
            raise ProviderError(
                f"No API key for provider {rm.provider!r}. Run `agent6 connect`"
                f" to store one, or set the {entry.api_key_env or 'provider'} env var."
            )
        return AnthropicProvider(
            api_key=key,
            model=model,
            prompt_caching=entry.prompt_caching,
            timeout_s=entry.http_timeout_s,
            transcript_sink=transcript_sink,
            budget=budget,
            thinking=rm.thinking,
        )
    return OpenAIProvider(
        api_key=key or "",
        model=model,
        base_url=entry.base_url,
        extra_headers=tuple(sorted(entry.extra_headers.items())),
        timeout_s=entry.http_timeout_s,
        transcript_sink=transcript_sink,
        budget=budget,
        reasoning_effort=rm.thinking,
    )


def _role_temperature(cfg: Config, role: RoleName) -> float | None:
    """The configured sampling temperature for *role* (worker fallback)."""
    rm = cfg.models.resolve(role)
    return rm.temperature if rm is not None else None


@dataclass(frozen=True, slots=True)
class _InstrumentedProvider:
    """Wraps any Provider with role.call / role.result / budget.update emission.

    Pure decoration; the inner provider is unchanged. Lives in cli.py
    because that is the only place that owns the EventSink and the
    BudgetTracker and the role -> model mapping all at once.
    """

    inner: Provider
    role: str
    model: str
    provider_name: str
    events: EventSink
    budget: BudgetTracker
    stream_text: bool = False

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        text_delta_callback: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        self.events.emit(
            "role.call",
            role=self.role,
            model=self.model,
            provider=self.provider_name,
        )
        # When the inner provider streams, fan text deltas
        # out as `role.text_delta` events. The TUI can subscribe to
        # these for a live-typing render; non-TUI consumers ignore the
        # event and see no behaviour change.
        role_for_event = self.role
        events = self.events

        def _on_delta(piece: str) -> None:
            events.emit("role.text_delta", role=role_for_event, text=piece)

        effective_delta_cb: Callable[[str], None] | None
        if text_delta_callback is not None:
            # Caller already passed one — chain through ours too.
            outer = text_delta_callback

            def _both(piece: str) -> None:
                _on_delta(piece)
                outer(piece)

            effective_delta_cb = _both
        else:
            effective_delta_cb = _on_delta if self.stream_text else None
        try:
            resp = self.inner.call(
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                text_delta_callback=effective_delta_cb,
            )
        except Exception as exc:
            self.events.emit("role.result", role=self.role, ok=False, error=str(exc)[:200])
            raise
        self.events.emit(
            "role.result",
            role=self.role,
            ok=True,
            tokens_in=resp.input_tokens,
            tokens_out=resp.output_tokens,
            cache_read=resp.cache_read_tokens,
            cache_creation=resp.cache_creation_tokens,
            stop_reason=resp.stop_reason,
        )
        snap = self.budget.snapshot()
        usd_total, usd_partial = self.budget.estimate_usd()
        self.events.emit(
            "budget.update",
            input_total=snap["input_total"],
            output_total=snap["output_total"],
            input_cap=snap["max_input_tokens"],
            output_cap=snap["max_output_tokens"],
            usd_total=usd_total,
            usd_partial=usd_partial,
        )
        return resp


def _build_critic_provider(
    cfg: Config,
    *,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
    events: EventSink,
) -> Provider | None:
    """critic-in-loop. Routes the reviewer role as the critic
    provider when ``workflow.critic != "off"``. Returns None when
    disabled so Workflow leaves the critic path inert."""
    if cfg.workflow.critic == "off":
        return None
    critic_inner = _build_role_provider(
        cfg, "reviewer", transcript_sink=transcript_sink, budget=budget
    )
    rm = cfg.models.resolve("reviewer")
    assert rm is not None  # critic only runs once a worker/reviewer model exists
    return _InstrumentedProvider(
        inner=critic_inner,
        role="critic",
        model=rm.model,
        provider_name=rm.provider,
        events=events,
        budget=budget,
    )


def _build_prompt_reviser_provider(
    cfg: Config,
    *,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
    events: EventSink,
) -> Provider | None:
    """Route the reviewer role as a one-shot prompt reviser."""
    if cfg.workflow.revise_prompt == "off":
        return None
    reviser_inner = _build_role_provider(
        cfg, "reviewer", transcript_sink=transcript_sink, budget=budget
    )
    rm = cfg.models.resolve("reviewer")
    assert rm is not None  # reviser only runs once a worker/reviewer model exists
    return _InstrumentedProvider(
        inner=reviser_inner,
        role="prompt_reviser",
        model=rm.model,
        provider_name=rm.provider,
        events=events,
        budget=budget,
    )


def _build_summariser_provider(
    cfg: Config,
    *,
    transcript_sink: TranscriptSink,
    budget: BudgetTracker,
    events: EventSink,
) -> Provider:
    """Route the reviewer role as the tier-2 context summariser. Always
    available (context compaction can fire on any run) and cheaper than the
    worker model."""
    summariser_inner = _build_role_provider(
        cfg, "reviewer", transcript_sink=transcript_sink, budget=budget
    )
    rm = cfg.models.resolve("reviewer")
    assert rm is not None  # summariser falls back to the worker model
    return _InstrumentedProvider(
        inner=summariser_inner,
        role="summariser",
        model=rm.model,
        provider_name=rm.provider,
        events=events,
        budget=budget,
    )
