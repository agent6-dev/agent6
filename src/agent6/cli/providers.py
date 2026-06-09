# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Construct role/critic/reviser/summariser providers for CLI commands."""

from __future__ import annotations

import sys
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


class _ConsoleStreamer:
    """Echo streamed reasoning + answer deltas to stderr in real time.

    Used by `plan` / `ask` / `machine create` / `--no-tui` runs (anything
    without the TUI) so the terminal shows the model thinking instead of
    sitting silent through a 30-120s reasoning call. Reasoning is dimmed and
    separated from the visible answer by a one-line header per phase switch.
    """

    def __init__(self, role: str) -> None:
        self.role = role
        self._phase: str | None = None  # None | "thinking" | "text"
        self._tty = sys.stderr.isatty()

    def write(self, piece: str, *, thinking: bool) -> None:
        want = "thinking" if thinking else "text"
        if self._phase != want:
            self._end_phase()
            self._phase = want
            label = "thinking" if thinking else "response"
            bar = f"── {self.role}: {label} ──"
            sys.stderr.write(f"\n\033[2m{bar}\033[0m\n" if self._tty else f"\n{bar}\n")
            if thinking and self._tty:
                sys.stderr.write("\033[2m")  # begin dim for the reasoning block
        sys.stderr.write(piece)
        sys.stderr.flush()

    def _end_phase(self) -> None:
        if self._phase == "thinking" and self._tty:
            sys.stderr.write("\033[0m")  # end dim
        if self._phase is not None:
            sys.stderr.write("\n")

    def close(self) -> None:
        self._end_phase()
        sys.stderr.flush()
        self._phase = None


@dataclass(frozen=True, slots=True)
class _InstrumentedProvider:
    """Wraps any Provider with role.call / role.result / budget.update emission.

    Pure decoration; the inner provider is unchanged. Lives in cli.py
    because that is the only place that owns the EventSink and the
    BudgetTracker and the role -> model mapping all at once. ``events`` may
    be None (e.g. the machine-agent subprocess has no logs.jsonl to feed),
    in which case only the optional console stream renders.
    """

    inner: Provider
    role: str
    model: str
    provider_name: str
    events: EventSink | None
    budget: BudgetTracker
    stream_text: bool = False
    console_stream: bool = False

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
        thinking_delta_callback: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        if self.events is not None:
            self.events.emit(
                "role.call",
                role=self.role,
                model=self.model,
                provider=self.provider_name,
            )
        # When the inner provider streams, fan visible text + reasoning deltas
        # out as `role.text_delta` / `role.thinking_delta` events (the TUI
        # subscribes to these) and, when `console_stream` is set, echo them
        # live to stderr (non-TUI plan/ask/machine-create). Any caller-passed
        # callback is chained through unchanged.
        role_for_event = self.role
        events = self.events
        console = _ConsoleStreamer(self.role) if self.console_stream else None

        def _on_text(piece: str) -> None:
            if events is not None:
                events.emit("role.text_delta", role=role_for_event, text=piece)
            if console is not None:
                console.write(piece, thinking=False)
            if text_delta_callback is not None:
                text_delta_callback(piece)

        def _on_thinking(piece: str) -> None:
            if events is not None:
                events.emit("role.thinking_delta", role=role_for_event, text=piece)
            if console is not None:
                console.write(piece, thinking=True)
            if thinking_delta_callback is not None:
                thinking_delta_callback(piece)

        stream = (
            self.stream_text
            or self.console_stream
            or text_delta_callback is not None
            or thinking_delta_callback is not None
        )
        effective_text_cb = _on_text if stream else None
        effective_thinking_cb = _on_thinking if stream else None
        try:
            resp = self.inner.call(
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                text_delta_callback=effective_text_cb,
                thinking_delta_callback=effective_thinking_cb,
            )
        except Exception as exc:
            if console is not None:
                console.close()
            if self.events is not None:
                self.events.emit("role.result", role=self.role, ok=False, error=str(exc)[:200])
            raise
        if console is not None:
            console.close()
        if self.events is not None:
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
