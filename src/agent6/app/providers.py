# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Construct role/reviser/summariser/review-seat providers for CLI commands."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from agent6 import budget as agent6_budget
from agent6 import event_log, secrets
from agent6.config import (
    AnthropicProviderEntry,
    ChatGPTProviderEntry,
    ClaudeCodeProviderEntry,
    Config,
    EffortLevel,
    RoleModel,
    RoleName,
    parse_seat_spec,
)
from agent6.harness import _reviewer
from agent6.models import registry as models_registry
from agent6.providers import (
    AnthropicProvider,
    ChatGPTCredential,
    ChatGPTProvider,
    ClaudeCodeProvider,
    CommandToken,
    OpenAIProvider,
    Provider,
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    TranscriptRecorder,
    TranscriptSink,
)


def resolve_compaction_thresholds(
    cfg: Config, rm: RoleModel | None, *, log: Callable[[str], None] | None = None
) -> tuple[int, int, int]:
    """Resolve the compaction thresholds the loop runs under.

    Explicit config values win; otherwise the thresholds are sized from the model's
    context window, or fall back to the fixed defaults when the window is unknown.
    An adaptive tier-2 threshold clamps the verbatim tail to half of itself, since a
    small window against the default tail would keep no verbatim turn at all.

    Args:
        cfg: The run's config.
        rm: The model driving the loop, or None when unresolved.
        log: Receives one line per adaptive choice.

    Returns:
        The `(drop_at_chars, summarise_at_chars, keep_recent_chars)` triple.
    """
    drop_override = cfg.context.drop_at_chars
    summarise_override = cfg.context.summarise_at_chars
    provider = rm.provider if rm is not None else ""
    model = rm.model if rm is not None else ""
    drop, summarise = models_registry.compaction_thresholds(
        provider,
        model,
        drop_override=drop_override,
        summarise_override=summarise_override,
    )
    if log is not None and drop_override is None:
        ctx = models_registry.context_window(provider, model) if model else None
        src = (
            f"adaptive from {model} (context {ctx:,} tok)"
            if ctx
            else "fixed default (context window unknown)"
        )
        # Thresholds compaction will fire at, not a compaction that happened.
        log(f"compaction thresholds: drop at {drop:,} chars, summarise at {summarise:,} [{src}]")
    keep = cfg.context.keep_recent_chars
    if summarise_override is None and keep > summarise // 2:
        keep = summarise // 2
        if log is not None:
            log(
                f"context.keep_recent_chars {cfg.context.keep_recent_chars:,} ->"
                f" {keep:,} chars: half this model's tier-2 threshold, so a restart"
                " shrinks the context instead of re-triggering on its own tail"
            )
    return drop, summarise, keep


def resolve_decompose(
    cfg: Config, rm: RoleModel | None, *, log: Callable[[str], None] | None = None
) -> Config:
    """Pin `prompt.decompose = "auto"` to on or off for this run.

    Auto turns on only when the model has a measured decompose win in the capability
    registry; an explicit on or off passes through untouched.

    Args:
        cfg: The run's config.
        rm: The worker model, or None when unresolved.
        log: Receives one line when auto turns on.

    Returns:
        The config with `prompt.decompose` pinned.
    """
    if cfg.prompt.decompose != "auto":
        return cfg
    on = rm is not None and models_registry.decompose_default(rm.model)
    if on and log is not None and rm is not None:
        log(f"decompose: auto-enabled for {rm.model} (measured win, bench/coreagent)")
    return cfg.with_decompose("on" if on else "off")


def build_role_provider(
    cfg: Config,
    role: RoleName,
    *,
    transcript_sink: TranscriptSink,
    budget: agent6_budget.BudgetTracker,
    seat: str = "",
) -> Provider:
    """Construct the configured provider for a role.

    The caller validates the route with `cfg.require_runnable(role)` first.

    Args:
        cfg: The run's config.
        role: The role whose route to build.
        transcript_sink: The recorder the provider's round-trips go to.
        budget: The tracker the provider bills.
        seat: The transcript seat stamp for an actor sharing the role's route; the role
            itself when empty.

    Returns:
        The provider, with the role's effort as its default reasoning effort.

    Raises:
        ProviderError: The role has no model or its provider entry is missing.
    """
    rm = cfg.models.resolve(role)
    if rm is None:  # pragma: no cover - blocked by require_runnable
        raise ProviderError(f"no model configured for role {role!r}")
    model = rm.model
    entry = cfg.providers.get(rm.provider)
    if entry is None:  # pragma: no cover - blocked by config validation
        raise ProviderError(
            f"models.{role}.provider = {rm.provider!r} but [providers.{rm.provider}] missing"
        )
    return _provider_from_entry(
        rm.provider,
        entry,
        model,
        rm.effort,
        # The conversation fold keeps the worker's seat and skips the side-calls' seats.
        transcript_sink=transcript_sink.for_seat(seat or role),
        budget=budget,
    )


def _provider_from_entry(
    provider_name: str,
    entry: Any,
    model: str,
    effort: EffortLevel | None,
    *,
    transcript_sink: TranscriptRecorder,
    budget: agent6_budget.BudgetTracker,
) -> Provider:
    """Build the provider for one `[providers.<name>]` entry, model and effort.

    Args:
        provider_name: The entry's name.
        entry: The parsed entry.
        model: The model id.
        effort: The default reasoning effort, or None.
        transcript_sink: The recorder the provider's round-trips go to.
        budget: The tracker the provider bills.

    Returns:
        The provider.

    Raises:
        ProviderError: The entry lacks a credential, or the effort has no equivalent.
    """
    budget.note_route(model, provider_name)
    if isinstance(entry, ClaudeCodeProviderEntry):
        if effort == "off":
            raise ProviderError(
                f"effort = off has no Claude Code equivalent ([providers.{provider_name}]"
                " passes --effort low..max); set the role's effort to low or unset it",
                fatal=True,
            )
        return ClaudeCodeProvider(
            model=model,
            binary=entry.binary,
            effort=effort,
            transcript_sink=transcript_sink,
            budget=budget,
            context_tokens=models_registry.context_window(provider_name, model),
        )
    extra_headers = tuple(sorted(entry.extra_headers.items()))
    extra_body = dict(entry.extra_body)
    extra_query = dict(entry.extra_query)
    if isinstance(entry, ChatGPTProviderEntry):
        chatgpt_credential = ChatGPTCredential(provider_name)
        account = chatgpt_credential.account_id()  # raises the connect hint when not signed in
        if not account:
            raise ProviderError(
                f"The stored ChatGPT sign-in for {provider_name!r} carries no account id;"
                f" run `agent6 connect {provider_name}` to sign in again."
            )
        return ChatGPTProvider(
            model=model,
            credential=chatgpt_credential,
            account_id=account,
            base_url=entry.base_url,
            extra_headers=extra_headers,
            extra_body=extra_body,
            extra_query=extra_query,
            timeout_s=entry.http_timeout_s,
            transcript_sink=transcript_sink,
            budget=budget,
            reasoning_effort=effort,
        )
    key = secrets.resolve_api_key(provider_name, entry.api_key_env)
    credential = (
        CommandToken(entry.token_command, ttl_s=entry.token_command_ttl_s)
        if entry.token_command
        else None
    )
    if isinstance(entry, AnthropicProviderEntry):
        # Anthropic needs a key, a token_command credential or `auth_style = "none"`.
        if not key and credential is None and entry.auth_style != "none":
            raise ProviderError(
                f"No API key for provider {provider_name!r}. Run `agent6 connect`"
                f" to store one, or set the {entry.api_key_env or 'provider'} env var."
            )
        return AnthropicProvider(
            api_key=key or "",
            model=model,
            base_url=entry.base_url,
            deployment=entry.deployment,
            auth_style=entry.auth_style,
            prompt_caching=entry.prompt_caching,
            timeout_s=entry.http_timeout_s,
            transcript_sink=transcript_sink,
            budget=budget,
            effort=effort,
            extra_headers=extra_headers,
            extra_body=extra_body,
            extra_query=extra_query,
            credential=credential,
        )
    return OpenAIProvider(
        api_key=key or "",
        model=model,
        base_url=entry.base_url,
        deployment=entry.deployment,
        auth_style=entry.auth_style,
        extra_headers=extra_headers,
        extra_body=extra_body,
        extra_query=extra_query,
        timeout_s=entry.http_timeout_s,
        transcript_sink=transcript_sink,
        budget=budget,
        reasoning_effort=effort,
        credential=credential,
    )


def close_provider(provider: Provider) -> None:
    """Release what a provider holds; the HTTP providers hold nothing and have no `close`."""
    close = getattr(provider, "close", None)
    if callable(close):
        close()


def role_temperature(cfg: Config, role: RoleName) -> float | None:
    """Return the configured sampling temperature for a role, or None."""
    rm = cfg.models.resolve(role)
    return rm.temperature if rm is not None else None


@dataclasses.dataclass(frozen=True, slots=True)
class InstrumentedProvider:
    """Wrap a provider with `role.call`, `role.result` and `budget.update` emission.

    Attributes:
        inner: The wrapped provider, unchanged.
        role: The role name the events carry.
        model: The model id the events carry.
        provider_name: The provider entry name the events carry.
        events: The sink the events go to; None emits nothing.
        budget: The tracker whose snapshot `budget.update` carries.
        stream_text: Whether to fan text and reasoning deltas out as events without a
            caller callback.
    """

    inner: Provider
    role: str
    model: str
    provider_name: str
    events: event_log.EventSink | None
    budget: agent6_budget.BudgetTracker
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
        thinking_delta_callback: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> ProviderResponse:
        """Call the inner provider and emit the role and budget events around it.

        Args:
            system: The system prompt.
            messages: The conversation so far.
            tools: The tool definitions offered.
            max_tokens: The response cap.
            temperature: The sampling temperature, or the provider's default.
            reasoning_effort: The reasoning effort, or the provider's default.
            text_delta_callback: Receives each visible text piece as it streams.
            thinking_delta_callback: Receives each reasoning piece as it streams.
            should_abort: Polled to abort the call.
            should_interrupt: Polled to interrupt the call.

        Returns:
            The inner provider's response.

        Raises:
            ProviderError: The inner call failed; the provider name is filled in.
        """
        if self.events is not None:
            self.events.emit(
                "role.call",
                role=self.role,
                model=self.model,
                provider=self.provider_name,
            )
        # Every live view subscribes to the delta events; a caller callback chains through.
        role_for_event = self.role
        events = self.events

        def _on_text(piece: str) -> None:
            if events is not None:
                events.emit("role.text_delta", role=role_for_event, text=piece)
            if text_delta_callback is not None:
                text_delta_callback(piece)

        def _on_thinking(piece: str) -> None:
            if events is not None:
                events.emit("role.thinking_delta", role=role_for_event, text=piece)
            if thinking_delta_callback is not None:
                thinking_delta_callback(piece)

        stream = (
            self.stream_text
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
                should_abort=should_abort,
                should_interrupt=should_interrupt,
            )
        except Exception as exc:
            if isinstance(exc, ProviderError) and not exc.provider:
                exc.provider = self.provider_name  # the hint names the config key
            if self.events is not None:
                self.events.emit("role.result", role=self.role, ok=False, error=str(exc)[:200])
            self._emit_budget()
            raise
        if self.events is not None:
            self.events.emit(
                "role.result",
                role=self.role,
                ok=True,
                # The settled prose: a headless run streams no deltas, so this is its only text.
                text=resp.text,
                tokens_in=resp.input_tokens,
                tokens_out=resp.output_tokens,
                cache_read=resp.cache_read_tokens,
                cache_creation=resp.cache_creation_tokens,
                stop_reason=resp.stop_reason,
            )
        self._emit_budget()
        return resp

    def close(self) -> None:
        """Release what the inner provider holds."""
        close_provider(self.inner)

    def _emit_budget(self) -> None:
        """Emit the cumulative spend after a call, whether it returned or raised.

        A stream cut after the provider reported usage is still billed, and these
        events are the only path spend takes to the cost meters and the machine's
        spend ledger.
        """
        if self.events is None:
            return
        snap = self.budget.snapshot()
        usd_total, usd_partial = self.budget.estimate_usd()
        plan = snap.plan_latest
        self.events.emit(
            "budget.update",
            input_total=snap.input_total,
            output_total=snap.output_total,
            cache_read_total=snap.cache_read_total,
            cache_creation_total=snap.cache_creation_total,
            usd_total=usd_total,
            usd_partial=usd_partial,
            usd_cap=snap.max_usd,
            tokens_unmetered=snap.unmetered_tokens,
            tokens_fallback_cap=snap.max_tokens_fallback,
            plan_used_percent=plan.used_percent if plan else 0.0,
            plan_consumed=snap.plan_consumed,
            plan_cap=snap.max_percent,
            plan_resets_at=plan.resets_at if plan else 0.0,
            # Every reported window, raw: the derived fields cannot show a second window.
            plan_windows=[
                {
                    "name": w.name,
                    "used_percent": w.used_percent,
                    "window_minutes": w.window_minutes,
                    "resets_at": w.resets_at,
                }
                for w in (plan.windows if plan else ())
            ],
            credits_has=plan.has_credits if plan else False,
            credits_unlimited=plan.credits_unlimited if plan else False,
            credits_balance=plan.credits_balance if plan else "",
        )


def reviewer_seat_provider(
    cfg: Config,
    seat: str,
    *,
    transcript_sink: TranscriptSink,
    budget: agent6_budget.BudgetTracker,
    events: event_log.EventSink | None,
) -> Provider:
    """Build the reviewer route under a seat's label, instrumented.

    Args:
        cfg: The run's config.
        seat: The seat stamp: a panel seat, the prompt reviser or the summariser.
        transcript_sink: The recorder the provider's round-trips go to.
        budget: The tracker the provider bills.
        events: The sink the role and budget events go to.

    Returns:
        The instrumented provider.
    """
    inner = build_role_provider(
        cfg, "reviewer", transcript_sink=transcript_sink, budget=budget, seat=seat
    )
    rm = cfg.models.resolve("reviewer")
    assert rm is not None  # "reviewer" resolves to the reviewer or worker model
    return InstrumentedProvider(
        inner=inner,
        role=seat,
        model=rm.model,
        provider_name=rm.provider,
        events=events,
        budget=budget,
    )


# The lenses cycled when no explicit seats are configured.
_DEFAULT_PERSONAS = ("security", "correctness", "tests", "over-engineering", "edge-cases")


def build_review_seats(
    cfg: Config,
    *,
    transcript_sink: TranscriptSink,
    budget: agent6_budget.BudgetTracker,
    n: int,
    personas: tuple[str, ...] = (),
    events: event_log.EventSink | None = None,
) -> list[_reviewer.ReviewSeat]:
    """Build the review-panel seats, one per roster entry.

    An entry is `persona[@provider/model]`: a pinned seat uses that route, a bare
    persona routes via `[models.reviewer]`. `cfg.review.seats` names the roster
    outright; otherwise `n` seats cycle the personas.

    Args:
        cfg: The run's config.
        transcript_sink: The recorder the seats' round-trips go to.
        budget: The tracker the seats bill.
        n: The seat count when the config names no roster.
        personas: The personas to cycle; the built-in set when empty.
        events: The sink each seat's events go to; None leaves the seats uninstrumented,
            so their spend reaches no surface.

    Returns:
        The seats in roster order.

    Raises:
        ProviderError: An entry does not parse or names a missing provider.
    """

    def _instrumented(provider: Provider, persona: str, model: str, provider_name: str) -> Provider:
        if events is None:
            return provider
        return InstrumentedProvider(
            inner=provider,
            role=f"review:{persona}",
            model=model,
            provider_name=provider_name,
            events=events,
            budget=budget,
        )

    if cfg.review.seats:
        specs = list(cfg.review.seats)
    else:
        pool = list(personas) if personas else list(_DEFAULT_PERSONAS)
        specs = [pool[i % len(pool)] for i in range(max(1, n))]
    parsed_specs: list[tuple[str, str, str]] = []
    for spec in specs:
        try:
            persona, provider_name, model = parse_seat_spec(spec)
        except ValueError as exc:
            raise ProviderError(f"review seat: {exc}") from exc
        persona = persona or "general"
        if provider_name and provider_name not in cfg.providers:
            raise ProviderError(
                f"review seat {spec!r} names provider {provider_name!r} but"
                f" [providers.{provider_name}] is missing"
            )
        parsed_specs.append((persona, provider_name, model))

    # A fully pinned panel needs no reviewer route.
    if any(not (provider_name and model) for _, provider_name, model in parsed_specs):
        cfg.require_runnable("reviewer")
    rm = cfg.models.resolve("reviewer")
    seats: list[_reviewer.ReviewSeat] = []
    for persona, provider_name, model in parsed_specs:
        if provider_name and model:
            entry = cfg.providers[provider_name]
            seat_model = model
            provider = _provider_from_entry(
                provider_name,
                entry,
                seat_model,
                None,
                # An unstamped sink records seat="", which the fold renders as worker turns.
                transcript_sink=transcript_sink.for_seat(f"review:{persona}"),
                budget=budget,
            )
            label = f"{provider_name}/{seat_model}"
            provider = _instrumented(provider, persona, seat_model, provider_name)
        else:
            provider = build_role_provider(
                cfg,
                "reviewer",
                transcript_sink=transcript_sink,
                budget=budget,
                seat=f"review:{persona}",
            )
            seat_model = rm.model if rm is not None else "reviewer"
            label = f"{rm.provider}/{seat_model}" if rm is not None else seat_model
            provider = _instrumented(
                provider, persona, seat_model, rm.provider if rm is not None else ""
            )
        seats.append(
            _reviewer.ReviewSeat(
                persona=persona, model=label, provider=provider, tier=cfg.review.tier
            )
        )
    return seats


def build_prompt_reviser_provider(
    cfg: Config,
    *,
    transcript_sink: TranscriptSink,
    budget: agent6_budget.BudgetTracker,
    events: event_log.EventSink,
) -> Provider | None:
    """Route the reviewer role as a one-shot prompt reviser.

    Args:
        cfg: The run's config.
        transcript_sink: The recorder the reviser's round-trips go to.
        budget: The tracker the reviser bills.
        events: The sink the role and budget events go to.

    Returns:
        The instrumented provider, or None when `prompt.revise_prompt` is off.
    """
    if cfg.prompt.revise_prompt == "off":
        return None
    return reviewer_seat_provider(
        cfg, "prompt_reviser", transcript_sink=transcript_sink, budget=budget, events=events
    )
