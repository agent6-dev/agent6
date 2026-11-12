# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Construct role/summariser/review-seat providers for CLI commands."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from agent6 import budget as agent6_budget
from agent6 import kinds, media
from agent6.app import conversation_state
from agent6.config import (
    AnthropicProviderEntry,
    ChatGPTProviderEntry,
    ClaudeCodeProviderEntry,
    Config,
    EffortLevel,
    ResponsesProviderEntry,
    ReviewerConfig,
    RoleModel,
    RoleName,
    secret_store,
)
from agent6.harness import provider_call, review_gate
from agent6.models import cache as models_cache
from agent6.models import registry as models_registry
from agent6.providers import (
    AnthropicProvider,
    ChatGPTProvider,
    ClaudeCodeProvider,
    CommandToken,
    OpenAIProvider,
    Provider,
    ProviderError,
    ProviderResponse,
    ResponsesProvider,
    ToolDefinition,
    TranscriptRecorder,
    TranscriptSink,
    oauth,
    provider_types,
    request_headers,
)
from agent6.sessions import event_log, journal


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
    drop_override = cfg.compaction.drop_at_chars
    summarise_override = cfg.compaction.summarise_at_chars
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
        key = models_registry.bundled_context_key(model)
        if key is not None and key not in (model, models_registry.normalize_model_id(model)):
            src += f"; family fallback from {key}"
        # Thresholds compaction will fire at, not a compaction that happened.
        log(f"compaction thresholds: drop at {drop:,} chars, summarise at {summarise:,} [{src}]")
    keep = cfg.compaction.keep_recent_chars
    if summarise_override is None and keep > summarise // 2:
        keep = summarise // 2
        if log is not None:
            log(
                f"compaction.keep_recent_chars {cfg.compaction.keep_recent_chars:,} ->"
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


def input_capabilities(cfg: Config, rm: RoleModel | None) -> media.InputCapabilities:
    """Resolve the calling role's media facts without mutating its dispatcher.

    Args:
        cfg: The effective configuration.
        rm: The resolved worker, planner or reviewer seat.

    Returns:
        The route's cached facts, unknown when no route is configured.
    """
    entry = cfg.providers.get(rm.provider) if rm is not None else None
    if rm is None or entry is None:
        return media.InputCapabilities()
    return models_cache.cached_input_capabilities(rm.provider, entry, rm.model)


def build_role_provider(
    cfg: Config,
    role: RoleName,
    *,
    transcript_sink: TranscriptSink,
    budget: agent6_budget.BudgetTracker,
    seat: str = "",
    events: event_log.EventSink | None = None,
) -> Provider:
    """Construct the configured provider for a role.

    Every provider requires `models.default`, even when the role pins its own route.

    Args:
        cfg: The run's config.
        role: The role whose route to build.
        transcript_sink: The recorder the provider's round-trips go to.
        budget: The tracker the provider bills.
        seat: The transcript seat stamp for an actor sharing the role's route; the role
            itself when empty.
        events: The run's journal sink; None emits no authentication events.

    Returns:
        The provider, with the role's effort as its default reasoning effort.

    Raises:
        ProviderError: The role has no model or its provider entry is missing.
    """
    cfg.require_runnable(role)
    rm = cfg.models.resolve(role)
    if rm is None:  # pragma: no cover - blocked by require_runnable
        raise ProviderError(f"no model configured for role {role!r}")
    model = rm.model
    entry = cfg.providers.get(rm.provider)
    if entry is None:  # pragma: no cover - blocked by config validation
        raise ProviderError(
            f"models.{role}.model = {f'{rm.provider}/{rm.model}'!r} but"
            f" [providers.{rm.provider}] missing"
        )
    return _provider_from_entry(
        rm.provider,
        entry,
        model,
        rm.effort,
        # The conversation fold keeps the worker's seat and skips the side-calls' seats.
        transcript_sink=transcript_sink.for_seat(seat or role),
        budget=budget,
        events=events,
    )


def _provider_from_entry(
    provider_name: str,
    entry: Any,
    model: str,
    effort: EffortLevel | None,
    *,
    transcript_sink: TranscriptRecorder,
    budget: agent6_budget.BudgetTracker,
    events: event_log.EventSink | None = None,
) -> Provider:
    """Build the provider for one `[providers.<name>]` entry, model and effort.

    Args:
        provider_name: The entry's name.
        entry: The parsed entry.
        model: The model id.
        effort: The default reasoning effort, or None.
        transcript_sink: The recorder the provider's round-trips go to.
        budget: The tracker the provider bills.
        events: The run's journal sink; None emits no authentication events.

    Returns:
        The provider.

    Raises:
        ProviderError: The entry lacks a credential, or the effort has no equivalent.
    """
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
            provider_name=provider_name,
            context_tokens=models_registry.context_window(provider_name, model),
        )

    def observe_auth(event: provider_types.AuthEvent) -> None:
        if events is not None:
            events.emit(
                journal.ProviderAuth(provider=provider_name, kind=event.kind, outcome=event.outcome)
            )

    auth_observer = observe_auth if events is not None else None
    extra_headers = request_headers.HeaderSet.from_config(entry.extra_headers)
    extra_body = dict(entry.extra_body)
    extra_query = dict(entry.extra_query)
    if isinstance(entry, ChatGPTProviderEntry):
        chatgpt_credential = oauth.OAuthCredential(
            provider_name, entry, auth_observer=auth_observer
        )
        account = chatgpt_credential.account_id()  # raises the connect hint when not signed in
        if not account:
            raise ProviderError(
                f"The stored ChatGPT sign-in for {provider_name!r} carries no account id;"
                f" run `agent6 connect {provider_name}` to sign in again."
            )
        return ChatGPTProvider(
            model=model,
            credential=chatgpt_credential,
            auth_observer=auth_observer,
            account_id=account,
            base_url=entry.base_url,
            extra_headers=extra_headers,
            extra_body=extra_body,
            extra_query=extra_query,
            timeout_s=entry.http_timeout_s,
            transcript_sink=transcript_sink,
            budget=budget,
            provider_name=provider_name,
            reasoning_effort=effort,
        )
    key = secret_store.resolve_api_key(provider_name, entry.api_key_env)
    credential: provider_types.BearerCredential | None = (
        CommandToken(entry.token_command, ttl_s=entry.token_command_ttl_s)
        if entry.token_command
        else None
    )
    if entry.oauth is not None:
        # A configured issuer is the one credential source: the stored, bound grant.
        credential = oauth.OAuthCredential(provider_name, entry, auth_observer=auth_observer)
        key = ""
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
            provider_name=provider_name,
            effort=effort,
            extra_headers=extra_headers,
            extra_body=extra_body,
            extra_query=extra_query,
            credential=credential,
            auth_observer=auth_observer,
        )
    if isinstance(entry, ResponsesProviderEntry):
        return ResponsesProvider(
            api_key=key or "",
            model=model,
            base_url=entry.base_url,
            auth_style=entry.auth_style,
            extra_headers=extra_headers,
            extra_body=extra_body,
            extra_query=extra_query,
            timeout_s=entry.http_timeout_s,
            transcript_sink=transcript_sink,
            budget=budget,
            provider_name=provider_name,
            reasoning_effort=effort,
            credential=credential,
            auth_observer=auth_observer,
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
        provider_name=provider_name,
        reasoning_effort=effort,
        credential=credential,
        auth_observer=auth_observer,
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


def _usage_fields(usage: provider_types.ProviderUsage | None) -> dict[str, Any]:
    """Return a result's reported usage fields; an unobserved usage stays absent."""
    if usage is None:
        return {}
    fields: dict[str, int | float] = {
        "tokens_in": usage.input_tokens,
        "tokens_out": usage.output_tokens,
        "cache_read": usage.cache_read_tokens,
        "cache_creation": usage.cache_creation_tokens,
    }
    if usage.cost_usd is not None:
        fields["cost_usd"] = usage.cost_usd
    return fields


@dataclasses.dataclass(frozen=True, slots=True)
class InstrumentedProvider:
    """Wrap a provider with `role.call`, `role.attempt`, `role.result` and `budget.update` emission.

    Attributes:
        inner: The wrapped provider, unchanged.
        role: The role name the events carry.
        model: The model id the events carry.
        provider_name: The provider entry name the events carry.
        events: The sink the events go to; None emits nothing.
        budget: The tracker whose snapshot `budget.update` carries.
        seat_index: The review panel seat's index, carried by its events; None outside a
            panel.
        stream_text: Whether to fan text and reasoning deltas out as events without a
            caller callback.
        conversations: The session's conversation state; a side call holds its own.
    """

    inner: Provider
    role: str
    model: str
    provider_name: str
    events: event_log.EventSink | None
    budget: agent6_budget.BudgetTracker
    stream_text: bool = False
    seat_index: int | None = None
    conversations: conversation_state.Conversations = dataclasses.field(
        default_factory=conversation_state.Conversations
    )

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
        context: request_headers.RequestContext | None = None,
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
            context: The caller's context; one without a conversation id names the initiator
                only, and the conversation supplies the rest; None is an agent-initiated call.

        Returns:
            The inner provider's response.

        Raises:
            ProviderError: The inner call failed; the provider name is filled in.
        """
        call_context = (context.call_context if context is not None else None) or (
            kinds.CallContext(seat_index=self.seat_index)
        )
        caller_observe = context.observe if context is not None else None
        request_ids: list[str] = []  # the attempt's HTTP request, for its role.attempt row

        def observe(facts: request_headers.RequestFacts) -> None:
            request_ids.append(facts.request_id)
            self._observe(facts)
            if caller_observe is not None:
                caller_observe(facts)

        # The caller's attempt counts survive the conversation state's rebuild of the context.
        attempt, attempts = (context.attempt, context.attempts) if context is not None else (1, 1)
        if context is None or not context.conversation_id:
            initiator = context.initiator if context is not None else "agent"
            context = self.conversations.context(
                self.role, self.provider_name, initiator=initiator, call_context=call_context
            )
        context = dataclasses.replace(
            context,
            call_context=call_context,
            observe=observe,
            attempt=attempt,
            attempts=attempts,
        )
        # One lifecycle per logical call: the first attempt opens it, the one that settles or
        # closes it ends it, and every attempt leaves its own row beneath.
        if self.events is not None and context.attempt == 1:
            self.events.emit(journal.RoleCall(**self._identity(call_context)))
        # Every live view subscribes to the delta events; a caller callback chains through.
        role_for_event = self.role
        events = self.events
        identity = call_context.event_fields()

        def _on_text(piece: str) -> None:
            if events is not None:
                events.emit(journal.RoleTextDelta(role=role_for_event, text=piece, **identity))
            if text_delta_callback is not None:
                text_delta_callback(piece)

        def _on_thinking(piece: str) -> None:
            if events is not None:
                events.emit(journal.RoleThinkingDelta(role=role_for_event, text=piece, **identity))
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
                context=context,
            )
        except Exception as exc:
            if isinstance(exc, ProviderError) and not exc.provider:
                exc.provider = self.provider_name  # the hint names the config key
            self._record_attempt(
                call_context,
                context,
                request_ids,
                {
                    "ok": False,
                    "error": str(exc)[:200],
                    **_usage_fields(exc.usage if isinstance(exc, ProviderError) else None),
                },
                ends_call=provider_call.closes_call(exc, context),
            )
            self._emit_budget()
            raise
        self.conversations.completed(self.role, self.provider_name, call_context=call_context)
        self._record_attempt(
            call_context,
            context,
            request_ids,
            {"ok": True, "stop_reason": resp.stop_reason, **_usage_fields(resp.usage)},
            ends_call=provider_call.settles_call(resp, context),
            # The settled prose: a headless run streams no deltas, so this is its only text.
            text=resp.text,
        )
        self._emit_budget()
        return resp

    def _record_attempt(
        self,
        call_context: kinds.CallContext,
        context: request_headers.RequestContext,
        request_ids: list[str],
        outcome: dict[str, Any],
        *,
        ends_call: bool,
        text: str | None = None,
    ) -> None:
        """Journal one attempt's row and, when it ends the logical call, the call's result.

        Args:
            call_context: The call's identity.
            context: The attempt's request context, with its number.
            request_ids: The HTTP requests observed so far; the last is this attempt's.
            outcome: The attempt's `ok`, usage and stop reason or error.
            ends_call: Whether no retry of the call follows.
            text: The response text the result carries; None for a failure.
        """
        if self.events is None:
            return
        identity = self._identity(call_context)
        self.events.emit(
            journal.RoleAttempt(
                **identity,
                attempt=context.attempt,
                request_id=request_ids[-1] if request_ids else None,
                **outcome,
            )
        )
        if ends_call:
            settled: dict[str, Any] = {"text": text} if text is not None else {}
            self.events.emit(journal.RoleResult(**identity, **outcome, **settled))

    def _identity(self, call_context: kinds.CallContext) -> dict[str, Any]:
        """Return the fields that pair a `role.call` with its `role.result`."""
        return {
            "role": self.role,
            "model": self.model,
            "provider": self.provider_name,
            **call_context.event_fields(),
        }

    def _observe(self, facts: request_headers.RequestFacts) -> None:
        """Emit `provider.request` for one HTTP attempt: its id and class, never a header value."""
        if self.events is not None:
            assert facts.context is not None and facts.context.call_context is not None
            self.events.emit(
                journal.ProviderRequest(
                    **facts.context.call_context.event_fields(),
                    role=self.role,
                    provider=self.provider_name,
                    model=facts.model,
                    request_id=facts.request_id,
                    target=facts.target,
                )
            )

    def close(self) -> None:
        """Release what the inner provider holds."""
        close_provider(self.inner)

    def _emit_budget(self) -> None:
        """Emit the cumulative spend after a call, whether it returned or raised.

        A stream cut after the provider reported usage is still billed, and these
        events are the only path spend takes to the cost meters and the machine's
        spend ledger.
        """
        if self.events is not None:
            self.events.emit(journal.BudgetUpdate(**budget_update_fields(self.budget)))


def budget_update_fields(budget: agent6_budget.BudgetTracker) -> dict[str, Any]:
    """Return a `budget.update` event's fields: the tracker's cumulative spend and plan reading."""
    snap = budget.snapshot()
    usd_total, usd_partial = budget.estimate_usd()
    account = None
    latest = next(reversed(snap.plans.items()), None)  # the last reading any provider gave
    if latest is not None:
        provider, spend = latest
        plan = spend.usage
        account = journal.AccountReading(
            provider=provider,
            used_percent=plan.used_percent,
            resets_at=plan.resets_at,
            windows=[
                journal.AccountWindow(
                    name=w.name,
                    used_percent=w.used_percent,
                    window_minutes=w.window_minutes,
                    resets_at=w.resets_at,
                )
                for w in plan.windows
            ],
            credits_has=plan.has_credits,
            credits_unlimited=plan.credits_unlimited,
            credits_balance=plan.credits_balance,
        )
    return dict(
        input_total=snap.input_total,
        output_total=snap.output_total,
        cache_read_total=snap.cache_read_total,
        cache_creation_total=snap.cache_creation_total,
        usd_total=usd_total,
        usd_partial=usd_partial,
        usd_cap=snap.max_usd,
        tokens_unmetered=snap.unmetered_tokens,
        tokens_fallback_cap=snap.max_tokens_fallback,
        plan_observed_points=snap.plan_observed_points,
        plan_cap=snap.max_plan_points,
        account=account,
    )


def helper_seat_provider(
    cfg: Config,
    seat: str,
    *,
    transcript_sink: TranscriptSink,
    budget: agent6_budget.BudgetTracker,
    events: event_log.EventSink | None,
    conversations: conversation_state.Conversations | None = None,
) -> Provider:
    """Build the helper route under a seat's label, instrumented.

    Args:
        cfg: The run's config.
        seat: The seat stamp: the summariser.
        transcript_sink: The recorder the provider's round-trips go to.
        budget: The tracker the provider bills.
        events: The sink the role and budget events go to.
        conversations: The session's conversation state; None holds one for this seat.

    Returns:
        The instrumented provider.
    """
    inner = build_role_provider(
        cfg, "helper", transcript_sink=transcript_sink, budget=budget, seat=seat, events=events
    )
    rm = cfg.models.resolve("helper")
    assert rm is not None
    return InstrumentedProvider(
        inner=inner,
        role=seat,
        model=rm.model,
        provider_name=rm.provider,
        events=events,
        budget=budget,
        conversations=conversations or conversation_state.Conversations(),
    )


def build_review_seats(
    cfg: Config,
    *,
    transcript_sink: TranscriptSink,
    budget: agent6_budget.BudgetTracker,
    events: event_log.EventSink | None = None,
    conversations: conversation_state.Conversations | None = None,
) -> list[review_gate.ReviewSeat]:
    """Build the review-panel seats, one per roster entry.

    `cfg.review.reviewers` names the roster; an empty roster builds one security seat.
    Each entry resolves its route and sampling settings before the provider is built.

    Args:
        cfg: The run's config.
        transcript_sink: The recorder the seats' round-trips go to.
        budget: The tracker the seats bill.
        events: The sink each seat's events go to; None leaves the seats uninstrumented,
            so their spend reaches no surface.
        conversations: The session's conversation state; None holds one per seat.

    Returns:
        The seats in roster order.

    Raises:
        ProviderError: An entry has no resolved route or names a missing provider.
    """
    cfg.require_runnable("reviewer")
    roster = cfg.review.reviewers or (ReviewerConfig(focus="security"),)
    reviewer = cfg.models.resolve("reviewer")
    resolved = [(entry.focus, entry.resolve(reviewer)) for entry in roster]
    for seat_focus, rm in resolved:
        if rm is None:
            raise ProviderError(
                f"reviewer {seat_focus!r} has no model; configure [models.reviewer]"
            )
        if rm.provider not in cfg.providers:
            raise ProviderError(
                f"reviewer {seat_focus!r} names provider {rm.provider!r} but"
                f" [providers.{rm.provider}] is missing"
            )
    seats: list[review_gate.ReviewSeat] = []
    for seat_index, (seat_focus, rm) in enumerate(resolved):
        assert rm is not None  # every route was checked before constructing any provider
        provider = _provider_from_entry(
            rm.provider,
            cfg.providers[rm.provider],
            rm.model,
            rm.effort,
            transcript_sink=transcript_sink.for_seat(f"review:{seat_focus}"),
            budget=budget,
            events=events,
        )
        if events is not None:
            provider = InstrumentedProvider(
                inner=provider,
                role=f"review:{seat_focus}",
                model=rm.model,
                provider_name=rm.provider,
                events=events,
                budget=budget,
                seat_index=seat_index,
                conversations=conversations or conversation_state.Conversations(),
            )
        seats.append(
            review_gate.ReviewSeat(
                focus=seat_focus,
                model=f"{rm.provider}/{rm.model}",
                provider=provider,
                scope=cfg.review.scope,
                temperature=rm.temperature,
                media_input=input_capabilities(cfg, rm),
            )
        )
    return seats
