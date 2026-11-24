# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Construct role providers and resolve review-panel settings."""

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
from agent6.harness import provider_call
from agent6.harness.compaction import window
from agent6.harness.review import gate as review_gate
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
    anthropic,
    chatgpt,
    claude_code,
    oauth,
    openai,
    provider_types,
    request_headers,
    responses,
)
from agent6.sessions import budget_events, event_log, journal


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
        The `(drop_at_chars, summarize_at_chars, keep_recent_chars)` triple.
    """
    drop_override = cfg.compaction.drop_at_chars
    summarize_override = cfg.compaction.summarize_at_chars
    provider = rm.provider if rm is not None else ""
    model = rm.model if rm is not None else ""
    drop, summarize = models_registry.compaction_thresholds(
        provider,
        model,
        drop_override=drop_override,
        summarize_override=summarize_override,
    )
    endpoint = endpoint_compaction(cfg, rm)
    if log is not None:
        ctx = models_registry.context_window(provider, model) if model else None
        src = (
            f"adaptive from {model} (context {ctx:,} tok)"
            if ctx
            else "fixed default (context window unknown)"
        )
        key = models_registry.bundled_context_key(model)
        if key is not None and key not in (model, models_registry.normalize_model_id(model)):
            src += f"; family fallback from {key}"
        writer = "endpoint" if endpoint.writes_summary else "agent6"
        source = src if drop_override is None else "configured"
        log(
            f"compaction thresholds: {writer} drops at {drop:,} chars, "
            f"summarizes at {summarize:,} [{source}]"
        )
    keep = cfg.compaction.keep_recent_chars
    if not endpoint.writes_summary and summarize_override is None and keep > summarize // 2:
        keep = summarize // 2
        if log is not None:
            log(
                f"compaction.keep_recent_chars {cfg.compaction.keep_recent_chars:,} ->"
                f" {keep:,} chars: half this model's tier-2 threshold, so a restart"
                " shrinks the context instead of re-triggering on its own tail"
            )
    return drop, summarize, keep


def helper_context_window(cfg: Config) -> int | None:
    """Return the helper's known window without querying the provider.

    Args:
        cfg: The effective configuration.

    Returns:
        The context window in tokens, or None when unknown.
    """
    role = cfg.models.resolve("helper")
    return models_registry.context_window(role.provider, role.model) if role else None


def idle_compaction(cfg: Config, rm: RoleModel | None) -> window.IdleCompaction | None:
    """Read the driving provider's local idle limits.

    Args:
        cfg: The effective config.
        rm: The resolved driving role.

    Returns:
        The limits, or None without an HTTP provider entry.
    """
    if rm is None:
        return None
    entry = cfg.providers.get(rm.provider)
    if entry is None or isinstance(entry, ClaudeCodeProviderEntry):
        return None
    return window.IdleCompaction(rm.provider, entry.idle_compact_s, entry.idle_compact_min_chars)


def endpoint_compaction(cfg: Config, rm: RoleModel | None) -> provider_types.EndpointCompaction:
    """Return the history operations of the driving route.

    Args:
        cfg: The effective configuration.
        rm: The driving model, if configured.

    Returns:
        The facts supplied by the backend for this model.
    """
    if rm is None or (entry := cfg.providers.get(rm.provider)) is None:
        return provider_types.EndpointCompaction()
    if isinstance(entry, AnthropicProviderEntry):
        return anthropic.endpoint_compaction(rm.model, server_compaction=entry.server_compaction)
    if isinstance(entry, ClaudeCodeProviderEntry):
        return claude_code.ENDPOINT_COMPACTION
    if isinstance(entry, ChatGPTProviderEntry):
        return chatgpt.ENDPOINT_COMPACTION
    if isinstance(entry, ResponsesProviderEntry):
        return responses.ENDPOINT_COMPACTION
    return openai.ENDPOINT_COMPACTION


def resolve_decompose(
    cfg: Config, rm: RoleModel | None, *, log: Callable[[str], None] | None = None
) -> Config:
    """Pin `harness.prompt.decompose = "auto"` to on or off for this run.

    Auto turns on only when the model has a measured decompose win in the capability
    registry; an explicit on or off passes through untouched.

    Args:
        cfg: The run's config.
        rm: The worker model, or None when unresolved.
        log: Receives one line when auto turns on.

    Returns:
        The config with `harness.prompt.decompose` pinned.
    """
    if cfg.harness.prompt.decompose != "auto":
        return cfg
    on = rm is not None and models_registry.decompose_default(rm.model)
    if on and log is not None and rm is not None:
        log(f"decompose: auto-enabled for {rm.model} (measured win, bench/coreagent)")
    return cfg.with_decompose("on" if on else "off")


def input_capabilities(cfg: Config, rm: RoleModel | None) -> media.InputCapabilities:
    """Resolve the calling role's media facts without mutating its dispatcher.

    Args:
        cfg: The effective configuration.
        rm: The resolved worker, planner or reviewer.

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
    transcript_sink: TranscriptSink | None,
    budget: agent6_budget.BudgetTracker,
    actor: str = "",
    events: event_log.EventSink | None = None,
) -> Provider:
    """Construct the configured provider for a role.

    Every provider requires `models.default`, even when the role pins its own route.

    Args:
        cfg: The run's config.
        role: The role whose route to build.
        transcript_sink: The run's recorder, or None for calls that write no transcript.
        budget: The tracker the provider bills.
        actor: The actor to record. An empty value takes the role name.
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
        transcript_sink=transcript_sink.for_actor(actor or role)
        if transcript_sink is not None
        else None,
        budget=budget,
        events=events,
    )


def _provider_from_entry(
    provider_name: str,
    entry: Any,
    model: str,
    effort: EffortLevel | None,
    *,
    transcript_sink: TranscriptRecorder | None,
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
            service_tier=entry.wire_tier,
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
        service_tier=entry.wire_tier,
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
    fields: dict[str, int | float | None] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }
    if usage.cost_usd is not None:
        fields["cost_usd"] = usage.cost_usd
    return {key: value for key, value in fields.items() if value is not None}


@dataclasses.dataclass(frozen=True, slots=True)
class InstrumentedProvider:
    """Wrap a provider to emit the `model.*` call events and `budget.update`.

    Attributes:
        inner: The wrapped provider, unchanged.
        actor: The actor recorded on each model call.
        model: The model id the events carry.
        provider_name: The provider entry name the events carry.
        events: The sink the events go to; None emits nothing.
        budget: The tracker whose snapshot `budget.update` carries.
        reviewer_index: The reviewer's index, carried by its events; None outside a
            panel.
        stream_text: Whether to fan text and reasoning deltas out as events without a
            caller callback.
        conversations: The session's conversation state; a side call holds its own.
    """

    inner: Provider
    actor: str
    model: str
    provider_name: str
    events: event_log.EventSink | None
    budget: agent6_budget.BudgetTracker
    stream_text: bool = False
    reviewer_index: int | None = None
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
        compaction: provider_types.CompactionRequest | None = None,
        context: request_headers.RequestContext | None = None,
        web_search: bool = False,
    ) -> ProviderResponse:
        """Call the inner provider and emit the model and budget events around it.

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
            compaction: The endpoint operation, if requested.
            context: The caller's context; one without a conversation id names the initiator
                only, and the conversation supplies the rest; None is an agent-initiated call.
            web_search: Whether the call offers the provider's hosted web search.

        Returns:
            The inner provider's response.

        Raises:
            ProviderError: The inner call failed; the provider name is filled in.
        """
        call_context = (context.call_context if context is not None else None) or (
            kinds.CallContext(reviewer_index=self.reviewer_index)
        )
        caller_observe = context.observe if context is not None else None
        request_ids: list[str] = []  # the attempt's HTTP request, for its model.attempt row

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
                self.actor, self.provider_name, initiator=initiator, call_context=call_context
            )
        context = dataclasses.replace(
            context,
            call_context=call_context,
            observe=observe,
            attempt=attempt,
            attempts=attempts,
        )
        # Retries share the first attempt's model.call.
        if self.events is not None and context.attempt == 1:
            self.events.emit(journal.ModelCall(**self._identity(call_context)))
        # Every live view subscribes to the delta events; a caller callback chains through.
        actor = self.actor
        events = self.events
        identity = call_context.event_fields()
        stream = (
            self.stream_text
            or text_delta_callback is not None
            or thinking_delta_callback is not None
        )

        def _on_text(piece: str) -> None:
            if events is not None:
                events.emit(journal.ModelTextDelta(actor=actor, text=piece, **identity))
            if text_delta_callback is not None:
                text_delta_callback(piece)

        def _on_thinking(piece: str) -> None:
            if events is not None:
                events.emit(journal.ModelThinkingDelta(actor=actor, text=piece, **identity))
            if thinking_delta_callback is not None:
                thinking_delta_callback(piece)

        process_reply = False

        def on_reply() -> None:
            nonlocal process_reply
            process_reply = True

        # Only an option the caller set is named: not every provider takes each one.
        options: dict[str, Any] = {
            **({"compaction": compaction} if compaction is not None else {}),
            **({"web_search": True} if web_search else {}),
        }
        if isinstance(self.inner, ClaudeCodeProvider) and events is not None:
            options["on_reply"] = on_reply
        try:
            resp = self.inner.call(
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                text_delta_callback=_on_text if stream else None,
                thinking_delta_callback=_on_thinking if stream else None,
                should_abort=should_abort,
                should_interrupt=should_interrupt,
                context=context,
                **options,
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
                    **(exc.serving.event_fields() if isinstance(exc, ProviderError) else {}),
                    "error": str(exc)[:200],
                    **_usage_fields(exc.usage if isinstance(exc, ProviderError) else None),
                },
                ends_call=provider_call.closes_call(exc, context),
                process_reply=process_reply,
            )
            self._emit_budget()
            raise
        self.conversations.completed(self.actor, self.provider_name, call_context=call_context)
        self._record_searches(call_context, resp)
        self._record_attempt(
            call_context,
            context,
            request_ids,
            {
                "ok": True,
                "stop_reason": resp.stop_reason,
                **resp.serving.event_fields(),
                **_usage_fields(resp.usage),
            },
            ends_call=isinstance(compaction, provider_types.Summarize)
            or provider_call.settles_call(resp, context),
            # The settled prose: a headless run streams no deltas, so this is its only text.
            response=resp,
        )
        self._emit_budget()
        return resp

    def _record_searches(self, call_context: kinds.CallContext, resp: ProviderResponse) -> None:
        """Journal each hosted web search the response holds, ahead of its attempt's row."""
        if self.events is None:
            return
        for search in resp.web_searches:
            self.events.emit(
                journal.ModelWebSearch(
                    **self._identity(call_context),
                    search_id=search.id,
                    query=search.query,
                    status=search.status,
                    urls=search.urls,
                )
            )

    def _record_attempt(
        self,
        call_context: kinds.CallContext,
        context: request_headers.RequestContext,
        request_ids: list[str],
        outcome: dict[str, Any],
        *,
        ends_call: bool,
        response: ProviderResponse | None = None,
        process_reply: bool = False,
    ) -> None:
        """Journal one attempt's row and, when it ends the logical call, the call's result.

        Args:
            call_context: The call's identity.
            context: The attempt's request context, with its number.
            request_ids: The HTTP requests observed so far; the last is this attempt's.
            outcome: The attempt's `ok`, usage and stop reason or error.
            ends_call: Whether no retry of the call follows.
            response: The endpoint result, absent on a failure.
            process_reply: The process produced output before it failed.
        """
        if self.events is None:
            return
        identity = self._identity(call_context)
        reply: dict[str, Any] = {"process_reply": True} if process_reply else {}
        self.events.emit(
            journal.ModelAttempt(
                **identity,
                attempt=context.attempt,
                request_id=request_ids[-1] if request_ids else None,
                **reply,
                **outcome,
            )
        )
        if ends_call:
            settled: dict[str, Any] = {}
            if response is not None:
                settled["text"] = response.text
                if response.thinking_blocks_dropped:
                    settled["thinking_blocks_dropped"] = response.thinking_blocks_dropped
                if response.tool_results_cleared:
                    settled["tool_results_cleared"] = response.tool_results_cleared
            self.events.emit(journal.ModelResult(**identity, **outcome, **settled))

    def record_stopped_call(
        self,
        call_context: kinds.CallContext,
        reason: ProviderError,
        serving: kinds.Serving,
        usage: provider_types.ProviderUsage | None,
    ) -> None:
        """Close a call stopped between attempts without charging another attempt.

        Args:
            call_context: The logical call's identity.
            reason: The stop or steer that ended the wait.
            serving: The last attempt's serving facts.
            usage: The last attempt's reported usage, if any.
        """
        if self.events is not None:
            self.events.emit(
                journal.ModelResult(
                    **self._identity(call_context),
                    ok=False,
                    error=str(reason)[:200],
                    **serving.event_fields(),
                    **_usage_fields(usage),
                )
            )

    def _identity(self, call_context: kinds.CallContext) -> dict[str, Any]:
        """Return the fields that pair a `model.call` with its `model.result`."""
        return {
            "actor": self.actor,
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
                    actor=self.actor,
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
            budget_events.emit_budget_update(self.budget, self.events.emit)


def helper_actor_provider(
    cfg: Config,
    actor: str,
    *,
    transcript_sink: TranscriptSink,
    budget: agent6_budget.BudgetTracker,
    events: event_log.EventSink | None,
    conversations: conversation_state.Conversations | None = None,
) -> Provider:
    """Build the instrumented provider for a helper actor.

    Args:
        cfg: The run's config.
        actor: The actor to record.
        transcript_sink: The recorder the provider's round-trips go to.
        budget: The tracker the provider bills.
        events: The sink for model and budget events.
        conversations: The session's conversation state; None holds one for this actor.

    Returns:
        The instrumented provider.
    """
    inner = build_role_provider(
        cfg, "helper", transcript_sink=transcript_sink, budget=budget, actor=actor, events=events
    )
    rm = cfg.models.resolve("helper")
    assert rm is not None
    return InstrumentedProvider(
        inner=inner,
        actor=actor,
        model=rm.model,
        provider_name=rm.provider,
        events=events,
        budget=budget,
        conversations=conversations or conversation_state.Conversations(),
    )


def resolve_reviewers(cfg: Config) -> list[review_gate.Reviewer]:
    """Resolve the ordered roster without constructing providers.

    Omitted fields inherit the reviewer role, including reviewers with a pinned route.

    Args:
        cfg: The effective configuration.

    Returns:
        The roster, or one security reviewer when the roster is empty.

    Raises:
        ProviderError: A reviewer has no route or names a missing provider.
    """
    cfg.require_runnable("reviewer")
    reviewer = cfg.models.resolve("reviewer")
    reviewers: list[review_gate.Reviewer] = []
    for entry in cfg.review.reviewers or (ReviewerConfig(focus="security"),):
        rm = entry.resolve(reviewer)
        if rm is None:
            raise ProviderError(
                f"reviewer {entry.focus!r} has no model; configure [models.reviewer]"
            )
        if rm.provider not in cfg.providers:
            raise ProviderError(
                f"reviewer {entry.focus!r} names provider {rm.provider!r} but"
                f" [providers.{rm.provider}] is missing"
            )
        reviewers.append(
            review_gate.Reviewer(
                focus=entry.focus,
                settings=kinds.ModelSettings(
                    kinds.ModelRoute(rm.provider, rm.model), rm.effort, rm.temperature
                ),
            )
        )
    return reviewers
