# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pre-spawn model validation: catch a bogus model id before any run or lane
spawns, with a did-you-mean, instead of dying at the first provider call where
the raw upstream 400 leaks.

`validate_configured_model` checks a configured `models.<role>.model` at run
start; `validate_spec_models` checks a `/parallel` spec's per-lane routes,
each against its own provider (the models the roles name on it, unioned with
its listing).

Matching is cache-first: exact id, or the registry's normalization so a
dated/tagged variant of a listed id (`...-20251001`, `...:free`) passes. A
MISS against an existing cache fetches the provider's live listing once (TTL
bypassed, ~1.5s cap) before any hard stop: `refused` always rests on a listing
fetched by this invocation, so a just-pulled local model or a just-published
listing entry is never refused off a stale snapshot. A failed fetch (offline,
provider down) degrades the miss to `warned` and the run proceeds -- the first
provider call is the final arbiter. With no cache at all nothing is fetched and
nothing blocks (a fresh/offline machine, or a provider that lists no models).
Never raises.

Lives in the models layer so all three front-ends and the coordinator's group
dispatcher share one policy without a UI or workflows dependency.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from agent6.config import ClaudeCodeProviderEntry, Config, ConfigError, RoleName
from agent6.config.layer import load_effective
from agent6.directive import DirectiveError, Segment, parse_spec
from agent6.models.cache import cached_models, fetch_models_live
from agent6.models.registry import normalize_model_id
from agent6.secrets import SecretsError, load_secrets, resolve_api_key
from agent6.types import ModelRoute

ROLES: tuple[RoleName, ...] = ("worker", "reviewer", "planner")

__all__ = [
    "ModelValidation",
    "configured_model_refusal",
    "directive_model_refusal",
    "refusal_message",
    "validate_configured_model",
    "validate_spec_models",
    "warning_message",
]

_MAX_SUGGESTIONS = 3


def _known_models(cfg: Config, provider: str) -> set[str]:
    """The model ids *provider* is known to serve, without touching the
    network: the models the roles name on it, unioned with its on-disk
    model-list cache snapshot."""
    named = {
        rm.model
        for role in ROLES
        if (rm := cfg.models.resolve(role)) is not None and rm.provider == provider
    }
    return named | set(cached_models(provider))


@dataclass(frozen=True, slots=True)
class ModelValidation:
    """Outcome of a model check.

    `unknown` lists the named models not found (deduped, in spec order);
    `suggestions` maps each to its closest known ids; `can_validate` is True
    when the miss was judged against real evidence (a matching cache, or a
    listing fetched live by this invocation). `refused` (unknown +
    can_validate) is a hard stop resting on a just-fetched listing; `warned`
    (unknown + no fresh evidence: no cache, or the live re-fetch failed)
    proceeds -- an offline machine is never blocked on a regenerable cache."""

    unknown: tuple[str, ...]
    suggestions: dict[str, tuple[str, ...]]
    can_validate: bool

    @property
    def refused(self) -> bool:
        return bool(self.unknown) and self.can_validate

    @property
    def warned(self) -> bool:
        return bool(self.unknown) and not self.can_validate


def _fresh_listing(cfg: Config, provider_name: str) -> list[str] | None:
    """The provider's LIVE model listing, fetched now (TTL bypassed): the
    evidence a hard refusal needs. None when the fetch fails -- the caller
    degrades to the warn path rather than refuse on a snapshot it could not
    freshen. Keyless (local) providers list without auth; a secrets problem
    just means an unauthenticated attempt."""
    entry = cfg.providers.get(provider_name)
    if entry is None or isinstance(entry, ClaudeCodeProviderEntry):
        return None  # no listing: the binary resolves model names itself
    try:
        secrets = load_secrets()
    except SecretsError:
        secrets = {}
    key = resolve_api_key(provider_name, entry.api_key_env, secrets=secrets)
    return fetch_models_live(provider_name, entry, key)


def _matches(model: str, pool: set[str], norm_pool: set[str]) -> bool:
    """True when *model* is listed: exact id, or its normalized form matches a
    listed id's (a dated/tagged variant of a listed model is provider-plausible,
    so it must never hard-refuse; the call itself is the final arbiter)."""
    return model in pool or normalize_model_id(model) in norm_pool


def _close_ids(typo: str, pool: list[str], bare_to_full: dict[str, list[str]]) -> tuple[str, ...]:
    """Closest known ids to *typo*: matched against the full provider-prefixed ids
    AND against the un-prefixed model segment (the part after the last `/`). The
    bare match catches a short nickname near-miss (`glm`, `kimi-typo`) that scores
    below difflib's cutoff against a full id, because the provider prefix dominates
    the ratio (`glm` vs `z-ai/glm-4.6`). Bare hits map back to full ids (what the
    operator must actually pass); full-id hits keep priority, capped overall."""
    close = list(difflib.get_close_matches(typo, pool, n=_MAX_SUGGESTIONS))
    bare_typo = typo.rsplit("/", 1)[-1]
    for bare in difflib.get_close_matches(bare_typo, sorted(bare_to_full), n=_MAX_SUGGESTIONS):
        close.extend(full for full in bare_to_full[bare] if full not in close)
    return tuple(close[:_MAX_SUGGESTIONS])


def _suggest(unknown: list[str], pool: list[str]) -> dict[str, tuple[str, ...]]:
    """Did-you-mean suggestions for each unknown model, drawn from *pool*."""
    bare_to_full: dict[str, list[str]] = {}
    for full in pool:
        bare_to_full.setdefault(full.rsplit("/", 1)[-1], []).append(full)
    return {model: _close_ids(model, pool, bare_to_full) for model in unknown}


def validate_spec_models(routes: Sequence[ModelRoute | None], cfg: Config) -> ModelValidation:
    """Check per-lane *routes* (`None` = the worker's own route, skipped), each
    against its provider's `_known_models`. A miss against an existing cache
    re-checks that provider's live listing once before refusing (see module
    docstring); a miss on a provider with no cache is unvalidated. `unknown`
    names each route as provider/model. A confirmed miss refuses even when
    another route stays unvalidated."""
    misses: list[ModelRoute] = []
    for route in routes:
        if route is None or route in misses:
            continue
        known = _known_models(cfg, route.provider)
        if not _matches(route.model, known, {normalize_model_id(m) for m in known}):
            misses.append(route)
    unknown: list[str] = []
    unvalidated: list[str] = []
    suggestions: dict[str, tuple[str, ...]] = {}
    fresh_by_provider: dict[str, list[str] | None] = {}
    for route in misses:
        if not cached_models(route.provider):
            # No snapshot to judge against: proceed with a warning, never block
            # a fresh/offline machine (and no fetch attempt: keyed providers got
            # one in check_provider_keys; a fetchable listing would be cached).
            unvalidated.append(route.spec)
            continue
        if route.provider not in fresh_by_provider:
            fresh_by_provider[route.provider] = _fresh_listing(cfg, route.provider)
        fresh = fresh_by_provider[route.provider]
        if fresh is None:
            unvalidated.append(route.spec)
            continue
        pool = _known_models(cfg, route.provider) | set(fresh)
        if _matches(route.model, pool, {normalize_model_id(m) for m in pool}):
            continue
        unknown.append(route.spec)
        specs = sorted(f"{route.provider}/{m}" for m in pool)
        suggestions.update(_suggest([route.spec], specs))
    if unknown:
        return ModelValidation(unknown=tuple(unknown), suggestions=suggestions, can_validate=True)
    if unvalidated:
        return ModelValidation(unknown=tuple(unvalidated), suggestions={}, can_validate=False)
    return ModelValidation(unknown=(), suggestions={}, can_validate=True)


def validate_configured_model(cfg: Config, role: RoleName) -> ModelValidation:
    """Check the CONFIGURED model for *role* against ITS provider's listing, so a
    typo'd `models.<role>.model` is caught at run start.

    Unlike `validate_spec_models` the pool EXCLUDES the model itself -- a
    configured model is trivially in `_known_models`, so that check can never
    fail. A miss against an existing cache re-checks the live listing once;
    `refused` always rests on a listing fetched by this invocation, `warned`
    means the re-fetch failed (the caller prints it and proceeds). No cache at
    all (a fresh/offline machine, or a provider that lists no models) stays a
    silent proceed, with no fetch attempt."""
    rm = cfg.models.resolve(role)
    if rm is None:
        return ModelValidation(unknown=(), suggestions={}, can_validate=False)
    cache = set(cached_models(rm.provider))
    if not cache:
        return ModelValidation(unknown=(), suggestions={}, can_validate=False)
    if _matches(rm.model, cache, {normalize_model_id(c) for c in cache}):
        return ModelValidation(unknown=(), suggestions={}, can_validate=True)
    fresh = _fresh_listing(cfg, rm.provider)
    if fresh is None:
        return ModelValidation(unknown=(rm.model,), suggestions={}, can_validate=False)
    fresh_set = set(fresh)
    if _matches(rm.model, fresh_set, {normalize_model_id(c) for c in fresh_set}):
        return ModelValidation(unknown=(), suggestions={}, can_validate=True)
    return ModelValidation(
        unknown=(rm.model,),
        suggestions=_suggest([rm.model], sorted(fresh_set)),
        can_validate=True,
    )


def configured_model_refusal(v: ModelValidation, role: str) -> str:
    """Refusal text for a typo'd CONFIGURED role model (a refused
    `validate_configured_model`): name the bad model, its closest known ids, and
    how to fix it. The listing was re-fetched live before this refusal, so
    refreshing the cache cannot fix it."""
    model = v.unknown[0]
    close = v.suggestions.get(model, ())
    suffix = f" Closest: {', '.join(close)}." if close else ""
    return (
        f"configured models.{role}.model {model!r} is not in its provider's model"
        f" listing (checked live).{suffix} Fix it in your config."
    )


def flag_model_refusal(v: ModelValidation, cfg: Config, role: RoleName, spec: str) -> str:
    """Refusal text for a typo'd `--model` (a refused `validate_configured_model`
    whose model the flag set): name what was typed, the provider whose listing
    was checked live, the closest known ids, and, when the value's first
    segment names no configured provider, that it was read as one model id."""
    model = v.unknown[0]
    route = cfg.models.resolve(role)
    provider = route.provider if route is not None else ""
    close = v.suggestions.get(model, ())
    suffix = f" Closest: {', '.join(close)}." if close else ""
    head, slash, _rest = spec.strip().partition("/")
    misread = ""
    if slash and head and head not in cfg.providers:
        known = ", ".join(sorted(cfg.providers)) or "(none)"
        misread = (
            f" No provider is named {head!r} (configured: {known}), so the whole value was"
            f" read as a model id on {provider}."
        )
    return (
        f"--model {spec.strip()!r} is not in {provider}'s model listing (checked live)."
        f"{misread}{suffix} Pass one of its ids, or provider/model for another provider."
    )


def refusal_message(v: ModelValidation, *, directive: bool) -> str:
    """The refusal text for an `unknown + can_validate` result: one line per
    unknown model with its closest matches. On a directive surface (the composers
    and the coordinator, where the same token could be task text) add the backtick
    hint."""
    lines = [
        f"unknown model {model!r} in /parallel spec"
        + (f"; closest: {', '.join(close)}" if (close := v.suggestions.get(model, ())) else "")
        for model in v.unknown
    ]
    if directive:
        lines.append("backtick the word if you meant it as task text.")
    return "\n".join(lines)


def directive_model_refusal(
    cwd: Path,
    segments: Sequence[Segment],
    config_path: Path | None = None,
    *,
    preset: str = "",
    model: str = "",
) -> str | None:
    """Refuse a `/parallel` directive that names a model the configured
    providers' cache says doesn't exist, before any spawn (the surface's normal
    error path, nothing spawned). None = every model checks out, or there is no
    cache to check against (a fresh/offline machine proceeds; the detached
    lane's own preflight warns). A malformed or over-`max_lanes` spec surfaces
    its grammar error. *preset* and *model* are the new run's overrides, so
    validation uses the same worker route as the child."""
    try:
        cfg = load_effective(cwd, config_path, preset=preset).config
        if model:
            cfg = cfg.with_model_route("worker", cfg.model_route("worker", model))
    except ConfigError:
        return None  # a broken config is its own separate error; don't mask it here
    try:
        cap = cfg.parallel.max_lanes
        routes = [
            cfg.model_route("worker", m) if m else None
            for seg in segments
            for m in parse_spec(seg.spec, limit=cap)
        ]
    except (ConfigError, DirectiveError) as exc:
        return str(exc)
    verdict = validate_spec_models(routes, cfg)
    return refusal_message(verdict, directive=True) if verdict.refused else None


def warning_message(v: ModelValidation) -> str:
    """The single warning line for an `unknown + not can_validate` result: no
    fresh listing to check against (no cache, or the live re-fetch failed), so
    proceed but name the unvalidated model(s)."""
    return (
        f"unvalidated model(s) {', '.join(v.unknown)}: no fresh provider listing"
        " to check against; proceeding (run `agent6 model` to refresh)."
    )
