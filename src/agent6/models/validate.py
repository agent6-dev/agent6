# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pre-spawn model validation: a bogus model id is refused with a did-you-mean.

`validate_configured_model` checks a configured `models.<role>.model` at run start;
`validate_spec_models` checks a `/parallel` spec's lane routes, each against its own
provider. Matching is cache-first, the id or its normalized form. A miss against an
existing cache fetches the live listing once, so `refused` always rests on a listing this
invocation fetched; a failed fetch degrades to `warned` and the run proceeds; with no cache
nothing is fetched and nothing blocks. Nothing here raises.
"""

from __future__ import annotations

import dataclasses
import difflib
import pathlib
from collections.abc import Sequence

from agent6 import directive as agent6_directive
from agent6 import kinds, secret_store
from agent6.config import ClaudeCodeProviderEntry, Config, ConfigError, RoleName, layer
from agent6.models import cache as models_cache
from agent6.models import registry

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
    """Return the ids the provider is known to serve: the roles' models plus its cache."""
    named = {
        rm.model
        for role in ROLES
        if (rm := cfg.models.resolve(role)) is not None and rm.provider == provider
    }
    return named | set(models_cache.cached_models(provider))


@dataclasses.dataclass(frozen=True, slots=True)
class ModelValidation:
    """The outcome of a model check.

    Attributes:
        unknown: The named models not found, deduped, in spec order.
        suggestions: Each unknown model's closest known ids.
        can_validate: The miss was judged against a cache or a listing fetched live now.
    """

    unknown: tuple[str, ...]
    suggestions: dict[str, tuple[str, ...]]
    can_validate: bool

    @property
    def refused(self) -> bool:
        """A hard stop: an unknown model judged against fresh evidence."""
        return bool(self.unknown) and self.can_validate

    @property
    def warned(self) -> bool:
        """A proceed with a warning: an unknown model with no fresh evidence."""
        return bool(self.unknown) and not self.can_validate


def _fresh_listing(cfg: Config, provider_name: str) -> list[str] | None:
    """Fetch the provider's live listing now, the evidence a refusal needs.

    A secrets problem means an unauthenticated attempt, as a keyless provider lists.

    Returns:
        The ids, or None when the fetch fails or the provider has no listing.
    """
    entry = cfg.providers.get(provider_name)
    if entry is None or isinstance(entry, ClaudeCodeProviderEntry):
        return None  # no listing: the binary resolves model names itself
    try:
        secrets = secret_store.load_secrets()
    except secret_store.SecretsError:
        secrets = {}
    key = secret_store.resolve_api_key(provider_name, entry.api_key_env, secrets=secrets)
    return models_cache.fetch_models_live(provider_name, entry, key)


def _matches(model: str, pool: set[str], norm_pool: set[str]) -> bool:
    """Return whether the model is listed, by exact id or normalized form."""
    return model in pool or registry.normalize_model_id(model) in norm_pool


def _close_ids(typo: str, pool: list[str], bare_to_full: dict[str, list[str]]) -> tuple[str, ...]:
    """Return the closest known ids to a typo.

    Matched against the full ids and against the segment after the last `/`: a short
    nickname (`glm`) scores below difflib's cutoff against `z-ai/glm-4.6`. Full-id hits keep
    priority.

    Args:
        typo: The unknown id.
        pool: The known ids.
        bare_to_full: Each bare segment's full ids.

    Returns:
        At most three full ids.
    """
    close = list(difflib.get_close_matches(typo, pool, n=_MAX_SUGGESTIONS))
    bare_typo = typo.rsplit("/", 1)[-1]
    for bare in difflib.get_close_matches(bare_typo, sorted(bare_to_full), n=_MAX_SUGGESTIONS):
        close.extend(full for full in bare_to_full[bare] if full not in close)
    return tuple(close[:_MAX_SUGGESTIONS])


def _suggest(unknown: list[str], pool: list[str]) -> dict[str, tuple[str, ...]]:
    """Return did-you-mean suggestions for each unknown model, drawn from the pool."""
    bare_to_full: dict[str, list[str]] = {}
    for full in pool:
        bare_to_full.setdefault(full.rsplit("/", 1)[-1], []).append(full)
    return {model: _close_ids(model, pool, bare_to_full) for model in unknown}


def validate_spec_models(routes: Sequence[kinds.ModelRoute | None], cfg: Config) -> ModelValidation:
    """Check a `/parallel` spec's lane routes, each against its own provider.

    A miss against an existing cache re-checks the live listing once; a miss on a provider
    with no cache is unvalidated. A confirmed miss refuses even when another route stays
    unvalidated.

    Args:
        routes: The lane routes; None is the worker's own route and is skipped.
        cfg: The effective config.

    Returns:
        The outcome, with each unknown route named as provider/model.
    """
    misses: list[kinds.ModelRoute] = []
    for route in routes:
        if route is None or route in misses:
            continue
        known = _known_models(cfg, route.provider)
        if not _matches(route.model, known, {registry.normalize_model_id(m) for m in known}):
            misses.append(route)
    unknown: list[str] = []
    unvalidated: list[str] = []
    suggestions: dict[str, tuple[str, ...]] = {}
    fresh_by_provider: dict[str, list[str] | None] = {}
    for route in misses:
        if not models_cache.cached_models(route.provider):
            # No snapshot to judge against: a fetchable listing would be cached by preflight.
            unvalidated.append(route.spec)
            continue
        if route.provider not in fresh_by_provider:
            fresh_by_provider[route.provider] = _fresh_listing(cfg, route.provider)
        fresh = fresh_by_provider[route.provider]
        if fresh is None:
            unvalidated.append(route.spec)
            continue
        pool = _known_models(cfg, route.provider) | set(fresh)
        if _matches(route.model, pool, {registry.normalize_model_id(m) for m in pool}):
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
    """Check the configured model of a role against its provider's listing.

    The pool excludes the model itself, which `_known_models` would list trivially. A miss
    against an existing cache re-checks the live listing once; no cache is a silent proceed.

    Args:
        cfg: The effective config.
        role: The role.

    Returns:
        The outcome; `warned` when the re-fetch failed.
    """
    rm = cfg.models.resolve(role)
    if rm is None:
        return ModelValidation(unknown=(), suggestions={}, can_validate=False)
    cache = set(models_cache.cached_models(rm.provider))
    if not cache:
        return ModelValidation(unknown=(), suggestions={}, can_validate=False)
    if _matches(rm.model, cache, {registry.normalize_model_id(c) for c in cache}):
        return ModelValidation(unknown=(), suggestions={}, can_validate=True)
    fresh = _fresh_listing(cfg, rm.provider)
    if fresh is None:
        return ModelValidation(unknown=(rm.model,), suggestions={}, can_validate=False)
    fresh_set = set(fresh)
    if _matches(rm.model, fresh_set, {registry.normalize_model_id(c) for c in fresh_set}):
        return ModelValidation(unknown=(), suggestions={}, can_validate=True)
    return ModelValidation(
        unknown=(rm.model,),
        suggestions=_suggest([rm.model], sorted(fresh_set)),
        can_validate=True,
    )


def configured_model_refusal(v: ModelValidation, role: str) -> str:
    """Return the refusal for a configured role model that its provider does not list.

    Args:
        v: A refused `validate_configured_model` outcome.
        role: The role.

    Returns:
        One line naming the model, its closest known ids and the fix.
    """
    model = v.unknown[0]
    close = v.suggestions.get(model, ())
    suffix = f" Closest: {', '.join(close)}." if close else ""
    return (
        f"configured models.{role}.model {model!r} is not in its provider's model"
        f" listing (checked live).{suffix} Fix it in your config."
    )


def flag_model_refusal(v: ModelValidation, cfg: Config, role: RoleName, spec: str) -> str:
    """Return the refusal for a `--model` value its provider does not list.

    Args:
        v: A refused `validate_configured_model` outcome for the flag's model.
        cfg: The effective config.
        role: The role the flag set.
        spec: The flag's value as typed.

    Returns:
        One line naming the value, the provider checked, the closest ids and, when the
        first segment names no configured provider, that the whole value was read as an id.
    """
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
    """Return the refusal for a `/parallel` spec, one line per unknown model.

    Args:
        v: A refused outcome.
        directive: The surface is a composer, where the token could be task text, so the
            backtick hint is added.

    Returns:
        The lines, joined.
    """
    lines = [
        f"unknown model {model!r} in /parallel spec"
        + (f"; closest: {', '.join(close)}" if (close := v.suggestions.get(model, ())) else "")
        for model in v.unknown
    ]
    if directive:
        lines.append("backtick the word if you meant it as task text.")
    return "\n".join(lines)


def directive_model_refusal(
    cwd: pathlib.Path,
    segments: Sequence[agent6_directive.Segment],
    config_path: pathlib.Path | None = None,
    *,
    preset: str = "",
    model: str = "",
) -> str | None:
    """Return the refusal for a `/parallel` directive naming an unlisted model, before any spawn.

    Args:
        cwd: The repo.
        segments: The directive's segments.
        config_path: An explicit config file, or None.
        preset: The new run's preset override, so validation uses the child's worker route.
        model: The new run's model override, likewise.

    Returns:
        The refusal, a malformed spec's grammar error, or None when every model checks out
        or there is no cache to check against (the lane's own preflight warns).
    """
    try:
        cfg = layer.load_effective(cwd, config_path, preset=preset).config
        if model:
            cfg = cfg.with_model_route("worker", cfg.model_route("worker", model))
    except ConfigError:
        return None  # a broken config is its own error elsewhere
    try:
        cap = cfg.parallel.max_lanes
        routes = [
            cfg.model_route("worker", m) if m else None
            for seg in segments
            for m in agent6_directive.parse_spec(seg.spec, limit=cap)
        ]
    except (ConfigError, agent6_directive.DirectiveError) as exc:
        return str(exc)
    verdict = validate_spec_models(routes, cfg)
    return refusal_message(verdict, directive=True) if verdict.refused else None


def warning_message(v: ModelValidation) -> str:
    """Return the warning line for a `warned` outcome, naming the unvalidated models."""
    return (
        f"unvalidated model(s) {', '.join(v.unknown)}: no fresh provider listing"
        " to check against; proceeding (run `agent6 model` to refresh)."
    )
