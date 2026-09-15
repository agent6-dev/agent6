# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The values a config leaf can take when the schema alone cannot say: a
provider's model ids, the ids a `/parallel` lane may run. One owner behind
the config editors' pickers, the web suggest route, and TAB completion, so
the surfaces offer the same lists."""

from __future__ import annotations

from pathlib import Path

from agent6.config import Config, ConfigError
from agent6.config.layer import EffectiveConfig, load_effective
from agent6.models.cache import cached_models, list_models
from agent6.models.validate import ROLES
from agent6.secrets import SecretsError, load_secrets, resolve_api_key
from agent6.types import session_kind


def provider_model_choices(cfg: Config, provider: str) -> list[str]:
    """Model ids for *provider*: the ones a role already names on it, unioned
    with the provider's listing (cache-first, refreshed from the live listing
    when stale; the fetch dials only that operator-configured base_url). A
    broken secrets file degrades to a keyless attempt, never a raise: this is
    a convenience list, and the authoritative SecretsError fires at run setup.
    An unconfigured provider yields its cache alone."""
    out: set[str] = set()
    for role in ROLES:
        rm = cfg.models.resolve(role)
        if rm is not None and rm.provider == provider:
            out.add(rm.model)
    entry = cfg.providers.get(provider)
    if entry is None:
        out.update(cached_models(provider))
    else:
        try:
            secrets = load_secrets()
        except SecretsError:
            secrets = {}
        api_key = resolve_api_key(provider, getattr(entry, "api_key_env", None), secrets=secrets)
        out.update(list_models(provider, entry, api_key))
    return sorted(out)


def route_choices(cfg: Config) -> list[str]:
    """Every `provider/model` a run can be pointed at (`--model` and the
    composers' model picker): each configured provider's cached listing plus
    the models the roles name on it. Cache-only, so a picker never waits on
    the network; `agent6 model <role> <provider>` refreshes a listing."""
    out: set[str] = set()
    for name in cfg.providers:
        out.update(f"{name}/{m}" for m in cached_models(name))
    for role in ROLES:
        rm = cfg.models.resolve(role)
        if rm is not None and rm.provider in cfg.providers:
            out.add(f"{rm.provider}/{rm.model}")
    return sorted(out)


def route_for(cfg: Config, mode: str) -> str:
    """The `provider/model` a session of *mode* runs under this config (the
    mode's role, worker fallback applied), or "" when the role is unset."""
    rm = cfg.models.resolve(session_kind(mode).role)
    return f"{rm.provider}/{rm.model}" if rm is not None else ""


def available_routes(cwd: Path, config_path: Path | None) -> list[str]:
    """`route_choices` for the config a hub at *cwd* runs under; empty on any
    config error (the hub's other lists degrade the same way)."""
    try:
        cfg = load_effective(cwd, config_path).config
    except ConfigError:
        return []
    return route_choices(cfg)


def default_route(cwd: Path, config_path: Path | None, mode: str, preset: str) -> str:
    """The `provider/model` a session of *mode* under *preset* runs from the
    config alone: what a hub's model picker shows until the operator picks
    another. Empty on any config error or an unset role."""
    try:
        cfg = load_effective(cwd, config_path, preset=preset).config
    except ConfigError:
        return ""
    return route_for(cfg, mode)


def model_role_provider(eff: EffectiveConfig, key: str) -> str | None:
    """The provider whose model ids a `models.<role>.model` leaf takes, else
    None. The TUI's editor and `config_value_choices` decide on it alike."""
    parts = key.split(".")
    if len(parts) != 3 or parts[0] != "models" or parts[2] != "model":
        return None
    role = getattr(eff.config.models, parts[1], None)
    return getattr(role, "provider", None) or None


def config_value_choices(eff: EffectiveConfig, key: str) -> list[str]:
    """What a chooser offers for an open-text config leaf: `models.<role>.model`
    is that role's provider's model ids; the pseudo-key `parallel.models` (a
    composer's `/parallel` autocomplete) is every `provider/model` a lane may
    run (`route_choices`, cache-only, so a keystroke never waits on the
    network). Enum leaves carry their choices in the config view; anything
    else offers nothing."""
    if key == "parallel.models":
        return route_choices(eff.config)
    provider = model_role_provider(eff, key)
    return provider_model_choices(eff.config, provider) if provider else []
