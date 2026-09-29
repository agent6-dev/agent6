# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The values a config leaf can take when the schema alone cannot say.

A provider's model ids and the routes a `/parallel` lane may run, behind the config
editors' pickers, the web suggest route and TAB completion, so every surface offers the
same lists.
"""

from __future__ import annotations

import pathlib

from agent6 import kinds
from agent6 import secrets as agent6_secrets
from agent6.config import Config, ConfigError, layer
from agent6.models import cache, validate
from agent6.sessions import manifest as sessions_manifest


def provider_model_choices(cfg: Config, provider: str) -> list[str]:
    """Return the model ids a provider serves: the roles' models plus its listing.

    The listing is cache-first, refreshed live when stale. A broken secrets file degrades
    to a keyless attempt; the authoritative SecretsError fires at run setup. An
    unconfigured provider yields its cache alone.

    Args:
        cfg: The effective config.
        provider: The provider's config name.

    Returns:
        The ids, sorted.
    """
    out: set[str] = set()
    for role in validate.ROLES:
        rm = cfg.models.resolve(role)
        if rm is not None and rm.provider == provider:
            out.add(rm.model)
    entry = cfg.providers.get(provider)
    if entry is None:
        out.update(cache.cached_models(provider))
    else:
        try:
            secrets = agent6_secrets.load_secrets()
        except agent6_secrets.SecretsError:
            secrets = {}
        api_key = agent6_secrets.resolve_api_key(
            provider, getattr(entry, "api_key_env", None), secrets=secrets
        )
        out.update(cache.list_models(provider, entry, api_key))
    return sorted(out)


def route_choices(cfg: Config) -> list[str]:
    """Return every `provider/model` a run can be pointed at.

    Cache-only, so a picker never waits on the network; `agent6 model <role> <provider>`
    refreshes a listing.

    Args:
        cfg: The effective config.

    Returns:
        Each configured provider's cached ids plus the roles' models, sorted.
    """
    out: set[str] = set()
    for name in cfg.providers:
        out.update(f"{name}/{m}" for m in cache.cached_models(name))
    for role in validate.ROLES:
        rm = cfg.models.resolve(role)
        if rm is not None and rm.provider in cfg.providers:
            out.add(f"{rm.provider}/{rm.model}")
    return sorted(out)


def route_for(cfg: Config, mode: str) -> str:
    """Return the `provider/model` a session of a mode runs under, or "" when unset.

    Args:
        cfg: The effective config.
        mode: The session mode; its role resolves with the worker fallback.

    Returns:
        The route, or "".
    """
    rm = cfg.models.resolve(kinds.session_kind(mode).role)
    return f"{rm.provider}/{rm.model}" if rm is not None else ""


def available_routes(cwd: pathlib.Path, config_path: pathlib.Path | None) -> list[str]:
    """Return `route_choices` for the config a hub at a directory runs under.

    Args:
        cwd: The hub's directory.
        config_path: An explicit config file, or None.

    Returns:
        The routes, or [] on any config error.
    """
    try:
        cfg = layer.load_effective(cwd, config_path).config
    except ConfigError:
        return []
    return route_choices(cfg)


def default_preset(cwd: pathlib.Path, config_path: pathlib.Path | None) -> str:
    """Return the preset the config at a directory selects.

    Args:
        cwd: The hub's directory.
        config_path: An explicit config file, or None.

    Returns:
        The selected preset, or "" when none is selected or on any config error.
    """
    try:
        return layer.preset_catalog(cwd, config_path).selected
    except ConfigError:
        return ""


def default_route(
    cwd: pathlib.Path, config_path: pathlib.Path | None, mode: str, preset: str
) -> str:
    """Return the route a session of a mode runs under a preset from the config alone.

    Args:
        cwd: The hub's directory.
        config_path: An explicit config file, or None.
        mode: The session mode.
        preset: The preset to load under.

    Returns:
        The route, or "" on any config error or an unset role.
    """
    try:
        cfg = layer.load_effective(cwd, config_path, preset=preset).config
    except ConfigError:
        return ""
    return route_for(cfg, mode)


def default_label(name: str, *, recorded: bool = False) -> str:
    """Return the label of a picker's no-flag entry, such as `quick (config default)`.

    Args:
        name: What the entry runs under; "" reads as `none`.
        recorded: A resume replays the run's own flag, so the origin reads `as recorded`.

    Returns:
        The label.
    """
    return f"{name or 'none'} ({'as recorded' if recorded else 'config default'})"


def resume_defaults(
    cwd: pathlib.Path,
    config_path: pathlib.Path | None,
    session_dir: pathlib.Path,
    *,
    preset: str = "",
) -> tuple[str, str]:
    """Return the (preset, model) labels of a resume row's no-flag entries.

    A preset or model the run set by flag is replayed; anything else is what the config
    resolves now. An unreadable manifest names the config's.

    Args:
        cwd: The hub's directory.
        config_path: An explicit config file, or None.
        session_dir: The session to resume.
        preset: The preset picked in the row, or "".

    Returns:
        The preset label and the model label.
    """
    try:
        manifest = sessions_manifest.read_manifest(session_dir)
        mode: str = manifest.session_mode()
    except sessions_manifest.ManifestError:
        replayed, driver, mode = "", None, "run"
    else:
        replayed, driver = manifest.harness.replay_preset, manifest.models.replay_driver
    route = (
        f"{driver.provider}/{driver.model}"
        if driver is not None
        else default_route(cwd, config_path, mode, preset or replayed)
    )
    return (
        default_label(replayed or default_preset(cwd, config_path), recorded=bool(replayed)),
        default_label(route, recorded=driver is not None),
    )


def model_role_provider(eff: layer.EffectiveConfig, key: str) -> str | None:
    """Return the provider whose model ids a `models.<role>.model` leaf takes.

    Args:
        eff: The effective config.
        key: The config leaf.

    Returns:
        The provider's name, or None for any other leaf.
    """
    parts = key.split(".")
    if len(parts) != 3 or parts[0] != "models" or parts[2] != "model":
        return None
    role = getattr(eff.config.models, parts[1], None)
    return getattr(role, "provider", None) or None


def config_value_choices(eff: layer.EffectiveConfig, key: str) -> list[str]:
    """Return what a chooser offers for an open-text config leaf.

    Enum leaves carry their choices in the config view.

    Args:
        eff: The effective config.
        key: The leaf, or the pseudo-key `parallel.models` for a `/parallel` autocomplete.

    Returns:
        The role's provider's model ids, every lane route for `parallel.models`, else [].
    """
    if key == "parallel.models":
        return route_choices(eff.config)
    provider = model_role_provider(eff, key)
    return provider_model_choices(eff.config, provider) if provider else []
