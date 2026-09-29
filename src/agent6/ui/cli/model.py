# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 model`: show or set role models, with an interactive prefill.

A piped invocation naming no model lists the provider's catalog instead, one id per line.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

from agent6.config import (
    ClaudeCodeProviderEntry,
    ConfigError,
    RoleName,
)
from agent6.config.layer import load_effective
from agent6.config.write import ConfigLeafValue, set_config_table
from agent6.models.choices import provider_model_choices
from agent6.paths import global_config_path, repo_config_path
from agent6.providers.claude_code import login_status
from agent6.secrets import load_oauth_tokens, resolve_api_key
from agent6.ui.cli._common import error, refuse, safe_input, warn


def _connected_providers(config_path: Path | None) -> list[str]:
    """Return the provider names the effective config declares; empty on any error."""
    try:
        eff = load_effective(Path.cwd(), config_path)
    except ConfigError:
        return []
    return sorted(eff.config.providers)


def _models_for(config_path: Path | None, provider: str) -> list[str]:
    """Return the known model ids for a provider; empty when the config does not load."""
    try:
        eff = load_effective(Path.cwd(), config_path)
    except ConfigError:
        return []
    return provider_model_choices(eff.config, provider)


def _prompt_for_provider(config_path: Path | None) -> str:
    """Return the provider picked interactively, defaulting to the first connected one."""
    providers = _connected_providers(config_path)
    if providers:
        print("Connected providers: " + ", ".join(providers))
        default = providers[0]
        choice = safe_input(f"Provider [{default}]: ")
        if choice is None:
            return ""
        return choice or default
    print("No providers connected yet; run `agent6 connect` first, or type a name.")
    return safe_input("Provider: ") or ""


def _prompt_for_model(config_path: Path | None, provider: str) -> str | None:
    """Return the model picked interactively from the provider's list.

    Returns:
        The model, "" when nothing was typed, or None after a refusal it printed (a number
        past the list).
    """
    options = _models_for(config_path, provider)
    if options:
        print(f"Models for {provider}:")
        for i, model in enumerate(options, 1):
            print(f"  {i:>2}. {model}")
        choice = safe_input("Model (name or number): ")
        if choice is None:
            return ""
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
            error(f"no model {choice}: the list has {len(options)}.")
            return None
        return choice
    print(f"No known models for {provider} (couldn't reach its API or none configured).")
    return safe_input("Model: ") or ""


def _show_assignments(config_path: Path | None) -> int:
    """Print the three role assignments with their config origin.

    Returns:
        The exit code, 0.
    """
    eff = load_effective(Path.cwd(), config_path)
    print("Role assignments (planner/reviewer fall back to worker when unset):\n")
    show_roles: tuple[RoleName, ...] = ("planner", "worker", "reviewer")
    for r in show_roles:
        rm = eff.config.models.resolve(r)
        source = eff.config.models.source_role(r)
        src = eff.sources.get(f"models.{source}.model", "default")
        if rm is None:
            print(f"  {r:<9} (unset)")
        else:
            effort = rm.effort or "-"
            origin = src if source == r else f"worker's, {src}"
            print(f"  {r:<9} {rm.provider}/{rm.model}  effort={effort}  [{origin}]")
    print(
        "\nSet one with: agent6 model worker provider/model"
        " [--effort low|medium|high|xhigh|max]  (prompted if omitted on a terminal)"
    )
    return 0


def _print_catalog(config_path: Path | None, role: str, provider: str) -> int:
    """Print the provider's model ids, one per line, with the set hint on stderr.

    The listing for a piped invocation naming no model: the one non-interactive way to
    discover model ids, for a `--parallel` spec among others.

    Args:
        config_path: The `--config` file, if any.
        role: The role the hint names.
        provider: The provider whose catalog to print.

    Returns:
        The exit code, 0.
    """
    options = _models_for(config_path, provider)
    if not options:
        error(
            f"no known models for {provider}: no listing reached and no role names one."
            f" Set one with: agent6 model {role} {provider}/<model>"
        )
        return 2
    for m in options:
        print(m)
    print(f"set one with: agent6 model {role} {provider}/<model>", file=sys.stderr)
    return 0


def _warn_unusable_provider(config_path: Path | None, provider: str) -> None:
    """Warn when a set names a keyless provider: config accepts it, the first run would refuse."""
    try:
        eff = load_effective(Path.cwd(), config_path)
    except ConfigError:
        return
    entry = eff.config.providers.get(provider)
    if entry is None:
        warn(f"provider {provider!r} is not configured; run `agent6 connect` first.")
        return
    if isinstance(entry, ClaudeCodeProviderEntry):
        if (err := login_status(entry.binary)) is not None:
            warn(f"provider {provider!r}: {err}")
        return
    if entry.auth_style == "none" or entry.token_command:
        return
    if entry.api_format == "chatgpt":
        if load_oauth_tokens(provider) is None:
            warn(
                f"provider {provider!r} has no ChatGPT sign-in;"
                f" run `agent6 connect {provider}` before using it."
            )
        return
    if resolve_api_key(provider, entry.api_key_env) is None:
        remedy = (
            f"export {entry.api_key_env} or run `agent6 connect`"
            if entry.api_key_env
            else "run `agent6 connect`"
        )
        warn(f"provider {provider!r} has no stored API key; {remedy} before using it.")


def _read_route(
    config_path: Path | None, role: str, route: str, *, interactive: bool
) -> tuple[str, str]:
    """Return `(provider, model)` from a `[PROVIDER/]MODEL` value.

    A configured provider at the first slash names the provider; any other slash stays in
    the model id on the role's current provider. A configured provider's name alone leaves
    the model to pick. A blank value prompts for the provider on a terminal.

    Args:
        config_path: The `--config` file, if any.
        role: The role being set.
        route: The value as typed.
        interactive: Both channels are a terminal.

    Returns:
        The provider and the model; the model is "" when left to pick.

    Raises:
        ConfigError: The value names nothing.
    """
    raw_route = route
    route = route.strip()
    if not route:
        if raw_route:
            raise ConfigError(f"{raw_route!r}: no model id.")
        provider = _prompt_for_provider(config_path) if interactive else ""
        if not provider:
            raise ConfigError("no provider given: name it as provider/model.")
        return provider, ""
    cfg = load_effective(Path.cwd(), config_path).config
    if "/" not in route and route in cfg.providers:
        return route, ""
    if not cfg.providers and "/" in route:
        provider, model = route.split("/", 1)
        if not provider:
            raise ConfigError(f"{route!r}: name the provider as provider/model.")
        if not model:
            raise ConfigError(f"{route!r}: no model id after the slash.")
        return provider, model
    route_role = cast("RoleName", "worker" if role == "all" else role)
    if "/" not in route and cfg.models.resolve(route_role) is None:
        known = ", ".join(sorted(cfg.providers)) or "(none)"
        raise ConfigError(
            f"{route!r}: no provider is set for {role}, so name one as provider/model"
            f" (configured providers: {known})."
        )
    parsed = cfg.model_route(route_role, route)
    return parsed.provider, parsed.model


def _cmd_model(
    config_path: Path | None,
    *,
    role: str | None,
    route: str,
    effort: str,
    to_repo: bool,
) -> int:
    """Show or set the model and reasoning effort for a role.

    Args:
        config_path: The `--config` file, if any.
        role: `planner`, `worker`, `reviewer` or `all`; None shows the assignments.
        route: The `[PROVIDER/]MODEL` value; "" prompts or lists.
        effort: The reasoning effort to set.
        to_repo: Write to the repo config instead of the global one.

    Returns:
        The exit code; 2 on a refused value.
    """
    if not role:
        return _show_assignments(config_path)
    # Interactive means both channels are a tty: a piped stdout gets the listing, not a prompt.
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    try:
        provider, model = _read_route(config_path, role, route, interactive=interactive)
    except ConfigError as exc:
        error(str(exc))
        return 2
    if not model and not interactive:
        if effort:
            # The flag only means something for a set; a silent drop would read as applied.
            print("note: --effort ignored (no model named; this is a listing).", file=sys.stderr)
        return _print_catalog(config_path, role, provider)
    if not model:
        picked = _prompt_for_model(config_path, provider)
        if not picked:
            if picked == "":
                error("no model given.")
            return 2
        model = picked
    target = repo_config_path(Path.cwd()) if to_repo else global_config_path()
    fields: dict[str, ConfigLeafValue] = {"provider": provider, "model": model}
    if effort:
        fields["effort"] = effort
    roles: tuple[RoleName, ...] = (
        ("planner", "worker", "reviewer") if role == "all" else (cast("RoleName", role),)
    )
    # The shared edit path re-validates and rolls back, so a bad route never breaks config.toml.
    for r in roles:
        err = set_config_table(Path.cwd(), f"models.{r}", fields, to_repo=to_repo)
        if err is not None:
            refuse(f"{provider}/{model} would make the config invalid:\n{err}")
            return 2
    where = "[models.*] (all roles)" if role == "all" else f"[models.{role}]"
    print(f"Set {where} = {provider}/{model}{f' (effort={effort})' if effort else ''} in {target}.")
    _warn_unusable_provider(config_path, provider)
    return 0
