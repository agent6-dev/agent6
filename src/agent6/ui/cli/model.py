# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 model`, show/set role models, with interactive prefill."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

from agent6.config import (
    Config,
    ConfigError,
    RoleName,
)
from agent6.config.layer import (
    load_effective,
    repo_config_path_for,
    set_config_table,
)
from agent6.models.cache import list_models
from agent6.paths import global_config_path
from agent6.secrets import resolve_api_key


def _safe_input(prompt: str) -> str | None:
    """``input`` that returns None instead of raising on EOF / non-interactive stdin."""
    try:
        return input(prompt).strip()
    except (EOFError, OSError):
        return None


def _connected_providers(config_path: Path | None) -> list[str]:
    """Provider names declared in the effective config (empty on any error)."""
    try:
        eff = load_effective(Path.cwd(), config_path)
    except ConfigError:
        return []
    return sorted(eff.config.providers)


def _configured_models_for(cfg: Config, provider: str) -> list[str]:
    """Models already assigned to *provider* across the three roles."""
    out: set[str] = set()
    roles: tuple[RoleName, ...] = ("worker", "reviewer", "planner")
    for role in roles:
        rm = cfg.models.resolve(role)
        if rm is not None and rm.provider == provider:
            out.add(rm.model)
    return sorted(out)


def _models_for(config_path: Path | None, provider: str) -> list[str]:
    """Known model ids for *provider*: configured ones unioned with the live list."""
    try:
        eff = load_effective(Path.cwd(), config_path)
    except ConfigError:
        return []
    options = set(_configured_models_for(eff.config, provider))
    entry = eff.config.providers.get(provider)
    if entry is not None:
        api_key = resolve_api_key(provider, entry.api_key_env)
        options.update(list_models(provider, entry, api_key))
    return sorted(options)


def _prompt_for_provider(config_path: Path | None) -> str:
    """Interactively pick a provider, defaulting to the first connected one."""
    providers = _connected_providers(config_path)
    if providers:
        print("Connected providers: " + ", ".join(providers))
        default = providers[0]
        choice = _safe_input(f"Provider [{default}]: ")
        if choice is None:
            return ""
        return choice or default
    print("No providers connected yet; run `agent6 connect` first, or type a name.")
    return _safe_input("Provider: ") or ""


def _prompt_for_model(config_path: Path | None, provider: str) -> str:
    """Interactively pick a model for *provider* from the live/configured list."""
    options = _models_for(config_path, provider)
    if options:
        print(f"Models for {provider}:")
        for i, model in enumerate(options, 1):
            print(f"  {i:>2}. {model}")
        choice = _safe_input("Model (name or number): ")
        if choice is None:
            return ""
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        return choice
    print(f"No known models for {provider} (couldn't reach its API or none configured).")
    return _safe_input("Model: ") or ""


def _cmd_model(
    config_path: Path | None,
    *,
    role: str | None,
    provider: str,
    model: str,
    thinking: str,
    to_repo: bool,
) -> int:
    """Show or set the model + thinking level for a role."""
    if not role:
        try:
            eff = load_effective(Path.cwd(), config_path)
        except ConfigError as exc:
            print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
            return 2
        print("Role assignments (planner/reviewer fall back to worker when unset):\n")
        show_roles: tuple[RoleName, ...] = ("planner", "worker", "reviewer")
        for r in show_roles:
            rm = eff.config.models.resolve(r)
            src = eff.sources.get(f"models.{r}.model", "default")
            if rm is None:
                print(f"  {r:<9} (unset)")
            else:
                think = rm.thinking or "-"
                print(f"  {r:<9} {rm.provider}/{rm.model}  thinking={think}  [{src}]")
        print(
            "\nSet one with: agent6 model worker <provider> <model>"
            " [--thinking low|medium|high]  (provider/model are prompted if omitted)"
        )
        return 0
    # `role` is validated by argparse `choices`: planner/worker/reviewer or the
    # pseudo-role "all" (no config field of that name, it expands to all three).
    # Positional provider/model are optional: prompt interactively when blank,
    # prefilling the provider list from connected providers and the model list
    # from that provider's live/configured catalog.
    if not provider:
        provider = _prompt_for_provider(config_path)
    if not provider:
        print("ERROR: no provider given.", file=sys.stderr)
        return 2
    if not model:
        model = _prompt_for_model(config_path, provider)
    if not model:
        print("ERROR: no model given.", file=sys.stderr)
        return 2
    target = repo_config_path_for(Path.cwd()) if to_repo else global_config_path()
    fields: dict[str, str | bool | None] = {"provider": provider, "model": model}
    if thinking:
        fields["thinking"] = thinking
    roles: tuple[RoleName, ...] = (
        ("planner", "worker", "reviewer") if role == "all" else (cast("RoleName", role),)
    )
    # Write through the shared edit path: each [models.<role>] table is persisted,
    # the merged config re-validated, and the file ROLLED BACK if the combination
    # is invalid -- so a bad provider/model never leaves config.toml broken (which
    # would fail every later command). The roles get identical fields, so the first
    # rejection rolls back with nothing partially applied.
    for r in roles:
        err = set_config_table(Path.cwd(), f"models.{r}", fields, to_repo=to_repo)
        if err is not None:
            print(
                f"Refusing: {provider}/{model} would make the config invalid:\n{err}",
                file=sys.stderr,
            )
            return 2
    where = "[models.*] (all roles)" if role == "all" else f"[models.{role}]"
    print(
        f"Set {where} = {provider}/{model}"
        f"{f' (thinking={thinking})' if thinking else ''} in {target}."
    )
    return 0
