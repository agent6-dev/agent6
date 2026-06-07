# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""argcomplete completers for the CLI parser."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent6.cli._common import _machines_dir
from agent6.cli.connect import _CONNECT_PRESETS
from agent6.cli.model import _connected_providers, _models_for
from agent6.config import (
    ConfigError,
)
from agent6.config_layer import (
    leaf_keys,
    load_effective,
)


def _complete_providers(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: connected provider names + known presets."""
    names = set(_connected_providers(None)) | set(_CONNECT_PRESETS)
    return sorted(n for n in names if n.startswith(prefix))


def _complete_models(
    prefix: str, parsed_args: argparse.Namespace | None = None, **_kw: object
) -> list[str]:
    """argcomplete: live + configured model ids for the already-typed provider."""
    provider = getattr(parsed_args, "provider", "") or ""
    if not provider:
        return []
    return [m for m in _models_for(None, provider) if m.startswith(prefix)]


# Dotted config leaves whose type is a Literal/enum, with their allowed values.
# Used by the `config set/add/remove` value completer so TAB offers the exact
# valid choices (e.g. `config set sandbox.agent_network <TAB>` -> providers/...).
_CONFIG_ENUM_CHOICES: dict[str, tuple[str, ...]] = {
    "sandbox.profile": ("auto", "strict", "hardened"),
    "sandbox.agent_network": ("providers", "local", "open"),
    "sandbox.tool_network": ("block", "only_explicit_states", "allow"),
    "sandbox.run_commands": ("yes", "no", "ask"),
    "git.commit_strategy": ("per_step", "squash", "stage", "none"),
    "workflow.critic": ("off", "on_verify_fail", "before_finish", "periodic"),
    "workflow.revise_prompt": ("off", "auto", "interactive"),
    "models.worker.thinking": ("off", "low", "medium", "high"),
    "models.reviewer.thinking": ("off", "low", "medium", "high"),
    "models.planner.thinking": ("off", "low", "medium", "high"),
}


def _complete_config_keys(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: known dotted config leaf paths (effective + enum keys)."""
    try:
        keys = set(leaf_keys(load_effective(Path.cwd(), None)))
    except ConfigError:
        keys = set()
    keys |= set(_CONFIG_ENUM_CHOICES)
    return sorted(k for k in keys if k.startswith(prefix))


def _complete_config_values(
    prefix: str, parsed_args: argparse.Namespace | None = None, **_kw: object
) -> list[str]:
    """argcomplete: the Literal choices for the config key already typed."""
    key = getattr(parsed_args, "key", "") or ""
    return [v for v in _CONFIG_ENUM_CHOICES.get(key, ()) if v.startswith(prefix)]


def _complete_machine_files(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: machine ``*.asm.toml`` files under cwd and the machines dir."""
    out: set[str] = set()
    try:
        for base in (Path.cwd(), _machines_dir(Path.cwd())):
            if base.is_dir():
                out.update(str(p) for p in base.rglob("*.asm.toml"))
    except OSError:
        return []
    return sorted(p for p in out if p.startswith(prefix))
