# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""argcomplete completers for the CLI parser."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent6.cli._common import _machines_dir, _runs_dir
from agent6.cli.model import _connected_providers, _models_for
from agent6.config import (
    ConfigError,
)
from agent6.config_layer import (
    PROVIDER_PRESETS,
    leaf_keys,
    load_effective,
)


def _complete_providers(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: connected provider names + known presets."""
    names = set(_connected_providers(None)) | set(PROVIDER_PRESETS)
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
    # `sandbox.profile` also accepts "none" (the unsandboxed opt-out, see
    # config.SandboxConfig.profile), deliberately omitted here: TAB should not put
    # "disable the sandbox" one keystroke away. Type it explicitly to set it.
    "sandbox.profile": ("auto", "strict", "hardened"),
    "sandbox.agent_network": ("providers", "local", "open"),
    "sandbox.tool_network": ("block", "only_explicit_states", "allow"),
    "sandbox.run_commands": ("yes", "no", "ask"),
    "git.merge_strategy": ("squash", "merge", "ff"),
    "review.trigger": ("off", "on_verify_fail", "before_finish", "periodic"),
    "prompt.revise_prompt": ("off", "auto", "interactive"),
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


# Presets offered for any `providers.<name>.extra_body` value (the provider name
# varies, so this is matched by suffix, not in _CONFIG_ENUM_CHOICES). The first
# is the recommended OpenRouter routing, a fast, prefix-caching backend.
_EXTRA_BODY_PRESETS: tuple[str, ...] = (
    '{ provider = { sort = "throughput" } }',
    '{ provider = { sort = "latency" } }',
    '{ provider = { sort = "price" } }',
)


def _complete_config_values(
    prefix: str, parsed_args: argparse.Namespace | None = None, **_kw: object
) -> list[str]:
    """argcomplete: the Literal choices for the config key already typed."""
    key = getattr(parsed_args, "key", "") or ""
    choices = list(_CONFIG_ENUM_CHOICES.get(key, ()))
    if key.endswith(".extra_body"):
        choices += list(_EXTRA_BODY_PRESETS)
    return [v for v in choices if v.startswith(prefix)]


def _complete_model_provider(
    prefix: str, parsed_args: argparse.Namespace | None = None, **_kw: object
) -> list[str]:
    """argcomplete for ``agent6 model <role> <provider>``.

    Only offer provider names once a valid role has been typed. argcomplete
    bleeds every nargs='?' positional's completer into the first slot, so
    without this gate `agent6 model <TAB>` would mix provider names into the
    role choices (and `agent6 model openrouter` then fails the role validator).
    """
    role = getattr(parsed_args, "role", None)
    if role not in ("planner", "worker", "reviewer", "all"):
        return []
    return _complete_providers(prefix)


def _complete_run_ids(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: run ids (directory names under the per-repo run-state dir)."""
    try:
        runs = _runs_dir(Path.cwd())
        if not runs.is_dir():
            return []
        return sorted(p.name for p in runs.iterdir() if p.is_dir() and p.name.startswith(prefix))
    except (OSError, ConfigError):
        return []


def _complete_plan_run_ids(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: run ids that hold a plan.md (for --from-plan / plan show/edit)."""
    try:
        runs = _runs_dir(Path.cwd())
        if not runs.is_dir():
            return []
        return sorted(
            p.name
            for p in runs.iterdir()
            if p.is_dir() and p.name.startswith(prefix) and (p / "plan.md").is_file()
        )
    except (OSError, ConfigError):
        return []


def _complete_machine_ids(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: live machine instance ids (dirs under the per-repo state dir's machines/)."""
    try:
        base = _machines_dir(Path.cwd())
        if not base.is_dir():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix))
    except (OSError, ConfigError):
        return []


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
