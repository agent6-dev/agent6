# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""argcomplete completers for the CLI parser."""

from __future__ import annotations

import argparse
import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent6.app.resume import resumable_bucket_dirs
from agent6.config import (
    ConfigError,
)
from agent6.config.layer import (
    available_preset_names,
    leaf_keys,
    load_effective,
    preset_catalog,
)
from agent6.config.write import PROVIDER_DEFAULTS
from agent6.ui.cli._common import (
    _machines_dir,
    _plans_dir,
    _state_dir,
    session_bucket_dirs,
)
from agent6.ui.cli.model import _connected_providers, _models_for
from agent6.ui.cli.skills_cmds import resolved_skill_names_for_completion


def _never_raises(fn: Callable[..., list[str]]) -> Callable[..., list[str]]:
    """Suggestions or nothing -- never an exception.

    argcomplete calls these on Tab, inside the operator's shell, where an
    exception is a traceback dumped over the command line. Every completer that
    touches the config or the filesystem wears this, instead of each growing
    its own try/except and several never growing one.
    """

    @functools.wraps(fn)
    def guarded(*args: Any, **kwargs: Any) -> list[str]:
        try:
            return fn(*args, **kwargs)
        except (OSError, ConfigError, ValueError):
            return []

    return guarded


@_never_raises
def _complete_providers(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: connected provider names + known presets."""
    names = set(_connected_providers(None)) | set(PROVIDER_DEFAULTS)
    return sorted(n for n in names if n.startswith(prefix))


@_never_raises
def _complete_presets(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: built-in presets + configured [presets.*] names."""
    return [n for n in available_preset_names(Path.cwd()) if n.startswith(prefix)]


@_never_raises
def _complete_skills(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: installed + extra_dirs skill names."""
    return [n for n in resolved_skill_names_for_completion(Path.cwd()) if n.startswith(prefix)]


@_never_raises
def _complete_models(
    prefix: str, parsed_args: argparse.Namespace | None = None, **_kw: object
) -> list[str]:
    """argcomplete: live + configured model ids for the already-typed provider."""
    provider = getattr(parsed_args, "provider", "") or ""
    if not provider:
        return []
    return [m for m in _models_for(None, provider) if m.startswith(prefix)]


def _all_parallel_model_names() -> list[str]:
    """Model ids a `/parallel` lane can actually run: the WORKER provider's
    catalog (lanes inherit the worker provider; only the model is overridden per
    lane), from the same live + configured source `agent6 model` completes from."""
    try:
        eff = load_effective(Path.cwd(), None)
    except ConfigError:
        return []
    worker = eff.config.models.worker
    if worker is None:
        return []
    return sorted(set(_models_for(None, worker.provider)))


@_never_raises
def _complete_parallel_models(prefix: str, **_kw: object) -> list[str]:
    """argcomplete for `run --parallel`: the worker provider's model ids,
    completing the token after the last comma so a `m1,m2,...` list completes
    member by member (an integer lane count is typed, not completed)."""
    head, sep, frag = prefix.rpartition(",")
    lead = head + sep  # "" for the first/only model, "m1," while extending a list
    return sorted(lead + m for m in _all_parallel_model_names() if m.startswith(frag))


# Dotted config leaves whose type is a Literal/enum, with their allowed values.
# Used by the `config set/add/remove` value completer so TAB offers the exact
# valid choices (e.g. `config set sandbox.tool_network <TAB>` -> auto/block/...).
_CONFIG_ENUM_CHOICES: dict[str, tuple[str, ...]] = {
    # `sandbox.isolation` also accepts "none" (the unsandboxed opt-out, see
    # config.Config.preset), deliberately omitted here: TAB should not put
    # "disable the sandbox" one keystroke away. Type it explicitly to set it.
    "sandbox.isolation": ("auto", "strict", "hardened"),
    "sandbox.tool_network": ("auto", "block", "only_explicit_states", "allow"),
    "sandbox.run_commands": ("yes", "no", "ask"),
    "git.merge_strategy": ("squash", "merge", "ff"),
    "review.trigger": ("off", "on_verify_fail", "before_finish", "periodic"),
    "prompt.revise_prompt": ("off", "auto", "interactive"),
    "models.worker.thinking": ("off", "low", "medium", "high"),
    "models.reviewer.thinking": ("off", "low", "medium", "high"),
    "models.planner.thinking": ("off", "low", "medium", "high"),
}


def _user_preset_names() -> list[str]:
    """USER-defined [presets.*] names only, for key completion. Built-in names
    are deliberately absent: writing presets.ultra.* creates a user table that
    REPLACES the built-in wholesale, a footgun TAB should not put one keystroke
    away (the same rule keeps `none` out of sandbox.isolation completion)."""
    try:
        return [p.name for p in preset_catalog(Path.cwd()).presets if p.origin != "built-in"]
    except ConfigError:
        return []


@_never_raises
def _complete_config_keys(prefix: str, *, include_presets: bool = True, **_kw: object) -> list[str]:
    """argcomplete: known dotted config leaf paths (effective + enum keys).
    From `preset` onward, also the user's presets.<name>.<leaf> paths (kept
    out of the bare-TAB listing, which is crowded enough already).

    ``include_presets=False`` for `config get`, which reads EFFECTIVE leaves:
    `[presets.*]` tables are stripped before validation, so it rejects them as
    "not a config leaf". A completer must offer what its command accepts.
    """
    try:
        keys = set(leaf_keys(load_effective(Path.cwd(), None)))
    except ConfigError:
        keys = set()
    keys |= set(_CONFIG_ENUM_CHOICES)
    if include_presets and prefix.startswith("preset"):
        pool = {k for k in keys if k != "preset"}
        keys |= {f"presets.{name}.{k}" for name in _user_preset_names() for k in pool}
    return sorted(k for k in keys if k.startswith(prefix))


# Presets offered for any `providers.<name>.extra_body` value (the provider name
# varies, so this is matched by suffix, not in _CONFIG_ENUM_CHOICES). The first
# is the recommended OpenRouter routing, a fast, prefix-caching backend.
_EXTRA_BODY_RECIPES: tuple[str, ...] = (
    '{ provider = { sort = "throughput" } }',
    '{ provider = { sort = "latency" } }',
    '{ provider = { sort = "price" } }',
)


@_never_raises
def _complete_config_values(
    prefix: str, parsed_args: argparse.Namespace | None = None, **_kw: object
) -> list[str]:
    """argcomplete: the Literal choices for the config key already typed."""
    key = getattr(parsed_args, "key", "") or ""
    if key == "preset":
        return _complete_presets(prefix)
    choices = list(_CONFIG_ENUM_CHOICES.get(key, ()))
    if key.endswith(".extra_body"):
        choices += list(_EXTRA_BODY_RECIPES)
    return [v for v in choices if v.startswith(prefix)]


@_never_raises
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


@_never_raises
def _complete_session_ids(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: ids across every session bucket (runs, asks, machine
    drafts). Offers exactly what `--from` accepts, so the two cannot drift."""
    out: list[str] = []
    for bucket in session_bucket_dirs(Path.cwd()):
        if not bucket.is_dir():
            continue
        out += [d.name for d in bucket.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    return sorted(out)


@_never_raises
def _complete_resumable_ids(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: ids `resume`/`fork` can actually pick up.

    Every bucket whose mode is resumable, so a plan and an ask are offered --
    but not a `machine create` draft, which resume refuses. Offering what a
    verb accepts, no less and no more.
    """
    out: list[str] = []
    for bucket in resumable_bucket_dirs(_state_dir(Path.cwd())):
        if not bucket.is_dir():
            continue
        out += [d.name for d in bucket.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    return sorted(out)


@_never_raises
def _complete_plan_session_ids(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: plan ids (for --from-plan / plan show/edit)."""
    plans = _plans_dir(Path.cwd())
    if not plans.is_dir():
        return []
    return sorted(
        p.name
        for p in plans.iterdir()
        if p.is_dir() and p.name.startswith(prefix) and (p / "plan.md").is_file()
    )


@_never_raises
def _complete_machine_ids(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: live machine instance ids (dirs under the per-repo state dir's machines/)."""
    base = _machines_dir(Path.cwd())
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix))


@_never_raises
def _complete_watch_targets(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: every session id plus every machine id -- what `attach`
    accepts. It resolves a session across all buckets, so offering only the
    runs bucket hid the plans and asks it opens happily."""
    return sorted(set(_complete_session_ids(prefix) + _complete_machine_ids(prefix)))


@_never_raises
def _complete_machine_files(prefix: str, **_kw: object) -> list[str]:
    """argcomplete: machine ``*.asm.toml`` files under cwd and the machines dir."""
    out: set[str] = set()
    for base in (Path.cwd(), _machines_dir(Path.cwd())):
        if base.is_dir():
            out.update(str(p) for p in base.rglob("*.asm.toml"))
    return sorted(p for p in out if p.startswith(prefix))
