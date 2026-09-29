# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The argcomplete completers for the CLI parser; each offers exactly what its verb accepts."""

from __future__ import annotations

import argparse
import contextlib
import functools
import pathlib
from collections.abc import Callable
from typing import Any

from agent6 import paths
from agent6.config import (
    Config,
    ConfigError,
    layer,
)
from agent6.models import choices as models_choices
from agent6.ui.cli import _common
from agent6.viewmodel import listing, machine_state


def _explicit_config(kw: dict[str, object]) -> pathlib.Path | None:
    """Return the `--config FILE` already typed on the line, so completions read that config."""
    parsed = kw.get("parsed_args")
    raw = getattr(parsed, "config", None)
    return raw if isinstance(raw, pathlib.Path) else None


def _never_raises(fn: Callable[..., list[str]]) -> Callable[..., list[str]]:
    """Return the completer wrapped to answer nothing on any exception.

    A traceback would land on the operator's command line.
    """

    @functools.wraps(fn)
    def guarded(*args: Any, **kwargs: Any) -> list[str]:
        """Return the suggestions, or none."""
        try:
            return fn(*args, **kwargs)
        except Exception:
            return []

    return guarded


@_never_raises
def _complete_providers(prefix: str, **kw: object) -> list[str]:
    """Return the connected provider names and the known presets."""
    from agent6.config import write  # noqa: PLC0415  # noqa: PLC0415
    from agent6.ui.cli import model  # noqa: PLC0415  # noqa: PLC0415

    names = set(model._connected_providers(_explicit_config(kw))) | set(write.PROVIDER_DEFAULTS)
    return sorted(n for n in names if n.startswith(prefix))


@_never_raises
def _complete_presets(prefix: str, **kw: object) -> list[str]:
    """Return the built-in presets and the configured `[presets.*]` names."""
    names = layer.available_preset_names(pathlib.Path.cwd(), _explicit_config(kw))
    return [n for n in names if n.startswith(prefix)]


@_never_raises
def _complete_skills(prefix: str, **_kw: object) -> list[str]:
    """Return the installed and extra-dir skill names."""
    from agent6.ui.cli import skills_cmds  # noqa: PLC0415  # noqa: PLC0415

    return [
        n
        for n in skills_cmds.resolved_skill_names_for_completion(pathlib.Path.cwd())
        if n.startswith(prefix)
    ]


@_never_raises
def _complete_mcp_servers(prefix: str, **kw: object) -> list[str]:
    """Return the configured MCP server names."""
    effective = layer.load_effective(pathlib.Path.cwd(), _explicit_config(kw))
    target = "repo" if getattr(kw.get("parsed_args"), "to_repo", False) else "global"
    from agent6.ui.cli import mcp_connect  # noqa: PLC0415  # noqa: PLC0415

    return sorted(
        name
        for name in effective.config.mcp.servers
        if name.startswith(prefix) and target in mcp_connect._layers_holding(effective, name)
    )


@_never_raises
def _complete_model_routes(prefix: str, **kw: object) -> list[str]:
    """Return every `provider/model` route the config can run, for `--model`."""
    try:
        cfg = layer.load_effective(pathlib.Path.cwd(), _explicit_config(kw)).config
    except ConfigError:
        return []
    return [r for r in models_choices.route_choices(cfg) if r.startswith(prefix)]


@_never_raises
def _complete_parallel_models(prefix: str, **kw: object) -> list[str]:
    """Return the routes for `run --parallel`, completing the entry after the last comma."""
    head, sep, frag = prefix.rpartition(",")
    lead = head + sep
    return [lead + r for r in _complete_model_routes(frag, **kw)]


# Values Tab must not put one keystroke away, though the schema allows them; typed explicitly.
_WITHHELD_ENUM_VALUES: dict[str, frozenset[str]] = {"sandbox.isolation": frozenset({"none"})}


def _config_enum_choices(config_path: pathlib.Path | None = None) -> dict[str, tuple[str, ...]]:
    """Return every closed-value leaf's allowed values, through the view the config surfaces render.

    A bool completes like an enum: `config set` takes exactly `true` or `false`.
    """
    from agent6.viewmodel import config_view  # noqa: PLC0415  # noqa: PLC0415

    try:
        view = config_view.build_config_view(layer.load_effective(pathlib.Path.cwd(), config_path))
    except ConfigError:
        # A config that does not load still completes: the schema carries the choices.
        view = config_view.build_config_view(
            layer.EffectiveConfig(
                config=Config(),
                sources={},
                layers=(),
                presets=tuple(layer.available_preset_names(pathlib.Path.cwd(), config_path)),
            )
        )
    out: dict[str, tuple[str, ...]] = {}
    for setting in view.settings:
        if setting.py_type == "bool":
            out[setting.key] = ("true", "false")
            continue
        if setting.py_type != "choice" or not setting.choices:
            continue
        withheld = _WITHHELD_ENUM_VALUES.get(setting.key, frozenset())
        out[setting.key] = tuple(c for c in setting.choices if c not in withheld)
    return out


def _config_list_keys(config_path: pathlib.Path | None = None) -> set[str]:
    """Return the list leaves `config add` and `config remove` accept."""
    from agent6.viewmodel import config_view  # noqa: PLC0415  # noqa: PLC0415

    effective = layer.load_effective(pathlib.Path.cwd(), config_path)
    return {
        setting.key
        for setting in config_view.build_config_view(effective).settings
        if setting.py_type == "list"
    }


def _user_preset_names(config_path: pathlib.Path | None = None) -> list[str]:
    """Return the user-defined `[presets.*]` names.

    A built-in name is withheld: writing `presets.<builtin>.*` replaces the built-in
    table wholesale.
    """
    try:
        return [
            p.name
            for p in layer.preset_catalog(pathlib.Path.cwd(), config_path).presets
            if p.origin != "built-in"
        ]
    except ConfigError:
        return []


@_never_raises
def _complete_config_keys(
    prefix: str, *, settable: bool = True, sections: bool = False, **kw: object
) -> list[str]:
    """Return the dotted config keys the verb accepts.

    Args:
        prefix: The typed prefix; from `preset` on, the user's `presets.<name>.<leaf>`
            paths are offered too.
        settable: Offer the enum keys and the preset paths, which `config get` rejects.
        sections: Offer section prefixes too, for `config show`.
        kw: argcomplete's keyword arguments.
    """
    explicit = _explicit_config(kw)
    try:
        keys = set(layer.leaf_keys(layer.load_effective(pathlib.Path.cwd(), explicit)))
    except ConfigError:
        keys = set()
    if settable:
        keys |= set(_config_enum_choices(explicit))
    command = getattr(kw.get("parsed_args"), "config_command", "")
    if command in ("add", "remove"):
        keys &= _config_list_keys(explicit)
    if sections:
        keys |= {
            ".".join(parts[:i])
            for key in keys
            for parts in (key.split("."),)
            for i in range(1, len(parts))
        }
    if settable and prefix.startswith("preset"):
        pool = {k for k in keys if k != "preset"}
        keys |= {f"presets.{name}.{k}" for name in _user_preset_names(explicit) for k in pool}
    if command in ("set", "unset", "add", "remove") and isinstance(
        getattr(kw.get("parsed_args"), "machine_file", None), pathlib.Path
    ):
        from agent6.machine import protected_overlay_key_error  # noqa: PLC0415

        keys = {key for key in keys if protected_overlay_key_error(key) is None}
    return sorted(k for k in keys if k.startswith(prefix))


# Offered for any `providers.<name>.extra_body` value, matched by suffix; OpenRouter routing.
_EXTRA_BODY_RECIPES: tuple[str, ...] = (
    '{ provider = { sort = "throughput" } }',
    '{ provider = { sort = "latency" } }',
    '{ provider = { sort = "price" } }',
)


@_never_raises
def _complete_config_values(
    prefix: str, parsed_args: argparse.Namespace | None = None, **_kw: object
) -> list[str]:
    """Return the values the config key already typed accepts.

    Its enum or configured choices, a role's provider's model ids, or the extra_body
    recipes.
    """
    key = getattr(parsed_args, "key", "") or ""
    if isinstance(getattr(parsed_args, "machine_file", None), pathlib.Path):
        from agent6.machine import protected_overlay_key_error  # noqa: PLC0415

        if protected_overlay_key_error(key) is not None:
            return []
    parts = key.split(".", 2)
    schema_key = parts[2] if len(parts) == 3 and parts[0] == "presets" else key
    raw = getattr(parsed_args, "config", None)
    config_path = raw if isinstance(raw, pathlib.Path) else None
    if getattr(parsed_args, "config_command", "") in (
        "add",
        "remove",
    ) and schema_key not in _config_list_keys(config_path):
        return []
    choices = list(_config_enum_choices(config_path).get(schema_key, ()))
    if key.endswith(".extra_body"):
        choices += list(_EXTRA_BODY_RECIPES)
    if not choices:
        with contextlib.suppress(ConfigError):
            choices = models_choices.config_value_choices(
                layer.load_effective(pathlib.Path.cwd(), config_path), schema_key
            )
    return [v for v in choices if v.startswith(prefix)]


@_never_raises
def _complete_model_verb_values(
    prefix: str, parsed_args: argparse.Namespace | None = None, **kw: object
) -> list[str]:
    """Return the provider names and routes for `model <role> [PROVIDER/]MODEL`.

    Only once a valid role is typed: argcomplete bleeds every optional positional's
    completer into the first slot.
    """
    role = getattr(parsed_args, "role", None)
    if role not in ("planner", "worker", "reviewer", "all"):
        return []
    from agent6.ui.cli import model  # noqa: PLC0415  # noqa: PLC0415

    providers = [
        name
        for name in model._connected_providers(_explicit_config({"parsed_args": parsed_args}))
        if name.startswith(prefix)
    ]
    return providers + _complete_model_routes(prefix, parsed_args=parsed_args, **kw)


@_never_raises
def _complete_session_ids(prefix: str, **_kw: object) -> list[str]:
    """Return the ids across every session bucket, what `--from` accepts."""
    return sorted(
        d.name for d in _common.all_session_dirs(pathlib.Path.cwd()) if d.name.startswith(prefix)
    )


@_never_raises
def _complete_session_ports(prefix: str, parsed_args: object = None, **_kw: object) -> list[str]:
    """Return the ports the session is listening on; only something inside its network sees them."""
    target = str(getattr(parsed_args, "target", "") or "")
    from agent6.sessions import ipc  # noqa: PLC0415  # noqa: PLC0415

    layout = _common.resolve_or_newest_layout(pathlib.Path.cwd(), target)
    if layout is None:
        return []
    return [str(p) for p in ipc.listening_ports(layout.session_dir) if str(p).startswith(prefix)]


@_never_raises
def _complete_resumable_ids(prefix: str, **_kw: object) -> list[str]:
    """Return the ids `resume` and `fork` pick up: every resumable bucket, not a machine draft."""
    out: list[str] = []
    from agent6.app import resume  # noqa: PLC0415  # noqa: PLC0415

    for bucket in resume.resumable_bucket_dirs(paths.state_dir(pathlib.Path.cwd())):
        if not bucket.is_dir():
            continue
        out += [d.name for d in bucket.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    return sorted(out)


@_never_raises
def _complete_live_session_ids(prefix: str, **_kw: object) -> list[str]:
    """Return the live sessions, through the same gate the verbs reaching a running one use."""
    return sorted(
        d.name
        for d in _common.all_session_dirs(pathlib.Path.cwd())
        if d.name.startswith(prefix) and listing.session_is_live(d)
    )


@_never_raises
def _complete_plan_session_ids(prefix: str, **_kw: object) -> list[str]:
    """Return the plan ids, for `plan show` and `plan edit`."""
    plans = _common._plans_dir(pathlib.Path.cwd())
    if not plans.is_dir():
        return []
    return sorted(
        p.name
        for p in plans.iterdir()
        if p.is_dir() and p.name.startswith(prefix) and (p / "plan.md").is_file()
    )


def _machine_instance_dirs(prefix: str) -> list[pathlib.Path]:
    """Return the machine instance dirs matching the prefix."""
    from agent6.sessions import layout as sessions_layout  # noqa: PLC0415  # noqa: PLC0415

    base = sessions_layout.machines_root(paths.state_dir(pathlib.Path.cwd()))
    if not base.is_dir():
        return []
    return [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]


@_never_raises
def _complete_machine_ids(prefix: str, **_kw: object) -> list[str]:
    """Return every machine instance id, finished ones included."""
    return sorted(p.name for p in _machine_instance_dirs(prefix))


def _machine_ids_taking(prefix: str, verb: machine_state.MachineVerb) -> list[str]:
    """Return the instances the verb acts on, through the verb's own refusal rule."""
    return sorted(
        p.name
        for p in _machine_instance_dirs(prefix)
        if not machine_state.machine_verb_refusal(p, p.name, verb)
    )


@_never_raises
def _complete_pokable_machine_ids(prefix: str, **_kw: object) -> list[str]:
    """Return the instances `machine poke` accepts: those with an open wait."""
    return _machine_ids_taking(prefix, "poke")


@_never_raises
def _complete_stoppable_machine_ids(prefix: str, **_kw: object) -> list[str]:
    """Return the instances `machine stop` has something to stop in: the running ones."""
    return _machine_ids_taking(prefix, "stop")


@_never_raises
def _complete_watch_targets(prefix: str, **_kw: object) -> list[str]:
    """Return every session id and every machine id, what `attach` accepts."""
    return sorted(set(_complete_session_ids(prefix) + _complete_machine_ids(prefix)))


@_never_raises
def _complete_machine_files(prefix: str, **_kw: object) -> list[str]:
    """Return the `*.asm.toml` files under cwd, spelled as the prefix is, and the machines dir's."""
    from agent6.sessions import layout as sessions_layout  # noqa: PLC0415  # noqa: PLC0415

    cwd = pathlib.Path.cwd()
    absolute = prefix.startswith("/")
    dotted = "./" if prefix.startswith("./") else ""
    out = {
        str(p) if absolute else dotted + str(p.relative_to(cwd)) for p in cwd.rglob("*.asm.toml")
    }
    machines = sessions_layout.machines_root(paths.state_dir(cwd))
    if machines.is_dir():
        out.update(str(p) for p in machines.rglob("*.asm.toml"))
    return sorted(p for p in out if p.startswith(prefix))
