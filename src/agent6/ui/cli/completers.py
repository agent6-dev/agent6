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
from agent6.app import review_actions
from agent6.config import (
    Config,
    ConfigError,
    _layer_data,
    _presets,
    io,
    layer,
    write,
)
from agent6.models import choices as models_choices
from agent6.sessions import id, process
from agent6.ui.cli import _common
from agent6.viewmodel import listing, machine_state


def complete_finding_ids(prefix: str, **kw: Any) -> list[str]:
    """Complete finding IDs from the selected session's saved evidence.

    Returns:
        Matching finding IDs, or an empty list without a readable session.
    """
    parsed = kw.get("parsed_args")
    target = getattr(parsed, "session_id", "") or getattr(parsed, "target", "")
    if not target:
        return []
    try:
        saved = _common.resolve_session_layout(pathlib.Path.cwd(), target)
        view = review_actions.read_review(saved.session_dir)
    except (id.SessionIdError, OSError, review_actions.ReviewActionError):
        return []
    return sorted(f.finding_id for f in view.findings if f.finding_id.startswith(prefix))


def _explicit_config(kw: dict[str, object]) -> tuple[pathlib.Path, ...]:
    """Return the `--config FILE` layers already typed on the line, before and after the verb.

    Args:
        kw: A completer's keyword arguments; its `parsed_args` is the line parsed so far.

    Returns:
        The files, those before the verb first.
    """
    parsed = kw.get("parsed_args")
    before = getattr(parsed, "config", None)
    after = getattr(parsed, "config_after", None)
    return (
        *(before if isinstance(before, list) else []),
        *(after if isinstance(after, list) else []),
    )


def _never_raises(fn: Callable[..., list[str]]) -> Callable[..., list[str]]:
    """Return the completer wrapped to answer nothing on any exception.

    A traceback would land on the operator's command line.

    Args:
        fn: The completer to guard.

    Returns:
        The guarded completer.
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
    names = _presets.available_preset_names(pathlib.Path.cwd(), _explicit_config(kw))
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
    from agent6.ui.cli import mcp_add  # noqa: PLC0415  # noqa: PLC0415

    return sorted(
        name
        for name in effective.config.mcp.servers
        if name.startswith(prefix) and target in mcp_add._layers_holding(effective, name)
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


def _config_enum_choices(
    config_paths: tuple[pathlib.Path, ...] = (),
) -> dict[str, tuple[str, ...]]:
    """Return every closed-value leaf's allowed values, through the view the config surfaces render.

    A bool completes like an enum: `config set` takes exactly `true` or `false`.

    Args:
        config_paths: The `--config` files already typed on the line.

    Returns:
        The values, keyed by the leaf's dotted key.
    """
    from agent6.viewmodel import config_view  # noqa: PLC0415  # noqa: PLC0415

    try:
        view = config_view.build_config_view(layer.load_effective(pathlib.Path.cwd(), config_paths))
    except ConfigError:
        # A config that does not load still completes: the schema carries the choices.
        view = config_view.build_config_view(
            _layer_data.EffectiveConfig(
                config=Config(),
                sources={},
                layers=(),
                presets=tuple(_presets.available_preset_names(pathlib.Path.cwd(), config_paths)),
            )
        )
    out: dict[str, tuple[str, ...]] = {}
    for setting in view.settings:
        if setting.py_type == "bool":
            out[setting.key] = ("true", "false")
            continue
        if setting.py_type != "choice" or not setting.choices:
            continue
        out[setting.key] = tuple(setting.choices)
    return out


def _config_list_keys(config_paths: tuple[pathlib.Path, ...] = ()) -> set[str]:
    """Return the list leaves `config add` and `config remove` accept."""
    from agent6.viewmodel import config_view  # noqa: PLC0415  # noqa: PLC0415

    effective = layer.load_effective(pathlib.Path.cwd(), config_paths)
    return {
        setting.key
        for setting in config_view.build_config_view(effective).settings
        if setting.py_type == "list"
    }


def _user_preset_names(config_paths: tuple[pathlib.Path, ...] = ()) -> list[str]:
    """Return the user-defined `[presets.*]` names.

    A built-in the operator has not replaced is withheld: writing `presets.<builtin>.*`
    replaces the built-in table wholesale.

    Args:
        config_paths: The `--config` files already typed on the line.

    Returns:
        The names; none when a config layer is unreadable or malformed.
    """
    try:
        return [
            p.name
            for p in _presets.preset_catalog(pathlib.Path.cwd(), config_paths).presets
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
        **kw: argcomplete's keyword arguments.

    Returns:
        The keys starting with prefix, sorted: only list keys for `config add` and
        `config remove`, and no operator-only key for a `--machine-file` write.
    """
    from agent6.viewmodel import config_view  # noqa: PLC0415

    explicit = _explicit_config(kw)
    try:
        keys = set(layer.leaf_keys(layer.load_effective(pathlib.Path.cwd(), explicit)))
        parents = {
            io.render_key(parts[:i])
            for key in keys
            for parts in (io.split_key(key),)
            for i in range(1, len(parts))
        }
        keys |= {
            key
            for key in parents
            if (schema := config_view.field_schema(Config, io.split_key(key))) is not None
            and schema.py_type == "table"
            and not write.names_a_section(key)
        }
    except ConfigError:
        keys = set()
    if settable:
        keys |= set(_config_enum_choices(explicit))
    command = getattr(kw.get("parsed_args"), "config_command", "")
    if command in ("add", "remove"):
        keys &= _config_list_keys(explicit)
    if sections:
        keys |= {
            io.render_key(parts[:i])
            for key in keys
            for parts in (io.split_key(key),)
            for i in range(1, len(parts))
        }
    if settable and prefix.startswith("preset"):
        pool = {k for k in keys if k != "preset"}
        keys |= {
            io.render_key(("presets", name, *io.split_key(k)))
            for name in _user_preset_names(explicit)
            for k in pool
        }
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

    Args:
        prefix: The text typed so far.
        parsed_args: The line parsed so far; its `key` is the config key.

    Returns:
        The key's enum or configured choices, a role's provider's model ids, or the
        extra_body recipes, those starting with prefix; none for a key the verb refuses.
    """
    key = getattr(parsed_args, "key", "") or ""
    if isinstance(getattr(parsed_args, "machine_file", None), pathlib.Path):
        from agent6.machine import protected_overlay_key_error  # noqa: PLC0415

        if protected_overlay_key_error(key) is not None:
            return []
    parts = io.split_key(key)
    schema_key = io.render_key(parts[2:] if len(parts) > 2 and parts[0] == "presets" else parts)
    config_paths = _explicit_config({"parsed_args": parsed_args})
    if getattr(parsed_args, "config_command", "") in (
        "add",
        "remove",
    ) and schema_key not in _config_list_keys(config_paths):
        return []
    choices = list(_config_enum_choices(config_paths).get(schema_key, ()))
    if key.endswith(".extra_body"):
        choices += list(_EXTRA_BODY_RECIPES)
    if not choices:
        with contextlib.suppress(ConfigError):
            choices = models_choices.config_value_choices(
                layer.load_effective(pathlib.Path.cwd(), config_paths), schema_key
            )
    return [v for v in choices if v.startswith(prefix)]


@_never_raises
def _complete_model_verb_values(
    prefix: str, parsed_args: argparse.Namespace | None = None, **kw: object
) -> list[str]:
    """Return the provider names and routes for `model <role> [PROVIDER/]MODEL`.

    Only once a valid role is typed: argcomplete bleeds every optional positional's
    completer into the first slot.

    Args:
        prefix: The text typed so far.
        parsed_args: The line parsed so far.
        **kw: The rest of argcomplete's keyword arguments.

    Returns:
        The config's provider names, then its routes, each starting with prefix.
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

    layout = _common.resolve_or_newest_layout(pathlib.Path.cwd(), target)
    if layout is None:
        return []
    return [
        str(p) for p in process.listening_ports(layout.session_dir) if str(p).startswith(prefix)
    ]


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
def complete_steer_ids(prefix: str, parsed_args: object = None, **_kw: object) -> list[str]:
    """Return live sessions, or resumable sessions for a finding action."""
    if getattr(parsed_args, "finding", None) or getattr(parsed_args, "resume", False):
        return _complete_resumable_ids(prefix)
    return _complete_live_session_ids(prefix)


@_never_raises
def _complete_exec_rest(prefix: str, parsed_args: argparse.Namespace, **_kw: object) -> list[str]:
    """Return the live sessions `exec` runs in; none after `--new`, which takes no session."""
    return [] if parsed_args.new else _complete_live_session_ids(prefix)


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


@_never_raises
def _complete_machine_targets(prefix: str, **_kw: object) -> list[str]:
    """Return the machine files `_complete_machine_files` offers, and the names they declare."""
    from agent6.app.machine import listing  # noqa: PLC0415

    cwd = pathlib.Path.cwd()
    names = {
        row.name
        for row in listing.machine_rows(cwd, paths.state_dir(cwd))
        if row.file is not None and row.name != "-"
    }
    return sorted({*_complete_machine_files(prefix), *(n for n in names if n.startswith(prefix))})
