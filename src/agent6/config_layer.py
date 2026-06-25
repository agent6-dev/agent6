# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Layered config resolution + auditing for agent6.

Config is assembled from up to four layers, lowest precedence first:

1. ``default``, the secure defaults baked into the pydantic model,
2. ``global`` , ``$XDG_CONFIG_HOME/agent6/config.toml`` (user-wide),
3. ``repo``   , the per-repo config under the state dir (out of the workspace,
   ``<state-base>/<repo-id>/config.toml``; see ``agent6.paths.state_dir``),
4. ``flag``   , an explicit ``--config FILE`` (power users / CI).

Raw TOML dicts are deep-merged in that order and validated **once**, so a
repo can override a single field without restating the rest. Every leaf
remembers which layer last set it, which powers ``agent6 config show``,
the audit surface that makes the effective config and its provenance
obvious at a glance.

A selected ``profile`` preset is injected just ABOVE the config layer that
SELECTED it (``--profile`` flag / repo / global ``[workflow].profile``), so the
profile OVERRIDES that config while a more-specific config layer (or an explicit
``--config FILE`` / machine overlay) still overrides the profile. Only the
most-specific source's profile is injected -- global and repo presets never
stack. See :func:`_apply_profile`.
"""

from __future__ import annotations

import contextlib
import json
import tomllib
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from agent6.config import (
    BUILTIN_PROFILES,
    AnthropicProviderEntry,
    Config,
    ConfigError,
    Deployment,
    OpenAIProviderEntry,
    resolve_profile,
    validate_config,
)
from agent6.config_io import (
    parse_cli_value,
    remove_toml_leaf,
    upsert_toml_leaf,
    upsert_toml_table,
)
from agent6.paths import (
    chown_to_real_user,
    global_config_path,
    repo_config_path,
    state_dir,
)

LayerName = Literal["default", "profile", "global", "repo", "flag", "machine"]

# Display order for `config show` / `config fill` (model definition order).
_SECTION_ORDER = (
    "agent6",
    "providers",
    "models",
    "sandbox",
    "git",
    "workflow",
    "budget",
    "notify",
    "mcp",
)


@dataclass(frozen=True, slots=True)
class Layer:
    name: LayerName
    path: Path | None
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EffectiveConfig:
    config: Config
    sources: dict[str, str]  # dotted leaf path -> layer name
    layers: tuple[Layer, ...]  # the layers that actually contributed (existing files)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file is not valid TOML ({path}): {exc}") from exc


def _global_state_dir() -> str | None:
    """Read ``[agent6].state_dir`` (the state BASE) from the GLOBAL config only.

    Resolved *before* the layered merge because it locates the directory the
    per-repo config lives in. Honored only from the global config;
    ``_forbid_repo_state_dir`` rejects it in any other layer.
    """
    gpath = global_config_path()
    if not gpath.is_file():
        return None
    data = _read_toml(gpath)
    section = data.get("agent6")
    if isinstance(section, dict):
        sd = section.get("state_dir")
        if isinstance(sd, str):
            # Raw pre-model read locating the per-repo config dir, so it runs
            # BEFORE the Config model (which also validates state_dir). Apply
            # the same absolute-path check here; fail loudly, don't drift.
            if not Path(sd).expanduser().is_absolute():
                raise ConfigError(
                    f"[agent6].state_dir in {gpath} must be an absolute path, got {sd!r}"
                )
            return sd
    return None


def _forbid_repo_state_dir(layer_name: str, data: dict[str, Any]) -> None:
    """Refuse ``state_dir`` in a repo/flag/overlay layer (global-only setting)."""
    section = data.get("agent6")
    if isinstance(section, dict) and "state_dir" in section:
        raise ConfigError(
            f"[agent6].state_dir may only be set in the global config"
            f" ({global_config_path()}), not in the {layer_name} config — it"
            " locates the directory the per-repo config itself lives in."
        )


def resolved_state_dir(repo_root: Path) -> Path:
    """The per-repo state dir for *repo_root*, honoring the global base override."""
    return state_dir(repo_root, _global_state_dir())


def repo_config_path_for(repo_root: Path) -> Path:
    """The per-repo config path for *repo_root* (out of the workspace)."""
    return repo_config_path(repo_root, _global_state_dir())


def discover_layers(repo_root: Path, explicit_path: Path | None) -> list[Layer]:
    """The config layers that exist, in precedence order (low -> high).

    The repo config lives out of the workspace under the state dir, whose base
    comes from the global config's ``[agent6].state_dir`` (or the XDG default).
    """
    layers: list[Layer] = []
    gpath = global_config_path()
    if gpath.is_file():
        layers.append(Layer("global", gpath, _read_toml(gpath)))
    base = _global_state_dir()
    rpath = repo_config_path(repo_root, base)
    if rpath.is_file():
        data = _read_toml(rpath)
        _forbid_repo_state_dir("repo", data)
        layers.append(Layer("repo", rpath, data))
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise ConfigError(f"--config file not found: {explicit_path}")
        data = _read_toml(explicit_path)
        _forbid_repo_state_dir("--config", data)
        layers.append(Layer("flag", explicit_path, data))
    return layers


def available_profile_names(repo_root: Path, explicit_path: Path | None = None) -> list[str]:
    """Profile names a chooser can offer: the built-ins plus the user's custom
    ``[profiles.<name>]`` tables (read from the config layers, the same source
    ``--profile`` resolves against), sorted + de-duplicated. A config-read failure
    degrades to the built-ins alone, so a caller (e.g. the TUI's new-work chooser)
    never blocks on a bad config."""
    names: set[str] = set(BUILTIN_PROFILES)
    with contextlib.suppress(Exception):
        for layer in discover_layers(repo_root, explicit_path):
            prof = layer.data.get("profiles")
            if isinstance(prof, dict):
                names.update(prof.keys())
    return sorted(names)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        existing = out.get(key)
        if isinstance(val, dict) and isinstance(existing, dict):
            # A discriminated dict (e.g. a [providers.<name>] entry) whose
            # `api_format` changes between layers must REPLACE, not deep-merge:
            # the lower layer's format-specific keys (an anthropic prompt_caching,
            # say) are invalid under the new format and would otherwise survive
            # the merge and surface as a confusing extra_forbidden error.
            if (
                "api_format" in val
                and "api_format" in existing
                and val.get("api_format") != existing.get("api_format")
            ):
                out[key] = val
            else:
                out[key] = _deep_merge(existing, val)
        else:
            out[key] = val
    return out


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted leaf paths.

    Lists (incl. arrays of tables) are treated as leaves, their provenance
    is the whole array, not individual elements.
    """
    out: dict[str, Any] = {}
    for key, val in data.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict) and val:
            out.update(_flatten(val, prefix=f"{path}."))
        else:
            out[path] = val
    return out


def _effective_from_layers(layers: list[Layer], *, source: str) -> EffectiveConfig:
    """Merge *layers* low->high, validate, and build the per-leaf source map."""
    merged: dict[str, Any] = {}
    source_of_leaf: dict[str, str] = {}
    for layer in layers:
        merged = _deep_merge(merged, layer.data)
        for leaf in _flatten(layer.data):
            source_of_leaf[leaf] = layer.name
    config = validate_config(merged, source=source)
    # Source map over the *effective* config: every leaf the model
    # produced, attributed to the layer that set it or "default".
    effective_leaves = _flatten(config.model_dump(mode="python"))
    sources = {leaf: source_of_leaf.get(leaf, "default") for leaf in effective_leaves}
    return EffectiveConfig(config=config, sources=sources, layers=tuple(layers))


def _own_profile(layer: Layer) -> str:
    """A layer's OWN raw ``[workflow].profile`` (not the merged value), or ""."""
    wf = layer.data.get("workflow")
    if isinstance(wf, dict):
        return str(wf.get("profile", "") or "")
    return ""


def _select_profile(cleaned: list[Layer], profile_override: str) -> tuple[str, str]:
    """Pick the (profile name, source) most-specific first from each layer's OWN
    raw ``[workflow].profile`` (never stacking global+repo): the ``--profile``
    flag, else the ``repo`` layer's field, else the ``global`` layer's field,
    else ("", "none")."""
    if profile_override:
        return profile_override, "flag"
    by_name = {layer.name: layer for layer in cleaned}
    for source in ("repo", "global"):
        layer = by_name.get(source)
        if layer is not None and (name := _own_profile(layer)):
            return name, source
    return "", "none"


def _insert_profile(cleaned: list[Layer], preset: Layer, source: str) -> list[Layer]:
    """Splice *preset* into *cleaned* at the position for its *source*.

    ``global``/``repo`` -> right AFTER that config layer (so the profile
    overrides it but the more-specific config layer / flag still wins). ``flag``
    (``--profile``) -> just BELOW an explicit ``--config FILE`` / machine overlay
    if present (those still win), else appended last (overrides all config).
    """
    out: list[Layer] = []
    inserted = False
    for layer in cleaned:
        if source == "flag" and not inserted and layer.name in ("flag", "machine"):
            out.append(preset)
            inserted = True
        out.append(layer)
        if not inserted and layer.name == source:  # source in {"global", "repo"}
            out.append(preset)
            inserted = True
    if not inserted:
        out.append(preset)
    return out


def _apply_profile(layers: list[Layer], profile_override: str) -> list[Layer]:
    """Strip ``[profiles]`` tables out of the user layers (they are meta-config,
    not part of the validated Config) and inject the selected profile preset
    just ABOVE the config layer that SELECTED it, so the profile OVERRIDES that
    config while a more-specific config layer (or an explicit ``--config FILE`` /
    machine overlay) still overrides the profile. Only the most-specific source's
    profile is injected -- global and repo presets never stack.

    Source is chosen by :func:`_select_profile` (``--profile`` flag > repo's own
    ``[workflow].profile`` > global's own), and the preset is spliced in by
    :func:`_insert_profile`. Resulting precedence (low->high): default <
    global-config < [profile if global-selected] < repo-config <
    [profile if repo-selected] < [profile if --flag] < flag(``--config FILE``) <
    machine-overlay.
    """
    cleaned: list[Layer] = []
    user_profiles: dict[str, Any] = {}
    for layer in layers:
        data = dict(layer.data)
        prof = data.pop("profiles", None)
        if isinstance(prof, dict):
            user_profiles = _deep_merge(user_profiles, prof)
        cleaned.append(Layer(layer.name, layer.path, data))

    name, source = _select_profile(cleaned, profile_override)
    overrides = resolve_profile(name, user_profiles)
    if not overrides:
        return cleaned
    return _insert_profile(cleaned, Layer("profile", None, overrides), source)


def load_effective(
    repo_root: Path, explicit_path: Path | None = None, *, profile: str = ""
) -> EffectiveConfig:
    """Merge + validate all layers and record per-leaf provenance. A named
    ``profile`` (CLI flag or ``[workflow].profile``) is injected just above the
    config layer that selected it, so it OVERRIDES that config (a more-specific
    config layer / flag still wins); ``[profiles.<name>]`` tables in config
    define custom ones. See :func:`_apply_profile`."""
    layers = discover_layers(repo_root, explicit_path)
    layers = _apply_profile(layers, profile)
    return _effective_from_layers(layers, source="(merged config layers)")


def load_effective_with_overlay(repo_root: Path, overlay: dict[str, Any]) -> EffectiveConfig:
    """Like :func:`load_effective` but with *overlay* as the highest layer.

    Used by `agent6 machine run` to apply a machine file's ``[config]``
    table on top of the repo/global/default layers. The overlay is merged
    and validated exactly like a config file; its leaves are labelled
    ``machine`` in the provenance map (``config show`` style).
    """
    layers = discover_layers(repo_root, None)
    if overlay:
        _forbid_repo_state_dir("machine overlay", overlay)
        layers = [*layers, Layer("machine", None, overlay)]
    # Apply the selected profile (and strip [profiles] tables) just like
    # load_effective, so a user's [profiles.<name>] + [workflow].profile work
    # under `machine run` / `config --machine` instead of failing validation.
    layers = _apply_profile(layers, "")
    return _effective_from_layers(layers, source="(merged config layers + machine overlay)")


def leaf_keys(eff: EffectiveConfig) -> list[str]:
    """Every dotted leaf path in the effective config, sorted (for completion)."""
    return sorted(_flatten(eff.config.model_dump(mode="python")))


def effective_leaf(eff: EffectiveConfig, dotted_key: str) -> tuple[Any, str] | None:
    """The ``(value, source-layer)`` for *dotted_key*, or None if it is not a leaf.

    Mirrors `config show`: the value comes from the merged+validated config and
    the source is the layer that set it (``default`` when no layer did).
    """
    leaves = _flatten(eff.config.model_dump(mode="python"))
    if dotted_key not in leaves:
        return None
    return leaves[dotted_key], eff.sources.get(dotted_key, "default")


def format_value(val: Any) -> str:
    """Render a config leaf value the same way `config show` does."""
    return _fmt_value(val)


# ---------------------------------------------------------------------------
# Schema introspection + the UI-agnostic view-model
# ---------------------------------------------------------------------------
#
# `config show` (CLI), the TUI config page, and a future web UI all render the
# SAME ConfigView, so config logic (provenance, defaults, enum choices, adaptive
# resolution) lives here once and the renderers stay thin -- the same split that
# keeps `ui.state` (a pure event-fold) independent of any widget toolkit.


@dataclass(frozen=True, slots=True)
class ConfigSetting:
    """One config leaf, fully described for display + editing."""

    key: str  # dotted leaf path, e.g. "sandbox.run_commands"
    section: str  # top-level section, e.g. "sandbox"
    value: Any  # raw effective value (None for an unset/adaptive setting)
    effective_value: Any  # resolved value; == value unless a caller resolves it
    default: Any  # the built-in default (None if unknown / no static default)
    source: str  # layer that set it: default/global/repo/flag/machine
    modified: bool  # a layer set it (source != "default")
    is_adaptive: bool  # effective_value was resolved away from the raw value
    py_type: str  # str | int | bool | float | choice | list | table
    choices: tuple[str, ...] | None  # enum options for a dropdown, else None


@dataclass(frozen=True, slots=True)
class ConfigView:
    """The whole effective config as a flat, section-ordered list of settings,
    plus the contributing layers for a provenance legend. The one structure
    every UI renders."""

    settings: tuple[ConfigSetting, ...]
    sections: tuple[str, ...]
    layers: tuple[Layer, ...]


def _unwrap_optional(ann: Any) -> Any:
    """``X | None`` -> ``X``; leave a genuine multi-member union (e.g. the
    provider-entry union) untouched."""
    if get_origin(ann) in (Union, types.UnionType):
        non_none = [a for a in get_args(ann) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return ann


def _literal_choices(ann: Any) -> tuple[str, ...] | None:
    ann = _unwrap_optional(ann)
    if get_origin(ann) is Literal:
        return tuple(str(a) for a in get_args(ann))
    return None


def _nested_model(ann: Any) -> type[BaseModel] | None:
    ann = _unwrap_optional(ann)
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann
    return None


def _value_models(ann: Any) -> tuple[type[BaseModel], ...]:
    """The model(s) a ``dict`` field maps to: one for a plain model value, or
    several for a discriminated union (e.g. ``providers`` ->
    Anthropic|OpenAI entry). Returns () when the value isn't model-shaped."""
    ann = _unwrap_optional(ann)
    if get_origin(ann) is not dict:
        return ()
    val = get_args(ann)[1]
    if get_origin(val) is Annotated:  # Annotated[Union[...], Discriminator(...)]
        val = get_args(val)[0]
    if get_origin(val) in (Union, types.UnionType):
        return tuple(m for m in get_args(val) if _nested_model(m) is not None)
    model = _nested_model(val)
    return (model,) if model is not None else ()


def _merge_field_schema(
    models: tuple[type[BaseModel], ...], parts: list[str]
) -> tuple[str, tuple[str, ...] | None, Any] | None:
    """Resolve a field across discriminated-union members, merging choices: a
    field shared by every member (e.g. ``auth_style``) keeps its choices; the
    discriminator itself (``api_format``: Literal['anthropic'] in one member,
    Literal['openai'] in the other) becomes the union of both."""
    results = [r for m in models if (r := _field_schema(m, parts)) is not None]
    if not results:
        return None
    if len(results) == 1:
        return results[0]
    choices: list[str] = []
    for _, member_choices, _default in results:
        for c in member_choices or ():
            if c not in choices:
                choices.append(c)
    merged = tuple(choices) or None
    py_type = "choice" if merged else results[0][0]
    default = next((d for _, _, d in results if d is not None), results[0][2])
    return py_type, merged, default


def _type_label(ann: Any) -> str:
    ann = _unwrap_optional(ann)
    origin = get_origin(ann)
    if origin is Literal:
        return "choice"
    if origin in (list, tuple):
        return "list"
    if origin is dict:
        return "table"
    if isinstance(ann, type):
        return ann.__name__
    return "str"


def _field_schema(
    model_cls: type[BaseModel], parts: list[str]
) -> tuple[str, tuple[str, ...] | None, Any] | None:
    """Walk *model_cls* down the dotted *parts* to a leaf field and return
    ``(py_type, choices, default)``, or None if the path can't be resolved
    (e.g. a dynamic provider key whose value model is a discriminated union)."""
    name = parts[0]
    fi = getattr(model_cls, "model_fields", {}).get(name)
    if fi is None:
        return None
    ann = fi.annotation
    if len(parts) == 1:
        default = fi.default
        if default is PydanticUndefined:
            default = fi.default_factory() if fi.default_factory is not None else None
        return _type_label(ann), _literal_choices(ann), default
    nested = _nested_model(ann)
    if nested is not None:
        return _field_schema(nested, parts[1:])
    models = _value_models(ann)
    if models and len(parts) >= 3:
        return _merge_field_schema(models, parts[2:])  # skip the dynamic dict key
    return None


def build_config_view(
    eff: EffectiveConfig, *, resolved: dict[str, Any] | None = None
) -> ConfigView:
    """Combine the effective config, its provenance, the schema (types + enum
    choices + defaults), and any caller-resolved values into the flat
    ConfigView every UI renders.

    *resolved* maps a dotted key to its resolved value for settings whose raw
    value is a placeholder for runtime resolution (compaction left unset ->
    adaptive, sized from the model's context window). It never changes
    provenance or the modified flag -- it only fills ``effective_value`` /
    ``is_adaptive`` so a UI can show the real number.
    """
    resolved = resolved or {}
    leaves = _flatten(eff.config.model_dump(mode="python"))
    by_section: dict[str, list[str]] = {}
    for leaf in leaves:
        by_section.setdefault(leaf.split(".", 1)[0], []).append(leaf)
    ordered = [s for s in _SECTION_ORDER if s in by_section]
    ordered += [s for s in by_section if s not in _SECTION_ORDER]

    settings: list[ConfigSetting] = []
    for section in ordered:
        for leaf in by_section[section]:
            value = leaves[leaf]
            source = eff.sources.get(leaf, "default")
            schema = _field_schema(eff.config.__class__, leaf.split("."))
            py_type, choices, default = schema if schema is not None else ("str", None, None)
            eff_val = resolved.get(leaf, value)
            settings.append(
                ConfigSetting(
                    key=leaf,
                    section=section,
                    value=value,
                    effective_value=eff_val,
                    default=default,
                    source=source,
                    modified=source != "default",
                    is_adaptive=leaf in resolved and eff_val != value,
                    py_type=py_type,
                    choices=choices,
                )
            )
    return ConfigView(settings=tuple(settings), sections=tuple(ordered), layers=eff.layers)


# ---------------------------------------------------------------------------
# Rendering: `config show`
# ---------------------------------------------------------------------------


def _fmt_value(val: Any) -> str:
    if val is None:
        return "(unset)"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (list, tuple)):
        if not val:
            return "[]"
        return "[" + ", ".join(_fmt_value(v) for v in val) + "]"
    if isinstance(val, dict):
        return "{...}" if val else "{}"
    return str(val)


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "\u2026"


def render_show(
    eff: EffectiveConfig, *, as_json: bool = False, resolved: dict[str, Any] | None = None
) -> str:
    """Render the effective config + provenance from the shared ConfigView.

    Plain mode is a section-grouped, fixed-width 3-column table
    (key / value / source) with a leading ``*`` on rows that override the
    default, no box drawing, so it never wraps badly. ``*`` rows are the
    ones to eyeball; settings whose value is resolved at runtime (compaction
    left adaptive) show their resolved value tagged ``(adaptive)``. JSON mode
    emits the full per-leaf view (value, effective, default, source, modified,
    adaptive, type, choices) -- the complete machine-readable picture.

    *resolved* maps dotted keys to their resolved values (e.g. adaptive
    compaction sized from the worker model); the caller computes it.
    """
    view = build_config_view(eff, resolved=resolved)
    if as_json:
        payload = {
            s.key: {
                "value": s.value,
                "effective": s.effective_value,
                "default": s.default,
                "source": s.source,
                "modified": s.modified,
                "adaptive": s.is_adaptive,
                "type": s.py_type,
                "choices": list(s.choices) if s.choices is not None else None,
            }
            for s in view.settings
        }
        return json.dumps(payload, indent=2, sort_keys=True, default=str)

    by_section: dict[str, list[ConfigSetting]] = {}
    for s in view.settings:
        by_section.setdefault(s.section, []).append(s)

    # Column widths (capped to avoid wide-terminal sprawl / narrow wrap).
    key_w = min(max((len(s.key) for s in view.settings), default=10) + 1, 40)
    val_w = 40
    lines: list[str] = []
    for section in view.sections:
        lines.append(f"[{section}]")
        for s in by_section[section]:
            if s.is_adaptive:
                value = f"{_fmt_value(s.effective_value)}  (adaptive)"
            else:
                value = _fmt_value(s.value)
            mark = "*" if s.modified else " "
            short_key = _truncate(s.key, key_w)
            short_val = _truncate(value, val_w)
            lines.append(f"{mark} {short_key:<{key_w}} {short_val:<{val_w}} {s.source}")
        lines.append("")
    legend_layers = ", ".join(
        f"{lyr.name}={lyr.path}" for lyr in view.layers if lyr.path is not None
    )
    lines.append("source: default | " + (legend_layers or "(no config files; all defaults)"))
    lines.append("* = overrides the built-in default")
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Materialize: `config fill`
# ---------------------------------------------------------------------------


def _toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _toml_str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    return str(value)


def _emit_table(path: str, data: dict[str, Any], lines: list[str]) -> None:
    """Emit one TOML table (and recurse into sub-tables / arrays of tables).

    ``None`` values are skipped, an unset optional field materializes as
    absent, i.e. "use the default".
    """
    scalars = {
        k: v
        for k, v in data.items()
        if v is not None and not isinstance(v, dict) and not _is_table_array(v)
    }
    subtables = {k: v for k, v in data.items() if isinstance(v, dict) and v}
    arraytables = {k: v for k, v in data.items() if _is_table_array(v)}
    # Emit a header for this path only when it carries scalar keys, or when
    # it is a genuine leaf table (no children at all). A pure parent table
    # like [providers] / [models] is left implicit so we don't print an
    # empty header above its [providers.<name>] children.
    is_leaf = not subtables and not arraytables
    if scalars or is_leaf:
        lines.append(f"[{path}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {_toml_scalar(value)}")
        lines.append("")
    for key, sub in subtables.items():
        _emit_table(f"{path}.{key}" if path else key, sub, lines)
    for key, arr in arraytables.items():
        for item in arr:
            lines.append(f"[[{path}.{key}]]" if path else f"[[{key}]]")
            for k2, v2 in item.items():
                if v2 is not None and not isinstance(v2, dict):
                    lines.append(f"{k2} = {_toml_scalar(v2)}")
            lines.append("")


def _is_table_array(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(v, dict) for v in value)
    )


def materialize(config: Config, *, for_repo: bool = False) -> str:
    """Render the fully-resolved config as a complete TOML document.

    Used by ``agent6 config fill`` to snapshot every effective value into
    one explicit file (handy before tightening defaults or for an audit).
    When ``for_repo`` is set, the global-only ``[agent6].state_dir``
    is dropped (it is invalid in a per-repo config).
    """
    data = config.model_dump(mode="python")
    if for_repo and isinstance(data.get("agent6"), dict):
        data["agent6"].pop("state_dir", None)
    lines: list[str] = [
        "# agent6 effective config, materialized by `agent6 config fill`.",
        "# Every value below is explicit; edit freely.",
        "",
    ]
    ordered = [s for s in _SECTION_ORDER if s in data]
    ordered += [s for s in data if s not in _SECTION_ORDER]
    for section in ordered:
        value = data[section]
        if isinstance(value, dict):
            if not value:
                continue
            _emit_table(section, value, lines)
        elif _is_table_array(value):
            for item in value:
                lines.append(f"[[{section}]]")
                for k2, v2 in item.items():
                    if v2 is not None and not isinstance(v2, dict):
                        lines.append(f"{k2} = {_toml_scalar(v2)}")
                lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Shared edit path: the `config` CLI and the TUI/web config editors write
# through these, so a value set from any UI is parsed, persisted (comment-
# preserving), re-validated, and rolled back on failure identically.
# ---------------------------------------------------------------------------


def _write_target(repo_root: Path, *, to_repo: bool) -> Path:
    return repo_config_path_for(repo_root) if to_repo else global_config_path()


def _revalidate(repo_root: Path, target: Path, prior: str | None) -> str | None:
    """Re-load the merged config after an edit; restore *prior* (or delete a
    freshly-created file) and return the error string if the edit is invalid."""
    try:
        load_effective(repo_root, None)
    except ConfigError as exc:
        if prior is None:
            target.unlink(missing_ok=True)
        else:
            target.write_text(prior, encoding="utf-8")
        return str(exc)
    return None


def set_config_value(
    repo_root: Path, dotted_key: str, raw_value: str, *, to_repo: bool = False
) -> str | None:
    """Set one leaf in the global (or, with *to_repo*, the repo) config.

    *raw_value* is interpreted exactly as ``config set`` interprets a CLI value
    (``true``/numbers/arrays parse; a bare word stays a string). Returns an
    error string when the edit produced an invalid config (the file is rolled
    back and left as it was), else None.
    """
    target = _write_target(repo_root, to_repo=to_repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    prior = target.read_text(encoding="utf-8") if target.is_file() else None
    try:
        upsert_toml_leaf(target, dotted_key, parse_cli_value(raw_value))
    except ValueError as exc:
        return str(exc)
    err = _revalidate(repo_root, target, prior)
    if err is None:
        chown_to_real_user(target.parent)
        chown_to_real_user(target)
    return err


def set_config_table(
    repo_root: Path,
    table: str,
    fields: dict[str, str | bool | None],
    *,
    to_repo: bool = False,
) -> str | None:
    """Insert/replace a whole ``[table]`` block in one shot (e.g. a new
    ``[providers.<name>]`` entry from the TUI's add-provider form). Revalidates
    the merged config and rolls the file back on failure. Returns an error string
    on invalid config, else None. ``None`` field values are omitted."""
    target = _write_target(repo_root, to_repo=to_repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    prior = target.read_text(encoding="utf-8") if target.is_file() else None
    try:
        upsert_toml_table(target, table, fields)
    except ValueError as exc:
        return str(exc)
    err = _revalidate(repo_root, target, prior)
    if err is None:
        chown_to_real_user(target.parent)
        chown_to_real_user(target)
    return err


def provider_choices() -> dict[str, list[str]]:
    """Fixed-choice fields for the add-provider form, read from the schema so
    they never drift: the api_format discriminator (per provider subclass) and
    the deployment profiles."""
    formats: list[str] = []
    for model in (AnthropicProviderEntry, OpenAIProviderEntry):
        formats.extend(get_args(model.model_fields["api_format"].annotation))
    return {"api_format": formats, "deployment": list(get_args(Deployment))}


# Known provider presets, keyed by the conventional provider NAME used as the
# [providers.<name>] table key. Maps a name to its api_format and, for
# OpenAI-compatible hosts, the default base_url. Both `agent6 connect` and the
# TUI add-provider form consult this so well-known names (openrouter, ollama)
# land on the right host instead of the bare (api_format, deployment) fallback
# in `config._default_base_url` -- which only knows api.openai.com for the
# `openai` format and would otherwise point an "openrouter" provider at OpenAI.
# Advanced deployments (vertex/azure/token_command) are hand-edited per CONFIG.md.
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "anthropic": {"api_format": "anthropic"},
    "openai": {"api_format": "openai", "base_url": "https://api.openai.com/v1"},
    "openrouter": {"api_format": "openai", "base_url": "https://openrouter.ai/api/v1"},
    "ollama": {"api_format": "openai", "base_url": "http://localhost:11434/v1"},
}


def unset_config_value(repo_root: Path, dotted_key: str, *, to_repo: bool = False) -> str | None:
    """Remove one leaf so it reverts to the next layer / built-in default.

    Re-validates and rolls back on failure. Returns None on success, including
    the no-op case where the key was not set in the target file.
    """
    target = _write_target(repo_root, to_repo=to_repo)
    if not target.is_file():
        return None
    prior = target.read_text(encoding="utf-8")
    try:
        removed = remove_toml_leaf(target, dotted_key)
    except ValueError as exc:
        return str(exc)
    if not removed:
        return None
    err = _revalidate(repo_root, target, prior)
    if err is None:
        chown_to_real_user(target)
    return err
