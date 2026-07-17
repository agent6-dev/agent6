# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The effective-config view-model: one structure every front-end renders.

`config show` (CLI), the TUI config page, and the web UI all render the SAME
ConfigView, so config display logic (provenance, defaults, enum choices,
adaptive resolution) lives here once and the renderers stay thin. Loading,
merging, and writing config stays in `agent6.config.layer`; this module only
reads an EffectiveConfig.
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from agent6.config.layer import (
    SECTION_ORDER,
    EffectiveConfig,
    Layer,
    flatten_leaves,
)


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
    leaves = flatten_leaves(eff.config.model_dump(mode="python"))
    by_section: dict[str, list[str]] = {}
    for leaf in leaves:
        by_section.setdefault(leaf.split(".", 1)[0], []).append(leaf)
    ordered = [s for s in SECTION_ORDER if s in by_section]
    ordered += [s for s in by_section if s not in SECTION_ORDER]

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


def render_key_detail(
    eff: EffectiveConfig,
    key: str,
    *,
    resolved: dict[str, Any] | None = None,
    color: bool = False,
) -> str | None:
    """Render one config leaf (or a whole section prefix) UNTRUNCATED, for
    ``agent6 config show <key>``: the full-width table clips long values (e.g. a
    verify_command), so a single-key view prints the whole value plus its source,
    default, and choices. Returns None when nothing matches *key*."""
    view = build_config_view(eff, resolved=resolved)
    matched = [s for s in view.settings if s.key == key or s.key.startswith(key + ".")]
    if not matched:
        return None
    lines: list[str] = []
    for s in matched:
        value = (
            f"{_fmt_value(s.effective_value)}  (adaptive)" if s.is_adaptive else _fmt_value(s.value)
        )
        header = f"{'*' if s.modified else ' '} {s.key}"
        lines.append(f"\x1b[1m{header}\x1b[0m" if color else header)
        lines.append(f"    value:   {value}")
        if s.modified:
            lines.append(f"    default: {_fmt_value(s.default)}")
        lines.append(f"    source:  {s.source}")
        if s.choices:
            lines.append(f"    choices: {', '.join(str(c) for c in s.choices)}")
    return "\n".join(lines) + "\n"


def render_show(
    eff: EffectiveConfig,
    *,
    as_json: bool = False,
    resolved: dict[str, Any] | None = None,
    color: bool = False,
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
    compaction sized from the worker model); the caller computes it. *color*
    dims the default rows (tty only; the caller passes ``isatty()``) so the
    ``*`` operator-set rows stand out.
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
            row = f"{mark} {short_key:<{key_w}} {short_val:<{val_w}} {s.source}"
            # Dim the default rows so the `*` operator-set values stand out of
            # what is otherwise a long dump of built-in defaults.
            lines.append(f"\x1b[2m{row}\x1b[0m" if color and not s.modified else row)
        lines.append("")
    legend_layers = ", ".join(
        f"{lyr.name}={lyr.path}" for lyr in view.layers if lyr.path is not None
    )
    lines.append("source: default | " + (legend_layers or "(no config files; all defaults)"))
    lines.append("* = overrides the built-in default")
    return "\n".join(lines).rstrip("\n") + "\n"
