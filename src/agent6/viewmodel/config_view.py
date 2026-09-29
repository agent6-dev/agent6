# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shape the effective config into the one view `config show`, the TUI and the web render.

Provenance, defaults, enum choices and adaptive resolution live here once, so the
renderers stay thin. Loading, merging and writing config stays in `agent6.config.layer`.
"""

from __future__ import annotations

import json
import textwrap
import types
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from agent6.config.layer import (
    SECTION_ORDER,
    EffectiveConfig,
    Layer,
    config_leaves,
    preset_names,
)


@dataclass(frozen=True, slots=True)
class ConfigSetting:
    """One config leaf, described for display and editing.

    Attributes:
        key: The dotted leaf path, such as "sandbox.run_commands".
        section: The top-level section, such as "sandbox".
        value: The raw effective value; None for an unset or adaptive setting.
        effective_value: The resolved value; equal to `value` unless a caller resolved it.
        default: The built-in default; None when unknown or not static.
        source: The layer that set it: default, preset, global, repo, flag or machine.
        modified: A layer set it, so `source` is not "default".
        is_adaptive: `effective_value` was resolved away from the raw value.
        py_type: str, int, bool, float, choice, list or table.
        choices: The enum options for a picker, else None.
        description: The leaf's operator-facing meaning, the docs table cell.
    """

    key: str
    section: str
    value: Any
    effective_value: Any
    default: Any
    source: str
    modified: bool
    is_adaptive: bool
    py_type: str
    choices: tuple[str, ...] | None
    description: str


@dataclass(frozen=True, slots=True)
class ConfigView:
    """The whole effective config as a flat, section-ordered list of settings.

    Attributes:
        settings: Every leaf, section by section.
        sections: The section names in display order.
        layers: The contributing layers, for a provenance legend.
    """

    settings: tuple[ConfigSetting, ...]
    sections: tuple[str, ...]
    layers: tuple[Layer, ...]


def _unwrap_optional(ann: Any) -> Any:
    """Return `X` for `X | None`, leaving a multi-member union untouched."""
    if get_origin(ann) in (Union, types.UnionType):
        non_none = [a for a in get_args(ann) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return ann


def _literal_choices(ann: Any) -> tuple[str, ...] | None:
    """Return a Literal annotation's members as strings, None for any other annotation."""
    ann = _unwrap_optional(ann)
    if get_origin(ann) is Literal:
        return tuple(str(a) for a in get_args(ann))
    return None


def _nested_model(ann: Any) -> type[BaseModel] | None:
    """Return the pydantic model an annotation names, None for any other annotation."""
    ann = _unwrap_optional(ann)
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann
    return None


def _value_models(ann: Any) -> tuple[type[BaseModel], ...]:
    """Return the models a `dict` field's values take.

    Args:
        ann: The field's annotation.

    Returns:
        One model for a plain model value, several for a discriminated union (the
        provider entries), empty when the value is not model-shaped.
    """
    ann = _unwrap_optional(ann)
    if get_origin(ann) is not dict:
        return ()
    val = get_args(ann)[1]
    if get_origin(val) is Annotated:
        val = get_args(val)[0]
    if get_origin(val) in (Union, types.UnionType):
        return tuple(m for m in get_args(val) if _nested_model(m) is not None)
    model = _nested_model(val)
    return (model,) if model is not None else ()


def _merge_field_schema(
    models: tuple[type[BaseModel], ...], parts: list[str]
) -> tuple[str, tuple[str, ...] | None, Any, str] | None:
    """Resolve a field across discriminated-union members, merging their choices.

    A field shared by every member keeps its choices; the discriminator itself becomes
    the union of every member's literal.

    Args:
        models: The union's members.
        parts: The dotted path below the union.

    Returns:
        `(py_type, choices, default, description)`, or None when no member has the field.
    """
    results = [r for m in models if (r := _field_schema(m, parts)) is not None]
    if not results:
        return None
    if len(results) == 1:
        return results[0]
    choices: list[str] = []
    for _, member_choices, _default, _desc in results:
        for c in member_choices or ():
            if c not in choices:
                choices.append(c)
    merged = tuple(choices) or None
    py_type = "choice" if merged else results[0][0]
    default = next((d for _, _, d, _ in results if d is not None), results[0][2])
    description = next((d for _, _, _, d in results if d), "")
    return py_type, merged, default, description


def _type_label(ann: Any) -> str:
    """Return the `py_type` word for an annotation."""
    ann = _unwrap_optional(ann)
    origin = get_origin(ann)
    if origin is Literal:
        return "choice"
    if origin in (list, tuple):
        return "list"
    if origin is dict:
        return "table"
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        # An unset optional section groups with the other structured leaves.
        return "table"
    if isinstance(ann, type):
        return ann.__name__
    return "str"


def _field_schema(
    model_cls: type[BaseModel], parts: list[str]
) -> tuple[str, tuple[str, ...] | None, Any, str] | None:
    """Walk a model down a dotted path to a leaf field's schema.

    Args:
        model_cls: The model to start from.
        parts: The dotted path's parts.

    Returns:
        `(py_type, choices, default, description)`, or None when the path cannot be
        resolved (a dynamic key whose value model is unknown).
    """
    name = parts[0]
    fi = getattr(model_cls, "model_fields", {}).get(name)
    if fi is None:
        return None
    ann = fi.annotation
    if len(parts) == 1:
        default = fi.default
        if default is PydanticUndefined:
            default = fi.default_factory() if fi.default_factory is not None else None
        return _type_label(ann), _literal_choices(ann), default, fi.description or ""
    nested = _nested_model(ann)
    if nested is not None:
        return _field_schema(nested, parts[1:])
    models = _value_models(ann)
    if models and len(parts) >= 3:
        return _merge_field_schema(models, parts[2:])
    return None


def _configured_choices(eff: EffectiveConfig, leaf: str) -> tuple[str, ...] | None:
    """Return the choices only the effective config can state.

    Every editor's picker and TAB completion read them from here; the model ids of a
    `models.<role>.model` leaf are a fetch, not a choice.

    Args:
        eff: The effective config.
        leaf: The dotted leaf path.

    Returns:
        The preset names for `preset`, the configured provider names for a
        `models.<role>.provider` leaf, else None.
    """
    if leaf == "preset":
        return eff.presets or tuple(preset_names(eff.layers))
    parts = leaf.split(".")
    if len(parts) == 3 and parts[0] == "models" and parts[2] == "provider":
        return tuple(sorted(eff.config.providers)) or None
    return None


def build_config_view(
    eff: EffectiveConfig, *, resolved: dict[str, Any] | None = None
) -> ConfigView:
    """Build the flat view every UI renders from the effective config.

    Args:
        eff: The effective config with its provenance.
        resolved: A dotted key to its resolved value, for settings whose raw value
            stands in for a runtime resolution (compaction sized from the model's
            context window); it fills `effective_value` and `is_adaptive` only, never
            provenance or the modified flag.

    Returns:
        The view.
    """
    resolved = resolved or {}
    leaves = config_leaves(eff.config)
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
            if schema is None:
                schema = ("str", None, None, "")
            py_type, choices, default, description = schema
            if choices is None:
                choices = _configured_choices(eff, leaf)
                if choices is not None:
                    py_type = "choice"
            eff_val = resolved.get(leaf, value)
            settings.append(
                ConfigSetting(
                    key=leaf,
                    section=section,
                    value=value,
                    effective_value=eff_val,
                    default=default,
                    source=source,
                    # Provenance, not "differs from the default": a layer may pin that value.
                    modified=source != "default",
                    is_adaptive=leaf in resolved and eff_val != value,
                    py_type=py_type,
                    choices=choices,
                    description=description,
                )
            )
    return ConfigView(settings=tuple(settings), sections=tuple(ordered), layers=eff.layers)


# Rendering: `config show`.


def format_value(val: Any) -> str:
    """Return a config leaf value as every surface prints it.

    Args:
        val: The raw value.

    Returns:
        `(unset)` for None, `(empty)` for an empty string, TOML booleans, `[a, b]`
        lists, `{...}` for a non-empty table.
    """
    if val is None:
        return "(unset)"
    if isinstance(val, str):
        return val or "(empty)"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(format_value(v) for v in val) + "]"
    if isinstance(val, dict):
        return "{...}" if val else "{}"
    return str(val)


def display_value(s: ConfigSetting) -> str:
    """Return the value column: the resolved value marked `(adaptive)`, else the raw value."""
    if s.is_adaptive:
        return f"{format_value(s.effective_value)}  (adaptive)"
    return format_value(s.value)


def _truncate(text: str, width: int) -> str:
    """Return the text cut to `width`, an ellipsis marking the cut."""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "\u2026"


def plain_description(description: str) -> str:
    """Return a leaf's description as a terminal shows it, markdown bold stripped."""
    return description.replace("**", "")


def _description_lines(description: str, indent: str) -> list[str]:
    """Return the description wrapped for a terminal at the given indent."""
    return textwrap.wrap(
        plain_description(description),
        width=92,
        initial_indent=indent,
        subsequent_indent=indent,
        break_on_hyphens=False,
        break_long_words=False,
    )


def _leaf_json(s: ConfigSetting) -> dict[str, Any]:
    """Return one leaf's JSON view, shared by the full dump and the single-key path.

    Args:
        s: The leaf.

    Returns:
        The leaf's fields, plus `display` and `default_display` as every surface prints
        them, so a client renders them verbatim.
    """
    return {
        "value": s.value,
        "effective": s.effective_value,
        "default": s.default,
        "source": s.source,
        "modified": s.modified,
        "adaptive": s.is_adaptive,
        "type": s.py_type,
        "choices": list(s.choices) if s.choices is not None else None,
        "description": s.description,
        "display": display_value(s),
        "default_display": format_value(s.default),
    }


def render_key_detail(
    eff: EffectiveConfig,
    keys: list[str],
    *,
    resolved: dict[str, Any] | None = None,
    color: bool = False,
    as_json: bool = False,
) -> str:
    """Render the leaves under the given keys untruncated, for `config show <key>...`.

    Args:
        eff: The effective config.
        keys: Leaf paths or section prefixes, in the order to print.
        resolved: The caller-resolved values, as for `build_config_view`.
        color: Bold the key and dim the meaning with ANSI.
        as_json: Print the leaves' JSON views instead.

    Returns:
        Each leaf's value, source, default, choices and meaning.

    Raises:
        KeyError: The first key that matches nothing.
    """
    view = build_config_view(eff, resolved=resolved)
    matched: list[ConfigSetting] = []
    for key in keys:
        hits = [s for s in view.settings if s.key == key or s.key.startswith(key + ".")]
        if not hits:
            raise KeyError(key)
        matched.extend(s for s in hits if s not in matched)
    if as_json:
        return json.dumps(
            {s.key: _leaf_json(s) for s in matched}, indent=2, sort_keys=True, default=str
        )
    lines: list[str] = []
    for s in matched:
        value = display_value(s)
        header = f"{'*' if s.modified else ' '} {s.key}"
        lines.append(f"\x1b[1m{header}\x1b[0m" if color else header)
        lines.append(f"    value:   {value}")
        if s.modified:
            lines.append(f"    default: {format_value(s.default)}")
        lines.append(f"    source:  {s.source}")
        if s.choices:
            lines.append(f"    choices: {', '.join(str(c) for c in s.choices)}")
        if s.description:
            wrapped = _description_lines(s.description, "             ")
            wrapped[0] = "    meaning: " + wrapped[0].lstrip()
            lines.extend(f"\x1b[2m{line}\x1b[0m" if color else line for line in wrapped)
    return "\n".join(lines) + "\n"


def render_show(
    eff: EffectiveConfig,
    *,
    as_json: bool = False,
    resolved: dict[str, Any] | None = None,
    color: bool = False,
    descriptions: bool = False,
) -> str:
    """Render the effective config and its provenance for `config show`.

    Args:
        eff: The effective config.
        as_json: Emit the full per-leaf JSON view instead of the table.
        resolved: The caller-resolved values, as for `build_config_view`.
        color: Dim the default rows with ANSI so the operator-set rows stand out.
        descriptions: Print each leaf's meaning wrapped under its row.

    Returns:
        A section-grouped, fixed-width table of key, value and source with a leading
        `*` on rows a layer set, then the provenance legend.
    """
    view = build_config_view(eff, resolved=resolved)
    if as_json:
        payload = {s.key: _leaf_json(s) for s in view.settings}
        return json.dumps(payload, indent=2, sort_keys=True, default=str)

    by_section: dict[str, list[ConfigSetting]] = {}
    for s in view.settings:
        by_section.setdefault(s.section, []).append(s)

    key_w = min(max((len(s.key) for s in view.settings), default=10) + 1, 40)
    val_w = 40
    lines: list[str] = []

    def emit(s: ConfigSetting) -> None:
        """Append one leaf's row, and its meaning when asked."""
        value = display_value(s)
        mark = "*" if s.modified else " "
        key, val = _truncate(s.key, key_w), _truncate(value, val_w)
        row = f"{mark} {key:<{key_w}} {val:<{val_w}} {s.source}"
        lines.append(f"\x1b[2m{row}\x1b[0m" if color and not s.modified else row)
        if descriptions and s.description:
            for line in _description_lines(s.description, "      "):
                lines.append(f"\x1b[2m{line}\x1b[0m" if color else line)

    # Top-level scalars first and headerless, as TOML requires, so the output pastes back.
    scalars = [s for s in view.settings if "." not in s.key]
    if scalars:
        for s in scalars:
            emit(s)
        lines.append("")
    for section in view.sections:
        rows = [s for s in by_section[section] if "." in s.key]
        if not rows:
            continue
        lines.append(f"[{section}]")
        for s in rows:
            emit(s)
        lines.append("")
    legend_layers = ", ".join(
        f"{lyr.name}={lyr.path}" for lyr in view.layers if lyr.path is not None
    )
    lines.append("source: default | " + (legend_layers or "(no config files; all defaults)"))
    lines.append("* = set by a config layer (see the source column)")
    for layer in eff.layers:
        # The source column's "flag" reads back to the file the operator typed.
        if layer.name == "flag" and layer.path is not None:
            lines.append(f"flag = {layer.path}")
    return "\n".join(lines).rstrip("\n") + "\n"
