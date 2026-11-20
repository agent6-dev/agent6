# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Project non-default config values onto read-only execution evidence."""

from __future__ import annotations

import functools
import math
from typing import Any, get_args

import pydantic
import pydantic_core

from agent6.config import Config, io, layer_data
from agent6.sessions import journal
from agent6.viewmodel import config_view

_Value = str | float | int | bool | None
_SOURCES = frozenset((*get_args(layer_data.LayerName), "execution", "mode"))


def _disclosed_value(
    schema: config_view.FieldSchema, value: object, *, local_name: bool = False
) -> _Value:
    if not schema.mask_value and (
        value is None
        or isinstance(value, int)
        or (isinstance(value, float) and math.isfinite(value))
        or (isinstance(value, str) and (local_name or value in (schema.choices or ())))
    ):
        return value
    return "***"


def observations(
    cfg: Config, effective: layer_data.EffectiveConfig | None
) -> tuple[journal.SettingObservation, ...]:
    """Select settings for the journal without copying credentials or free-form text.

    Args:
        cfg: The config this execution uses.
        effective: The config before execution policy and the origins of its values.

    Returns:
        One observation per non-default setting. Arbitrary strings and maps stay masked.
    """
    defaults = layer_data.config_leaves(Config())
    original = layer_data.config_leaves(effective.config) if effective else {}
    rows: list[journal.SettingObservation] = []
    for key, value in sorted(layer_data.config_leaves(cfg).items()):
        parts = io.split_key(key)
        schema = config_view.field_schema(Config, parts) or config_view.FieldSchema()
        if value == defaults.get(key, schema.default):
            continue
        origin = effective.origin(key) if effective else None
        source = effective.sources.get(key, "default") if effective else "execution"
        file = str(origin.path) if origin and origin.path else None
        if source == "preset":
            source = f"preset {cfg.agent6.preset}"
        elif source.startswith("flag ("):
            source = "flag"
        if effective and key in effective.applied_sources:
            source, file = effective.applied_sources[key], None
        if key in original and original[key] != value:
            source, file = "execution", None
        if key not in defaults and source == "default" and key not in original:
            source = "execution"
        local_name = key == "agent6.preset" or (parts[0] == "models" and parts[-1] == "model")
        shown = _disclosed_value(schema, value, local_name=local_name)
        rows.append(journal.SettingObservation(key=key, value=shown, source=source, file=file))
    return tuple(rows)


@functools.cache
def _fixed_fields(model: type[pydantic.BaseModel]) -> dict[str, pydantic_core.SchemaValidator]:
    """Index fields with schema-defined names. Mapping entry names can contain secrets.

    Args:
        model: The config section.

    Returns:
        Field validators by canonical dotted key, without descending into mappings.
    """
    # The compiled field schema includes @field_validator rules that the annotation omits.
    section: Any = model.__pydantic_core_schema__
    definitions = section.get("definitions", [])
    while section["type"] != "model-fields":
        section = section["schema"]
    fields: dict[str, pydantic_core.SchemaValidator] = {}
    for name, field in model.model_fields.items():
        if field.exclude:
            continue
        annotation = field.annotation
        members = get_args(annotation)
        if len(members) == 2 and type(None) in members:
            annotation = next(member for member in members if member is not type(None))
        if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
            fields.update(
                {f"{name}.{key}": leaf for key, leaf in _fixed_fields(annotation).items()}
            )
        else:
            leaf = section["fields"][name]["schema"]
            if definitions:
                leaf = pydantic_core.core_schema.definitions_schema(leaf, definitions)
            fields[name] = pydantic_core.SchemaValidator(leaf, {"allow_inf_nan": False})
    return fields


def export_settings(rows: list[object]) -> list[dict[str, _Value]]:
    """Project stored observations onto the settings log without trusting journal text.

    Args:
        rows: The untrusted settings list from a journal event.

    Returns:
        Fixed config keys, their disclosed scalar values and source categories.
    """
    fields = _fixed_fields(Config)
    result: list[dict[str, _Value]] = []
    for item in rows:
        try:
            row = journal.SettingObservation.model_validate(item, strict=True)
        except pydantic.ValidationError:
            continue
        source = "preset" if row.source.startswith("preset ") else row.source
        if source not in _SOURCES or row.key not in fields:
            continue
        schema = config_view.field_schema(Config, row.key.split("."))
        if schema is None:
            continue
        value = _disclosed_value(schema, row.value)
        if value != "***":
            try:
                checked = fields[row.key].validate_python(value, strict=True)
            except pydantic.ValidationError:
                value = "***"
            else:
                # Strict float fields accept integers, even when conversion rounds them.
                value = checked if checked == value else "***"
        result.append({"key": row.key, "source": source, "value": value})
    return result
