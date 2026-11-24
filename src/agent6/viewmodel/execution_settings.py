# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Project non-default config values onto read-only execution evidence."""

from __future__ import annotations

import functools
import math
from typing import Any, get_args

import pydantic
import pydantic_core

from agent6 import terminal
from agent6.config import io, layer_data
from agent6.config import model as config_model
from agent6.sessions import journal
from agent6.viewmodel import config_view

_Value = str | float | int | bool | None
_SOURCES = frozenset((*get_args(layer_data.LayerName), "execution", "mode"))


def summary(rows: tuple[journal.SettingObservation, ...]) -> str:
    """Summarize non-default settings for the CLI.

    Args:
        rows: The settings observed at one start.

    Returns:
        One line, with the full values and sources available in JSON.
    """
    defaults = {
        r.key.removeprefix("models.default."): r.value
        for r in rows
        if r.key.startswith("models.default.")
    }
    distinct = [
        r
        for r in rows
        if not (
            r.key.startswith("models.")
            and not r.key.startswith("models.default.")
            and defaults.get(r.key.rpartition(".")[2]) == r.value
        )
    ]
    parts: list[str] = []
    for row in sorted(distinct, key=lambda row: (not row.key.startswith("models."), row.key)):
        # Provider setup can include token commands. It belongs in the requested detail.
        if row.key.startswith("providers."):
            continue
        part = " ".join(terminal.scrub_terminal_controls(f"{row.key}={row.value}").split())
        if len(", ".join([*parts, part])) <= 96:
            parts.append(part)
    if omitted := len(distinct) - len(parts):
        parts.append(f"{omitted} more in sessions show --json")
    return ", ".join(parts)


def _recorded_value(schema: config_view.FieldSchema, value: object) -> _Value:
    """Return the value as `agent6 config show` prints it: masked where that masks it."""
    if schema.mask_value:
        return "***"
    if value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return config_view.format_value(value, display_safe=schema.display_safe)


def _exported_value(schema: config_view.FieldSchema, value: object) -> _Value:
    """Return the value telemetry may carry off the machine: numbers and the schema's choices."""
    if not schema.mask_value and (
        value is None
        or isinstance(value, int)
        or (isinstance(value, float) and math.isfinite(value))
        or (isinstance(value, str) and value in (schema.choices or ()))
    ):
        return value
    return "***"


def observations(
    cfg: config_model.Config, effective: layer_data.EffectiveConfig | None
) -> tuple[journal.SettingObservation, ...]:
    """Select the non-default settings for the journal, as `agent6 config show` prints them.

    Args:
        cfg: The config this execution uses.
        effective: The config before execution policy and the origins of its values.

    Returns:
        One observation per non-default setting; a value `config show` masks stays masked.
    """
    defaults = layer_data.config_leaves(config_model.Config())
    original = layer_data.config_leaves(effective.config) if effective else {}
    rows: list[journal.SettingObservation] = []
    for key, value in sorted(layer_data.config_leaves(cfg).items()):
        parts = io.split_key(key)
        schema = config_view.field_schema(config_model.Config, parts) or config_view.FieldSchema()
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
        shown = _recorded_value(schema, value)
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
    fields = _fixed_fields(config_model.Config)
    result: list[dict[str, _Value]] = []
    for item in rows:
        try:
            row = journal.SettingObservation.model_validate(item, strict=True)
        except pydantic.ValidationError:
            continue
        source = "preset" if row.source.startswith("preset ") else row.source
        if source not in _SOURCES or row.key not in fields:
            continue
        schema = config_view.field_schema(config_model.Config, row.key.split("."))
        if schema is None:
            continue
        value = _exported_value(schema, row.value)
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
