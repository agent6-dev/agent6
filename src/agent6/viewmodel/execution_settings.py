# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Project non-default config values onto read-only execution evidence."""

from __future__ import annotations

from agent6.config import Config, io, layer_data
from agent6.sessions import journal
from agent6.viewmodel import config_view


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
        shown: str | float | int | bool | None = "***"
        if not schema.mask_value and (
            value is None
            or isinstance(value, (bool, float, int))
            or (
                isinstance(value, str)
                and (
                    value in (schema.choices or ())
                    or parts[0] == "models"
                    or key == "agent6.preset"
                )
            )
        ):
            shown = value
        rows.append(journal.SettingObservation(key=key, value=shown, source=source, file=file))
    return tuple(rows)
