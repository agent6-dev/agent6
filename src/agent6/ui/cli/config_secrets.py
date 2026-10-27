# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The config secret target accepts telemetry header keys and never displays their values."""

from __future__ import annotations

import pathlib

import pydantic

from agent6 import secret_store
from agent6.config import ConfigError, TelemetryConfig, io, layer
from agent6.ui.cli import _common


def write_value(key: str, value: str | None) -> int:
    """Set or unset a telemetry header in the private secrets file.

    Args:
        key: A `telemetry.headers.<name>` dotted key.
        value: The literal header value; None removes it.

    Returns:
        Zero on success, 2 for a refused key, value or secrets file.
    """
    try:
        parts = io.split_key(key)
    except ConfigError:
        parts = []
    valid = len(parts) == 3 and parts[:2] == ["telemetry", "headers"]
    if valid:
        try:
            TelemetryConfig(headers=(parts[2],))
        except pydantic.ValidationError:
            valid = False
    if not valid:
        _common.error(
            "--secret accepts only telemetry.headers.<name> with a header name telemetry accepts"
        )
        return 2
    try:
        target = secret_store.write_telemetry_header(parts[2], value)
    except secret_store.SecretsError as exc:
        _common.error(str(exc))
        return 2
    verb = "Set" if value is not None else "Unset"
    print(f"{verb} {io.render_key(parts)} in {target}")
    return 0


def keys(config_paths: tuple[pathlib.Path, ...]) -> list[str]:
    """Return configured and saved telemetry header keys, never their values.

    Args:
        config_paths: The explicit config layers already typed on the command line.

    Returns:
        The sorted dotted keys, with non-bare header names quoted.
    """
    try:
        names = set(layer.load_effective(pathlib.Path.cwd(), config_paths).config.telemetry.headers)
    except ConfigError:
        names: set[str] = set()
    telemetry = secret_store.load_secrets().get("telemetry")
    headers = telemetry.get("headers") if isinstance(telemetry, dict) else None
    if isinstance(headers, dict):
        names.update(headers)
    accepted: list[str] = []
    for name in names:
        try:
            TelemetryConfig(headers=(name,))
        except pydantic.ValidationError:
            continue
        accepted.append(io.render_key(("telemetry", "headers", name)))
    return sorted(accepted)
