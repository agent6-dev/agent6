# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The TUI's own preferences: the theme and the copy method.

Stored in `<global-config-dir>/ui.toml`, beside `config.toml`, never in the agent
config: a viewer preference is not agent behaviour and never goes through the
shareable config layers. Everything is best-effort, since a preference must never
break the TUI: a missing or corrupt file reads as the defaults, and a failed save
is swallowed. Writes are atomic and chowned back to the real user under sudo.
"""

from __future__ import annotations

import tomllib
from typing import Any

from agent6.paths import (
    RealUser,
    chown_to_real_user,
    effective_user,
    mkdir_for_real_user,
    ui_settings_path,
)
from agent6.portable import atomic_write, toml_basic_string

DEFAULT_THEME = "agent6-dark"
DEFAULT_COPY_METHOD = "auto"


def load_ui_settings(user: RealUser | None = None) -> dict[str, Any]:
    """Return the parsed `ui.toml`, or {} when absent, unreadable or corrupt."""
    path = ui_settings_path(user)
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def get_theme(default: str = DEFAULT_THEME) -> str:
    """Return the persisted theme name, or the default when unset."""
    ui = load_ui_settings().get("ui")
    name = ui.get("theme") if isinstance(ui, dict) else None
    return name if isinstance(name, str) and name else default


def _save_ui_key(key: str, value: str, user: RealUser | None = None) -> None:
    """Persist one `[ui]` key atomically; a failed save is swallowed."""
    user = user or effective_user()
    path = ui_settings_path(user)
    data = load_ui_settings(user)
    ui = data.get("ui")
    if not isinstance(ui, dict):
        ui = {}
    ui[key] = value
    data["ui"] = ui
    try:
        mkdir_for_real_user(path.parent, user)
        # mkstemp (O_EXCL, unpredictable name): a planted `ui.toml.tmp` symlink cannot redirect
        # the write, which can run as root under sudo; a fixed `.tmp` would follow it.
        atomic_write(path, _render_ui_toml(data))
        chown_to_real_user(path.parent, user)
        chown_to_real_user(path, user)
    except OSError:
        pass


def save_theme(name: str, user: RealUser | None = None) -> None:
    """Persist the theme name."""
    _save_ui_key("theme", name, user)


def get_copy_method(default: str = DEFAULT_COPY_METHOD) -> str:
    """Return the persisted copy method, or the default when unset."""
    ui = load_ui_settings().get("ui")
    name = ui.get("copy_method") if isinstance(ui, dict) else None
    return name if isinstance(name, str) and name else default


def save_copy_method(name: str, user: RealUser | None = None) -> None:
    """Persist the copy method."""
    _save_ui_key("copy_method", name, user)


def _render_ui_toml(data: dict[str, Any]) -> str:
    """Return the flat `[ui]` table as TOML; a hand serializer, so no `tomli_w` dependency."""
    lines = ["# agent6 UI preferences (theme, etc.). Written by the TUI.", ""]
    ui = data.get("ui")
    if isinstance(ui, dict) and ui:
        lines.append("[ui]")
        for key in sorted(ui):
            value = ui[key]
            if isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            elif isinstance(value, str):
                lines.append(f"{key} = {toml_basic_string(value)}")
            elif isinstance(value, int):
                lines.append(f"{key} = {value}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
