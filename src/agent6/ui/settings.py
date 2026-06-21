# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""UI-only preferences for the TUI (currently just the theme).

Stored in ``<global-config-dir>/ui.toml`` — a sibling of ``config.toml`` and
``secrets.toml``, NOT part of the agent config. A theme is a viewer preference,
not agent behavior, so it must never go through the config schema or the
(shareable, per-repo) config layers. This module is the whole contract:

    get_theme() / save_theme(name)

Everything is best-effort: a missing, unreadable, or corrupt ``ui.toml`` simply
degrades to the default — a UI preference must never break the TUI. Writes are
atomic and ``chown``-ed back to the real user under sudo (same idiom as
``secrets.py``); there's deliberately no ``tomli_w`` dependency, the writer is a
tiny hand-rolled serializer for the one flat ``[ui]`` table.
"""

from __future__ import annotations

import tomllib
from typing import Any

from agent6.paths import RealUser, chown_to_real_user, effective_user, ui_settings_path

DEFAULT_THEME = "agent6-dark"


def load_ui_settings(user: RealUser | None = None) -> dict[str, Any]:
    """Read ``ui.toml``; ``{}`` if absent, unreadable, or corrupt (never raises)."""
    path = ui_settings_path(user)
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def get_theme(default: str = DEFAULT_THEME) -> str:
    """The persisted theme name, or *default* when unset/invalid."""
    ui = load_ui_settings().get("ui")
    name = ui.get("theme") if isinstance(ui, dict) else None
    return name if isinstance(name, str) and name else default


def save_theme(name: str, user: RealUser | None = None) -> None:
    """Persist ``[ui].theme = name`` atomically. Best-effort: a failed save is
    swallowed so it can never break the UI."""
    user = user or effective_user()
    path = ui_settings_path(user)
    data = load_ui_settings(user)
    ui = data.get("ui")
    if not isinstance(ui, dict):
        ui = {}
    ui["theme"] = name
    data["ui"] = ui
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(_render_ui_toml(data), encoding="utf-8")
        tmp.replace(path)
        chown_to_real_user(path.parent, user)
        chown_to_real_user(path, user)
    except OSError:
        pass  # a theme is not worth a crash


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _render_ui_toml(data: dict[str, Any]) -> str:
    """Render the flat ``[ui]`` table back to TOML (no ``tomli_w`` dependency)."""
    lines = ["# agent6 UI preferences (theme, etc.). Written by the TUI.", ""]
    ui = data.get("ui")
    if isinstance(ui, dict) and ui:
        lines.append("[ui]")
        for key in sorted(ui):
            value = ui[key]
            if isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{_toml_escape(value)}"')
            elif isinstance(value, int):
                lines.append(f"{key} = {value}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
