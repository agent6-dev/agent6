# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The environment a process agent6 spawns OUTSIDE the jail inherits.

A leaf, because the two callers sit on opposite sides of the layering: the
operator's notify hooks (``app``) and the MCP servers (``tools``). One owner,
so their env-scope claims cannot drift apart.

Jailed commands do not come here -- ``sandbox.jail`` builds their env from the
policy, which is narrower still.
"""

from __future__ import annotations

import os

# Enough to execute a program (PATH, HOME, locale, user identity) and to reach
# the desktop bus (notify-send needs DISPLAY/DBUS). Never the whole
# environment: it carries the provider API keys resolved via
# `[providers.*].api_key_env`, and a child that logs or forwards its env -- a
# shell wrapper, a webhook poster, an MCP server -- would carry the key with it.
_KEEP = (
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "DBUS_SESSION_BUS_ADDRESS",
    "XDG_RUNTIME_DIR",
)


def curated_env(
    *, passthrough: tuple[str, ...] = (), extra: dict[str, str] | None = None
) -> dict[str, str]:
    """The base environment, plus *passthrough* names and *extra* values.

    ``passthrough`` is how an operator hands one child a variable it genuinely
    needs (an MCP server's API token). Named one at a time in config, because
    naming each one is the point: a provider key is never among them, since
    nobody would write it down.
    """
    env = {k: v for k in _KEEP if (v := os.environ.get(k)) is not None}
    env.update({k: v for k in passthrough if (v := os.environ.get(k)) is not None})
    env.update(extra or {})
    return env
