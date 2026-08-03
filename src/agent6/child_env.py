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

# Enough to execute a program. Never the whole environment: it carries the
# provider API keys resolved via `[providers.*].api_key_env`, and a child that
# logs or forwards its env -- a shell wrapper, a webhook poster, an MCP server
# -- would carry the key with it.
_KEEP = (
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
)

# How to reach the operator's desktop session. A notify hook needs these --
# `notify-send` talks to the session bus. A CONFINED child must not have them:
# the session bus reaches `systemd --user`, which is NOT confined and will
# gladly run a command on the caller's behalf. Landlock gates filesystem
# paths, not `connect()` to a unix socket, so a server denied `/etc/passwd`
# directly could still have systemd read it and write the result anywhere.
# Proved end to end before this split existed.
_DESKTOP = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "DBUS_SESSION_BUS_ADDRESS",
    "XDG_RUNTIME_DIR",
)


def curated_env(
    *,
    passthrough: tuple[str, ...] = (),
    extra: dict[str, str] | None = None,
    desktop: bool = True,
) -> dict[str, str]:
    """The base environment, plus *passthrough* names and *extra* values.

    ``passthrough`` is how an operator hands one child a variable it genuinely
    needs (an MCP server's API token). Named one at a time in config, because
    naming each one is the point: a provider key is never among them, since
    nobody would write it down.

    ``desktop=False`` also drops the session-bus and display addresses. Pass it
    for a child that is meant to be CONFINED: those addresses reach processes
    that are not, and delegating to one walks straight out of any sandbox.
    """
    keep = (*_KEEP, *_DESKTOP) if desktop else _KEEP
    env = {k: v for k in keep if (v := os.environ.get(k)) is not None}
    env.update({k: v for k in passthrough if (v := os.environ.get(k)) is not None})
    env.update(extra or {})
    return env
