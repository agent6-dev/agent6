# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the environment a process spawned outside the jail inherits.

A leaf, since its callers sit on both sides of the layering; one owner, so
their claims cannot drift. Jailed commands get a narrower env from the policy.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

# Enough to execute a program; the whole environment carries the provider keys.
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

# The operator's desktop session. A confined child must not have these: the session bus reaches
# `systemd --user`, which is unconfined and runs commands on the caller's behalf, and Landlock
# does not gate `connect()` to a unix socket.
_DESKTOP = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "DBUS_SESSION_BUS_ADDRESS",
    "XDG_RUNTIME_DIR",
)


# The `api_key_env` names this run resolves keys from; no child needs one. Mutated, never rebound.
_provider_key_env: set[str] = set()


def set_provider_key_env(names: Iterable[str]) -> None:
    """Register the provider-key variable names agent6 keeps out of every child."""
    _provider_key_env.clear()
    _provider_key_env.update(n for n in names if n)


def without_provider_keys(env: dict[str, str]) -> dict[str, str]:
    """Return the environment minus the registered provider-key names."""
    return {k: v for k, v in env.items() if k not in _provider_key_env}


def curated_env(
    *,
    passthrough: tuple[str, ...] = (),
    extra: dict[str, str] | None = None,
    desktop: bool = True,
) -> dict[str, str]:
    """Return the base environment, plus passed-through names and extra values.

    Args:
        passthrough: Variables the operator named in config for this child, one at a
            time, so a provider key reaches a child only when written down.
        extra: Values set outright.
        desktop: Keep the session-bus and display addresses; a confined child gets
            none, since they reach unconfined processes.

    Returns:
        The environment.
    """
    keep = (*_KEEP, *_DESKTOP) if desktop else _KEEP
    env = {k: v for k in keep if (v := os.environ.get(k)) is not None}
    env.update({k: v for k in passthrough if (v := os.environ.get(k)) is not None})
    env.update(extra or {})
    return env
