# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 web` command: serve the browser front-end over loopback (or an
operator-chosen bind). Thin wrapper around `web.run_web`."""

from __future__ import annotations

from agent6.web import run_web

# Loopback + a fixed default port until the `[web]` config section lands.
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8901


def _cmd_web(target: str, *, host: str, port: int) -> int:
    """Serve the web UI on host:port. Binds loopback by default."""
    return run_web(target, host=host, port=port)
