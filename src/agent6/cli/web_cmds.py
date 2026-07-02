# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 web` command: serve the browser front-end over the `[web]` bind
(loopback by default). Thin wrapper: resolve host/port from config, let
`--host`/`--port` override, then hand off to `web.run_web`."""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.config import ConfigError
from agent6.config_layer import load_effective
from agent6.web import run_web


def _cmd_web(
    target: str,
    *,
    config_path: Path | None,
    host: str | None,
    port: int | None,
) -> int:
    """Serve the web UI. `--host`/`--port` override the `[web]` config section.

    A non-loopback host set in config is already gated by `[web].allow_non_loopback`
    at load time; a `--host` flag overriding it here inherits the same intent (the
    operator typed it explicitly), and `run_web` warns on any non-loopback bind."""
    try:
        eff = load_effective(Path.cwd(), config_path)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    web = eff.config.web
    return run_web(
        target,
        host=host if host is not None else web.host,
        port=port if port is not None else web.port,
    )
