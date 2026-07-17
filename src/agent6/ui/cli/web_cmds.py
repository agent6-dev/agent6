# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 web` command: serve the browser front-end over the `[web]` bind
(loopback by default). Thin wrapper: resolve host/port from config, let
`--host`/`--port` override, then hand off to `web.run_web`."""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.config import is_loopback_host
from agent6.ui.cli._common import load_config_or_exit
from agent6.ui.web import run_web


def _cmd_web(
    target: str,
    *,
    config_path: Path | None,
    host: str | None,
    port: int | None,
    allow_non_loopback: bool,
) -> int:
    """Serve the web UI. `--host`/`--port` override the `[web]` config section.

    A non-loopback bind is gated the same whether it comes from `[web].host`
    (refused at config load) or `--host` (refused here): both need the opt-in,
    either `--allow-non-loopback` or `[web].allow_non_loopback = true`. Prefer
    `tailscale serve` in front of a loopback bind over any raw non-loopback bind."""
    eff = load_config_or_exit(Path.cwd(), config_path)
    if isinstance(eff, int):
        return eff
    web = eff.config.web
    eff_host = host if host is not None else web.host
    eff_port = port if port is not None else web.port
    if not is_loopback_host(eff_host) and not (allow_non_loopback or web.allow_non_loopback):
        print(
            f"agent6 web: refusing to bind non-loopback host {eff_host!r} without opt-in."
            " Pass --allow-non-loopback (or set [web].allow_non_loopback = true), and prefer"
            " `tailscale serve` in front of a 127.0.0.1 bind.",
            file=sys.stderr,
        )
        return 2
    return run_web(target, host=eff_host, port=eff_port)
