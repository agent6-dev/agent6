# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 web`: serve the browser front-end over the `[web]` bind, loopback by default.

Resolves host and port from config, lets `--host` and `--port` override, and hands off to
`web.run_web`.
"""

from __future__ import annotations

import pathlib

import pydantic

from agent6.config import WebConfig, is_loopback_host, layer
from agent6.ui.cli import _common
from agent6.ui.web import run_web


def _cmd_web(
    target: str,
    *,
    config_path: pathlib.Path | None,
    host: str | None,
    port: int | None,
    allow_non_loopback: bool,
) -> int:
    """Serve the web UI.

    A non-loopback bind is gated the same whether it comes from `[web].host` (refused at
    config load) or `--host` (refused here): both need `--allow-non-loopback` or
    `[web].allow_non_loopback = true`. Prefer `tailscale serve` in front of a loopback bind.

    Args:
        target: The session or machine to open first.
        config_path: The `--config` file, if any.
        host: The `--host` override.
        port: The `--port` override.
        allow_non_loopback: The `--allow-non-loopback` opt-in.

    Returns:
        The exit code; 2 for a bad port or an unapproved non-loopback host.
    """
    eff = layer.load_effective(pathlib.Path.cwd(), config_path)
    web = eff.config.web
    eff_host = host if host is not None else web.host
    eff_port = port if port is not None else web.port
    try:
        # The leaf's own bounds; allow_non_loopback=True leaves the host refusal to the check below.
        WebConfig(host=eff_host, port=eff_port, allow_non_loopback=True)
    except pydantic.ValidationError:
        _common.error(f"--port {eff_port} is out of range (1-65535).")
        return 2
    if not is_loopback_host(eff_host) and not (allow_non_loopback or web.allow_non_loopback):
        _common.refuse(
            f"binding non-loopback host {eff_host!r} requires opt-in."
            " Pass --allow-non-loopback (or set [web].allow_non_loopback = true), and prefer"
            " `tailscale serve` in front of a 127.0.0.1 bind."
        )
        return 2
    return run_web(target, host=eff_host, port=eff_port, config_path=config_path)
