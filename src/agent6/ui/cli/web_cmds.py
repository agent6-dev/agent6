# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 web`: serve the browser front-end over the `[web]` bind, loopback by default.

Resolves host and port from config, lets `--host` and `--port` override, resolves the
sign-in secret (`--secret`, else `AGENT6_WEB_SECRET`, else a minted one), and hands off
to `web.run_web`.
"""

from __future__ import annotations

import os
import pathlib

import pydantic

from agent6.config import WebConfig, is_loopback_host, layer
from agent6.sessions import id
from agent6.ui.cli import common
from agent6.ui.web import auth, model, run_web


def _cmd_web(
    target: str,
    *,
    config_paths: tuple[pathlib.Path, ...],
    host: str | None,
    port: int | None,
    allow_non_loopback: bool,
    secret: str | None,
) -> int:
    """Serve the web UI.

    A non-loopback `[web].host` needs `[web].allow_non_loopback = true` at config load.
    `--allow-non-loopback` opts in a non-loopback `--host` for this invocation.
    Prefer `tailscale serve` in front of a loopback bind.

    Args:
        target: The session id or prefix, or machine name to open first.
        config_paths: The `--config` files in command-line order, if any.
        host: The `--host` override.
        port: The `--port` override.
        allow_non_loopback: The `--allow-non-loopback` opt-in.
        secret: The `--secret` value; None reads `AGENT6_WEB_SECRET`, else mints one.

    Returns:
        The exit code; 2 for a bad target, port, unapproved non-loopback host or short secret.
    """
    cwd = pathlib.Path.cwd()
    if target:
        try:
            target = common.resolve_session_layout(cwd, target).session_id
        except id.SessionIdError as exc:
            if not exc.no_match or model.machine_dir_for(cwd, target) is None:
                common.error(str(exc))
                return 2
    eff = layer.load_effective(cwd, config_paths)
    web = eff.config.web
    eff_host = host if host is not None else web.host
    eff_port = port if port is not None else web.port
    try:
        # The leaf's own bounds; allow_non_loopback=True leaves the host refusal to the check below.
        WebConfig(host=eff_host, port=eff_port, allow_non_loopback=True)
    except pydantic.ValidationError:
        common.error(f"--port {eff_port} is out of range (1-65535).")
        return 2
    if not is_loopback_host(eff_host) and not (allow_non_loopback or web.allow_non_loopback):
        common.refuse(
            f"binding non-loopback host {eff_host!r} requires opt-in."
            " Pass --allow-non-loopback (or set [web].allow_non_loopback = true), and prefer"
            " `tailscale serve` in front of a 127.0.0.1 bind."
        )
        return 2
    try:
        web_secret = auth.resolve(secret, os.environ.get(auth.SECRET_ENV))
    except auth.WebSecretError as exc:
        common.refuse(str(exc))
        return 2
    return run_web(
        target, host=eff_host, port=eff_port, secret=web_secret, config_paths=config_paths
    )
