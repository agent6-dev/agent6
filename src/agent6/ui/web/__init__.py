# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Serve the browser front-end.

`agent6 web [target]` serves one page from a stdlib `http.server`, fed by JSON and
SSE endpoints over the same folds (`agent6.viewmodel`), spawn (`agent6.ui.spawn`)
and answer files (`agent6.sessions.ipc`) every other front-end uses.

Layout:
    model.py   the JSON payload builders (hub, run, machine, conversation, config).
    actions.py the write side: answers, steers, spawns, the fixed-argv CLI bridge.
    server.py  the threading server, routing and POST bodies (`run_web`).
    _sse.py    the run and machine server-sent-event streams.
    page.py    the HTML, CSS and JS single-page app.

The server binds loopback; a non-loopback bind is opt-in under `[web]`, with remote
access expected behind `tailscale serve`. No secret is ever served.
"""

from __future__ import annotations

from agent6.ui.web.server import run_web

__all__ = ["run_web"]
