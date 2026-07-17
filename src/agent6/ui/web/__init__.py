# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent6 web UI: the browser front-end.

`agent6 web [target]` serves a single page (web.page) from a stdlib
`http.server` (web.server), fed by JSON + SSE endpoints that fold the same
`<run>/logs.jsonl` and machine journals every other front-end reads (via
`agent6.viewmodel`) and driven by the same contracts the CLI and TUI use: the
detached spawn (`agent6.ui.spawn`) and the approval / steer answer files
(`agent6.runs.ipc`). It is a thin renderer of shared state.

Layout:
    model.py   pure JSON payload builders (hub / run / machine / conversation / config).
    server.py  ThreadingHTTPServer + routing + SSE (`run_web`).
    page.py    the embedded HTML/CSS/JS single-page app.

Secure by default: binds loopback; a non-loopback bind is opt-in via `[web]`
config and remote access is expected behind `tailscale serve` (the tailnet
identity is the access control). No secrets are ever served.
"""

from __future__ import annotations

from agent6.ui.web.server import run_web

__all__ = ["run_web"]
