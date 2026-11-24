# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Serve the browser front-end.

A front-end: it may import every layer and the shared `ui` helpers, never another front-end.
`agent6 web [target]` serves one page from a stdlib `http.server`, fed by JSON and SSE endpoints.
The endpoints use the folds (`agent6.viewmodel`), spawn (`agent6.ui.spawn`) and answer files
(`agent6.sessions.ipc`) every other front-end uses.
The server binds loopback, and a non-loopback bind is opt-in under `[web]`.
Remote access is expected behind `tailscale serve`.
No secret is ever served.
Main modules: `server`, `page`, `model`, `actions`, `sse`.
"""

from __future__ import annotations

from agent6.ui.web.server import run_web  # re-export

__all__ = ["run_web"]
