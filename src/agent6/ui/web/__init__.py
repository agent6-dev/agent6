# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Serve the browser front-end.

`agent6 web [target]` serves one page from a stdlib `http.server`, fed by JSON and
SSE endpoints over the same folds (`agent6.viewmodel`), spawn (`agent6.ui.spawn`)
and answer files (`agent6.sessions.ipc`) every other front-end uses.

The server binds loopback; a non-loopback bind is opt-in under `[web]`, with remote
access expected behind `tailscale serve`. No secret is ever served.
"""

from __future__ import annotations

from agent6.ui.web.server import run_web  # re-export

__all__ = ["run_web"]
