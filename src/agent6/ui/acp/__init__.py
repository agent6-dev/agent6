# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Serve the Agent Client Protocol: agent6 driven by an editor.

A front-end: it may import every layer and the shared `ui` helpers, never another front-end.
It reports through the viewmodel fold and acts through `SessionFrontend`, like the other three.
ACP's `initialize` is a capability exchange, so it maps onto `FrontendCapabilities`.
Every path is absolute and line numbers are 1-based, as the spec states.
ACP's client-owned `fs/*` and `terminal/*` are not adopted: the operator's jail does that work.
Main modules: `server`, `session`, `prompt`, `runner`, `stream`, `updates`, `model`, `wire`.
"""

from __future__ import annotations

from agent6.ui.acp.runner import serve_acp  # re-export
from agent6.ui.acp.server import ACPServer  # re-export

__all__ = ["ACPServer", "serve_acp"]
