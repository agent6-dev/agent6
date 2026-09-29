# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Serve the Agent Client Protocol: agent6 driven by an editor.

The front-end reports through the viewmodel fold and acts through
`SessionFrontend` and `FrontendCapabilities`, like the other three; ACP's
`initialize` is a capability exchange, so it maps onto the second. Every path is
absolute and line numbers are 1-based, as the spec states.

ACP's client-owned filesystem and terminal (`fs/*`, `terminal/*`) are not adopted:
the agent owns a jail the operator configured, so an editor cannot be talked into
doing the model's filesystem work. `session/load` is not implemented, and
`initialize` reports it absent.
"""

from __future__ import annotations

from agent6.ui.acp.runner import serve_acp  # noqa: ICN003  # re-export
from agent6.ui.acp.server import ACPServer  # noqa: ICN003  # re-export

__all__ = ["ACPServer", "serve_acp"]
