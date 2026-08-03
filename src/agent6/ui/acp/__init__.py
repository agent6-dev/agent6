# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Agent Client Protocol front-end: agent6 driven by an editor.

A FOURTH front-end beside `ui/cli`, `ui/tui` and `ui/web`, over the same two
seams they use -- the viewmodel fold for everything it reports, and
`RunFrontend` + `FrontendCapabilities` for everything it is asked to do. ACP's
`initialize` IS a capability exchange, so it maps onto the second rather than
needing plumbing of its own.

Two constraints the spec states once and this module obeys everywhere:
every path is ABSOLUTE, and line numbers are 1-BASED.

What agent6 deliberately does NOT adopt: ACP has the CLIENT own the filesystem
and the terminal (`fs/*`, `terminal/*`). That is the inverse of agent6's model,
on purpose -- the AGENT owns a jail the operator configured, precisely so an
editor cannot be talked into doing the model's filesystem work. An adapter maps
those onto the jailed tools; it never lets the client do the work. A future
contributor's instinct will be to "fix" this by wiring them straight through.
Do not.

`session/load` is not implemented. It is exactly what ACP v2 reorganises, and
resume is where agent6 has the most of its own semantics; `initialize` reports
the capability as absent rather than half-answering it.
"""

from __future__ import annotations

from agent6.ui.acp.runner import serve_acp
from agent6.ui.acp.server import ACPServer

__all__ = ["ACPServer", "serve_acp"]
