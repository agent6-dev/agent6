# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Present agent6: the four front-ends and the helpers beside them.

The top layer and the composition root: it imports every layer, and no layer imports it.
The front-ends are `cli`, `tui`, `web` and `acp`, and `telemetry` exports session journals.
Main modules: `spawn`, `steer`, `directive_actions`, `keymap`, `notify`, `btw`, `mcp_server`.
"""

from __future__ import annotations
