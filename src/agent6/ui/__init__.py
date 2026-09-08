# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent6 UI layer: the four front-ends (cli, tui, web, acp), the shared
read-model fold (viewmodel), and the helpers beside them (spawn: launch the CLI
detached; notify: desktop notification; steer: the file-bridge steer seam; btw:
the side-question runner; mcp_server: agent6 as an MCP server). Everything here
is the top of the dependency graph: it may depend on the engine (workflows,
tools, sandbox), and the engine never depends on it."""

from __future__ import annotations
