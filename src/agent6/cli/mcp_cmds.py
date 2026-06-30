# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 mcp-serve` command (spawn an MCP stdio server)."""

from __future__ import annotations

from pathlib import Path

from agent6.mcp_server import run_server as _mcp_run_server


def _cmd_mcp_serve(config_path: Path | None) -> int:
    """Spawn an MCP stdio server against ``config_path``'s
    workspace. Thin wrapper so dispatch stays uniform with the other
    ``_cmd_*`` helpers."""
    return _mcp_run_server(config_path)
