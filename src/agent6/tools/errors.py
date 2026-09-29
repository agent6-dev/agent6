# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool-layer exceptions.

Homed here so handler modules and the MCP server import them without the dispatcher.
"""

from __future__ import annotations


class ToolError(Exception):
    """The LLM tried something the tool layer refused."""


class ToolDeniedError(ToolError):
    """A tool call refused by policy before it ran.

    The approval gate did not approve: a human said no, or the ask policy auto-denied an
    unattended run. A command, an MCP server's tool and a `fetch` whose host is outside
    `sandbox.fetch_hosts` all raise it. Nothing executed, so the loop's sandbox-reachability
    heuristic must not count it as a tool that fails in the jail, and the repeat-error nudge
    says "refused, stop retrying" instead of "your call is malformed".
    """


class OperatorCommandUnexecutableError(Exception):
    """An operator-configured verify or metric command cannot run in the jail.

    Raised for a command not found on PATH (/usr/bin:/bin) or a path that escapes the sandbox.
    Distinct from ToolError, which the loop surfaces to the model and continues: the model
    cannot fix the operator's config, so the loop aborts loudly.
    """
