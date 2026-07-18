# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool-layer exceptions, homed here so callers (mcp_server, the loop,
tools/__init__) import them without importing the whole dispatch module."""

from __future__ import annotations


class ToolError(Exception):
    """The LLM tried something the tool layer refused."""


class ToolDenied(ToolError):
    """A run_command refused by POLICY before execution: the approval gate did
    not approve (a human said no, or the ask-policy auto-denied an unattended
    run), or the git guard refused a mutating command. The command never
    executed, so the loop's sandbox-reachability heuristic must not count it as
    a tool that "fails in the jail", and the repeat-error nudge says "refused,
    stop retrying" instead of "your call is malformed"."""


class OperatorCommandUnexecutable(Exception):
    """An operator-configured verify/metric command could not be executed in the
    jail (not found on PATH /usr/bin:/bin, or a path that escapes the sandbox).

    Distinct from ToolError (which the loop surfaces to the model and continues):
    the model cannot fix the operator's config, so the loop must abort loudly
    rather than let the worker flail against a verify that never actually runs.
    """
