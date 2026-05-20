# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Summarizer sub-agent: compress long tool output."""

from __future__ import annotations

from agent6.agents._common import call_for_model
from agent6.models import Summary
from agent6.providers import Provider

_SYSTEM = """You compress long command output for a downstream coding agent.

Rules:
- Preserve concrete error messages, file:line references, and test failure names verbatim.
- Drop ANSI escape codes, progress noise, repeated lines.
- Output a single paragraph (or short bullets) describing what happened.
"""


def summarizer_compress(provider: Provider, *, tool_output: str, max_tokens: int = 512) -> Summary:
    user = f"OUTPUT (verbatim):\n{tool_output[:50_000]}\n"
    return call_for_model(
        provider, system=_SYSTEM, user=user, output_model=Summary, max_tokens=max_tokens
    )
