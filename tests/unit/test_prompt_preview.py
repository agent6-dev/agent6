# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The harness package preserves its prompt preview exports."""

from __future__ import annotations

from agent6 import harness
from agent6.harness import prompt_preview


def test_package_exports_the_preview_module_definitions() -> None:
    """Package callers and module callers share the same preview definitions."""
    assert harness.ModelExchange is prompt_preview.ModelExchange
    assert harness.model_exchange_for is prompt_preview.model_exchange_for
    assert harness.system_prompt_for is prompt_preview.system_prompt_for
    assert harness.ModelExchange.__module__ == "agent6.harness.prompt_preview"
