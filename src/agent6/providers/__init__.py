# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Model providers behind one `provider_calls.Provider` protocol.

A shared package: it imports `sandbox` and no layer above it.
The providers are interchangeable, and `[models.<role>]` routes each role.
Main modules: `anthropic`, `openai`, `responses`, `chatgpt`, `claude_code`, `provider_types`.
"""

from __future__ import annotations
