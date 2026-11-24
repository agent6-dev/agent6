# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run's system prompt and first message: the block fillers, the repo context and the preview.

Part of `harness`: it imports `tools` and the layers below, never `app` or `ui`.
The text the fillers render is the leaf package `agent6.prompts`.
Main modules: `blocks`, `repo_context`, `preview`.
"""

from __future__ import annotations
