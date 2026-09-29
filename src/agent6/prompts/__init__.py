# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Static prompt text: the strings agent6 sends to models.

The system-prompt bases and context blocks (`loop`), the review seats (`review`), the
compare judge (`judge`), the revision, summariser and restart prompts (`revision`) and the
machine-authoring grammar (`machine`). Pure text with `{...}` placeholders; the harness and
machine layers assemble them. The package imports nothing from agent6, so it stays a leaf.
"""

from __future__ import annotations
