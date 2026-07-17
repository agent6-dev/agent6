# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Static prompt and template text: the pure, dependency-free strings agent6
sends to models.

The agent-loop system-prompt bases and context blocks (`loop`), the
review-panel seat prompts (`review`), the compare-judge prompt (`judge`), the
auxiliary critic / prompt-revision / summariser / restart prompts (`revision`),
and the machine-authoring grammar reference (`machine`). Pure text with `{...}`
format placeholders; the workflow and machine layers own assembly. This
package imports nothing from agent6 so it stays a leaf.
"""

from __future__ import annotations
