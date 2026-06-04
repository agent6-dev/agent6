# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Sub-agents — each a typed function calling a Provider with a validated output.

: only ``code_review`` remains (used by ``agent6 review``). The
legacy sub-agent cascade (planner/critic/architect/editor/reviewer/triage/
alignment/explore/worker/summarizer) was removed when the single-loop agent
became the sole agent loop.
"""

from __future__ import annotations
