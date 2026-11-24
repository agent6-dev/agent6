# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The review panel the loop runs: its gate, reviewers and history.

Part of `harness`: it imports `tools` and the layers below, never `app` or `ui`.
The reviews an operator starts (`/review`, `sessions review`) are `app.review`.
Main modules: `gate`, `panel`, `verdict`, `history`.
"""

from __future__ import annotations
