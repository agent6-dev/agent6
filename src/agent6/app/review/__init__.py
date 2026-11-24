# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The reviews an operator starts or acts on: finding actions, `/review` and `sessions review`.

Part of `app`: it imports `harness` and the layers below, never `ui`.
The panel the loop runs itself is `harness.review`.
Main modules: `actions`, `directive`, `recap`.
"""

from __future__ import annotations
