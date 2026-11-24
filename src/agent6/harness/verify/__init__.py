# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The verify gate: when the harness runs it, its scoped retry and its verdict.

Part of `harness`: it imports `tools` and the layers below, never `app` or `ui`.
Main modules: `gate`, `scope`, `verdict`, `child`.
"""

from __future__ import annotations
