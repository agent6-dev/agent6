# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Session identity and on-disk state.

A shared package: any layer may import it, and it imports no layer.
The package re-exports nothing: import the submodules directly.
Main modules: `id`, `layout`, `manifest`, `journal`, `event_log`, `lock`, `ipc`, `process`.
"""

from __future__ import annotations
