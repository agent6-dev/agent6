# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What the run and resume lifecycles share: a session's preparation, runtime, config and pruning.

Part of `app`: it imports `harness` and the layers below, never `ui`.
Main modules: `prepare`, `runtime`, `setup`, `prune`.
"""

from __future__ import annotations
