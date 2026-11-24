# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The config models, file IO and layered resolve.

A shared package: any layer may import it, and it imports no layer.
The IO and layering are imported from `agent6.config.io` and `agent6.config.layer`.
Main modules: `model`, `layer`, `io`, `write`, `secret_store`.
"""

from __future__ import annotations
