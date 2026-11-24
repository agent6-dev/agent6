# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The provider model catalog: cached listings, prices, capabilities and validation.

A shared package: any layer may import it, and it imports no layer.
The package re-exports nothing.
That keeps `budget -> models.pricing` and `models.cache -> providers -> budget` acyclic.
Main modules: `cache`, `pricing`, `registry`, `validate`, `choices`.
"""

from __future__ import annotations
