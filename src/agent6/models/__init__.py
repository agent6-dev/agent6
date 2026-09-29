# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The provider model catalog.

Cached model listings (`cache`), cache-only price lookups (`pricing`), the curated
capability registry (`registry`) and pre-spawn model validation (`validate`). The package
re-exports nothing, so `budget -> models.pricing` and `models.cache -> providers -> budget`
stay acyclic.
"""

from __future__ import annotations
