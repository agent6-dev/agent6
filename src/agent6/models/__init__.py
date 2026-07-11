# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Provider model catalog: live cached model listings (`agent6.models.cache`),
cache-only price lookups over the same on-disk cache
(`agent6.models.pricing`), and the curated capability registry
(`agent6.models.registry`: context windows, compaction sizing, per-model
prompting defaults). Import the submodules directly; the package re-exports
nothing so `budget -> models.pricing` and
`models.cache -> providers -> budget` stay acyclic."""

from __future__ import annotations
