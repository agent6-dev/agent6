# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""worker_models: the code-writing models a journal records, first worker first, deduplicated."""

from __future__ import annotations

from agent6.viewmodel import transcript


def test_first_seen_unique_workers_only() -> None:
    events = [
        {"type": "model.call", "actor": "worker", "model": "m1"},
        {"type": "model.call", "actor": "reviewer", "model": "mc"},
        {"type": "model.call", "actor": "worker", "model": "m2"},
        {"type": "model.call", "actor": "worker", "model": "m1"},
        {"type": "model.call", "actor": "worker"},  # no model field: skipped, never blank
    ]
    assert transcript.worker_models(events) == ("m1", "m2")
    assert transcript.worker_models([]) == ()
