# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""worker_models: the code-writing models a journal records, first worker first, deduplicated."""

from __future__ import annotations

from agent6.viewmodel import worker_models


def test_first_seen_unique_workers_only() -> None:
    events = [
        {"type": "model.call", "role": "worker", "model": "m1"},
        {"type": "model.call", "role": "reviewer", "model": "mc"},
        {"type": "model.call", "role": "worker", "model": "m2"},
        {"type": "model.call", "role": "worker", "model": "m1"},
        {"type": "model.call", "role": "worker"},  # no model field: skipped, never blank
    ]
    assert worker_models(events) == ("m1", "m2")
    assert worker_models([]) == ()
