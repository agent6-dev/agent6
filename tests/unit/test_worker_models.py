# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""worker_models: the code-writing models a journal records, first worker
first, deduplicated; message-writing roles never join."""

from __future__ import annotations

from agent6.viewmodel import worker_models


def test_first_seen_unique_workers_only() -> None:
    events = [
        {"type": "role.call", "role": "worker", "model": "m1"},
        {"type": "role.call", "role": "reviewer", "model": "mc"},
        {"type": "role.call", "role": "worker", "model": "m2"},
        {"type": "role.call", "role": "worker", "model": "m1"},
        {"type": "role.call", "role": "worker"},  # an older log: no model field
    ]
    assert worker_models(events) == ("m1", "m2")
    assert worker_models([]) == ()
