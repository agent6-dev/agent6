# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""ULID generation: shape and same-millisecond monotonicity."""

from __future__ import annotations

from agent6.graph.ulid import new_ulid

CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def test_ulid_shape() -> None:
    u = new_ulid()
    assert len(u) == 26
    assert all(c in CROCKFORD for c in u)


def test_ulids_strictly_increase_within_a_millisecond() -> None:
    # _first_ready_subtask sorts node ids as creation order; several ids per
    # millisecond is routine (an add_task is a handful of fsyncs), so same-ms
    # ids must still sort in creation order.
    ids = [new_ulid() for _ in range(5000)]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
