# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""SpiralGuard transitions: the streak and reset rules the dispatch loop
relies on, pinned at the object so call sites cannot drift."""

from __future__ import annotations

from agent6.workflows._spiral_guards import SpiralGuard


def test_repeat_streak_extends_and_resets() -> None:
    g = SpiralGuard()
    g.note_call("read:a")
    g.note_call("read:a")
    assert g.call_streak == 2
    g.note_call("read:b")
    assert g.call_streak == 1 and g.last_call_sig == "read:b"


def test_stub_repeat_needs_streak_identity_and_size() -> None:
    g = SpiralGuard()
    g.note_call("read:a")
    g.note_success("x" * 500)
    g.note_call("read:a")  # back-to-back repeat
    assert g.stub_repeat("x" * 500, min_chars=100)  # identical, big enough
    assert not g.stub_repeat("y" * 500, min_chars=100)  # changed result serves whole
    assert not g.stub_repeat("x" * 500, min_chars=1000)  # too small to bother
    g.note_call("read:b")  # different call: no repeat
    assert not g.stub_repeat("x" * 500, min_chars=100)


def test_success_clears_the_whole_error_spiral() -> None:
    g = SpiralGuard()
    g.note_error("boom", denial=True, content="{}")
    g.note_error("boom", denial=False, content="{}")
    assert g.error_streak == 2
    g.error_nudges_used = 1
    g.note_success("ok")
    assert g.error_streak == 0 and g.error_sig is None
    assert g.error_nudges_used == 0 and g.last_error_was_denial is False
    assert g.last_served_content == "ok"


def test_a_new_error_signature_rearms_the_nudge_allowance() -> None:
    g = SpiralGuard()
    g.note_error("sig-a", denial=False, content="{}")
    g.error_nudges_used = 2
    g.note_error("sig-b", denial=False, content="{}")
    assert g.error_streak == 1 and g.error_nudges_used == 0
