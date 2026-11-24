# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The steer inbox: steers sent while one waits queue in order and reach the loop one per step.

One slot held every steer; a second one was refused or overwrote the first.
"""

from __future__ import annotations

import pathlib

from agent6.harness import operator_bridge
from agent6.sessions import ipc
from agent6.ui import directive_actions


def _slot(session_dir: pathlib.Path) -> str | None:
    try:
        return (session_dir / ipc.STEER_ANSWER_FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def test_steers_behind_the_one_in_flight_queue_in_order_and_promote_one_at_a_time(
    tmp_path: pathlib.Path,
) -> None:
    assert ipc.submit_steer(tmp_path, "one")
    assert ipc.submit_steer(tmp_path, "two")
    assert ipc.submit_steer(tmp_path, "three")
    assert _slot(tmp_path) == "one" and not ipc.steer_interrupt_pending(tmp_path)
    assert [entry.text for entry in ipc.queued_steers(tmp_path)] == ["two", "three"]
    # The slot is occupied: nothing moves until the loop takes the steer.
    assert ipc.promote_queued_steer(tmp_path) is False
    assert ipc.take_steer_answer(tmp_path) == "one"
    ipc.clear_steer_request(tmp_path)
    assert ipc.promote_queued_steer(tmp_path) is True
    assert _slot(tmp_path) == "two" and ipc.steer_request_pending(tmp_path)
    assert [entry.text for entry in ipc.queued_steers(tmp_path)] == ["three"]
    ipc.clear_steer_answer(tmp_path)
    ipc.clear_steer_request(tmp_path)
    assert ipc.promote_queued_steer(tmp_path) is True
    assert _slot(tmp_path) == "three" and ipc.steer_request_pending(tmp_path)
    ipc.clear_steer_answer(tmp_path)
    ipc.clear_steer_request(tmp_path)
    assert ipc.promote_queued_steer(tmp_path) is False
    assert ipc.queued_steers(tmp_path) == ()


def test_a_verb_takes_the_slot_whatever_it_holds(tmp_path: pathlib.Path) -> None:
    """A Stop must not wait behind a queued instruction."""
    assert ipc.submit_steer(tmp_path, "keep going")
    assert ipc.submit_steer(tmp_path, "abort", now=True)
    assert _slot(tmp_path) == "abort" and ipc.steer_answer_is_abort(tmp_path)
    assert ipc.queued_steers(tmp_path) == ()


def test_an_urgent_steer_takes_the_slot_and_the_displaced_one_goes_first(
    tmp_path: pathlib.Path,
) -> None:
    assert ipc.submit_steer(tmp_path, "first")
    assert ipc.submit_steer(tmp_path, "second")
    assert ipc.submit_steer(tmp_path, "urgent", now=True)
    assert _slot(tmp_path) == "urgent" and ipc.steer_interrupt_pending(tmp_path)
    assert [entry.text for entry in ipc.queued_steers(tmp_path)] == ["first", "second"]


def test_a_demoted_steer_goes_back_to_the_head_of_the_inbox(tmp_path: pathlib.Path) -> None:
    """A parked start left its steer in the slot, which the next start clears as stale."""
    assert ipc.submit_steer(tmp_path, "first", now=True)
    assert ipc.submit_steer(tmp_path, "second")
    ipc.demote_steer(tmp_path)
    assert _slot(tmp_path) is None and not ipc.steer_request_pending(tmp_path)
    assert [(entry.text, entry.now) for entry in ipc.queued_steers(tmp_path)] == [
        ("first", True),
        ("second", False),
    ]
    # Nothing in the slot: nothing moves.
    ipc.demote_steer(tmp_path)
    assert [entry.text for entry in ipc.queued_steers(tmp_path)] == ["first", "second"]


def test_a_demotion_leaves_a_verb_in_the_slot(tmp_path: pathlib.Path) -> None:
    """A Stop that ended a start is for that worker: the next start must not read it."""
    assert ipc.submit_steer(tmp_path, "abort", now=True)
    ipc.demote_steer(tmp_path)
    assert _slot(tmp_path) == "abort" and ipc.steer_request_pending(tmp_path)
    assert ipc.queued_steers(tmp_path) == ()


def test_the_bridge_acts_on_the_words_the_inbox_lets_through(tmp_path: pathlib.Path) -> None:
    """The verbs the loop acts on and the verbs that preempt the inbox are one set."""
    assert set(operator_bridge.STEER_VERBS) == set(ipc.STEER_VERBS)


def test_an_unreadable_inbox_entry_is_dropped(tmp_path: pathlib.Path) -> None:
    assert ipc.submit_steer(tmp_path, "one")
    assert ipc.submit_steer(tmp_path, "two")
    (path,) = (tmp_path / ipc.STEERS_DIR_NAME).glob("*.steer")
    path.write_text("not json", encoding="utf-8")
    assert ipc.queued_steers(tmp_path) == ()
    assert not path.exists()


def test_the_composer_says_how_many_steers_are_ahead(tmp_path: pathlib.Path) -> None:
    did, said = directive_actions.submit_composer_line(tmp_path, "first")
    assert did and said == "steer queued, not yet delivered"
    did, said = directive_actions.submit_composer_line(tmp_path, "second")
    assert did and said == "steer queued behind 1 other; not yet delivered"
    did, said = directive_actions.submit_composer_line(tmp_path, "third")
    assert did and said == "steer queued behind 2 others; not yet delivered"
    assert [entry.text for entry in ipc.queued_steers(tmp_path)] == ["second", "third"]
