# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The id a front-end gives an operator's message travels with it into the journal."""

from __future__ import annotations

import pathlib
from unittest import mock

from agent6 import operator_input
from agent6.sessions import ipc, journal
from tests.unit import test_harness, test_operator_image_delivery


def test_a_steer_with_an_id_waits_in_the_inbox_and_keeps_it_into_the_slot(
    tmp_path: pathlib.Path,
) -> None:
    """Written straight to the slot, the words would arrive with nothing to name them."""
    assert ipc.submit_steer(tmp_path, "tighten the test", message_id="m-1")
    assert not ipc.steer_request_pending(tmp_path), "only the worker pairs the slot with an id"
    (queued,) = ipc.queued_steers(tmp_path)
    assert (queued.text, queued.message_id) == ("tighten the test", "m-1")

    assert ipc.promote_queued_steer(tmp_path)

    assert ipc.take_steer_answer(tmp_path) == "tighten the test"
    assert ipc.take_steer_message_id(tmp_path) == "m-1"
    assert ipc.take_steer_message_id(tmp_path) == "", "consumed with its steer"


def test_a_steer_with_an_id_stays_behind_the_steers_left_waiting(
    tmp_path: pathlib.Path,
) -> None:
    ipc.submit_steer(tmp_path, "left from the last run", message_id="m-old")
    assert ipc.submit_steer(tmp_path, "this execution's prompt", message_id="m-new")

    assert [entry.message_id for entry in ipc.queued_steers(tmp_path)] == ["m-old", "m-new"]
    assert ipc.promote_queued_steer(tmp_path)
    assert ipc.take_steer_answer(tmp_path) == "left from the last run"


def test_a_steer_displaced_from_the_slot_keeps_its_id(tmp_path: pathlib.Path) -> None:
    ipc.submit_steer(tmp_path, "first", message_id="m-1")
    ipc.promote_queued_steer(tmp_path)
    assert ipc.submit_steer(tmp_path, "do this now", now=True)

    (displaced,) = ipc.queued_steers(tmp_path)
    assert (displaced.text, displaced.message_id) == ("first", "m-1")
    assert ipc.take_steer_message_id(tmp_path) == "", "the urgent steer has no id of its own"


def test_dropping_a_steer_drops_its_id(tmp_path: pathlib.Path) -> None:
    ipc.submit_steer(tmp_path, "first", message_id="m-1")
    ipc.promote_queued_steer(tmp_path)
    ipc.clear_steer_answer(tmp_path)
    assert not (tmp_path / ipc.STEER_MESSAGE_ID_FILE).exists()


def test_an_injected_steer_is_journaled_with_its_id(tmp_path: pathlib.Path) -> None:
    """The row is how a front-end finds the message again: on a replay, on a retried prompt."""
    events: list[journal.Event] = []
    ipc.submit_steer(tmp_path, "tighten the test", message_id="m-1")
    ipc.promote_queued_steer(tmp_path)
    test_operator_image_delivery._handle(test_operator_image_delivery._steering(tmp_path, events))
    (injected,) = [e for e in events if isinstance(e, journal.SteerInjected)]
    assert (injected.text, injected.message_id) == ("tighten the test", "m-1")

    events.clear()
    ipc.submit_steer(tmp_path, "no id here")
    test_operator_image_delivery._handle(test_operator_image_delivery._steering(tmp_path, events))
    (plain,) = [e for e in events if isinstance(e, journal.SteerInjected)]
    assert "message_id" not in plain.model_dump(exclude_unset=True), "a row names an id it has"


def test_an_opening_is_journaled_with_the_id_its_front_end_recorded(tmp_path: pathlib.Path) -> None:
    message = operator_input.OperatorInput(parts=("fix the parser",), message_id="m-1")
    ipc.write_launch_input(tmp_path, message)
    events: list[journal.Event] = []

    class _Events:
        path = tmp_path / "logs.jsonl"

        def emit(self, event: journal.Event) -> None:
            events.append(event)

    provider = mock.MagicMock()
    provider.call.return_value = test_harness._resp("done")
    wf = test_harness._wf(mode="run", budget=None, provider=provider, events=_Events())

    wf.run(message.text)

    (start,) = [e for e in events if isinstance(e, journal.SessionStart)]
    assert start.message_id == "m-1"
    assert not (tmp_path / ipc.LAUNCH_INPUT_FILE).exists()
    events.clear()
    wf.run("a task no front-end recorded")
    (bare,) = [e for e in events if isinstance(e, journal.SessionStart)]
    assert "message_id" not in bare.model_dump(exclude_unset=True)


def test_a_launch_record_names_its_message_only_to_the_execution_it_was_written_for(
    tmp_path: pathlib.Path,
) -> None:
    message = operator_input.OperatorInput(parts=("first task",), message_id="m-1")
    ipc.write_launch_input(tmp_path, message)
    assert ipc.take_launch_input(tmp_path, "another task").message_id == ""
    ipc.write_launch_input(tmp_path, message)
    assert ipc.take_launch_input(tmp_path, "first task").message_id == "m-1"
