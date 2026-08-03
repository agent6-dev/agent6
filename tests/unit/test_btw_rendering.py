# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A btw answer lands whole, at a clean break -- never through the transcript."""

from __future__ import annotations

import io

from agent6.ui.cli._console_view import ConsoleView


def _view() -> tuple[ConsoleView, io.StringIO]:
    out = io.StringIO()
    return ConsoleView(out, color=False), out


def test_an_answer_queued_mid_stream_waits_for_the_turn_boundary() -> None:
    """A btw finishes while the run is streaming. Printing it then would cut a
    sentence in half; it waits and lands whole."""
    view, out = _view()
    view.feed({"type": "role.text_delta", "text": "the first half "})
    view.queue_btw("\n--- btw: why\nbecause\n--- end btw\n")
    view.feed({"type": "role.text_delta", "text": "and the second half"})
    mid = out.getvalue()
    assert "btw" not in mid, "a btw must never interrupt streaming prose"
    assert "the first half and the second half" in mid.replace("\n", " ").replace("  ", " ")

    view.feed({"type": "role.result"})  # the turn boundary
    after = out.getvalue()
    assert "--- btw: why" in after
    assert after.index("and the second half") < after.index("--- btw: why")


def test_the_block_lands_in_one_piece() -> None:
    view, out = _view()
    view.queue_btw("\n--- btw: q\nline one\nline two\n--- end btw\n")
    view.feed({"type": "role.result"})
    text = out.getvalue()
    start, end = text.index("--- btw: q"), text.index("--- end btw")
    assert "line one\nline two" in text[start:end]


def test_two_answers_both_land_and_only_once() -> None:
    view, out = _view()
    view.queue_btw("\n--- btw: a\nfirst\n--- end btw\n")
    view.queue_btw("\n--- btw: b\nsecond\n--- end btw\n")
    view.feed({"type": "role.result"})
    view.feed({"type": "role.result"})  # nothing left to drain
    text = out.getvalue()
    assert text.count("--- btw: a") == 1
    assert text.count("--- btw: b") == 1


def test_nothing_queued_prints_nothing() -> None:
    view, out = _view()
    view.feed({"type": "role.result"})
    assert out.getvalue() == ""
