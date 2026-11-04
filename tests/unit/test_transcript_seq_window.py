# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`sessions transcript --seq` windows N and N-M; a reversed M-N refuses instead of printing."""

from __future__ import annotations

import pytest

from agent6.ui.cli import sessions_transcript  # pyright: ignore[reportPrivateUsage]


def test_seq_window_shapes() -> None:
    assert sessions_transcript.parse_seq_window("") is None
    assert sessions_transcript.parse_seq_window("5") == (5, 5)
    assert sessions_transcript.parse_seq_window("3-7") == (3, 7)


def test_a_reversed_seq_window_is_an_error() -> None:
    with pytest.raises(ValueError, match="reversed"):
        sessions_transcript.parse_seq_window("9-3")
