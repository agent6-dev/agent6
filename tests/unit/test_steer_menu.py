# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The mid-run Ctrl-C menu maps operator input to a canonical steer action."""

from __future__ import annotations

from agent6.cli._steer import (
    _normalize_steer_choice,  # pyright: ignore[reportPrivateUsage]
)


def test_stop_keys_map_to_abort() -> None:
    for key in ("q", "Q", "quit", "stop", "abort", "  ABORT  "):
        assert _normalize_steer_choice(key) == "abort"


def test_detach_keys_map_to_detach() -> None:
    for key in ("d", "D", "detach", " Detach "):
        assert _normalize_steer_choice(key) == "detach"


def test_blank_continues() -> None:
    assert _normalize_steer_choice("") == ""
    assert _normalize_steer_choice("   ") == ""


def test_none_stays_none() -> None:
    assert _normalize_steer_choice(None) is None


def test_instruction_passes_through() -> None:
    assert _normalize_steer_choice("focus on the parser") == "focus on the parser"
    # a sentence that merely starts with a keyword is an instruction, not a command
    assert _normalize_steer_choice("abort the current plan") == "abort the current plan"
