# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the friendly run-id module."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent6.sessions.id import (
    SessionIdError,
    friendly_token,
    resolve_session_id,
    validate_explicit_session_id,
)

_PATTERN = re.compile(r"^[a-z]+-[a-z]+-[0-9A-Z]{6}$")


def test_validate_explicit_run_id_rejects_traversal() -> None:
    for bad in ("../escape", "..", ".", "a/b", "/abs/path", "x\\y", ""):
        with pytest.raises(SessionIdError):
            validate_explicit_session_id(bad)
    # A normal slug (and the generated shape) passes through unchanged.
    assert validate_explicit_session_id("my-run-1") == "my-run-1"
    assert validate_explicit_session_id(friendly_token())


def test_friendly_token_shape() -> None:
    for _ in range(50):
        rid = friendly_token()
        assert _PATTERN.match(rid), rid


def test_friendly_token_varies() -> None:
    """Catches a constant or an unseeded generator. NOT a uniqueness guarantee:
    within one millisecond the space is ~30M, so 500 draws collide about once
    in 200 -- which is what made the old 500-draw assertion flaky. What must
    never collide is the DIRECTORY, and `_unused_session_id` owns that
    (tests/unit/test_generated_id_collision.py)."""
    seen = {friendly_token() for _ in range(20)}
    assert len(seen) == 20


def test_friendly_token_suffix_time_sortable() -> None:
    """Suffixes from ids minted in order should sort in time order."""
    import time

    suffixes: list[str] = []
    for _ in range(10):
        suffixes.append(friendly_token().rsplit("-", 1)[1])
        time.sleep(0.002)
    assert suffixes == sorted(suffixes)


def test_resolve_exact_match(tmp_path: Path) -> None:
    (tmp_path / "sunny-otter-K4Q7B2").mkdir()
    assert resolve_session_id(tmp_path, "sunny-otter-K4Q7B2") == "sunny-otter-K4Q7B2"


def test_resolve_unambiguous_prefix(tmp_path: Path) -> None:
    (tmp_path / "sunny-otter-K4Q7B2").mkdir()
    (tmp_path / "calm-river-AAAA11").mkdir()
    assert resolve_session_id(tmp_path, "sunny") == "sunny-otter-K4Q7B2"
    assert resolve_session_id(tmp_path, "calm-riv") == "calm-river-AAAA11"


def test_resolve_ambiguous_prefix(tmp_path: Path) -> None:
    (tmp_path / "sunny-otter-K4Q7B2").mkdir()
    (tmp_path / "sunny-otter-AAAA11").mkdir()
    with pytest.raises(SessionIdError, match="ambiguous"):
        resolve_session_id(tmp_path, "sunny")


def test_resolve_no_match(tmp_path: Path) -> None:
    (tmp_path / "sunny-otter-K4Q7B2").mkdir()
    with pytest.raises(SessionIdError, match="no session matches"):
        resolve_session_id(tmp_path, "zzz")


def test_resolve_empty_query(tmp_path: Path) -> None:
    with pytest.raises(SessionIdError, match="empty"):
        resolve_session_id(tmp_path, "")
