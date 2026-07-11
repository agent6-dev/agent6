# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the friendly run-id module."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent6.runs.id import (
    RunIdError,
    new_friendly_id,
    resolve_run_id,
    validate_explicit_run_id,
)

_PATTERN = re.compile(r"^[a-z]+-[a-z]+-[0-9A-Z]{6}$")


def test_validate_explicit_run_id_rejects_traversal() -> None:
    for bad in ("../escape", "..", ".", "a/b", "/abs/path", "x\\y", ""):
        with pytest.raises(RunIdError):
            validate_explicit_run_id(bad)
    # A normal slug (and the generated shape) passes through unchanged.
    assert validate_explicit_run_id("my-run-1") == "my-run-1"
    assert validate_explicit_run_id(new_friendly_id())


def test_new_friendly_id_shape() -> None:
    for _ in range(50):
        rid = new_friendly_id()
        assert _PATTERN.match(rid), rid


def test_new_friendly_id_unique() -> None:
    seen = {new_friendly_id() for _ in range(500)}
    assert len(seen) == 500


def test_new_friendly_id_suffix_time_sortable() -> None:
    """Suffixes from ids minted in order should sort in time order."""
    import time

    suffixes: list[str] = []
    for _ in range(10):
        suffixes.append(new_friendly_id().rsplit("-", 1)[1])
        time.sleep(0.002)
    assert suffixes == sorted(suffixes)


def test_resolve_exact_match(tmp_path: Path) -> None:
    (tmp_path / "sunny-otter-K4Q7B2").mkdir()
    assert resolve_run_id(tmp_path, "sunny-otter-K4Q7B2") == "sunny-otter-K4Q7B2"


def test_resolve_unambiguous_prefix(tmp_path: Path) -> None:
    (tmp_path / "sunny-otter-K4Q7B2").mkdir()
    (tmp_path / "calm-river-AAAA11").mkdir()
    assert resolve_run_id(tmp_path, "sunny") == "sunny-otter-K4Q7B2"
    assert resolve_run_id(tmp_path, "calm-riv") == "calm-river-AAAA11"


def test_resolve_ambiguous_prefix(tmp_path: Path) -> None:
    (tmp_path / "sunny-otter-K4Q7B2").mkdir()
    (tmp_path / "sunny-otter-AAAA11").mkdir()
    with pytest.raises(RunIdError, match="ambiguous"):
        resolve_run_id(tmp_path, "sunny")


def test_resolve_no_match(tmp_path: Path) -> None:
    (tmp_path / "sunny-otter-K4Q7B2").mkdir()
    with pytest.raises(RunIdError, match="no run matches"):
        resolve_run_id(tmp_path, "zzz")


def test_resolve_empty_query(tmp_path: Path) -> None:
    with pytest.raises(RunIdError, match="empty"):
        resolve_run_id(tmp_path, "")
