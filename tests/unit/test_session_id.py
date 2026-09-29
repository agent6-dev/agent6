# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the friendly run-id module."""

from __future__ import annotations

import pathlib
import re

import pytest

from agent6.sessions import id

_PATTERN = re.compile(r"^[a-z]+-[a-z]+-[0-9A-Z]{6}$")


def test_validate_explicit_run_id_rejects_traversal() -> None:
    for bad in ("../escape", "..", ".", "a/b", "/abs/path", "x\\y", ""):
        with pytest.raises(id.SessionIdError):
            id.validate_explicit_session_id(bad)
    # A normal slug (and the generated shape) passes through unchanged.
    assert id.validate_explicit_session_id("my-run-1") == "my-run-1"
    assert id.validate_explicit_session_id(id.friendly_token())


def test_validate_explicit_run_id_rejects_git_forbidden_names() -> None:
    """An id git's ref grammar rejects is refused up front.

    The id becomes a branch and a chain ref; accepted, every `update-ref` of the run would fail
    while it reported success. The traversal check alone misses all of these.
    """
    for bad in (
        "has space",
        "ti~lde",
        "ca^ret",
        "col:on",
        "quest?ion",
        "star*x",
        "brack[et",
        "end.lock",
        "trailing.",
        "dou..ble",
        "at@{brace",
        "-leading",
        ".leading",
    ):
        with pytest.raises(id.SessionIdError):
            id.validate_explicit_session_id(bad)


def test_validate_explicit_run_id_accepts_only_ids_git_can_ref(tmp_path: pathlib.Path) -> None:
    """Whatever the validator accepts works as both refs the run builds."""
    import subprocess

    def git_accepts(ref: str) -> bool:
        return (
            subprocess.run(
                ["git", "check-ref-format", ref], capture_output=True, cwd=tmp_path, check=False
            ).returncode
            == 0
        )

    for good in ("my-run-1", "sunny-otter-K4Q7B2", "machine-foo", "a.b.c", "UPPER_case-1"):
        assert id.validate_explicit_session_id(good) == good
        assert git_accepts(f"refs/heads/agent6/{good}"), good  # the run branch
        assert git_accepts(f"refs/agent6/{good}/head"), good  # the chain ref


def test_friendly_token_shape() -> None:
    for _ in range(50):
        rid = id.friendly_token()
        assert _PATTERN.match(rid), rid


def test_friendly_token_varies() -> None:
    """Catches a constant or an unseeded generator.

    NOT a uniqueness guarantee: within one millisecond the space is ~30M, so 500 draws collide about
    once in 200, which makes a 500-draw assertion flaky. What must never collide is the DIRECTORY,
    and `_unused_session_id` owns that (tests/unit/test_generated_id_collision.py).
    """
    seen = {id.friendly_token() for _ in range(20)}
    assert len(seen) == 20


def test_friendly_token_suffix_time_sortable() -> None:
    """Suffixes from ids minted in order should sort in time order."""
    import time

    suffixes: list[str] = []
    for _ in range(10):
        suffixes.append(id.friendly_token().rsplit("-", 1)[1])
        time.sleep(0.002)
    assert suffixes == sorted(suffixes)


def _bucket(state_dir: pathlib.Path, *ids: str) -> None:
    for sid in ids:
        (state_dir / "sessions" / "plans" / sid).mkdir(parents=True)


def test_resolve_exact_match(tmp_path: pathlib.Path) -> None:
    _bucket(tmp_path, "sunny-otter-K4Q7B2")
    layout = id.resolve_session(tmp_path, "sunny-otter-K4Q7B2", buckets=("plans",))
    assert (layout.session_id, layout.subdir) == ("sunny-otter-K4Q7B2", "plans")


def test_resolve_unambiguous_prefix(tmp_path: pathlib.Path) -> None:
    _bucket(tmp_path, "sunny-otter-K4Q7B2", "calm-river-AAAA11")
    assert (
        id.resolve_session(tmp_path, "sunny", buckets=("plans",)).session_id == "sunny-otter-K4Q7B2"
    )
    assert (
        id.resolve_session(tmp_path, "calm-riv", buckets=("plans",)).session_id
        == "calm-river-AAAA11"
    )


def test_resolve_ambiguous_prefix(tmp_path: pathlib.Path) -> None:
    _bucket(tmp_path, "sunny-otter-K4Q7B2", "sunny-otter-AAAA11")
    with pytest.raises(id.SessionIdError, match="ambiguous"):
        id.resolve_session(tmp_path, "sunny", buckets=("plans",))


def test_resolve_no_match(tmp_path: pathlib.Path) -> None:
    _bucket(tmp_path, "sunny-otter-K4Q7B2")
    with pytest.raises(id.SessionIdError, match="no session matches"):
        id.resolve_session(tmp_path, "nope", buckets=("plans",))


def test_a_bucket_scoped_query_ignores_the_other_buckets(tmp_path: pathlib.Path) -> None:
    """`plan show` resolves through the shared resolver with the buckets it takes.

    A plans-only twin had drifted in wording; a plans-only prefix is not ambiguous against a
    run of the same prefix.
    """
    _bucket(tmp_path, "sunny-otter-K4Q7B2")
    (tmp_path / "sessions" / "runs" / "sunny-otter-AAAA11").mkdir(parents=True)
    assert (
        id.resolve_session(tmp_path, "sunny", buckets=("plans",)).session_id == "sunny-otter-K4Q7B2"
    )
    with pytest.raises(id.SessionIdError, match="ambiguous"):
        id.resolve_session(tmp_path, "sunny")


def test_resolve_empty_query(tmp_path: pathlib.Path) -> None:
    with pytest.raises(id.SessionIdError, match="empty"):
        id.resolve_session(tmp_path, "")
