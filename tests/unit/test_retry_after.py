# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`parse_retry_after` header parsing (both RFC 7231 forms)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

from agent6.providers.anthropic import parse_retry_after


def test_retry_after_seconds_form() -> None:
    assert parse_retry_after({"retry-after": "120"}) == 120.0
    assert parse_retry_after({"Retry-After": "0"}) == 0.0
    # whitespace tolerated
    assert parse_retry_after({"retry-after": "  45 "}) == 45.0


def test_retry_after_absent_or_garbage() -> None:
    assert parse_retry_after({}) is None
    assert parse_retry_after({"retry-after": ""}) is None
    assert parse_retry_after({"retry-after": "soon"}) is None
    assert parse_retry_after({"retry-after": "120, 60"}) is None  # not a single value


def test_retry_after_rejects_non_finite() -> None:
    # A malformed inf/nan must not propagate (it would dodge the loop's clamp).
    assert parse_retry_after({"retry-after": "inf"}) is None
    assert parse_retry_after({"retry-after": "nan"}) is None
    assert parse_retry_after({"retry-after": "-inf"}) is None


def test_retry_after_negative_clamped_to_zero() -> None:
    # A past delta must not produce a negative sleep.
    assert parse_retry_after({"retry-after": "-5"}) == 0.0


def test_retry_after_http_date_form() -> None:
    when = datetime.now(tz=UTC) + timedelta(seconds=40)
    got = parse_retry_after({"Retry-After": format_datetime(when)})
    assert got is not None
    assert 30.0 <= got <= 45.0  # ~40s, allowing for test execution slack


def test_retry_after_past_http_date_clamped() -> None:
    past = datetime.now(tz=UTC) - timedelta(hours=1)
    assert parse_retry_after({"retry-after": format_datetime(past)}) == 0.0
