# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`parse_retry_after` header parsing (both RFC 7231 forms)."""

from __future__ import annotations

import datetime
from email import utils

from agent6.providers import types


def test_retry_after_seconds_form() -> None:
    assert types.parse_retry_after({"retry-after": "120"}) == 120.0
    assert types.parse_retry_after({"Retry-After": "0"}) == 0.0
    # whitespace tolerated
    assert types.parse_retry_after({"retry-after": "  45 "}) == 45.0


def test_retry_after_absent_or_garbage() -> None:
    assert types.parse_retry_after({}) is None
    assert types.parse_retry_after({"retry-after": ""}) is None
    assert types.parse_retry_after({"retry-after": "soon"}) is None
    assert types.parse_retry_after({"retry-after": "120, 60"}) is None  # not a single value


def test_retry_after_rejects_non_finite() -> None:
    # A malformed inf/nan must not propagate (it would dodge the loop's clamp).
    assert types.parse_retry_after({"retry-after": "inf"}) is None
    assert types.parse_retry_after({"retry-after": "nan"}) is None
    assert types.parse_retry_after({"retry-after": "-inf"}) is None


def test_retry_after_negative_clamped_to_zero() -> None:
    # A past delta must not produce a negative sleep.
    assert types.parse_retry_after({"retry-after": "-5"}) == 0.0


def test_retry_after_http_date_form() -> None:
    when = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(seconds=40)
    got = types.parse_retry_after({"Retry-After": utils.format_datetime(when)})
    assert got is not None
    assert 30.0 <= got <= 45.0  # ~40s, allowing for test execution slack


def test_retry_after_past_http_date_clamped() -> None:
    past = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(hours=1)
    assert types.parse_retry_after({"retry-after": utils.format_datetime(past)}) == 0.0
