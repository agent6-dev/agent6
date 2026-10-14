# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A plan without its page is followed while it runs and resumed once it stopped."""

from __future__ import annotations

import pathlib

import pytest

from agent6.ui.cli import plan_watch  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("live", "expected"),
    [
        (True, "it is still running (`agent6 attach plan-AAAA11` follows it)"),
        (False, "`agent6 resume plan-AAAA11` continues it"),
    ],
)
def test_the_next_step_follows_a_live_plan_and_resumes_a_stopped_one(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, live: bool, expected: str
) -> None:
    """`plan show` told the operator to resume a plan that was still writing its page."""
    monkeypatch.chdir(tmp_path)

    def _is_live(_d: pathlib.Path) -> bool:
        return live

    monkeypatch.setattr(plan_watch, "session_is_live", _is_live)
    assert plan_watch._plan_next("plan-AAAA11") == expected  # pyright: ignore[reportPrivateUsage]
