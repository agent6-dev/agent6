# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Integration tests for triage + profile dispatch in ImplementWorkflow.

Drives ``_select_profile`` directly with stubbed triage providers and a
mock workflow, without spinning up git / sandbox / curator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent6.agents.triage import TaskClass
from agent6.providers import ProviderResponse
from agent6.types import RepoSummary
from agent6.workflows.implement import ImplementWorkflow
from agent6.workflows.profiles import DEFAULT_PROFILE, PROFILES, Profile


def _silent(_msg: str) -> None:
    return None


def _repo(tmp_path: Path) -> RepoSummary:
    return RepoSummary(
        root=tmp_path,
        branch="main",
        head_sha="0" * 40,
        file_count=1,
        top_level=(),
        agents_md="",
        recent_log="",
    )


class _Stub:
    """Provider stub returning a fixed JSON payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        return ProviderResponse(
            text=json.dumps(self._payload),
            input_tokens=1,
            output_tokens=1,
            stop_reason="end_turn",
            tool_uses=(),
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )


def _wf(tmp_path: Path, **kw: Any) -> ImplementWorkflow:
    return ImplementWorkflow(
        root=tmp_path,
        config=MagicMock(),
        planner=MagicMock(),
        worker=MagicMock(),
        reviewer=MagicMock(),
        critic=MagicMock(),
        dispatcher=MagicMock(),
        logger=_silent,
        **kw,
    )


def test_select_profile_uses_pin_when_set(tmp_path: Path) -> None:
    pinned = PROFILES[TaskClass.TRIVIAL]
    wf = _wf(
        tmp_path,
        profile=pinned,
        triage=_Stub({"task_class": "multi", "reasoning": "x", "confidence": 0.9}),
    )
    out = wf._select_profile(  # pyright: ignore[reportPrivateUsage]
        user_task="t", repo=_repo(tmp_path)
    )
    assert out is pinned, "operator pin must win over triage"


def test_select_profile_falls_back_to_default_without_triage(tmp_path: Path) -> None:
    wf = _wf(tmp_path)
    out = wf._select_profile(  # pyright: ignore[reportPrivateUsage]
        user_task="t", repo=_repo(tmp_path)
    )
    assert out is DEFAULT_PROFILE


def test_select_profile_routes_to_trivial_on_classifier_signal(tmp_path: Path) -> None:
    wf = _wf(
        tmp_path,
        triage=_Stub({"task_class": "trivial", "reasoning": "one-liner", "confidence": 0.9}),
    )
    out = wf._select_profile(  # pyright: ignore[reportPrivateUsage]
        user_task="fix typo", repo=_repo(tmp_path)
    )
    assert out is PROFILES[TaskClass.TRIVIAL]


def test_select_profile_routes_to_multi_step(tmp_path: Path) -> None:
    wf = _wf(
        tmp_path,
        triage=_Stub({"task_class": "multi", "reasoning": "cross-file", "confidence": 0.8}),
    )
    out = wf._select_profile(  # pyright: ignore[reportPrivateUsage]
        user_task="refactor", repo=_repo(tmp_path)
    )
    assert out is PROFILES[TaskClass.MULTI_STEP]


def test_select_profile_recovers_from_classifier_failure(tmp_path: Path) -> None:
    """A malformed classifier response must NOT abort the run — fall back."""
    bad = _Stub({"task_class": "garbage", "reasoning": "x", "confidence": 0.5})
    wf = _wf(tmp_path, triage=bad)
    out = wf._select_profile(  # pyright: ignore[reportPrivateUsage]
        user_task="t", repo=_repo(tmp_path)
    )
    assert out is DEFAULT_PROFILE


def test_default_profile_keeps_full_pipeline(tmp_path: Path) -> None:
    """Sanity: DEFAULT_PROFILE must not silently downgrade to skip-everything."""
    p: Profile = DEFAULT_PROFILE
    assert not p.skip_critic
    assert not p.skip_planner
