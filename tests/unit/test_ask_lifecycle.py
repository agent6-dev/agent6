# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An ask is a session like any other: findable, resumable, still an ask."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.runs.layout import session_layout
from agent6.runs.manifest import ManifestError, RunManifest, read_manifest


def _session(state: Path, bucket: str, sid: str, mode: str) -> Path:
    d = state / bucket / sid
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": mode, "run_id": sid}), encoding="utf-8"
    )
    return d


def test_an_ask_id_resolves_to_its_own_bucket(tmp_path: Path) -> None:
    """Resume looked only under runs/, so an ask could not be continued at all."""
    _session(tmp_path, "asks", "quiet-fox-AAAAAA", "ask")
    _session(tmp_path, "runs", "brave-elk-BBBBBB", "run")
    ask = session_layout(tmp_path, "quiet-fox-AAAAAA")
    run = session_layout(tmp_path, "brave-elk-BBBBBB")
    assert ask is not None and ask.subdir == "asks"
    assert run is not None and run.subdir == "runs"
    assert ask.run_dir.is_dir()


def test_a_unique_prefix_resolves_across_buckets(tmp_path: Path) -> None:
    _session(tmp_path, "asks", "quiet-fox-AAAAAA", "ask")
    found = session_layout(tmp_path, "quiet-fox")
    assert found is not None and found.run_id == "quiet-fox-AAAAAA"


def test_an_ambiguous_prefix_resolves_to_nothing_rather_than_guessing(tmp_path: Path) -> None:
    _session(tmp_path, "asks", "quiet-fox-AAAAAA", "ask")
    _session(tmp_path, "runs", "quiet-fox-BBBBBB", "run")
    assert session_layout(tmp_path, "quiet-fox") is None


def test_an_unknown_id_resolves_to_nothing(tmp_path: Path) -> None:
    assert session_layout(tmp_path, "nope") is None
    assert session_layout(tmp_path, "") is None


def test_ask_is_a_mode_resume_and_fork_may_act_on(tmp_path: Path) -> None:
    """The privilege gate refused "ask" outright, so an ask was a dead end: no
    resume, no fork. It is LESS privileged than plan, not unknown."""
    d = _session(tmp_path, "asks", "quiet-fox-AAAAAA", "ask")
    assert read_manifest(d).validated_mode() == "ask"


def test_an_unknown_mode_is_still_refused(tmp_path: Path) -> None:
    """The gate's whole point: a damaged manifest must not fall open to the
    privileged write mode."""
    d = _session(tmp_path, "asks", "odd-AAAAAA", "wat")
    with pytest.raises(ManifestError, match="unknown run mode"):
        read_manifest(d).validated_mode()


def test_a_resumed_ask_is_still_clamped() -> None:
    """The clamp lives with the mode, not with one lifecycle, so continuing an
    ask cannot hand it the auto-approval a fresh one never had."""
    from agent6.app._session import session_config

    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}})
    assert session_config(cfg, "ask").sandbox.run_commands == "ask"
    assert session_config(cfg, "run").sandbox.run_commands == "yes"


def test_an_ask_records_a_mode_that_survives_a_round_trip(tmp_path: Path) -> None:
    m = RunManifest(mode="ask", run_id="x")
    assert m.validated_mode() == "ask"
