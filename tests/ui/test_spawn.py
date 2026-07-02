# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the shared spawn+locate helper behind the hub's "start a run" and the
machines page's "create" -- both spawn the CLI detached, then watch the new log dir."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.ui import _spawn


def test_spawn_and_locate_finds_new_log_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "dirs"
    base.mkdir()

    class _Proc:
        def __init__(self) -> None:
            # The "child" produces a new dir with a logs.jsonl the moment it starts.
            (base / "new").mkdir()
            (base / "new" / "logs.jsonl").write_text("", encoding="utf-8")

        def poll(self) -> int | None:
            return None  # still running

    def _popen(*_a: object, **_k: object) -> _Proc:
        return _Proc()

    monkeypatch.setattr(_spawn.subprocess, "Popen", _popen)
    found, err = _spawn.spawn_and_locate(
        ["agent6", "x"],
        tmp_path,
        before=set(),
        list_dirs=lambda: [p for p in base.iterdir() if p.is_dir()],
    )
    assert err == ""
    assert found == base / "new"


def test_spawn_and_locate_ignores_preexisting_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "dirs"
    base.mkdir()
    (base / "old").mkdir()
    (base / "old" / "logs.jsonl").write_text("", encoding="utf-8")
    before = {base / "old"}

    class _Proc:
        returncode = 0

        def poll(self) -> int:
            return 0  # exits immediately without producing a NEW dir

    def _popen(*_a: object, **_k: object) -> _Proc:
        return _Proc()

    monkeypatch.setattr(_spawn.subprocess, "Popen", _popen)
    found, err = _spawn.spawn_and_locate(
        ["agent6", "machine", "create", "x"],
        tmp_path,
        before=before,
        list_dirs=lambda: [p for p in base.iterdir() if p.is_dir()],
    )
    assert found is None  # the only dir was already in `before`
    assert "exited" in err  # surfaced the early exit, not a 25s timeout


def test_spawn_and_locate_surfaces_spawn_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> object:
        raise OSError("no exec")

    monkeypatch.setattr(_spawn.subprocess, "Popen", _boom)
    found, err = _spawn.spawn_and_locate(
        ["agent6", "run", "x"], tmp_path, before=set(), list_dirs=list
    )
    assert found is None
    assert "failed to start agent6 run" in err
