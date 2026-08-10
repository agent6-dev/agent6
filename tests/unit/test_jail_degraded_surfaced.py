# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A run whose jail came up degraded says so once, at session open.

`JailSession.open()` reads the launcher's setup stderr at its ready handshake
(a refused /proc mount under rootless podman, a skipped grant) and stores it.
The dispatcher surfaces that ONCE when it opens the run's single session --
not per command, where it would repeat -- so a jail that still runs but is
weaker than asked does not surface only as a puzzling command failure later.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.events import EventSink
from agent6.tools.dispatch import ToolDispatcher


class _StubSession:
    def __init__(self, startup_stderr: str) -> None:
        self.startup_stderr = startup_stderr

    def close(self) -> None:  # pragma: no cover - dispatcher teardown
        pass


def _patch_open(monkeypatch: pytest.MonkeyPatch, stub: _StubSession) -> None:
    def fake_open(cls: object, policy: object, *, session_net: object = None) -> _StubSession:
        return stub

    monkeypatch.setattr("agent6.tools.dispatch.JailSession.open", classmethod(fake_open))


def _events(path: Path, kind: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        e
        for line in path.read_text(encoding="utf-8").splitlines()
        if (e := json.loads(line)).get("type") == kind
    ]


def _dispatcher(tmp_path: Path, events: EventSink, stub: _StubSession) -> ToolDispatcher:
    # network = "host" so the policy needs no real session netns for this unit
    # test; isolation must be strict for a session to open at all.
    (tmp_path / "s").mkdir(exist_ok=True)
    return ToolDispatcher(
        root=tmp_path,
        config=Config.model_validate({"sandbox": {"network": "host"}}),
        isolation="strict",
        events=events,
        session_dir=tmp_path / "s",
        use_jail_session=True,
    )


def test_a_degraded_session_emits_jail_degraded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warning = "[agent6-jail] warning: fresh /proc mount failed (EPERM)"
    stub = _StubSession(warning)
    _patch_open(monkeypatch, stub)
    log = tmp_path / "e.jsonl"
    d = _dispatcher(tmp_path, EventSink(log), stub)
    try:
        assert d._run_session() is stub  # pyright: ignore[reportPrivateUsage]
        d._run_session()  # already open -> no second emit  # pyright: ignore[reportPrivateUsage]
    finally:
        d.close()
    degraded = _events(log, "jail.degraded")
    assert len(degraded) == 1, degraded
    assert warning in str(degraded[0].get("detail"))


def test_a_clean_session_emits_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubSession("")
    _patch_open(monkeypatch, stub)
    log = tmp_path / "e.jsonl"
    d = _dispatcher(tmp_path, EventSink(log), stub)
    try:
        d._run_session()  # pyright: ignore[reportPrivateUsage]
    finally:
        d.close()
    assert _events(log, "jail.degraded") == []


def test_concurrent_callers_open_exactly_one_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lazy open is a check-then-set, so two threads reaching it together
    each started a launcher and one was dropped on the floor.

    A dropped `JailSession` leaks its launcher process, its namespaces and
    (under `network = "session"`) its network holder, with nothing left holding
    a handle to close them. Nothing calls a command tool concurrently on one
    dispatcher TODAY -- review seats are read-only -- but that was an invariant
    written in a comment, not enforced. The sleep widens the window the race
    needs; without the lock this opens two.
    """
    import threading
    import time

    opened: list[_StubSession] = []

    def slow_open(cls: object, policy: object, *, session_net: object = None) -> _StubSession:
        time.sleep(0.05)  # the window a check-then-set leaves open
        stub = _StubSession("")
        opened.append(stub)
        return stub

    monkeypatch.setattr("agent6.tools.dispatch.JailSession.open", classmethod(slow_open))
    d = _dispatcher(tmp_path, EventSink(tmp_path / "e.jsonl"), _StubSession(""))
    seen: list[object] = []
    try:
        threads = [
            threading.Thread(
                target=lambda: seen.append(d._run_session())  # pyright: ignore[reportPrivateUsage]
            )
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        d.close()

    assert len(opened) == 1, f"opened {len(opened)} sessions; {len(opened) - 1} leaked"
    assert seen == [opened[0]] * 4
