# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the best-effort desktop-notification helper."""

from __future__ import annotations

from typing import Any

import pytest

from agent6.frontend import notify


def _which_none(_name: str) -> str | None:
    return None


def _which_found(_name: str) -> str | None:
    return "/usr/bin/notify-send"


def test_desktop_notify_noop_when_notify_send_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(notify.shutil, "which", _which_none)
    called: list[Any] = []

    def fake_popen(*a: Any, **_k: Any) -> Any:
        called.append(a)

    monkeypatch.setattr(notify.subprocess, "Popen", fake_popen)
    assert notify.desktop_notify("t", "b") is False
    assert called == []


def test_desktop_notify_fires_fixed_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(notify.shutil, "which", _which_found)
    seen: list[list[str]] = []

    def fake_popen(argv: list[str], **_kw: Any) -> Any:
        seen.append(argv)
        return object()

    monkeypatch.setattr(notify.subprocess, "Popen", fake_popen)
    assert notify.desktop_notify("agent6: m", "attention") is True
    # Fixed argv: exe + `--` end-of-options + two positional data args, no shell.
    assert seen == [["/usr/bin/notify-send", "--", "agent6: m", "attention"]]


def test_desktop_notify_dash_leading_message_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    # A message beginning with '-' must not be parsed as a notify-send option.
    monkeypatch.setattr(notify.shutil, "which", _which_found)
    seen: list[list[str]] = []

    def fake_popen(argv: list[str], **_kw: Any) -> Any:
        seen.append(argv)

    monkeypatch.setattr(notify.subprocess, "Popen", fake_popen)
    notify.desktop_notify("agent6: m", "-t 0 --hint=x")
    assert seen == [["/usr/bin/notify-send", "--", "agent6: m", "-t 0 --hint=x"]]


def test_desktop_notify_swallows_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(notify.shutil, "which", _which_found)

    def boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("no display")

    monkeypatch.setattr(notify.subprocess, "Popen", boom)
    assert notify.desktop_notify("t", "b") is False
