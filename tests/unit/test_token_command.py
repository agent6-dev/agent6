# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.providers.token_command.CommandToken."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.providers import ProviderError
from agent6.providers.token_command import CommandToken


def _counter_argv(tmp_path: Path) -> list[str]:
    """argv that prints ``tok1``, ``tok2``, ... on successive runs."""
    counter = tmp_path / "counter"
    script = (
        f'n=$(cat "{counter}" 2>/dev/null || echo 0); '
        f'n=$((n + 1)); printf %s "$n" > "{counter}"; printf "tok%s" "$n"'
    )
    return ["sh", "-c", script]


def test_token_runs_command_and_strips_output() -> None:
    cred = CommandToken(["printf", "  sk-minted\n"])
    assert cred.token() == "sk-minted"


def test_token_is_cached_within_ttl(tmp_path: Path) -> None:
    cred = CommandToken(_counter_argv(tmp_path), ttl_s=1000.0)
    assert cred.token() == "tok1"
    assert cred.token() == "tok1"  # cached: the command did not re-run


def test_invalidate_forces_rerun(tmp_path: Path) -> None:
    cred = CommandToken(_counter_argv(tmp_path), ttl_s=1000.0)
    assert cred.token() == "tok1"
    cred.invalidate()
    assert cred.token() == "tok2"


def test_token_refreshes_after_ttl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 1000.0}
    monkeypatch.setattr("agent6.providers.token_command.time.monotonic", lambda: clock["t"])
    cred = CommandToken(_counter_argv(tmp_path), ttl_s=300.0)
    assert cred.token() == "tok1"
    clock["t"] += 299.0
    assert cred.token() == "tok1"  # still within TTL
    clock["t"] += 2.0  # 301s elapsed -> stale
    assert cred.token() == "tok2"


def test_nonzero_exit_raises_with_stderr() -> None:
    cred = CommandToken(["sh", "-c", "echo boom >&2; exit 3"])
    with pytest.raises(ProviderError, match=r"exited 3.*boom"):
        cred.token()


def test_empty_output_raises() -> None:
    cred = CommandToken(["true"])
    with pytest.raises(ProviderError, match="no output"):
        cred.token()


def test_missing_command_raises() -> None:
    cred = CommandToken(["agent6-nonexistent-token-binary-xyz"])
    with pytest.raises(ProviderError, match="not found"):
        cred.token()
