# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.secrets (storage, permissions, key resolution)."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from agent6 import secrets
from agent6.secrets import SecretsError


@pytest.fixture
def gcfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    return tmp_path / "g"


def test_save_secret_is_0600(gcfg: Path) -> None:
    p = secrets.save_secret("anthropic", "sk-ant-xyz")
    assert p.is_file()
    mode = stat.S_IMODE(p.stat().st_mode)
    assert mode == 0o600
    assert secrets.resolve_api_key("anthropic", None) == "sk-ant-xyz"


def test_save_secret_preserves_other_providers(gcfg: Path) -> None:
    secrets.save_secret("anthropic", "sk-ant-1")
    secrets.save_secret("openrouter", "sk-or-2")
    assert secrets.resolve_api_key("anthropic", None) == "sk-ant-1"
    assert secrets.resolve_api_key("openrouter", None) == "sk-or-2"


def test_env_takes_precedence_over_secrets(gcfg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secrets.save_secret("anthropic", "from-secrets")
    monkeypatch.setenv("MY_KEY", "from-env")
    assert secrets.resolve_api_key("anthropic", "MY_KEY") == "from-env"
    # Empty env falls back to secrets.
    monkeypatch.setenv("MY_KEY", "")
    assert secrets.resolve_api_key("anthropic", "MY_KEY") == "from-secrets"


def test_resolve_missing_returns_none(gcfg: Path) -> None:
    assert secrets.resolve_api_key("nope", None) is None


def test_load_secrets_refuses_group_readable(gcfg: Path) -> None:
    p = secrets.save_secret("anthropic", "sk-ant-xyz")
    p.chmod(0o644)
    with pytest.raises(SecretsError, match="unsafe permissions"):
        secrets.load_secrets()


def test_load_secrets_absent_is_empty(gcfg: Path) -> None:
    assert secrets.load_secrets() == {}
