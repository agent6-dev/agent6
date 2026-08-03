# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.secrets (storage, permissions, key resolution)."""

from __future__ import annotations

import stat
import threading
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


def test_save_secret_escapes_control_chars(gcfg: Path) -> None:
    # A control char in a pasted key must not write unparseable secrets.toml.
    # A raw newline/\x01 in a basic string is illegal TOML, so the whole file
    # fails to parse and EVERY provider's key reads back missing -- while the
    # save reported success.
    secrets.save_secret("openrouter", "sk-or-clean")
    secrets.save_secret("anthropic", "sk-\x01\nbroken")
    assert secrets.resolve_api_key("anthropic", None) == "sk-\x01\nbroken"
    assert secrets.resolve_api_key("openrouter", None) == "sk-or-clean"


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


def test_save_secret_does_not_follow_a_planted_tmp_symlink(gcfg: Path, tmp_path: Path) -> None:
    """A pre-planted `secrets.toml.tmp` symlink must not redirect the write to
    its target (the sudo-connect symlink-redirect vector). atomic_write uses an
    unpredictable mkstemp name, so a fixed-name symlink is simply ignored."""
    victim = tmp_path / "victim"
    victim.write_text("KEEP ME\n", encoding="utf-8")
    gcfg.mkdir(parents=True, exist_ok=True)
    (gcfg / "secrets.toml.tmp").symlink_to(victim)
    secrets.save_secret("anthropic", "sk-ant-xyz")
    assert victim.read_text(encoding="utf-8") == "KEEP ME\n"  # untouched
    assert secrets.resolve_api_key("anthropic", None) == "sk-ant-xyz"
    assert not (gcfg / "secrets.toml").is_symlink()


def test_concurrent_save_secret_loses_no_provider(gcfg: Path) -> None:
    """Two concurrent connects both read the same base file and the later
    publish silently dropped the earlier provider's credential (lost update).
    save_secret serializes on portable.locked_file, removed on release."""
    n = 8
    barrier = threading.Barrier(n)

    def save(i: int) -> None:
        barrier.wait()
        secrets.save_secret(f"prov{i}", f"sk-{i}")

    threads = [threading.Thread(target=save, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    for i in range(n):
        assert secrets.resolve_api_key(f"prov{i}", None) == f"sk-{i}"
    p = secrets.save_secret("final", "sk-final")
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert not p.with_name(p.name + ".lock").exists()
