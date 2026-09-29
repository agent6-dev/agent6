# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.secret_store (storage, permissions, key resolution)."""

from __future__ import annotations

import pathlib
import stat
import threading

import pytest

from agent6 import secret_store


@pytest.fixture
def gcfg(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    return tmp_path / "g"


def test_save_secret_is_0600(gcfg: pathlib.Path) -> None:
    p = secret_store.save_secret("anthropic", "sk-ant-xyz")
    assert p.is_file()
    mode = stat.S_IMODE(p.stat().st_mode)
    assert mode == 0o600
    assert secret_store.resolve_api_key("anthropic", None) == "sk-ant-xyz"


def test_an_unreadable_secrets_file_is_a_named_refusal(gcfg: pathlib.Path) -> None:
    """Root-owned after a `sudo connect`, or a plain chmod 000.

    The operator's environment, not a bug in agent6. It escaped as an unexpected PermissionError
    with a saved traceback and an invitation to report it, and no run could start.
    """
    path = secret_store.save_secret("anthropic", "sk-ant-xyz")
    path.chmod(0o000)
    try:
        with pytest.raises(secret_store.SecretsError, match="could not read"):
            secret_store.load_secrets()
    finally:
        path.chmod(0o600)


def test_save_secret_preserves_other_providers(gcfg: pathlib.Path) -> None:
    secret_store.save_secret("anthropic", "sk-ant-1")
    secret_store.save_secret("openrouter", "sk-or-2")
    assert secret_store.resolve_api_key("anthropic", None) == "sk-ant-1"
    assert secret_store.resolve_api_key("openrouter", None) == "sk-or-2"


def test_save_secret_escapes_control_chars(gcfg: pathlib.Path) -> None:
    # A control char in a pasted key must not write unparseable secrets.toml.
    # A raw newline/\x01 in a basic string is illegal TOML, so the whole file
    # fails to parse and EVERY provider's key reads back missing -- while the
    # save reported success.
    secret_store.save_secret("openrouter", "sk-or-clean")
    secret_store.save_secret("anthropic", "sk-\x01\nbroken")
    assert secret_store.resolve_api_key("anthropic", None) == "sk-\x01\nbroken"
    assert secret_store.resolve_api_key("openrouter", None) == "sk-or-clean"


def test_env_takes_precedence_over_secrets(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_store.save_secret("anthropic", "from-secrets")
    monkeypatch.setenv("MY_KEY", "from-env")
    assert secret_store.resolve_api_key("anthropic", "MY_KEY") == "from-env"
    # Empty env falls back to secrets.
    monkeypatch.setenv("MY_KEY", "")
    assert secret_store.resolve_api_key("anthropic", "MY_KEY") == "from-secrets"


def test_resolve_missing_returns_none(gcfg: pathlib.Path) -> None:
    assert secret_store.resolve_api_key("nope", None) is None


def test_load_secrets_refuses_group_readable(gcfg: pathlib.Path) -> None:
    p = secret_store.save_secret("anthropic", "sk-ant-xyz")
    p.chmod(0o644)
    with pytest.raises(secret_store.SecretsError, match="unsafe permissions"):
        secret_store.load_secrets()


def test_load_secrets_absent_is_empty(gcfg: pathlib.Path) -> None:
    assert secret_store.load_secrets() == {}


def test_save_secret_does_not_follow_a_planted_tmp_symlink(
    gcfg: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """save_secret does not follow a planted tmp symlink.

    A pre-planted `secrets.toml.tmp` symlink must not redirect the write to its target (the sudo-
    connect symlink-redirect vector); atomic_write uses an unpredictable mkstemp name, so a fixed-
    name symlink is ignored.
    """
    victim = tmp_path / "victim"
    victim.write_text("KEEP ME\n", encoding="utf-8")
    gcfg.mkdir(parents=True, exist_ok=True)
    (gcfg / "secrets.toml.tmp").symlink_to(victim)
    secret_store.save_secret("anthropic", "sk-ant-xyz")
    assert victim.read_text(encoding="utf-8") == "KEEP ME\n"  # untouched
    assert secret_store.resolve_api_key("anthropic", None) == "sk-ant-xyz"
    assert not (gcfg / "secrets.toml").is_symlink()


def test_concurrent_save_secret_loses_no_provider(gcfg: pathlib.Path) -> None:
    """Concurrent save_secret calls lose no provider.

    Two concurrent connects reading the same base file would let the later publish drop the earlier
    provider's credential (a lost update); save_secret serializes on portable.locked_file, removed
    on release.
    """
    n = 8
    barrier = threading.Barrier(n)

    def save(i: int) -> None:
        barrier.wait()
        secret_store.save_secret(f"prov{i}", f"sk-{i}")

    threads = [threading.Thread(target=save, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    for i in range(n):
        assert secret_store.resolve_api_key(f"prov{i}", None) == f"sk-{i}"
    p = secret_store.save_secret("final", "sk-final")
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert not p.with_name(p.name + ".lock").exists()


def test_oauth_tokens_round_trip_beside_api_keys(gcfg: pathlib.Path) -> None:
    """OAuth tokens replace their provider's entry, preserve siblings, stay 0600."""
    secret_store.save_secret("anthropic", "sk-ant-123")
    tokens = secret_store.OAuthTokens(
        access_token="eyJ.access", refresh_token="rt-1", expires_at=1755.5, account_id="acct-9"
    )
    path = secret_store.save_oauth_tokens("chatgpt", tokens)
    assert (path.stat().st_mode & 0o777) == 0o600
    assert secret_store.load_oauth_tokens("chatgpt") == tokens
    assert secret_store.resolve_api_key("anthropic", None) == "sk-ant-123"
    # Re-connect rotates the whole entry; no stale fields survive.
    secret_store.save_oauth_tokens(
        "chatgpt", secret_store.OAuthTokens("a2", "r2", 2000.0, "acct-9")
    )
    loaded = secret_store.load_oauth_tokens("chatgpt")
    assert loaded is not None and loaded.access_token == "a2" and loaded.refresh_token == "r2"


def test_load_oauth_tokens_absent_or_mangled_is_none(gcfg: pathlib.Path) -> None:
    """load_oauth_tokens reads an absent or mangled entry as None.

    No entry, an api-key-only entry and an unparseable expiry all read as absent; the caller's
    repair path is `agent6 connect` either way.
    """
    assert secret_store.load_oauth_tokens("chatgpt") is None
    secret_store.save_secret("chatgpt", "sk-not-oauth")
    assert secret_store.load_oauth_tokens("chatgpt") is None
    assert (
        secret_store.load_oauth_tokens(
            "chatgpt",
            secrets={
                "providers": {
                    "chatgpt": {
                        "oauth_access_token": "a",
                        "oauth_refresh_token": "r",
                        "oauth_expires_at": "not-a-float",
                    }
                }
            },
        )
        is None
    )


def test_delete_provider_secrets_preserves_siblings(gcfg: pathlib.Path) -> None:
    secret_store.save_secret("anthropic", "sk-1")
    secret_store.save_oauth_tokens("chatgpt", secret_store.OAuthTokens("a", "r", 100.0, "id"))
    assert secret_store.delete_provider_secrets("chatgpt") is True
    assert secret_store.delete_provider_secrets("chatgpt") is False
    assert secret_store.load_oauth_tokens("chatgpt") is None
    assert secret_store.resolve_api_key("anthropic", None) == "sk-1"


def test_a_logout_on_a_fresh_machine_creates_no_open_config_dir(gcfg: pathlib.Path) -> None:
    """A logout on a fresh machine creates no open config dir.

    `delete_provider_secrets` takes its lock before anything has created the config dir, and the
    lock file's own parent walk would make `$XDG_CONFIG_HOME/agent6` at the umask's 755 with nothing
    left to tighten it; the config dir is created by the state tree's one creator, 0700, or not at
    all.
    """
    import os

    old = os.umask(0o022)
    try:
        assert secret_store.delete_provider_secrets("nobody") is False
        assert not (gcfg / "agent6").exists()
        secret_store.save_secret("anthropic", "sk-1")
    finally:
        os.umask(old)
    assert stat.S_IMODE((gcfg / "agent6").stat().st_mode) == 0o700
