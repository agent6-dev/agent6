# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Store provider API keys and OAuth tokens in `secrets.toml`.

The file sits beside the config and is treated like an SSH private key: a
regular file owned by the operator, `0600` or refused, written atomically at
`0600` and chowned back to the real user under `sudo`. A provider's key comes
from its `api_key_env` variable first, then `secrets.toml`, else nothing.
Secrets never reach transcripts, `config show` or the jail.
"""

from __future__ import annotations

import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent6.paths import (
    RealUser,
    chown_to_real_user,
    effective_user,
    mkdir_for_real_user,
    secrets_path,
)
from agent6.portable import atomic_write, locked_file, toml_basic_string


class SecretsError(Exception):
    """The secrets file is malformed, unreadable or unsafely permitted."""


def _require_safe_perms(path: Path, user: RealUser) -> None:
    """Refuse a secrets file that others can read or that the operator does not own.

    Raises:
        SecretsError: The file is not regular, has group or other bits, or has another owner.
    """
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise SecretsError(f"{path} is not a regular file; refusing to read secrets from it.")
    if st.st_mode & 0o077:
        raise SecretsError(
            f"{path} has unsafe permissions {stat.S_IMODE(st.st_mode):#o}"
            f" (group/other accessible). Run: chmod 600 {path}"
        )
    # Under sudo the read takes the real user's file as root.
    if os.geteuid() != 0 and st.st_uid != user.uid:
        raise SecretsError(
            f"{path} is owned by uid {st.st_uid}, not you (uid {user.uid});"
            " refusing to read secrets you do not own."
        )


def _read_secrets_toml(path: Path) -> dict[str, Any]:
    """Parse the secrets file, the one reader.

    Returns:
        The parsed TOML.

    Raises:
        SecretsError: The file cannot be read or is not valid TOML; an unreadable file
            is the operator's environment, never a crash.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SecretsError(f"could not read {path}: {exc}") from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SecretsError(f"{path} is not valid TOML: {exc}") from exc


def load_secrets(user: RealUser | None = None) -> dict[str, Any]:
    """Return the validated secrets; empty when the file is absent."""
    user = user or effective_user()
    path = secrets_path(user)
    if not path.exists():
        return {}
    _require_safe_perms(path, user)
    return _read_secrets_toml(path)


def resolve_api_key(
    provider_name: str,
    api_key_env: str | None,
    *,
    secrets: dict[str, Any] | None = None,
    user: RealUser | None = None,
) -> str | None:
    """Return one provider's API key, the env variable first, then the secrets file, else None."""
    if api_key_env:
        env_val = os.environ.get(api_key_env, "").strip()
        if env_val:
            return env_val
    data = secrets if secrets is not None else load_secrets(user)
    providers = data.get("providers")
    if isinstance(providers, dict):
        entry = providers.get(provider_name)
        if isinstance(entry, dict):
            key = entry.get("api_key")
            if isinstance(key, str) and key.strip():
                return key.strip()
    return None


def save_secret(
    provider_name: str,
    api_key: str,
    *,
    extra: dict[str, str] | None = None,
    user: RealUser | None = None,
) -> Path:
    """Write a provider's `api_key` and any extra string fields, replacing its entry.

    Returns:
        The secrets file's path.
    """
    return _save_provider_entry(provider_name, {"api_key": api_key, **(extra or {})}, user)


def _save_provider_entry(provider_name: str, entry: dict[str, str], user: RealUser | None) -> Path:
    """Replace one provider's entry, rewriting the whole file under the lock.

    Two concurrent writers (a `connect` beside a run refreshing tokens) would
    otherwise read the same base file and the later publish drop the earlier
    credential.

    Returns:
        The secrets file's path.
    """
    user = user or effective_user()
    path = secrets_path(user)
    # Created 0700 and handed back here, before the lock's parent walk creates it at the umask.
    mkdir_for_real_user(path.parent, user)
    path.parent.chmod(0o700)
    with locked_file(path):
        data: dict[str, Any] = {}
        if path.exists():
            _require_safe_perms(path, user)
            data = _read_secrets_toml(path)
        providers = data.get("providers")
        if not isinstance(providers, dict):
            providers = {}
        providers[provider_name] = entry
        data["providers"] = providers

        text = _render_secrets_toml(data)
        # mkstemp opens an unpredictable name O_EXCL, so a planted `.tmp` symlink cannot redirect.
        atomic_write(path, text)
        path.chmod(0o600)
    chown_to_real_user(path.parent, user)
    chown_to_real_user(path, user)
    return path


def delete_provider_secrets(provider_name: str, *, user: RealUser | None = None) -> bool:
    """Remove one provider's entry.

    Returns:
        True when it existed.
    """
    user = user or effective_user()
    path = secrets_path(user)
    if not path.exists():
        return False  # before the lock, whose parent walk would create the config dir
    with locked_file(path):
        if not path.exists():
            return False
        _require_safe_perms(path, user)
        data: dict[str, Any] = _read_secrets_toml(path)
        providers = data.get("providers")
        if not isinstance(providers, dict) or provider_name not in providers:
            return False
        del providers[provider_name]
        atomic_write(path, _render_secrets_toml(data))
        path.chmod(0o600)
    chown_to_real_user(path, user)
    return True


@dataclass(frozen=True, slots=True)
class OAuthTokens:
    """One provider's OAuth grant as stored.

    Attributes:
        access_token: The access token.
        refresh_token: The refresh token.
        expires_at: When the access token expires, as a Unix time.
        account_id: The backend account the tokens are bound to; "" when unknown.
    """

    access_token: str
    refresh_token: str
    expires_at: float
    account_id: str = ""


def save_oauth_tokens(
    provider_name: str, tokens: OAuthTokens, *, user: RealUser | None = None
) -> Path:
    """Write a provider's OAuth tokens, replacing its entry.

    Returns:
        The secrets file's path.
    """
    return _save_provider_entry(
        provider_name,
        {
            "oauth_access_token": tokens.access_token,
            "oauth_refresh_token": tokens.refresh_token,
            "oauth_expires_at": repr(tokens.expires_at),
            "oauth_account_id": tokens.account_id,
        },
        user,
    )


def load_oauth_tokens(
    provider_name: str,
    *,
    secrets: dict[str, Any] | None = None,
    user: RealUser | None = None,
) -> OAuthTokens | None:
    """Return one provider's stored OAuth tokens, or None when absent or mangled.

    A mangled entry reads as absent: every caller's None path says to run
    `agent6 connect`, which is also the repair.
    """
    data = secrets if secrets is not None else load_secrets(user)
    providers = data.get("providers")
    entry = providers.get(provider_name) if isinstance(providers, dict) else None
    if not isinstance(entry, dict):
        return None
    access = entry.get("oauth_access_token")
    refresh = entry.get("oauth_refresh_token")
    if not (isinstance(access, str) and access and isinstance(refresh, str) and refresh):
        return None
    try:
        expires_at = float(str(entry.get("oauth_expires_at", "0")))
    except ValueError:
        return None
    return OAuthTokens(
        access_token=access,
        refresh_token=refresh,
        expires_at=expires_at,
        account_id=str(entry.get("oauth_account_id", "") or ""),
    )


def _render_secrets_toml(data: dict[str, Any]) -> str:
    """Return the secrets as TOML: a flat `[providers.<name>]` table of strings each."""
    lines = [
        "# agent6 secrets. Written by `agent6 connect`.",
        "# Keep this file private: it is enforced 0600 and owner-only.",
        "",
    ]
    providers = data.get("providers")
    if isinstance(providers, dict):
        for name in sorted(providers):
            entry = providers[name]
            if not isinstance(entry, dict):
                continue
            lines.append(f"[providers.{name}]")
            for field in sorted(entry):
                value = entry[field]
                if isinstance(value, str):
                    lines.append(f"{field} = {toml_basic_string(value)}")
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
