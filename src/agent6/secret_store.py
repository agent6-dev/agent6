# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Store provider credentials and telemetry headers in `secrets.toml`.

The file sits beside the config and is treated like an SSH private key: a
regular file owned by the operator, `0600` or refused, written atomically at
`0600` and chowned back to the real user under `sudo`. A provider's key comes
from its `api_key_env` variable first, then `secrets.toml`, else nothing.
Secrets never reach transcripts, `config show` or the jail.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import os
import pathlib
import re
import stat
import tomllib
from collections.abc import Generator, Iterable
from typing import Any

from agent6 import paths, portable

# A TOML bare key; any other header name is written quoted.
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
# What a header value may hold on the wire: visible ASCII and spaces.
_HEADER_VALUE = re.compile(r"[\x20-\x7e]+")


class SecretsError(Exception):
    """The secrets file is malformed, unreadable or unsafely permitted."""


def _read_secrets_toml(path: pathlib.Path, user: paths.RealUser) -> dict[str, Any]:
    """Parse the secrets file, the one reader; the file is judged by the descriptor read.

    The open follows no link and the checks run on the opened descriptor, so a path
    swapped between a check and a read cannot hand the reader another file.

    Args:
        path: The secrets file.
        user: The operator, who must own the file unless agent6 runs as root.

    Returns:
        The parsed TOML.

    Raises:
        FileNotFoundError: There is no secrets file.
        SecretsError: The path is a link or not a regular file, has group or other bits,
            has another owner, cannot be read, or is not valid TOML.
    """
    # O_NONBLOCK: a FIFO at the path opens at once and fails the regular-file check below.
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise SecretsError(
                f"{path} is a symlink; refusing to read secrets through it."
            ) from exc
        raise SecretsError(f"could not read {path}: {exc}") from exc
    try:
        st = os.fstat(fd)
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
        try:
            chunks: list[bytes] = []
            while chunk := os.read(fd, 1 << 16):  # a network file may answer a short read
                chunks.append(chunk)
            text = b"".join(chunks).decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SecretsError(f"could not read {path}: {exc}") from exc
    finally:
        os.close(fd)
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SecretsError(f"{path} is not valid TOML: {exc}") from exc


def _lock_refusal(path: pathlib.Path) -> str:
    """Return the refusal text for a rewrite whose lock could not be taken."""
    return (
        f"could not take the lock beside {path}; refusing to rewrite the secrets file"
        " unserialized (remove a stale .lock sibling if one is left over)"
    )


def load_secrets(user: paths.RealUser | None = None) -> dict[str, Any]:
    """Return the validated secrets; empty when the file is absent."""
    user = user or paths.effective_user()
    try:
        return _read_secrets_toml(paths.secrets_path(user), user)
    except FileNotFoundError:
        return {}


def resolve_api_key(
    provider_name: str,
    api_key_env: str | None,
    *,
    secrets: dict[str, Any] | None = None,
    user: paths.RealUser | None = None,
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


def telemetry_headers(
    names: Iterable[str],
    *,
    secrets: dict[str, Any] | None = None,
    user: paths.RealUser | None = None,
) -> dict[str, str]:
    """Return the telemetry export's headers, each value from `[telemetry.headers]`.

    Args:
        names: The header names `[telemetry].headers` configures.
        secrets: The loaded secrets; None loads the file.
        user: The operator; None resolves the effective one.

    Returns:
        The headers, name to value.

    Raises:
        SecretsError: A named header has no value in the secrets file.
    """
    data = secrets if secrets is not None else load_secrets(user)
    table = data.get("telemetry")
    values = table.get("headers") if isinstance(table, dict) else None
    out: dict[str, str] = {}
    for name in names:
        value = values.get(name) if isinstance(values, dict) else None
        if not isinstance(value, str) or not value.strip():
            raise SecretsError(
                f'telemetry header {name!r} has no value: add `{name} = "..."` under'
                f" [telemetry.headers] in {paths.secrets_path(user)}"
            )
        if not _HEADER_VALUE.fullmatch(value.strip()):
            raise SecretsError(
                f"telemetry header {name!r} has a value outside visible ASCII; a header value"
                " carries no control character and no non-ASCII text"
            )
        out[name] = value.strip()
    return out


def save_secret(
    provider_name: str,
    api_key: str,
    *,
    extra: dict[str, str] | None = None,
    user: paths.RealUser | None = None,
) -> pathlib.Path:
    """Write a provider's `api_key` and any extra string fields, replacing its entry.

    Args:
        provider_name: The provider's name under `[providers]`.
        api_key: The provider's API key.
        extra: More fields, by name.
        user: The operator; None resolves the effective one.

    Returns:
        The secrets file's path.
    """
    return _save_provider_entry(provider_name, {"api_key": api_key, **(extra or {})}, user)


@contextlib.contextmanager
def _writing_secrets(user: paths.RealUser) -> Generator[dict[str, Any]]:
    """Hold the secrets lock across a validated read, an edit and a private atomic write.

    Args:
        user: The operator who owns the file.

    Yields:
        The secrets to edit; unchanged data causes no write.

    Raises:
        SecretsError: The lock cannot be taken or the file refuses to be read.
    """
    path = paths.secrets_path(user)
    # Created 0700 and handed back here, before the lock's parent walk creates it at the umask.
    paths.mkdir_for_real_user(path.parent, user)
    path.parent.chmod(0o700)
    with portable.locked_file(path) as held:
        if not held:
            raise SecretsError(_lock_refusal(path))
        try:
            data = _read_secrets_toml(path, user)
        except FileNotFoundError:
            data = {}
        before = _render_secrets_toml(data)
        yield data
        text = _render_secrets_toml(data)
        if text != before:
            # mkstemp opens an unpredictable name O_EXCL; the mode rides the atomic rename.
            portable.atomic_write(path, text, mode=0o600)
            paths.chown_to_real_user(path, user)
    paths.chown_to_real_user(path.parent, user)


def _save_provider_entry(
    provider_name: str, entry: dict[str, str], user: paths.RealUser | None
) -> pathlib.Path:
    """Replace one provider's entry under the shared secrets lock.

    Args:
        provider_name: The provider's name under `[providers]`.
        entry: The replacement string fields.
        user: The operator; None resolves the effective one.

    Returns:
        The secrets file's path.
    """
    user = user or paths.effective_user()
    with _writing_secrets(user) as data:
        providers = data.get("providers")
        if not isinstance(providers, dict):
            providers = {}
        providers[provider_name] = entry
        data["providers"] = providers
    return paths.secrets_path(user)


def write_telemetry_header(
    name: str, value: str | None, *, user: paths.RealUser | None = None
) -> pathlib.Path:
    """Set a literal telemetry header value, or remove it when the value is None.

    Args:
        name: The header name validated against the telemetry config schema.
        value: The literal value; None removes the entry.
        user: The operator; None resolves the effective one.

    Returns:
        The secrets file's path.

    Raises:
        SecretsError: The value is not a usable header, a table is malformed, or the file
            or its lock refuses access.
    """
    if value is not None:
        telemetry_headers((name,), secrets={"telemetry": {"headers": {name: value}}})
    user = user or paths.effective_user()
    path = paths.secrets_path(user)
    if value is None and not path.exists():
        return path
    with _writing_secrets(user) as data:
        telemetry = data.setdefault("telemetry", {})
        headers = telemetry.setdefault("headers", {}) if isinstance(telemetry, dict) else None
        if not isinstance(headers, dict):
            raise SecretsError("[telemetry.headers] in secrets.toml must be a table")
        if value is None:
            headers.pop(name, None)
        else:
            headers[name] = value
    return path


def delete_provider_secrets(provider_name: str, *, user: paths.RealUser | None = None) -> bool:
    """Remove one provider's entry.

    Args:
        provider_name: The provider's name under `[providers]`.
        user: The operator; None resolves the effective one.

    Returns:
        True when it existed.

    Raises:
        SecretsError: The lock beside the file could not be taken, or the file refuses to be read.
    """
    user = user or paths.effective_user()
    path = paths.secrets_path(user)
    if not path.exists():
        return False  # before the lock, whose parent walk would create the config dir
    with _writing_secrets(user) as data:
        providers = data.get("providers")
        if not isinstance(providers, dict) or provider_name not in providers:
            return False
        del providers[provider_name]
    return True


@dataclasses.dataclass(frozen=True, slots=True)
class StoredGrant:
    """One provider's OAuth grant as stored, bound to its settings.

    Attributes:
        binding: The fingerprint of the settings the grant was made under.
        access_token: The issuer's access token.
        refresh_token: The refresh token; "" when the issuer gave none.
        issued_at: When the issuer grant was issued, as Unix time.
        expires_at: When it expires; None when the issuer named no expiry.
        exchange_access_token: The second exchange's token; "" when none is configured or made.
        exchange_issued_at: When that token was issued; 0.0 when none.
        exchange_expires_at: When it expires; None when none.
        account_id: The account extracted from the grant; None when unknown.
    """

    binding: str
    access_token: str
    refresh_token: str
    issued_at: float
    expires_at: float | None
    exchange_access_token: str
    exchange_issued_at: float
    exchange_expires_at: float | None
    account_id: str | None = None


GRANT_VERSION = "1"


def save_oauth_grant(
    provider_name: str, grant: StoredGrant, *, user: paths.RealUser | None = None
) -> pathlib.Path:
    """Write a provider's bound OAuth grant, replacing its entry.

    Args:
        provider_name: The provider's name under `[providers]`.
        grant: The grant to store.
        user: The operator; None resolves the effective one.

    Returns:
        The secrets file's path.
    """
    entry = {
        "oauth_version": GRANT_VERSION,
        "oauth_binding": grant.binding,
        "oauth_access_token": grant.access_token,
        "oauth_issued_at": repr(grant.issued_at),
    }
    if grant.account_id is not None:
        entry["oauth_account_id"] = grant.account_id
    if grant.refresh_token:
        entry["oauth_refresh_token"] = grant.refresh_token
    if grant.expires_at is not None:
        entry["oauth_expires_at"] = repr(grant.expires_at)
    if grant.exchange_access_token:
        entry["oauth_exchange_access_token"] = grant.exchange_access_token
        entry["oauth_exchange_issued_at"] = repr(grant.exchange_issued_at)
        if grant.exchange_expires_at is not None:
            entry["oauth_exchange_expires_at"] = repr(grant.exchange_expires_at)
    return _save_provider_entry(provider_name, entry, user)


def _optional_time(entry: dict[str, Any], key: str) -> float | None:
    """Return a stored Unix time field, None when absent."""
    raw = entry.get(key)
    return float(str(raw)) if raw not in (None, "") else None


def load_oauth_grant(
    provider_name: str,
    *,
    secrets: dict[str, Any] | None = None,
    user: paths.RealUser | None = None,
) -> StoredGrant | None:
    """Return one provider's stored bound OAuth grant, or None when absent or mangled.

    Args:
        provider_name: The provider's name under `[providers]`.
        secrets: The loaded secrets; None loads the operator's.
        user: The operator; None resolves the effective one.

    Returns:
        The grant; None when absent, of another version, or mangled, which
        `agent6 connect` repairs.
    """
    data = secrets if secrets is not None else load_secrets(user)
    providers = data.get("providers")
    entry = providers.get(provider_name) if isinstance(providers, dict) else None
    if not isinstance(entry, dict) or str(entry.get("oauth_version", "")) != GRANT_VERSION:
        return None
    access = entry.get("oauth_access_token")
    binding = entry.get("oauth_binding")
    if not (isinstance(access, str) and access and isinstance(binding, str) and binding):
        return None
    account = entry.get("oauth_account_id")
    if account is not None and not isinstance(account, str):
        return None
    try:
        return StoredGrant(
            binding=binding,
            access_token=access,
            refresh_token=str(entry.get("oauth_refresh_token", "") or ""),
            issued_at=float(str(entry.get("oauth_issued_at", "0"))),
            expires_at=_optional_time(entry, "oauth_expires_at"),
            exchange_access_token=str(entry.get("oauth_exchange_access_token", "") or ""),
            exchange_issued_at=float(str(entry.get("oauth_exchange_issued_at", "0") or "0")),
            exchange_expires_at=_optional_time(entry, "oauth_exchange_expires_at"),
            account_id=account,
        )
    except ValueError:
        return None


def _render_secrets_toml(data: dict[str, Any]) -> str:
    """Return provider entries and telemetry headers as TOML string tables."""
    lines = [
        "# agent6 secrets. Written by `agent6 connect` and `agent6 config set --secret`.",
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
                    lines.append(f"{field} = {portable.toml_basic_string(value)}")
            lines.append("")
    telemetry = data.get("telemetry")
    headers = telemetry.get("headers") if isinstance(telemetry, dict) else None
    if isinstance(headers, dict) and headers:
        lines.append("[telemetry.headers]")
        for name in sorted(headers):
            value = headers[name]
            if isinstance(value, str):
                key = name if _BARE_KEY.fullmatch(name) else portable.toml_basic_string(name)
                lines.append(f"{key} = {portable.toml_basic_string(value)}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
