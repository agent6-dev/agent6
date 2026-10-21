# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hold the secret `agent6 web` requires: mint it, normalize it, and check a session cookie.

A browser signs in once with the secret and keeps a cookie holding a keyed digest of it,
so the server stores and compares the digest, and no cookie carries the secret.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import re
import secrets
from typing import Literal

from agent6 import child_env
from agent6._data import words

SECRET_ENV = child_env.WEB_SECRET_ENV
MIN_SECRET_CHARS = 12
_MINTED_PAIRS = 3  # about 44 bits over the wordlist
_SESSION_MESSAGE = b"agent6-web-session-v1"
# `Secure` only behind a proxy that terminated TLS: the server speaks plain HTTP on loopback,
# where a browser would drop a Secure cookie.
_COOKIE_ATTRIBUTES = "HttpOnly; SameSite=Strict; Path=/; Max-Age=34560000"
_WORD_BREAKS = re.compile(r"[\s-]+")


class WebSecretError(Exception):
    """An operator secret `agent6 web` refuses to start with."""


@dataclasses.dataclass(frozen=True, slots=True)
class Secret:
    """One launch's secret.

    Attributes:
        value: The normalized secret.
        origin: Where it came from: minted here, the `--secret` flag, or the variable.
    """

    value: str
    origin: Literal["minted", "flag", "env"]


def normalize(text: str) -> str:
    """Return the secret as both sides compare it: lowercase words joined by one `-`."""
    return "-".join(part for part in _WORD_BREAKS.split(text.strip().casefold()) if part)


def mint() -> str:
    """Return a new secret: three `<adjective>-<noun>` pairs from the session-id wordlist."""
    pairs = (
        f"{secrets.choice(words.ADJECTIVES)}-{secrets.choice(words.NOUNS)}"
        for _ in range(_MINTED_PAIRS)
    )
    return "-".join(pairs)


def _operator_secret(raw: str, name: str) -> str:
    """Return the operator's secret normalized.

    Args:
        raw: The secret as the operator gave it.
        name: The flag or variable, named in the refusal.

    Returns:
        The secret as lowercase words joined by `-`.

    Raises:
        WebSecretError: It is under `MIN_SECRET_CHARS`, too short to resist guessing.
    """
    value = normalize(raw)
    if len(value) < MIN_SECRET_CHARS:
        raise WebSecretError(
            f"the {name} value is {len(value)} characters; a secret needs at least"
            f" {MIN_SECRET_CHARS} characters, counted in lowercase with its words joined"
            f" by `-`. Leave {name} out to have `agent6 web` mint one."
        )
    return value


def resolve(flag: str | None, env: str | None) -> Secret:
    """Return the launch's secret: the flag's, else the variable's, else a minted one.

    Args:
        flag: The `--secret` value, or None.
        env: The `AGENT6_WEB_SECRET` value, or None.

    Returns:
        The normalized secret and its origin.

    Raises:
        WebSecretError: The operator's secret is under `MIN_SECRET_CHARS` once normalized.
    """
    if flag is not None:
        return Secret(_operator_secret(flag, "--secret"), "flag")
    if env is not None:
        return Secret(_operator_secret(env, SECRET_ENV), "env")
    return Secret(mint(), "minted")


def session_token(secret: str) -> str:
    """Return the session cookie's value for a secret: a keyed digest, stable across restarts."""
    return hmac.new(normalize(secret).encode(), _SESSION_MESSAGE, hashlib.sha256).hexdigest()


def token_matches(candidate: str, token: str) -> bool:
    """Return whether a presented cookie value or login digest equals the server's token."""
    return hmac.compare_digest(candidate.encode(), token.encode())


def cookie_name(port: int) -> str:
    """Return the cookie's name; cookies ignore ports, so two servers on a host get two names."""
    return f"agent6_web_{port}"


def set_cookie(port: int, token: str, *, secure: bool = False) -> str:
    """Return the `Set-Cookie` header a successful login answers with.

    Args:
        port: The server's port, which names the cookie.
        token: The session token.
        secure: Add `Secure`, for a browser that reached the server over HTTPS.

    Returns:
        The header's value.
    """
    attributes = f"{_COOKIE_ATTRIBUTES}; Secure" if secure else _COOKIE_ATTRIBUTES
    return f"{cookie_name(port)}={token}; {attributes}"


def has_session(cookie_headers: list[str], port: int, token: str) -> bool:
    """Return whether the request's `Cookie` headers carry this server's session token.

    Args:
        cookie_headers: Every `Cookie` header the request sent.
        port: The server's port, which names its cookie.
        token: The server's session token.

    Returns:
        True when the cookie named for `port` holds `token`.
    """
    name = cookie_name(port)
    for header in cookie_headers:
        for part in header.split(";"):
            key, _, value = part.strip().partition("=")
            if key == name and token_matches(value, token):
                return True
    return False
