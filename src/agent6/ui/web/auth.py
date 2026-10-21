# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hold the secret `agent6 web` requires: mint it, normalize it, and check a sign-in token.

A browser signs in once with the secret and keeps the keyed digest the login answers, in
storage scoped to the server's origin (host and port), and sends it as a bearer header. A
cookie would reach every port of the host, so a page another local server serves would
receive it. The server stores and compares the digest; nothing carries the secret after
the sign-in.
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
# v2: the v1 digest rode a cookie, which a browser sends to every port of the host; a cookie a
# browser kept from then must not equal the token.
_SESSION_MESSAGE = b"agent6-web-token-v2"
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
    """Return the sign-in token for a secret: a keyed digest, stable across restarts."""
    return hmac.new(normalize(secret).encode(), _SESSION_MESSAGE, hashlib.sha256).hexdigest()


def token_matches(candidate: str, token: str) -> bool:
    """Return whether a presented token or login digest equals the server's token."""
    return hmac.compare_digest(candidate.encode(), token.encode())


def expired_cookie(port: int) -> str:
    """Return the `Set-Cookie` that clears the cookie an earlier version set for this port."""
    return f"agent6_web_{port}=; Max-Age=0; Path=/"


def bearer_matches(authorization: str | None, token: str) -> bool:
    """Return whether an `Authorization` header carries this server's sign-in token.

    Args:
        authorization: The request's `Authorization` header, or None.
        token: The server's token.

    Returns:
        True when the header is `Bearer <token>`.
    """
    scheme, _, value = (authorization or "").strip().partition(" ")
    return scheme.lower() == "bearer" and token_matches(value.strip(), token)
