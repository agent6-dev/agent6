# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The OAuth wire shapes every sign-in shares: PKCE, the callback, the token grant.

A profile (the ChatGPT sign-in) or a configured issuer supplies the endpoints, the client
registration and the label its errors carry; this module owns the requests and reads back
nothing a remote returns except the validated grant fields and a scrubbed error text.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import secrets
from typing import Any
from urllib import parse

import httpx2

from agent6 import secret_store
from agent6.providers import types

TOKEN_TIMEOUT_S = 30.0
# The refresh token itself is dead; re-consent is the only repair.
PERMANENT_REFRESH_CODES = frozenset(
    {"refresh_token_expired", "refresh_token_reused", "refresh_token_invalidated", "invalid_grant"}
)


@dataclasses.dataclass(frozen=True, slots=True)
class TokenGrant:
    """One token-endpoint response, from an exchange or a refresh.

    Attributes:
        access_token: The bearer.
        refresh_token: The rotating refresh token; "" when the response omitted it.
        expires_in: The bearer's lifetime in seconds.
        id_token: The identity JWT; "" when absent.
    """

    access_token: str
    refresh_token: str
    expires_in: float
    id_token: str = ""


def pkce_challenge(verifier: str) -> str:
    """Return the RFC 7636 S256 challenge for a verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def pkce_pair() -> tuple[str, str]:
    """Return a fresh RFC 7636 `(code_verifier, code_challenge)` pair."""
    verifier = secrets.token_urlsafe(64)
    return verifier, pkce_challenge(verifier)


def parse_callback(pasted: str, *, state: str) -> str:
    """Read the authorization code off the callback URL the browser landed on.

    Args:
        pasted: The full callback URL, or just its query string.
        state: The state the sign-in started with.

    Returns:
        The code.

    Raises:
        ValueError: A state mismatch, a refusal carried in the `error` parameter,
            or no code; the message names which.
    """
    text = pasted.strip()
    query = parse.urlsplit(text).query if "?" in text else text
    params = dict(parse.parse_qsl(query, keep_blank_values=True))
    # The state is checked first, so a callback this sign-in did not start gets nothing reflected.
    if params.get("state", "") != state:
        raise ValueError("state mismatch: this callback is not from the sign-in agent6 started")
    if params.get("error"):
        detail = params.get("error_description") or params["error"]
        raise ValueError(f"sign-in was refused: {detail}")
    code = params.get("code", "")
    if not code:
        raise ValueError("no `code` parameter found; paste the full URL the browser landed on")
    return code


def post_form(url: str, data: dict[str, str], timeout_s: float) -> httpx2.Response:
    """Return a token endpoint's response to a form POST; the seam tests stub."""
    return httpx2.post(
        url,
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "accept": "application/json",
        },
        content=parse.urlencode(data).encode("ascii"),
        timeout=timeout_s,
    )


def post_json(url: str, data: dict[str, str], timeout_s: float) -> httpx2.Response:
    """Return an endpoint's response to a JSON POST; the seam tests stub."""
    return httpx2.post(url, json=data, headers={"accept": "application/json"}, timeout=timeout_s)


def grant_from_response(resp: httpx2.Response, *, operation: str, label: str) -> TokenGrant:
    """Parse a 2xx token response.

    Args:
        resp: The response.
        operation: "exchange" or "refresh", for the error.
        label: The sign-in the error names ("ChatGPT", a provider name).

    Returns:
        The grant.

    Raises:
        ProviderError: The body is not JSON or a field is unusable.
    """  # noqa: DOC501  # the ValueError is raised and caught in the same try
    try:
        data: Any = resp.json()
    except ValueError as exc:
        raise types.ProviderError(f"{label} token {operation} returned a non-JSON body") from exc
    access = data.get("access_token") if isinstance(data, dict) else None
    if not isinstance(access, str) or not access:
        raise types.ProviderError(f"{label} token {operation} response carried no access_token")
    refresh = data.get("refresh_token")
    if refresh is None:
        refresh = ""
    if not isinstance(refresh, str):
        raise types.ProviderError(
            f"{label} token {operation} response carried an unusable refresh_token"
        )
    identity = data.get("id_token")
    if identity is None:
        identity = ""
    if not isinstance(identity, str):
        raise types.ProviderError(
            f"{label} token {operation} response carried an unusable id_token"
        )
    expires = data.get("expires_in", 3600.0)
    try:
        expires_in = float(expires)
        if isinstance(expires, bool) or not math.isfinite(expires_in) or expires_in <= 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise types.ProviderError(
            f"{label} token {operation} response carried an unusable expires_in"
        ) from exc
    return TokenGrant(
        access_token=access,
        refresh_token=refresh,
        expires_in=expires_in,
        id_token=identity,
    )


def scrub(text: str, secrets: tuple[str, ...]) -> str:
    """Return the text with every echoed credential value replaced.

    Raw and JSON-escaped spellings are covered; values under 8 characters are skipped.

    Args:
        text: A response body.
        secrets: The credential values the request sent.

    Returns:
        The text, each value replaced by `<REDACTED>`.
    """
    for value in secrets:
        if len(value) < 8:
            continue
        for spelling in {value, json.dumps(value)[1:-1]}:
            text = text.replace(spelling, "<REDACTED>")
    return text


def token_error(
    resp: httpx2.Response,
    *,
    operation: str,
    provider: str,
    label: str,
    secrets: tuple[str, ...] = (),
) -> types.ProviderError:
    """Classify a non-2xx token response.

    Args:
        resp: The response.
        operation: "exchange" or "refresh", for the error.
        provider: The provider name, for the reconnect hint.
        label: The sign-in the error names.
        secrets: The request's credential values, scrubbed from an echoed body.

    Returns:
        A 401 naming the reconnect for a dead refresh token; otherwise an error
        carrying the response's status for the retry policy.
    """
    body = scrub(resp.text[:2000], secrets)
    code = ""
    try:
        err = json.loads(body).get("error")
        code = str(err.get("code") if isinstance(err, dict) else err or "")
    except (ValueError, AttributeError):
        pass
    if resp.status_code == 401 or code in PERMANENT_REFRESH_CODES:
        return types.ProviderError(
            f"{label} sign-in is no longer valid ({code or f'HTTP {resp.status_code}'});"
            f" run `agent6 connect {provider}` to sign in again.",
            status_code=401,
        )
    return types.ProviderError(
        f"{label} token {operation} failed: HTTP {resp.status_code}: {body[:300]}",
        status_code=resp.status_code,
    )


def exchange_code(
    token_endpoint: str,
    client_id: str,
    *,
    code: str,
    verifier: str,
    redirect_uri: str,
    provider: str,
    label: str,
    timeout_s: float = TOKEN_TIMEOUT_S,
) -> TokenGrant:
    """Exchange an authorization code for a token grant.

    Args:
        token_endpoint: The token endpoint.
        client_id: The client registration.
        code: The authorization code.
        verifier: The PKCE verifier the code was minted against.
        redirect_uri: The redirect of the flow that minted the code.
        provider: The provider name, for the reconnect hint.
        label: The sign-in the errors name.
        timeout_s: The request timeout.

    Returns:
        The grant.

    Raises:
        ProviderError: The endpoint was unreachable or refused the exchange.
    """  # noqa: DOC501  # `token_error` builds the ProviderError named above
    try:
        resp = post_form(
            token_endpoint,
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "code_verifier": verifier,
            },
            timeout_s,
        )
    except httpx2.HTTPError as exc:
        raise types.ProviderError(f"could not reach {token_endpoint}: {exc}") from exc
    if resp.status_code >= 400:
        raise token_error(
            resp, operation="exchange", provider=provider, label=label, secrets=(code, verifier)
        )
    return grant_from_response(resp, operation="exchange", label=label)


def refresh_grant(
    token_endpoint: str,
    client_id: str,
    refresh_token: str,
    *,
    provider: str,
    label: str,
    timeout_s: float = TOKEN_TIMEOUT_S,
) -> TokenGrant:
    """Trade a refresh token for a fresh grant; tokens may rotate.

    Args:
        token_endpoint: The token endpoint.
        client_id: The client registration.
        refresh_token: The refresh token.
        provider: The provider name, for the reconnect hint.
        label: The sign-in the errors name.
        timeout_s: The request timeout.

    Returns:
        The grant.

    Raises:
        ProviderError: The endpoint was unreachable or refused.
    """  # noqa: DOC501  # `token_error` builds the ProviderError named above
    try:
        resp = post_form(
            token_endpoint,
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            },
            timeout_s,
        )
    except httpx2.HTTPError as exc:
        raise types.ProviderError(f"could not reach {token_endpoint}: {exc}") from exc
    if resp.status_code >= 400:
        raise token_error(
            resp, operation="refresh", provider=provider, label=label, secrets=(refresh_token,)
        )
    return grant_from_response(resp, operation="refresh", label=label)


def revoke_tokens(
    revocation_endpoint: str,
    client_id: str,
    tokens: secret_store.OAuthTokens,
    *,
    timeout_s: float = TOKEN_TIMEOUT_S,
) -> str | None:
    """Revoke the grant at sign-out, best effort.

    The refresh token kills the whole grant; the access token is the fallback.

    Args:
        revocation_endpoint: The revocation endpoint.
        client_id: The client registration.
        tokens: The stored tokens.
        timeout_s: The request timeout.

    Returns:
        None on success, else the error's description; the caller removes the
        local tokens either way.
    """
    token, hint = (
        (tokens.refresh_token, "refresh_token")
        if tokens.refresh_token
        else (tokens.access_token, "access_token")
    )
    body: dict[str, str] = {"token": token, "token_type_hint": hint}
    if hint == "refresh_token":
        body["client_id"] = client_id
    try:
        resp = httpx2.post(revocation_endpoint, json=body, timeout=timeout_s)
    except httpx2.HTTPError as exc:
        return f"could not reach {revocation_endpoint}: {exc}"
    if resp.status_code >= 400:
        return f"HTTP {resp.status_code}: {scrub(resp.text, (token,))[:200]}"
    return None


def error_code_of(resp: httpx2.Response) -> str:
    """Return the error code a response body carries; "" when it has none."""
    try:
        err = resp.json().get("error")
    except (ValueError, AttributeError):
        return ""
    return str(err.get("code") if isinstance(err, dict) else err or "")


def jwt_claims(token: str) -> dict[str, Any]:
    """Return a JWT's payload claims; `{}` on any malformation.

    No signature check: agent6 is the OAuth client, not a verifier, and the
    claims are only read back for an account id or a plan type.

    Args:
        token: An access token or id_token.

    Returns:
        The decoded payload.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        decoded: Any = json.loads(
            base64.b64decode(payload.encode("ascii"), altchars=b"-_", validate=True)
        )
    except (ValueError, UnicodeDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}
