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
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib import parse

import httpx2

from agent6.providers import _transport, types

TOKEN_TIMEOUT_S = 30.0
# The refresh token itself is dead; re-consent is the only repair.
PERMANENT_REFRESH_CODES = frozenset(
    {"refresh_token_expired", "refresh_token_reused", "refresh_token_invalidated", "invalid_grant"}
)


class OAuthError(types.ProviderError):
    """A token endpoint refusal with an allowlisted refresh-error code.

    Attributes:
        code: One of PERMANENT_REFRESH_CODES, or None for an unclassified refusal.
    """

    def __init__(self, message: str, *, status_code: int, code: str | None = None) -> None:
        super().__init__(message, status_code=status_code)
        self.code = code


@dataclasses.dataclass(frozen=True, slots=True)
class TokenGrant:
    """One token-endpoint response, from an exchange or a refresh.

    Attributes:
        access_token: The bearer.
        refresh_token: The rotating refresh token; "" when the response omitted it.
        expires_in: The bearer's lifetime in seconds; None when the issuer named none.
        id_token: The identity JWT; "" when absent.
    """

    access_token: str
    refresh_token: str
    expires_in: float | None
    id_token: str = ""


def pkce_challenge(verifier: str) -> str:
    """Return the RFC 7636 S256 challenge for a verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def pkce_pair() -> tuple[str, str]:
    """Return a fresh RFC 7636 `(code_verifier, code_challenge)` pair."""
    verifier = secrets.token_urlsafe(64)
    return verifier, pkce_challenge(verifier)


def parse_callback(pasted: str, *, state: str, redirect_uri: str = "") -> str:
    """Read the authorization code off the callback URL the browser landed on.

    Args:
        pasted: The full callback URL, or just its query string.
        state: The state the sign-in started with.
        redirect_uri: The redirect the sign-in registered; a full URL must match its
            scheme, host, port and path. "" accepts any origin (a query-only paste).

    Returns:
        The code.

    Raises:
        ValueError: Another origin or path, more or fewer than one state, a state mismatch,
            a refusal carried in the `error` parameter, more or fewer than one code, or a
            code beside an error; the message names which.
    """
    text = pasted.strip()
    if "?" in text or text.startswith(("http://", "https://")):
        parts = parse.urlsplit(text)
        if redirect_uri:
            want = parse.urlsplit(redirect_uri)
            same = (
                parts.scheme == want.scheme
                and (parts.hostname or "").lower() == (want.hostname or "").lower()
                and parts.port == want.port
                and parts.path == want.path
            )
            if not same:
                raise ValueError(
                    "this URL is not the sign-in's redirect; paste the URL the browser landed on"
                )
        query = parts.query
    else:
        query = text
    pairs = parse.parse_qsl(query, keep_blank_values=True)
    states = [v for k, v in pairs if k == "state"]
    codes = [v for k, v in pairs if k == "code"]
    errors = [v for k, v in pairs if k == "error"]
    # The state is checked first, so a callback this sign-in did not start gets nothing reflected.
    if len(states) > 1:
        raise ValueError("the callback must carry exactly one state")
    if not states or states[0] != state:
        raise ValueError("state mismatch: this callback is not from the sign-in agent6 started")
    if codes and errors:
        raise ValueError("the callback carries both a code and an error; it must carry one of them")
    if errors:
        descriptions = [v for k, v in pairs if k == "error_description"]
        detail = printable((descriptions[0] if descriptions else "") or errors[0])
        raise ValueError(f"sign-in was refused: {detail}")
    if len(codes) > 1:
        raise ValueError("the callback must carry exactly one code")
    if not codes:
        raise ValueError("no `code` parameter found; paste the full URL the browser landed on")
    return codes[0]


def post_form(
    url: str, data: dict[str, str], timeout_s: float, headers: Mapping[str, str] | None = None
) -> httpx2.Response:
    """Return a token endpoint's response to a form POST; the seam tests stub."""
    return _transport.http_request(
        "POST",
        url,
        headers={
            **(headers or {}),
            "content-type": "application/x-www-form-urlencoded",
            "accept": "application/json",
        },
        content=parse.urlencode(data).encode("ascii"),
        timeout=timeout_s,
        cap=_transport.CREDENTIAL_BODY_BYTES,
    )


def post_json(
    url: str, data: dict[str, str], timeout_s: float, headers: Mapping[str, str] | None = None
) -> httpx2.Response:
    """Return an endpoint's response to a JSON POST; the seam tests stub."""
    return _transport.http_request(
        "POST",
        url,
        headers={**(headers or {}), "accept": "application/json"},
        json=dict(data),
        timeout=timeout_s,
        cap=_transport.CREDENTIAL_BODY_BYTES,
    )


HeadersFactory = Callable[[], Mapping[str, str]]


def _post(
    url: str,
    data: dict[str, str],
    *,
    encoding: str,
    timeout_s: float,
    headers: HeadersFactory | None = None,
) -> httpx2.Response:
    """Send a device or token request in the configured encoding.

    Args:
        url: The endpoint.
        data: The parameters.
        encoding: "form" or "json".
        timeout_s: The request timeout.
        headers: Renders the operator's headers addressed to this request; None sends none.

    Returns:
        The response.
    """
    poster = post_json if encoding == "json" else post_form
    return poster(url, data, timeout_s, headers=dict(headers()) if headers is not None else {})


def authorization_url(
    endpoint: str,
    client_id: str,
    *,
    redirect_uri: str,
    scopes: tuple[str, ...],
    state: str,
    challenge: str,
    extra: dict[str, str] | None = None,
) -> str:
    """Build the browser URL that starts an authorization-code sign-in with PKCE.

    Args:
        endpoint: The authorization endpoint.
        client_id: The client registration.
        redirect_uri: The loopback redirect the code returns to.
        scopes: The scopes requested; empty sends no `scope`.
        state: The state the callback must echo.
        challenge: The PKCE challenge.
        extra: Static parameters sent after the protocol's own.

    Returns:
        The URL, the endpoint's own query kept ahead of the sign-in's.
    """
    params = [("response_type", "code"), ("client_id", client_id), ("redirect_uri", redirect_uri)]
    if scopes:
        params.append(("scope", " ".join(scopes)))
    params += [("code_challenge", challenge), ("code_challenge_method", "S256"), ("state", state)]
    params += list((extra or {}).items())
    joiner = "&" if parse.urlsplit(endpoint).query else "?"
    return f"{endpoint}{joiner}{parse.urlencode(params)}"


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
    if not header_safe(access):
        raise types.ProviderError(
            f"{label} token {operation} response carried an access_token that cannot ride a"
            " header (a control character or a non-ASCII byte; the value is not shown)"
        )
    refresh = data.get("refresh_token")
    if refresh is None:
        refresh = ""
    if not isinstance(refresh, str) or not header_safe(refresh or "x"):
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
    expires = data.get("expires_in")
    expires_in: float | None = None
    if expires is not None:
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


def header_safe(token: str) -> bool:
    """Return whether a token can ride an HTTP header value: visible ASCII, no space."""
    return bool(token) and all(0x21 <= ord(ch) <= 0x7E for ch in token)


def scrub(text: str, secrets: tuple[str, ...]) -> str:
    """Return the text with every echoed credential value replaced.

    Raw and JSON-escaped spellings are covered, including short device codes.

    Args:
        text: A response body.
        secrets: The credential values the request sent.

    Returns:
        The text, each value replaced by `<REDACTED>`.
    """
    for value in secrets:
        if not value:
            continue
        for spelling in {value, json.dumps(value)[1:-1]}:
            text = text.replace(spelling, "<REDACTED>")
    return text


def redirect_refusal(resp: httpx2.Response, *, operation: str, label: str) -> types.ProviderError:
    """Return the error for a token endpoint that answered with a redirect, never followed."""
    return types.ProviderError(
        f"{label} token {operation} redirected (HTTP {resp.status_code}); refusing to follow"
        " with credentials (the Location is not shown: it may echo them)",
        status_code=resp.status_code,
    )


def printable(text: str, limit: int = 300) -> str:
    """Return remote text fit for a terminal: printable characters only, clipped."""
    return "".join(ch for ch in text if ch.isprintable())[:limit]


def token_error(
    resp: httpx2.Response,
    *,
    operation: str,
    provider: str,
    label: str,
    secrets: tuple[str, ...] = (),
) -> OAuthError:
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
    body = scrub(resp.text, secrets)[:2000]
    raw_code = error_code_of(resp)
    code = raw_code if raw_code in PERMANENT_REFRESH_CODES else None
    detail = printable(scrub(raw_code, secrets), 80)
    if resp.status_code == 401 or code is not None:
        return OAuthError(
            f"{label} sign-in is no longer valid ({detail or f'HTTP {resp.status_code}'});"
            f" run `agent6 connect {provider}` to sign in again.",
            status_code=401,
            code=code,
        )
    return OAuthError(
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
    encoding: str = "form",
    headers: HeadersFactory | None = None,
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
        encoding: "form" or "json".
        headers: Renders the operator's headers addressed to this request; None sends none.
        timeout_s: The request timeout.

    Returns:
        The grant.

    Raises:
        ProviderError: The endpoint was unreachable or refused the exchange.
    """  # noqa: DOC501  # `token_error` builds the ProviderError named above
    try:
        resp = _post(
            token_endpoint,
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "code_verifier": verifier,
            },
            encoding=encoding,
            timeout_s=timeout_s,
            headers=headers,
        )
    except httpx2.HTTPError as exc:
        raise types.ProviderError(
            f"could not reach {token_endpoint}: {scrub(str(exc), (code, verifier))}"
        ) from exc
    if 300 <= resp.status_code < 400:
        raise redirect_refusal(resp, operation="exchange", label=label)
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
    encoding: str = "form",
    headers: HeadersFactory | None = None,
    timeout_s: float = TOKEN_TIMEOUT_S,
) -> TokenGrant:
    """Trade a refresh token for a fresh grant; tokens may rotate.

    Args:
        token_endpoint: The token endpoint.
        client_id: The client registration.
        refresh_token: The refresh token.
        provider: The provider name, for the reconnect hint.
        label: The sign-in the errors name.
        encoding: "form" or "json".
        headers: Renders the operator's headers addressed to this request; None sends none.
        timeout_s: The request timeout.

    Returns:
        The grant.

    Raises:
        ProviderError: The endpoint was unreachable or refused.
    """  # noqa: DOC501  # `token_error` builds the ProviderError named above
    try:
        resp = _post(
            token_endpoint,
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            },
            encoding=encoding,
            timeout_s=timeout_s,
            headers=headers,
        )
    except httpx2.HTTPError as exc:
        raise types.ProviderError(
            f"could not reach {token_endpoint}: {scrub(str(exc), (refresh_token,))}"
        ) from exc
    if 300 <= resp.status_code < 400:
        raise redirect_refusal(resp, operation="refresh", label=label)
    if resp.status_code >= 400:
        raise token_error(
            resp, operation="refresh", provider=provider, label=label, secrets=(refresh_token,)
        )
    return grant_from_response(resp, operation="refresh", label=label)


def revoke_tokens(
    revocation_endpoint: str,
    client_id: str,
    *,
    access_token: str,
    refresh_token: str,
    encoding: str,
    headers: HeadersFactory | None = None,
    timeout_s: float = TOKEN_TIMEOUT_S,
) -> str | None:
    """Revoke the grant at sign-out, best effort.

    The refresh token kills the whole grant; the access token is the fallback.

    Args:
        revocation_endpoint: The revocation endpoint.
        client_id: The client registration.
        access_token: The stored access token.
        refresh_token: The stored refresh token; "" when the grant has none.
        encoding: "form" (RFC 7009) or "json".
        headers: Renders the operator's headers addressed to this request; None sends none.
        timeout_s: The request timeout.

    Returns:
        None on success, else the error's description; the caller removes the
        local tokens either way.
    """
    token, hint = (
        (refresh_token, "refresh_token") if refresh_token else (access_token, "access_token")
    )
    body: dict[str, str] = {"token": token, "token_type_hint": hint}
    if hint == "refresh_token":
        body["client_id"] = client_id
    try:
        resp = _post(
            revocation_endpoint, body, encoding=encoding, timeout_s=timeout_s, headers=headers
        )
    except (httpx2.HTTPError, types.ProviderError) as exc:
        return f"could not reach {revocation_endpoint}: {scrub(str(exc), (token,))}"
    if 300 <= resp.status_code < 400:
        return f"redirected (HTTP {resp.status_code}); not followed, so nothing was revoked"
    if resp.status_code >= 400:
        return f"HTTP {resp.status_code}: {scrub(resp.text, (token,))[:200]}"
    return None


# A device grant waits this long at most, whatever expiry the issuer reports.
DEVICE_CEILING_S = 15 * 60.0
# One poll waits this long at most, whatever interval the issuer asks for or adds.
POLL_INTERVAL_CEILING_S = 60.0


@dataclasses.dataclass(frozen=True, slots=True)
class DeviceGrant:
    """A started device-authorization grant (RFC 8628).

    Attributes:
        device_code: The code the poll sends; never shown.
        user_code: The code the person types at the verification URI.
        verification_uri: Where the person types it.
        verification_uri_complete: The same with the code filled in; "" when absent.
        interval_s: The polling interval the issuer asked for.
        expires_at: When the codes die, as a monotonic deadline.
    """

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    interval_s: float
    expires_at: float


def start_device_grant(
    endpoint: str,
    client_id: str,
    *,
    scopes: tuple[str, ...],
    encoding: str,
    label: str,
    headers: HeadersFactory | None = None,
    timeout_s: float = TOKEN_TIMEOUT_S,
    now: Callable[[], float] = time.monotonic,
) -> DeviceGrant:
    """Begin the device grant: ask the issuer for a user code.

    Args:
        endpoint: The device authorization endpoint.
        client_id: The client registration.
        scopes: The scopes requested; empty sends no `scope`.
        encoding: "form" or "json".
        headers: Renders the operator's headers addressed to this request; None sends none.
        label: The sign-in the errors name.
        timeout_s: The request timeout.
        now: The monotonic clock.

    Returns:
        The started grant.

    Raises:
        ProviderError: The endpoint was unreachable, redirected, refused, or answered malformed.
    """  # noqa: DOC501  # `redirect_refusal` builds the ProviderError named above
    body = {"client_id": client_id}
    if scopes:
        body["scope"] = " ".join(scopes)
    try:
        resp = _post(endpoint, body, encoding=encoding, timeout_s=timeout_s, headers=headers)
    except httpx2.HTTPError as exc:
        raise types.ProviderError(f"could not reach {endpoint}: {exc}") from exc
    if 300 <= resp.status_code < 400:
        raise redirect_refusal(resp, operation="device start", label=label)
    if resp.status_code >= 400:
        raise types.ProviderError(
            f"{label} device sign-in refused: HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        data: Any = resp.json()
        uri = str(data["verification_uri"])
        complete = str(data.get("verification_uri_complete") or "")
        if parse.urlsplit(complete).scheme != "https":
            complete = ""
        grant = DeviceGrant(
            device_code=str(data["device_code"]),
            user_code=str(data["user_code"]),
            verification_uri=uri,
            verification_uri_complete=complete,
            interval_s=min(max(1.0, float(data.get("interval") or 5.0)), POLL_INTERVAL_CEILING_S),
            expires_at=now() + min(float(data.get("expires_in") or 600.0), DEVICE_CEILING_S),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise types.ProviderError(f"{label} device sign-in response was malformed") from exc
    uris = (grant.verification_uri, grant.verification_uri_complete)
    if parse.urlsplit(grant.verification_uri).scheme != "https" or any(
        u != printable(u, 2048) for u in uris
    ):
        raise types.ProviderError(f"{label} device sign-in named an unusable verification URI")
    if not grant.user_code.isprintable() or len(grant.user_code) > 64:
        raise types.ProviderError(f"{label} device sign-in response carried an unusable user code")
    return grant


def poll_device_grant(
    token_endpoint: str,
    client_id: str,
    grant: DeviceGrant,
    *,
    encoding: str,
    label: str,
    headers: HeadersFactory | None = None,
    timeout_s: float = TOKEN_TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> TokenGrant:
    """Poll the token endpoint until the person approves, refuses, or the codes expire.

    `authorization_pending` waits the interval, `slow_down` adds five seconds to it; the
    issuer may answer either as a 4xx or as a 200 whose body carries the error.

    Args:
        token_endpoint: The token endpoint.
        client_id: The client registration.
        grant: The started grant.
        encoding: "form" or "json".
        headers: Renders the operator's headers addressed to this request; None sends none.
        label: The sign-in the errors name.
        timeout_s: The request timeout.
        sleep: The wait between polls.
        now: The monotonic clock.

    Returns:
        The token grant.

    Raises:
        ProviderError: The endpoint was unreachable or redirected, the person refused, the
            codes expired, or the issuer answered an unknown error.
    """  # noqa: DOC501  # `redirect_refusal` builds the ProviderError named above
    interval = grant.interval_s
    while now() < grant.expires_at:
        body = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": grant.device_code,
            "client_id": client_id,
        }
        try:
            resp = _post(
                token_endpoint, body, encoding=encoding, timeout_s=timeout_s, headers=headers
            )
        except httpx2.HTTPError as exc:
            raise types.ProviderError(
                f"could not reach {token_endpoint}: {scrub(str(exc), (grant.device_code,))}"
            ) from exc
        if 300 <= resp.status_code < 400:
            raise redirect_refusal(resp, operation="device poll", label=label)
        code = error_code_of(resp)
        if resp.status_code < 400 and not code:
            return grant_from_response(resp, operation="device", label=label)
        if code == "slow_down":
            interval = min(interval + 5.0, POLL_INTERVAL_CEILING_S)
        if code in ("authorization_pending", "slow_down"):
            left = grant.expires_at - now()
            if left <= 0:
                break
            sleep(min(interval, left))
            continue
        if code == "access_denied":
            raise types.ProviderError(f"{label} sign-in was refused at the issuer")
        if code == "expired_token":
            raise types.ProviderError(f"{label} device code expired before it was entered")
        raise types.ProviderError(
            f"{label} device sign-in failed: HTTP {resp.status_code}: "
            f"{scrub(resp.text, (grant.device_code,))[:200]}"
        )
    raise types.ProviderError(f"{label} device code expired before it was entered")


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
