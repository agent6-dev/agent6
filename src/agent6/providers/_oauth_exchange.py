# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Exchange an issuer bearer for the model endpoint's token."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any
from urllib import parse

import httpx2

from agent6.config import _oauth as oauth_config
from agent6.providers import _oauth_protocol, _transport, types


def _expiry_of(
    data: dict[str, Any], exchange: oauth_config.OAuthExchangeConfig, *, now: float
) -> float:
    """Return the exchanged token's expiry as Unix time.

    Args:
        data: The exchange's JSON answer.
        exchange: The configured exchange, naming the field and its format.
        now: The current Unix time.

    Returns:
        The expiry.

    Raises:
        ProviderError: The field is missing, not a number, or already past.
    """
    raw = data.get(exchange.expires_field)
    value = None
    if isinstance(raw, (int, float, str)) and not isinstance(raw, bool):
        try:
            value = float(raw)
        except (ValueError, OverflowError):
            value = None
    if value is None or not math.isfinite(value):
        raise types.ProviderError(
            f"credential exchange answered without a usable `{exchange.expires_field}`"
        )
    expires_at = value if exchange.expires_format == "unix_seconds" else now + value
    if not (expires_at > now):
        raise types.ProviderError("credential exchange answered with a token already expired")
    return expires_at


def run_exchange(
    exchange: oauth_config.OAuthExchangeConfig,
    issuer_token: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = _oauth_protocol.TOKEN_TIMEOUT_S,
    now: Callable[[], float] = time.time,
) -> tuple[str, float]:
    """Trade the issuer's bearer for the model endpoint's token.

    Args:
        exchange: The configured exchange.
        issuer_token: The issuer's access token, sent as the bearer.
        headers: Extra headers addressed to the exchange.
        timeout_s: The request timeout.
        now: The clock.

    Returns:
        The exchanged token and its expiry as Unix time.

    Raises:
        ProviderError: The endpoint was unreachable, redirected, refused (a 401 keeps its
            status so the caller can refresh the issuer grant once), or answered without a
            usable token or expiry.
    """
    sent = {
        **(headers or {}),
        "authorization": f"Bearer {issuer_token}",
        "accept": "application/json",
    }
    is_get = exchange.method == "GET"
    as_json = not is_get and exchange.request_encoding == "json"
    form = None if is_get or as_json else parse.urlencode(exchange.parameters).encode("ascii")
    if form is not None:
        sent["content-type"] = "application/x-www-form-urlencoded"
    try:
        resp = _transport.http_request(
            exchange.method,
            exchange.endpoint,
            headers=sent,
            params=(exchange.parameters or None) if is_get else None,
            json=exchange.parameters if as_json else None,
            content=form,
            timeout=timeout_s,
            cap=_transport.CREDENTIAL_BODY_BYTES,
        )
    except httpx2.HTTPError as exc:
        raise types.ProviderError(
            f"could not reach {exchange.endpoint}:"
            f" {_oauth_protocol.scrub(str(exc), (issuer_token,))}"
        ) from exc
    if 300 <= resp.status_code < 400:
        raise types.ProviderError(
            f"credential exchange redirected (HTTP {resp.status_code}); refusing to follow"
            " with credentials",
            status_code=resp.status_code,
        )
    if resp.status_code >= 400:
        raise types.ProviderError(
            f"credential exchange failed: HTTP {resp.status_code}: "
            f"{_oauth_protocol.scrub(resp.text, (issuer_token,))[:200]}",
            status_code=resp.status_code,
        )
    try:
        data: Any = resp.json()
    except ValueError as exc:
        raise types.ProviderError("credential exchange returned a non-JSON body") from exc
    token = data.get(exchange.access_token_field) if isinstance(data, dict) else None
    if not isinstance(token, str) or not _oauth_protocol.header_safe(token):
        raise types.ProviderError(
            f"credential exchange answered without a usable `{exchange.access_token_field}`"
        )
    return token, _expiry_of(data, exchange, now=now())
