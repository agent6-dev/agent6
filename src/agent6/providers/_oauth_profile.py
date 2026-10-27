# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Resolved OAuth settings and the policies a credential holder preserves."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Literal

from agent6.config import _oauth as oauth_config
from agent6.providers import _oauth_protocol


@dataclasses.dataclass(frozen=True, slots=True)
class OAuthProfile:
    """An issuer's effective settings, independent of public configuration validation.

    Attributes:
        issuer: The issuer identifier; never discovered.
        client_id: The public client registration.
        scopes: The scopes requested.
        authorization_endpoint: The browser authorization URL, when supported.
        device_authorization_endpoint: The device-start URL, when supported.
        token_endpoint: The authorization-code exchange and refresh URL.
        revocation_endpoint: The revocation URL, when supported.
        redirect_uri: The registered browser callback URI.
        callback_host: The literal loopback address the listener binds.
        authorization_params: Static authorization parameters, sorted by name.
        request_encoding: The token request encoding.
        revocation_encoding: The revocation request encoding.
        device_flow: The device wire protocol.
        device_poll_endpoint: The device polling URL, when supported.
        device_redirect_uri: The redirect paired with proprietary device codes.
        refresh_skew_s: The margin before expiry.
        exchange: The optional exchange for a model endpoint's bearer.
        label: The sign-in label; None uses the provider key.
        account_id_of: Extracts an account before identity tokens are discarded.
        require_refresh_token: Refuses an initial grant without a nonblank refresh token.
        default_expires_in: The lifetime when the issuer omits expiry; None is indefinite.
        cap_refresh_skew: Caps the margin at half the issued lifetime.
    """

    issuer: str
    client_id: str
    scopes: tuple[str, ...]
    authorization_endpoint: str | None
    device_authorization_endpoint: str | None
    token_endpoint: str
    revocation_endpoint: str | None
    redirect_uri: str
    callback_host: str
    authorization_params: tuple[tuple[str, str], ...]
    request_encoding: Literal["form", "json"]
    revocation_encoding: Literal["form", "json"]
    device_flow: Literal["rfc8628", "chatgpt"]
    device_poll_endpoint: str | None
    device_redirect_uri: str | None
    refresh_skew_s: float
    exchange: oauth_config.OAuthExchangeConfig | None = None
    label: str | None = None
    account_id_of: Callable[[_oauth_protocol.TokenGrant], str] | None = None
    require_refresh_token: bool = False
    default_expires_in: float | None = None
    cap_refresh_skew: bool = True
