# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[providers.<name>].oauth` model: a configured issuer, its endpoints, one second exchange.

Every endpoint here is a credential destination the operator named by hand: HTTPS, or plain
HTTP on a literal loopback address for a local service or a test. There is no discovery, no
client secret and no redirect following; `agent6 connect <name>` dials exactly these URLs.
"""

from __future__ import annotations

from typing import Literal
from urllib import parse

import pydantic

from agent6.config import _base

# Parameters the sign-in itself sets; an operator-supplied value for one is a refusal.
RESERVED_AUTHORIZATION_PARAMS = frozenset(
    {
        "client_id",
        "client_secret",
        "code",
        "code_challenge",
        "code_challenge_method",
        "code_verifier",
        "device_code",
        "grant_type",
        "redirect_uri",
        "refresh_token",
        "response_type",
        "scope",
        "state",
    }
)
DEFAULT_REDIRECT_URI = "http://127.0.0.1:0/auth/callback"
_LOOPBACK_LITERALS = frozenset({"127.0.0.1", "::1"})


def credential_url_refusal(url: str, *, field: str) -> str:
    """Return why a URL cannot be a credential destination, or "".

    Args:
        url: The configured URL.
        field: The field name for the message.

    Returns:
        The refusal: not http(s), no host, userinfo, a fragment, or plain http off a literal
        loopback address; "" when the URL is acceptable.
    """
    try:
        parts = parse.urlsplit(url)
        port = parts.port
    except ValueError as exc:
        return f"{field} {url!r}: {exc}"
    checks = (
        (
            parts.scheme not in ("http", "https"),
            "must start with https:// (http:// only on a loopback address)",
        ),
        (not parts.hostname, "has no host"),
        (port is not None and not (1 <= port <= 65535), "has an invalid port"),
        (
            parts.username is not None or parts.password is not None,
            "carries userinfo; credentials never ride a URL",
        ),
        (bool(parts.fragment), "carries a fragment, which no request can send"),
        (
            parts.scheme == "http" and parts.hostname not in _LOOPBACK_LITERALS,
            "is plain http off a loopback address; a credential would cross the wire"
            " readable (https, or http://127.0.0.1 for a local service)",
        ),
    )
    return next((f"{field} {url!r} {why}" for failed, why in checks if failed), "")


def _require_credential_url(url: str, field: str) -> str:
    """Return the URL unchanged.

    Args:
        url: The configured URL.
        field: The field name for the message.

    Returns:
        The URL.

    Raises:
        ValueError: The refusal `credential_url_refusal` names.
    """
    refusal = credential_url_refusal(url, field=field)
    if refusal:
        raise ValueError(refusal)
    return url


class OAuthExchangeConfig(pydantic.BaseModel):
    """One bounded second exchange: the issuer's bearer buys the token the model endpoint takes.

    The request is one GET or POST authenticated by the issuer's access token; the answer's
    named top-level fields are the new token and its expiry. Nothing in the answer can change
    a destination, a header or a command.
    """

    model_config = _base.MODEL_CONFIG

    endpoint: str = pydantic.Field(
        description="the exact URL the issuer's bearer is sent to for the model endpoint's token",
    )
    method: Literal["GET", "POST"] = pydantic.Field(
        default="POST", description="the request method; `GET` sends `parameters` as the query"
    )
    request_encoding: Literal["form", "json"] = pydantic.Field(
        default="form", description="how a `POST` body is encoded"
    )
    parameters: dict[str, str] = pydantic.Field(
        default_factory=dict,
        description="static, non-secret request parameters (query for `GET`, body for `POST`)",
    )
    access_token_field: str = pydantic.Field(
        default="access_token",
        min_length=1,
        description="the top-level JSON field holding the token the model endpoint takes",
    )
    expires_field: str = pydantic.Field(
        default="expires_in",
        min_length=1,
        description="the top-level JSON field holding that token's expiry; required in the answer",
    )
    expires_format: Literal["relative_seconds", "unix_seconds"] = pydantic.Field(
        default="relative_seconds",
        description="whether `expires_field` is a lifetime in seconds or a Unix time",
    )

    @pydantic.field_validator("endpoint")
    @classmethod
    def _endpoint_is_a_credential_url(cls, v: str) -> str:
        """Refuse an endpoint that cannot carry a credential safely.

        Args:
            v: The URL.

        Returns:
            The URL unchanged.

        Raises:
            ValueError: The URL fails `credential_url_refusal`.
        """
        return _require_credential_url(v, "oauth.exchange.endpoint")


class OAuthConfig(pydantic.BaseModel):
    """A configured OAuth issuer and public-client registration for a provider's sign-in.

    `agent6 connect <name>` runs authorization code with PKCE against `authorization_endpoint`
    (a loopback redirect or a pasted callback), or the standard device grant against
    `device_authorization_endpoint`; `token_endpoint` serves the exchange, the device polling
    and the refresh. The stored grant is bound to these settings and to the provider's
    `base_url`: a change to any of them needs a new `agent6 connect <name>` before the stored
    credential is sent anywhere.
    """

    model_config = _base.MODEL_CONFIG

    issuer: str = pydantic.Field(
        description=(
            "the issuer identifier (https, or http on a loopback literal for a local service);"
            " names the sign-in and binds the stored grant; never fetched (no discovery)"
        ),
    )
    client_id: str = pydantic.Field(
        min_length=1, description="the public-client registration the operator made at the issuer"
    )
    scopes: _base.StrTuple = pydantic.Field(
        default=(), description="the scopes requested, joined with spaces; none are implied"
    )
    authorization_endpoint: str | None = pydantic.Field(
        default=None,
        description="enables authorization code with PKCE: the URL the browser opens",
    )
    device_authorization_endpoint: str | None = pydantic.Field(
        default=None,
        description="enables the device grant (RFC 8628): the URL that mints the user code",
    )
    token_endpoint: str = pydantic.Field(
        description="the URL for the code exchange, the device polling and the refresh",
    )
    revocation_endpoint: str | None = pydantic.Field(
        default=None,
        description="the URL `agent6 connect <name> --logout` revokes the grant at, when set",
    )
    redirect_uri: str = pydantic.Field(
        default=DEFAULT_REDIRECT_URI,
        description=(
            "the loopback redirect the registration names; port 0 takes a free port before the"
            " authorization starts; `http://127.0.0.1` or `http://[::1]`, a path, no query"
        ),
    )
    authorization_params: dict[str, str] = pydantic.Field(
        default_factory=dict,
        description=(
            "static, non-secret authorization parameters (an audience, a prompt); never"
            " `client_id`, `state`, `redirect_uri`, `response_type`, `scope` or a PKCE parameter"
        ),
    )
    request_encoding: Literal["form", "json"] = pydantic.Field(
        default="form", description="how the device and token requests are encoded"
    )
    refresh_skew_s: float = pydantic.Field(
        default=60.0,
        gt=0.0,
        description=(
            "seconds before an expiry the grant is refreshed, capped at half the issued lifetime"
        ),
    )
    exchange: OAuthExchangeConfig | None = pydantic.Field(
        default=None,
        description="one second exchange of the issuer's bearer for the model endpoint's token",
    )

    @pydantic.field_validator(
        "issuer",
        "authorization_endpoint",
        "device_authorization_endpoint",
        "token_endpoint",
        "revocation_endpoint",
    )
    @classmethod
    def _endpoints_are_credential_urls(
        cls, v: str | None, info: pydantic.ValidationInfo
    ) -> str | None:
        """Refuse an issuer or endpoint URL that cannot carry a credential safely.

        Args:
            v: The URL, or None for an unset optional endpoint.
            info: The field being validated.

        Returns:
            The URL unchanged.

        Raises:
            ValueError: The URL fails `credential_url_refusal`.
        """
        if v is None:
            return None
        return _require_credential_url(v, f"oauth.{info.field_name}")

    @pydantic.field_validator("authorization_endpoint")
    @classmethod
    def _endpoint_query_leaves_the_protocol_alone(cls, v: str | None) -> str | None:
        """Refuse an authorization endpoint whose own query sets a sign-in parameter.

        Args:
            v: The URL, or None.

        Returns:
            The URL unchanged.

        Raises:
            ValueError: The query names one of `RESERVED_AUTHORIZATION_PARAMS`.
        """
        if v is None:
            return None
        keys = {k for k, _ in parse.parse_qsl(parse.urlsplit(v).query, keep_blank_values=True)}
        clash = sorted(keys & RESERVED_AUTHORIZATION_PARAMS)
        if clash:
            raise ValueError(
                f"oauth.authorization_endpoint {v!r} sets the sign-in's own parameters in its"
                f" query: {', '.join(clash)}"
            )
        return v

    @pydantic.field_validator("redirect_uri")
    @classmethod
    def _redirect_is_a_loopback_path(cls, v: str) -> str:
        """Refuse a redirect that is not plain http to a literal loopback address and path.

        Args:
            v: The redirect URI.

        Returns:
            The URI unchanged.

        Raises:
            ValueError: Another scheme or host, no path, or a query or fragment.
        """
        try:
            parts = parse.urlsplit(v)
            parts.port  # noqa: B018  # raises ValueError on an out-of-range port
        except ValueError as exc:
            raise ValueError(f"oauth.redirect_uri {v!r}: {exc}") from exc
        if parts.scheme != "http" or parts.hostname not in _LOOPBACK_LITERALS:
            raise ValueError(
                f"oauth.redirect_uri {v!r} must be http://127.0.0.1 or http://[::1] with a port"
            )
        if parts.username is not None or parts.password is not None:
            raise ValueError(f"oauth.redirect_uri {v!r} carries userinfo")
        if not parts.path or parts.path == "/":
            raise ValueError(f"oauth.redirect_uri {v!r} needs a path, such as /auth/callback")
        if parts.query or parts.fragment:
            raise ValueError(f"oauth.redirect_uri {v!r} takes no query or fragment")
        return v

    @pydantic.field_validator("authorization_params")
    @classmethod
    def _params_leave_the_protocol_alone(cls, v: dict[str, str]) -> dict[str, str]:
        """Refuse a parameter the sign-in itself sets.

        Args:
            v: The parameters.

        Returns:
            The parameters unchanged.

        Raises:
            ValueError: A key is one of `RESERVED_AUTHORIZATION_PARAMS`.
        """
        clash = sorted(k for k in v if k in RESERVED_AUTHORIZATION_PARAMS)
        if clash:
            raise ValueError(
                "oauth.authorization_params may not set the sign-in's own parameters:"
                f" {', '.join(clash)}"
            )
        return v

    @pydantic.model_validator(mode="after")
    def _one_flow_at_least(self) -> OAuthConfig:
        """Refuse an issuer with no way to sign in.

        Returns:
            The model unchanged.

        Raises:
            ValueError: Neither flow endpoint is set.
        """
        if self.authorization_endpoint is None and self.device_authorization_endpoint is None:
            raise ValueError(
                "oauth needs authorization_endpoint (PKCE in a browser) or"
                " device_authorization_endpoint (a code typed on another device), or both"
            )
        return self

    def flows(self) -> tuple[str, ...]:
        """Return the sign-in flows the endpoints enable, in `auto`'s order of preference."""
        return tuple(
            name
            for name, endpoint in (
                ("authorization_code", self.authorization_endpoint),
                ("device", self.device_authorization_endpoint),
            )
            if endpoint is not None
        )
