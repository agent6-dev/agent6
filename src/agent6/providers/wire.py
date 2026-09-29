# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the auth header and the request URL every provider dials.

Three knobs stay orthogonal: `api_format` selects the wire dialect (the provider
modules own body shaping, parsing and SSE), `deployment` decides the URL shape and
whether the model id rides in the path or the body, and `auth` decides the header.
This is the one place a credential becomes a header and `base_url` becomes the URL
dialled, so the egress allow-list (derived from the same host) and the redaction
set stay honest.
"""

from __future__ import annotations

import re
from typing import Literal
from urllib import parse

from agent6.providers import types

ApiFormat = Literal["anthropic", "openai", "chatgpt"]
Deployment = Literal["direct", "vertex", "azure"]
AuthStyle = Literal["x_api_key", "bearer", "api_key_header", "none"]

# RFC 7230 field-value bytes minus obs-text; real credentials (JWT, hex, base64) are ASCII.
_HEADER_SAFE_VALUE = re.compile(r"[\t\x20-\x7e]*")


def auth_header(style: AuthStyle, token: str) -> tuple[str, str] | None:
    """Turn a credential into its auth header.

    The name is lowercased here so callers and the transcript redaction set agree
    on the spelling. This is the one place a credential is checked header-safe,
    before a transport error could echo it.

    Args:
        style: The auth style.
        token: The credential; "" sends no header.

    Returns:
        The `(name, value)` pair, or None for the `none` style or an empty token.

    Raises:
        ProviderError: The token holds a control, newline or non-ASCII byte (the
            message never shows the value).
    """
    if style == "none" or not token:
        return None
    if not _HEADER_SAFE_VALUE.fullmatch(token):
        raise types.ProviderError(
            "provider credential is not a valid HTTP header value: it holds a"
            " control character, newline, or non-ASCII byte (a stray newline"
            " from copy-paste is the usual cause). The value is not shown, to"
            " keep it out of logs."
        )
    if style == "bearer":
        return ("authorization", f"Bearer {token}")
    if style == "x_api_key":
        return ("x-api-key", token)
    if style == "api_key_header":
        return ("api-key", token)
    return None  # pragma: no cover - exhaustive over AuthStyle


def _merge_query(url: str, extra_query: dict[str, str]) -> str:
    """Return the URL with the extra query parameters merged in."""
    if not extra_query:
        return url
    parts = parse.urlsplit(url)
    query = dict(parse.parse_qsl(parts.query, keep_blank_values=True))
    query.update(extra_query)
    return parse.urlunsplit(parts._replace(query=parse.urlencode(query)))


def request_url(
    *,
    api_format: ApiFormat,
    deployment: Deployment,
    base_url: str,
    model: str,
    streaming: bool,
    extra_query: dict[str, str] | None = None,
) -> tuple[str, bool]:
    """Build the request URL and say whether the model id goes in the body.

    Args:
        api_format: The wire dialect.
        deployment: The URL profile.
        base_url: The endpoint's base URL.
        model: The model or deployment id.
        streaming: Selects the streaming verb where the deployment has one.
        extra_query: Query parameters merged into the URL.

    Returns:
        `(url, model_in_body)`; when the model id rides in the URL path (Vertex
        `:rawPredict`, Azure `/deployments/{model}`) it must be left out of the body.
    """
    base = base_url.rstrip("/")
    # A path-carried id is quoted to one segment so it cannot reshape the URL off the base host.
    if deployment == "vertex" and api_format == "anthropic":
        verb = "streamRawPredict" if streaming else "rawPredict"
        url, model_in_body = f"{base}/{parse.quote(model, safe='')}:{verb}", False
    elif deployment == "azure":
        # api_format is validated to be "openai" for azure at config load.
        seg = parse.quote(model, safe="")
        url, model_in_body = f"{base}/openai/deployments/{seg}/chat/completions", False
    elif api_format == "anthropic":
        url, model_in_body = f"{base}/messages", True
    elif api_format == "chatgpt":
        url, model_in_body = f"{base}/responses", True
    else:
        url, model_in_body = f"{base}/chat/completions", True
    return _merge_query(url, extra_query or {}), model_in_body
