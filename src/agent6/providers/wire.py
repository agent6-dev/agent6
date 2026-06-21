# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Orthogonal provider wiring: auth-header + request-URL construction.

agent6 separates three concerns that used to be fused into one ``kind``:

- **api_format** (``anthropic`` | ``openai``) selects the wire dialect; that
  lives in the two provider modules (body shaping, response parsing, SSE).
- **deployment** (``direct`` | ``vertex`` | ``azure``) is a named profile that
  decides the request URL shape and whether the model id rides in the URL path
  vs the JSON body. Adding a deployment (e.g. ``bedrock``) is a new branch here
  plus an auth style, with no config-shape change.
- **auth** (``style`` + a static key or a refreshable ``token_command``) decides
  the auth header.

This module is the small, security-relevant core shared by both providers: it is
where the auth credential becomes a header and where ``base_url`` becomes the
exact URL dialled. Keeping it in one place keeps the egress allow-list (which is
derived from the same ``base_url`` host) and the redaction set honest.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ApiFormat = Literal["anthropic", "openai"]
Deployment = Literal["direct", "vertex", "azure"]
AuthStyle = Literal["x_api_key", "bearer", "api_key_header", "none"]


def auth_header(style: AuthStyle, token: str) -> tuple[str, str] | None:
    """The ``(lowercased-header-name, value)`` for an auth style, or None.

    Returns None for ``none`` or an empty token (an unauthenticated local
    endpoint sends no auth header). httpx lowercases header names anyway; we do
    it here so callers and the transcript-redaction set agree on the spelling.
    """
    if style == "none" or not token:
        return None
    if style == "bearer":
        return ("authorization", f"Bearer {token}")
    if style == "x_api_key":
        return ("x-api-key", token)
    if style == "api_key_header":
        return ("api-key", token)
    return None  # pragma: no cover - exhaustive over AuthStyle


def _merge_query(url: str, extra_query: dict[str, str]) -> str:
    if not extra_query:
        return url
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(extra_query)
    return urlunsplit(parts._replace(query=urlencode(query)))


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

    Returns ``(url, model_in_body)``. ``model_in_body=False`` means the model id
    is carried in the URL path (Vertex-Anthropic ``:rawPredict``/
    ``:streamRawPredict``, Azure ``/deployments/{model}``) and MUST be omitted
    from the JSON body.
    """
    base = base_url.rstrip("/")
    if deployment == "vertex" and api_format == "anthropic":
        verb = "streamRawPredict" if streaming else "rawPredict"
        url, model_in_body = f"{base}/{model}:{verb}", False
    elif deployment == "azure":
        # api_format is validated to be "openai" for azure at config load.
        url, model_in_body = f"{base}/openai/deployments/{model}/chat/completions", False
    elif api_format == "anthropic":
        url, model_in_body = f"{base}/messages", True
    else:
        url, model_in_body = f"{base}/chat/completions", True
    return _merge_query(url, extra_query or {}), model_in_body
