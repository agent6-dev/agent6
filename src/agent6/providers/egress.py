# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Process-wide egress routing for provider HTTP calls.

When ``[sandbox] agent_network = "providers"`` is active the agent process
runs inside a fresh, empty network namespace (see
``agent6.sandbox.broker``). It has **zero** IP connectivity of its own;
the only way out is a set of ``AF_UNIX`` sockets, one per allow-listed
provider endpoint, served by a trusted broker process that stayed behind
in the host network namespace.

This module is the transport seam. The provider call sites ask it to
issue every request; if an endpoint has a broker socket registered, the
request is dialled over that unix-domain socket (TLS stays end-to-end —
the broker only ever splices ciphertext). If no socket is registered the
request falls through to a plain ``httpx`` call, byte-for-byte identical
to the un-sandboxed path, so runs without it are unaffected.

Fail-closed: the kernel network namespace is the actual security
boundary. If the registry is empty or wrong while the process is
isolated, a provider call simply fails to connect (no route in the empty
netns) — it can never silently leak to an unexpected host.

The registry holds plain strings (host, port, socket path) and imports
nothing from ``agent6``; it lives in the providers package only so the
provider modules can consult it without crossing a module boundary.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Generator
from urllib.parse import urlsplit

import httpx

# (host, port) -> unix-domain socket path. Guarded by `_LOCK` because the
# providers create per-call streaming watchdog threads; registration
# happens once at startup but reads happen from worker threads.
_BROKER_ROUTES: dict[tuple[str, int], str] = {}
_LOCK = threading.Lock()


def parse_endpoint(url: str) -> tuple[str, int]:
    """Return the ``(host, port)`` a provider URL connects to.

    Ports default to 443 for ``https`` and 80 for ``http`` when the URL
    omits an explicit port. Raises ``ValueError`` if no host is present.
    """
    parts = urlsplit(url)
    host = parts.hostname
    if not host:
        raise ValueError(f"cannot determine host from URL: {url!r}")
    port = parts.port
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    return host, port


def register_route(host: str, port: int, uds_path: str) -> None:
    """Route requests to ``host:port`` over the broker's unix socket."""
    with _LOCK:
        _BROKER_ROUTES[(host, port)] = uds_path


def clear_routes() -> None:
    """Drop all broker routes (teardown / test isolation)."""
    with _LOCK:
        _BROKER_ROUTES.clear()


def _uds_for_url(url: str) -> str | None:
    try:
        key = parse_endpoint(url)
    except ValueError:
        return None
    with _LOCK:
        return _BROKER_ROUTES.get(key)


def http_post(
    url: str,
    *,
    headers: dict[str, str],
    content: bytes,
    timeout: float,
) -> httpx.Response:
    """POST ``url``, transparently dialling the broker socket if one is
    registered for the URL's endpoint. TLS (when the URL is ``https``)
    is performed against the URL host as usual, so certificate
    verification and SNI are unchanged regardless of routing."""
    uds = _uds_for_url(url)
    if uds is None:
        return httpx.post(url, headers=headers, content=content, timeout=timeout)
    transport = httpx.HTTPTransport(uds=uds)
    with httpx.Client(transport=transport, timeout=timeout) as client:
        return client.post(url, headers=headers, content=content)


@contextlib.contextmanager
def http_stream(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    content: bytes,
    timeout: float,
) -> Generator[httpx.Response]:
    """Streaming counterpart of :func:`http_post`, yielding the open
    response so the caller can iterate SSE lines."""
    uds = _uds_for_url(url)
    if uds is None:
        with httpx.stream(method, url, headers=headers, content=content, timeout=timeout) as resp:
            yield resp
        return
    transport = httpx.HTTPTransport(uds=uds)
    with (
        httpx.Client(transport=transport, timeout=timeout) as client,
        client.stream(method, url, headers=headers, content=content) as resp,
    ):
        yield resp
