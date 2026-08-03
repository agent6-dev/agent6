# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read one URL, for a worker whose commands have no network.

Under the default `tool_network`, a jailed command has no network at all, so
the worker cannot read a linked spec, an API's docs or a changelog. Its only
move is to ask the operator and wait. This runs in the AGENT process, which
already has egress, and hands the bytes back as a tool result.

Not a crawler and not a client: one URL, GET only, no redirects followed, no
headers or body from the model, and no credential ever sent. Every refusal is
a default-deny -- a scheme that is not https, an address that is not global, a
body that is not text -- rather than a list of bad things.

It is still an egress channel a model drives: a GET can encode data in its
path. That is why a host is either on the operator's allow-list or asked
about, and why an absent operator is a no.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx2

# A body is context the operator pays for, and a fetch is meant to answer one
# question. Beyond this the read is refused, never silently truncated.
MAX_BYTES = 1 << 20
TIMEOUT_S = 20.0
# What a model can read: prose and structured data. A binary blob in the
# context window is noise, so it is refused by what it IS, not by extension.
_TEXTUAL = ("text/", "application/json", "application/xml", "application/xhtml+xml")
# The allow-list value that means "any host". Empty means NONE, so opting out
# has to be written down and shows up in `agent6 config show` as a choice.
ANY_HOST = "*"


class FetchRefused(Exception):
    """The URL was not fetched, and why."""


@dataclass(frozen=True, slots=True)
class Fetched:
    """One URL's response."""

    url: str
    status: int
    content_type: str
    body: str


def host_allowed(host: str, allowed: tuple[str, ...]) -> bool:
    """Whether *host* is on the operator's list.

    Hosts, never URL prefixes: a prefix invites `evil.com/docs.python.org`. A
    leading dot allows subdomains, so `.readthedocs.io` covers the project
    pages without covering `notreadthedocs.io`.
    """
    if ANY_HOST in allowed:
        return True
    host = host.lower().rstrip(".")
    for entry in allowed:
        pattern = entry.lower().rstrip(".")
        if pattern.startswith("."):
            if host == pattern[1:] or host.endswith(pattern):
                return True
        elif host == pattern:
            return True
    return False


def check_url(url: str) -> str:
    """Return *url*'s host, or raise ``FetchRefused``.

    Everything a URL must be before anyone is asked about it: https, a real
    host, and a host that resolves only to addresses on the public internet.
    The last one is what keeps a fetch away from the cloud metadata endpoint
    (169.254.169.254), a loopback admin port, or the operator's LAN.
    """
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise FetchRefused(f"only https is fetched, not {parts.scheme or 'a bare path'!r}")
    host = parts.hostname
    if not host:
        raise FetchRefused("no host in the URL")
    try:
        infos = socket.getaddrinfo(host, parts.port or 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise FetchRefused(f"{host} does not resolve: {exc}") from exc
    for info in infos:
        addr = ipaddress.ip_address(str(info[4][0]))
        if not addr.is_global:
            raise FetchRefused(f"{host} resolves to {addr}, which is not a public address")
    return host


def fetch(url: str) -> Fetched:
    """GET *url*, refusing anything that is not a bounded text response.

    Redirects are returned, not followed: a 30x hands its Location back for the
    model to decide on, which re-runs every check. Following them silently is
    how one allowed host becomes an open proxy to every other.

    ``check_url`` must have passed first. Between its DNS answer and this
    connection a hostile resolver could still move the name to a private
    address; closing that needs pinning the checked address and carrying the
    name through TLS, which is worth doing if this tool grows past reading
    docs.
    """
    try:
        with (
            httpx2.Client(follow_redirects=False, timeout=TIMEOUT_S) as client,
            client.stream("GET", url) as response,
        ):
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith(_TEXTUAL):
                raise FetchRefused(f"not a text response: content-type {content_type!r}")
            body = bytearray()
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > MAX_BYTES:
                    raise FetchRefused(f"response is larger than {MAX_BYTES} bytes")
            return Fetched(
                url=url,
                status=response.status_code,
                content_type=content_type,
                body=bytes(body).decode(response.encoding or "utf-8", errors="replace"),
            )
    except httpx2.HTTPError as exc:
        raise FetchRefused(f"could not fetch {url}: {exc}") from exc
