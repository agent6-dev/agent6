# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read one URL, for a worker whose commands have no network.

Under the default `network` a jailed command has no network, so the worker cannot read a
linked spec or an API's docs. The fetch runs in the agent process, which has egress, and hands
the bytes back as a tool result.

One URL, GET only, no redirects followed, no header or body the model chose, and no credential
sent. Every refusal is a default-deny (a scheme that is not https, an address that is not
global, a body that is not text), not a list of bad things. It is still an egress channel a
model drives: a GET can encode data in its path, so a host is either on the operator's
allow-list or asked about, and an absent operator is a no.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import socket
import time
from urllib import parse

import httpx2

from agent6.tools import http_body

# Beyond the cap the read is refused, never silently truncated.
MAX_BYTES = 1 << 20
TIMEOUT_S = 20.0
# A binary blob is refused by what it is, not by extension.
_TEXTUAL = ("text/", "application/json", "application/xml", "application/xhtml+xml")
# An empty allow-list means no host, so opting out is written down and shows in `config show`.
ANY_HOST = "*"


class FetchRefusedError(Exception):
    """The URL was not fetched, and why."""


@dataclasses.dataclass(frozen=True, slots=True)
class Fetched:
    """One URL's response.

    Attributes:
        url: The URL as vetted.
        status: The HTTP status.
        content_type: The response's content type.
        body: The decoded text.
        location: A 30x's target, handed back for the model to decide on rather than followed.
    """

    url: str
    status: int
    content_type: str
    body: str
    location: str = ""


def host_allowed(host: str, allowed: tuple[str, ...]) -> bool:
    """Return whether the host is on the operator's list.

    Hosts, never URL prefixes: a prefix invites `evil.com/docs.python.org`. A leading dot allows
    subdomains, so `.readthedocs.io` covers the project pages without `notreadthedocs.io`.

    Args:
        host: The parsed host.
        allowed: The `sandbox.fetch_hosts` entries; `ANY_HOST` allows every host.

    Returns:
        True when an entry matches.
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


@dataclasses.dataclass(frozen=True, slots=True)
class Checked:
    """A vetted URL that has not touched the network: the gate's input.

    Attributes:
        url: The URL.
        host: The parsed host, the name every check is proved against.
    """

    url: str
    host: str

    def prompt(self) -> str:
        """Return the approval line: the parsed host, a port other than 443, the path and query.

        The host shown is the one the connection is proved against and the one `fetch_hosts`
        would name. A GET carries data out in its query string, so clipping the path or dropping
        the query is consent to an exfiltration the operator never saw.
        """
        parts = parse.urlsplit(self.url)
        tail = parts.path or "/"
        if parts.query:
            tail += f"?{parts.query}"
        port = "" if parts.port in (None, 443) else f":{parts.port}"
        return f"{self.host}{port} {tail}"


def check_url(url: str) -> Checked:
    """Vet a URL without touching the network.

    Everything the string alone can prove: https, no credentials, a real host, and a literal
    address that is public. A name is not resolved here: the DNS query for
    `<data>.attacker.example` delivers its label to that name's authoritative server, so
    resolving ahead of the operator's gate is itself an egress channel. `fetch` resolves behind
    the gate.

    Args:
        url: The URL the model asked for.

    Returns:
        The URL and its host.

    Raises:
        FetchRefusedError: The URL is malformed, not https, carries credentials, has no host,
            or names a literal address that is not public.
    """
    try:
        parts = parse.urlsplit(url)
        _ = parts.port  # a port outside 0-65535 raises here, before the approval
    except ValueError as exc:
        # A malformed literal ("http://[::1") or a bad port reads like every other refusal.
        raise FetchRefusedError(f"the URL cannot be read: {exc}") from exc
    if parts.scheme != "https":
        raise FetchRefusedError(f"only https is fetched, not {parts.scheme or 'a bare path'!r}")
    if parts.username or parts.password:
        # httpx turns userinfo into an Authorization header: a credential the model chose.
        raise FetchRefusedError("a URL with credentials in it is not fetched")
    host = parts.hostname
    if not host:
        raise FetchRefusedError("no host in the URL")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        return Checked(url=url, host=host)  # a name: resolved behind the gate
    if not literal.is_global:
        raise FetchRefusedError(f"{host} is not a public address")
    return Checked(url=url, host=host)


def fetch(checked: Checked) -> Fetched:
    """GET a vetted URL, refusing anything that is not a bounded text response.

    The host resolves here, behind every gate, to public addresses only, which keeps a fetch
    away from the cloud metadata endpoint (169.254.169.254), a loopback admin port and the
    operator's LAN. The connection dials exactly the address chosen, with the original name in
    SNI and Host, so the certificate is proved against the name and no second DNS answer can
    move it. Handing the name onward lets two resolvers disagree: CPython's `getaddrinfo`
    encodes an international name with IDNA2003 and httpx with UTS-46, so `ßeta.example.com`
    vets as `sseta.example.com` and connects to `xn--eta-4ka.example.com`, a bypass needing no
    race. Re-resolving also reopens the rebinding window.

    Redirects are returned, not followed: a 30x hands its Location back for the model, which
    re-runs every check. Following them is how one allowed host becomes a proxy to every other.

    Args:
        checked: The vetted URL.

    Returns:
        The response's status, content type, decoded body and any redirect target.

    Raises:
        FetchRefusedError: The host does not resolve or resolves to a non-public address, the
            response is not text, the body is compressed, too large or too slow, or the request
            fails.
    """
    parts = parse.urlsplit(checked.url)
    port = 443 if parts.port is None else parts.port
    try:
        infos = socket.getaddrinfo(checked.host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise FetchRefusedError(f"{checked.host} does not resolve: {exc}") from exc
    address = ""
    for info in infos:
        addr = ipaddress.ip_address(str(info[4][0]))
        if not addr.is_global:
            raise FetchRefusedError(
                f"{checked.host} resolves to {addr}, which is not a public address"
            )
        address = address or str(addr)
    if not address:
        raise FetchRefusedError(f"{checked.host} resolves to nothing")
    literal = f"[{address}]" if ":" in address else address
    dialled = parts._replace(netloc=f"{literal}:{port}").geturl()
    try:
        with (
            httpx2.Client(follow_redirects=False, timeout=TIMEOUT_S, verify=True) as client,
            client.stream(
                "GET",
                dialled,
                # Compression is declined here and refused by read_capped if sent anyway.
                headers={"Host": checked.host, "Accept-Encoding": "identity"},
                extensions={"sni_hostname": checked.host},
            ) as response,
        ):
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith(_TEXTUAL):
                raise FetchRefusedError(f"not a text response: content-type {content_type!r}")
            deadline = time.monotonic() + TIMEOUT_S
            body = http_body.read_capped(
                response, cap=MAX_BYTES, deadline=deadline, timeout_s=TIMEOUT_S
            )
            return Fetched(
                url=checked.url,
                status=response.status_code,
                content_type=content_type,
                body=body.decode(response.encoding or "utf-8", errors="replace"),
                location=response.headers.get("location", ""),
            )
    except http_body.BodyRefusedError as exc:
        raise FetchRefusedError(str(exc)) from exc
    except httpx2.HTTPError as exc:
        raise FetchRefusedError(f"could not fetch {checked.url}: {exc}") from exc
