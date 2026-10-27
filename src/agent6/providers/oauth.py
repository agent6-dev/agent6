# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The configured-OAuth credential: a bound, refreshing bearer over a stored grant.

`OAuthCredential` is the `BearerCredential` a provider with `[providers.<name>.oauth]`
holds. It reads the grant `agent6 connect <name>` stored, refuses it when the binding no
longer matches the configuration, refreshes the issuer grant before its expiry under the
interprocess lock, runs the configured second exchange when the model endpoint takes
another token, and recovers once from a 401. Nothing a remote returns is executed; token
values never reach an error text.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol
from urllib import parse

import httpx2

from agent6 import paths, portable, secret_store
from agent6.config import _oauth as oauth_config
from agent6.providers import _oauth_protocol, _transport, types

# A rotation refused for want of the lock: the message names the file to clear.
_LOCK_HINT = "remove a stale .lock sibling if one is left over"


class _OAuthEntry(Protocol):
    """What the credential reads off a provider entry."""

    @property
    def base_url(self) -> str: ...
    @property
    def deployment(self) -> str: ...
    @property
    def api_format(self) -> str: ...
    @property
    def auth_style(self) -> str: ...
    @property
    def extra_query(self) -> dict[str, str]: ...
    @property
    def oauth(self) -> oauth_config.OAuthConfig | None: ...


def binding_fingerprint(entry: _OAuthEntry) -> str:
    """Return the fingerprint a stored grant is bound to.

    It covers the effective transport (format, deployment, the resolved URL and query, the
    auth style) and every OAuth setting but the refresh margin, so a changed overlay cannot
    send a stored grant somewhere its sign-in never agreed to.

    Args:
        entry: The provider entry, whose `oauth` is set.

    Returns:
        A hex digest.
    """
    oauth = entry.oauth
    assert oauth is not None
    facts: dict[str, Any] = {
        "api_format": entry.api_format,
        "deployment": entry.deployment,
        "base_url": entry.base_url,
        "extra_query": dict(sorted(entry.extra_query.items())),
        "auth_style": entry.auth_style,
        "oauth": oauth.model_dump(mode="json", exclude={"refresh_skew_s"}),
    }
    canonical = json.dumps(facts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def needs_refresh(*, issued_at: float, expires_at: float | None, skew_s: float, now: float) -> bool:
    """Return whether a token with this expiry should be replaced before use.

    Args:
        issued_at: When the token was issued, as Unix time.
        expires_at: When it expires; None means the issuer named no expiry.
        skew_s: The configured margin, capped at half the token's lifetime.
        now: The clock.

    Returns:
        True inside the margin before the expiry; False for a token with no expiry.
    """
    if expires_at is None:
        return False
    margin = min(skew_s, max(expires_at - issued_at, 0.0) / 2)
    return now >= expires_at - margin


def _header_safe(token: str) -> bool:
    """Return whether a token can ride an HTTP header value."""
    return bool(token) and all(0x21 <= ord(ch) <= 0x7E for ch in token)


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
    trust_env = _transport.trust_env_for(exchange.endpoint)
    try:
        if exchange.method == "GET":
            resp = httpx2.get(
                exchange.endpoint,
                params=exchange.parameters or None,
                headers=sent,
                timeout=timeout_s,
                follow_redirects=False,
                trust_env=trust_env,
            )
        elif exchange.request_encoding == "json":
            resp = httpx2.post(
                exchange.endpoint,
                json=exchange.parameters,
                headers=sent,
                timeout=timeout_s,
                follow_redirects=False,
                trust_env=trust_env,
            )
        else:
            resp = httpx2.post(
                exchange.endpoint,
                content=parse.urlencode(exchange.parameters).encode("ascii"),
                headers={**sent, "content-type": "application/x-www-form-urlencoded"},
                timeout=timeout_s,
                follow_redirects=False,
                trust_env=trust_env,
            )
    except httpx2.HTTPError as exc:
        raise types.ProviderError(f"could not reach {exchange.endpoint}: {exc}") from exc
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
    if not isinstance(token, str) or not _header_safe(token):
        raise types.ProviderError(
            f"credential exchange answered without a usable `{exchange.access_token_field}`"
        )
    return token, _expiry_of(data, exchange, now=now())


class OAuthCredential:
    """A cached, refreshing bearer over a provider's stored configured-OAuth grant.

    Thread-safe in process; the reload-refresh-save transaction holds the interprocess
    lock beside `secrets.toml`, since a rotating refresh token is single-use. A stored
    grant whose binding differs from the configuration refuses with the reconnect hint
    before anything is sent. After a 401 the exchanged token is replaced first; an
    exchange the issuer refuses with 401 refreshes the issuer grant once. A 403 changes
    nothing a rotation could mend, so it never refreshes.
    """

    __slots__ = (
        "_entry",
        "_fingerprint",
        "_force",
        "_grant",
        "_last_returned",
        "_lock",
        "_now",
        "_provider",
    )

    def __init__(
        self, provider_name: str, entry: _OAuthEntry, *, now: Callable[[], float] = time.time
    ) -> None:
        """Bind the credential to a provider's stored sign-in; nothing is read yet."""
        assert entry.oauth is not None
        self._provider = provider_name
        self._entry = entry
        self._fingerprint = binding_fingerprint(entry)
        self._lock = threading.Lock()
        self._grant: secret_store.StoredGrant | None = None
        self._force = False
        self._last_returned = ""
        self._now = now

    @property
    def _oauth(self) -> oauth_config.OAuthConfig:
        oauth = self._entry.oauth
        assert oauth is not None
        return oauth

    def _reconnect(self, why: str) -> types.ProviderError:
        return types.ProviderError(
            f"{why}; run `agent6 connect {self._provider}` to sign in again.", status_code=401
        )

    def _stored(self) -> secret_store.StoredGrant:
        """Load the stored grant and check its binding.

        Returns:
            The grant.

        Raises:
            ProviderError: No grant is stored, or its binding differs from the configuration.
        """
        grant = secret_store.load_oauth_grant(self._provider)
        if grant is None:
            raise self._reconnect(f"No sign-in stored for provider {self._provider!r}")
        if grant.binding != self._fingerprint:
            raise self._reconnect(
                f"The stored sign-in for {self._provider!r} was made under other provider"
                " settings (its issuer, endpoints or base_url changed since)"
            )
        return grant

    def _bearer(self, grant: secret_store.StoredGrant) -> str:
        """Return the token the model endpoint takes from a grant."""
        return grant.exchange_access_token if self._oauth.exchange else grant.access_token

    def _adopt(self, grant: secret_store.StoredGrant) -> str:
        self._grant = grant
        self._force = False
        self._last_returned = self._bearer(grant)
        return self._last_returned

    def _fresh(self, grant: secret_store.StoredGrant) -> bool:
        """Return whether the grant's bearer can be sent as it is."""
        now = self._now()
        skew = self._oauth.refresh_skew_s
        if self._oauth.exchange:
            if not grant.exchange_access_token:
                return False
            return not needs_refresh(
                issued_at=grant.exchange_issued_at,
                expires_at=grant.exchange_expires_at,
                skew_s=skew,
                now=now,
            )
        return not needs_refresh(
            issued_at=grant.issued_at, expires_at=grant.expires_at, skew_s=skew, now=now
        )

    def token(self) -> str:
        """Return a fresh-enough bearer, rotating or exchanging the stored grant when needed.

        Returns:
            The bearer the model endpoint takes.

        Raises:
            ProviderError: No usable sign-in is stored, the binding differs, the lock could
                not be taken, or a refresh or exchange failed.
        """
        with self._lock:
            grant = self._grant or self._stored()
            if not self._force and self._fresh(grant):
                return self._adopt(grant)
            with portable.locked_file(paths.secrets_path()) as held:
                if not held:
                    raise types.ProviderError(
                        "could not take the credential-refresh lock beside"
                        f" {paths.secrets_path()}; refusing to rotate the sign-in ({_LOCK_HINT})"
                    )
                stored = self._stored()
                # A sibling process that finished first is adopted rather than rotating again.
                if self._fresh(stored) and self._bearer(stored) != self._last_returned:
                    return self._adopt(stored)
                return self._adopt(self._renew(stored))

    def _renew(self, grant: secret_store.StoredGrant) -> secret_store.StoredGrant:
        """Replace what is stale: the exchanged token first, the issuer grant when it needs it.

        Args:
            grant: The stored grant, read under the lock.

        Returns:
            The grant as stored after the renewal.

        Raises:
            ProviderError: A refresh or an exchange failed, or an expired grant has no
                refresh token.
        """
        oauth = self._oauth
        now = self._now()
        issuer_stale = needs_refresh(
            issued_at=grant.issued_at,
            expires_at=grant.expires_at,
            skew_s=oauth.refresh_skew_s,
            now=now,
        )
        if issuer_stale or (self._force and not oauth.exchange):
            grant = self._refresh_issuer(grant)
        if not oauth.exchange:
            return grant
        try:
            return self._exchange(grant, oauth.exchange)
        except types.ProviderError as exc:
            if exc.status_code != 401 or issuer_stale:
                raise
            # The issuer's bearer was refused: refresh it once, then exchange again.
            return self._exchange(self._refresh_issuer(grant), oauth.exchange)

    def _refresh_issuer(self, grant: secret_store.StoredGrant) -> secret_store.StoredGrant:
        """Trade the refresh token for a new issuer grant and persist it before any exchange.

        Args:
            grant: The stored grant.

        Returns:
            The renewed grant, with the exchange fields cleared.

        Raises:
            ProviderError: The grant has no refresh token, or the issuer refused.
        """
        if not grant.refresh_token:
            raise self._reconnect(
                f"The sign-in for {self._provider!r} has expired and carries no refresh token"
            )
        oauth = self._oauth
        fresh = _oauth_protocol.refresh_grant(
            oauth.token_endpoint,
            oauth.client_id,
            grant.refresh_token,
            provider=self._provider,
            label=self._provider,
            encoding=oauth.request_encoding,
        )
        now = self._now()
        renewed = secret_store.StoredGrant(
            binding=grant.binding,
            access_token=fresh.access_token,
            # A refresh that omits the rotated token keeps the previous one.
            refresh_token=fresh.refresh_token or grant.refresh_token,
            issued_at=now,
            expires_at=now + fresh.expires_in if fresh.expires_in is not None else None,
            exchange_access_token="",
            exchange_issued_at=0.0,
            exchange_expires_at=None,
        )
        secret_store.save_oauth_grant(self._provider, renewed)
        return renewed

    def _exchange(
        self, grant: secret_store.StoredGrant, exchange: oauth_config.OAuthExchangeConfig
    ) -> secret_store.StoredGrant:
        """Run the second exchange and persist its token beside the issuer grant.

        Args:
            grant: The stored grant whose issuer token is fresh.
            exchange: The configured exchange.

        Returns:
            The grant with the exchanged token.
        """
        token, expires_at = run_exchange(exchange, grant.access_token, now=self._now)
        renewed = secret_store.StoredGrant(
            binding=grant.binding,
            access_token=grant.access_token,
            refresh_token=grant.refresh_token,
            issued_at=grant.issued_at,
            expires_at=grant.expires_at,
            exchange_access_token=token,
            exchange_issued_at=self._now(),
            exchange_expires_at=expires_at,
        )
        secret_store.save_oauth_grant(self._provider, renewed)
        return renewed

    def invalidate(self, status: int = 401) -> bool:
        """Arm recovery for the next `token()` after an auth failure.

        Args:
            status: The 401 or 403 the transport saw.

        Returns:
            True on a 401, the bearer itself being bad; False on a 403, which is
            permission or entitlement and which no rotation can change.
        """
        if status != 401:
            return False
        with self._lock:
            self._force = True
        return True


def listing_bearer(provider_name: str, entry: _OAuthEntry) -> str:
    """Return the bearer a model listing sends for a configured-OAuth entry.

    The listing goes through the credential, so the binding check and the refresh apply
    to it as they do to a model call.

    Args:
        provider_name: The provider's config name.
        entry: Its entry, whose `oauth` is set.

    Returns:
        The bearer.

    Raises:
        ProviderError: As `OAuthCredential.token` raises.
    """
    return OAuthCredential(provider_name, entry).token()


def stored_grant_for(
    grant: _oauth_protocol.TokenGrant, entry: _OAuthEntry, *, now: float
) -> secret_store.StoredGrant:
    """Return the grant `agent6 connect` stores after a sign-in, bound to the entry.

    Args:
        grant: The issuer's grant.
        entry: The provider entry, whose `oauth` is set.
        now: The Unix time of the sign-in.

    Returns:
        The grant to store, its exchange fields empty.
    """
    return secret_store.StoredGrant(
        binding=binding_fingerprint(entry),
        access_token=grant.access_token,
        refresh_token=grant.refresh_token,
        issued_at=now,
        expires_at=now + grant.expires_in if grant.expires_in is not None else None,
        exchange_access_token="",
        exchange_issued_at=0.0,
        exchange_expires_at=None,
    )
