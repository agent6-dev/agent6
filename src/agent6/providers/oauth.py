# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Bound OAuth grants, profile resolution and the shared refreshing bearer.

`OAuthCredential` validates the stored binding and pins the first account it reads.
Rotations hold both locks and persist before returning a bearer. Profile policies set the
expiry margin and initial refresh-token requirement. Nothing a remote returns is executed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol
from urllib import parse

from agent6 import paths, portable, secret_store
from agent6.config import _headers as header_config
from agent6.config import _oauth as oauth_config
from agent6.providers import (
    _headers,
    _oauth_exchange,
    _oauth_profile,
    _oauth_protocol,
    chatgpt_oauth,
    types,
)

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
    def extra_headers(self) -> Mapping[str, str | header_config.HeaderTemplate]: ...
    @property
    def oauth(self) -> oauth_config.OAuthConfig | None: ...


def profile_for(entry: _OAuthEntry) -> _oauth_profile.OAuthProfile:
    """Resolve a configured issuer or the fixed ChatGPT registration.

    Args:
        entry: A provider entry with configured OAuth or the ChatGPT format.

    Returns:
        The effective settings and sign-in policies.

    Raises:
        ProviderError: The entry has no OAuth sign-in.
    """
    if entry.api_format == "chatgpt":
        return chatgpt_oauth.PROFILE
    config = entry.oauth
    if config is None:
        raise types.ProviderError("this provider has no OAuth sign-in")
    return _oauth_profile.OAuthProfile(
        issuer=config.issuer,
        client_id=config.client_id,
        scopes=config.scopes,
        authorization_endpoint=config.authorization_endpoint,
        device_authorization_endpoint=config.device_authorization_endpoint,
        token_endpoint=config.token_endpoint,
        revocation_endpoint=config.revocation_endpoint,
        redirect_uri=config.redirect_uri,
        callback_host=parse.urlsplit(config.redirect_uri).hostname or "127.0.0.1",
        authorization_params=tuple(sorted(config.authorization_params.items())),
        request_encoding=config.request_encoding,
        revocation_encoding=config.request_encoding,
        device_flow="rfc8628",
        device_poll_endpoint=config.token_endpoint
        if config.device_authorization_endpoint
        else None,
        device_redirect_uri=None,
        refresh_skew_s=config.refresh_skew_s,
        exchange=config.exchange,
    )


def binding_fingerprint(entry: _OAuthEntry) -> str:
    """Return the fingerprint a stored grant is bound to.

    It covers the transport and configured OAuth settings except the refresh margin.
    Built-in profiles bind their endpoints, registration and encodings, never callable hooks.

    Args:
        entry: The provider entry with configured OAuth or the ChatGPT format.

    Returns:
        A hex digest.
    """
    if entry.api_format == "chatgpt":
        profile = profile_for(entry)
        settings: dict[str, Any] = {
            "issuer": profile.issuer,
            "client_id": profile.client_id,
            "scopes": profile.scopes,
            "authorization_endpoint": profile.authorization_endpoint,
            "device_authorization_endpoint": profile.device_authorization_endpoint,
            "token_endpoint": profile.token_endpoint,
            "revocation_endpoint": profile.revocation_endpoint,
            "redirect_uri": profile.redirect_uri,
            "authorization_params": dict(profile.authorization_params),
            "request_encoding": profile.request_encoding,
            "revocation_encoding": profile.revocation_encoding,
            "device_flow": profile.device_flow,
            "device_poll_endpoint": profile.device_poll_endpoint,
            "device_redirect_uri": profile.device_redirect_uri,
        }
    else:
        config = entry.oauth
        assert config is not None
        settings = config.model_dump(mode="json", exclude={"refresh_skew_s"})
    facts: dict[str, Any] = {
        "api_format": entry.api_format,
        "deployment": entry.deployment,
        "base_url": entry.base_url,
        "extra_query": dict(sorted(entry.extra_query.items())),
        "auth_style": entry.auth_style,
        "extra_headers": {
            name: value if isinstance(value, str) else value.model_dump(mode="json")
            for name, value in sorted(entry.extra_headers.items())
        },
        "oauth": settings,
    }
    canonical = json.dumps(facts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _reconnect(provider: str, why: str) -> types.ProviderError:
    return types.ProviderError(
        f"{why}; run `agent6 connect {provider}` to sign in again.", status_code=401
    )


def load_bound_grant(
    provider_name: str, entry: _OAuthEntry, *, secrets: dict[str, Any] | None = None
) -> secret_store.StoredGrant:
    """Read a bound grant without refreshing or writing it.

    Args:
        provider_name: The provider key used to store the grant and name its reconnect.
        entry: The effective provider entry.
        secrets: Loaded secrets; None reads the operator's secrets file.

    Returns:
        The grant, with its account claim filled in when metadata omitted it.

    Raises:
        ProviderError: The grant is absent, legacy, rebound or contradicts its account claim.
    """  # noqa: DOC501  # `_reconnect` builds the ProviderError named above
    profile = profile_for(entry)
    label = f"{profile.label} " if profile.label else ""
    data = secrets if secrets is not None else secret_store.load_secrets()
    grant = secret_store.load_oauth_grant(provider_name, secrets=data)
    if grant is None:
        providers = data.get("providers")
        raw = providers.get(provider_name) if isinstance(providers, dict) else None
        if (
            entry.api_format == "chatgpt"
            and isinstance(raw, dict)
            and raw.get("oauth_access_token")
            and not raw.get("oauth_version")
        ):
            raise _reconnect(provider_name, "This ChatGPT sign-in predates credential binding")
        raise _reconnect(provider_name, f"No {label}sign-in stored for provider {provider_name!r}")
    if grant.binding != binding_fingerprint(entry):
        raise _reconnect(
            provider_name,
            f"The stored {label}sign-in for {provider_name!r} was made under other provider"
            " settings (its issuer, endpoints or base_url changed since)",
        )
    claimed = (
        profile.account_id_of(_oauth_protocol.TokenGrant(grant.access_token, "", None))
        if profile.account_id_of
        else None
    )
    if claimed and grant.account_id and claimed != grant.account_id:
        raise _reconnect(
            provider_name,
            f"The stored {label}sign-in for {provider_name!r} carries an account id"
            " that does not match its own token",
        )
    return dataclasses.replace(grant, account_id=claimed or grant.account_id)


def needs_refresh(
    *, issued_at: float, expires_at: float | None, skew_s: float, now: float, cap_skew: bool = True
) -> bool:
    """Return whether a token with this expiry should be replaced before use.

    Args:
        issued_at: When the token was issued, as Unix time.
        expires_at: When it expires; None means the issuer named no expiry.
        skew_s: The requested margin before expiry.
        now: The clock.
        cap_skew: Whether the margin is capped at half the lifetime.

    Returns:
        True inside the margin before the expiry; False for a token with no expiry.
    """
    if expires_at is None:
        return False
    margin = min(skew_s, max(expires_at - issued_at, 0.0) / 2) if cap_skew else skew_s
    return now >= expires_at - margin


class OAuthCredential:
    """A cached, refreshing bearer over a provider's bound OAuth grant.

    Thread-safe in process; the reload-refresh-save transaction holds the interprocess
    lock beside `secrets.toml`, since a rotating refresh token is single-use. A stored
    grant whose binding differs from the configuration refuses with the reconnect hint
    before anything is sent. After a 401 the exchanged token is replaced first; an
    exchange the issuer refuses with 401 refreshes the issuer grant once. A 403 changes
    nothing a rotation could mend, so it never refreshes.
    """

    __slots__ = (
        "_account",
        "_auth_observer",
        "_entry",
        "_force",
        "_grant",
        "_header_set",
        "_last_returned",
        "_lock",
        "_now",
        "_profile",
        "_provider",
    )

    def __init__(
        self,
        provider_name: str,
        entry: _OAuthEntry,
        *,
        now: Callable[[], float] = time.time,
        auth_observer: types.AuthObserver | None = None,
    ) -> None:
        """Bind the credential to a provider's stored sign-in; nothing is read yet."""
        self._profile = profile_for(entry)
        self._provider = provider_name
        self._entry = entry
        self._account: str | None = None
        self._header_set = _headers.HeaderSet.from_config(entry.extra_headers)
        self._lock = threading.Lock()
        self._grant: secret_store.StoredGrant | None = None
        self._force = False
        self._last_returned = ""
        self._now = now
        self._auth_observer = auth_observer

    def _stored(self) -> secret_store.StoredGrant:
        return self._same_account(load_bound_grant(self._provider, self._entry))

    def _same_account(self, grant: secret_store.StoredGrant) -> secret_store.StoredGrant:
        """Pin the first account and refuse another or an unidentified sign-in.

        Args:
            grant: A grant whose metadata agrees with its access-token claim.

        Returns:
            The grant, unchanged.

        Raises:
            ProviderError: The grant names another account or lacks the pinned identity.
        """  # noqa: DOC501  # `_reconnect` builds the ProviderError named above
        account = grant.account_id
        if not self._account:
            self._account = account
        elif not account:
            raise _reconnect(
                self._provider,
                f"The stored grant for {self._provider!r} is an unidentified sign-in;"
                " it cannot replace this run's pinned account",
            )
        elif account != self._account:
            label = f"{self._profile.label} " if self._profile.label else ""
            raise _reconnect(
                self._provider,
                f"The stored {label}sign-in for {self._provider!r} belongs to a different"
                " account than this run started under",
            )
        return grant

    def account_id(self) -> str:
        """Return the pinned account id, or an empty string when none is known."""
        with self._lock:
            self._grant = self._grant or self._stored()
            return self._grant.account_id or ""

    def _bearer(self, grant: secret_store.StoredGrant) -> str:
        """Return the token the model endpoint takes from a grant."""
        return grant.exchange_access_token if self._profile.exchange else grant.access_token

    def _adopt(self, grant: secret_store.StoredGrant) -> str:
        self._grant = grant
        self._force = False
        self._last_returned = self._bearer(grant)
        return self._last_returned

    def _fresh(self, grant: secret_store.StoredGrant) -> bool:
        """Return whether the grant's bearer can be sent as it is."""
        now = self._now()
        skew = self._profile.refresh_skew_s
        if self._profile.exchange:
            if not grant.exchange_access_token:
                return False
            return not needs_refresh(
                issued_at=grant.exchange_issued_at,
                expires_at=grant.exchange_expires_at,
                skew_s=skew,
                now=now,
            )
        return not self._issuer_stale(grant)

    def _issuer_stale(self, grant: secret_store.StoredGrant) -> bool:
        expires_at = grant.expires_at
        if expires_at is None and self._profile.default_expires_in is not None:
            expires_at = grant.issued_at + self._profile.default_expires_in
        return needs_refresh(
            issued_at=grant.issued_at,
            expires_at=expires_at,
            skew_s=self._profile.refresh_skew_s,
            now=self._now(),
            cap_skew=self._profile.cap_refresh_skew,
        )

    def token(self, *, refresh: bool = True) -> str:
        """Return the bound bearer, renewing it unless refresh is disabled.

        Args:
            refresh: False reads the bound bearer without rotating or exchanging it.

        Returns:
            The bearer the model endpoint takes.

        Raises:
            ProviderError: No usable sign-in is stored, the binding differs, the lock could
                not be taken, or a refresh or exchange failed.
        """
        outcomes: list[types.AuthEvent] = []
        try:
            with self._lock:
                grant = self._grant or self._stored()
                if not refresh or (not self._force and self._fresh(grant)):
                    return self._adopt(grant)
                with portable.locked_file(paths.secrets_path()) as held:
                    if not held:
                        raise types.ProviderError(
                            "could not take the credential-refresh lock beside"
                            f" {paths.secrets_path()}; refusing to rotate the sign-in"
                            f" ({_LOCK_HINT})"
                        )
                    stored = self._stored()
                    # A sibling process that finished first is adopted rather than rotating again.
                    if self._fresh(stored) and self._bearer(stored) != self._last_returned:
                        return self._adopt(stored)
                    return self._adopt(self._renew(stored, outcomes))
        finally:
            # Observers may read the holder or the store; neither lock may be held here.
            if self._auth_observer is not None:
                for outcome in outcomes:
                    self._auth_observer(outcome)

    def _renew(
        self, grant: secret_store.StoredGrant, outcomes: list[types.AuthEvent]
    ) -> secret_store.StoredGrant:
        """Replace what is stale: the exchanged token first, the issuer grant when it needs it.

        Args:
            grant: The stored grant, read under the lock.
            outcomes: Settled operations to deliver after both locks release.

        Returns:
            The grant as stored after the renewal.

        Raises:
            ProviderError: A refresh or an exchange failed, or an expired grant has no
                refresh token.
        """
        oauth = self._profile
        issuer_stale = self._issuer_stale(grant)
        if issuer_stale or (self._force and not oauth.exchange):
            grant = self._refresh_issuer(grant, outcomes)
        if not oauth.exchange:
            return grant
        try:
            return self._exchange(grant, oauth.exchange, outcomes)
        except types.ProviderError as exc:
            if exc.status_code != 401 or issuer_stale:
                raise
            # The issuer's bearer was refused: refresh it once, then exchange again.
            return self._exchange(self._refresh_issuer(grant, outcomes), oauth.exchange, outcomes)

    def _refresh_issuer(
        self, grant: secret_store.StoredGrant, outcomes: list[types.AuthEvent]
    ) -> secret_store.StoredGrant:
        """Trade the refresh token for a new issuer grant and persist it before any exchange.

        Args:
            grant: The stored grant.
            outcomes: Settled operations to deliver after both locks release.

        Returns:
            The renewed grant, with the exchange fields cleared.

        Raises:
            ProviderError: The grant has no refresh token, or the issuer refused.
            OAuthError: The issuer refused and no usable sibling grant rescued the rotation.
        """  # noqa: DOC501  # `_reconnect` builds the ProviderError named above
        if not grant.refresh_token:
            raise _reconnect(
                self._provider,
                f"The sign-in for {self._provider!r} has expired and carries no refresh token",
            )
        oauth = self._profile
        try:
            fresh = _oauth_protocol.refresh_grant(
                oauth.token_endpoint,
                oauth.client_id,
                grant.refresh_token,
                provider=self._provider,
                label=oauth.label or self._provider,
                encoding=oauth.request_encoding,
                headers=self._header_set.renderer("oauth"),
            )
        except types.ProviderError as exc:
            if exc.status_code is not None:
                outcomes.append(types.AuthEvent("refresh", "failure"))
            if (
                not isinstance(exc, _oauth_protocol.OAuthError)
                or exc.code != "refresh_token_reused"
            ):
                raise
            # A rotation on another host can reach the secrets file after this host's refusal.
            time.sleep(1.0)
            rescued = self._stored()
            if rescued.access_token == grant.access_token or self._issuer_stale(rescued):
                raise
            return rescued
        renewed = self._same_account(
            stored_grant_for(fresh, self._entry, now=self._now(), previous=grant)
        )
        secret_store.save_oauth_grant(self._provider, renewed)
        outcomes.append(types.AuthEvent("refresh", "success"))
        return renewed

    def _exchange(
        self,
        grant: secret_store.StoredGrant,
        exchange: oauth_config.OAuthExchangeConfig,
        outcomes: list[types.AuthEvent],
    ) -> secret_store.StoredGrant:
        """Run the second exchange and persist its token beside the issuer grant.

        Args:
            grant: The stored grant whose issuer token is fresh.
            exchange: The configured exchange.
            outcomes: Settled operations to deliver after both locks release.

        Returns:
            The grant with the exchanged token.

        Raises:
            ProviderError: The exchange failed.
        """
        try:
            token, expires_at = _oauth_exchange.run_exchange(
                exchange,
                grant.access_token,
                headers=self._header_set.render(_headers.plain_facts("exchange")),
                now=self._now,
            )
        except types.ProviderError as exc:
            if exc.status_code is not None:
                outcomes.append(types.AuthEvent("exchange", "failure"))
            raise
        renewed = dataclasses.replace(
            grant,
            exchange_access_token=token,
            exchange_issued_at=self._now(),
            exchange_expires_at=expires_at,
        )
        secret_store.save_oauth_grant(self._provider, renewed)
        outcomes.append(types.AuthEvent("exchange", "success"))
        return renewed

    def invalidate(self, status: int, rejected_bearer: str) -> bool:
        """Return True and arm recovery only when a 401 rejects the last returned bearer."""
        if status != 401:
            return False
        with self._lock:
            if rejected_bearer != self._last_returned:
                return False
            self._force = True
        return True


def stored_grant_for(
    grant: _oauth_protocol.TokenGrant,
    entry: _OAuthEntry,
    *,
    now: float,
    previous: secret_store.StoredGrant | None = None,
) -> secret_store.StoredGrant:
    """Return the grant `agent6 connect` stores after a sign-in, bound to the entry.

    Args:
        grant: The issuer's grant.
        entry: The provider entry with configured OAuth or the ChatGPT format.
        now: The Unix time of the sign-in or refresh.
        previous: The grant being refreshed; omitted tokens and account claims carry over.

    Returns:
        The grant to store, its exchange fields empty.

    Raises:
        ProviderError: The profile requires an initial refresh token and the grant has none.
    """
    profile = profile_for(entry)
    if previous is None and profile.require_refresh_token and not grant.refresh_token.strip():
        raise types.ProviderError(
            f"{profile.label or 'OAuth'} token exchange response carried no refresh_token"
        )
    lifetime = grant.expires_in if grant.expires_in is not None else profile.default_expires_in
    account = profile.account_id_of(grant) if profile.account_id_of else None
    return secret_store.StoredGrant(
        binding=binding_fingerprint(entry),
        access_token=grant.access_token,
        refresh_token=grant.refresh_token or (previous.refresh_token if previous else ""),
        issued_at=now,
        expires_at=now + lifetime if lifetime is not None else None,
        exchange_access_token="",
        exchange_issued_at=0.0,
        exchange_expires_at=None,
        account_id=account or (previous.account_id if previous else None),
    )
