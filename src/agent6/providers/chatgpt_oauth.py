# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The ChatGPT sign-in profile over the shared OAuth primitives.

`agent6 connect` owns the interaction; `_oauth_protocol` owns the wire shapes; this
module owns what is ChatGPT's: the authorize URL's parameters, the proprietary device
flow, the account and plan claims, and the `ChatGPTCredential` the provider holds.
The issuer, the client id and the redirect (pinned by the client registration) are
constants, so the profile dials only OpenAI's hosts. Nothing a remote returns is
executed. Tokens live in `secrets.toml` at 0600 and never reach a transcript or the jail.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable
from typing import Any
from urllib import parse

import httpx2

from agent6 import paths, portable, secret_store
from agent6.providers import _oauth_protocol, types

CHATGPT_ISSUER = "https://auth.openai.com"
# The Codex CLI's public client registration; its redirect is pinned to localhost:1455.
CHATGPT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REDIRECT_URI = "http://localhost:1455/auth/callback"
CALLBACK_PORT = 1455
# The device flow's fixed paths: the code entry page and the redirect paired with its codes.
DEVICE_VERIFY_PATH = "/codex/device"
_DEVICE_USERCODE_PATH = "/api/accounts/deviceauth/usercode"
_DEVICE_TOKEN_PATH = "/api/accounts/deviceauth/token"  # noqa: S105 - a URL path, not a secret
_DEVICE_REDIRECT_PATH = "/deviceauth/callback"
_DEVICE_TIMEOUT_S = 15 * 60.0
OAUTH_SCOPE = "openid profile email offline_access"
# The namespaced JWT claim OpenAI tokens carry the ChatGPT identity under.
_CLAIMS_KEY = "https://api.openai.com/auth"
# A token is refreshed this long before its nominal expiry, so it never dies mid-call.
_REFRESH_SKEW_S = 300.0
_LABEL = "ChatGPT"


def token_endpoint(issuer: str) -> str:
    """Return the issuer's token endpoint, the one path every grant comes from."""
    return f"{issuer.rstrip('/')}/oauth/token"


def revocation_endpoint(issuer: str) -> str:
    """Return the issuer's revocation endpoint."""
    return f"{issuer.rstrip('/')}/oauth/revoke"


def authorize_url(issuer: str, client_id: str, *, challenge: str, state: str) -> str:
    """Build the browser URL that starts the sign-in.

    The `id_token_add_organizations` and `codex_cli_simplified_flow` parameters are
    what the issuer expects from this client; without them a workspace account
    gets an id_token with no account claim.

    Args:
        issuer: The OAuth issuer.
        client_id: The client registration.
        challenge: The PKCE challenge.
        state: The state the callback must echo.

    Returns:
        The authorize URL.
    """
    query = parse.urlencode(
        [
            ("response_type", "code"),
            ("client_id", client_id),
            ("redirect_uri", REDIRECT_URI),
            ("scope", OAUTH_SCOPE),
            ("code_challenge", challenge),
            ("code_challenge_method", "S256"),
            ("state", state),
            ("id_token_add_organizations", "true"),
            ("codex_cli_simplified_flow", "true"),
            ("originator", "agent6"),
        ]
    )
    return f"{issuer.rstrip('/')}/oauth/authorize?{query}"


def exchange_code(
    issuer: str,
    client_id: str,
    *,
    code: str,
    verifier: str,
    provider: str,
    redirect_uri: str = REDIRECT_URI,
    timeout_s: float = _oauth_protocol.TOKEN_TIMEOUT_S,
) -> _oauth_protocol.TokenGrant:
    """Exchange an authorization code for a token grant.

    Args:
        issuer: The OAuth issuer.
        client_id: The client registration.
        code: The authorization code.
        verifier: The PKCE verifier the code was minted against.
        provider: The provider name, for the reconnect hint.
        redirect_uri: The redirect of the flow that minted the code.
        timeout_s: The request timeout.

    Returns:
        The grant, always carrying a refresh token.

    Raises:
        ProviderError: The issuer was unreachable, refused the exchange, or answered
            without a refresh token.
    """
    grant = _oauth_protocol.exchange_code(
        token_endpoint(issuer),
        client_id,
        code=code,
        verifier=verifier,
        redirect_uri=redirect_uri,
        provider=provider,
        label=_LABEL,
        timeout_s=timeout_s,
    )
    if not grant.refresh_token.strip():
        raise types.ProviderError("ChatGPT token exchange response carried no refresh_token")
    return grant


@dataclasses.dataclass(frozen=True, slots=True)
class DeviceAuth:
    """A started device-code sign-in.

    Attributes:
        device_auth_id: The issuer's id for the attempt.
        user_code: The code the person types on the verify page.
        interval_s: The polling interval the issuer asked for.
    """

    device_auth_id: str
    user_code: str
    interval_s: float


def start_device_auth(issuer: str, client_id: str) -> DeviceAuth | None:
    """Begin the code-entry sign-in.

    Args:
        issuer: The OAuth issuer.
        client_id: The client registration.

    Returns:
        The started attempt, or None when the issuer answers 404 (the caller falls
        back to the pasted callback).

    Raises:
        ProviderError: The issuer was unreachable, refused, or answered malformed.
    """
    url = f"{issuer.rstrip('/')}{_DEVICE_USERCODE_PATH}"
    try:
        resp = _oauth_protocol.post_json(
            url, {"client_id": client_id}, _oauth_protocol.TOKEN_TIMEOUT_S
        )
    except httpx2.HTTPError as exc:
        raise types.ProviderError(f"could not reach {url}: {exc}") from exc
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        raise types.ProviderError(
            f"device sign-in refused: HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        data: Any = resp.json()
        return DeviceAuth(
            device_auth_id=str(data["device_auth_id"]),
            user_code=str(data["user_code"]),
            interval_s=max(5.0, float(data.get("interval") or 5.0)),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise types.ProviderError(f"device sign-in response was malformed: {exc!r}") from exc


def poll_device_auth(
    issuer: str,
    client_id: str,
    device: DeviceAuth,
    *,
    provider: str,
    timeout_s: float = _DEVICE_TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> _oauth_protocol.TokenGrant:
    """Wait for the person to enter the code, then exchange the grant.

    The issuer answers pending as a 403, a 404 or `deviceauth_authorization_pending`
    and hands back the code and its verifier once approved.

    Args:
        issuer: The OAuth issuer.
        client_id: The client registration.
        device: The started attempt.
        provider: The provider name, for the reconnect hint.
        timeout_s: How long the code stays valid.
        sleep: The wait between polls.

    Returns:
        The grant.

    Raises:
        ProviderError: The issuer was unreachable or refused, the code expired
            unentered, or the exchange failed.
    """
    url = f"{issuer.rstrip('/')}{_DEVICE_TOKEN_PATH}"
    deadline = time.monotonic() + timeout_s
    interval = device.interval_s
    while time.monotonic() < deadline:
        try:
            resp = _oauth_protocol.post_json(
                url,
                {"device_auth_id": device.device_auth_id, "user_code": device.user_code},
                _oauth_protocol.TOKEN_TIMEOUT_S,
            )
        except httpx2.HTTPError as exc:
            raise types.ProviderError(f"could not reach {url}: {exc}") from exc
        if resp.status_code < 400:
            try:
                data: Any = resp.json()
                code = str(data["authorization_code"])
                verifier = str(data["code_verifier"])
            except (ValueError, KeyError, TypeError) as exc:
                raise types.ProviderError(
                    f"device sign-in response was malformed: {exc!r}"
                ) from exc
            return exchange_code(
                issuer,
                client_id,
                code=code,
                verifier=verifier,
                provider=provider,
                redirect_uri=f"{issuer.rstrip('/')}{_DEVICE_REDIRECT_PATH}",
            )
        detail = _oauth_protocol.error_code_of(resp)
        if resp.status_code in (403, 404) or detail == "deviceauth_authorization_pending":
            sleep(interval)
            continue
        if detail == "slow_down":
            interval += 5.0
            sleep(interval)
            continue
        raise types.ProviderError(
            "device sign-in failed: "
            f"HTTP {resp.status_code}:"
            f" {_oauth_protocol.scrub(resp.text, (device.device_auth_id,))[:200]}"
        )
    raise types.ProviderError(
        "device sign-in expired before the code was entered; run connect again"
    )


def account_id_of(grant: _oauth_protocol.TokenGrant) -> str:
    """Return the account id a grant is bound to; "" when absent.

    The claim rides in the access token and, for a workspace account, the id_token.

    Args:
        grant: The grant whose tokens are read, access token first.

    Returns:
        The `chatgpt_account_id` claim.
    """
    for token in (grant.access_token, grant.id_token):
        auth = _oauth_protocol.jwt_claims(token).get(_CLAIMS_KEY)
        if isinstance(auth, dict):
            # `user_id` is a user id, not an account id; a guessed header is worse than none.
            account = auth.get("chatgpt_account_id")
            if isinstance(account, str) and account:
                return account
    return ""


def plan_type_of(grant: _oauth_protocol.TokenGrant) -> str:
    """Return the plan the grant reports ("plus", "pro"); "" when absent."""
    for token in (grant.id_token, grant.access_token):
        auth = _oauth_protocol.jwt_claims(token).get(_CLAIMS_KEY)
        if isinstance(auth, dict):
            plan = auth.get("chatgpt_plan_type")
            if isinstance(plan, str) and plan:
                return plan
    return ""


def tokens_from_grant(
    grant: _oauth_protocol.TokenGrant, *, previous: secret_store.OAuthTokens | None = None
) -> secret_store.OAuthTokens:
    """Build the storable tokens for a grant.

    Args:
        grant: The grant.
        previous: The stored tokens; a refresh may omit the rotated refresh token
            or the identity claim, and both carry over rather than being erased.

    Returns:
        The tokens to store.
    """
    account = account_id_of(grant) or (previous.account_id if previous else "")
    refresh = grant.refresh_token or (previous.refresh_token if previous else "")
    return secret_store.OAuthTokens(
        access_token=grant.access_token,
        refresh_token=refresh,
        expires_at=time.time() + grant.expires_in,
        account_id=account,
    )


class ChatGPTCredential:
    """A cached, refreshing bearer over the stored ChatGPT OAuth tokens.

    The `BearerCredential` twin of `CommandToken`. Thread-safe in process; the
    reload-refresh-save transaction also holds an interprocess lock beside
    `secrets.toml`, because the refresh token is single-use and two processes
    submitting the same one kill the sign-in for both. The first account id read
    pins the credential: a grant bound to another account refuses with the
    reconnect hint rather than sending a bearer under a stale account header.
    After a 401, recovery adopts a newer stored grant first and refreshes only
    when none exists; a 403 never refreshes, since rotation cannot change what the
    account is allowed to do.
    """

    __slots__ = (
        "_account",
        "_client_id",
        "_force_refresh",
        "_issuer",
        "_last_returned",
        "_lock",
        "_provider",
        "_tokens",
    )

    def __init__(
        self,
        provider_name: str,
        *,
        issuer: str = CHATGPT_ISSUER,
        client_id: str = CHATGPT_CLIENT_ID,
    ) -> None:
        """Bind the credential to a provider's stored sign-in; nothing is read yet."""
        self._provider = provider_name
        self._issuer = issuer
        self._client_id = client_id
        self._lock = threading.Lock()
        self._tokens: secret_store.OAuthTokens | None = None
        self._force_refresh = False
        self._account = ""
        self._last_returned = ""

    def _stored(self) -> secret_store.OAuthTokens:
        """Load the stored tokens and check them against the pinned account.

        Returns:
            The stored tokens.

        Raises:
            ProviderError: No sign-in is stored, the stored account id contradicts
                the token's own claim, or the grant belongs to another account.
        """
        tokens = secret_store.load_oauth_tokens(self._provider)
        if tokens is None:
            raise types.ProviderError(
                f"No ChatGPT sign-in stored for provider {self._provider!r};"
                f" run `agent6 connect {self._provider}`.",
                status_code=401,
            )
        # A stored entry can hold a user id where the account id belongs; the repair is a reconnect.
        claimed = account_id_of(_oauth_protocol.TokenGrant(tokens.access_token, "", 0.0, ""))
        if claimed and tokens.account_id and claimed != tokens.account_id:
            raise types.ProviderError(
                f"The stored ChatGPT sign-in for {self._provider!r} carries an account id"
                f" that does not match its own token; run `agent6 connect {self._provider}`"
                " to sign in again.",
                status_code=401,
            )
        return self._same_account(tokens, claimed=claimed)

    def _same_account(
        self, tokens: secret_store.OAuthTokens, *, claimed: str = ""
    ) -> secret_store.OAuthTokens:
        """Pin on the first account id seen.

        Args:
            tokens: The grant's tokens.
            claimed: The account id the token itself claims, when read.

        Returns:
            The tokens, unchanged.

        Raises:
            ProviderError: The grant belongs to another account.
        """
        account = claimed or tokens.account_id
        if not self._account:
            self._account = account
        elif account and account != self._account:
            raise types.ProviderError(
                f"The stored ChatGPT sign-in for {self._provider!r} now belongs to a"
                f" different account than this run started under;"
                f" run `agent6 connect {self._provider}` to sign in again.",
                status_code=401,
            )
        return tokens

    def _adopt(self, tokens: secret_store.OAuthTokens) -> str:
        """Return the tokens' bearer after taking them as current."""
        self._tokens = tokens
        self._force_refresh = False
        self._last_returned = tokens.access_token
        return tokens.access_token

    def token(self) -> str:
        """Return a fresh-enough bearer, rotating the stored grant when needed.

        Returns:
            The access token.

        Raises:
            ProviderError: No usable sign-in is stored, the refresh lock could not
                be taken, or the refresh failed.
        """
        with self._lock:
            tokens = self._tokens or self._stored()
            if not self._force_refresh and time.time() < tokens.expires_at - _REFRESH_SKEW_S:
                return self._adopt(tokens)
            # An unserialised rotation kills every process's sign-in, so an unheld lock refuses.
            with portable.locked_file(paths.secrets_path()) as held:
                if not held:
                    raise types.ProviderError(
                        "could not take the credential-refresh lock beside"
                        f" {paths.secrets_path()}; refusing to rotate the single-use"
                        " ChatGPT refresh token (remove a stale .lock sibling"
                        " if one is left over)"
                    )
                # A sibling that finished first is adopted rather than burning another rotation.
                stored = self._stored()
                fresh_enough = time.time() < stored.expires_at - _REFRESH_SKEW_S
                if fresh_enough and stored.access_token != self._last_returned:
                    return self._adopt(stored)
                tokens = stored  # a stored sign-in always carries its refresh token
                try:
                    grant = _oauth_protocol.refresh_grant(
                        token_endpoint(self._issuer),
                        self._client_id,
                        tokens.refresh_token,
                        provider=self._provider,
                        label=_LABEL,
                    )
                except types.ProviderError as exc:
                    if "refresh_token_reused" not in str(exc):
                        raise
                    # The lock covers one host; a rotation from another host wins after a beat.
                    time.sleep(1.0)
                    rescued = self._stored()
                    if (
                        rescued.access_token == tokens.access_token
                        or time.time() >= rescued.expires_at - _REFRESH_SKEW_S
                    ):
                        raise
                    return self._adopt(rescued)
                fresh = self._same_account(tokens_from_grant(grant, previous=tokens))
                secret_store.save_oauth_tokens(self._provider, fresh)
                return self._adopt(fresh)

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
            self._force_refresh = True
        return True

    def account_id(self) -> str:
        """Return the account id the backend requires as `chatgpt-account-id`."""
        with self._lock:
            tokens = self._tokens or self._stored()
            self._tokens = tokens
        if tokens.account_id:
            return tokens.account_id
        return account_id_of(_oauth_protocol.TokenGrant(tokens.access_token, "", 0.0))
