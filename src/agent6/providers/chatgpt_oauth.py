# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The ChatGPT sign-in profile over the shared OAuth primitives.

`agent6 connect` owns the interaction; `_oauth_protocol` owns the wire shapes; this
module owns what is ChatGPT's: the authorize URL's parameters, the proprietary device
flow and the account and plan claims. It owns no credential cache or persistence.
The issuer, the client id and the redirect (pinned by the client registration) are
constants, so the profile dials only OpenAI's hosts. Nothing a remote returns is
executed. Tokens live in `secrets.toml` at 0600 and never reach a transcript or the jail.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from typing import Any

import httpx2

from agent6.providers import _oauth_profile, _oauth_protocol, types

CHATGPT_ISSUER = "https://auth.openai.com"
# The Codex CLI's public client registration; its redirect is pinned to localhost:1455.
CHATGPT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REDIRECT_URI = "http://localhost:1455/auth/callback"
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
        ProviderError: The issuer was unreachable, redirected, refused, or answered malformed.
    """  # noqa: DOC501  # `redirect_refusal` builds the ProviderError named above
    url = f"{issuer.rstrip('/')}{_DEVICE_USERCODE_PATH}"
    try:
        resp = _oauth_protocol.post_json(
            url, {"client_id": client_id}, _oauth_protocol.TOKEN_TIMEOUT_S
        )
    except httpx2.HTTPError as exc:
        raise types.ProviderError(f"could not reach {url}: {exc}") from exc
    if 300 <= resp.status_code < 400:
        raise _oauth_protocol.redirect_refusal(resp, operation="device start", label=_LABEL)
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        raise types.ProviderError(
            f"device sign-in refused: HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        data: Any = resp.json()
        device_id = data["device_auth_id"]
        user_code = data["user_code"]
        interval = max(5.0, float(data.get("interval") or 5.0))
    except (ValueError, KeyError, TypeError) as exc:
        raise types.ProviderError("device sign-in response was malformed") from exc
    if not isinstance(device_id, str) or not _oauth_protocol.header_safe(device_id):
        raise types.ProviderError("device sign-in response carried an unusable device id")
    if (
        not isinstance(user_code, str)
        or not _oauth_protocol.header_safe(user_code)
        or len(user_code) > 64
    ):
        raise types.ProviderError("device sign-in response carried an unusable user code")
    return DeviceAuth(device_id, user_code, interval)


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
        ProviderError: The issuer was unreachable, redirected or refused, the code expired
            unentered, or the exchange failed.
    """  # noqa: DOC501  # `redirect_refusal` builds the ProviderError named above
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
            detail = _oauth_protocol.scrub(str(exc), (device.device_auth_id, device.user_code))
            raise types.ProviderError(f"could not reach {url}: {detail}") from exc
        if 300 <= resp.status_code < 400:
            raise _oauth_protocol.redirect_refusal(resp, operation="device poll", label=_LABEL)
        if resp.status_code < 400:
            try:
                data: Any = resp.json()
                code = data["authorization_code"]
                verifier = data["code_verifier"]
            except (ValueError, KeyError, TypeError) as exc:
                raise types.ProviderError("device sign-in response was malformed") from exc
            if any(
                not isinstance(value, str) or not _oauth_protocol.header_safe(value)
                for value in (code, verifier)
            ):
                raise types.ProviderError("device sign-in response was malformed")
            return _oauth_protocol.exchange_code(
                f"{issuer.rstrip('/')}/oauth/token",
                client_id,
                code=code,
                verifier=verifier,
                provider=provider,
                label=_LABEL,
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
            f" {_oauth_protocol.scrub(resp.text, (device.device_auth_id, device.user_code))[:200]}"
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


PROFILE = _oauth_profile.OAuthProfile(
    issuer=CHATGPT_ISSUER,
    client_id=CHATGPT_CLIENT_ID,
    scopes=tuple(OAUTH_SCOPE.split()),
    authorization_endpoint=f"{CHATGPT_ISSUER}/oauth/authorize",
    device_authorization_endpoint=f"{CHATGPT_ISSUER}{_DEVICE_USERCODE_PATH}",
    token_endpoint=f"{CHATGPT_ISSUER}/oauth/token",
    revocation_endpoint=f"{CHATGPT_ISSUER}/oauth/revoke",
    redirect_uri=REDIRECT_URI,
    callback_host="127.0.0.1",
    authorization_params=(
        ("codex_cli_simplified_flow", "true"),
        ("id_token_add_organizations", "true"),
        ("originator", "agent6"),
    ),
    request_encoding="form",
    revocation_encoding="json",
    device_flow="chatgpt",
    device_poll_endpoint=f"{CHATGPT_ISSUER}{_DEVICE_TOKEN_PATH}",
    device_redirect_uri=f"{CHATGPT_ISSUER}{_DEVICE_REDIRECT_PATH}",
    refresh_skew_s=_REFRESH_SKEW_S,
    label=_LABEL,
    account_id_of=account_id_of,
    require_refresh_token=True,
    default_expires_in=3600.0,
    cap_refresh_skew=False,
)


def plan_type_of(grant: _oauth_protocol.TokenGrant) -> str:
    """Return the plan the grant reports ("plus", "pro"); "" when absent."""
    for token in (grant.id_token, grant.access_token):
        auth = _oauth_protocol.jwt_claims(token).get(_CLAIMS_KEY)
        if isinstance(auth, dict):
            plan = auth.get("chatgpt_plan_type")
            if isinstance(plan, str) and plan:
                return plan
    return ""
