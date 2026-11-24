# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[providers.*]` model: one entry per backend."""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Literal
from urllib import parse

import pydantic

from agent6 import kinds
from agent6.config import header_config, oauth_config, sandbox_config, validation

_BACKEND_MEANINGS: dict[str, pydantic.JsonValue] = {
    "anthropic": "Use Anthropic Messages API",
    "openai": "Use OpenAI Chat Completions API",
    "chatgpt": "Use ChatGPT subscription API",
    "responses": "Use OpenAI Responses API",
    "claude_code": "Use installed signed-in Claude Code binary",
}

Backend = Literal["anthropic", "openai", "chatgpt", "responses", "claude_code"]
Deployment = Literal["direct", "vertex", "azure"]
AuthStyle = Literal["x_api_key", "bearer", "api_key_header", "none"]


def provider_name_refusal(name: str) -> str:
    """Return why a provider name cannot be one dotted-key component, or "".

    Args:
        name: The proposed provider key.

    Returns:
        The refusal, or "" for a non-empty TOML bare key.
    """
    if not name:
        return "provider name is required"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        return (
            f"provider name {name!r} is not a valid TOML bare key"
            " (use only ASCII letters, digits, '-', '_')."
        )
    return ""


_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_CHATGPT_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _default_base_url(backend: str, deployment: str) -> str | None:
    """Return the default `base_url` for a format and deployment.

    Args:
        backend: The wire format.
        deployment: The deployment profile.

    Returns:
        The fixed endpoint of a `direct` deployment; None for vertex and azure, whose URL
        carries the project, resource or region and must be configured.
    """
    if deployment != "direct":
        return None
    if backend == "anthropic":
        return _ANTHROPIC_DEFAULT_BASE_URL
    if backend == "responses":
        return None  # no vendor default: the endpoint is whatever the operator stood up
    return _CHATGPT_DEFAULT_BASE_URL if backend == "chatgpt" else _OPENAI_DEFAULT_BASE_URL


def _default_auth_style(backend: str, deployment: str) -> str:
    """Return the default `auth_style` for a format and deployment.

    Args:
        backend: The wire format.
        deployment: The deployment profile.

    Returns:
        The auth style name.
    """
    if deployment == "azure":
        return "api_key_header"
    if deployment == "vertex":
        return "bearer"
    return "x_api_key" if backend == "anthropic" else "bearer"


_BACKEND_DESCRIPTION = (
    "Provider backend; Chat Completions supports OpenAI, OpenRouter, Ollama, vLLM, LM Studio, "
    "llama.cpp and Gemini's OpenAI endpoint; ChatGPT subscriptions use the Codex Responses "
    "backend; configured Responses endpoints require `base_url`; Claude Code uses a Claude "
    "subscription without an HTTP endpoint or key"
)


def _require_json_shaped(value: Any, path: str) -> None:
    """Refuse a value JSON cannot carry, recursively.

    Args:
        value: The parsed TOML value.
        path: The value's dotted path under `extra_body`, for the message.

    Raises:
        ValueError: The value or a nested item is non-finite or has a non-JSON type.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"extra_body{path} holds a non-finite float, which JSON cannot carry")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _require_json_shaped(item, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_json_shaped(item, f"{path}.{key}")
        return
    raise ValueError(
        f"extra_body{path} holds a {type(value).__name__}, which JSON cannot carry"
        " (a TOML date/time is the usual cause); quote it as a string"
    )


def _idle_limit(seconds: float) -> float:
    """Require a positive duration or the no-limit sentinel.

    Args:
        seconds: The configured idle limit.

    Returns:
        The validated limit.

    Raises:
        ValueError: The limit is neither positive nor -1.
    """
    if seconds == -1 or seconds > 0:
        return seconds
    raise ValueError("must be positive, or exactly -1 for no idle limit")


IdleCompactSeconds = Annotated[
    float,
    pydantic.AfterValidator(_idle_limit),
    pydantic.Field(
        allow_inf_nan=False,
        description=(
            "Seconds from worker request start to compact during a wait or treat a resume as "
            "cold. Use -1 for no idle limit. Defaults: 240 for anthropic and responses, "
            "-1 for openai and chatgpt. Neither idle setting applies to claude_code."
        ),
    ),
]


class _ProviderBase(pydantic.BaseModel):
    """The transport and auth fields every HTTP provider shares.

    `backend` selects the wire dialect, `deployment` the URL profile, and the auth fields
    the credential; the three compose freely. `base_url` and `auth_style` default from the
    first two in `_fill_defaults`, so an entry naming only `backend` is usable.
    """

    model_config = validation.MODEL_CONFIG

    # Declared here so backend leads every subclass's model_fields; each subclass narrows it.
    backend: Backend
    idle_compact_s: IdleCompactSeconds = -1
    idle_compact_min_chars: int = pydantic.Field(
        default=240000,
        ge=0,
        description=(
            "Character floor for compaction during a wait, including system and tools. "
            "Does not apply on resume."
        ),
    )
    deployment: Deployment = pydantic.Field(
        json_schema_extra={
            "choice_meanings": {
                "direct": "Connect directly to provider",
                "vertex": "Route through Google Vertex AI",
                "azure": "Route through Microsoft Azure",
            }
        },
        default="direct",
        description=(
            "Provider URL shape and placement of model name and API version; Azure supports "
            "only the OpenAI format"
        ),
    )
    # Never empty after validation; the host also feeds the egress allow-list.
    base_url: str = pydantic.Field(
        default="",
        description=(
            "provider API endpoint's host and path prefix (`https://api.anthropic.com/v1`); "
            'required for `deployment = "vertex"` or `"azure"` and for `backend = "responses"`; '
            "a ChatGPT sign-in also dials its fixed OAuth authority, and a configured `oauth` "
            "table its issuer and exchange endpoints"
        ),
    )
    auth_style: AuthStyle = pydantic.Field(
        json_schema_extra={
            "choice_meanings": {
                "x_api_key": "Send x-api-key authentication header",
                "bearer": "Send Authorization Bearer header",
                "api_key_header": "Send api-key authentication header",
                "none": "Send no authentication header",
            }
        },
        default="bearer",
        description=(
            "Authentication header style; resolved per format and deployment when omitted; "
            "Anthropic uses x-api-key, compatible APIs use bearer, and Azure uses api-key"
        ),
    )
    # Secrets live here and in token_command, never in base_url, extra_headers or extra_query.
    api_key_env: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "environment variable holding the API key; precedence over `secrets.toml`; unset "
            "for a key stored by `agent6 connect` or an unauthenticated local endpoint"
        ),
    )
    token_command: validation.Argv = pydantic.Field(
        default=(),
        description=(
            "argv command printing a short-lived bearer token to stdout; re-run after "
            "`token_command_ttl_s` and once after a `401` or `403`; precedence over "
            "`api_key_env`"
        ),
    )
    token_command_ttl_s: float = pydantic.Field(
        gt=0.0,
        default=300.0,
        description="Seconds a `token_command` token is reused before the command runs again.",
    )
    extra_headers: dict[str, str | header_config.HeaderTemplate] = pydantic.Field(
        default_factory=dict,
        description=(
            "extra HTTP headers: a literal string rides every API request; a template table"
            " (`template`, `targets`, `when`) renders from request facts for the request classes"
            " it names; no secrets, config file not `0600`"
        ),
    )
    extra_body: dict[str, Any] = pydantic.Field(
        default_factory=dict,
        description=(
            "provider-specific JSON merged last into every request body; precedence for tuning "
            "keys (`max_tokens`, `temperature`); agent6-owned structural keys "
            "filtered out (messages, model, stream, tools, tool choice, response shape); "
            "JSON-shaped values only, non-finite floats and TOML dates and times refused; "
            "OpenRouter routing options here"
        ),
    )
    extra_query: dict[str, str] = pydantic.Field(
        default_factory=dict,
        description="Extra URL query parameters on every request (Azure's `api-version`).",
    )
    oauth: oauth_config.OAuthConfig | None = pydantic.Field(
        default=None,
        description=(
            "a configured OAuth issuer and public-client registration: `agent6 connect <name>`"
            " signs in there and stores the grant in `secrets.toml`; requires"
            ' `auth_style = "bearer"`, replaces `api_key_env` and `token_command`; a stored'
            " grant is bound to these settings and the entry's transport settings"
        ),
    )
    # The connect phase is bounded by providers.transport.CONNECT_TIMEOUT_S instead.
    http_timeout_s: float = pydantic.Field(
        gt=0.0,
        default=600.0,
        description=(
            "Seconds one HTTP call may take to read or write; the connect phase is capped at 20 s."
        ),
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, data: Any) -> Any:
        """Fill `base_url` and `auth_style` from the format and deployment, refusing bad pairs.

        Args:
            data: The raw table.

        Returns:
            The table with the defaults filled in; a non-dict value unchanged.

        Raises:
            ValueError: Anthropic on azure, chatgpt off `direct`, a deployment with no default
                `base_url` and none configured, or azure without `extra_query["api-version"]`.
        """
        if not isinstance(data, dict):
            return data
        fmt = data.get("backend")
        dep = data.get("deployment", "direct")
        if fmt == "anthropic" and dep == "azure":
            raise ValueError("deployment 'azure' requires backend 'openai'")
        if fmt in ("chatgpt", "responses") and dep != "direct":
            raise ValueError(f"backend {fmt!r} supports deployment 'direct' only")
        if fmt == "anthropic":
            data = dict(data)
            data.setdefault("server_compaction", dep == "direct" and "base_url" not in data)
        if not data.get("base_url"):
            default = _default_base_url(fmt, dep) if isinstance(fmt, str) else None
            if default is None:
                what = f"backend {fmt!r}" if fmt == "responses" else f"deployment {dep!r}"
                raise ValueError(f"base_url is required for {what}")
            data["base_url"] = default
        if not data.get("auth_style") and isinstance(fmt, str):
            data["auth_style"] = _default_auth_style(fmt, dep)
        if dep == "azure" and "api-version" not in (data.get("extra_query") or {}):
            raise ValueError("deployment 'azure' requires extra_query['api-version']")
        return data

    @pydantic.field_validator("base_url")
    @classmethod
    def _check_base_url(cls, v: str) -> str:
        """Validate a configured `base_url`.

        Args:
            v: The URL, or "" before `_fill_defaults` ran.

        Returns:
            The URL unchanged.

        Raises:
            ValueError: The URL fails `sandbox_config.validate_base_url`.
        """
        if v:
            sandbox_config.validate_base_url(v)
        return v

    @pydantic.field_validator("extra_body")
    @classmethod
    def _check_extra_body_json_shaped(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Refuse an `extra_body` value JSON cannot carry, at load instead of mid-request.

        Args:
            v: The table.

        Returns:
            The table unchanged.

        Raises:
            ValueError: A value is non-finite or has a non-JSON type, such as a TOML date or time.
        """
        if "service_tier" in v:
            raise ValueError(
                "set service_tier on an openai or responses entry, not extra_body.service_tier"
            )
        for key, value in v.items():
            _require_json_shaped(value, f".{key}")
        return v

    @pydantic.model_validator(mode="after")
    def _none_auth_takes_no_credential(self) -> _ProviderBase:
        """Refuse a credential source beside `auth_style = "none"`, which sends no header.

        Returns:
            The model unchanged.

        Raises:
            ValueError: `api_key_env` or `token_command` is set with `auth_style = "none"`.
        """
        if self.auth_style == "none" and (self.api_key_env or self.token_command):
            named = "api_key_env" if self.api_key_env else "token_command"
            raise ValueError(
                f"auth_style = 'none' sends no auth header, so {named} would never"
                " be used; drop one or the other"
            )
        return self

    @pydantic.field_validator("extra_headers")
    @classmethod
    def _headers_are_sendable(
        cls, v: dict[str, str | header_config.HeaderTemplate]
    ) -> dict[str, str | header_config.HeaderTemplate]:
        """Refuse a header the transport owns, an invalid name, a duplicate or an unsendable value.

        Args:
            v: The configured headers.

        Returns:
            The headers unchanged.

        Raises:
            ValueError: The refusal `extra_headers_refusal` names.
        """
        if refusal := header_config.extra_headers_refusal(v):
            raise ValueError(refusal)
        return v

    @pydantic.model_validator(mode="after")
    def _oauth_is_the_one_credential_source(self) -> _ProviderBase:
        """Refuse credentials or header targets that conflict with the `oauth` setting.

        Returns:
            The model unchanged.

        Raises:
            ValueError: A template targets `oauth` or `exchange` without an `oauth` setting,
                or `oauth` is set with `api_key_env`, `token_command` or an `auth_style`
                other than `bearer`.
        """
        if self.oauth is None:
            for name, value in self.extra_headers.items():
                if isinstance(value, header_config.HeaderTemplate):
                    for target in value.targets:
                        if target in ("exchange", "oauth"):
                            raise ValueError(
                                f"extra_headers.{name} targets {target!r}, but the entry has"
                                " no oauth setting"
                            )
            return self
        if refusal := oauth_config.credential_url_refusal(self.base_url, field="base_url"):
            raise ValueError(f"with oauth set, the grant rides every request: {refusal}")
        if self.api_key_env or self.token_command:
            named = "api_key_env" if self.api_key_env else "token_command"
            raise ValueError(
                f"oauth signs in at its issuer and stores the grant, so {named} would never"
                " be used; drop one or the other"
            )
        if self.auth_style != "bearer":
            raise ValueError(
                "oauth sends its grant as a bearer header (`Authorization: Bearer`); auth_style"
                f" {self.auth_style!r} is not honoured, drop it"
            )
        return self


class PricedProviderEntry(_ProviderBase):
    """An API whose listing or reference listing can price its calls."""

    price_from: str | None = pydantic.Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]+$",
        description=(
            "Configured provider whose listing supplies token rates instead of this entry's "
            "own listing; default: openrouter for direct Anthropic and OpenAI endpoints"
        ),
    )

    @property
    def direct_vendor(self) -> str:
        """The vendor of a direct API endpoint, or an empty string for a gateway."""
        if self.deployment != "direct":
            return ""
        host = parse.urlsplit(self.base_url).hostname
        if self.backend == "anthropic" and host == "api.anthropic.com":
            return "anthropic"
        return (
            "openai" if self.backend in ("openai", "responses") and host == "api.openai.com" else ""
        )

    @pydantic.model_validator(mode="after")
    def _default_price_source(self) -> PricedProviderEntry:
        if "price_from" not in self.model_fields_set and self.direct_vendor:
            object.__setattr__(self, "price_from", "openrouter")
        return self


class AnthropicProviderEntry(PricedProviderEntry):
    """The Anthropic Messages wire format.

    `direct` dials api.anthropic.com; `vertex` is Claude on Vertex (the model id in the URL,
    `anthropic_version` in the body, a Google OAuth bearer via `token_command`).
    """

    # The narrowing override is sound: the model is frozen, so nothing writes the wider type.
    backend: Literal["anthropic"] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        pydantic.Field(
            json_schema_extra={"choice_meanings": _BACKEND_MEANINGS},
            description=_BACKEND_DESCRIPTION,
        )
    )
    idle_compact_s: IdleCompactSeconds = 240
    prompt_caching: bool = pydantic.Field(
        default=True,
        description=(
            'prompt caching for `backend = "anthropic"` only; system prompt, tools and '
            "growing conversation re-read at 0.1x the input price"
        ),
    )
    server_compaction: bool = pydantic.Field(
        default=True,
        description=(
            "true on a supported model: the endpoint clears old tool results and writes "
            "the summary; agent6 changes no earlier message there; false or an unsupported "
            'model: agent6 compacts by itself; default: true for `deployment = "direct"` '
            "without a configured `base_url`, false otherwise"
        ),
    )


class _TieredProviderEntry(PricedProviderEntry):
    """An OpenAI wire that carries a service tier."""

    service_tier: kinds.ServiceTier | None = pydantic.Field(
        default=None,
        description=(
            "Processing tier on Chat Completions and Responses: standard, flex, fast, or auto "
            "for the project's choice; default: standard at api.openai.com, unset elsewhere"
        ),
        json_schema_extra={
            "choice_meanings": {
                "standard": "Use standard processing",
                "fast": "Request priority processing",
                "flex": "Request flex processing",
                "auto": "Let the project choose",
            }
        },
    )

    @pydantic.model_validator(mode="after")
    def _default_tier(self) -> _TieredProviderEntry:
        if self.service_tier is None and self.direct_vendor == "openai":
            object.__setattr__(self, "service_tier", "standard")
        return self

    @property
    def wire_tier(self) -> str | None:
        """The requested tier in the endpoint's vocabulary."""
        return kinds.TIER_WIRE[self.service_tier] if self.service_tier is not None else None


class OpenAIProviderEntry(_TieredProviderEntry):
    """The OpenAI Chat Completions wire format.

    `direct` serves OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp and Gemini's
    OpenAI endpoint; `vertex` is Gemini's Vertex OpenAPI endpoint; `azure` is Azure OpenAI
    (the deployment name in the URL, the api-version query, the `api-key` header).
    """

    backend: Literal["openai"] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        pydantic.Field(
            json_schema_extra={"choice_meanings": _BACKEND_MEANINGS},
            description=_BACKEND_DESCRIPTION,
        )
    )


class ChatGPTProviderEntry(_ProviderBase):
    """The ChatGPT-subscription Codex backend.

    The Responses wire format, authorized by the OAuth tokens `agent6 connect <name>` stores
    in `secrets.toml`; usage draws on the account's plan limits. The provider dials only
    `base_url` and OpenAI's fixed OAuth authority (the issuer and client id are constants).
    """

    backend: Literal["chatgpt"] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        pydantic.Field(
            json_schema_extra={"choice_meanings": _BACKEND_MEANINGS},
            description=_BACKEND_DESCRIPTION,
        )
    )

    @pydantic.field_validator("base_url")
    @classmethod
    def _chatgpt_base_url_is_https(cls, v: str) -> str:
        """Refuse a cleartext URL off loopback: the bearer and account id ride every request.

        Args:
            v: The URL.

        Returns:
            The URL unchanged.

        Raises:
            ValueError: The URL is plain http to a host other than loopback.
        """
        if sandbox_config.is_cleartext_url(v) and not sandbox_config.is_loopback_url(v):
            raise ValueError(
                "a chatgpt base_url must use https (plain http is allowed only"
                " for a loopback test endpoint)"
            )
        return v

    @pydantic.field_validator("extra_headers")
    @classmethod
    def _reserved_headers_stay_structural(
        cls, v: dict[str, str | header_config.HeaderTemplate]
    ) -> dict[str, str | header_config.HeaderTemplate]:
        """Refuse an override of the auth headers, which would re-route or mislabel every call.

        Args:
            v: The extra headers.

        Returns:
            The headers unchanged.

        Raises:
            ValueError: A header names chatgpt-account-id, originator or session-id in any
                case, or fails the shared `extra_headers_refusal`.
        """
        reserved = frozenset({"chatgpt-account-id", "originator", "session-id"})
        if refusal := header_config.extra_headers_refusal(v, reserved=reserved):
            raise ValueError(refusal)
        return v

    @pydantic.model_validator(mode="after")
    def _oauth_takes_no_key_source(self) -> ChatGPTProviderEntry:
        """Refuse a key source or a non-bearer auth style beside the stored OAuth tokens.

        Returns:
            The model unchanged.

        Raises:
            ValueError: `api_key_env` or `token_command` is set, or `auth_style` is not `bearer`.
        """
        if self.oauth is not None:
            raise ValueError(
                "backend 'chatgpt' signs in at its fixed issuer; a custom oauth table"
                " belongs on an openai, anthropic or responses entry"
            )
        if self.api_key_env or self.token_command:
            named = "api_key_env" if self.api_key_env else "token_command"
            raise ValueError(
                f"backend 'chatgpt' authenticates with the OAuth tokens"
                f" `agent6 connect <name>` stores, so {named} would never be used;"
                " drop it"
            )
        if self.auth_style != "bearer":
            raise ValueError(
                "backend 'chatgpt' always sends its OAuth token as"
                f" `Authorization: Bearer`; auth_style {self.auth_style!r} is not honoured,"
                " drop it"
            )
        return self


class ResponsesProviderEntry(_TieredProviderEntry):
    """The Responses API on a configured endpoint.

    `/responses` is appended to `base_url`, which has no vendor default; `direct` is the only
    deployment. A key, a `token_command` or a configured `oauth` issuer authenticates it, and
    its usage meters tokens like any keyed endpoint.
    """

    idle_compact_s: IdleCompactSeconds = 240
    backend: Literal["responses"] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        pydantic.Field(
            json_schema_extra={"choice_meanings": _BACKEND_MEANINGS},
            description=_BACKEND_DESCRIPTION,
        )
    )


class ClaudeCodeProviderEntry(pydantic.BaseModel):
    """The operator's installed Claude Code binary.

    It dials no endpoint and holds no credential, so the transport and auth fields do not
    exist on it and `extra="forbid"` refuses each by name; usage draws on the binary's own
    Claude login.
    """

    model_config = validation.MODEL_CONFIG

    backend: Literal["claude_code"] = pydantic.Field(
        json_schema_extra={"choice_meanings": _BACKEND_MEANINGS}, description=_BACKEND_DESCRIPTION
    )
    binary: str = pydantic.Field(
        default="claude",
        min_length=1,
        description=(
            'executable for `backend = "claude_code"` only; a name on PATH or an absolute path'
        ),
    )


ProviderEntry = Annotated[
    AnthropicProviderEntry
    | OpenAIProviderEntry
    | ChatGPTProviderEntry
    | ResponsesProviderEntry
    | ClaudeCodeProviderEntry,
    pydantic.Discriminator("backend"),
]


def plan_metered(entry: object) -> bool:
    """Return whether calls through the entry draw on a subscription plan.

    Args:
        entry: A provider entry.

    Returns:
        True for the plan-metered formats, whose calls count plan percent at an authoritative
        $0 and are never priced per token.
    """
    return isinstance(entry, (ChatGPTProviderEntry, ClaudeCodeProviderEntry))
