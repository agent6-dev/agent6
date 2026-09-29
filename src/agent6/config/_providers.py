# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[providers.*]` model: one entry per endpoint, discriminated by wire format."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Discriminator, Field, field_validator, model_validator

from agent6.config._base import MODEL_CONFIG, Argv
from agent6.config._sandbox import is_cleartext_url, is_loopback_url

ApiFormat = Literal["anthropic", "openai", "chatgpt", "claude_code"]
Deployment = Literal["direct", "vertex", "azure"]
AuthStyle = Literal["x_api_key", "bearer", "api_key_header", "none"]


def validate_base_url(url: str, field: str = "base_url") -> None:
    """Refuse a provider URL that is not an http(s) URL with a host and a valid port.

    The usual paste error is an API key or a bare host in the field, which would otherwise
    fail much later as an opaque HTTP error.

    Args:
        url: The configured URL.
        field: The field name for the message.

    Raises:
        ValueError: No http(s) scheme, no host, or an out-of-range port.
    """
    try:
        parts = urlsplit(url)
        port = parts.port  # raises ValueError on an out-of-range port
    except ValueError as exc:
        raise ValueError(f"invalid {field} {url!r}: {exc}") from exc
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"{field} {url!r} must start with http:// or https://")
    if not parts.hostname:
        raise ValueError(f"{field} {url!r} has no host")
    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"{field} {url!r} has an invalid port")


_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_CHATGPT_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _default_base_url(api_format: str, deployment: str) -> str | None:
    """Return the default `base_url` for a format and deployment.

    Args:
        api_format: The wire format.
        deployment: The deployment profile.

    Returns:
        The fixed endpoint of a `direct` deployment; None for vertex and azure, whose URL
        carries the project, resource or region and must be configured.
    """
    if deployment != "direct":
        return None
    if api_format == "anthropic":
        return _ANTHROPIC_DEFAULT_BASE_URL
    return _CHATGPT_DEFAULT_BASE_URL if api_format == "chatgpt" else _OPENAI_DEFAULT_BASE_URL


def _default_auth_style(api_format: str, deployment: str) -> str:
    """Return the default `auth_style` for a format and deployment.

    Args:
        api_format: The wire format.
        deployment: The deployment profile.

    Returns:
        The auth style name.
    """
    if deployment == "azure":
        return "api_key_header"
    if deployment == "vertex":
        return "bearer"
    return "x_api_key" if api_format == "anthropic" else "bearer"


_API_FORMAT_DESCRIPTION = (
    "The wire format: `anthropic` (the Messages API), `openai` (Chat Completions: OpenAI, "
    "OpenRouter, Ollama, vLLM, LM Studio, llama.cpp, Gemini's OpenAI endpoint), `chatgpt` "
    "(the ChatGPT-subscription Codex backend, Responses API), or `claude_code` (the installed, "
    "signed-in Claude Code binary on a Claude subscription; no HTTP endpoint, no key)."
)


def _require_json_shaped(value: Any, path: str) -> None:
    """Refuse a value JSON cannot carry, recursively.

    Args:
        value: The parsed TOML value.
        path: The value's dotted path under `extra_body`, for the message.

    Raises:
        ValueError: The value or a nested item is not JSON-shaped (a TOML date or time).
    """
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


class _ProviderBase(BaseModel):
    """The transport and auth fields every HTTP provider shares.

    `api_format` selects the wire dialect, `deployment` the URL profile, and the auth fields
    the credential; the three compose freely. `base_url` and `auth_style` default from the
    first two in `_fill_defaults`, so an entry naming only `api_format` is usable.
    """

    model_config = MODEL_CONFIG

    # Declared here so api_format leads every subclass's model_fields; each subclass narrows it.
    api_format: ApiFormat
    deployment: Deployment = Field(
        default="direct",
        description=(
            "`direct`, `vertex` (Google Vertex AI), or `azure` (Azure OpenAI; `openai` format "
            "only): the URL shape and where the model name and API version go."
        ),
    )
    # Never empty after validation; the host also feeds the egress allow-list.
    base_url: str = Field(
        default="",
        description=(
            "The endpoint's host and path prefix (`https://api.anthropic.com/v1`); required for "
            "`vertex` and `azure`. The provider API destination; a ChatGPT sign-in also dials "
            "its fixed OAuth authority."
        ),
    )
    auth_style: AuthStyle = Field(
        default="bearer",
        description=(
            "How the key is sent: `x_api_key` (Anthropic), `bearer` (`Authorization: Bearer`, the "
            "OpenAI style), `api_key_header` (Azure), or `none` (an unauthenticated local "
            "endpoint). `agent6 connect` sets it."
        ),
    )
    # Secrets live here and in token_command, never in base_url, extra_headers or extra_query.
    api_key_env: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The environment variable holding the API key; it wins over `secrets.toml`. Unset for "
            "a key `agent6 connect` stored, or an unauthenticated local endpoint."
        ),
    )
    token_command: Argv = Field(
        default=(),
        description=(
            "A command (argv) that prints a short-lived bearer token to stdout, re-run when "
            "`token_command_ttl_s` expires and once after a `401` or `403`. Wins over "
            "`api_key_env`."
        ),
    )
    token_command_ttl_s: float = Field(
        gt=0.0,
        default=300.0,
        description="Seconds a `token_command` token is reused before the command runs again.",
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra HTTP headers on every request to this provider. Never a secret: the config file "
            "is not `0600`."
        ),
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider-specific JSON merged last into every request body, so tuning keys "
            "(`max_tokens`, `temperature`) win; the structural keys agent6 owns (messages, model, "
            "stream, tools, tool choice, response shape) are filtered out. Values must be "
            "JSON-shaped (a TOML date or time is refused). OpenRouter's routing options go here."
        ),
    )
    extra_query: dict[str, str] = Field(
        default_factory=dict,
        description="Extra URL query parameters on every request (Azure's `api-version`).",
    )
    # The connect phase is bounded by providers._transport.CONNECT_TIMEOUT_S instead.
    http_timeout_s: float = Field(
        gt=0.0,
        default=600.0,
        description=(
            "Seconds one HTTP call may take to read or write; the connect phase is capped at 20 s."
        ),
    )

    @model_validator(mode="before")
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
        fmt = data.get("api_format")
        dep = data.get("deployment", "direct")
        if fmt == "anthropic" and dep == "azure":
            raise ValueError("deployment 'azure' requires api_format 'openai'")
        if fmt == "chatgpt" and dep != "direct":
            raise ValueError("api_format 'chatgpt' supports deployment 'direct' only")
        if not data.get("base_url"):
            default = _default_base_url(fmt, dep) if isinstance(fmt, str) else None
            if default is None:
                raise ValueError(f"base_url is required for deployment {dep!r}")
            data["base_url"] = default
        if not data.get("auth_style") and isinstance(fmt, str):
            data["auth_style"] = _default_auth_style(fmt, dep)
        if dep == "azure" and "api-version" not in (data.get("extra_query") or {}):
            raise ValueError("deployment 'azure' requires extra_query['api-version']")
        return data

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, v: str) -> str:
        """Validate a configured `base_url`.

        Args:
            v: The URL, or "" before `_fill_defaults` ran.

        Returns:
            The URL unchanged.

        Raises:
            ValueError: The URL fails `validate_base_url`.
        """
        if v:
            validate_base_url(v)
        return v

    @field_validator("extra_body")
    @classmethod
    def _check_extra_body_json_shaped(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Refuse an `extra_body` value JSON cannot carry, at load instead of mid-request.

        Args:
            v: The table.

        Returns:
            The table unchanged.

        Raises:
            ValueError: A value is a TOML date or time, or another non-JSON type.
        """
        for key, value in v.items():
            _require_json_shaped(value, f".{key}")
        return v

    @model_validator(mode="after")
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


class AnthropicProviderEntry(_ProviderBase):
    """The Anthropic Messages wire format.

    `direct` dials api.anthropic.com; `vertex` is Claude on Vertex (the model id in the URL,
    `anthropic_version` in the body, a Google OAuth bearer via `token_command`).
    """

    # The narrowing override is sound: the model is frozen, so nothing writes the wider type.
    api_format: Literal["anthropic"] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        Field(description=_API_FORMAT_DESCRIPTION)
    )
    prompt_caching: bool = Field(
        default=True,
        description=(
            "Anthropic prompt caching: the system prompt, the tools, and the growing conversation "
            "are re-read at 0.1x the input price. `anthropic` format only."
        ),
    )


class OpenAIProviderEntry(_ProviderBase):
    """The OpenAI Chat Completions wire format.

    `direct` serves OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp and Gemini's
    OpenAI endpoint; `vertex` is Gemini's Vertex OpenAPI endpoint; `azure` is Azure OpenAI
    (the deployment name in the URL, the api-version query, the `api-key` header).
    """

    api_format: Literal["openai"] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        Field(description=_API_FORMAT_DESCRIPTION)
    )


class ChatGPTProviderEntry(_ProviderBase):
    """The ChatGPT-subscription Codex backend.

    The Responses wire format, authorized by the OAuth tokens `agent6 connect <name>` stores
    in `secrets.toml`; usage draws on the account's plan limits. The provider dials only
    `base_url` and OpenAI's fixed OAuth authority (the issuer and client id are constants).
    """

    api_format: Literal["chatgpt"] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        Field(description=_API_FORMAT_DESCRIPTION)
    )

    @field_validator("base_url")
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
        if is_cleartext_url(v) and not is_loopback_url(v):
            raise ValueError(
                "a chatgpt base_url must use https (plain http is allowed only"
                " for a loopback test endpoint)"
            )
        return v

    @field_validator("extra_headers")
    @classmethod
    def _reserved_headers_stay_structural(cls, v: dict[str, str]) -> dict[str, str]:
        """Refuse an override of the auth headers, which would re-route or mislabel every call.

        Args:
            v: The extra headers.

        Returns:
            The headers unchanged.

        Raises:
            ValueError: A header names authorization, chatgpt-account-id, originator or
                session-id in any case.
        """
        reserved = {"authorization", "chatgpt-account-id", "originator", "session-id"}
        clash = sorted(k for k in v if k.lower() in reserved)
        if clash:
            raise ValueError(
                f"extra_headers may not override the chatgpt auth headers: {', '.join(clash)}"
            )
        return v

    @model_validator(mode="after")
    def _oauth_takes_no_key_source(self) -> ChatGPTProviderEntry:
        """Refuse a key source or a non-bearer auth style beside the stored OAuth tokens.

        Returns:
            The model unchanged.

        Raises:
            ValueError: `api_key_env` or `token_command` is set, or `auth_style` is not `bearer`.
        """
        if self.api_key_env or self.token_command:
            named = "api_key_env" if self.api_key_env else "token_command"
            raise ValueError(
                f"api_format 'chatgpt' authenticates with the OAuth tokens"
                f" `agent6 connect <name>` stores, so {named} would never be used;"
                " drop it"
            )
        if self.auth_style != "bearer":
            raise ValueError(
                "api_format 'chatgpt' always sends its OAuth token as"
                f" `Authorization: Bearer`; auth_style {self.auth_style!r} is not honoured,"
                " drop it"
            )
        return self


class ClaudeCodeProviderEntry(BaseModel):
    """The operator's installed Claude Code binary.

    It dials no endpoint and holds no credential, so the transport and auth fields do not
    exist on it and `extra="forbid"` refuses each by name; usage draws on the binary's own
    Claude login.
    """

    model_config = MODEL_CONFIG

    api_format: Literal["claude_code"] = Field(description=_API_FORMAT_DESCRIPTION)
    binary: str = Field(
        default="claude",
        min_length=1,
        description=(
            "The Claude Code executable: a name on PATH or an absolute path. `claude_code`"
            " format only."
        ),
    )


ProviderEntry = Annotated[
    AnthropicProviderEntry | OpenAIProviderEntry | ChatGPTProviderEntry | ClaudeCodeProviderEntry,
    Discriminator("api_format"),
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
