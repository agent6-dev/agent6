# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[providers.*]` model: one entry per endpoint, discriminated by wire format."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Discriminator, Field, field_validator, model_validator

from agent6.config._base import MODEL_CONFIG

ApiFormat = Literal["anthropic", "openai"]
Deployment = Literal["direct", "vertex", "azure"]
AuthStyle = Literal["x_api_key", "bearer", "api_key_header", "none"]


def validate_base_url(url: str) -> None:
    """Reject a ``[providers.*].base_url`` that is not an http(s) URL with a host.

    A provider's
    ``base_url`` is the host+path prefix the HTTP client posts to (the
    deployment profile appends ``/chat/completions``, ``/messages``, etc.), so
    it must carry an explicit
    ``http://`` / ``https://`` scheme and a host. The common paste error this
    catches is dropping an API key (or a bare host) into the field, which would
    otherwise be accepted and only fail much later as an opaque HTTP error.
    """
    try:
        parts = urlsplit(url)
        port = parts.port  # urlsplit raises ValueError on an out-of-range port
    except ValueError as exc:
        raise ValueError(f"invalid base_url {url!r}: {exc}") from exc
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"base_url {url!r} must start with http:// or https://")
    if not parts.hostname:
        raise ValueError(f"base_url {url!r} has no host")
    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"base_url {url!r} has an invalid port")


_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _default_base_url(api_format: str, deployment: str) -> str | None:
    """Default ``base_url`` for a (format, deployment), or None if required.

    Only the ``direct`` deployment has a sensible fixed endpoint; vertex/azure
    (and future bedrock) carry project/resource/region in the URL, so the
    operator must supply ``base_url``.
    """
    if deployment != "direct":
        return None
    return _ANTHROPIC_DEFAULT_BASE_URL if api_format == "anthropic" else _OPENAI_DEFAULT_BASE_URL


def _default_auth_style(api_format: str, deployment: str) -> str:
    """Default ``auth_style`` for a (format, deployment)."""
    if deployment == "azure":
        return "api_key_header"
    if deployment == "vertex":
        return "bearer"
    return "x_api_key" if api_format == "anthropic" else "bearer"


_API_FORMAT_DESCRIPTION = (
    '`"anthropic"` (Messages) or `"openai"` (Chat Completions: OpenAI, OpenRouter, '
    "Ollama, vLLM, LM Studio, llama.cpp, Gemini's OpenAI endpoint, …)."
)


class _ProviderBase(BaseModel):
    """Transport + auth fields shared by every provider, independent of format.

    Three orthogonal concerns: ``api_format`` (the discriminator) selects the
    wire dialect; ``deployment`` selects the URL /
    model-placement profile; and the auth fields (``auth_style`` + a static
    ``api_key_env`` or a refreshable ``token_command``) select the credential.
    They compose freely -- e.g. Claude-on-Vertex and Gemini-on-Vertex differ
    only in ``api_format`` (both ``deployment = "vertex"``). ``base_url`` and
    ``auth_style`` default from (api_format, deployment) in ``_fill_defaults`` so
    a minimal entry (just ``api_format``) is fully usable. Each block is
    one endpoint; configure as many as you like under any names and reference
    them from ``[models.*]``.
    """

    model_config = MODEL_CONFIG

    # Declared on the base only to fix the FIELD ORDER: a redeclared field
    # keeps its base position, so api_format leads every subclass's
    # model_fields (the docs table and `config show` print that order). Each
    # subclass narrows it to its own literal, which is what discriminates.
    api_format: ApiFormat
    deployment: Deployment = Field(
        default="direct",
        description=(
            '`"direct"`, `"vertex"`, or `"azure"` (`openai` only). Selects URL shape + '
            "model/version placement."
        ),
    )
    # Resolved by _fill_defaults from (api_format, deployment) when omitted;
    # never empty post-validation. The host also feeds the egress allow-list.
    base_url: str = Field(
        default="",
        description=(
            "Endpoint host + path prefix; required for vertex/azure. Its host is the only network "
            "destination the agent dials for this provider."
        ),
    )
    # Auth header style; defaults from (api_format, deployment) in _fill_defaults.
    auth_style: AuthStyle = Field(
        default="bearer",
        description=(
            '`"x_api_key"`, `"bearer"`, `"api_key_header"` (Azure), or `"none"` (local). '
            "Rarely set by hand."
        ),
    )
    # Static key: env var name (falls back to secrets.toml by provider name).
    # Secrets live here, never in base_url/extra_headers/extra_query.
    api_key_env: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Env var holding the key (wins over `secrets.toml`). Omit for `agent6 connect` keys or "
            "unauthenticated local endpoints."
        ),
    )
    token_command: list[str] | None = Field(
        default=None,
        description=(
            "argv that prints a short-lived bearer to stdout; re-run on TTL and once on "
            "`401`/`403`. Wins over `api_key_env`."
        ),
    )
    token_command_ttl_s: float = Field(
        gt=0.0,
        default=300.0,
        description="Seconds to cache `token_command` output.",
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Extra HTTP headers on every request. Not for secrets.",
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider-specific JSON merged into every request body (load-bearing keys filtered), "
            "e.g. OpenRouter routing options."
        ),
    )
    extra_query: dict[str, str] = Field(
        default_factory=dict,
        description="Extra URL query params (e.g. Azure's `api-version`).",
    )
    # Per-HTTP-call read/write budget in seconds; the connect phase is bounded
    # separately (providers._transport.CONNECT_TIMEOUT_S) so a blackholed
    # connect fails in seconds, not this. Default 600s streams a long response;
    # lower it on benches that should fail fast.
    http_timeout_s: float = Field(
        gt=0.0,
        default=600.0,
        description="Per-HTTP-call timeout (read/write; connect is bounded at 20s).",
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        fmt = data.get("api_format")
        dep = data.get("deployment", "direct")
        if fmt == "anthropic" and dep == "azure":
            raise ValueError("deployment 'azure' requires api_format 'openai'")
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
        if v:
            validate_base_url(v)
        return v

    @field_validator("token_command")
    @classmethod
    def _check_token_command(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and (not v or any(not arg.strip() for arg in v)):
            raise ValueError("token_command must be a non-empty argv of non-empty strings")
        return v


class AnthropicProviderEntry(_ProviderBase):
    """``api_format = "anthropic"`` -- the Anthropic Messages wire format.

    ``deployment = "direct"`` (default) hits api.anthropic.com; ``"vertex"``
    is Claude-on-Vertex (model id in the URL, ``anthropic_version`` in the body,
    a Google-OAuth bearer via ``token_command``).
    """

    # The narrowing override is sound: the model is frozen, so the attribute
    # can never be written back through the wider base type.
    api_format: Literal["anthropic"] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        Field(description=_API_FORMAT_DESCRIPTION)
    )
    prompt_caching: bool = Field(
        default=True,
        description=(
            "(`anthropic`) Prompt caching: system prompt, tools, and the growing conversation "
            "re-read at 0.1x input price."
        ),
    )


class OpenAIProviderEntry(_ProviderBase):
    """``api_format = "openai"`` -- any OpenAI Chat Completions wire format.

    ``deployment = "direct"`` works against OpenAI, OpenRouter, Ollama, vLLM,
    LM Studio, llama.cpp, Gemini's OpenAI-compatible endpoint, GitHub Copilot,
    etc.; ``"vertex"`` is Gemini's Vertex OpenAPI endpoint; ``"azure"`` is Azure
    OpenAI (deployment-name in the URL, api-version query param, ``api-key``
    header).
    """

    api_format: Literal["openai"] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        Field(description=_API_FORMAT_DESCRIPTION)
    )


ProviderEntry = Annotated[
    AnthropicProviderEntry | OpenAIProviderEntry,
    Discriminator("api_format"),
]
