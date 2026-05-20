# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config loading — TOML to pydantic, every field required, no defaults.

This is a trust boundary (untrusted text -> structured types), so we use pydantic
and surface field-pointing errors.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, ValidationError, model_validator


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or fails validation."""


_BASE_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

ProviderKind = Literal["anthropic", "openai"]
RoleName = Literal["planner", "worker", "critic", "reviewer", "summarizer"]


class AnthropicProviderEntry(BaseModel):
    """`kind = "anthropic"` — the Anthropic Messages endpoint.

    Model names live in `[models.<role>]`, not here — this block only
    carries auth and Anthropic-specific knobs.
    """

    model_config = _BASE_MODEL_CONFIG

    kind: Literal["anthropic"]
    api_key_env: str = Field(min_length=1)
    prompt_caching: bool


class OpenAIProviderEntry(BaseModel):
    """`kind = "openai"` — any OpenAI Chat Completions-compatible endpoint.

    Works against OpenAI itself, OpenRouter, Ollama (`/v1`), vLLM, LM
    Studio, llama.cpp's server, etc. Each [providers.<name>] block is one
    endpoint; configure as many as you want under whatever names you like
    (e.g. `[providers.openai]`, `[providers.openrouter]`,
    `[providers.ollama]`) and reference them by name from `[models.<role>]`.

    `api_key_env` is optional: leave it unset (or point at an unset env var)
    for unauthenticated local endpoints like Ollama. `extra_headers` is
    forwarded verbatim — OpenRouter, for example, asks for `HTTP-Referer`
    and `X-Title`. Header names are sent lowercased by httpx.
    """

    model_config = _BASE_MODEL_CONFIG

    kind: Literal["openai"]
    api_key_env: str | None = Field(
        default=None,
        description=(
            "Env var holding the API key. Set to None / omit for unauthenticated"
            " local endpoints (Ollama, llama.cpp)."
        ),
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        min_length=1,
        description="Base URL of an OpenAI Chat Completions-compatible endpoint.",
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Extra HTTP headers to attach to every request (e.g. OpenRouter's).",
    )


ProviderEntry = Annotated[
    AnthropicProviderEntry | OpenAIProviderEntry,
    Discriminator("kind"),
]


class RoleModel(BaseModel):
    """One role's `(provider, model)` assignment.

    `provider` is the name (TOML table key) of an entry in `[providers.*]`.
    """

    model_config = _BASE_MODEL_CONFIG

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)


class ModelsConfig(BaseModel):
    """Per-role provider + model routing.

    Every sub-agent role gets an explicit `(provider, model)` pair. Any of
    the configured providers may serve any role; there is no fixed
    "primary" / "secondary" split.
    """

    model_config = _BASE_MODEL_CONFIG

    planner: RoleModel
    worker: RoleModel
    critic: RoleModel
    reviewer: RoleModel
    summarizer: RoleModel

    def all(self) -> dict[RoleName, RoleModel]:
        return {
            "planner": self.planner,
            "worker": self.worker,
            "critic": self.critic,
            "reviewer": self.reviewer,
            "summarizer": self.summarizer,
        }


class SandboxConfig(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    profile: Literal["auto", "strict", "hardened"]
    # `provider_only` restricts egress to the host:port of every configured
    # `[providers.*]` block.
    network: Literal["no", "provider_only", "allow"]
    run_commands: Literal["yes", "no", "ask"]


class GitCommitConfig(BaseModel):
    """Optional overrides for the author/committer identity on agent6 commits.

    All three fields default to None, meaning "use whatever the project's
    `git config user.name` / `user.email` already resolves to". The startup
    check in `agent6 run` refuses to proceed if neither an override nor a
    resolvable git-config identity is present — we will not silently commit
    as `(no author) <(none)>`.

    Set `name` / `email` to override the identity on commits made by this
    agent (e.g. to commit as `agent6-bot <bot@example.com>`). Set
    `coauthor` to append a `Co-authored-by:` trailer naming the human
    operator (e.g. `"Alice <alice@example.com>"`).
    """

    model_config = _BASE_MODEL_CONFIG

    name: str | None = None
    email: str | None = None
    coauthor: str | None = None


class GitConfig(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    require_clean_worktree: bool
    auto_stash: bool
    branch_per_run: bool
    commit_strategy: Literal["per_step", "squash", "stage", "none"]
    allow_push: bool
    allow_force: bool
    allow_history_rewrite: bool
    commit: GitCommitConfig = Field(default_factory=GitCommitConfig)


class WorkflowConfig(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    default: Literal["implement"]
    verify_command: tuple[str, ...] = Field(min_length=1)


class BudgetConfig(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)


class Agent6Section(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    config_version: int = Field(ge=1, le=1)


class Config(BaseModel):
    model_config = _BASE_MODEL_CONFIG

    agent6: Agent6Section
    providers: dict[str, ProviderEntry] = Field(min_length=1)
    models: ModelsConfig
    sandbox: SandboxConfig
    git: GitConfig
    workflow: WorkflowConfig
    budget: BudgetConfig

    @model_validator(mode="after")
    def _cross_validate_provider_routing(self) -> Config:
        for role, rm in self.models.all().items():
            if rm.provider not in self.providers:
                known = ", ".join(sorted(self.providers)) or "(none)"
                raise ValueError(
                    f"models.{role}.provider = {rm.provider!r} but"
                    f" [providers.{rm.provider}] is not configured."
                    f" Known providers: {known}."
                )
        return self


def _format_validation_error(err: ValidationError, path: Path) -> str:
    lines = [f"Config validation failed: {path}"]
    for issue in err.errors():
        loc = ".".join(str(part) for part in issue["loc"]) or "<root>"
        lines.append(f"  - {loc}: {issue['msg']} (type={issue['type']})")
    return "\n".join(lines)


def load_config(path: Path) -> Config:
    """Load and strictly validate the TOML config at *path*.

    Raises ConfigError on any problem; never returns a partially valid config.
    """
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file is not valid TOML ({path}): {exc}") from exc
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, path)) from exc
