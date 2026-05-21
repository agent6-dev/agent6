# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.config — strict pydantic loading from TOML."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import ConfigError, load_config

_VALID_TOML = """
[agent6]
config_version = 1

[providers.anthropic]
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true

[models.planner]
provider = "anthropic"
model = "claude-x"

[models.worker]
provider = "anthropic"
model = "claude-x"

[models.critic]
provider = "anthropic"
model = "claude-x"

[models.reviewer]
provider = "anthropic"
model = "claude-x"

[models.summarizer]
provider = "anthropic"
model = "claude-x"

[sandbox]
profile = "auto"
network = "provider_only"
run_commands = "ask"
protect_git = true
protect_agent6 = true

[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
commit_strategy = "per_step"
allow_push = false
allow_force = false
allow_history_rewrite = false

[workflow]
default = "implement"
verify_command = ["true"]
[budget]
max_input_tokens = 100000
max_output_tokens = 10000
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "agent6.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_loads_valid_config(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.agent6.config_version == 1
    assert cfg.sandbox.profile == "auto"
    assert cfg.workflow.verify_command == ("true",)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_extra_key_forbidden(tmp_path: Path) -> None:
    body = _VALID_TOML.replace("[git]", "[git]\nextra_key = true")
    with pytest.raises(ConfigError, match="extra"):
        load_config(_write(tmp_path, body))


def test_missing_required_key(tmp_path: Path) -> None:
    body = _VALID_TOML.replace("config_version = 1\n", "")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_invalid_enum_literal(tmp_path: Path) -> None:
    body = _VALID_TOML.replace('profile = "auto"', 'profile = "lax"')
    with pytest.raises(ConfigError, match=r"sandbox\.profile"):
        load_config(_write(tmp_path, body))


def test_verify_command_min_length(tmp_path: Path) -> None:
    body = _VALID_TOML.replace('verify_command = ["true"]', "verify_command = []")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_role_routes_to_unconfigured_provider_rejected(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        '[models.reviewer]\nprovider = "anthropic"\nmodel = "claude-x"',
        '[models.reviewer]\nprovider = "openrouter"\nmodel = "gpt-x"',
    )
    with pytest.raises(ConfigError, match="openrouter"):
        load_config(_write(tmp_path, body))


def test_no_providers_configured_rejected(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        '[providers.anthropic]\nkind = "anthropic"\n'
        'api_key_env = "ANTHROPIC_API_KEY"\nprompt_caching = true\n',
        "",
    )
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_openai_provider_with_no_api_key_env_loads(tmp_path: Path) -> None:
    """Ollama-style local endpoint: api_key_env is omitted entirely."""
    body = _VALID_TOML.replace(
        '[providers.anthropic]\nkind = "anthropic"\n'
        'api_key_env = "ANTHROPIC_API_KEY"\nprompt_caching = true\n',
        '[providers.ollama]\nkind = "openai"\nbase_url = "http://localhost:11434/v1"\n',
    )
    # Re-route every role to the ollama provider since anthropic is now gone.
    body = body.replace('provider = "anthropic"', 'provider = "ollama"')
    cfg = load_config(_write(tmp_path, body))
    ollama = cfg.providers["ollama"]
    from agent6.config import OpenAIProviderEntry

    assert isinstance(ollama, OpenAIProviderEntry)
    assert ollama.api_key_env is None


def test_multiple_openai_providers_load(tmp_path: Path) -> None:
    """Both OpenAI and OpenRouter side-by-side, distinct keys, routed per role."""
    body = _VALID_TOML.replace(
        '[providers.anthropic]\nkind = "anthropic"\n'
        'api_key_env = "ANTHROPIC_API_KEY"\nprompt_caching = true\n',
        (
            '[providers.openai]\nkind = "openai"\n'
            'api_key_env = "OPENAI_API_KEY"\n\n'
            '[providers.openrouter]\nkind = "openai"\n'
            'api_key_env = "OPENROUTER_API_KEY"\n'
            'base_url = "https://openrouter.ai/api/v1"\n'
        ),
    )
    body = body.replace(
        '[models.planner]\nprovider = "anthropic"\nmodel = "claude-x"',
        '[models.planner]\nprovider = "openai"\nmodel = "gpt-x"',
    )
    body = body.replace('provider = "anthropic"', 'provider = "openrouter"')
    cfg = load_config(_write(tmp_path, body))
    assert set(cfg.providers) == {"openai", "openrouter"}
    assert cfg.models.planner.provider == "openai"
    assert cfg.models.worker.provider == "openrouter"
