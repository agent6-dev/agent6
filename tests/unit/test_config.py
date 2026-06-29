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
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true

[models.worker]
provider = "anthropic"
model = "claude-x"

[models.reviewer]
provider = "anthropic"
model = "claude-x"

[sandbox]
profile = "auto"
agent_network = "providers"
run_commands = "ask"
protect_git = true

[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
allow_push = false
allow_force = false
allow_history_rewrite = false

[workflow]
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


def test_security_field_defaults_to_safe_value(tmp_path: Path) -> None:
    # `allow_push` is a security field; omitting it must default to the SAFE
    # (disabled) value rather than failing to load (secure-by-default).
    body = _VALID_TOML.replace("allow_push = false\n", "")
    cfg = load_config(_write(tmp_path, body))
    assert cfg.git.allow_push is False


def test_invalid_enum_literal(tmp_path: Path) -> None:
    body = _VALID_TOML.replace('profile = "auto"', 'profile = "lax"')
    with pytest.raises(ConfigError, match=r"sandbox\.profile"):
        load_config(_write(tmp_path, body))


def test_network_defaults_are_secure(tmp_path: Path) -> None:
    body = _VALID_TOML.replace('agent_network = "providers"\n', "")
    cfg = load_config(_write(tmp_path, body))
    assert cfg.sandbox.agent_network == "providers"  # confined to providers
    assert cfg.sandbox.tool_network == "block"  # no jailed-command network


def test_tool_network_allow_requires_agent_open(tmp_path: Path) -> None:
    # run_command runs inside the agent process; it can't reach the network
    # while the agent is confined, so allow requires agent_network=open.
    body = _VALID_TOML.replace(
        'agent_network = "providers"', 'agent_network = "providers"\ntool_network = "allow"'
    )
    with pytest.raises(ConfigError, match="tool_network = 'allow'"):
        load_config(_write(tmp_path, body))


def test_mcp_server_name_rejects_double_underscore(tmp_path: Path) -> None:
    # `__` separates server from tool in the LLM-visible mcp__<server>__<tool>;
    # a server name containing it would break routing, so it's rejected at load.
    body = _VALID_TOML + ('\n[[mcp.servers]]\nname = "bad__name"\ncommand = ["true"]\n')
    with pytest.raises(ConfigError, match="__"):
        load_config(_write(tmp_path, body))


def test_agent_network_local_refuses_allow_urls(tmp_path: Path) -> None:
    # The docstring promises `local` refuses allow_urls; it must be enforced,
    # not silently ignored (there is nothing external to allow-list offline).
    body = _VALID_TOML.replace(
        'agent_network = "providers"',
        'agent_network = "local"\nallow_urls = ["example.com:443"]',
    )
    with pytest.raises(ConfigError, match="agent_network = 'local'"):
        load_config(_write(tmp_path, body))


def test_tool_network_explicit_states_ok_with_confined_agent(tmp_path: Path) -> None:
    # only_explicit_states is exempt: machine tool states are jailed by the
    # host-netns engine, not the confined agent.
    body = _VALID_TOML.replace(
        'agent_network = "providers"',
        'agent_network = "providers"\ntool_network = "only_explicit_states"',
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.sandbox.tool_network == "only_explicit_states"


def test_allow_urls_defaults_empty(tmp_path: Path) -> None:
    # Secure default: no extra egress destinations beyond the providers.
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.sandbox.allow_urls == ()


def test_allow_urls_accepts_host_hostport_and_url(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        "protect_git = true",
        'protect_git = true\nallow_urls = ["example.com", "h.com:8443", "https://api.x.com/v1"]',
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.sandbox.allow_urls == ("example.com", "h.com:8443", "https://api.x.com/v1")


def test_allow_urls_rejects_portless_garbage(tmp_path: Path) -> None:
    body = _VALID_TOML.replace("protect_git = true", 'protect_git = true\nallow_urls = [""]')
    with pytest.raises(ConfigError, match=r"allow_urls"):
        load_config(_write(tmp_path, body))


def test_allow_urls_rejects_bad_port(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        "protect_git = true", 'protect_git = true\nallow_urls = ["h.com:99999"]'
    )
    with pytest.raises(ConfigError, match=r"allow_urls"):
        load_config(_write(tmp_path, body))


def test_extra_read_paths_accepts_clean_absolute(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        "protect_git = true",
        'protect_git = true\nextra_read_paths = ["/opt/toolchain", "/usr/local/go"]',
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.sandbox.extra_read_paths == ("/opt/toolchain", "/usr/local/go")


def test_extra_read_paths_rejects_relative(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        "protect_git = true", 'protect_git = true\nextra_read_paths = ["opt/toolchain"]'
    )
    with pytest.raises(ConfigError, match=r"extra_read_paths"):
        load_config(_write(tmp_path, body))


def test_extra_read_paths_rejects_dotdot_traversal(tmp_path: Path) -> None:
    # FINDING 2: extra_read_paths are bind-mounted read+EXECUTE into the jail, so
    # a `..` component (which could traverse outside the apparent target) must be
    # rejected at config validation even though the path is absolute.
    body = _VALID_TOML.replace(
        "protect_git = true", 'protect_git = true\nextra_read_paths = ["/opt/../etc/shadow"]'
    )
    with pytest.raises(ConfigError, match=r"extra_read_paths.*'\.\.'"):
        load_config(_write(tmp_path, body))


def test_openai_base_url_accepts_http_and_https(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        "[models.worker]",
        '[providers.local]\napi_format = "openai"\nbase_url = "http://localhost:11434/v1"\n\n[models.worker]',
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.providers["local"].base_url == "http://localhost:11434/v1"  # type: ignore[union-attr]


def test_openai_base_url_rejects_schemeless(tmp_path: Path) -> None:
    # The classic paste error: an API key dropped into the base_url field.
    body = _VALID_TOML.replace(
        "[models.worker]",
        '[providers.bad]\napi_format = "openai"\nbase_url = "sk-or-v1-not-a-url"\n'
        "\n[models.worker]",
    )
    with pytest.raises(ConfigError, match=r"base_url"):
        load_config(_write(tmp_path, body))


def test_openai_base_url_rejects_hostless(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        "[models.worker]",
        '[providers.bad]\napi_format = "openai"\nbase_url = "https://"\n\n[models.worker]',
    )
    with pytest.raises(ConfigError, match=r"base_url"):
        load_config(_write(tmp_path, body))


def test_role_temperature_defaults_to_zero(tmp_path: Path) -> None:
    # Finding C / Amp 2: agent6's tool-use loop is a feedback loop;
    # default temperature is pinned to 0.0 so OpenRouter-routed models
    # don't run at their (often high) provider default.
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.models.worker is not None
    assert cfg.models.reviewer is not None
    assert cfg.models.worker.temperature == 0.0
    assert cfg.models.reviewer.temperature == 0.0


def test_role_temperature_override(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"',
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\ntemperature = 0.7',
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.models.worker is not None
    assert cfg.models.reviewer is not None
    assert cfg.models.worker.temperature == 0.7
    assert cfg.models.reviewer.temperature == 0.0  # unchanged


def test_role_temperature_null_passes_through(tmp_path: Path) -> None:
    # Operators who specifically want the provider's default can set None.
    body = _VALID_TOML.replace(
        '[models.reviewer]\nprovider = "anthropic"\nmodel = "claude-x"',
        '[models.reviewer]\nprovider = "anthropic"\nmodel = "claude-x"\ntemperature = nan',
    )
    # nan is rejected by ge/le bounds; the canonical "use provider default"
    # path is to omit the field (default 0.0) or explicitly set null via
    # the python API. Document that nan / out-of-range floats fail loud.
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_role_temperature_out_of_range(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"',
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\ntemperature = 3.0',
    )
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_empty_verify_command_loads_and_is_runnable(tmp_path: Path) -> None:
    # verify_command is OPTIONAL: an empty one loads AND is runnable. `agent6
    # run`/`plan` infer one (or fall back to a gateless run), so require_runnable
    # must NOT block on it -- only providers/model are required.
    body = _VALID_TOML.replace('verify_command = ["true"]', "verify_command = []")
    cfg = load_config(_write(tmp_path, body))
    assert cfg.workflow.verify_command == ()
    cfg.require_runnable("worker")  # does not raise


def test_with_inferred_verify_injects_in_memory(tmp_path: Path) -> None:
    # An inferred verify command is injected in-memory for one run, never
    # mutating the original config.
    body = _VALID_TOML.replace('verify_command = ["true"]', "verify_command = []")
    cfg = load_config(_write(tmp_path, body))
    injected = cfg.with_inferred_verify(("pytest", "-q"))
    assert injected.workflow.verify_command == ("pytest", "-q")
    assert cfg.workflow.verify_command == ()  # original untouched
    assert cfg.with_inferred_verify(()) is cfg  # empty argv is a no-op


def test_verify_timeout_s_defaults_to_600(tmp_path: Path) -> None:
    """Default verify_timeout_s matches jail default (600s)."""
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.workflow.verify_timeout_s == 600.0


def test_verify_timeout_s_overridable(tmp_path: Path) -> None:
    """Bench configs set verify_timeout_s = 30 for fast failure on
    infinite-loop edits."""
    body = _VALID_TOML.replace(
        'verify_command = ["true"]',
        'verify_command = ["true"]\nverify_timeout_s = 30.0',
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.workflow.verify_timeout_s == 30.0


def test_verify_timeout_s_must_be_positive(tmp_path: Path) -> None:
    """0 or negative timeout is rejected (gt=0.0 constraint)."""
    body = _VALID_TOML.replace(
        'verify_command = ["true"]',
        'verify_command = ["true"]\nverify_timeout_s = 0.0',
    )
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_revise_prompt_defaults_off(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.prompt.revise_prompt == "off"


@pytest.mark.parametrize("mode", ["off", "auto", "interactive"])
def test_revise_prompt_modes_load(tmp_path: Path, mode: str) -> None:
    body = _VALID_TOML + f'\n[prompt]\nrevise_prompt = "{mode}"\n'
    cfg = load_config(_write(tmp_path, body))
    assert cfg.prompt.revise_prompt == mode


def test_revise_prompt_invalid_mode_rejected(tmp_path: Path) -> None:
    body = _VALID_TOML + '\n[prompt]\nrevise_prompt = "always"\n'
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_role_routes_to_unconfigured_provider_rejected(tmp_path: Path) -> None:
    body = _VALID_TOML.replace(
        '[models.reviewer]\nprovider = "anthropic"\nmodel = "claude-x"',
        '[models.reviewer]\nprovider = "openrouter"\nmodel = "gpt-x"',
    )
    with pytest.raises(ConfigError, match="openrouter"):
        load_config(_write(tmp_path, body))


def test_no_providers_loads_but_not_runnable(tmp_path: Path) -> None:
    # Secure-by-default: a config with no providers is valid (a global config
    # may define them); require_runnable refuses to start without one.
    body = _VALID_TOML.replace(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        'api_key_env = "ANTHROPIC_API_KEY"\nprompt_caching = true\n',
        "",
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.providers == {}
    with pytest.raises(ConfigError):
        cfg.require_runnable("worker")


def test_openai_provider_with_no_api_key_env_loads(tmp_path: Path) -> None:
    """Ollama-style local endpoint: api_key_env is omitted entirely."""
    body = _VALID_TOML.replace(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        'api_key_env = "ANTHROPIC_API_KEY"\nprompt_caching = true\n',
        '[providers.ollama]\napi_format = "openai"\nbase_url = "http://localhost:11434/v1"\n',
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
        '[providers.anthropic]\napi_format = "anthropic"\n'
        'api_key_env = "ANTHROPIC_API_KEY"\nprompt_caching = true\n',
        (
            '[providers.openai]\napi_format = "openai"\n'
            'api_key_env = "OPENAI_API_KEY"\n\n'
            '[providers.openrouter]\napi_format = "openai"\n'
            'api_key_env = "OPENROUTER_API_KEY"\n'
            'base_url = "https://openrouter.ai/api/v1"\n'
        ),
    )
    body = body.replace(
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"',
        '[models.worker]\nprovider = "openai"\nmodel = "gpt-x"',
    )
    body = body.replace('provider = "anthropic"', 'provider = "openrouter"')
    cfg = load_config(_write(tmp_path, body))
    assert set(cfg.providers) == {"openai", "openrouter"}
    assert cfg.models.worker is not None
    assert cfg.models.reviewer is not None
    assert cfg.models.worker.provider == "openai"
    assert cfg.models.reviewer.provider == "openrouter"


def test_metric_block_loads(tmp_path: Path) -> None:
    body = _VALID_TOML + (
        "\n[workflow.metric]\n"
        'command = ["/usr/bin/python3", "bench.py"]\n'
        'pattern = "CYCLES:\\\\s*(\\\\d+)"\n'
        'goal = "minimize"\n'
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.workflow.metric is not None
    assert cfg.workflow.metric.command == ("/usr/bin/python3", "bench.py")
    assert cfg.workflow.metric.goal == "minimize"


def test_metric_block_absent_is_none(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.workflow.metric is None


def test_metric_goal_invalid(tmp_path: Path) -> None:
    body = _VALID_TOML + (
        '\n[workflow.metric]\ncommand = ["true"]\npattern = "x"\ngoal = "sideways"\n'
    )
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_operational_fields_have_defaults(tmp_path: Path) -> None:
    """Operational fields with safe defaults can be omitted from the TOML.

    Security fields (allow_*, providers.*, sandbox.*, models.*,
    budget.max_*_tokens, workflow.verify_command) still hard-fail when
    missing; the test_missing_required_key family covers those.
    """
    body = """
[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[models.worker]
provider = "anthropic"
model = "claude-x"

[models.reviewer]
provider = "anthropic"
model = "claude-x"

[sandbox]
profile = "auto"
agent_network = "providers"
run_commands = "ask"
protect_git = true

[git]
allow_push = false
allow_force = false
allow_history_rewrite = false

[workflow]
verify_command = ["true"]

[budget]
max_input_tokens = 100000
max_output_tokens = 10000
"""
    cfg = load_config(_write(tmp_path, body))
    # Defaulted fields:
    assert cfg.agent6.config_version == 1
    assert cfg.git.require_clean_worktree is True
    assert cfg.git.auto_stash is False
    assert cfg.git.branch_per_run is True
    assert cfg.git.merge_strategy == "squash"
    assert cfg.workflow.verify_timeout_s == 600.0
    anthro = cfg.providers["anthropic"]
    from agent6.config import AnthropicProviderEntry

    assert isinstance(anthro, AnthropicProviderEntry)
    assert anthro.prompt_caching is True
    assert anthro.http_timeout_s == 600.0


def test_compaction_defaults(tmp_path: Path) -> None:
    # Default is now None == adaptive (sized from the worker model's context
    # window at run construction; see models_cache.compaction_thresholds).
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.context.drop_at_chars is None
    assert cfg.context.summarise_at_chars is None
    assert cfg.context.summary_max_tokens == 2048


def test_compaction_both_or_neither(tmp_path: Path) -> None:
    # A lone threshold is ambiguous (is the other adaptive or fixed?); the
    # loader must reject setting only one.
    body = _VALID_TOML + "\n[context]\ndrop_at_chars = 100000\n"
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body))
    assert "BOTH" in str(exc.value) or "NEITHER" in str(exc.value)


def test_compaction_thresholds_overridable(tmp_path: Path) -> None:
    body = (
        _VALID_TOML
        + "\n[context]\n"
        + "drop_at_chars = 100000\n"
        + "summarise_at_chars = 300000\n"
        + "summary_max_tokens = 1024\n"
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.context.drop_at_chars == 100000
    assert cfg.context.summarise_at_chars == 300000
    assert cfg.context.summary_max_tokens == 1024


def test_compaction_threshold_must_be_positive(tmp_path: Path) -> None:
    body = _VALID_TOML + "\n[context]\ndrop_at_chars = 0\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_compaction_summarise_must_exceed_drop(tmp_path: Path) -> None:
    # Inverted ordering (tier-2 <= tier-1) is the misconfiguration that made
    # tier-2 unreachable; the loader must reject it.
    body = _VALID_TOML + "\n[context]\ndrop_at_chars = 300000\nsummarise_at_chars = 200000\n"
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body))
    assert "must be greater than" in str(exc.value)


def test_with_budget_overrides(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    out = cfg.with_budget_overrides(max_input_tokens=5, max_output_tokens=7)
    assert out.budget.max_input_tokens == 5
    assert out.budget.max_output_tokens == 7
    # Original is unchanged (frozen, returns a copy).
    assert cfg.budget.max_input_tokens == 100000


def test_with_budget_overrides_noop_returns_self(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    assert cfg.with_budget_overrides() is cfg


def test_with_machine_agent_overrides(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    out = cfg.with_machine_agent_overrides(
        model="claude-y",
        thinking="high",
        temperature=0.5,
        max_usd=2.0,
    )
    assert out.models.worker is not None
    assert out.models.worker.model == "claude-y"
    assert out.models.worker.thinking == "high"
    assert out.models.worker.temperature == 0.5
    assert out.budget.best_effort_usd_limit == 2.0
    # Provider name untouched when not overridden.
    assert out.models.worker.provider == "anthropic"


def test_with_machine_agent_overrides_provider(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    out = cfg.with_machine_agent_overrides(provider="anthropic", model="claude-z")
    assert out.models.worker is not None
    assert out.models.worker.provider == "anthropic"
    assert out.models.worker.model == "claude-z"


def _with_openai_provider(block: str) -> str:
    return _VALID_TOML.replace("[models.worker]", f"{block}\n\n[models.worker]")


def test_token_command_parses(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.gw]\napi_format = "openai"\nbase_url = "https://gw.example.com/v1"\n'
        'token_command = ["mint-token", "--json"]\ntoken_command_ttl_s = 60.0'
    )
    cfg = load_config(_write(tmp_path, body))
    entry = cfg.providers["gw"]
    assert entry.token_command == ["mint-token", "--json"]  # type: ignore[union-attr]
    assert entry.token_command_ttl_s == 60.0  # type: ignore[union-attr]


def test_token_command_ttl_defaults_to_300(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.gw]\napi_format = "openai"\ntoken_command = ["mint-token"]'
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.providers["gw"].token_command_ttl_s == 300.0  # type: ignore[union-attr]


def test_token_command_rejects_empty_list(tmp_path: Path) -> None:
    body = _with_openai_provider('[providers.gw]\napi_format = "openai"\ntoken_command = []')
    with pytest.raises(ConfigError, match="token_command"):
        load_config(_write(tmp_path, body))


def test_token_command_rejects_blank_arg(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.gw]\napi_format = "openai"\ntoken_command = ["mint", "  "]'
    )
    with pytest.raises(ConfigError, match="token_command"):
        load_config(_write(tmp_path, body))


def test_token_command_ttl_must_be_positive(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.gw]\napi_format = "openai"\ntoken_command = ["mint"]\ntoken_command_ttl_s = 0'
    )
    with pytest.raises(ConfigError, match="token_command_ttl_s"):
        load_config(_write(tmp_path, body))


_VERTEX_CLAUDE = (
    "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5"
    "/publishers/anthropic/models"
)


def test_deployment_and_auth_defaults(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _VALID_TOML))
    a = cfg.providers["anthropic"]
    assert a.deployment == "direct"
    assert a.auth_style == "x_api_key"  # type: ignore[union-attr]
    assert a.base_url == "https://api.anthropic.com/v1"  # type: ignore[union-attr]


def test_vertex_anthropic_defaults_bearer(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.v]\napi_format = "anthropic"\ndeployment = "vertex"\n'
        f'base_url = "{_VERTEX_CLAUDE}"'
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.providers["v"].deployment == "vertex"
    assert cfg.providers["v"].auth_style == "bearer"  # type: ignore[union-attr]


def test_non_direct_deployment_requires_base_url(tmp_path: Path) -> None:
    body = _with_openai_provider('[providers.v]\napi_format = "anthropic"\ndeployment = "vertex"')
    with pytest.raises(ConfigError, match="base_url is required"):
        load_config(_write(tmp_path, body))


def test_azure_requires_openai_format(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.a]\napi_format = "anthropic"\ndeployment = "azure"\n'
        'base_url = "https://r.openai.azure.com"\nextra_query = { "api-version" = "2024-06-01" }'
    )
    with pytest.raises(ConfigError, match="api_format 'openai'"):
        load_config(_write(tmp_path, body))


def test_azure_requires_api_version_query(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.a]\napi_format = "openai"\ndeployment = "azure"\nbase_url = "https://r.openai.azure.com"'
    )
    with pytest.raises(ConfigError, match="extra_query"):
        load_config(_write(tmp_path, body))


def test_azure_defaults_api_key_header(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.a]\napi_format = "openai"\ndeployment = "azure"\n'
        'base_url = "https://r.openai.azure.com"\nextra_query = { "api-version" = "2024-06-01" }'
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.providers["a"].auth_style == "api_key_header"  # type: ignore[union-attr]


def test_unknown_deployment_rejected(tmp_path: Path) -> None:
    body = _with_openai_provider(
        '[providers.x]\napi_format = "openai"\ndeployment = "bedrock"\nbase_url = "https://x.example.com"'
    )
    with pytest.raises(ConfigError, match="deployment"):
        load_config(_write(tmp_path, body))


def test_explicit_auth_style_preserved(tmp_path: Path) -> None:
    body = _with_openai_provider('[providers.x]\napi_format = "openai"\nauth_style = "none"')
    cfg = load_config(_write(tmp_path, body))
    assert cfg.providers["x"].auth_style == "none"  # type: ignore[union-attr]
