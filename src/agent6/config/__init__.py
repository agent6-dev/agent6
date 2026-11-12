# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The config models, file IO and layered resolve.

The models are the package's public API and are re-exported here; the IO and
layering are imported from `agent6.config.io` and `agent6.config.layer`.
"""

from __future__ import annotations

from agent6.config import validation
from agent6.config.container_config import ContainerConfig  # re-export
from agent6.config.git_config import (  # re-export
    GitCommitConfig,
    GitConfig,
    GitWorktreeConfig,
)
from agent6.config.harness_config import (  # re-export
    BudgetConfig,
    CompactionConfig,
    HarnessConfig,
    MetricConfig,
    ParallelConfig,
    PromptConfig,
    ReviewConfig,
    ReviewerConfig,
    ReviewScope,
    RoleModel,
    VerifyConfig,
)
from agent6.config.header_config import HeaderTemplate  # re-export
from agent6.config.model import (  # re-export
    Agent6Section,
    Config,
    load_config,
    validate_config,
)
from agent6.config.model_roles import ModelsConfig  # re-export
from agent6.config.oauth_config import OAuthConfig, OAuthExchangeConfig  # re-export
from agent6.config.provider_config import (  # re-export
    AnthropicProviderEntry,
    ChatGPTProviderEntry,
    ClaudeCodeProviderEntry,
    OpenAIProviderEntry,
    ProviderEntry,
    ResponsesProviderEntry,
    plan_metered,
    provider_name_refusal,
    validate_base_url,
)
from agent6.config.sandbox_config import (  # re-export
    MCPConfig,
    MCPServerEntry,
    SandboxConfig,
    is_cleartext_url,
    is_loopback_url,
    mcp_server_name_refusal,
)
from agent6.config.surface_config import (  # re-export
    MachineConfig,
    NotifyConfig,
    TelemetryConfig,
    WebConfig,
    is_loopback_host,
)
from agent6.kinds import EffortLevel, RoleName  # re-export

ConfigError = validation.ConfigError

__all__ = [
    "Agent6Section",
    "AnthropicProviderEntry",
    "BudgetConfig",
    "ChatGPTProviderEntry",
    "ClaudeCodeProviderEntry",
    "CompactionConfig",
    "Config",
    "ConfigError",
    "ContainerConfig",
    "EffortLevel",
    "GitCommitConfig",
    "GitConfig",
    "GitWorktreeConfig",
    "HarnessConfig",
    "HeaderTemplate",
    "MCPConfig",
    "MCPServerEntry",
    "MachineConfig",
    "MetricConfig",
    "ModelsConfig",
    "NotifyConfig",
    "OAuthConfig",
    "OAuthExchangeConfig",
    "OpenAIProviderEntry",
    "ParallelConfig",
    "PromptConfig",
    "ProviderEntry",
    "ResponsesProviderEntry",
    "ReviewConfig",
    "ReviewScope",
    "ReviewerConfig",
    "RoleModel",
    "RoleName",
    "SandboxConfig",
    "TelemetryConfig",
    "VerifyConfig",
    "WebConfig",
    "is_cleartext_url",
    "is_loopback_host",
    "is_loopback_url",
    "load_config",
    "mcp_server_name_refusal",
    "plan_metered",
    "provider_name_refusal",
    "validate_base_url",
    "validate_config",
]
