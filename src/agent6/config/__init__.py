# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The config models, file IO and layered resolve.

The models are the package's public API and are re-exported here; the IO and
layering are imported from `agent6.config.io` and `agent6.config.layer`.
"""

from __future__ import annotations

from agent6.config._git import GitCommitConfig, GitConfig  # noqa: ICN003  # re-export
from agent6.config._harness import (  # noqa: ICN003  # re-export
    BudgetConfig,
    ContextConfig,
    EffortLevel,
    HarnessConfig,
    MetricConfig,
    PromptConfig,
    ReviewConfig,
    ReviewerConfig,
    ReviewScope,
    RoleModel,
)
from agent6.config._providers import (  # noqa: ICN003  # re-export
    AnthropicProviderEntry,
    ChatGPTProviderEntry,
    ClaudeCodeProviderEntry,
    OpenAIProviderEntry,
    ProviderEntry,
    plan_metered,
    provider_name_refusal,
    validate_base_url,
)
from agent6.config._sandbox import (  # noqa: ICN003  # re-export
    MCPConfig,
    MCPServerEntry,
    SandboxConfig,
    is_cleartext_url,
    is_loopback_url,
    mcp_server_name_refusal,
)
from agent6.config._surfaces import (  # noqa: ICN003  # re-export
    MachineConfig,
    MachineNotifyConfig,
    NotifyConfig,
    ParallelConfig,
    WebConfig,
    WorkspaceConfig,
    is_loopback_host,
)
from agent6.config.model import (  # noqa: ICN003  # re-export
    Agent6Section,
    Config,
    ConfigError,
    ModelsConfig,
    load_config,
    validate_config,
)
from agent6.kinds import RoleName  # noqa: ICN003  # re-export

__all__ = [
    "Agent6Section",
    "AnthropicProviderEntry",
    "BudgetConfig",
    "ChatGPTProviderEntry",
    "ClaudeCodeProviderEntry",
    "Config",
    "ConfigError",
    "ContextConfig",
    "EffortLevel",
    "GitCommitConfig",
    "GitConfig",
    "HarnessConfig",
    "MCPConfig",
    "MCPServerEntry",
    "MachineConfig",
    "MachineNotifyConfig",
    "MetricConfig",
    "ModelsConfig",
    "NotifyConfig",
    "OpenAIProviderEntry",
    "ParallelConfig",
    "PromptConfig",
    "ProviderEntry",
    "ReviewConfig",
    "ReviewScope",
    "ReviewerConfig",
    "RoleModel",
    "RoleName",
    "SandboxConfig",
    "WebConfig",
    "WorkspaceConfig",
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
