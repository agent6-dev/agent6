# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config package: the models (`model`), file IO (`io`), and the layered
resolve/view/write (`layer`). The Config models are the package's public API and
are re-exported here, so `from agent6.config import Config` keeps working; the IO
and layering live at `agent6.config.io` / `agent6.config.layer`."""

from __future__ import annotations

from agent6.config._git import GitCommitConfig, GitConfig
from agent6.config._providers import (
    AnthropicProviderEntry,
    OpenAIProviderEntry,
    ProviderEntry,
    validate_base_url,
)
from agent6.config._sandbox import (
    MCPConfig,
    MCPServerEntry,
    SandboxConfig,
    mcp_server_name_refusal,
)
from agent6.config.model import (
    Agent6Section,
    BudgetConfig,
    Config,
    ConfigError,
    ContextConfig,
    MachineConfig,
    MachineNotifyConfig,
    MetricConfig,
    ModelsConfig,
    NotifyConfig,
    ParallelConfig,
    PromptConfig,
    ReviewConfig,
    ReviewTier,
    RoleModel,
    RoleName,
    ThinkingLevel,
    WebConfig,
    WorkflowConfig,
    is_loopback_host,
    load_config,
    validate_config,
)

__all__ = [
    "Agent6Section",
    "AnthropicProviderEntry",
    "BudgetConfig",
    "Config",
    "ConfigError",
    "ContextConfig",
    "GitCommitConfig",
    "GitConfig",
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
    "ReviewTier",
    "RoleModel",
    "RoleName",
    "SandboxConfig",
    "ThinkingLevel",
    "WebConfig",
    "WorkflowConfig",
    "is_loopback_host",
    "load_config",
    "mcp_server_name_refusal",
    "validate_base_url",
    "validate_config",
]
