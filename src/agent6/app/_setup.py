# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run/resume lifecycle setup shared by the front-end adapters: sandbox env
detection, provider-key preflight, per-invocation budget/sandbox override
values, and MCP server startup."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace

from agent6.config import (
    AnthropicProviderEntry,
    Config,
)
from agent6.models.cache import list_models
from agent6.models.pricing import lookup_price
from agent6.sandbox import strict_namespaces_work
from agent6.sandbox.detect import Environment, detect
from agent6.secrets import SecretsError, load_secrets, resolve_api_key
from agent6.tools.mcp_client import MCPManager


def detect_env() -> Environment:
    """`detect()` with an authoritative userns re-check via the jail binary.

    `detect.probe_userns_supported` uses `unshare -U -r true`, which
    under-reports on an AppArmor-restricted host (Ubuntu 24.04+) where a profile
    grants the *agent6-jail* binary userns but not `/usr/bin/unshare`. When the
    cheap probe says "no" on a Linux host, confirm with the real jail binary so
    a correctly-profiled host gets `strict` instead of silently dropping to
    `hardened`. Every CLI profile-selection path uses this instead of `detect()`.
    """
    env = detect()
    if env.sandbox_available and not env.userns_supported and strict_namespaces_work():
        return replace(env, userns_supported=True)
    return env


@dataclass(frozen=True, slots=True)
class BudgetOverrides:
    """Per-run budget overrides parsed from ``--max-*`` flags."""

    max_usd: float | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BudgetOverrides:
        return cls(
            max_usd=getattr(args, "max_usd", None),
            max_input_tokens=getattr(args, "max_input_tokens", None),
            max_output_tokens=getattr(args, "max_output_tokens", None),
        )

    def apply(self, cfg: Config) -> Config:
        return cfg.with_budget_overrides(
            max_usd=self.max_usd,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
        )


@dataclass(frozen=True, slots=True)
class SandboxOverrides:
    """Per-invocation sandbox/approval overrides from CLI flags.

    ``--dangerously-disable-sandbox`` runs unconfined; ``--auto-approve``
    auto-approves ``run_command``. The env setter for the sandbox is read in
    ``detect.select_profile`` (so it also reaches machine subprocesses), so
    ``from_args`` reads only the flags. Flags and env are structurally
    LLM-unreachable."""

    disable_sandbox: bool = False
    auto_approve: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SandboxOverrides:
        return cls(
            disable_sandbox=bool(getattr(args, "dangerously_disable_sandbox", False)),
            auto_approve=bool(getattr(args, "auto_approve", False)),
        )

    def apply(self, cfg: Config) -> Config:
        return cfg.with_sandbox_overrides(
            disable_sandbox=self.disable_sandbox,
            auto_approve=self.auto_approve,
        )


def check_provider_keys(cfg: Config) -> str | None:
    """Return an error message if any referenced provider has no resolvable key.

    A key may come from the env var named by ``api_key_env`` or from
    ``secrets.toml`` (via ``agent6 connect``). Only providers actually
    referenced by a configured ``[models.<role>]`` are checked.
    OpenAI-compat providers with no key configured at all are skipped
    (unauthenticated local endpoints like Ollama).
    """
    try:
        secrets = load_secrets()
    except SecretsError as exc:
        return str(exc)
    needed = {rm.provider for rm in cfg.models.configured().values()}
    for name, entry in cfg.providers.items():
        if name not in needed:
            continue
        key = resolve_api_key(name, entry.api_key_env, secrets=secrets)
        if key:
            # Opportunistically refresh this provider's models cache (TTL-gated
            # inside, ~1.5s timeout, never raises). This is what keeps model
            # PRICING fresh for budget sizing + cost reports: prices live only
            # in this cache, fetched from the provider's models endpoint.
            list_models(name, entry, key)
            continue
        if entry.token_command or entry.auth_style == "none":
            # Auth is minted by a command (checked at call time) or not required.
            continue
        if isinstance(entry, AnthropicProviderEntry):
            return (
                f"no API key for [providers.{name}] (Anthropic). Run"
                f" `agent6 connect` or set the {entry.api_key_env or 'API key'} env var."
            )
        # OpenAI-compatible: a missing key is only an error if the endpoint
        # clearly expects one; local endpoints legitimately need none, so we
        # do not block here.
    return None


def explicit_usd_flag_error(explicit_usd: float | None, cfg: Config) -> str | None:
    """Refusal message when an explicit --max-usd cannot be enforced.

    The config field is best-effort by name (best_effort_usd_limit), but a
    flag typed on the command line is a promise for THIS run. With no price
    data for the worker model the limit only binds if the provider happens to
    report per-call cost, so refuse up front instead of maybe overspending.
    Called after check_provider_keys so the models cache (which carries the
    pricing) has been refreshed.
    """
    if explicit_usd is None or explicit_usd <= 0:
        return None
    worker = cfg.models.resolve("worker")
    if worker is None or lookup_price(worker.model) is not None:
        return None
    return (
        f"--max-usd {explicit_usd:g} cannot be enforced: no price data for"
        f" {worker.model!r} (its provider's models endpoint publishes none, and"
        " agent6 keeps no price table). Set [budget] best_effort_usd_limit in"
        " config for a best-effort limit, or bound the run with"
        " --max-input-tokens / --max-output-tokens."
    )


def start_mcp_manager_if_enabled(cfg: Config) -> MCPManager | None:
    """Spawn all enabled MCP servers from ``cfg.mcp``. Returns None when
    MCP is disabled or no servers are configured (so callers can skip
    teardown entirely). Each server's startup failure is logged and
    silently skipped; one bad server doesn't poison the run.
    """
    if not cfg.mcp.enabled or not cfg.mcp.servers:
        return None
    configs = [
        (srv.name, srv.command, srv.startup_timeout_s, srv.call_timeout_s)
        for srv in cfg.mcp.servers
        if srv.enabled
    ]
    if not configs:
        return None
    return MCPManager.start(configs, logger=lambda m: print(m, file=sys.stderr))
