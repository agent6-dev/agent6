# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run/resume lifecycle setup shared by the front-end adapters: sandbox env
detection, provider-key preflight, per-invocation budget/sandbox override
values, and MCP server startup."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.config import (
    AnthropicProviderEntry,
    Config,
)
from agent6.models.cache import list_models
from agent6.sandbox import landlock_abi, strict_namespaces_work
from agent6.sandbox.detect import Environment, detect
from agent6.secrets import SecretsError, load_secrets, resolve_api_key
from agent6.tools.mcp_client import MCPConfinement, MCPManager, MCPServerSpec
from agent6.tools.mcp_http import HttpTransport


def detect_env() -> Environment:
    """`detect()` with an authoritative userns re-check via the jail binary.

    `detect.probe_userns_supported` uses `unshare -U -r true`, which
    under-reports on an AppArmor-restricted host (Ubuntu 24.04+) where an AppArmor
    profile grants the *agent6-jail* binary userns but not `/usr/bin/unshare`. When the
    cheap probe says "no" on a Linux host, confirm with the real jail binary so
    a correctly-profiled host gets `strict` instead of silently dropping to
    `hardened`. Every CLI isolation-selection path uses this instead of `detect()`.
    """
    env = detect()
    if env.sandbox_available and not env.userns_supported and strict_namespaces_work():
        return replace(env, userns_supported=True)
    return env


@dataclass(frozen=True, slots=True)
class BudgetOverrides:
    """Per-run budget overrides parsed from ``--max-*`` flags."""

    max_usd: float | None = None
    max_tokens_fallback: int | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BudgetOverrides:
        return cls(
            max_usd=getattr(args, "max_usd", None),
            max_tokens_fallback=getattr(args, "max_tokens_fallback", None),
        )

    def apply(self, cfg: Config) -> Config:
        return cfg.with_budget_overrides(
            max_usd=self.max_usd,
            max_tokens_fallback=self.max_tokens_fallback,
        )


@dataclass(frozen=True, slots=True)
class SandboxOverrides:
    """Per-invocation sandbox/approval overrides from CLI flags.

    ``--dangerously-disable-sandbox`` runs unconfined; ``--auto-approve``
    auto-approves every jailed command; ``--no-commands`` withholds them
    entirely (what `/btw` spawns its side question with). The env setter for the sandbox is read in
    ``detect.resolve_isolation`` (so it also reaches machine subprocesses), so
    ``from_args`` reads only the flags. Flags and env are structurally
    LLM-unreachable."""

    disable_sandbox: bool = False
    auto_approve: bool = False
    no_commands: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SandboxOverrides:
        return cls(
            disable_sandbox=bool(getattr(args, "dangerously_disable_sandbox", False)),
            auto_approve=bool(getattr(args, "auto_approve", False)),
            no_commands=bool(getattr(args, "no_commands", False)),
        )

    def apply(self, cfg: Config) -> Config:
        return cfg.with_sandbox_overrides(
            disable_sandbox=self.disable_sandbox,
            auto_approve=self.auto_approve,
            no_commands=self.no_commands,
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


def start_mcp_manager_if_enabled(
    cfg: Config, *, reporter: Reporter = STDIO_REPORTER
) -> MCPManager | None:
    """Spawn all enabled MCP servers from ``cfg.mcp``. Returns None when
    MCP is disabled or no servers are configured (so callers can skip
    teardown entirely). Each server's startup failure is logged and
    silently skipped; one bad server doesn't poison the run.
    """
    if not cfg.mcp.enabled or not cfg.mcp.servers:
        return None
    configs = [
        MCPServerSpec(
            name=name,
            command=srv.command,
            startup_timeout_s=srv.startup_timeout_s,
            call_timeout_s=srv.call_timeout_s,
            pass_env=srv.pass_env,
            confine=(
                MCPConfinement(
                    read_paths=srv.sandbox.read_paths,
                    write_paths=srv.sandbox.write_paths,
                    require=srv.sandbox.require,
                    network=srv.sandbox.network,
                )
                if srv.sandbox is not None
                else None
            ),
            http=(
                HttpTransport(name=name, url=srv.url, token_env=srv.token_env) if srv.url else None
            ),
        )
        for name, srv in cfg.mcp.servers.items()
        if srv.enabled
    ]
    if not configs:
        return None
    _warn_unconfinable(cfg, reporter=reporter)
    return MCPManager.start(configs, logger=reporter.err)


def _warn_unconfinable(cfg: Config, *, reporter: Reporter) -> None:
    """Say so when a server asked to be confined cannot be, on this kernel.

    Decided HERE, not in the shim: the shim's stderr goes to /dev/null (a
    chatty server would otherwise spam every run), so a warning printed there
    is a warning nobody reads -- and the manager's next line says the server
    started, which reads as success.
    """
    if landlock_abi() > 0:
        return
    for name, srv in sorted(cfg.mcp.servers.items()):
        if srv.enabled and srv.sandbox is not None and not srv.sandbox.require:
            reporter.err(
                f"[agent6] WARNING: MCP server {name!r} asked to be confined, but this"
                " kernel has no Landlock: it is running with your full filesystem."
                " Set its sandbox.require = true to refuse instead."
            )
