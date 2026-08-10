# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run/resume lifecycle setup shared by the front-end adapters: sandbox env
detection, provider-key preflight, per-invocation budget/sandbox override
values, and MCP server startup."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import ValidationError

from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.child_env import curated_env
from agent6.config import (
    AnthropicProviderEntry,
    Config,
    ConfigError,
    MCPServerEntry,
)
from agent6.events import EventSink
from agent6.git_ops import set_provider_key_env, set_repo_hook_policy
from agent6.models.cache import list_models
from agent6.sandbox import strict_namespaces_work
from agent6.sandbox.detect import Environment, detect
from agent6.sandbox.jail import SessionNetwork
from agent6.secrets import SecretsError, load_secrets, resolve_api_key
from agent6.tools.mcp_client import MCPManager, MCPServerSpec
from agent6.tools.mcp_http import HttpTransport
from agent6.tools.policy import jail_policy
from agent6.types import IsolationLevel, JailPolicy, NetworkMode


def detect_env() -> Environment:
    """`detect()` with an authoritative strict re-check via the jail binary.

    `detect.probe_userns_supported` runs `unshare -U -r true`, which answers a
    narrower question than "can the jail set up a strict sandbox", and is wrong
    in BOTH directions:

    - It under-reports on an AppArmor-restricted host (Ubuntu 24.04+) where a
      profile grants the *agent6-jail* binary userns but not `/usr/bin/unshare`.
    - It over-reports inside Docker with a relaxed seccomp profile, where
      `unshare` succeeds and the default AppArmor profile then denies the jail's
      `mount`. Measured: every command died with a raw "namespace setup failed:
      EACCES" instead of the run degrading to `hardened`.

    So the real jail binary settles it either way. It costs one short jail spawn
    at startup, cached for the process lifetime.
    """
    env = detect()
    if not env.sandbox_available:
        return env
    works = strict_namespaces_work()
    if works != env.userns_supported:
        return replace(env, userns_supported=works)
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
        try:
            return cfg.with_budget_overrides(
                max_usd=self.max_usd,
                max_tokens_fallback=self.max_tokens_fallback,
            )
        except ValidationError as exc:
            # The schema speaks in config keys; the operator typed a flag. Name
            # what they typed, and refuse the way `config set` refuses rather
            # than escaping to the crash reporter.
            raise ConfigError(self._flag_error(exc)) from exc

    def _flag_error(self, exc: ValidationError) -> str:
        flags = {"max_usd": "--max-usd", "max_tokens_fallback": "--max-tokens-fallback"}
        parts: list[str] = []
        for err in exc.errors():
            field = str(err["loc"][-1]) if err["loc"] else ""
            parts.append(f"{flags.get(field, field)}: {err['msg']}")
        return "; ".join(parts) or str(exc)


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


def apply_git_egress_policy(cfg: Config) -> None:
    """Set how agent6's OWN git ops (run outside the jail) treat repo-controlled
    host code and provider secrets, from the run's config. One call per entry
    point (run, resume, merge, machine), so the policy is set the same way
    everywhere; git_ops itself stays config-free.

    - Repo `.git/hooks/*` fire only under `git.run_repo_hooks` (default off): a
      hook is repo-controlled host code, an RCE vector on an untrusted repo.
    - The configured provider-key env vars are stripped from git's environment:
      git never needs a provider key, and a git subprocess (a credential
      helper, a content driver we could not neutralize) should not inherit one.
    """
    set_repo_hook_policy(cfg.git.run_repo_hooks)
    set_provider_key_env(p.api_key_env for p in cfg.providers.values() if p.api_key_env)


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


def wants_session_network(cfg: Config, isolation: IsolationLevel) -> bool:
    """Whether this run needs its own network: any child that would join one.

    Asked once, before anything spawns, because the network has to exist before
    its first member. Only strict can provide one; elsewhere every child shares
    the host's (preflight has already warned or refused).
    """
    if isolation != "strict":
        return False
    if cfg.sandbox.network != "host":
        return True
    return cfg.mcp.enabled and any(
        srv.enabled and srv.sandbox is not None and srv.sandbox.network == "session"
        for srv in cfg.mcp.servers.values()
    )


def mcp_server_policy(
    cfg: Config, root: Path, isolation: IsolationLevel, srv: MCPServerEntry
) -> JailPolicy | None:
    """The sandbox for one spawned server, or None when the operator opted it
    out with `unconfined = true`.

    The same `jail_policy` a jailed command gets, plus this server's additive
    grants -- so the block names only what is extra and never has to describe
    the interpreter, the tool dirs, or a writable HOME.

    Its env is the CURATED set rather than a command's passthrough: a server
    is third-party code that may log or forward what it was given, so it gets
    the base plus the variables named in `pass_env`, and never the desktop
    addresses (the session bus reaches an unconfined `systemd --user` that
    runs commands on request, which walks straight out of any sandbox).
    """
    sandbox = srv.sandbox
    if sandbox is not None and sandbox.unconfined:
        return None
    read_paths = sandbox.read_paths if sandbox else ()
    write_paths = sandbox.write_paths if sandbox else ()
    # auto and none both mean "a network of its own"; they differ only in what
    # happens when the host cannot provide one (warn vs refuse, which preflight
    # owns). `session` is the run's shared one; `host` is the machine's.
    configured = sandbox.network if sandbox else "auto"
    network: NetworkMode = "none" if configured == "auto" else configured
    return jail_policy(
        root,
        cfg,
        isolation,
        srv.command,
        extra_ro_paths=tuple(Path(p).expanduser() for p in read_paths),
        extra_rw_paths=tuple(Path(p).expanduser() for p in write_paths),
        network=network,
        env_base=curated_env(passthrough=srv.pass_env, desktop=False),
    )


def start_mcp_manager_if_enabled(
    cfg: Config,
    root: Path,
    isolation: IsolationLevel,
    *,
    reporter: Reporter = STDIO_REPORTER,
    events: EventSink | None = None,
    session_net: SessionNetwork | None = None,
) -> MCPManager | None:
    """Spawn all enabled MCP servers from ``cfg.mcp``. Returns None when
    MCP is disabled or no servers are configured (so callers can skip
    teardown entirely). One bad server doesn't poison the run: it is skipped,
    and the run simply does not see its tools.

    A skipped server also becomes an ``mcp.server_unavailable`` journal event
    when *events* is given. Stderr is only visible from a terminal -- under an
    editor it is a log pane, and the operator sees a run quietly missing the
    tools they configured.
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
            policy=mcp_server_policy(cfg, root, isolation, srv),
            http=(
                HttpTransport(name=name, url=srv.url, token_env=srv.token_env) if srv.url else None
            ),
        )
        for name, srv in cfg.mcp.servers.items()
        if srv.enabled
    ]
    if not configs:
        return None
    _warn_servers_that_keep_the_network(cfg, isolation, reporter=reporter)
    manager = MCPManager.start(configs, logger=reporter.err, session_net=session_net)
    if events is not None:
        for failure in manager.failures:
            events.emit("mcp.server_unavailable", server=failure.name, error=failure.error)
    return manager


def _warn_servers_that_keep_the_network(
    cfg: Config, isolation: IsolationLevel, *, reporter: Reporter
) -> None:
    """`network = "auto"` is the secure default and cannot be honoured without
    a network namespace, so where it degrades it says so -- per server, here,
    where the operator is already being told about their servers. An explicit
    `block` refused long before this (check_mcp_network_support)."""
    if isolation == "strict":
        return
    for name, srv in sorted(cfg.mcp.servers.items()):
        if srv.enabled and srv.sandbox is not None and srv.sandbox.network == "auto":
            reporter.err(
                f"[agent6] WARNING: MCP server {name!r} keeps this host's network:"
                f" taking it away needs the network namespace only 'strict' has, and"
                f" this host resolved to {isolation!r}. Set its sandbox.network ="
                " 'block' to refuse rather than run connected."
            )
