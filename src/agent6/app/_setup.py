# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Set up what the run and resume lifecycles share.

Sandbox environment detection, the provider-key preflight, the per-invocation budget and
sandbox overrides, and MCP server startup.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
from collections.abc import Iterable

import pydantic

from agent6 import budget as agent6_budget
from agent6 import child_env, event_log, git_ops, kinds
from agent6 import secrets as agent6_secrets
from agent6.app import reporter as app_reporter
from agent6.config import (
    AnthropicProviderEntry,
    ChatGPTProviderEntry,
    ClaudeCodeProviderEntry,
    Config,
    ConfigError,
    MCPServerEntry,
    ProviderEntry,
    layer,
    parse_seat_spec,
    plan_metered,
)
from agent6.models import cache
from agent6.providers import claude_code
from agent6.sandbox import detect, jail, strict_namespaces_work
from agent6.tools import mcp_client, mcp_http, policy


def detect_env() -> detect.Environment:
    """Detect the host environment, with the jail binary settling whether strict works.

    The `unshare -U -r true` probe is wrong in both directions: an AppArmor profile can grant
    the jail binary userns but not `unshare`, and Docker with a relaxed seccomp profile lets
    `unshare` succeed while AppArmor denies the jail's `mount`. One jail spawn at startup,
    cached for the process, settles it; a binary the kernel cannot execute raises out of it.

    Returns:
        The environment, with `userns_supported` as the jail binary found it.

    Raises:
        JailBinaryError: The jail binary cannot be executed on this host.
    """
    env = detect.detect()
    if not env.sandbox_available:
        return env
    works = strict_namespaces_work()
    if works != env.userns_supported:
        return dataclasses.replace(env, userns_supported=works)
    return env


def budget_tracker(cfg: Config, *, max_usd: float | None = None) -> agent6_budget.BudgetTracker:
    """Return a run's meter from `[budget]`; a `--max-usd` flag overrides the cap."""
    return agent6_budget.BudgetTracker(
        max_usd=cfg.budget.max_usd if max_usd is None else max_usd,
        max_percent=cfg.budget.max_percent,
        allow_paid_credits=cfg.budget.allow_paid_credits,
        max_tokens_fallback=cfg.budget.max_tokens_fallback,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class BudgetOverrides:
    """Hold the per-run budget overrides the `--max-*` flags set.

    Attributes:
        max_usd: The dollar cap.
        max_tokens_fallback: The token cap for an unpriced model.
        max_percent: The plan-window cap.
    """

    max_usd: float | None = None
    max_tokens_fallback: int | None = None
    max_percent: float | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BudgetOverrides:
        """Return the overrides a parsed argv sets."""
        return cls(
            max_usd=getattr(args, "max_usd", None),
            max_tokens_fallback=getattr(args, "max_tokens_fallback", None),
            max_percent=getattr(args, "max_percent", None),
        )

    def apply(self, cfg: Config) -> Config:
        """Return the config with these overrides applied.

        Raises:
            ConfigError: A flag's value fails validation, named by the flag the operator typed.
        """
        try:
            return cfg.with_budget_overrides(
                max_usd=self.max_usd,
                max_tokens_fallback=self.max_tokens_fallback,
                max_percent=self.max_percent,
            )
        except pydantic.ValidationError as exc:
            raise ConfigError(self._flag_error(exc)) from exc

    def argv(self) -> list[str]:
        """Return these overrides as the flags that set them, for a detached resume to carry."""
        out: list[str] = []
        if self.max_usd is not None:
            out += ["--max-usd", str(self.max_usd)]
        if self.max_tokens_fallback is not None:
            out += ["--max-tokens-fallback", str(self.max_tokens_fallback)]
        if self.max_percent is not None:
            out += ["--max-percent", str(self.max_percent)]
        return out

    def _flag_error(self, exc: pydantic.ValidationError) -> str:
        """Return the validation errors worded by flag."""
        flags = {
            "max_usd": "--max-usd",
            "max_tokens_fallback": "--max-tokens-fallback",
            "max_percent": "--max-percent",
        }
        parts: list[str] = []
        for err in exc.errors():
            field = str(err["loc"][-1]) if err["loc"] else ""
            parts.append(f"{flags.get(field, field)}: {err['msg']}")
        return "; ".join(parts) or str(exc)


def override_flags(
    budget: BudgetOverrides | None, sandbox: SandboxOverrides | None, route: kinds.ModelRoute | None
) -> list[str]:
    """Return the flags a detached resume carries so it runs under this invocation's overrides."""
    return [
        *(budget.argv() if budget else []),
        *(sandbox.argv() if sandbox else []),
        *(["--model", route.spec] if route else []),
    ]


def route_text(model: str | kinds.ModelRoute | None) -> str:
    """Return a `--model` as typed, or a recorded pair as `provider/model`; "" for no flag."""
    return model.spec if isinstance(model, kinds.ModelRoute) else (model or "")


def flag_route(
    cfg: Config, mode: str, model: str | kinds.ModelRoute | None
) -> kinds.ModelRoute | None:
    """Return the pair the mode's role runs on when a `--model` set it, else None."""
    if not model:
        return None
    rm = cfg.models.resolve(kinds.session_kind(mode).role)
    return kinds.ModelRoute(rm.provider, rm.model) if rm is not None else None


@dataclasses.dataclass(frozen=True, slots=True)
class SandboxOverrides:
    """Hold the per-invocation sandbox and approval overrides the flags set.

    The sandbox env setter is read in `detect.resolve_isolation`, where it also reaches machine
    subprocesses, so `from_args` reads only the flags. Flags and env are LLM-unreachable.

    Attributes:
        disable_sandbox: `--dangerously-disable-sandbox`, run unconfined.
        auto_approve: `--auto-approve`, approve every jailed command.
        no_commands: `--no-commands`, withhold the command tools.
    """

    disable_sandbox: bool = False
    auto_approve: bool = False
    no_commands: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SandboxOverrides:
        """Return the overrides a parsed argv sets."""
        return cls(
            disable_sandbox=bool(getattr(args, "dangerously_disable_sandbox", False)),
            auto_approve=bool(getattr(args, "auto_approve", False)),
            no_commands=bool(getattr(args, "no_commands", False)),
        )

    def argv(self) -> list[str]:
        """Return these overrides as the flags that set them."""
        flags = (
            ("--dangerously-disable-sandbox", self.disable_sandbox),
            ("--auto-approve", self.auto_approve),
            ("--no-commands", self.no_commands),
        )
        return [flag for flag, on in flags if on]

    def apply(self, cfg: Config) -> Config:
        """Return the config with these overrides applied."""
        return cfg.with_sandbox_overrides(
            disable_sandbox=self.disable_sandbox,
            auto_approve=self.auto_approve,
            no_commands=self.no_commands,
        )


def apply_git_ops_policy(cfg: Config) -> None:
    """Set how agent6's own git ops treat repo-controlled host code and provider secrets.

    One call per entry point, so git_ops itself stays config-free. Repo hooks and content
    drivers are repo-controlled host code, off by default; the provider-key env vars are
    stripped from git's environment, since a credential helper must not inherit one.
    """
    git_ops.set_repo_hook_policy(cfg.git.run_repo_hooks)
    git_ops.set_repo_filter_policy(cfg.git.run_repo_filters)
    child_env.set_provider_key_env(
        p.api_key_env
        for p in cfg.providers.values()
        if not isinstance(p, ClaudeCodeProviderEntry) and p.api_key_env
    )


def session_config(cfg: Config, mode: str, overrides: SandboxOverrides | None = None) -> Config:
    """Return the effective config for a session of the mode.

    The interactive-mode clamp catches a standing `run_commands = "yes"` nobody is watching;
    the operator's flags land last, so an explicit `--auto-approve` stays in force and
    `--no-commands` still pins "no".
    """
    clamped = cfg.with_run_commands_clamped() if kinds.session_kind(mode).clamps_commands else cfg
    return clamped if overrides is None else overrides.apply(clamped)


def load_session_config(
    cwd: pathlib.Path,
    config_path: pathlib.Path | None,
    *,
    mode: str,
    preset: str = "",
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    model: str | kinds.ModelRoute | None = None,
) -> layer.EffectiveConfig:
    """Load the config a session starts or resumes under, the same way at every entry point.

    Args:
        cwd: The workspace.
        config_path: An explicit config file, else the effective one.
        mode: The session mode.
        preset: The preset to apply.
        budget_overrides: The `--max-*` flags.
        sandbox_overrides: The sandbox and approval flags, landing last.
        model: A typed `[provider/]model`, parsed once here, or a recorded pair applied as is.

    Returns:
        The effective config, checked runnable for the mode's role.

    Raises:
        ConfigError: The layers do not load, a flag fails validation, or the role cannot run.
    """
    effective = layer.load_effective(cwd, config_path, preset=preset)
    cfg = effective.config
    apply_git_ops_policy(cfg)
    if budget_overrides is not None:
        cfg = budget_overrides.apply(cfg)
    if model:
        role = kinds.session_kind(mode).role
        route = cfg.model_route(role, model) if isinstance(model, str) else model
        cfg = cfg.with_model_route(role, route)
    cfg = session_config(cfg, mode, sandbox_overrides)
    cfg.require_runnable(kinds.session_kind(mode).role)
    return dataclasses.replace(effective, config=cfg)


def check_provider_keys(cfg: Config, extra_providers: Iterable[str] = ()) -> str | None:
    """Return why a provider the run can reach cannot run, else None.

    Every provider the run can statically reach is checked: the configured roles, the review
    seats, and a machine's per-state pins, so a route found mid-run cannot fail after spend has
    started.

    Args:
        cfg: The resolved config.
        extra_providers: A machine's per-state pins.

    Returns:
        The refusal, or None when every provider can run.
    """
    try:
        secrets = agent6_secrets.load_secrets()
    except agent6_secrets.SecretsError as exc:
        return str(exc)
    needed = {rm.provider for rm in cfg.models.configured().values()}
    for spec in cfg.review.seats:
        _persona, seat_provider, _model = parse_seat_spec(spec)
        if seat_provider:
            needed.add(seat_provider)
    needed.update(p for p in extra_providers if p)
    if absent := sorted(needed - cfg.providers.keys()):
        return (
            f"no [providers.{absent[0]}] entry, but a model route references it"
            " (a role, a review seat, or a machine state pin). Add the provider"
            " or fix the reference."
        )
    for name in sorted(needed):
        if (err := _provider_refusal(name, cfg.providers[name], secrets)) is not None:
            return err
    if "openrouter" not in needed and any(
        rm.model.startswith("claude-")
        and "/" not in rm.model
        and not plan_metered(cfg.providers.get(rm.provider))
        for rm in cfg.models.configured().values()
    ):
        # Bare claude-* ids price through the OpenRouter catalog, which nothing above fetched.
        cache.refresh_pricing_catalog()
    return None


def _provider_refusal(name: str, entry: ProviderEntry, secrets: dict[str, str]) -> str | None:
    """Return why one routed provider cannot run, or None.

    A keyed or local endpoint passes and refreshes its model listing on the way (TTL-gated,
    about 1.5 s): that cache feeds completion, context sizing and the prices the budget meters.
    """
    if isinstance(entry, ClaudeCodeProviderEntry):
        err = claude_code.login_status(entry.binary)
        return f"[providers.{name}]: {err}" if err is not None else None
    if isinstance(entry, ChatGPTProviderEntry):
        if agent6_secrets.load_oauth_tokens(name, secrets=secrets) is None:
            return f"no ChatGPT sign-in stored for [providers.{name}]; run `agent6 connect {name}`."
        cache.list_models(name, entry, None)
        return None
    key = agent6_secrets.resolve_api_key(name, entry.api_key_env, secrets=secrets)
    if key:
        cache.list_models(name, entry, key)
        return None
    if (
        isinstance(entry, AnthropicProviderEntry)
        and not entry.token_command
        and entry.auth_style != "none"
    ):
        return (
            f"no API key for [providers.{name}] (Anthropic). Run"
            f" `agent6 connect` or set the {entry.api_key_env or 'API key'} env var."
        )
    # Minted by a command, not required, or a local OpenAI-compatible endpoint.
    return None


def wants_session_network(cfg: Config, isolation: kinds.IsolationLevel) -> bool:
    """Return whether the run needs its own network: any child would join one.

    Asked once before anything spawns, since the network exists before its first member. Only
    strict can provide one.
    """
    if isolation != "strict":
        return False
    if cfg.sandbox.network != "host":
        return True
    return cfg.mcp.enabled and any(
        srv.enabled and srv.effective_network == "session" for srv in cfg.mcp.servers.values()
    )


def mcp_server_policy(
    cfg: Config,
    root: pathlib.Path,
    isolation: kinds.IsolationLevel,
    srv: MCPServerEntry,
    *,
    readonly: bool = False,
) -> kinds.JailPolicy | None:
    """Return the sandbox for one spawned server, or None when it opted out as unconfined.

    The `jail_policy` a command gets plus the server's additive grants. Its env is the curated
    set, never the desktop addresses: the session bus reaches an unconfined `systemd --user`.

    Args:
        cfg: The resolved config.
        root: The workspace.
        isolation: The isolation level.
        srv: The server's config block.
        readonly: Bind the workspace read-only, for a probe that must not write it.

    Returns:
        The policy, or None for an unconfined server.
    """
    sandbox = srv.sandbox
    if sandbox is not None and sandbox.unconfined:
        return None
    read_paths = sandbox.read_paths if sandbox else ()
    write_paths = sandbox.write_paths if sandbox else ()
    # auto and none both mean a network of its own; preflight owns the warn-or-refuse difference.
    configured = srv.effective_network
    network: kinds.NetworkMode = "none" if configured == "auto" else configured
    return policy.jail_policy(
        root,
        cfg,
        isolation,
        srv.command,
        extra_ro_paths=tuple(pathlib.Path(p).expanduser() for p in read_paths),
        extra_rw_paths=tuple(pathlib.Path(p).expanduser() for p in write_paths),
        extra_protect_paths=(root,) if readonly else (),
        network=network,
        env_base=child_env.curated_env(passthrough=srv.pass_env, desktop=False),
    )


def mcp_server_spec(
    cfg: Config,
    root: pathlib.Path,
    isolation: kinds.IsolationLevel,
    name: str,
    srv: MCPServerEntry,
    *,
    readonly: bool = False,
) -> mcp_client.MCPServerSpec:
    """Return what starting the server takes; every surface spawns it the same way.

    Args:
        cfg: The resolved config.
        root: The workspace.
        isolation: The isolation level.
        name: The server's name.
        srv: The server's config block.
        readonly: Bind the workspace read-only, for the `check mcp` and `mcp connect` probes.

    Returns:
        The spec the manager starts the server from.
    """
    return mcp_client.MCPServerSpec(
        name=name,
        command=srv.command,
        startup_timeout_s=srv.startup_timeout_s,
        call_timeout_s=srv.call_timeout_s,
        pass_env=srv.pass_env,
        # A `url` server is the operator's own process: nothing to confine.
        policy=(
            None if srv.url else mcp_server_policy(cfg, root, isolation, srv, readonly=readonly)
        ),
        http=(
            mcp_http.HttpTransport(
                name=name,
                url=srv.url,
                token_env=srv.token_env,
                httpx_trust_env=srv.httpx_trust_env,
            )
            if srv.url
            else None
        ),
    )


def no_jail_cause(cfg: Config, env: detect.Environment) -> str:
    """Return why this host resolved to isolation `none`."""
    if detect.sandbox_disabled_by_env():
        return "AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1 is set"
    if cfg.sandbox.isolation == "none":
        return "sandbox.isolation = none"
    return detect.degrade_reason(env) or "this host has no jail"


def start_mcp_manager_if_enabled(
    cfg: Config,
    root: pathlib.Path,
    isolation: kinds.IsolationLevel,
    *,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
    events: event_log.EventSink | None = None,
    session_net: jail.SessionNetwork | None = None,
) -> mcp_client.MCPManager | None:
    """Spawn every enabled MCP server; a server that fails is skipped and its tools absent.

    Args:
        cfg: The resolved config.
        root: The workspace.
        isolation: The isolation level.
        reporter: Where a failed server is logged.
        events: The run's event sink; a skipped server becomes an `mcp.server_unavailable`
            event there, since stderr is a log pane under an editor.
        session_net: The session's network namespace, when one exists.

    Returns:
        The manager, or None when MCP is disabled or no server is configured.
    """
    if not cfg.mcp.enabled or not cfg.mcp.servers:
        return None
    configs = [
        mcp_server_spec(cfg, root, isolation, name, srv)
        for name, srv in cfg.mcp.servers.items()
        if srv.enabled
    ]
    if not configs:
        return None
    _warn_servers_that_keep_the_network(cfg, isolation, reporter=reporter)
    manager = mcp_client.MCPManager.start(configs, logger=reporter.err, session_net=session_net)
    if events is not None:
        for failure in manager.failures:
            events.emit("mcp.server_unavailable", server=failure.name, error=failure.error)
    return manager


def _warn_servers_that_keep_the_network(
    cfg: Config, isolation: kinds.IsolationLevel, *, reporter: app_reporter.Reporter
) -> None:
    """Warn per server whose `network = "auto"` degrades to the host's network.

    An explicit `none` or `session` refused before this.
    """
    if isolation == "strict":
        return
    for name, srv in sorted(cfg.mcp.servers.items()):
        if srv.enabled and srv.effective_network == "auto":
            reporter.warn(
                f"MCP server {name!r} keeps this host's network:"
                f" taking it away needs the network namespace only 'strict' has, and"
                f" this host resolved to {isolation!r}. On 'hardened', setting its"
                " sandbox.network = 'none' refuses the run instead of connecting it."
            )
