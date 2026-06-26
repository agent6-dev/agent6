# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-cutting CLI helpers: run dirs, budget flags, key/root checks."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from agent6.config import (
    AnthropicProviderEntry,
    Config,
)
from agent6.config_layer import (
    resolved_state_dir,
)
from agent6.detect import Environment, detect
from agent6.graph.storage import RunLayout
from agent6.models_cache import list_models
from agent6.paths import (
    effective_user,
    is_root,
    root_optin_enabled,
)
from agent6.pricing import lookup_price
from agent6.run_id import RunIdError, resolve_run_id
from agent6.sandbox import strict_namespaces_work
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


def _add_budget_flags(parser: argparse.ArgumentParser) -> None:
    """Add per-run budget override flags (override ``[budget]`` config)."""
    parser.add_argument(
        "--max-usd",
        type=float,
        default=None,
        metavar="USD",
        help=(
            "Override [budget].best_effort_usd_limit for this run (0 disables)."
            " Passing the flag explicitly refuses to start when the worker model"
            " has no price data, since the limit could not be enforced."
        ),
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Override [budget].max_input_tokens for this run.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Override [budget].max_output_tokens for this run.",
    )


@dataclass(frozen=True, slots=True)
class _BudgetOverrides:
    """Per-run budget overrides parsed from ``--max-*`` flags."""

    max_usd: float | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> _BudgetOverrides:
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


def _state_dir(repo_root: Path) -> Path:
    """The per-repo agent6 state dir (config + run state), out of the workspace.

    Resolved from the global ``[agent6].state_dir`` base (default
    ``$XDG_STATE_HOME/agent6``) plus a per-repo id, so this is cheap and works
    for read-only commands (``runs``/``history``/...) without a full config
    merge.
    """
    return resolved_state_dir(repo_root)


def _runs_dir(repo_root: Path) -> Path:
    """The ``runs/`` directory under the per-repo state dir."""
    return _state_dir(repo_root) / "runs"


def _machines_dir(repo_root: Path) -> Path:
    """The ``machines/`` directory under the per-repo state dir."""
    return _state_dir(repo_root) / "machines"


def resolve_run_layout(repo_root: Path, query: str) -> RunLayout:
    """Resolve a run id (or unique prefix) across BOTH ``runs/`` and ``asks/``,
    returning a ``RunLayout`` with the matching subdir.

    `agent6 run`/`plan` live under ``runs/`` and `agent6 ask` under ``asks/``;
    read-only commands (``runs graph``/``history search``) use this so an ask's
    state is findable too. Raises ``RunIdError`` if no run matches in either.
    """
    state = _state_dir(repo_root)
    for subdir in ("runs", "asks"):
        d = state / subdir
        if not d.is_dir():
            continue
        try:
            rid = resolve_run_id(d, query)
        except RunIdError:
            continue
        return RunLayout(state_dir=state, run_id=rid, subdir=subdir)
    raise RunIdError(f"no run matches {query!r} under {state}/(runs|asks)")


def _check_provider_keys(cfg: Config) -> str | None:
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


def _explicit_usd_flag_error(explicit_usd: float | None, cfg: Config) -> str | None:
    """Refusal message when an explicit --max-usd cannot be enforced.

    The config field is best-effort by name (best_effort_usd_limit), but a
    flag typed on the command line is a promise for THIS run. With no price
    data for the worker model the limit only binds if the provider happens to
    report per-call cost, so refuse up front instead of maybe overspending.
    Called after _check_provider_keys so the models cache (which carries the
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


def _enforce_root_policy(allow_root: bool) -> int | None:
    """Gate running as root behind an explicit opt-in.

    Returns a non-zero exit code (to refuse) when running as root without
    ``--allow-root`` / ``AGENT6_ALLOW_ROOT=1``; returns None to proceed. When
    proceeding as root it prints a loud banner. We deliberately do NOT drop
    privileges: under sudo the LLM's verify/run commands need to run as root
    inside the jail, so the jail, not the process uid, is the boundary.
    """
    if not is_root():
        return None
    if not root_optin_enabled(allow_root):
        print(
            "[agent6] REFUSING to run as root. Running an LLM-driven agent as root"
            " is dangerous. If a task genuinely needs it, re-run with --allow-root"
            " (or set AGENT6_ALLOW_ROOT=1).",
            file=sys.stderr,
        )
        return 2
    user = effective_user()
    who = f" on behalf of {user.name} (uid {user.uid})" if user.via_sudo else ""
    print(
        f"[agent6] WARNING: running as root{who}. The LLM's commands execute as"
        " root inside the jail; files agent6 writes under the repo are chowned"
        " back to you when invoked via sudo. Proceed with care.",
        file=sys.stderr,
    )
    return None


def _start_mcp_manager_if_enabled(cfg: Config) -> MCPManager | None:
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
