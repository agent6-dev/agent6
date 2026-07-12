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
from agent6.config.layer import (
    resolved_state_dir,
)
from agent6.models.cache import list_models
from agent6.models.pricing import lookup_price
from agent6.paths import (
    effective_user,
    is_root,
    root_optin_enabled,
)
from agent6.runs.id import RunIdError, list_run_ids
from agent6.runs.layout import RunLayout
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


def _add_sandbox_flags(parser: argparse.ArgumentParser, *, auto_approve: bool = True) -> None:
    """Add the per-invocation sandbox/approval override flags.

    ``--dangerously-disable-sandbox`` runs the agent's commands UNCONFINED on
    the host (equivalent to a one-off ``sandbox.profile = "none"``); the env
    ``AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1`` does the same. ``--auto-approve``
    auto-approves ``run_command`` for this run (``sandbox.run_commands = yes``),
    which stays safe because the command is still jailed; pass ``auto_approve``
    False on commands (like ``machine run``) where approvals are config-driven.
    """
    parser.add_argument(
        "--dangerously-disable-sandbox",
        action="store_true",
        help=(
            "Run the agent's commands UNCONFINED on the host (no Landlock/"
            "seccomp/namespaces). Only for a disposable or already-isolated"
            " machine; the host becomes the only boundary."
        ),
    )
    if auto_approve:
        parser.add_argument(
            "--auto-approve",
            action="store_true",
            help=(
                "Auto-approve run_command for this run instead of prompting"
                " (same as sandbox.run_commands = yes). Safe while sandboxed;"
                " combined with --dangerously-disable-sandbox it hands the agent"
                " unprompted host access."
            ),
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


@dataclass(frozen=True, slots=True)
class _SandboxOverrides:
    """Per-invocation sandbox/approval overrides from CLI flags.

    ``--dangerously-disable-sandbox`` runs unconfined; ``--auto-approve``
    auto-approves ``run_command``. The env setter for the sandbox is read in
    ``detect.select_profile`` (so it also reaches machine subprocesses), so
    ``from_args`` reads only the flags. Flags and env are structurally
    LLM-unreachable."""

    disable_sandbox: bool = False
    auto_approve: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> _SandboxOverrides:
        return cls(
            disable_sandbox=bool(getattr(args, "dangerously_disable_sandbox", False)),
            auto_approve=bool(getattr(args, "auto_approve", False)),
        )

    def apply(self, cfg: Config) -> Config:
        return cfg.with_sandbox_overrides(
            disable_sandbox=self.disable_sandbox,
            auto_approve=self.auto_approve,
        )


def sgr(text: str, code: str) -> str:
    """Wrap *text* in an ANSI style, tty only, so piped output stays plain.
    The one place the CLI's faded/bold hints are styled (was duplicated in
    skills_cmds / memory_cmds)."""
    return f"\x1b[{code}m{text}\x1b[0m" if sys.stdout.isatty() else text


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


# Every run-style bucket a listing spans: `run`/`plan` under runs/, `ask` under
# asks/, `machine create` authoring logs under machine-drafts/. Kept in one place
# so "most recent" and "search all history" match what `agent6 runs` lists.
RUN_BUCKETS: tuple[str, ...] = ("runs", "asks", "machine-drafts")


def all_run_dirs(repo_root: Path) -> list[Path]:
    """Every run directory across all RUN_BUCKETS. So latest-run resolution and
    history search cover asks and machine-drafts, not just runs/ (a bare `attach`
    or `history search` right after an `ask` must find that ask)."""
    state = _state_dir(repo_root)
    dirs: list[Path] = []
    for subdir in RUN_BUCKETS:
        bucket = state / subdir
        if bucket.is_dir():
            dirs.extend(p for p in bucket.iterdir() if p.is_dir())
    return dirs


def _machines_dir(repo_root: Path) -> Path:
    """The ``machines/`` directory under the per-repo state dir."""
    return _state_dir(repo_root) / "machines"


def resolve_run_layout(repo_root: Path, query: str) -> RunLayout:
    """Resolve a run id (or unique prefix) across every run-style bucket --
    ``runs/``, ``asks/``, and ``machine-drafts/`` -- returning a ``RunLayout``
    with the matching subdir.

    `agent6 run`/`plan` live under ``runs/``, `agent6 ask` under ``asks/``, and
    `machine create` authoring logs under ``machine-drafts/``; read-only
    commands (``runs show``/``watch``/``history search``) use this so anything
    a listing shows is also inspectable by id. Raises ``RunIdError`` if no run
    matches in any bucket.
    """
    if not query:
        raise RunIdError("empty run id")
    state = _state_dir(repo_root)
    exact: list[tuple[str, str]] = []
    prefix: list[tuple[str, str]] = []
    for subdir in ("runs", "asks", "machine-drafts"):
        d = state / subdir
        if not d.is_dir():
            continue
        for rid in list_run_ids(d):
            if rid == query:
                exact.append((subdir, rid))
            elif rid.startswith(query):
                prefix.append((subdir, rid))
    if len(exact) == 1:
        subdir, rid = exact[0]
        return RunLayout(state_dir=state, run_id=rid, subdir=subdir)
    if len(exact) > 1:
        preview = ", ".join(f"{subdir}/{rid}" for subdir, rid in sorted(exact)[:5])
        raise RunIdError(
            f"run id {query!r} is ambiguous ({len(exact)} exact matches): {preview}",
            ambiguous=True,
        )
    if len(prefix) == 1:
        subdir, rid = prefix[0]
        return RunLayout(state_dir=state, run_id=rid, subdir=subdir)
    if len(prefix) > 1:
        preview = ", ".join(f"{subdir}/{rid}" for subdir, rid in sorted(prefix)[:5])
        raise RunIdError(
            f"run id {query!r} is ambiguous ({len(prefix)} matches): {preview}",
            ambiguous=True,
        )
    raise RunIdError(f"no run matches {query!r} under {state}/(runs|asks|machine-drafts)")


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
