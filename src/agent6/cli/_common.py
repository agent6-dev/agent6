# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-cutting CLI helpers: run dirs, budget flags, key/root/gitignore checks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from agent6.config import (
    AnthropicProviderEntry,
    Config,
)
from agent6.config_layer import (
    resolved_agent6_dir,
)
from agent6.detect import Environment, detect
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    commit_paths,
    is_git_repo,
)
from agent6.paths import (
    effective_user,
    is_root,
    root_optin_enabled,
)
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
        help="Override [budget].max_usd for this run (dollar cap; 0 disables).",
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


def _agent6_dir(repo_root: Path) -> Path:
    """The in-repo agent6 dir (config + run state), honoring the global rename.

    The directory name comes solely from the global config's
    ``[agent6].workspace_subdir`` (default ``.agent6``), so this is cheap and
    works for read-only commands (``watch``/``history``/...) without a full
    config merge.
    """
    return resolved_agent6_dir(repo_root)


def _runs_dir(repo_root: Path) -> Path:
    """The ``runs/`` directory under the resolved agent6 dir."""
    return _agent6_dir(repo_root) / "runs"


def _machines_dir(repo_root: Path) -> Path:
    """The ``machines/`` directory under the resolved agent6 dir."""
    return _agent6_dir(repo_root) / "machines"


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


def _enforce_root_policy(allow_root: bool) -> int | None:
    """Gate running as root behind an explicit opt-in.

    Returns a non-zero exit code (to refuse) when running as root without
    ``--allow-root`` / ``AGENT6_ALLOW_ROOT=1``; returns None to proceed. When
    proceeding as root it prints a loud banner. We deliberately do NOT drop
    privileges: under sudo the LLM's verify/run commands need to run as root
    inside the jail, so the jail — not the process uid — is the boundary.
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


def _ensure_agent6_gitignored(
    root: Path,
    *,
    agent6_dir: Path,
    identity: CommitIdentity | None = None,
    logger: Callable[[str], None] = print,
) -> None:
    """Make sure the agent6 dir is in `.gitignore` before we write under it.

    `agent6 run` and `agent6 plan` create ``<agent6-dir>/runs/<id>/`` early in
    startup (transcripts, run log). If the project's `.gitignore` doesn't
    already exclude the agent6 dir, those files become untracked content and
    the `require_clean_worktree` pre-flight check then refuses to proceed — a
    self-DoS that confuses first-time users.

    Append the entry, then commit `.gitignore` immediately so the worktree
    stays clean for the subsequent dirty-tree check. We commit on the user's
    current branch *before* `branch_per_run` cuts the agent's working branch,
    so this single housekeeping commit lands on the parent branch where it
    belongs.
    """
    gitignore = root / ".gitignore"
    name = agent6_dir.name
    entry = f"{name}/"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
    if any(line.strip() in {entry, f"/{entry}", name} for line in existing.splitlines()):
        return
    suffix = "" if existing.endswith("\n") or not existing else "\n"
    gitignore.write_text(
        existing + suffix + "# agent6 run state (transcripts, run logs, graph)\n" + entry + "\n",
        encoding="utf-8",
    )
    # Commit on the current branch only if we are inside a git repo; otherwise
    # writing the file is enough (the workflow's git pre-flight will refuse
    # to proceed anyway, with a clearer error than "dirty worktree").
    try:
        if is_git_repo(root):
            commit_paths(
                root,
                "chore: ignore .agent6/ run state (added by agent6)",
                (".gitignore",),
                identity=identity,
            )
            logger(f"[agent6] added {entry!r} to {gitignore.name} (committed)")
            return
    except GitError as exc:
        logger(f"[agent6] WARNING: wrote {entry!r} to .gitignore but commit failed: {exc}")
        return
    logger(f"[agent6] added {entry!r} to {gitignore.name}")


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
