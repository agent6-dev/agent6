# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one description of what a confined child may do.

`dispatch` (commands) and `mcp_client` (servers) both build their policy here.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from agent6.config import Config
from agent6.kinds import IsolationLevel, JailPolicy, NetworkMode
from agent6.memory import DECISIONS_NAME
from agent6.paths import (
    effective_user,
    hidden_paths,
    jail_cache_home,
    linked_worktree_git_dir,
    mkdir_for_real_user,
    private_dirs,
)
from agent6.sandbox.jail import JailUnavailableError
from agent6.sandbox.tool_paths import operator_tool_paths
from agent6.tools._path_safety import Workspace
from agent6.tools._result_format import passthrough_env

# strict's default HOME: the launcher creates it inside the run's private /tmp tmpfs.
JAIL_TMP_HOME = Path("/tmp/agent6-home")  # noqa: S108 - resolved inside the jail


def persistent_jail_home(config: Config, isolation: IsolationLevel) -> Path | None:
    """Return the persistent HOME a jailed command gets, or None where `JAIL_TMP_HOME` applies.

    Only `strict` has a private /tmp for a throwaway HOME; every other level gets
    `agent6.paths.jail_cache_home`, and strict opts into it with `[sandbox].home = "cache"`.
    `jail_policy` creates it and grants it read-write at its real path.
    """
    if isolation == "strict" and config.sandbox.home == "tmp":
        return None
    return jail_cache_home()


def jail_home_refusal(home: Path) -> str | None:
    """Return why a path cannot be the jail's persistent HOME, or None when it can.

    An absent path is fine (`jail_policy` creates it). Refused: a symlink, a non-directory,
    another user's directory, a mode with any group or other bit (a jailed command may chmod
    the dir, and an open one lets another local user plant what the next run consumes; checked
    on every build, never restored), or a path inside an agent6-private dir, which the strict
    mask would re-bind writable. Checked on resolved paths so a symlinked ancestor cannot dodge
    it; the config validator refuses the same for `extra_write_paths`.
    """
    real = home.resolve()
    for private in private_dirs():
        if real.is_relative_to(private.resolve()):
            where = "" if real == home else f" (really {str(real)!r})"
            return (
                f"the jail's HOME {str(home)!r}{where} is inside agent6's private dir"
                f" {str(private)!r} (secrets/state). Point XDG_CACHE_HOME elsewhere."
            )
    try:
        st = os.lstat(home)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"the jail's HOME {str(home)!r} cannot be read: {exc}"
    uid = effective_user().uid
    mode = stat.S_IMODE(st.st_mode)
    fix = "Remove it, or point XDG_CACHE_HOME at a directory of your own."
    if stat.S_ISLNK(st.st_mode):
        problem = (
            f"is a symlink (to {str(home.readlink())!r}): jailed commands would write through it"
        )
    elif not stat.S_ISDIR(st.st_mode):
        problem = "is not a directory"
    elif st.st_uid != uid:
        problem = (
            f"is owned by uid {st.st_uid}, not you (uid {uid}):"
            " another user's directory is never bound"
        )
    elif mode & 0o077:
        problem = (
            f"has mode {mode:04o}, open to other users, who could plant a ~/.gitconfig or"
            " cache content for the next jailed run"
        )
        fix = f"Run: chmod 700 {home}"
    else:
        return None
    return f"the jail's HOME {str(home)!r} {problem}. {fix}"


def workspace_for(config: Config, root: Path, *, memory_dir: Path | None = None) -> Workspace:
    """Build the in-process file boundary for a run.

    The tools are the front door of the file axis (an untrusted model reaches files through
    them with no approval) and the jail is the fence that stops a command working around it.
    Both read the same hidden set (`agent6.paths.hidden_paths`), so the two cannot disagree.
    Derived from config values, never from the isolation level: a degradation must never widen
    what the tools may touch.

    Args:
        config: The run's config.
        root: The workspace root.
        memory_dir: The per-repo memory dir, model-writable by design through the in-process
            tools only; `jail_policy` never mounts it.

    Returns:
        The workspace.
    """
    sb = config.sandbox
    denied = hidden_paths(Path(p) for p in sb.hide_paths)
    writable = tuple(Path(p).resolve() for p in sb.extra_write_paths)
    mem = (memory_dir.resolve(),) if memory_dir is not None else ()
    return Workspace(
        root=root.resolve(),
        denied=tuple(p.resolve() for p in denied),
        # Write implies read, matching the grants the jail mounts.
        read_roots=(*(Path(p).resolve() for p in sb.extra_read_paths), *writable, *mem),
        write_roots=(*writable, *mem),
        exempt=mem,
        read_only=(memory_dir.resolve() / DECISIONS_NAME,) if memory_dir is not None else (),
    )


def resolve_network(
    config: Config, isolation: IsolationLevel, *, override: NetworkMode | None = None
) -> NetworkMode:
    """Return the network a child gets.

    Clamped to what the level can provide: only strict has namespaces, so everywhere else the
    child shares this process's network and the policy says so (preflight has refused an
    explicit setting it cannot honour and warned about an automatic one).

    Args:
        config: The run's config.
        isolation: The resolved isolation level.
        override: A caller's own answer; an MCP server's reachability is the operator's
            per-server choice.

    Returns:
        "host" outside strict or under `network = "host"`, else the override or "session".
    """
    if isolation != "strict":
        return "host"
    if override is not None:
        return override
    return "host" if config.sandbox.network == "host" else "session"


def jail_policy(
    root: Path,
    config: Config,
    isolation: IsolationLevel,
    argv: tuple[str, ...],
    *,
    timeout_s: float | None = None,
    extra_ro_paths: tuple[Path, ...] = (),
    extra_rw_paths: tuple[Path, ...] = (),
    extra_protect_paths: tuple[Path, ...] = (),
    network: NetworkMode | None = None,
    env_base: dict[str, str] | None = None,
    worktree_git_dir: Path | None = None,
) -> JailPolicy:
    """Build the sandbox policy every LLM-influenced argv runs under.

    One owner, so a foreground command, a detached one, the baseline gate re-run and a spawned
    MCP server get the same protect paths, env, tool mounts and memory cap. Callers name only
    what is extra: the system dirs, the operator's tool dirs and a writable HOME are here.

    Args:
        root: The workspace root, the child's cwd.
        config: The run's config.
        isolation: The resolved isolation level.
        argv: The command.
        timeout_s: The child's wall-clock limit; the policy's default when None.
        extra_ro_paths: Paths granted read-only beyond the config's.
        extra_rw_paths: Paths granted read-write beyond the config's.
        extra_protect_paths: Paths re-bound read-only beyond `.git`.
        network: A caller's own network answer; see `resolve_network`.
        env_base: The child's env instead of the passthrough set; the jail defaults (PATH, HOME,
            the uv and bytecode settings) apply either way.
        worktree_git_dir: The repository git dir agent6 recorded when it added the root as a
            fork's linked worktree, granted read-only once the worktree's own `.git` pointer
            still resolves to it. A linked worktree with no record gets no grant.

    Returns:
        The policy.

    Raises:
        JailUnavailableError: The persistent HOME is refused or cannot be created, or the
            worktree's `.git` pointer names anything but the recorded git dir (a rewritten
            pointer grants nothing).
    """
    network = resolve_network(config, isolation, override=network)
    protect_paths: list[Path] = []
    # A writable `.git` lets a jailed command plant a clean filter that the auto-commit runs on
    # the host. Strict re-binds `.git` read-only, which needs a mount namespace; hardened has only
    # Landlock, whose recursive root grant cannot exclude `.git` without costing every top-level
    # write. The in-process edit tools refuse `.git` writes on both levels.
    if config.sandbox.protect_git and isolation == "strict":
        protect_paths.append((root / ".git").resolve())
    protect_paths.extend(extra_protect_paths)
    policy_kwargs: dict[str, Any] = {}
    if timeout_s is not None:
        policy_kwargs["timeout_s"] = timeout_s
    env = passthrough_env() if env_base is None else dict(env_base)
    # HOME is forced like PATH: the operator's HOME is not in the jail, and toolchains need a
    # writable cache root under it (go-build, CARGO_HOME, uv).
    persistent = persistent_jail_home(config, isolation)
    if persistent is not None:
        # Created here so every policy builder gets a HOME that exists: the launcher skips a
        # missing rw path silently. Inspected before creation and again after it.
        refusal = jail_home_refusal(persistent)
        if refusal is None and not os.path.lexists(persistent):
            try:
                mkdir_for_real_user(persistent)
            except OSError as exc:
                raise JailUnavailableError(
                    f"the jail's HOME {str(persistent)!r} cannot be created: {exc}"
                ) from exc
            refusal = jail_home_refusal(persistent)
        if refusal is not None:
            raise JailUnavailableError(refusal)
    env["HOME"] = str(persistent or JAIL_TMP_HOME)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # The jail is offline, so `uv run` must use the venv the operator already synced.
    env.setdefault("UV_NO_SYNC", "1")
    # A controlled PATH plus the operator's tool dirs as RO+exec mounts; else `uv run` dies 127.
    tool_path, tool_mounts = operator_tool_paths()
    env["PATH"] = tool_path
    git_dir: Path | None = None
    if worktree_git_dir is not None:
        pointed = linked_worktree_git_dir(root)
        if pointed != worktree_git_dir:
            raise JailUnavailableError(
                f"the worktree's .git at {root / '.git'} points at {pointed}, not the"
                f" repository git dir agent6 recorded for it ({worktree_git_dir});"
                " a rewritten pointer grants nothing. Restore the pointer, or fork again."
            )
        git_dir = worktree_git_dir
    return JailPolicy(
        cwd=root,
        argv=argv,
        isolation=isolation,
        env=tuple(sorted(env.items())),
        network=network,
        extra_protect_paths=tuple(protect_paths),
        extra_ro_paths=(
            *(Path(p) for p in config.sandbox.extra_read_paths),
            *extra_ro_paths,
            *((git_dir,) if git_dir is not None else ()),
        ),
        extra_rw_paths=(
            *(Path(p) for p in config.sandbox.extra_write_paths),
            *extra_rw_paths,
            *(() if persistent is None else (persistent,)),
        ),
        extra_device_paths=tuple(Path(p) for p in config.sandbox.extra_device_paths),
        tool_paths=tool_mounts,
        hide_paths=tuple(Path(p) for p in config.sandbox.hide_paths),
        memory_limit_mb=config.sandbox.memory_limit_mb,
        **policy_kwargs,
    )
