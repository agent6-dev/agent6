# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Filesystem path + identity resolution for agent6.

Single source of truth for:

- the global (user-level) config + secrets directory under XDG
  (``$XDG_CONFIG_HOME/agent6`` or ``~/.config/agent6``),
- the per-repo config path (``./.agent6/config.toml``),
- the run-state directory (``./.agent6`` by default, overridable from the
  global config), and
- the *real* operator when agent6 is invoked through ``sudo``, so we read
  the user's config/secrets (not root's) and never leave root-owned files
  scattered in their repository.

Security model (see SECURITY.md):

- Running an LLM-driven agent as root is dangerous. agent6 refuses to run
  as root unless the operator explicitly opts in via ``--allow-root`` or
  ``AGENT6_ALLOW_ROOT=1``, and prints a loud banner either way.
- When ``euid == 0`` and the process was launched through ``sudo`` we
  resolve the invoking user from ``SUDO_UID`` / ``SUDO_GID`` / ``SUDO_USER``
  and ``chown`` anything we create back to them. We do NOT drop privileges
  in-process: the whole point of ``sudo agent6`` is that verify/run
  commands need root, and those run inside the jail as root regardless, so
  juggling euid in the bookkeeping code would be theatre. The jail remains
  the real boundary.
"""

from __future__ import annotations

import contextlib
import os
import pwd
from dataclasses import dataclass
from pathlib import Path

# Environment overrides. All optional; documented in CONFIG.md.
_ALLOW_ROOT_ENV = "AGENT6_ALLOW_ROOT"
_GLOBAL_DIR_ENV = "AGENT6_CONFIG_HOME"  # points at the agent6 global dir itself


@dataclass(frozen=True, slots=True)
class RealUser:
    """The human operator agent6 is acting on behalf of.

    Differs from the process euid only when agent6 runs under ``sudo``:
    there ``uid``/``gid``/``home`` describe the user who typed ``sudo``,
    not root.
    """

    uid: int
    gid: int
    name: str
    home: Path
    via_sudo: bool


def _passwd_home(uid: int) -> Path | None:
    try:
        return Path(pwd.getpwuid(uid).pw_dir)
    except KeyError:
        return None


def effective_user() -> RealUser:
    """Resolve the operator agent6 should act as.

    Under ``sudo`` (euid 0 + ``SUDO_UID`` set) this is the invoking user;
    otherwise it is the current process user.
    """
    euid = os.geteuid()
    sudo_uid = os.environ.get("SUDO_UID")
    if euid == 0 and sudo_uid and sudo_uid.isdigit():
        uid = int(sudo_uid)
        gid_raw = os.environ.get("SUDO_GID", "")
        gid = int(gid_raw) if gid_raw.isdigit() else uid
        name = os.environ.get("SUDO_USER", "") or (
            pwd.getpwuid(uid).pw_name if _passwd_home(uid) else str(uid)
        )
        home = _passwd_home(uid) or Path(os.environ.get("HOME", "/")).resolve()
        return RealUser(uid=uid, gid=gid, name=name, home=home, via_sudo=True)
    uid = os.getuid()
    gid = os.getgid()
    home_env = os.environ.get("HOME")
    home = Path(home_env) if home_env else (_passwd_home(uid) or Path("/"))
    try:
        name = pwd.getpwuid(uid).pw_name
    except KeyError:
        name = str(uid)
    return RealUser(uid=uid, gid=gid, name=name, home=home, via_sudo=False)


def global_config_dir(user: RealUser | None = None) -> Path:
    """The agent6 global config directory.

    Precedence: ``AGENT6_CONFIG_HOME`` > ``$XDG_CONFIG_HOME/agent6`` (only
    when not running through sudo, where root's XDG would be wrong) >
    ``<real-user-home>/.config/agent6``.
    """
    override = os.environ.get(_GLOBAL_DIR_ENV)
    if override:
        return Path(override).expanduser()
    user = user or effective_user()
    if not user.via_sudo:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg:
            return Path(xdg) / "agent6"
    return user.home / ".config" / "agent6"


def global_config_path(user: RealUser | None = None) -> Path:
    return global_config_dir(user) / "config.toml"


def secrets_path(user: RealUser | None = None) -> Path:
    return global_config_dir(user) / "secrets.toml"


_CACHE_DIR_ENV = "AGENT6_CACHE_HOME"  # points at the agent6 cache dir itself


def cache_dir(user: RealUser | None = None) -> Path:
    """The agent6 user cache directory (for non-authoritative, regenerable data).

    Precedence mirrors :func:`global_config_dir`: ``AGENT6_CACHE_HOME`` >
    ``$XDG_CACHE_HOME/agent6`` (only when not running through sudo) >
    ``<real-user-home>/.cache/agent6``. Holds throwaway caches such as the
    provider model-list snapshots used for shell completion; safe to delete.
    """
    override = os.environ.get(_CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    user = user or effective_user()
    if not user.via_sudo:
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            return Path(xdg) / "agent6"
    return user.home / ".cache" / "agent6"


# Default name of the in-repo agent6 directory (config + run state). The
# operator can rename it via ``[agent6].workspace_subdir`` in the GLOBAL
# config only (the per-repo config lives inside this dir, so it cannot name
# the dir that contains it).
DEFAULT_WORKSPACE_SUBDIR = ".agent6"


def validate_workspace_subdir(name: str) -> str:
    """Validate a ``[agent6].workspace_subdir`` value.

    Must be a bare directory name living directly under the repo root: no
    path separators, no ``.``/``..``, not absolute. Returns the name on
    success; raises ``ValueError`` otherwise.
    """
    if not name or name in (".", ".."):
        raise ValueError("workspace_subdir must be a non-empty directory name")
    if "/" in name or "\\" in name or (os.altsep and os.altsep in name):
        raise ValueError(
            f"workspace_subdir must be a bare directory name (no path separators): {name!r}"
        )
    if Path(name).is_absolute():
        raise ValueError(f"workspace_subdir must be relative to the repo, not absolute: {name!r}")
    return name


def agent6_dir(repo_root: Path, workspace_subdir: str | None = None) -> Path:
    """The in-repo agent6 directory holding config + run state.

    ``<repo_root>/.agent6`` by default, or ``<repo_root>/<workspace_subdir>``
    when the global config renames it.
    """
    return repo_root / (workspace_subdir or DEFAULT_WORKSPACE_SUBDIR)


def repo_config_path(repo_root: Path, workspace_subdir: str | None = None) -> Path:
    """The per-repo config file (``<repo>/<agent6-dir>/config.toml``).

    The config lives inside the (possibly renamed) agent6 dir; the dir name
    itself comes from the global config (see ``config_layer``).
    """
    return agent6_dir(repo_root, workspace_subdir) / "config.toml"


def is_root() -> bool:
    return os.geteuid() == 0


def root_optin_enabled(cli_flag: bool) -> bool:
    """True when the operator has explicitly allowed running as root."""
    if cli_flag:
        return True
    val = os.environ.get(_ALLOW_ROOT_ENV, "").strip().lower()
    return val not in ("", "0", "false", "no")


def chown_to_real_user(path: Path, user: RealUser | None = None) -> None:
    """Recursively ``chown`` *path* back to the real operator.

    No-op unless the process is root and was launched through sudo. Uses
    ``lchown`` so we never follow symlinks out of the tree. Best-effort:
    permission errors are swallowed (the file is still usable by root).
    """
    if os.geteuid() != 0:
        return
    user = user or effective_user()
    if not user.via_sudo:
        return
    targets: list[Path] = [path]
    if path.is_dir():
        targets.extend(path.rglob("*"))
    for target in targets:
        # Best effort: a file we cannot chown is still owned by root and
        # readable by root; we never weaken perms to compensate.
        with contextlib.suppress(OSError):
            os.lchown(target, user.uid, user.gid)
