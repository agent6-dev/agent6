# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Filesystem path + identity resolution for agent6.

Single source of truth for:

- the global (user-level) config + secrets directory under XDG
  (``$XDG_CONFIG_HOME/agent6`` or ``~/.config/agent6``),
- the per-repo config path (``<state_dir>/config.toml``, out of the repo),
- the run-state directory (``$XDG_STATE_HOME/agent6/<repo-id>`` by default,
  overridable from the global config), and
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
import hashlib
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


def _passwd_entry(uid: int) -> pwd.struct_passwd | None:
    try:
        return pwd.getpwuid(uid)
    except KeyError:
        return None


def _passwd_home(uid: int) -> Path | None:
    entry = _passwd_entry(uid)
    return Path(entry.pw_dir) if entry else None


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
        # One passwd lookup, not three; and use the entry's existence (not "does
        # its home dir resolve") to decide whether we have a real name/home.
        entry = _passwd_entry(uid)
        name = os.environ.get("SUDO_USER", "") or (entry.pw_name if entry else str(uid))
        home = Path(entry.pw_dir) if entry else Path(os.environ.get("HOME", "/")).resolve()
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


# Per-repo agent6 state lives OUT of the workspace, under an XDG state base,
# namespaced by a per-repo id. Nothing the agent runs (a jailed command on its
# own cwd) can reach it, and a checkout never carries an `.agent6/` dir.
_STATE_DIR_ENV = "AGENT6_STATE_HOME"  # points at the agent6 state BASE dir itself


def state_base(user: RealUser | None = None) -> Path:
    """The agent6 state BASE directory (per-repo config + run state).

    Precedence: ``AGENT6_STATE_HOME`` > ``$XDG_STATE_HOME/agent6`` (only when
    not running through sudo, where root's XDG would be wrong) >
    ``<real-user-home>/.local/state/agent6``. Each repo gets ``<base>/<repo-id>/``.
    """
    override = os.environ.get(_STATE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    user = user or effective_user()
    if not user.via_sudo:
        xdg = os.environ.get("XDG_STATE_HOME")
        if xdg:
            return Path(xdg) / "agent6"
    return user.home / ".local" / "state" / "agent6"


def repo_id(repo_root: Path) -> str:
    """Stable per-repo id: ``<folder>-<12 hex of sha256(canonical path)>``.

    Keyed on the resolved path, so two checkouts at different paths get
    separate state and never collide. Moving or renaming a checkout changes
    its id: its prior runs are simply not found from the new path.
    """
    real = repo_root.resolve()
    return f"{real.name}-{hashlib.sha256(str(real).encode('utf-8')).hexdigest()[:12]}"


def state_dir(repo_root: Path, base_override: str | None = None) -> Path:
    """The per-repo agent6 state directory (``<base>/<repo-id>``).

    ``base_override`` is the global ``[agent6].state_dir`` (an absolute base
    path); when set it replaces the XDG base. ``repo_id`` is always appended,
    so one global base namespaces every repo without collision.
    """
    base = Path(base_override).expanduser() if base_override else state_base()
    return base / repo_id(repo_root)


def repo_config_path(repo_root: Path, base_override: str | None = None) -> Path:
    """The per-repo config file (``<state_dir>/config.toml``), out of the repo."""
    return state_dir(repo_root, base_override) / "config.toml"


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
