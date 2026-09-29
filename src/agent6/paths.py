# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Resolve agent6's paths and the operator's identity.

The one owner of the global config and secrets directory, the per-repo config
path, the state directory and the real operator under `sudo`: there the
invoking user is read from `SUDO_UID`, `SUDO_GID` and `SUDO_USER` and what
agent6 creates is chowned back to them. Privileges are never dropped
in-process, since the jail is the boundary; see docs/security.md.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import pathlib
import pwd
from collections.abc import Iterable

_ALLOW_ROOT_ENV = "AGENT6_ALLOW_ROOT"


@dataclasses.dataclass(frozen=True, slots=True)
class RealUser:
    """The operator agent6 acts for.

    Attributes:
        uid: The operator's uid; under `sudo`, the invoking user's, not root's.
        gid: The operator's gid.
        name: The operator's login name, or the uid as text.
        home: The operator's home.
        via_sudo: The process runs as root through `sudo`.
    """

    uid: int
    gid: int
    name: str
    home: pathlib.Path
    via_sudo: bool


def _passwd_entry(uid: int) -> pwd.struct_passwd | None:
    """Return the passwd entry for a uid, or None when there is none."""
    try:
        return pwd.getpwuid(uid)
    except KeyError:
        return None


def _passwd_home(uid: int) -> pathlib.Path | None:
    """Return the passwd home for a uid, or None when there is no entry."""
    entry = _passwd_entry(uid)
    return pathlib.Path(entry.pw_dir) if entry else None


def effective_user() -> RealUser:
    """Return the operator agent6 acts as: under `sudo` the invoking user, else the process user."""
    euid = os.geteuid()
    sudo_uid = os.environ.get("SUDO_UID")
    if euid == 0 and sudo_uid and sudo_uid.isdigit():
        uid = int(sudo_uid)
        gid_raw = os.environ.get("SUDO_GID", "")
        gid = int(gid_raw) if gid_raw.isdigit() else uid
        entry = _passwd_entry(uid)
        name = os.environ.get("SUDO_USER", "") or (entry.pw_name if entry else str(uid))
        home = (
            pathlib.Path(entry.pw_dir)
            if entry
            else pathlib.Path(os.environ.get("HOME", "/")).resolve()
        )
        return RealUser(uid=uid, gid=gid, name=name, home=home, via_sudo=True)
    uid = os.getuid()
    gid = os.getgid()
    home_env = os.environ.get("HOME")
    home = pathlib.Path(home_env) if home_env else (_passwd_home(uid) or pathlib.Path("/"))
    try:
        name = pwd.getpwuid(uid).pw_name
    except KeyError:
        name = str(uid)
    return RealUser(uid=uid, gid=gid, name=name, home=home, via_sudo=False)


def _user_dir(user: RealUser | None, xdg_env: str, *home_parts: str) -> pathlib.Path:
    """Return one agent6 user dir: the XDG variable's, else under the operator's home.

    Under `sudo` the XDG variable is root's and is skipped.

    Args:
        user: The operator; None resolves the effective one.
        xdg_env: The XDG variable's name.
        *home_parts: The path under the home when the variable is unset.

    Returns:
        The directory.
    """
    user = user or effective_user()
    if not user.via_sudo:
        xdg = os.environ.get(xdg_env)
        if xdg:
            return pathlib.Path(xdg) / "agent6"
    return user.home.joinpath(*home_parts) / "agent6"


def global_config_dir(user: RealUser | None = None) -> pathlib.Path:
    """Return the global config directory, `$XDG_CONFIG_HOME/agent6` or `~/.config/agent6`."""
    return _user_dir(user, "XDG_CONFIG_HOME", ".config")


def global_config_path(user: RealUser | None = None) -> pathlib.Path:
    """Return the global config file's path."""
    return global_config_dir(user) / "config.toml"


def secrets_path(user: RealUser | None = None) -> pathlib.Path:
    """Return the secrets file's path."""
    return global_config_dir(user) / "secrets.toml"


def ui_settings_path(user: RealUser | None = None) -> pathlib.Path:
    """Return the UI preferences file's path, beside the config.

    A theme is a viewer preference, not agent behaviour, so it never enters the
    config schema or the shareable config layers.
    """
    return global_config_dir(user) / "ui.toml"


def cache_dir(user: RealUser | None = None) -> pathlib.Path:
    """Return the cache directory, `$XDG_CACHE_HOME/agent6` or `~/.cache/agent6`.

    It holds regenerable data such as the provider model lists; safe to delete.
    """
    return _user_dir(user, "XDG_CACHE_HOME", ".cache")


def jail_cache_home(user: RealUser | None = None) -> pathlib.Path:
    """Return the persistent HOME a jailed command gets, `<cache>/home`.

    Used under `hardened` and `none`, which have no private /tmp, and under `strict`
    with `[sandbox].home = "cache"`. Model-writable across runs, 0700, never the
    operator's own home; `app.confine.check_jail_home` creates it and refuses a
    symlink or another user's directory there.
    """
    return cache_dir(user) / "home"


def data_dir(user: RealUser | None = None) -> pathlib.Path:
    """Return the data directory, `$XDG_DATA_HOME/agent6` or `~/.local/share/agent6`.

    It holds installed skills; unlike the cache it is not regenerable.
    """
    return _user_dir(user, "XDG_DATA_HOME", ".local", "share")


def state_base(user: RealUser | None = None) -> pathlib.Path:
    """Return the state base, `$XDG_STATE_HOME/agent6` or `~/.local/state/agent6`.

    Each repo gets `<base>/<repo-id>/`, out of the workspace; the jail masks the base.
    """
    return _user_dir(user, "XDG_STATE_HOME", ".local", "state")


def private_dirs() -> tuple[pathlib.Path, ...]:
    """Return the directories a jailed command must never see: the config dir and the state base.

    One owner: the jail masks them, the tool-mount scan refuses them and the config
    validator rejects grants inside them. The data dir holds skills the model is meant
    to run and the cache holds regenerable lists, so neither is private. Read per call,
    since the XDG variables are per-process.
    """
    return (global_config_dir(), state_base())


def hidden_paths(extra: Iterable[pathlib.Path]) -> tuple[pathlib.Path, ...]:
    """Return every tree hidden from a run: `[sandbox].hide_paths` plus the private dirs.

    One owner, because the jail and the in-process `Workspace` both enforce it, and a
    boundary they disagree about is a hole.
    """
    return (*extra, *private_dirs())


# The filesystem limit is 255 bytes per component, and CJK or emoji run 3 to 4 bytes a character.
_ID_BYTES_MAX = 100
# Only the elided form needs a hash; 12 hex chars is 48 bits, past casual brute force.
_ID_HASH_LEN = 12


def repo_id(repo_root: pathlib.Path) -> str:
    """Return a directory name that identifies the repo root, and only it.

    `/` becomes `-`, and a trailing hex tag records which dashes were slashes, one
    bit per dash: `/a/b/c` is `a-b-c-3`, `/a/b-c` is `a-b-c-2`. The mapping is
    reversible, so nothing collides. Keyed on the resolved path, so a moved checkout
    gets a new id. A path too long for one component has its middle elided and a
    hash of the full path appended.
    """
    real = str(repo_root.resolve()).strip("/")
    flat = real.replace("/", "-")
    if len(flat.encode()) > _ID_BYTES_MAX:
        digest = hashlib.sha256(real.encode("utf-8")).hexdigest()[:_ID_HASH_LEN]
        head, tail = _ID_BYTES_MAX // 3, _ID_BYTES_MAX - _ID_BYTES_MAX // 3 - 2
        return f"{_head_bytes(flat, head)}--{_tail_bytes(flat, tail)}-{digest}"
    marks = "".join("1" if ch == "/" else "0" for ch in real if ch in "/-")
    tag = f"{int(marks or '0', 2):x}"
    # `/` flattens to nothing; the bare tag cannot collide, since every other id carries a dash.
    return f"{flat}-{tag}" if flat else tag


def repo_root_of_id(state_dir_name: str) -> pathlib.Path | None:
    """Return the repo root a state-dir name encodes, `repo_id`'s inverse.

    The decoded candidate must re-encode to exactly the name, so a wrong read is
    impossible.

    Args:
        state_dir_name: The state directory's name.

    Returns:
        The root, or None for a name `repo_id` cannot have produced.
    """
    flat, _, tag = state_dir_name.rpartition("-")
    try:
        bits_val = int(tag, 16)
    except ValueError:
        return None
    dashes = flat.count("-")
    if bits_val >= (1 << dashes):
        return None
    marks = format(bits_val, f"0{dashes}b") if dashes else ""
    out: list[str] = []
    it = iter(marks)
    for ch in flat:
        out.append(("/" if next(it) == "1" else "-") if ch == "-" else ch)
    candidate = pathlib.Path("/" + "".join(out))
    return candidate if repo_id(candidate) == state_dir_name else None


def _head_bytes(s: str, limit: int) -> str:
    """Return the longest prefix fitting in the byte limit, never splitting a character."""
    return s.encode()[:limit].decode(errors="ignore")


def _tail_bytes(s: str, limit: int) -> str:
    """Return the longest suffix fitting in the byte limit, never splitting a character."""
    raw = s.encode()[-limit:]
    while raw:
        try:
            return raw.decode()
        except UnicodeDecodeError:
            raw = raw[1:]
    return ""


def project_root(start: pathlib.Path) -> pathlib.Path:
    """Return the project the path is inside, or the path itself outside one.

    Walks for `.git` rather than asking git, since this is on every command's path.
    A linked worktree is its repository's project; any other `.git` file makes its
    directory a project of its own. No stop at `$HOME`: under `git init $HOME` every
    directory is one repo, and splitting it would give two runs separate locks on one
    working tree.
    """
    root = checkout_root(start)
    if (root / ".git").is_file():
        git_dir = linked_worktree_git_dir(root)
        if git_dir is not None and git_dir.name == ".git":
            return git_dir.parent
    return root


def checkout_root(start: pathlib.Path) -> pathlib.Path:
    """Return the nearest directory holding a `.git`, or the path itself outside one."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def linked_worktree_git_dir(root: pathlib.Path) -> pathlib.Path | None:
    """Return the repository git dir a linked worktree points into, or None for a plain checkout.

    Resolved as git does, from the `.git` file and the entry's `commondir`. Both files
    are writable by a jailed command under hardened, so the pointer keys state and
    verifies a recorded grant, and never makes one.
    """
    pointer = root / ".git"
    if not pointer.is_file():
        return None
    try:
        text = pointer.read_text(encoding="utf-8", errors="replace").strip()
        if not text.startswith("gitdir:"):
            return None
        admin = pathlib.Path(text[len("gitdir:") :].strip())
        if not admin.is_absolute():
            admin = root / admin
        common = pathlib.Path((admin / "commondir").read_text(encoding="utf-8").strip())
        if not common.is_absolute():
            common = admin / common
        return common.resolve()
    except OSError:
        return None


def state_dir(repo_root: pathlib.Path) -> pathlib.Path:
    """Return the per-repo state directory, keyed on the project rather than the cwd."""
    return state_base() / repo_id(project_root(repo_root))


def repo_config_path(repo_root: pathlib.Path) -> pathlib.Path:
    """Return the per-repo config file, `<state_dir>/config.toml`, out of the repo."""
    return state_dir(repo_root) / "config.toml"


def is_root() -> bool:
    """Return whether the process runs as root."""
    return os.geteuid() == 0


def root_optin_enabled(cli_flag: bool) -> bool:
    """Return whether the operator allowed root: `--allow-root` or `AGENT6_ALLOW_ROOT=1`."""
    return cli_flag or os.environ.get(_ALLOW_ROOT_ENV) == "1"


def mkdir_for_real_user(path: pathlib.Path, user: RealUser | None = None) -> None:
    """Create the directory and its missing ancestors, chowning what was created to the operator.

    Under `sudo` a root-owned base would block every later non-root sibling. The
    handover covers the topmost directory created, recursively; a pre-existing
    directory is never rechowned or rechmodded.

    Args:
        path: The directory.
        user: The operator; None resolves the effective one.
    """
    missing: list[pathlib.Path] = []
    cur = path
    while not cur.exists():
        missing.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    # 0700: a non-traversable base shields the files inside whatever the umask.
    for d in reversed(missing):
        d.mkdir(mode=0o700, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    chown_to_real_user(missing[-1] if missing else path, user)


def chown_to_real_user(path: pathlib.Path, user: RealUser | None = None) -> None:
    """Chown a tree back to the operator, when the process is root through `sudo`.

    Every target is named relative to an open directory fd and no link is followed,
    so a jailed command holding write on the tree cannot swap a component between
    the walk and the call. Permission errors are swallowed; permissions are never
    weakened to compensate.

    Args:
        path: The tree's root.
        user: The operator; None resolves the effective one.
    """
    if os.geteuid() != 0:
        return
    user = user or effective_user()
    if not user.via_sudo:
        return
    with contextlib.suppress(OSError):
        os.lchown(path, user.uid, user.gid)
    if path.is_symlink() or not path.is_dir():
        return
    for _dirpath, dirnames, filenames, dir_fd in os.fwalk(path, follow_symlinks=False):
        for name in (*dirnames, *filenames):
            with contextlib.suppress(OSError):
                os.chown(name, user.uid, user.gid, dir_fd=dir_fd, follow_symlinks=False)
