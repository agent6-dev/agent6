# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one description of what a confined child may do.

`dispatch` (commands) and `mcp_client` (servers) both build their policy here.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import stat
from collections.abc import Sequence
from typing import Any

from agent6 import kinds, memory, paths
from agent6.config import Config
from agent6.sandbox import detect, jail, tool_paths
from agent6.tools import _path_safety, _result_format

# strict's default HOME: the launcher creates it inside the run's private /tmp tmpfs.
JAIL_TMP_HOME = pathlib.Path("/tmp/agent6-home")  # noqa: S108 - resolved inside the jail
# The host's /tmp, which a hardened run's commands write: a link there is re-pointable.
_TMP = pathlib.Path("/tmp").resolve()  # noqa: S108

# What a resolver and a TLS client read: the resolver settings and the CA store.
_HOST_NETWORK_FILES = (
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/ssl",
    "/etc/pki",
    "/etc/ca-certificates",
)


def host_network_reads() -> tuple[pathlib.Path, ...]:
    """Return the files a child on the host network reads to resolve names and verify TLS.

    Those that exist on this host, each followed by the file it points to when it is a symlink
    (a resolver file often points into `/run`): strict binds the name the resolver opens, and
    Landlock needs the real file.
    """
    out: dict[pathlib.Path, None] = {}
    for name in _HOST_NETWORK_FILES:
        path = pathlib.Path(name)
        if path.exists():
            out[path] = None
            out[path.resolve()] = None
    return tuple(out)


def persistent_jail_home(
    config: Config,
    isolation: kinds.IsolationLevel,
    root: pathlib.Path,
    worktree_git_dir: pathlib.Path | None = None,
) -> pathlib.Path | None:
    """Return the persistent HOME a jailed command gets, or None where `JAIL_TMP_HOME` applies.

    Only `strict` has a private /tmp for a throwaway HOME; every other level gets the
    repository's `agent6.paths.jail_cache_home`, and strict opts into it with
    `[sandbox].home = "cache"`. `jail_policy` creates it and grants it read-write at its real
    path. A working tree agent6 made for a repository (a worktree, a lane's clone, a machine
    state's clone) gets that repository's HOME, so a warmed tool cache serves all of them.

    Args:
        config: The run's config.
        isolation: The resolved isolation level.
        root: The workspace.
        worktree_git_dir: The recorded repository git dir of a session in a linked worktree.
            It names the repository; the worktree's own `.git` pointer never does, since a
            jailed command can rewrite it under hardened and aim HOME at another repository's.

    Returns:
        The repository's persistent HOME, or None.
    """
    if isolation == "strict" and config.sandbox.home == "tmp":
        return None
    if worktree_git_dir is not None and worktree_git_dir.name == ".git":
        return paths.jail_cache_home(worktree_git_dir.parent)
    # Named by where agent6 put the tree, which no jailed command can move: the first
    # directory below the parallel workdir is the owning repository's id.
    base = paths.parallel_base(config.parallel.workdir)
    try:
        owner = root.resolve().relative_to(base.resolve()).parts[:1]
    except ValueError:
        owner = ()
    if owner:
        return paths.jail_cache_home_of(owner[0])
    return paths.jail_cache_home(paths.checkout_root(root))


def jail_home_refusal(home: pathlib.Path) -> str | None:
    """Return why a path cannot be the jail's persistent HOME, or None when it can.

    An absent path is fine (`jail_policy` creates it). Refused: a symlink, a non-directory,
    another user's directory, a mode with any group or other bit (a jailed command may chmod
    the dir, and an open one lets another local user plant what the next run consumes; checked
    on every build, never restored), or a path inside an agent6-private dir, which the strict
    mask would re-bind writable. Checked on resolved paths so a symlinked ancestor cannot dodge
    it; the config validator refuses the same for `extra_write_paths`.
    """
    real = home.resolve()
    for private in paths.private_dirs():
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
    uid = paths.effective_user().uid
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


@dataclasses.dataclass(frozen=True, slots=True)
class DroppedGrant:
    """A configured grant a run leaves out.

    Attributes:
        key: The dotted config key that lists it.
        path: The configured path.
        real: The path it resolves to.
        region: The writable region it sits inside, where a jailed command can re-point it.
        via_link: Only a link on the way to it sits in the region; the grant ends outside.
    """

    key: str
    path: pathlib.Path
    real: pathlib.Path
    region: pathlib.Path
    via_link: bool = False

    def line(self) -> str:
        """Return the sentence a run prints for the grant at its start."""
        where = "" if self.real == self.path else f" (it resolves to {self.real})"
        who = "a hardened run's commands" if self.region == _TMP else "jailed commands"
        pin = " Pin sandbox.isolation = strict to keep it." if self.region == _TMP else ""
        if self.via_link:
            return (
                f"{self.key} entry {self.path}{where} is not granted: a link on its"
                f" way sits inside {self.region}, which {who} write, so a command"
                f" could point it anywhere. Grant {self.real} by its own path.{pin}"
            )
        if self.real == self.region:
            return (
                f"{self.key} entry {self.path}{where} is not granted on its own: it is"
                f" {self.region} itself, which the run already holds, and a second bind of it"
                " would lack the protections the first carries."
            )
        # A link inside the region that leads out: what it held is not part of the region.
        still = (
            f" What it holds stays reachable as part of {self.region}."
            if _path_safety.path_within(self.real, self.region)
            else f" Grant {self.real} by its own path."
        )
        return (
            f"{self.key} entry {self.path}{where} is not granted on its own: it sits"
            f" inside {self.region}, which {who} write, so a command could point it"
            f" anywhere.{still}{pin}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Grants:
    """The config's path grants as one run holds them, resolved once.

    Every jail launch and the file tools bind these same directories for the whole run: a
    grant resolved per launch follows whatever a link on its way points to at that moment.

    Attributes:
        read: Each read grant kept, as (configured path, resolved path).
        write: Each write grant kept, the same way.
        dropped: The grants left out.
    """

    read: tuple[tuple[pathlib.Path, pathlib.Path], ...] = ()
    write: tuple[tuple[pathlib.Path, pathlib.Path], ...] = ()
    dropped: tuple[DroppedGrant, ...] = ()


def _link_dirs(path: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Return the resolved directory of every symlink a resolution of the path passes through."""
    dirs: list[pathlib.Path] = []
    current = pathlib.Path(path.anchor)
    pending = list(path.parts[1:])
    # The kernel gives up after 40 links too.
    while pending and len(dirs) <= 40:
        name = pending.pop(0)
        step = current.parent if name == ".." else current / name
        if not step.is_symlink():
            current = step
            continue
        try:
            target = pathlib.PurePosixPath(step.readlink())
        except OSError:
            break
        dirs.append(current)
        if target.is_absolute():
            current = pathlib.Path(target.anchor)
        pending = [*(target.parts[1:] if target.is_absolute() else target.parts), *pending]
    return tuple(dirs)


def _resolved_isolation(requested: str) -> str:
    """Return the level the run would get on this host; an unhonourable one as requested."""
    try:
        return detect.resolve_isolation(requested, detect.environment())
    except detect.IsolationUnavailableError:
        return requested


def resolve_grants(
    config: Config,
    root: pathlib.Path,
    *,
    server: tuple[str, Sequence[str], Sequence[str]] | None = None,
) -> Grants:
    """Resolve `sandbox.extra_read_paths` and `extra_write_paths`, once for a run.

    A grant spelled or resolving inside a region the run writes (the workspace root, a write
    grant), or that is one of them under another name, or reached through a link that sits in
    one, is dropped: a jailed command can swap a component there for a symlink out of every
    grant, and an alias of the workspace would be bound without its read-only `.git`. `/tmp`
    is such a region when the run would not be `strict` on this host, whose private `/tmp` no
    command can re-point: a `hardened` run's commands write the host's, and what a grant there
    held stays reachable as part of it. A grant elsewhere resolves as the operator linked it.

    Args:
        config: The run's config.
        root: The workspace root.
        server: An MCP server's own grants, as (name, read paths, write paths), resolved with
            the config's by the same rule; its write paths are writable regions too.

    Returns:
        The grants kept, each with its resolved path, and the grants dropped.
    """
    sb = config.sandbox
    # Each group: the dotted key that lists it, whether it grants writes, its entries.
    groups: list[tuple[str, bool, Sequence[str]]] = [
        ("sandbox.extra_read_paths", False, sb.extra_read_paths),
        ("sandbox.extra_write_paths", True, sb.extra_write_paths),
    ]
    if server is not None:
        name, reads, writes = server
        groups += [
            (f"mcp.servers.{name}.sandbox.read_paths", False, reads),
            (f"mcp.servers.{name}.sandbox.write_paths", True, writes),
        ]
    write_regions = tuple(
        pathlib.Path(p).expanduser().resolve()
        for _key, writes, entries in groups
        if writes
        for p in entries
    )
    regions = (root.resolve(), *write_regions)
    if _resolved_isolation(sb.isolation) != "strict":
        regions += (_TMP,)
    kept: dict[bool, list[tuple[pathlib.Path, pathlib.Path]]] = {False: [], True: []}
    dropped: list[DroppedGrant] = []
    for key, writes, listed in groups:
        entries = kept[writes]
        for grant in listed:
            path = pathlib.Path(grant).expanduser()
            real = path.resolve()
            # Inside a region by its spelling (an ancestor resolves into one) or by its target,
            # or the region itself under another name; a write grant is not inside itself.
            inside = (real, *(step.resolve() for step in path.parents))
            others = tuple(w for w in regions if not (writes and w == real and w != regions[0]))
            region = next(
                (w for w in others if any(_path_safety.path_within(p, w) for p in inside)), None
            )
            links = () if region is not None else _link_dirs(path)
            linked = next(
                (w for w in regions if any(_path_safety.path_within(d, w) for d in links)), None
            )
            if region is not None or linked is not None:
                dropped.append(
                    DroppedGrant(
                        key=key,
                        path=path,
                        real=real,
                        region=region or linked or root,
                        via_link=region is None,
                    )
                )
            else:
                entries.append((path, real))
    return Grants(read=tuple(kept[False]), write=tuple(kept[True]), dropped=tuple(dropped))


def workspace_for(
    config: Config,
    root: pathlib.Path,
    *,
    memory_dir: pathlib.Path | None = None,
    grants: Grants | None = None,
) -> _path_safety.Workspace:
    """Build the in-process file boundary for a run.

    The tools are the front door of the file axis (an untrusted model reaches files through
    them with no approval) and the jail is the fence that stops a command working around it.
    Both read the same hidden set (`agent6.paths.hidden_paths`), so the two cannot disagree.
    Derived from config values, never from the isolation level: a degradation must never widen
    what the tools may touch. A grant `resolve_grants` drops reads as outside the boundary, and
    `jail_policy` leaves it out of the mounts.

    Args:
        config: The run's config.
        root: The workspace root.
        memory_dir: The per-repo memory dir, model-writable by design through the in-process
            tools only; `jail_policy` never mounts it.
        grants: The run's resolved grants; resolved here when None.

    Returns:
        The workspace.
    """
    denied = paths.hidden_paths(pathlib.Path(p) for p in config.sandbox.hide_paths)
    if grants is None:
        grants = resolve_grants(config, root)
    readable = tuple(real for _path, real in grants.read)
    writable = tuple(real for _path, real in grants.write)
    mem = (memory_dir.resolve(),) if memory_dir is not None else ()
    return _path_safety.Workspace(
        root=root.resolve(),
        denied=tuple(p.resolve() for p in denied),
        # Write implies read, matching the grants the jail mounts.
        read_roots=(*readable, *writable, *mem),
        write_roots=(*writable, *mem),
        exempt=mem,
        read_only=(memory_dir.resolve() / memory.DECISIONS_NAME,) if memory_dir is not None else (),
    )


def resolve_network(
    config: Config, isolation: kinds.IsolationLevel, *, override: kinds.NetworkMode | None = None
) -> kinds.NetworkMode:
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
    root: pathlib.Path,
    config: Config,
    isolation: kinds.IsolationLevel,
    argv: tuple[str, ...],
    *,
    timeout_s: float | None = None,
    extra_ro_paths: tuple[pathlib.Path, ...] = (),
    extra_rw_paths: tuple[pathlib.Path, ...] = (),
    extra_protect_paths: tuple[pathlib.Path, ...] = (),
    network: kinds.NetworkMode | None = None,
    env_base: dict[str, str] | None = None,
    worktree_git_dir: pathlib.Path | None = None,
    grants: Grants | None = None,
) -> kinds.JailPolicy:
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
            the `[git.commit]` identity, the uv and bytecode settings) apply either way.
        worktree_git_dir: The repository git dir agent6 recorded when it added the root as a
            fork's linked worktree, granted read-only once the worktree's own `.git` pointer
            still resolves to it. A linked worktree with no record gets no grant.
        grants: The run's resolved grants, so every launch of the run binds the same
            directories; resolved here when None, for a policy built once.

    Returns:
        The policy.

    Raises:
        JailUnavailableError: The persistent HOME is refused or cannot be created, or the
            worktree's `.git` pointer names anything but the recorded git dir (a rewritten
            pointer grants nothing).
    """
    network = resolve_network(config, isolation, override=network)
    protect_paths: list[pathlib.Path] = []
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
    env = _result_format.passthrough_env() if env_base is None else dict(env_base)
    # HOME is forced like PATH: the operator's HOME is not in the jail, and toolchains need a
    # writable cache root under it (go-build, CARGO_HOME, uv).
    persistent = persistent_jail_home(config, isolation, root, worktree_git_dir)
    if persistent is not None:
        # Created here so every policy builder gets a HOME that exists: the launcher skips a
        # missing rw path silently. Inspected before creation and again after it.
        refusal = jail_home_refusal(persistent)
        if refusal is None and not os.path.lexists(persistent):
            try:
                paths.mkdir_for_real_user(persistent)
            except OSError as exc:
                raise jail.JailUnavailableError(
                    f"the jail's HOME {str(persistent)!r} cannot be created: {exc}"
                ) from exc
            refusal = jail_home_refusal(persistent)
        if refusal is not None:
            raise jail.JailUnavailableError(refusal)
    env["HOME"] = str(persistent or JAIL_TMP_HOME)
    # That HOME holds no git identity, so a `git commit` the model runs (`git.control = "model"`)
    # carries the operator's `[git.commit]` one; an unset field leaves the identity to git (a
    # repo-local user.name still serves). The trailer stays agent6's, added only where it commits.
    if config.git.commit.name:
        env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = config.git.commit.name
    if config.git.commit.email:
        env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = config.git.commit.email
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # The jail is offline, so `uv run` must use the venv the operator already synced.
    env.setdefault("UV_NO_SYNC", "1")
    # A controlled PATH plus the operator's tool dirs as RO+exec mounts; else `uv run` dies 127.
    tool_path, tool_mounts = tool_paths.operator_tool_paths()
    env["PATH"] = tool_path
    git_dir: pathlib.Path | None = None
    if worktree_git_dir is not None:
        pointed = paths.linked_worktree_git_dir(root)
        if pointed != worktree_git_dir:
            raise jail.JailUnavailableError(
                f"the worktree's .git at {root / '.git'} points at {pointed}, not the"
                f" repository git dir agent6 recorded for it ({worktree_git_dir});"
                " a rewritten pointer grants nothing. Restore the pointer, or fork again."
            )
        git_dir = worktree_git_dir
    if grants is None:
        grants = resolve_grants(config, root)
    return kinds.JailPolicy(
        cwd=root,
        argv=argv,
        isolation=isolation,
        env=tuple(sorted(env.items())),
        network=network,
        extra_protect_paths=tuple(protect_paths),
        extra_ro_paths=(
            *(path for path, _real in grants.read),
            *extra_ro_paths,
            *((git_dir,) if git_dir is not None else ()),
            # The host network needs these to resolve names and verify TLS; no other network does.
            *(host_network_reads() if network == "host" and isolation != "none" else ()),
        ),
        extra_rw_paths=(
            *(path for path, _real in grants.write),
            *extra_rw_paths,
            *(() if persistent is None else (persistent,)),
        ),
        grant_sources=tuple(
            (path, real) for path, real in (*grants.read, *grants.write) if path != real
        ),
        extra_device_paths=tuple(pathlib.Path(p) for p in config.sandbox.extra_device_paths),
        tool_paths=tool_mounts,
        hide_paths=tuple(pathlib.Path(p) for p in config.sandbox.hide_paths),
        memory_limit_mb=config.sandbox.memory_limit_mb,
        **policy_kwargs,
    )
