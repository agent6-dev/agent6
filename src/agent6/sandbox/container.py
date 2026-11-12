# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Discover containers and prepare their launcher transport without acquiring ownership."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import pathlib
import re
import selectors
import shlex
import shutil
import subprocess
import time
from typing import Annotated, Self

import pydantic

from agent6 import child_env, kinds
from agent6.sandbox import container_install, jail, jail_session

QUERY_TIMEOUT_S = 10.0
MAX_QUERY_BYTES = 1024 * 1024
_RUNTIME_ENV = (
    "CONTAINER_CONNECTION",
    "CONTAINER_HOST",
    "DOCKER_CERT_PATH",
    "DOCKER_CONFIG",
    "DOCKER_CONTEXT",
    "DOCKER_HOST",
    "DOCKER_TLS_VERIFY",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
)
_ID = re.compile(r"[0-9a-f]{64}")
_User = Annotated[
    str,
    pydantic.StringConstraints(
        pattern=r"^[a-zA-Z0-9_][a-zA-Z0-9_.\-$]*(?::[a-zA-Z0-9_][a-zA-Z0-9_.\-$]*)?$"
    ),
]
_EnvName = Annotated[str, pydantic.StringConstraints(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")]
_EnvValue = Annotated[str, pydantic.StringConstraints(pattern=r"^[^\x00]*$")]


class ContainerError(jail.JailUnavailableError):
    """Container discovery or preparation refused execution."""


class ContainerRuntimeUnavailableError(ContainerError):
    """The selected runtime executable is absent."""


class ContainerRuntimeError(ContainerError):
    """A runtime query failed, timed out or exceeded its output bound."""


class ContainerNotFoundError(ContainerError):
    """Runtime queries succeeded but found no running labelled container."""


class ContainerAmbiguousError(ContainerError):
    """Several running labelled containers match the workspace."""


class ContainerMetadataError(ContainerError):
    """Runtime output or dev container metadata is malformed or inconsistent."""


class ContainerMappingError(ContainerError):
    """The workspace has no unambiguous command-side path."""


class ContainerLifecycleError(ContainerError):
    """An operator start or stop command failed, timed out or exceeded its output bound."""


@dataclasses.dataclass(frozen=True, slots=True)
class ContainerOptions:
    """Hold operator-owned selection inputs, without activating container execution."""

    exec: tuple[str, ...] = ()
    runtime: tuple[str, ...] = ()
    workspace: pathlib.Path | None = None
    start: tuple[str, ...] = ()
    stop: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class ContainerDiscovery:
    """Hold unprepared target facts and an opaque prefix for later launcher preparation.

    remote_env holds unexpanded metadata, not an environment safe to execute with.
    """

    exec: tuple[str, ...]
    target: kinds.ExecutionTarget
    runtime: tuple[str, ...] = ()
    remote_env: tuple[tuple[str, str | None], ...] = ()
    # Inspected bind mounts as (host source, container destination); empty behind a manual exec.
    mounts: tuple[tuple[str, str], ...] = ()


class _InspectModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore", strict=True, frozen=True)


class _Config(_InspectModel):
    user: str = pydantic.Field(alias="User")
    labels: dict[str, str] | None = pydantic.Field(default=None, alias="Labels")


class _State(_InspectModel):
    running: bool = pydantic.Field(alias="Running")


class _Mount(_InspectModel):
    type: str = pydantic.Field(alias="Type")
    source: str = pydantic.Field(default="", alias="Source")
    destination: str = pydantic.Field(alias="Destination")

    @pydantic.field_validator("destination")
    @classmethod
    def absolute_destination(cls, value: str) -> str:
        if not _absolute_container_path(value):
            raise ValueError("mount destination must be an absolute container path")
        return value

    @pydantic.model_validator(mode="after")
    def absolute_bind_source(self) -> Self:
        if self.type == "bind" and not _absolute_container_path(self.source):
            raise ValueError("bind source must be an absolute host path")
        return self


class _Inspect(_InspectModel):
    id: str = pydantic.Field(alias="Id", pattern=r"^[0-9a-f]{64}$")
    config: _Config = pydantic.Field(alias="Config")
    state: _State = pydantic.Field(alias="State")
    mounts: tuple[_Mount, ...] = pydantic.Field(alias="Mounts")


class _Metadata(_InspectModel):
    remote_user: _User | None = pydantic.Field(default=None, alias="remoteUser")
    remote_env: dict[_EnvName, _EnvValue | None] = pydantic.Field(
        default_factory=dict, alias="remoteEnv"
    )

    @pydantic.field_validator("remote_user")
    @classmethod
    def nonnull_remote_user(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("remoteUser must be a user name or UID")
        return value


_INSPECT = pydantic.TypeAdapter(tuple[_Inspect, ...])
_METADATA = pydantic.TypeAdapter(_Metadata | tuple[_Metadata, ...])
_USER = pydantic.TypeAdapter(_User)


def _absolute_container_path(value: str) -> bool:
    path = pathlib.PurePosixPath(value)
    return path.is_absolute() and ".." not in path.parts and not any(ord(c) < 32 for c in value)


def _query(runtime: tuple[str, ...], args: tuple[str, ...]) -> str:
    argv = (*runtime, *args)
    env = child_env.without_provider_keys(
        child_env.curated_env(desktop=False, passthrough=_RUNTIME_ENV)
    )
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env=env,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise ContainerRuntimeUnavailableError(
            f"runtime executable {runtime[0]!r} is missing; set container.runtime"
        ) from exc
    except OSError as exc:
        raise ContainerRuntimeError(f"cannot run container.runtime {runtime!r}: {exc}") from exc
    assert proc.stdout is not None and proc.stderr is not None
    try:
        output, errors = _capture_query(proc)
        if proc.returncode:
            reason = errors.decode("utf-8", errors="replace")[-2000:]
            raise ContainerRuntimeError(
                f"container.runtime {runtime!r} {args[0]} failed ({proc.returncode}): {reason}\n"
                "Check the runtime daemon and socket access; for Docker, check the docker group "
                "or set container.runtime to an operator-owned argv prefix."
            )
        return output.decode("utf-8")
    except subprocess.TimeoutExpired as exc:
        raise ContainerRuntimeError(f"container.runtime {runtime!r} {args[0]} timed out") from exc
    except UnicodeDecodeError as exc:
        raise ContainerMetadataError("runtime output is not UTF-8") from exc
    finally:
        proc.stdout.close()
        proc.stderr.close()
        if proc.returncode is None:
            jail.signal_group(proc.pid)
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired as exc:
                raise ContainerRuntimeError(
                    "runtime client did not exit after cancellation"
                ) from exc


def _capture_query(
    proc: subprocess.Popen[bytes], timeout_s: float | None = None
) -> tuple[bytearray, bytearray]:
    assert proc.stdout is not None and proc.stderr is not None
    buffers = {proc.stdout.fileno(): bytearray(), proc.stderr.fileno(): bytearray()}
    total = 0
    if timeout_s is None:
        timeout_s = QUERY_TIMEOUT_S  # read at call time: tests and operators may change it
    deadline = time.monotonic() + timeout_s
    with selectors.DefaultSelector() as selector:
        for stream in (proc.stdout, proc.stderr):
            selector.register(stream, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, timeout_s)
            for key, _ in selector.select(remaining):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                total += len(chunk)
                if total > MAX_QUERY_BYTES:
                    raise ContainerRuntimeError("container.runtime query exceeded the output limit")
                buffers[key.fd].extend(chunk)
        proc.wait(timeout=max(0, deadline - time.monotonic()))
    return buffers[proc.stdout.fileno()], buffers[proc.stderr.fileno()]


def _runtimes(prefix: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    if prefix:
        if exe := shutil.which(prefix[0]):
            return ((str(pathlib.Path(exe).absolute()), *prefix[1:]),)
        raise ContainerRuntimeUnavailableError(
            f"runtime executable {prefix[0]!r} is unavailable; set container.runtime"
        )
    found: dict[tuple[int, int], tuple[str, ...]] = {}
    for name in ("docker", "podman"):
        if exe := shutil.which(name):
            path = pathlib.Path(exe).absolute()
            try:
                info = path.stat()
            except OSError as exc:
                raise ContainerRuntimeError(
                    f"cannot identify runtime executable {path}: {exc}; set container.runtime"
                ) from exc
            found.setdefault((info.st_dev, info.st_ino), (str(path),))
    if not found:
        raise ContainerRuntimeUnavailableError(
            "neither Docker nor Podman is on PATH; set container.runtime or container.exec"
        )
    return tuple(found.values())


def _matches(
    runtimes: tuple[tuple[str, ...], ...], roots: tuple[pathlib.Path, ...], *, stopped: bool = False
) -> tuple[tuple[tuple[str, ...], str], ...]:
    found: list[tuple[tuple[str, ...], str]] = []
    for runtime in runtimes:
        ids: set[str] = set()
        for root in roots:
            output = _query(
                runtime,
                (
                    "ps",
                    *(("-a",) if stopped else ()),
                    "--quiet",
                    "--no-trunc",
                    "--filter",
                    f"label=devcontainer.local_folder={root}",
                ),
            )
            for line in output.splitlines():
                if not _ID.fullmatch(line):
                    raise ContainerMetadataError("runtime ps returned an invalid container ID")
                ids.add(line)
        found.extend((runtime, id_) for id_ in sorted(ids))
    return tuple(found)


def _inspect(runtime: tuple[str, ...], id_: str, roots: tuple[pathlib.Path, ...]) -> _Inspect:
    raw = _query(runtime, ("inspect", "--type", "container", id_))
    try:
        rows = _INSPECT.validate_json(raw)
    except pydantic.ValidationError:
        raise ContainerMetadataError("malformed container inspect data") from None
    if len(rows) != 1 or rows[0].id != id_:
        raise ContainerMetadataError("container inspect did not return the requested identity")
    row = rows[0]
    if not row.state.running:
        raise ContainerMetadataError(f"container {id_} stopped during discovery")
    if (row.config.labels or {}).get("devcontainer.local_folder") not in {str(r) for r in roots}:
        raise ContainerMetadataError("container inspect has a different devcontainer.local_folder")
    return row


def _metadata(row: _Inspect) -> tuple[str, tuple[tuple[str, str | None], ...]]:
    raw = (row.config.labels or {}).get("devcontainer.metadata")
    user = row.config.user or "0"
    env: dict[str, str | None] = {}
    try:
        if raw is not None:
            parsed = _METADATA.validate_json(raw)
            for entry in (parsed,) if isinstance(parsed, _Metadata) else parsed:
                if entry.remote_user is not None:
                    user = entry.remote_user
                env.update(entry.remote_env)
        user = _USER.validate_python(user)
    except pydantic.ValidationError:
        raise ContainerMetadataError("invalid devcontainer.metadata or runtime user") from None
    return user, tuple(sorted(env.items()))


def _mapped_root(host_root: pathlib.Path, mounts: tuple[_Mount, ...]) -> pathlib.Path:
    root = host_root.resolve()
    candidates: set[pathlib.Path] = set()
    for mount in mounts:
        if mount.type != "bind":
            continue
        source = pathlib.Path(mount.source).resolve()
        if root.is_relative_to(source):
            candidates.add(pathlib.Path(mount.destination) / root.relative_to(source))
    valid: set[pathlib.Path] = set()
    for candidate in candidates:
        covering = [m for m in mounts if candidate.is_relative_to(m.destination)]
        deepest = max(len(pathlib.Path(m.destination).parts) for m in covering)
        visible = [m for m in covering if len(pathlib.Path(m.destination).parts) == deepest]
        if all(
            m.type == "bind"
            and (pathlib.Path(m.source) / candidate.relative_to(m.destination)).resolve() == root
            for m in visible
        ):
            valid.add(candidate)
    if len(valid) != 1:
        raise ContainerMappingError(
            "workspace bind mounts are missing, shadowed or ambiguous; set container.workspace "
            "to the existing container path (this creates no mount)"
        )
    return valid.pop()


LIFECYCLE_TIMEOUT_S = 120.0


def run_lifecycle(argv: tuple[str, ...], *, what: str) -> None:
    """Run an operator's start or stop command: fixed argv, no shell, bounded time and output.

    Args:
        argv: The operator's command, as configured.
        what: The setting it came from, for the error.

    Raises:
        ContainerLifecycleError: The command is missing, failed, timed out or flooded.
    """
    env = child_env.without_provider_keys(
        child_env.curated_env(desktop=False, passthrough=_RUNTIME_ENV)
    )
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        raise ContainerLifecycleError(f"{what} {argv!r} did not start: {exc}") from exc
    assert proc.stdout is not None and proc.stderr is not None
    try:
        _, errors = _capture_query(proc, LIFECYCLE_TIMEOUT_S)
        if proc.returncode:
            reason = errors.decode("utf-8", errors="replace")[-2000:]
            raise ContainerLifecycleError(f"{what} failed ({proc.returncode}): {reason}")
    except subprocess.TimeoutExpired as exc:
        raise ContainerLifecycleError(f"{what} timed out after {LIFECYCLE_TIMEOUT_S:.0f}s") from exc
    except ContainerRuntimeError as exc:
        raise ContainerLifecycleError(f"{what}: {exc}") from exc
    finally:
        proc.stdout.close()
        proc.stderr.close()
        if proc.returncode is None:
            jail.signal_group(proc.pid)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=1)


def check_checkout(
    discovery: ContainerDiscovery, checkout: pathlib.Path, git_dir: pathlib.Path | None
) -> None:
    """Refuse a checkout the container cannot reach before any command runs in it.

    A linked worktree, fork, review worktree or machine clone outside the mounted workspace
    has no path inside the container, and a `.git` pointer to a git dir outside it breaks
    git inside.

    Args:
        discovery: The selected container and its host workspace.
        checkout: The checkout commands would run in.
        git_dir: The checkout's repository git dir, when it is a linked worktree.

    Raises:
        ContainerMappingError: The checkout or its git dir lies outside the mounted workspace.
    """
    root = discovery.target.host_root.resolve()
    for label, path in (("checkout", checkout), ("git dir", git_dir)):
        if path is not None and not path.resolve().is_relative_to(root):
            raise ContainerMappingError(
                f"the {label} {path} is outside the mounted workspace {root}; a container"
                " reaches only paths under the workspace it mounts"
            )


def check_hidden_paths(discovery: ContainerDiscovery, hide_paths: tuple[str, ...]) -> None:
    """Refuse a hidden host path the container's mounts expose to commands.

    An explicit restriction is enforced or refused: a hidden path under a bind mount's
    source is reachable inside, and behind a manual exec prefix nothing can be inspected,
    so a non-empty list refuses there too.

    Args:
        discovery: The selected container, with the mounts discovery inspected.
        hide_paths: `[sandbox].hide_paths`, host paths.

    Raises:
        ContainerMappingError: An entry is exposed, or the mounts cannot be inspected.
    """
    if not hide_paths:
        return
    if not discovery.mounts and discovery.target.container_id is None:
        raise ContainerMappingError(
            "sandbox.hide_paths cannot be enforced behind container.exec: the container's "
            "mounts cannot be inspected; unset the hidden paths or select the container by "
            "discovery"
        )
    for hidden in hide_paths:
        path = pathlib.Path(hidden).resolve()
        for source, destination in discovery.mounts:
            mounted = pathlib.Path(source).resolve()
            if path.is_relative_to(mounted) or mounted.is_relative_to(path):
                raise ContainerMappingError(
                    f"sandbox.hide_paths entry {hidden} is exposed to commands through the "
                    f"container mount {source} at {destination}; unset it or change the mount"
                )


def exposures(discovery: ContainerDiscovery) -> tuple[str, ...]:
    """Name what the container exposes to commands beyond the workspace.

    Args:
        discovery: The selected container.

    Returns:
        One line per host path mounted from outside the workspace, sockets named as such;
        empty when the mounts were not inspected or hold only the workspace.
    """
    root = discovery.target.host_root.resolve()
    lines: list[str] = []
    for source, destination in discovery.mounts:
        if pathlib.Path(source).resolve().is_relative_to(root):
            continue
        kind = "socket" if source.endswith(".sock") else "host path"
        lines.append(f"{kind} {source} mounted at {destination}")
    return tuple(lines)


def discover(host_root: pathlib.Path, options: ContainerOptions) -> ContainerDiscovery:
    """Select a container without starting it, executing metadata or opening repo definitions.

    Args:
        host_root: The host workspace, including its lexical spelling for label lookup.
        options: Internal typed inputs supplied by the operator-facing layer.

    Returns:
        Unprepared facts and an exec prefix; HOME and live launcher identity remain unknown.

    Raises:
        ContainerRuntimeUnavailableError: The runtime executable is absent.
        ContainerRuntimeError: A runtime query failed or exceeded its bounds.
        ContainerMetadataError: Runtime output or metadata is malformed or inconsistent.
        ContainerMappingError: The workspace has no unambiguous container path.
        ContainerNotFoundError: Queries succeeded but no running container matched.
        ContainerAmbiguousError: Several running containers matched.
    """
    host_root = host_root.absolute()
    if options.workspace is not None and not _absolute_container_path(str(options.workspace)):
        raise ContainerMappingError("container.workspace must be an absolute container path")
    if options.exec:
        return ContainerDiscovery(
            exec=options.exec,
            target=kinds.ExecutionTarget(
                transport="container",
                host_root=host_root,
                command_root=options.workspace or host_root,
                user=None,
                home=None,
                environment=kinds.ExecutionEnvironment(variables=(), inherit=True),
            ),
        )
    runtimes = _runtimes(options.runtime)
    roots = tuple(dict.fromkeys((host_root, host_root.resolve())))
    matches = _matches(runtimes, roots)
    if not matches:
        stopped = _matches(runtimes, roots, stopped=True)
        definitions = (
            host_root / ".devcontainer.json",
            host_root / ".devcontainer/devcontainer.json",
        )
        if stopped or any(path.exists() for path in definitions):
            ids = ", ".join(id_ for _, id_ in stopped)
            raise ContainerNotFoundError(
                f"no running dev container for {host_root}; labelled containers: {ids or 'none'}. "
                "Start it in VS Code, with devcontainer up, or with container.start."
            )
        raise ContainerNotFoundError(
            "no matching container and no definition at .devcontainer.json or "
            ".devcontainer/devcontainer.json; set container.exec to the argv prefix "
            "for an existing container"
        )
    if len(matches) != 1:
        identities = ", ".join(f"{runtime!r}: {id_}" for runtime, id_ in matches)
        raise ContainerAmbiguousError(
            f"multiple running dev containers: {identities}; select one with container.exec"
        )
    runtime, id_ = matches[0]
    row = _inspect(runtime, id_, roots)
    user, remote_env = _metadata(row)
    command_root = options.workspace or _mapped_root(host_root, row.mounts)
    return ContainerDiscovery(
        exec=(*runtime, "exec", "-i", "--user", user, id_),
        runtime=runtime,
        remote_env=remote_env,
        mounts=tuple((m.source, m.destination) for m in row.mounts if m.type == "bind"),
        target=kinds.ExecutionTarget(
            transport="container",
            host_root=host_root,
            command_root=command_root,
            user=user,
            home=None,
            runtime_id=shlex.join(runtime),
            container_id=id_,
            environment=kinds.ExecutionEnvironment(variables=(), inherit=True),
        ),
    )


def _environment(
    inherited: dict[str, str], remote: tuple[tuple[str, str | None], ...]
) -> tuple[tuple[str, str], ...]:
    resolved = inherited.copy()
    substitutions = re.compile(r"\$\{([^}]*)\}")

    def substitute(match: re.Match[str]) -> str:
        expression = match[1]
        if not expression.startswith("containerEnv:"):
            raise ContainerMetadataError(f"unsupported remoteEnv substitution ${{{expression}}}")
        name = expression.removeprefix("containerEnv:")
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name):
            raise ContainerMetadataError(f"unsupported remoteEnv substitution ${{{expression}}}")
        return inherited.get(name, "")

    for name, value in remote:
        if value is None:
            resolved.pop(name, None)
        else:
            if "${" in substitutions.sub("", value):
                raise ContainerMetadataError(f"unsupported remoteEnv substitution in {name}")
            resolved[name] = substitutions.sub(substitute, value)
    # Temp and capture storage belong to the installation, never metadata or the host.
    resolved["TMPDIR"] = inherited["TMPDIR"]
    if not resolved.get("HOME", "").startswith("/") or "PATH" not in resolved:
        raise ContainerMetadataError("container user needs an absolute HOME and a PATH")
    return tuple(sorted(resolved.items()))


def prepare(
    discovery: ContainerDiscovery,
    *,
    memory_limit_mib: int = 0,
    controls: tuple[kinds.EffectiveControl, ...] = (),
) -> jail_session.JailSession:
    """Install one static launcher and prove its target before handing over the session.

    This deliberately bypasses the host jail. No model argv enters the bootstrap,
    no mounts are created, and attachment grants no container lifecycle authority.

    Args:
        discovery: Operator-selected exec prefix and unprepared facts.
        memory_limit_mib: The per-command memory cap enforced by the launcher.
        controls: Effective controls resolved by the caller, not local confinement policy.

    Returns:
        The ready command-session owner, with target facts and environment populated.

    Raises:
        ContainerError: Installation, compatibility or mount proof failed.
    """
    session: jail_session.JailSession | None = None
    try:
        binary, architecture = container_install.launcher()
        # A terminal corrupts the framed JSON channel; opaque prefixes are never rewritten.
        if any(arg in ("-t", "-it", "-ti", "--tty", "--tty=true") for arg in discovery.exec):
            raise ContainerError("container.exec must not allocate a tty (-t) for the JSON channel")
        if not discovery.exec:
            raise ContainerError("container.exec is empty after selection")
        executable = shutil.which(discovery.exec[0])
        if executable is None:
            raise ContainerError(f"container.exec executable {discovery.exec[0]!r} is missing")
        env = child_env.without_provider_keys(
            child_env.curated_env(desktop=False, passthrough=_RUNTIME_ENV)
        )
        proc = subprocess.Popen(
            (
                str(pathlib.Path(executable).absolute()),
                *discovery.exec[1:],
                "/bin/sh",
                "-c",
                container_install.BOOTSTRAP,
                "agent6-bootstrap",
                str(len(binary)),
                architecture,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env=env,
            start_new_session=True,
        )
        target = dataclasses.replace(
            discovery.target,
            controls=controls,
            launcher_digest="sha256:" + hashlib.sha256(binary).hexdigest(),
        )
        session = jail_session.JailSession.connect(
            proc,
            kinds.JailPolicy(
                cwd=target.command_root,
                argv=(),
                isolation="none",
                network="host",
                memory_limit_mib=memory_limit_mib,
            ),
            target=target,
            bootstrap=binary,
        )
        probe = session.probe()
        if probe.architecture != architecture:
            raise ContainerError("launcher probe architecture differs from the installed ELF")
        environment = _environment(probe.environment, discovery.remote_env)
        assert session.target is not None
        session.target = dataclasses.replace(
            session.target,
            user=f"{probe.uid}:{probe.gid}",
            home=dict(environment)["HOME"],
            environment=kinds.ExecutionEnvironment(environment, inherit=True),
        )
        container_install.prove_mapping(session, target.host_root)
        return session
    except (OSError, jail.JailUnavailableError) as exc:
        if session is not None:
            session.close()
        raise ContainerError(f"container preparation refused: {exc}") from exc
