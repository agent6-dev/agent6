# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Provider-only egress broker + agent-process Landlock setup for runs."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from agent6.config import Config
from agent6.config_layer import resolved_state_dir
from agent6.detect import Environment
from agent6.providers.egress import clear_routes, parse_endpoint, register_route
from agent6.sandbox import (
    BrokerHandle,
    EgressBrokerError,
    Endpoint,
    LandlockNotSupportedError,
    apply_agent_landlock,
    enter_network_isolation,
    start_egress_broker,
)
from agent6.sandbox.jail import _locate_jail_binary
from agent6.types import SandboxProfile


def _provider_endpoints(cfg: Config) -> set[Endpoint]:
    """The set of ``host:port`` endpoints every configured provider dials.

    Used to build the provider-only egress allow-list: one broker socket per
    endpoint. Every provider (both api_formats, every deployment) carries its
    effective endpoint host in ``base_url`` -- which is exactly the host the
    provider dials -- so the allow-list is derived uniformly from it. The
    deployment profile only appends path/model to that host, so the host:port
    is unchanged by it.
    """
    eps: set[Endpoint] = set()
    for entry in cfg.providers.values():
        host, port = parse_endpoint(entry.base_url)
        eps.add(Endpoint(host=host, port=port))
    return eps


def _allow_url_endpoints(cfg: Config) -> set[Endpoint]:
    """Extra ``host:port`` endpoints from ``sandbox.allow_urls``.

    Each entry is already validated by ``SandboxConfig`` as a host, host:port,
    or URL. We normalize a missing scheme to ``https://`` (so a bare host
    defaults to 443), kept in lock-step with ``config._validate_allow_url``,
    then reuse ``parse_endpoint``. Folded into the provider-only egress
    allow-list alongside the provider endpoints (union); the winning config
    tier already decided which ``allow_urls`` list applies.
    """
    eps: set[Endpoint] = set()
    for entry in cfg.sandbox.allow_urls:
        url = entry if "://" in entry else f"https://{entry}"
        host, port = parse_endpoint(url)
        eps.add(Endpoint(host=host, port=port))
    return eps


def _warn_if_unsandboxed(selected_profile: SandboxProfile) -> None:
    """Print a prominent warning when running without the kernel sandbox.

    The `none` profile is reached either on a non-Linux host (no kernel sandbox)
    or when the operator EXPLICITLY sets `profile = "none"` on Linux (the
    unsandboxed opt-out, intended for inside a container). Either way commands
    run as plain subprocesses with no agent6 confinement, so say so loudly.
    """
    if selected_profile != "none":
        return
    print(
        "[agent6] WARNING: running UNSANDBOXED (sandbox.profile = 'none'). "
        "Commands -- including the LLM's run_command and verify_command -- "
        "execute as plain subprocesses with NO filesystem, network, or syscall "
        "confinement; the agent is contained only by the surrounding environment "
        "(e.g. the container it runs in). Use 'auto'/'strict'/'hardened' for "
        "kernel-enforced isolation.",
        file=sys.stderr,
    )


def _is_loopback(host: str) -> bool:
    """True for a loopback host (a local model endpoint, e.g. Ollama)."""
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.startswith("127.")  # noqa: S104


def _check_network_profile(cfg: Config, selected_profile: SandboxProfile) -> str | None:
    """A refusal message if the network config can't be enforced on this profile.

    ``agent_network = "local"`` (loopback-pinning) and ``tool_network =
    "only_explicit_states"`` (singling one tool out) both need a network
    namespace, which only the ``strict`` profile provides. On ``hardened`` (a
    real sandbox that can't provide them) we refuse rather than silently
    under-confine; on ``none`` (no sandbox at all) the unsandboxed warning
    already covers it and we run. Returns None when fine.
    """
    if selected_profile != "hardened":
        return None
    sb = cfg.sandbox
    if sb.agent_network == "local":
        return (
            "sandbox.agent_network = 'local' requires the strict profile (loopback"
            " pinning needs the egress broker), but this host supports only"
            " 'hardened'. Use 'providers' or 'open'."
        )
    if sb.tool_network == "only_explicit_states":
        return (
            "sandbox.tool_network = 'only_explicit_states' requires the strict"
            " profile (a per-tool network namespace singles one tool out), but"
            " this host supports only 'hardened'. Use 'block' or 'allow'."
        )
    return None


def _maybe_start_egress(
    cfg: Config, selected_profile: SandboxProfile
) -> tuple[BrokerHandle | None, Path | None, str | None]:
    """Confine the agent process's egress via the broker, if configured.

    Returns ``(broker, sock_dir, error)``. ``error`` non-None ⇒ the caller must
    refuse the run. Only acts on the ``strict`` profile under
    ``agent_network ∈ {providers, local}``, on ``open`` nothing is confined,
    and on ``hardened`` the agent-process Landlock (see
    :func:`_maybe_apply_agent_landlock`) provides port-level confinement
    instead. ``local`` restricts to loopback provider endpoints and refuses any
    non-local provider.

    Must run before any network object is built and while single-threaded
    (``unshare(CLONE_NEWUSER)``). On success the process is left inside an empty
    network namespace whose only routes are the broker's per-endpoint sockets.
    """
    mode = cfg.sandbox.agent_network
    if mode == "open" or selected_profile != "strict":
        return None, None, None
    if mode == "local":
        eps = _provider_endpoints(cfg)
        non_local = sorted(f"{e.host}:{e.port}" for e in eps if not _is_loopback(e.host))
        if non_local:
            return (
                None,
                None,
                "sandbox.agent_network = 'local' permits only loopback providers,"
                f" but these are non-local: {', '.join(non_local)}. Use a local"
                " model (e.g. Ollama) or set agent_network = 'providers'.",
            )
        endpoints = eps
    else:  # providers
        endpoints = _provider_endpoints(cfg) | _allow_url_endpoints(cfg)
    sock_dir = Path(tempfile.mkdtemp(prefix="agent6-egress-"))
    broker: BrokerHandle | None = None
    try:
        broker = start_egress_broker(endpoints, sock_dir=sock_dir)
        enter_network_isolation()
    except (EgressBrokerError, OSError) as exc:
        # OSError covers a socket bind/listen failure inside start_egress_broker
        # (resource exhaustion, permissions) AND a failure of
        # enter_network_isolation AFTER the broker child has been forked. Fail
        # closed: reap the broker if it started, clean up the socket dir, and
        # refuse the run rather than leak a process/dir or run unconfined.
        if broker is not None:
            broker.close()
        shutil.rmtree(sock_dir, ignore_errors=True)
        return None, None, f"could not establish agent-network confinement: {exc}"
    for ep in endpoints:
        uds = broker.uds_for(ep.host, ep.port)
        if uds is not None:
            register_route(ep.host, ep.port, uds)
    return broker, sock_dir, None


def _stop_egress(broker: BrokerHandle | None, sock_dir: Path | None) -> None:
    """Tear down the egress broker and clear its routes. Idempotent."""
    if broker is not None:
        broker.close()
    clear_routes()
    if sock_dir is not None:
        shutil.rmtree(sock_dir, ignore_errors=True)


def _maybe_apply_agent_landlock(
    cfg: Config, selected_profile: SandboxProfile, env: Environment
) -> str | None:
    """Confine the agent's OWN process with Landlock on hardened hosts.

    Returns ``None`` when nothing is to be done or confinement succeeds, or a
    ready-to-print error message when the run must be refused.

    Only the ``hardened`` profile takes this path. The ``strict`` profile
    instead runs every child command in its own user+mount+pid+net namespace
    (a stronger boundary) and confines provider egress with the broker;
    Landlocking the agent there would break the jail's ``pivot_root(2)`` /
    ``mount(2)`` on kernels at ABI >= 7. Irrevocable, and applied before any
    provider or network object is built so it covers the whole run and every
    child it spawns.
    """
    if selected_profile != "hardened" or not env.kernel.supports_landlock_fs:
        return None
    cwd = Path.cwd().resolve()
    # The agent (and the curator subprocess) persist run state OUT of the
    # workspace, under the per-repo state dir; grant it read+write so they can
    # write transcripts, snapshots, and the graph. Created here so the Landlock
    # O_PATH open below finds it. Because state lives OUT of cwd by default,
    # jailed children (whose hardened ruleset grants RW only recursively under
    # cwd) do not get this path, so the agent's grant does not leak to them
    # (Landlock rulesets intersect). Caveat: an operator who points
    # [agent6].state_dir at an absolute path nested under the repo would bring it
    # inside the child's cwd grant; the validator enforces absoluteness only.
    state = resolved_state_dir(cwd)
    state.mkdir(parents=True, exist_ok=True)
    # Landlock allow-root, not a temp file we create: children (git, the jail
    # launcher, the curator socket dir) legitimately read and write under /tmp.
    tmp = Path("/tmp")  # noqa: S108
    dev_files = tuple(
        p
        for p in (
            Path("/dev/null"),
            Path("/dev/zero"),
            Path("/dev/urandom"),
            Path("/dev/random"),
            Path("/dev/tty"),
        )
        if p.exists()
    )
    run_paths = (Path("/run"),) if Path("/run").exists() else ()
    proc_paths = (Path("/proc"),) if Path("/proc").exists() else ()
    # The agent process (and the curator subprocess it re-execs) must be able to
    # READ its own Python install for lazy imports. A `uv tool` install lives
    # under $HOME (already covered), but a venv outside $HOME, a dev checkout,
    # /opt, a system venv, would otherwise fail when agent6 is run from an
    # unrelated cwd (PermissionError importing e.g. a pydantic submodule).
    py_paths = tuple(
        p
        for p in {
            Path(sys.prefix),
            Path(sys.base_prefix),
            Path(sys.executable).resolve().parent,
            # The directory that CONTAINS the agent6 package (the sys.path entry
            # the import finder scandir()s). For an editable/dev install this is
            # the source root (e.g. <repo>/src), outside the venv, which the
            # curator subprocess (-m agent6.graph.server) must read to import.
            Path(__file__).resolve().parents[2],
        }
        if p.exists()
    )
    # The jailed-command launcher itself: run_in_jail execs it from THIS
    # (Landlocked) process, so its directory must be in the read+exec set or the
    # jail cannot start. py_paths cover the bundled (venv) and dev-checkout
    # binaries; an AGENT6_JAIL_BIN override to an out-of-tree path would
    # otherwise EACCES under the agent-process Landlock.
    jail_bin = _locate_jail_binary()
    jail_paths = (jail_bin.resolve().parent,) if jail_bin is not None else ()
    read_paths = (
        cwd,
        state,
        Path.home(),
        Path("/usr"),
        Path("/etc"),
        tmp,
        *dev_files,
        *run_paths,
        *proc_paths,
        *py_paths,
        *jail_paths,
    )
    write_paths = (cwd, state, tmp, *dev_files, *proc_paths)
    # Hardened can't run the broker, so we fall back to Landlock TCP-connect
    # rules: under `providers` confine to the provider ports (host-level, weaker
    # than the broker but the best hardened offers); under `open` impose no TCP
    # restriction. (`local` is refused on hardened by `_check_network_profile`.)
    ports: tuple[int, ...] = (
        ()
        if cfg.sandbox.agent_network == "open"
        else tuple(sorted({ep.port for ep in _provider_endpoints(cfg)}))
    )
    try:
        report = apply_agent_landlock(
            read_paths=read_paths,
            write_paths=write_paths,
            tcp_connect_ports=ports,
        )
    except LandlockNotSupportedError:
        print(
            "[agent6] WARNING: Landlock unavailable; agent process is NOT "
            "filesystem/network confined",
            file=sys.stderr,
        )
        return None
    except OSError as exc:
        return f"could not apply agent Landlock confinement: {exc}"
    tcp_note = (
        f", tcp connect ports {report.tcp_connect_ports}"
        if report.tcp_supported
        else " (kernel too old for Landlock TCP rules)"
    )
    print(
        f"[agent6] agent-process Landlock: ABI {report.abi}, "
        f"{len(report.fs_read)} read / {len(report.fs_write)} write roots{tcp_note}",
        file=sys.stderr,
    )
    return None
