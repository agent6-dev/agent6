# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Provider-only egress broker + agent-process Landlock setup for runs."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from agent6.config import (
    AnthropicProviderEntry,
    Config,
)
from agent6.detect import Environment
from agent6.providers.anthropic import ANTHROPIC_URL
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
from agent6.types import SandboxProfile


def _provider_endpoints(cfg: Config) -> set[Endpoint]:
    """The set of ``host:port`` endpoints every configured provider dials.

    Used to build the provider-only egress allow-list: one broker socket
    per endpoint. Anthropic's endpoint is fixed; OpenAI-compatible
    providers carry it in ``base_url``.
    """
    eps: set[Endpoint] = set()
    for entry in cfg.providers.values():
        url = ANTHROPIC_URL if isinstance(entry, AnthropicProviderEntry) else entry.base_url
        host, port = parse_endpoint(url)
        eps.add(Endpoint(host=host, port=port))
    return eps


def _allow_url_endpoints(cfg: Config) -> set[Endpoint]:
    """Extra ``host:port`` endpoints from ``sandbox.allow_urls``.

    Each entry is already validated by ``SandboxConfig`` as a host, host:port,
    or URL. We normalize a missing scheme to ``https://`` (so a bare host
    defaults to 443) — kept in lock-step with ``config._validate_allow_url`` —
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

    The `none` profile is only reached on non-Linux hosts (see
    `agent6.detect.select_profile`); commands run as plain subprocesses with
    no confinement, so the operator must be told loudly.
    """
    if selected_profile != "none":
        return
    print(
        "[agent6] WARNING: the Linux kernel sandbox is unavailable on this "
        f"platform ({sys.platform}); running UNSANDBOXED. Commands, including "
        "the LLM's run_command tool and verify_command, execute as plain "
        "subprocesses with NO filesystem, network, or syscall confinement. "
        "Run agent6 on Linux for kernel-enforced isolation.",
        file=sys.stderr,
    )


def _maybe_start_egress(
    cfg: Config, selected_profile: SandboxProfile
) -> tuple[BrokerHandle | None, Path | None, str | None]:
    """Establish provider-only egress confinement, if configured.

    Returns ``(broker, sock_dir, error)``. When ``error`` is non-None the
    caller must refuse the run (the message is ready to print). When
    ``sandbox.network != "provider_only"`` returns ``(None, None, None)``
    and nothing is confined.

    Must be called before any network-using object is built and while the
    process is single-threaded (``unshare(CLONE_NEWUSER)`` requires it).
    On success this process is left inside an empty network namespace and
    the egress routes are registered so provider calls reach the broker.
    """
    if cfg.sandbox.network != "provider_only":
        return None, None, None
    if selected_profile != "strict":
        return (
            None,
            None,
            (
                "sandbox.network = 'provider_only' requires the strict profile "
                "(unprivileged user namespaces) to confine egress, but this host "
                f"only supports the {selected_profile!r} profile. Set "
                "sandbox.network = 'allow' or 'no', or run on a Linux host with "
                "user namespaces enabled."
            ),
        )
    endpoints = _provider_endpoints(cfg) | _allow_url_endpoints(cfg)
    sock_dir = Path(tempfile.mkdtemp(prefix="agent6-egress-"))
    try:
        broker = start_egress_broker(endpoints, sock_dir=sock_dir)
        enter_network_isolation()
    except EgressBrokerError as exc:
        shutil.rmtree(sock_dir, ignore_errors=True)
        return None, None, f"could not establish provider-only egress: {exc}"
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
    read_paths = (
        cwd,
        Path.home(),
        Path("/usr"),
        Path("/etc"),
        tmp,
        *dev_files,
        *run_paths,
        *proc_paths,
    )
    write_paths = (cwd, tmp, *dev_files, *proc_paths)
    # Allow connecting only to the ports the configured providers dial,
    # rather than blanket-allowing 443: a self-hosted gateway on another
    # port still works, and nothing else can open a TCP connection.
    ports = tuple(sorted({ep.port for ep in _provider_endpoints(cfg)}))
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
