# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Agent-process confinement: the hardened Landlock layer, and the warnings for
what an isolation level does NOT confine.

There is no agent-process network isolation. It existed once -- an empty netns
plus a broker proxying provider calls -- and was deleted: under `strict` the
agent process has no filesystem confinement, so code execution there could
write `~/.ssh/authorized_keys` or a cron entry and exfiltrate on its own
schedule. Blocking the socket while leaving that open is a partial mitigation
that reads as a guarantee, and it cost four special cases (a host spawner, lane
launching, /btw, MCP servers) plus a HIGH finding of its own.

What confines untrusted work is the JAIL, per command, and that is untouched:
each jailed child still gets its own network namespace under `strict`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.config import Config
from agent6.config.layer import resolved_state_dir
from agent6.sandbox import LandlockNotSupportedError, apply_agent_landlock
from agent6.sandbox.detect import Environment
from agent6.sandbox.jail import locate_jail_binary, operator_tool_paths
from agent6.types import IsolationLevel


def warn_sandbox_gaps(
    isolation: IsolationLevel,
    env: Environment,
    cfg: Config,
    *,
    reporter: Reporter = STDIO_REPORTER,
) -> None:
    """Print a prominent warning when the isolation confines less than it promises.

    `none` is reached on a host with no confinement mechanism at all
    (non-Linux, or a Linux kernel offering neither userns nor Landlock), or
    when the operator EXPLICITLY sets `isolation = "none"` (the unsandboxed
    opt-out, intended for inside a container). Either way commands run as
    plain subprocesses with no agent6 confinement, so say so loudly.

    `strict` needs only userns; on a kernel without Landlock the jail's
    best-effort ruleset enforces nothing (`restrict_self` returns NotEnforced)
    while namespaces + the pivoted read-only rootfs + seccomp still confine.
    That is a documented layer going missing, so it is loud too -- here, once
    per run, not in the launcher: a per-spawn stderr warning would land in
    every tool result and prompt the model to fight the sandbox.

    `tool_network = "auto"` DEGRADES on a netns-less isolation: with no per-child
    network namespace, a jailed run_command shares the (agent-scoped) host
    network instead of being offline, so say so once per run. Explicit `block`
    never reaches here (check_network_support refused it on hardened).
    """
    if isolation == "none":
        reporter.err(
            "[agent6] WARNING: running UNSANDBOXED (sandbox.isolation = 'none'). "
            "Commands -- including the LLM's run_command and verify_command -- "
            "execute as plain subprocesses with NO filesystem, network, or syscall "
            "confinement; the agent is contained only by the surrounding environment "
            "(e.g. the container it runs in). Use 'auto'/'strict'/'hardened' for "
            "kernel-enforced isolation."
        )
    elif isolation == "strict" and env.landlock_abi < 1:
        reporter.err(
            "[agent6] WARNING: 'strict' is running WITHOUT its Landlock layer: "
            "this kernel offers no Landlock (needs Linux >= 5.13 with the "
            "Landlock LSM enabled). Namespaces, the pivoted read-only rootfs, "
            "and seccomp still confine commands; the in-jail Landlock "
            "defense-in-depth is absent."
        )
    if isolation == "hardened" and cfg.sandbox.tool_network == "auto":
        reporter.err(
            "[agent6] WARNING: 'hardened' has no network namespace, so "
            "sandbox.tool_network = 'auto' cannot make a jailed run_command "
            "offline: it shares this process's host network, which hardened "
            "does not confine. Run on 'strict' for a network-free tool sandbox "
            "and provider-only agent egress, or set sandbox.tool_network = "
            "'block' to refuse rather than run here."
        )


def check_network_support(cfg: Config, isolation: IsolationLevel) -> str | None:
    """A refusal message if the network config EXPLICITLY enforces something
    this isolation cannot provide, else None.

    Only jailed commands have a network boundary. ``tool_network =
    "only_explicit_states"`` (singling one tool out) and ``"block"`` (no jailed
    network at all) both need a per-command network namespace, which only
    ``strict`` provides. On ``hardened`` we refuse rather than silently
    under-confine, naming what is unsupported and the fix; ``"auto"`` is the
    secure default that DEGRADES with a warning instead. On ``none`` the
    unsandboxed warning already covers it.
    """
    if isolation != "hardened":
        return None
    sb = cfg.sandbox
    if sb.tool_network == "only_explicit_states":
        return (
            "sandbox.tool_network = 'only_explicit_states' requires the strict"
            " isolation (per-command network namespaces), but this host supports"
            " only 'hardened'. Use 'auto' or 'allow'."
        )
    if sb.tool_network == "block":
        return (
            "sandbox.tool_network = 'block' requires the strict isolation (a"
            " per-command network namespace), but this host supports only"
            " 'hardened', where a jailed command shares this process's network."
            " Use 'auto' to run with a warning, or 'allow' to accept it."
        )
    return None


def maybe_apply_agent_landlock(
    cfg: Config,
    isolation: IsolationLevel,
    *,
    reporter: Reporter = STDIO_REPORTER,
) -> str | None:
    """Confine the agent's OWN process with Landlock on hardened hosts.

    Returns ``None`` when nothing is to be done or confinement succeeds, or a
    ready-to-print error message when the run must be refused.

    Only the ``hardened`` isolation takes this path, and isolation resolution
    (``detect.resolve_isolation``) only selects hardened when the Landlock probe
    succeeded. The ``strict`` isolation instead runs every child command in its
    own user+mount+pid+net namespace (a stronger boundary) and confines
    provider egress with the broker; Landlocking the agent there would break
    the jail's ``pivot_root(2)`` / ``mount(2)`` on kernels at ABI >= 7.
    Irrevocable, and applied before any provider or network object is built so
    it covers the whole run and every child it spawns.
    """
    if isolation != "hardened":
        return None
    cwd = Path.cwd().resolve()
    # The agent persists run state (including the in-process curator's graph)
    # OUT of the workspace, under the per-repo state dir; grant it read+write
    # so it can write transcripts, snapshots, and the graph. Created here so
    # the Landlock O_PATH open below finds it. Because state lives OUT of cwd
    # by default, jailed children (whose hardened ruleset grants RW only
    # recursively under cwd) do not get this path, so the agent's grant does
    # not leak to them (Landlock rulesets intersect). Caveat: an operator who
    # points [agent6].state_dir at an absolute path nested under the repo
    # would bring it inside the child's cwd grant; the validator enforces
    # absoluteness only.
    state = resolved_state_dir(cwd)
    state.mkdir(parents=True, exist_ok=True)
    # Landlock allow-root, not a temp file we create: children (git, the jail
    # launcher) legitimately read and write under /tmp.
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
    # The jail launcher (agent6-jail, hardened isolation) grants the CHILD
    # read+execute on its ro_paths by opening each one from inside THIS
    # already-Landlocked process (PathFd::new in apply_landlock_hardened), and
    # nested Landlock rulesets INTERSECT. If a dir is not in the agent's own
    # read set, the child's rule for it is denied/stripped and the child cannot
    # exec ANY binary that needs it -- every run_command / verify / commit then
    # fails with execve EACCES (returncode 127) on a no-userns host. So the
    # agent read set must be a SUPERSET of the jail child's read+exec roots:
    # the fixed system dirs here (/usr + /etc are above; /dev is the one that
    # bites on a merged-/usr host, the others on split-/usr), plus the two
    # DYNAMIC sets appended below, sourced from the same producers that build
    # the jail policy so they cannot drift.
    sys_exec_dirs = tuple(
        p
        for p in (
            Path("/bin"),
            Path("/sbin"),
            Path("/lib"),
            Path("/lib64"),
            Path("/dev"),
        )
        if p.exists()
    )
    # The agent process must be able to READ its own Python install for lazy
    # imports. A `uv tool` install lives
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
            # agent process must be able to read for its lazy imports.
            Path(__file__).resolve().parents[2],
        }
        if p.exists()
    )
    # The jailed-command launcher itself: run_in_jail execs it from THIS
    # (Landlocked) process, so its directory must be in the read+exec set or the
    # jail cannot start. py_paths cover the bundled (venv) and dev-checkout
    # binaries; an AGENT6_JAIL_BIN override to an out-of-tree path would
    # otherwise EACCES under the agent-process Landlock.
    jail_bin = locate_jail_binary()
    jail_paths = (jail_bin.resolve().parent,) if jail_bin is not None else ()
    # The jail child's dynamic read+exec grants: operator tool mounts
    # (uv/node/... outside the system dirs) and sandbox.extra_read_paths.
    # Read+exec only, never write. Nonexistent grants are skipped exactly as
    # the jail skips them.
    tool_mounts = operator_tool_paths()[1]
    extra_read = tuple(p for p in (Path(x) for x in cfg.sandbox.extra_read_paths) if p.exists())
    read_paths = (
        cwd,
        state,
        Path.home(),
        Path("/usr"),
        Path("/etc"),
        tmp,
        *sys_exec_dirs,
        *dev_files,
        *run_paths,
        *proc_paths,
        *py_paths,
        *jail_paths,
        *tool_mounts,
        *extra_read,
    )
    write_paths = (cwd, state, tmp, *dev_files, *proc_paths)
    # Filesystem only. Hardened cannot run the broker, and Landlock filters
    # connects by PORT, so the one rule available here was "any host on the
    # provider ports" -- no barrier to exfiltration (one HTTPS endpoint is
    # enough, and every host offers one) but a real obstacle to a legitimate
    # tool on another port. Egress is bounded on `strict`, structurally; it is
    # not bounded on hardened, and no longer claims to be.
    try:
        report = apply_agent_landlock(read_paths=read_paths, write_paths=write_paths)
    except (LandlockNotSupportedError, OSError) as exc:
        # Fail closed: hardened's only filesystem boundary is Landlock, so a
        # kernel that cannot apply it refuses the run. (LandlockNotSupported
        # is a can't-happen safety net here -- isolation resolution already
        # probed the ABI before selecting hardened.)
        return f"could not apply agent Landlock confinement: {exc}"
    reporter.err(
        f"[agent6] agent-process Landlock: ABI {report.abi}, "
        f"{len(report.fs_read)} read / {len(report.fs_write)} write roots"
    )
    return None
