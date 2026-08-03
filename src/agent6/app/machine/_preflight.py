# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Host-side preflight for `machine run`/`create`.

Before the engine composition drives a machine, these checks refuse a run that
can't be honored -- a tool-network need the isolation cannot enforce
(`machine_network_refusal`) -- and they resolve the machine's own read-only
protect paths (`machine_protect_paths`) and the operator notify hook
(`build_machine_notify_hook`). Pure computations plus the one host subprocess
(the notify hook, whose argv comes from `[machine.notify]`, never LLM output).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from agent6.app.confine import check_network_support
from agent6.app.finalize import hook_env
from agent6.app.machine._bundle import is_inside
from agent6.config import Config
from agent6.machine import ToolState
from agent6.types import IsolationLevel


def machine_network_refusal(
    cfg: Config, isolation: IsolationLevel, tool_states: list[ToolState]
) -> str | None:
    """A refusal message if this machine's tool-network needs can't be honored.

    Layers machine-specific rules on top of `check_network_support` (which
    handles agent_network=local / tool_network=only_explicit_states on
    `hardened`). On `hardened` per-tool isolation is impossible, so we refuse,
    rather than silently mis-confine, whenever isolation is *required*: by the
    operator (`tool_network = "block"`) or by a state (`allow_network = "block"`).
    A networked state under `tool_network` in {"block", "auto"} (both intend no
    tool network) is a config conflict and is refused on any isolation. Returns
    None when fine.
    """
    net_err = check_network_support(cfg, isolation)
    if net_err is not None:
        return net_err
    tn = cfg.sandbox.tool_network
    no_tool_net = tn in ("block", "auto")  # both intend the tool has no network
    has_allow = any(s.allow_network == "allow" for s in tool_states)
    has_block = any(s.allow_network == "block" for s in tool_states)
    if has_allow and no_tool_net:
        # Name the ACTUAL value (auto or block), not a hardcoded 'block': the
        # default is now 'auto', so telling the operator their setting is 'block'
        # misstates their config on a refusal surface.
        if isolation == "hardened":
            return (
                'a tool state sets allow_network = "allow" but sandbox.tool_network ='
                f" {tn!r}. The hardened isolation cannot single out one tool's"
                " network namespace; let tools share the host network with"
                " sandbox.tool_network = 'allow' and sandbox.agent_network = 'open',"
                " or run on strict for explicit per-tool egress."
            )
        return (
            'a tool state sets allow_network = "allow" but sandbox.tool_network ='
            f" {tn!r}. Set sandbox.tool_network = 'only_explicit_states' for"
            " explicit per-tool egress."
        )
    if tool_states and tn == "block" and isolation == "hardened":
        return (
            "isolating a machine's tool-state network requires the strict isolation"
            " (a per-tool network namespace); this host supports only 'hardened'."
            " Run on strict, or let tools share the host network with"
            " sandbox.tool_network = 'allow' (which also requires"
            " sandbox.agent_network = 'open')."
        )
    if has_block and isolation == "hardened":
        return (
            'a tool state sets allow_network = "block" (network must be denied),'
            " but the hardened isolation can't isolate one tool's network. Run on"
            ' strict, or use allow_network = "auto" to tolerate the host network.'
        )
    return None


def machine_protect_paths(machine_path: Path, cwd: Path) -> tuple[Path, ...]:
    """The machine's own ``.asm.toml`` + ``scripts/`` bundle, to mark read-only
    in run jails. Only paths under the jail-mounted cwd are enforceable (a path
    outside cwd isn't in the child's view, so it can't edit it anyway)."""
    cwd_r = cwd.resolve()
    out: list[Path] = []
    for p in (machine_path, machine_path.parent / "scripts"):
        rp = p.resolve()
        if rp.exists() and is_inside(rp, cwd_r):
            out.append(rp)
    return tuple(out)


def build_machine_notify_hook(
    cfg: Config, machine_id: str, root: Path
) -> Callable[[str, str, str, str], None] | None:
    """The operator notify hook fired on `machine.notify`/`machine.end`, or None.

    The argv comes from `[machine.notify].on_event`, operator-controlled and
    never LLM output, so it runs on the host OUTSIDE the jail (mirror of
    `[notify].on_complete`). Failures are logged and never change the exit code.
    """
    notify = cfg.machine.notify
    if not notify.on_event:
        return None

    def fire(kind: str, state: str, message: str, level: str) -> None:
        env = hook_env(
            AGENT6_MACHINE_ID=machine_id,
            AGENT6_MACHINE_DIR=str(root),
            AGENT6_MACHINE_EVENT=kind,
            AGENT6_MACHINE_STATE=state,
            AGENT6_MACHINE_MESSAGE=message,
            AGENT6_MACHINE_LEVEL=level,
        )
        try:
            subprocess.run(list(notify.on_event), env=env, timeout=notify.timeout_s, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"[agent6] machine.notify hook failed: {exc}", file=sys.stderr)

    return fire
