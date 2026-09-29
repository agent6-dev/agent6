# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Check a machine on the host before `machine run` or `create` drives it.

Refuses a run whose tool-network need the isolation cannot enforce, resolves the machine's
read-only protect paths, and builds the operator notify hook. The hook runs through
`finalize.run_notify_hook`, the one runner both notify hooks share.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from collections.abc import Callable, Mapping

from agent6 import kinds
from agent6.app import confine, finalize
from agent6.config import Config
from agent6.machine import StateSpec, ToolState


def machine_pass_env_refusal(cfg: Config, states: Mapping[str, StateSpec]) -> str | None:
    """Return the refusal naming every tool state asking for a disallowed variable, else None.

    The allowlist is `[machine].pass_env` in the global or repo config, never the machine's file.
    """
    allowed = set(cfg.machine.pass_env)
    asks = [
        f"[states.{name}] asks for {', '.join(n for n in state.pass_env if n not in allowed)}"
        for name, state in states.items()
        if isinstance(state, ToolState) and any(n not in allowed for n in state.pass_env)
    ]
    if not asks:
        return None
    return (
        f"{'; '.join(asks)}. [machine].pass_env does not allow these environment"
        " variables; list them there, in the global or repo config (never a machine"
        " overlay), to pass them."
    )


@dataclasses.dataclass(frozen=True, slots=True)
class NetworkRefusal:
    """Describe a run this host cannot honor.

    Attributes:
        message: The refusal the operator reads.
        fix: The config leaves that clear it, in an order sequential `config set` writes accept;
            empty when only a different isolation can.
    """

    message: str
    fix: tuple[tuple[str, str], ...] = ()


def machine_network_refusal(
    cfg: Config, isolation: kinds.IsolationLevel, tool_states: list[ToolState]
) -> NetworkRefusal | None:
    """Return the refusal when the machine's tool-network needs cannot be honored, else None.

    Layers the machine rules over `check_network_support`. `hardened` cannot isolate one
    tool's network, so a state that requires isolation is refused rather than mis-confined.
    A networked state under `network` in {"session", "auto"} is a config conflict on any
    isolation. Each refusal carries the fix its message names.

    Args:
        cfg: The resolved config.
        isolation: The isolation level the run resolved.
        tool_states: The machine's tool states.

    Returns:
        The refusal, or None when the host can honor the machine.
    """
    has_allow = any(s.network == "host" for s in tool_states)
    has_block = any(s.network == "none" for s in tool_states)
    # No config leaf gives hardened a per-tool netns; only an asking state justifies 'host'.
    hardened_fix: tuple[tuple[str, str], ...] = (
        () if has_block else (("sandbox.network", "host" if has_allow else "auto"),)
    )
    net_err = confine.check_network_support(cfg, isolation)
    if net_err is not None:
        return NetworkRefusal(net_err, hardened_fix)
    tn = cfg.sandbox.network
    no_tool_net = tn in ("session", "auto")  # both keep the tool off the host network
    if has_allow and no_tool_net:
        if isolation == "hardened":
            return NetworkRefusal(
                'a tool state sets network = "host" but sandbox.network ='
                f" {tn!r}. The hardened isolation cannot single out one tool's"
                " network namespace; let tools share the host network with"
                " sandbox.network = 'host', or run on strict for explicit"
                " per-tool egress.",
                hardened_fix,
            )
        return NetworkRefusal(
            'a tool state sets network = "host" but sandbox.network ='
            f" {tn!r}. Set sandbox.network = 'only_explicit_states' for"
            " explicit per-tool egress.",
            (("sandbox.network", "only_explicit_states"),),
        )
    if has_block and isolation == "hardened":
        return NetworkRefusal(
            'a tool state sets network = "none" (network must be denied),'
            " but the hardened isolation can't isolate one tool's network. Run on"
            ' strict, or use network = "auto" to tolerate the host network.'
        )
    return None


def machine_protect_paths(
    machine_path: pathlib.Path, cwd: pathlib.Path
) -> tuple[pathlib.Path, ...]:
    """Return the machine file and its `scripts/` bundle, to mark read-only in run jails.

    Only paths under the jail-mounted cwd are listed: a path outside it is not in the child's
    view.
    """
    cwd_r = cwd.resolve()
    out: list[pathlib.Path] = []
    for p in (machine_path, machine_path.parent / "scripts"):
        rp = p.resolve()
        if rp.exists() and rp.is_relative_to(cwd_r):
            out.append(rp)
    return tuple(out)


def _stderr_note(message: str) -> None:
    """Write a notice to stderr; the machine's stdout is the operator's run output."""
    print(f"[agent6] {message}", file=sys.stderr)


def build_machine_notify_hook(
    cfg: Config, machine_id: str, root: pathlib.Path
) -> Callable[[str, str, str, str], None] | None:
    """Return the operator hook fired on `machine.notify` and `machine.end`, or None.

    The argv is `[machine.notify].on_event`, the mirror of `[notify].on_complete`.
    """
    notify = cfg.machine.notify
    if not notify.on_event:
        return None

    def fire(kind: str, state: str, message: str, level: str) -> None:
        env = finalize.hook_env(
            AGENT6_MACHINE_ID=machine_id,
            AGENT6_MACHINE_DIR=str(root),
            AGENT6_MACHINE_EVENT=kind,
            AGENT6_MACHINE_STATE=state,
            AGENT6_MACHINE_MESSAGE=message,
            AGENT6_MACHINE_LEVEL=level,
        )
        finalize.run_notify_hook(
            notify.on_event,
            env,
            timeout_s=notify.timeout_s,
            label="machine.notify hook",
            note=_stderr_note,
        )

    return fire
