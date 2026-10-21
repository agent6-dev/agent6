# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Check a machine on the host before `machine run` or `create` drives it.

Refuses a state's `pass_env` the operator did not allow, resolves the machine's read-only
protect paths, and builds the operator notify hook. The hook runs through
`finalize.run_notify_hook`, the one runner both notify hooks share.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable, Mapping

from agent6.app import finalize
from agent6.config import Config
from agent6.machine import StateSpec, ToolState


def machine_pass_env_refusal(cfg: Config, states: Mapping[str, StateSpec]) -> str | None:
    """Return the refusal naming every tool state asking for a disallowed variable, else None.

    Args:
        cfg: The operator's config: the allowlist is its `[machine].pass_env`, from the global
            or repo config, never the machine's file.
        states: The machine's states, by name.

    Returns:
        The refusal, or None when the allowlist covers every variable asked for.
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


def machine_protect_paths(
    machine_path: pathlib.Path, cwd: pathlib.Path, config_paths: tuple[pathlib.Path, ...] = ()
) -> tuple[pathlib.Path, ...]:
    """Return the machine file, its `scripts/` bundle and the `--config` files, read-only in jails.

    The `--config` files are part of the operator's ceiling, so a state must not rewrite them.

    Args:
        machine_path: The machine file.
        cwd: The working directory the jails mount.
        config_paths: The `--config` files.

    Returns:
        Those that exist under `cwd`, resolved.
    """
    cwd_r = cwd.resolve()
    out: list[pathlib.Path] = []
    for p in (machine_path, machine_path.parent / "scripts", *config_paths):
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

    Args:
        cfg: The machine's resolved config.
        machine_id: The hook's `AGENT6_MACHINE_ID`.
        root: The instance dir, passed as `AGENT6_MACHINE_DIR`.

    Returns:
        The hook, called with the event (`notify` or `end`), the state, the message and the
        level; None when `on_event` is empty.
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
