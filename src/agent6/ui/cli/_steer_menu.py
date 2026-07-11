# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The interactive pause menu for a foreground CLI run (Ctrl-C, then decide).

Line input comes from ``_menu_input`` on Unix: editing, in-process history
(recall an earlier steer with Up), and a fish-style Tab preview of the slash
commands (Tab cycles the matches, descriptions shown). Windows has no termios,
so it keeps the plain one-line prompt (``_steer`` gates on
:func:`agent6.ui.cli._menu_input.menu_capable`). Info commands answer from the
run's event log and re-prompt, so the operator can inspect the run before
steering it.

Parsing rule: a command fires only when it is the WHOLE line (one ``/token``;
a unique prefix like ``/sta`` works, an ambiguous one re-asks). Any line with
a space -- or not starting with ``/`` -- is sent to the run verbatim as the
steering instruction, so no quoting is ever needed:

    /status   run status: tasks, tools, cost, ctx, profile
    /tasks    the task graph with statuses
    /compact  compact the context before the next model call
    /continue resume unchanged (same as Enter)
    /stop     stop the run now (resumable with `agent6 resume`)
    /detach   keep the run going in the background
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from agent6.config.layer import load_effective
from agent6.models.registry import context_window
from agent6.paths import data_dir
from agent6.skills import discover_skills, resolve_states, skill_search_dirs
from agent6.ui.bridge.approval import request_compact
from agent6.ui.cli._menu_input import menu_capable, menu_input
from agent6.ui.viewmodel import fold_run, tail_events
from agent6.ui.viewmodel.format import TASK_STATUS_GLYPH, format_cost
from agent6.ui.viewmodel.state import RunState, run_status_label

PROMPT = "[agent6] paused: Enter=continue · type to steer · /help: "

# Command -> one-line help. The Tab preview menu and /help both read this table.
COMMANDS: dict[str, str] = {
    "/status": "run status: tasks, tools, cost, context, profile",
    "/tasks": "the task graph with statuses",
    "/compact": "compact the context before the next model call",
    "/continue": "resume the run unchanged (same as Enter)",
    "/stop": "stop the run now (resume later with `agent6 resume`)",
    "/detach": "keep the run going in the background",
    "/help": "this list",
}


def skill_menu_table() -> dict[str, tuple[str, str]]:
    """``/name`` -> (description, full SKILL.md text) for enabled skills.

    Built-in commands always win a name collision, so ``/status`` can never
    be shadowed by a skill. A broken config or store degrades to no skill
    commands, loudly, without breaking the pause prompt.
    """
    try:
        cfg = load_effective(Path.cwd()).config
        if not cfg.skills.enabled:
            return {}
        found, _warns = discover_skills(
            skill_search_dirs(cfg.skills.extra_dirs, data_dir() / "skills")
        )
        resolved = resolve_states(found, cfg.skills.state)
    except Exception as exc:  # the pause prompt must survive any config error
        print(f"[agent6] skill commands unavailable: {exc}")
        return {}
    return {
        f"/{s.name}": (s.description, s.text)
        for s in (*resolved.enabled, *resolved.always)
        if f"/{s.name}" not in COMMANDS
    }


def skill_steer_payload(name: str, text: str, args: str) -> str:
    """The steer message a ``/skill-name [args]`` menu line injects."""
    args_line = f"\nSkill arguments: {args}" if args else ""
    return (
        f"Apply the operator-installed skill {name!r} for the rest of this run."
        f"{args_line}\n\n"
        f'<skill name="{name}">\n{text.rstrip()}\n</skill>'
    )


def normalize_steer_choice(line: str | None) -> str | None:
    """Map a mid-run menu line to a canonical action: None/'' continue,
    'abort' stop, 'detach' keep-running-in-background, else the instruction."""
    if line is None:
        return None
    choice = line.strip()
    low = choice.lower()
    if low in ("q", "quit", "stop", "abort"):
        return "abort"
    if low in ("d", "detach"):
        return "detach"
    return choice


# Steer lines accepted this process, for Up-arrow recall across pauses.
_HISTORY: list[str] = []


def _menu_read(prompt: str) -> str:
    return menu_input(prompt, COMMANDS, _HISTORY)


def _fold(run_dir: Path) -> RunState:
    return fold_run(tail_events(run_dir / "logs.jsonl", follow=False))


def _read_profile(run_dir: Path) -> str:
    """The effective profile the run started with (manifest.json), or ""."""
    try:
        data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    profile = data.get("profile")
    return profile if isinstance(profile, str) else ""


def _print_status(run_dir: Path) -> None:
    s = _fold(run_dir)
    label = run_status_label(s) if s.finished else "running"
    done = sum(1 for t in s.tasks if t.status in ("passed", "skipped"))
    tasks = f"{done}/{len(s.tasks)}" if s.tasks else "—"
    role = s.last_role
    model = f"{role.role}/{role.model}" if role else "—"
    cost = format_cost(s.budget.usd_total, partial=s.budget.usd_partial)
    ctx = ""
    if role is not None and role.ctx_tokens > 0:
        window = context_window(role.provider, role.model) if role.model else None
        pct = f" ({min(100, round(100 * role.ctx_tokens / window))}%)" if window else ""
        ctx = f" · ctx {role.ctx_tokens:,} tok{pct}"
    profile = _read_profile(run_dir)
    prof = f" · profile {profile}" if profile else ""
    print(f"[agent6] {label} · tasks {tasks} · {len(s.tool_calls)} tools · cost {cost}{ctx}{prof}")
    print(f"         model {model} · task: {s.user_task[:80]}")


def _print_tasks(run_dir: Path) -> None:
    s = _fold(run_dir)
    if not s.tasks:
        print("[agent6] (no tasks yet)")
        return
    for tv in s.tasks:
        icon = TASK_STATUS_GLYPH.get(tv.status, "·")
        marker = "▸ " if tv.is_cursor else ""
        print(f"  {'  ' * tv.depth}{marker}{icon} {tv.title}")


def _print_help() -> None:
    width = max(len(c) for c in COMMANDS)
    for cmd, what in COMMANDS.items():
        print(f"  {cmd:<{width}}  {what}")
    print("  anything else is sent to the run as a steering instruction")


# Commands that end the menu, mapped to the canonical steer action.
_ACTIONS: dict[str, str] = {"/continue": "", "/stop": "abort", "/detach": "detach"}


def _run_info_command(cmd: str, run_dir: Path) -> None:
    """Run a print-and-re-prompt command (everything not in ``_ACTIONS``)."""
    if cmd == "/help":
        _print_help()
    elif cmd == "/status":
        _print_status(run_dir)
    elif cmd == "/tasks":
        _print_tasks(run_dir)
    elif cmd == "/compact":
        request_compact(run_dir)
        print("[agent6] compaction requested — applies before the next model call")


def pause_menu(  # noqa: PLR0911, PLR0912
    run_dir: Path, *, input_fn: Callable[[str], str] | None = None
) -> str | None:
    """The interactive pause menu. Returns the canonical steer action: None/''
    continue, 'abort' stop now, 'detach' background, else the instruction sent
    verbatim. A command must be the whole line (unique prefixes fire, ambiguous
    ones re-ask); info commands print and re-prompt. EOF (Ctrl-D) continues."""
    skills = skill_menu_table()
    if input_fn is None:
        if menu_capable():
            display = {**COMMANDS, **{c: d[:70] for c, (d, _t) in skills.items()}}
            input_fn = lambda p: menu_input(p, display, _HISTORY)  # noqa: E731
        else:
            input_fn = input
    while True:
        try:
            line = input_fn(PROMPT)
        except EOFError:
            return None
        stripped = line.strip()
        if not stripped:
            return ""  # Enter: continue the run unchanged
        if not stripped.startswith("/"):
            return stripped  # a steering instruction, sent verbatim
        first, _, args = stripped.partition(" ")
        word = first.lower()
        if word in ("/h", "/?"):
            word = "/help"
        if args:
            # Only a skill command takes arguments; any other line with spaces
            # stays a verbatim steer (the pre-skills contract).
            smatches = [word] if word in skills else [c for c in skills if c.startswith(word)]
            if len(smatches) == 1:
                return skill_steer_payload(smatches[0][1:], skills[smatches[0]][1], args.strip())
            return stripped
        if word in COMMANDS or word in skills:  # exact match (never both: the
            # table builder drops skills that collide with a built-in)
            matches = [word]
        else:
            builtin = [c for c in COMMANDS if c.startswith(word)]
            matches = builtin + [c for c in skills if c.startswith(word) and c not in builtin]
        if len(matches) > 1:
            print(f"[agent6] ambiguous: {'  '.join(matches)} — type a bit more")
        elif not matches:
            print(
                f"[agent6] unknown command {word!r} — /help lists them"
                " (a line with spaces is sent as a steer)"
            )
        elif matches[0] in _ACTIONS:
            return _ACTIONS[matches[0]]
        elif matches[0] in skills:
            return skill_steer_payload(matches[0][1:], skills[matches[0]][1], "")
        else:
            _run_info_command(matches[0], run_dir)
