# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The interactive pause menu for a foreground CLI run (Ctrl-C, then decide).

Readline-backed on Unix: line editing, in-process history (recall an earlier
steer with Up), and Tab completion of the slash commands. Windows has no
``readline``, so it keeps the plain one-line prompt (``_steer`` gates on
:func:`readline_capable`). Info commands answer from the run's event log and
re-prompt, so the operator can inspect the run before steering it.

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

import contextlib
import json
import sys
from collections.abc import Callable, Generator
from pathlib import Path

try:  # Unix line editing; absent on Windows -> the caller uses the plain prompt
    import readline
except ImportError:  # pragma: no cover - exercised only on Windows
    readline = None  # type: ignore[assignment]

from agent6.models.registry import context_window
from agent6.ui.bridge.approval import request_compact
from agent6.ui.viewmodel import fold_run, tail_events
from agent6.ui.viewmodel.format import TASK_STATUS_GLYPH, format_cost
from agent6.ui.viewmodel.state import RunState, run_status_label

PROMPT = "[agent6] paused: Enter=continue · type to steer · /help: "

# Command -> one-line help. The completer and /help both read this table.
COMMANDS: dict[str, str] = {
    "/status": "run status: tasks, tools, cost, context, profile",
    "/tasks": "the task graph with statuses",
    "/compact": "compact the context before the next model call",
    "/continue": "resume the run unchanged (same as Enter)",
    "/stop": "stop the run now (resume later with `agent6 resume`)",
    "/detach": "keep the run going in the background",
    "/help": "this list",
}


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


def readline_capable() -> bool:
    """True when the pause menu can own the terminal line: readline exists
    (Unix) and both std streams are the interactive terminal (a redirected
    stream means the prompt must go through /dev/tty instead)."""
    return readline is not None and sys.stdin.isatty() and sys.stdout.isatty()


def _complete(text: str, state: int) -> str | None:
    """Readline completer for the FIRST word only: Tab on an empty line lists
    every command, Tab on a ``/prefix`` cycles the matches, and Tab inside
    steer text is inert (commands never fire mid-line)."""
    if readline is None:  # pragma: no cover - Windows
        return None
    head = readline.get_line_buffer()[: readline.get_begidx()]
    if head.strip():
        return None  # not the first word: the line is a steering instruction
    if text and not text.startswith("/"):
        return None
    matches = [c for c in COMMANDS if c.startswith(text)]
    return matches[state] if state < len(matches) else None


@contextlib.contextmanager
def _line_editing() -> Generator[None]:
    """Install the slash-command completer for the menu, restoring the prior
    readline state after (the $EDITOR hook or an embedding REPL may own it)."""
    if readline is None:  # pragma: no cover - Windows
        yield
        return
    old_completer = readline.get_completer()
    old_delims = readline.get_completer_delims()
    readline.set_completer(_complete)
    readline.set_completer_delims(" \t")  # keep "/" inside the completed word
    readline.parse_and_bind("tab: complete")
    try:
        yield
    finally:
        readline.set_completer(old_completer)
        readline.set_completer_delims(old_delims)


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


def pause_menu(run_dir: Path, *, input_fn: Callable[[str], str] = input) -> str | None:
    """The readline pause menu. Returns the canonical steer action: None/''
    continue, 'abort' stop now, 'detach' background, else the instruction sent
    verbatim. A command must be the whole line (unique prefixes fire, ambiguous
    ones re-ask); info commands print and re-prompt. EOF (Ctrl-D) continues."""
    with _line_editing():
        while True:
            try:
                line = input_fn(PROMPT)
            except EOFError:
                return None
            stripped = line.strip()
            if not stripped:
                return ""  # Enter: continue the run unchanged
            if not stripped.startswith("/") or " " in stripped:
                return stripped  # a steering instruction, sent verbatim
            word = stripped.lower()
            if word in ("/h", "/?"):
                word = "/help"
            matches = [word] if word in COMMANDS else [c for c in COMMANDS if c.startswith(word)]
            if len(matches) > 1:
                print(f"[agent6] ambiguous: {'  '.join(matches)} — type a bit more")
            elif not matches:
                print(
                    f"[agent6] unknown command {word!r} — /help lists them"
                    " (a line with spaces is sent as a steer)"
                )
            elif matches[0] in _ACTIONS:
                return _ACTIONS[matches[0]]
            else:
                _run_info_command(matches[0], run_dir)
