# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The pause menu of a foreground CLI run: Ctrl-C, then decide.

Line input comes from `_menu_input` where termios can own the line: editing,
history recall seeded from the session journal, and a Tab preview of the slash
commands; elsewhere the plain one-line prompt. A command fires only when it is
the whole line, typed in full. A line with a space is answered here when its
word is `/compact`, `/btw` or a skill; `/pin` and `/parallel` travel with their
word lowercased; any other line goes to the run verbatim as the steering
instruction, so no quoting is needed. Info commands print and re-prompt.
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
from collections.abc import Callable

from agent6 import budget, directive, paths
from agent6 import skills as agent6_skills
from agent6.config import layer
from agent6.graph import order
from agent6.sessions import ipc, layout, manifest
from agent6.tools import background
from agent6.ui import directives
from agent6.ui.cli import _common, _menu_input
from agent6.viewmodel import (
    fold_session,
    format,
    operator_inputs,
    restate,
    state,
    status_for_session_dir,
    tail_events,
    task_snippet,
)

PROMPT = "[agent6] paused: Enter=continue · type to steer · /help: "

# The commands only a paused run can offer; `/pin` is the one word the menu describes differently.
MENU_ONLY_HELP: dict[str, str] = {
    "/status": "run status: tasks, tools, cost, context, preset",
    "/tasks": "the task graph with statuses",
    "/pin": "list pinned instructions (pin one with `/pin <text>`)",
    "/continue": "resume the run unchanged (same as Enter)",
    "/exit": "stop the run and leave (no follow-up prompt; resume later)",
    "/detach": "keep the run going in the background",
    "/help": "this list",
}

# Command to help line, read by the Tab preview and /help; a shared word's line is the owner's.
MENU_COMMANDS: dict[str, str] = {
    "/status": MENU_ONLY_HELP["/status"],
    "/tasks": MENU_ONLY_HELP["/tasks"],
    "/pin": MENU_ONLY_HELP["/pin"],
    "/compact": directive.STEER_COMMANDS["/compact"],
    "/parallel": directive.STEER_COMMANDS["/parallel"],
    "/btw": directive.STEER_COMMANDS["/btw"],
    "/task": directive.STEER_COMMANDS["/task"],
    "/standing": directive.STEER_COMMANDS["/standing"],
    "/retire": directive.STEER_COMMANDS["/retire"],
    "/shells": directive.STEER_COMMANDS["/shells"],
    "/restate": directive.STEER_COMMANDS["/restate"],
    "/undo": directive.STEER_COMMANDS["/undo"],
    "/continue": MENU_ONLY_HELP["/continue"],
    "/stop": directive.STEER_COMMANDS["/stop"],
    "/exit": MENU_ONLY_HELP["/exit"],
    "/detach": MENU_ONLY_HELP["/detach"],
    "/help": MENU_ONLY_HELP["/help"],
}


def _without_btw() -> dict[str, str]:
    """Return the menu without `/btw`, for a surface with nothing to spawn one from."""
    return {cmd: help_ for cmd, help_ in MENU_COMMANDS.items() if cmd != "/btw"}


def skill_menu_table(config_path: pathlib.Path | None = None) -> dict[str, tuple[str, str]]:
    """Return `/name` to (description, SKILL.md text) for the enabled skills.

    A built-in command wins a name collision. A broken config or store degrades to
    no skill commands, loudly.

    Args:
        config_path: The invocation's `--config`.
    """
    try:
        cfg = layer.load_effective(pathlib.Path.cwd(), config_path).config
        resolved = agent6_skills.operator_skills(
            cfg.skills.enabled, cfg.skills.extra_dirs, cfg.skills.state, paths.data_dir() / "skills"
        )
    except Exception as exc:  # the pause prompt survives any config error
        print(f"[agent6] skill commands unavailable: {exc}")
        return {}
    return {
        f"/{s.name}": (s.description, s.text)
        for s in (*resolved.enabled, *resolved.always)
        if f"/{s.name}" not in MENU_COMMANDS
    }


@dataclasses.dataclass(slots=True)
class _Recall:
    """The pause prompt's history, seeded once per session from its journal, then grown."""

    lines: list[str] = dataclasses.field(default_factory=list)
    seeded_from: str | None = None

    def seed(self, session_dir: pathlib.Path) -> None:
        """Seed the task and every steer, once; reseeding would drop the lines accepted since."""
        if self.seeded_from == str(session_dir):
            return
        self.seeded_from = str(session_dir)
        recorded = operator_inputs(tail_events(session_dir / layout.LOGS_NAME, follow=False))
        self.lines[:] = [" ".join(text.split()) for text in recorded]


_RECALL = _Recall()


def _fold(session_dir: pathlib.Path) -> state.SessionState:
    """Return the session's state, folded from its log."""
    return fold_session(tail_events(session_dir / layout.LOGS_NAME, follow=False))


def _read_preset(session_dir: pathlib.Path) -> str:
    """Return the preset the run started with, or ""."""
    try:
        return manifest.read_manifest(session_dir).harness.preset
    except manifest.ManifestError:
        return ""


def _print_status(session_dir: pathlib.Path) -> None:
    """Print the run's status line: tasks, tools, cost, context and preset."""
    s = _fold(session_dir)
    # The dir's probes too: a gone worker reads as running from the fold alone.
    label = format.status_label(*status_for_session_dir(session_dir, state.status_facts(s)))
    done = sum(1 for t in s.tasks if t.status in order.DONE_STATUSES)
    tasks = f"{done}/{len(s.tasks)}" if s.tasks else "—"
    role = s.last_role
    model = f"{role.role}/{role.model}" if role else "—"
    cost = budget.format_usd(s.budget.usd_total, partial=s.budget.usd_partial)
    ctx = ""
    if role is not None and role.ctx_tokens > 0:
        fill = state.context_fill(s)
        pct = f" ({fill}%)" if fill is not None else ""
        ctx = f" · ctx {role.ctx_tokens:,} tok{pct}"
    if s.compact_elided:
        ctx += f" · elided {s.compact_elided} ({s.compact_gists_live} gists)"
    if s.pins:
        ctx += f" · pins {len(s.pins)}"
    preset = _read_preset(session_dir)
    prof = f" · preset {preset}" if preset else ""
    calls = _common.plural(len(s.tool_calls), "tool")
    print(f"[agent6] {label} · tasks {tasks} · {calls} · cost {cost}{ctx}{prof}")
    print(f"         model {model} · task: {task_snippet(s.user_task, max_chars=80)}")


def _print_pins(session_dir: pathlib.Path) -> None:
    """Print the recorded pins for a bare `/pin`; `/pin <text>` travels as a steer."""
    s = _fold(session_dir)
    if not s.pins:
        print("[agent6] no pinned instructions; pin one with `/pin <text>`")
        return
    print(f"[agent6] {len(s.pins)} pinned (survive context compaction; `/pin <text>` adds):")
    for i, pin in enumerate(s.pins, start=1):
        print(f"  {i}. {pin}")


def _print_tasks(session_dir: pathlib.Path) -> None:
    """Print the task tree with statuses."""
    s = _fold(session_dir)
    if not s.tasks:
        print("[agent6] (no tasks yet)")
        return
    # The fold's views carry their depth; the id leads the line, as `/retire` names a task.
    for tv in s.tasks:
        icon = format.TASK_STATUS_GLYPH.get(tv.status, "·")
        marker = "▸ " if tv.is_cursor else ""
        print(f"  {format.short_task_id(tv.id):>3}  {'  ' * tv.depth}{marker}{icon} {tv.title}")


def _print_help(offered: dict[str, str]) -> None:
    """Print the offered commands and the key hints."""
    width = max(len(c) for c in offered)
    for cmd, what in offered.items():
        print(f"  {cmd:<{width}}  {what}")
    print("  anything else is sent to the run as a steering instruction")
    print("  Up recalls this session's messages · Ctrl-R searches them · Tab previews commands")


# Starts a btw and delivers its answer to the console view; None makes `/btw` say so.
BtwRunner = Callable[[str, pathlib.Path], tuple[bool, str]]


def _print_shells(session_dir: pathlib.Path) -> None:
    """Print the run's background commands, read off disk like every other surface."""
    lines = background.roster_from_dir(session_dir / background.SHELLS_DIR)
    if not lines:
        print("[agent6] no background commands this run")
        return
    for line in lines:
        print(f"  {line}")


def _start_btw(cmd: str, session_dir: pathlib.Path, runner: BtwRunner | None) -> str:
    """Return the line to print after starting a btw through the runner."""
    question = directive.parse_btw(cmd)
    if not question:
        return "[agent6] ask something: `/btw <question>`"
    if runner is None:
        return "[agent6] /btw needs a live run with a terminal"
    return runner(question, session_dir)[1]


# Commands that end the menu, mapped to the canonical steer action.
_ACTIONS: dict[str, str] = {
    "/continue": "",
    "/stop": "abort",
    "/exit": "exit",
    "/detach": "detach",
    # Verbatim: the harness parses the directive itself.
    "/undo": "/undo",
}


def _run_info_command(
    cmd: str,
    session_dir: pathlib.Path,
    btw_runner: BtwRunner | None = None,
) -> None:
    """Run a command that prints and re-prompts (everything not in `_ACTIONS`)."""
    if cmd == "/help":
        _print_help(MENU_COMMANDS if btw_runner is not None else _without_btw())
    elif cmd == "/status":
        _print_status(session_dir)
    elif cmd == "/tasks":
        _print_tasks(session_dir)
    elif cmd == "/pin":
        _print_pins(session_dir)
    elif cmd == "/parallel":
        print("[agent6] fan out needs a task: `/parallel [N|models] <task>`")
    elif cmd == "/shells":
        _print_shells(session_dir)
    elif cmd == "/restate":
        print(restate(list(tail_events(session_dir / layout.LOGS_NAME, follow=False))))
    elif cmd.startswith("/btw"):
        print(_start_btw(cmd, session_dir, btw_runner))
    elif cmd.startswith(("/task", "/standing", "/retire")):
        # The one owner of what a composer line does; `/btw` stays local for its runner.
        _did, said = directives.act_on_directive(session_dir, cmd) or (False, "")
        print(f"[agent6] {said}")


def _line_reader(
    session_dir: pathlib.Path, offered: dict[str, str], skills: dict[str, tuple[str, str]]
) -> Callable[[str], str]:
    """Return the terminal's line reader: the menu, polling the session's steer file.

    Args:
        session_dir: The run's dir.
        offered: The commands and their help.
        skills: The skill commands.
    """
    arrived = functools.partial(ipc.steer_answer_written, session_dir)
    _RECALL.seed(session_dir)
    display = {**offered, **{c: d[:70] for c, (d, _t) in skills.items()}}
    return lambda p: _menu_input.read_line(p, display, _RECALL.lines, until=arrived)


def pause_menu(
    session_dir: pathlib.Path,
    *,
    input_fn: Callable[[str], str] | None = None,
    btw_runner: BtwRunner | None = None,
    config_path: pathlib.Path | None = None,
) -> str | None:
    """Run the pause menu until a line ends it.

    A steer a front-end writes while the menu is open ends it and is the answer.

    Args:
        session_dir: The run's dir.
        input_fn: Reads one line for a prompt; the terminal's reader when None.
        btw_runner: Starts a btw; None withholds `/btw`.
        config_path: The invocation's `--config`.

    Returns:
        The steer action: "" or None continues, "abort" stops, "exit" stops and leaves,
        "detach" backgrounds, anything else is the instruction sent verbatim.
    """
    skills = skill_menu_table(config_path)
    # A surface that cannot spawn a sibling session never offers `/btw`.
    offered = MENU_COMMANDS if btw_runner is not None else _without_btw()
    if input_fn is None:
        input_fn = _line_reader(session_dir, offered, skills)
    while True:
        try:
            line = input_fn(PROMPT)
        except EOFError:
            return None
        except _menu_input.LineSuperseded:
            print("[agent6] a steer arrived from a front-end; taking it")
            return ipc.take_steer_answer(session_dir) or ""
        answer = _answer_line(line, session_dir, btw_runner, skills)
        if not isinstance(answer, _Again):
            return answer


@dataclasses.dataclass(frozen=True, slots=True)
class _Again:
    """A line that printed: the menu asks again, the plain prompt continues the run."""


AGAIN = _Again()
# Parsed out of steer text by case-sensitive parsers, so their word travels lowercased.
_LOOP_DIRECTIVES = ("/pin", "/parallel")


def _answer_line(  # noqa: PLR0911, PLR0912
    line: str,
    session_dir: pathlib.Path,
    btw_runner: BtwRunner | None,
    skills: dict[str, tuple[str, str]],
) -> str | _Again:
    """Return one typed line's answer, the same at both prompts; `pause_menu` has the contract."""
    stripped = line.strip()
    if not stripped:
        return ""
    if not stripped.startswith("/"):
        return stripped
    first, _, args = stripped.partition(" ")
    word = first.lower()
    if word in ("/h", "/?"):
        word = "/help"
    if args:
        # `/btw` is the menu's own; every other directive belongs to the owner composers share.
        if word == "/btw":
            print(_start_btw(stripped, session_dir, btw_runner))
            return AGAIN
        acted = directives.act_on_directive(session_dir, stripped)
        if acted is not None:
            print(f"[agent6] {acted[1]}")
            return AGAIN
        if word in skills:
            # A skill command travels as typed; the harness expands it.
            return stripped
        if word in _LOOP_DIRECTIVES:
            return f"{word} {args.strip()}"
        if word == "/now":
            # No call is in flight at a boundary, so the steer lands at once without the word.
            return args.strip()
        return stripped
    if word == "/compact":
        # Complete on its own; the owner composers share acts on it.
        acted = directives.act_on_directive(session_dir, stripped)
        assert acted is not None
        print(f"[agent6] {acted[1]}")
        return AGAIN
    if word not in MENU_COMMANDS and word not in skills:
        # A prefix drives Tab completion, never an action, so a new command re-points no habit.
        near = sorted(c for c in (*MENU_COMMANDS, *skills) if c.startswith(word) and c != word)
        hint = f"; did you mean {'  '.join(near)}?" if near else "; /help lists them"
        print(f"[agent6] unknown command {word!r}{hint} (a line with spaces is sent as a steer)")
        return AGAIN
    if word in _ACTIONS:
        return _ACTIONS[word]
    if word in skills:
        return word
    _run_info_command(word, session_dir, btw_runner)
    return AGAIN


def pause_line(
    line: str | None,
    session_dir: pathlib.Path,
    *,
    btw_runner: BtwRunner | None = None,
    config_path: pathlib.Path | None = None,
) -> str | None:
    """Return the plain prompt's answer for one line; a line that printed continues the run.

    Args:
        line: The typed line; None on EOF continues.
        session_dir: The run's dir.
        btw_runner: Starts a btw; None withholds `/btw`.
        config_path: The invocation's `--config`.

    Returns:
        The steer action, as `pause_menu` returns it.
    """
    if line is None:
        return None
    answer = _answer_line(line, session_dir, btw_runner, skill_menu_table(config_path))
    return "" if isinstance(answer, _Again) else answer
