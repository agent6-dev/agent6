# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Machines page: browse, view, run, watch and create state machines.

Run and Create shell out to `agent6 machine run|create`, detached; View parses the
machine file in-process, the one reason ui depends on `agent6.machine`.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Callable
from typing import ClassVar

try:
    import textual
    from rich import text as rich_text
    from textual import app as textual_app
    from textual import binding as textual_binding
    from textual import containers, events, screen
    from textual import widgets as textual_widgets
    from textual.notifications import SeverityLevel
except ImportError as e:  # pragma: no cover - clear runtime message
    raise ImportError(
        "agent6 TUI requires the 'textual' package (part of the base install)."
        " Reinstall agent6, or `pip install textual`."
    ) from e

from agent6 import budget, paths
from agent6.app.machine import listing
from agent6.machine import (
    JournalError,
    MachineError,
    MachineJournal,
    MachineSpec,
    load_machine,
    render_mermaid,
    validate_semantics,
    write_stop_request,
)
from agent6.sessions import ipc, layout
from agent6.ui import notify, spawn
from agent6.ui.tui import composer, forms, menubar, modals, screen_chrome, theme
from agent6.ui.tui import prompts as tui_prompts
from agent6.viewmodel import (
    MachineState,
    MachineWatchCursor,
    NewestExecutionFold,
    fold_machine,
    format,
    machine_spend,
    machine_state,
    machine_verb_refusal,
    machine_verb_refusals,
    newest_state_log,
    probe_instance,
    verb_answer,
)
from agent6.viewmodel import events as viewmodel_events

_VERB_ACTIONS: dict[str, machine_state.MachineVerb] = {
    "steer": "steer",
    "poke": "poke",
    "stop": "stop",
}


def _list_drafts(agent6_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return the machine-create draft dirs, newest first."""
    drafts = layout.bucket_dir(agent6_dir, "machines")
    if not drafts.is_dir():
        return []
    out = [p for p in drafts.iterdir() if p.is_dir()]
    out.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return out


def _discrete_log_line(evt: dict[str, object], *, in_flight: bool = False) -> rich_text.Text | None:
    """Return a compact line for a non-streaming agent-log event, or None to skip it.

    Args:
        evt: The event.
        in_flight: The turn is still open; only then is a role.call marked as thinking,
            since a replayed finished turn would otherwise read as live.
    """
    t = evt.get("type")
    if t == "role.call":
        mark = " thinking…" if in_flight else ""
        return rich_text.Text(
            f"  → {evt.get('role', '')}/{evt.get('model', '')}{mark}", style="cyan"
        )
    if t == "tool.call":
        args = json.dumps(evt.get("args", {}), default=str)
        if len(args) > 80:
            args = args[:77] + "…"
        return rich_text.Text(f"  ⚙ {evt.get('name', '')} {args}", style="yellow")
    if t == "tool.result":
        ok = viewmodel_events.tool_result_ok(evt.get("ok"))
        mark = "✓" if ok else "✗"
        return rich_text.Text(f"  {mark} {evt.get('summary', '')}", style="green" if ok else "red")
    return None


# An answer submitted after the machine stopped: the per-state answer file has no reader.
_ANSWER_LOST = (
    "machine is not running; the answer reached nothing (poke it to wake a waiting machine)"
)


class MachineWatchScreen(composer.ApprovalKeys, screen_chrome.ScreenChrome, screen.Screen[None]):
    """The live view of a machine: its states, each transition, the agent state's reasoning.

    Polls every 0.5s. While open it is an answer front-end for the instance: the
    current agent state's approvals dock as the run views' answer row and its
    questions pop as modals; the machine's notifies and its end notify too.
    """

    HELP_TITLE: ClassVar = "agent6 machine — keys & actions"
    MENUS: ClassVar = (
        menubar.Menu("File", (menubar.MenuItem("Back", "close"),)),
        menubar.Menu(
            "Machine",
            (
                menubar.MenuItem("Steer the running state", "steer"),
                menubar.MenuItem("Message a waiting machine", "poke"),
                menubar.MenuItem("Stop at the next transition", "stop"),
            ),
        ),
        menubar.Menu(
            "Help",
            (
                menubar.MenuItem("Keys & actions", "help"),
                menubar.MenuItem("Command palette", "command_palette"),
            ),
        ),
    )
    COMMANDS: ClassVar = screen.Screen.COMMANDS | {screen_chrome.MenuCommands}
    FOOTER: ClassVar = (
        ("steer", "Steer"),
        ("poke", "Message"),
        ("stop", "Stop"),
        ("help", "Help"),
        ("close", "Back"),
    )
    BINDINGS: ClassVar = [
        *menubar.menu_bindings("machine watch", MENUS, footer=FOOTER),
        *composer.APPROVAL_KEY_BINDINGS,  # an open approval answers from any non-text focus
    ]
    APPROVAL_DOCK_BEFORE: ClassVar = "Footer"
    APPROVAL_LOST: ClassVar = _ANSWER_LOST
    APPROVAL_ROW_HINT: ClassVar = "(or click)"  # no composer here: the keys always answer
    CSS = (
        theme.PALETTE_CSS
        + """
    MachineWatchScreen { layers: base; }
    #mw-head { height: 3; border: round $primary; padding: 0 1; }
    #mw-states { width: 32%; border: round $primary; }
    #mw-log { width: 1fr; border: round $primary; padding: 0 1; }
    """
    )

    def __init__(self, instance_dir: pathlib.Path, spec: MachineSpec) -> None:
        super().__init__()
        self._root = instance_dir
        self._spec = spec
        self._journal = MachineJournal(instance_dir)
        self._cursor = MachineWatchCursor()
        self._pending = ""  # thinking and answer text, flushed in readable chunks
        self._ended = False
        # Every verb's refusal, read once per poll: the footer, the keys and the prompt gate agree.
        self._refusals = machine_verb_refusals(self._root, self._root.name)
        # The newest state log, folded incrementally, one fold per poll.
        self._execution_fold = NewestExecutionFold()
        self._prompts = tui_prompts.PromptDispatcher(
            self.app, answerable=self._answerable, lost=_ANSWER_LOST
        )
        self._end_notified = False
        self._steer_open = False

    def compose(self) -> textual_app.ComposeResult:
        """Yield the menu bar, the header, the state table beside the log, and the footer."""
        yield menubar.MenuBar(self.MENUS)
        yield textual_widgets.Static(id="mw-head")
        with containers.Horizontal():
            yield textual_widgets.DataTable(id="mw-states", cursor_type="none")
            yield textual_widgets.RichLog(id="mw-log", wrap=True, markup=False, highlight=False)
        yield textual_widgets.Footer()

    def on_mount(self) -> None:
        """Fill the state table, claim the instance as a front-end, seed the cursor and poll."""
        table = self.query_one("#mw-states", textual_widgets.DataTable)
        table.add_column(" ", key="mark")
        table.add_column("state", key="state")
        table.add_column("kind", key="kind")
        for name, state in self._spec.states.items():
            table.add_row("", name, state.kind, key=name)
        # The dir may not exist yet when watching a just-spawned run.
        paths.mkdir_for_real_user(self._root)
        # A per-process claim; concurrent web and TUI watchers each hold their own.
        ipc.register_frontend(self._root, os.getpid())
        try:
            events = self._journal.read()
        except JournalError:
            events = []  # the first poll surfaces the corruption in the header
        seeded = fold_machine(self._spec, events)
        self._cursor.seed_notifications(seeded)
        # An end that predates the open is history, not news.
        self._end_notified = seeded.ended is not None
        self._poll()
        self._poll_timer = self.set_interval(0.5, self._poll)

    def on_unmount(self) -> None:
        """Drop this process's front-end claim."""
        ipc.unregister_frontend(self._root, os.getpid())

    def _current_state_dir(self) -> pathlib.Path | None:
        """Return the current agent state's dir, where its answer files live."""
        log = newest_state_log(self._root)
        return log.parent if log is not None else None

    def action_close(self) -> None:
        """Pop the screen, or exit when `agent6 attach --tui` mounted it on the base screen."""
        if len(self.app.screen_stack) > 2:
            self.app.pop_screen()
        else:
            self.app.exit()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Return None to dim a verb's footer key where the viewmodel gate refuses it."""
        del parameters
        verb = _VERB_ACTIONS.get(action)
        if verb is None or not self._refusals[verb]:
            return True
        return None

    def on_key(self, event: events.Key) -> None:
        """A dimmed verb's key shows its refusal; a stop's is the CLI's note."""
        verb = self._verb_for_key(event.key)
        if verb is None or not self._refusals[verb]:
            return
        event.prevent_default()
        event.stop()
        self._explain_refusal(verb)

    def on_click(self, event: events.Click) -> None:
        """Show a dimmed footer key's refusal on click; the footer alone would only ring."""
        widget = event.widget
        action = getattr(widget, "action", None)
        if widget is None or not widget.has_class("-disabled") or not isinstance(action, str):
            return
        verb = _VERB_ACTIONS.get(action)
        if verb is not None and self._refusals[verb]:
            self._explain_refusal(verb)

    def _explain_refusal(self, verb: machine_state.MachineVerb) -> None:
        """Notify the verb's refusal; a stop's is a note, not a warning."""
        severity: SeverityLevel = "information" if verb == "stop" else "warning"
        self.app.notify(self._refusals[verb], severity=severity, timeout=6.0)

    def _verb_for_key(self, key: str) -> machine_state.MachineVerb | None:
        """Return the machine verb the key is bound to in BINDINGS."""
        for binding in self.BINDINGS:
            if isinstance(binding, textual_binding.Binding) and binding.key == key:
                return _VERB_ACTIONS.get(binding.action)
        return None

    def _set_refusals(self, refusals: dict[machine_state.MachineVerb, str]) -> None:
        """Take the poll's reading; the footer repaints on any verb's edge."""
        changed = any(bool(refusals[v]) != bool(self._refusals[v]) for v in _VERB_ACTIONS.values())
        self._refusals = refusals
        if changed:
            self.refresh_bindings()

    def _steerable(self) -> bool:
        """Return whether a steer reaches an open agent state, per the poll's refusals."""
        return not self._refusals["steer"]

    def _answerable(self) -> bool:
        """Return whether an answer reaches the worker, read fresh since it can die mid-prompt."""
        return not machine_verb_refusal(self._root, self._root.name, "answer")

    def action_steer(self) -> None:
        """Drop a steer request and open the steer box for the current agent state."""
        ok, refusal = verb_answer(self._root, self._root.name, "steer")
        if not ok or refusal:
            self.app.notify(refusal, severity="warning", timeout=6.0)
            return
        state_dir = self._current_state_dir()
        if state_dir is None or self._steer_open:
            self.app.notify("no agent state to steer", severity="warning", timeout=4.0)
            return
        ipc.clear_steer_answer(state_dir)
        if not ipc.request_steer(state_dir):
            self.app.notify("could not write the steer request", severity="warning", timeout=4.0)
            return
        self._steer_open = True
        self.app.push_screen(modals.SteerModal(), self._on_steer(state_dir))

    def _on_steer(self, state_dir: pathlib.Path) -> Callable[[str | None], None]:
        """Return the steer modal's callback, which re-reads the gate before writing."""

        def cb(answer: str | None) -> None:
            self._steer_open = False
            refusal = machine_verb_refusal(self._root, self._root.name, "steer")
            if refusal:
                self.app.notify(refusal, severity="warning", timeout=6.0)
                return
            ipc.write_steer_answer(state_dir, answer or "")

        return cb

    def action_stop(self) -> None:
        """Ask the machine to park at its next transition; the instance stays resumable."""
        ok, answer = verb_answer(self._root, self._root.name, "stop")
        if not ok:
            self.app.notify(answer, severity="warning", timeout=6.0)
            return
        if answer:
            self.app.notify(answer, timeout=6.0)
            return
        write_stop_request(self._root)
        self.app.notify("stop requested; the machine parks at its next boundary", timeout=4.0)

    def action_poke(self) -> None:
        """Open the message box for a waiting machine."""
        ok, refusal = verb_answer(self._root, self._root.name, "poke")
        if not ok or refusal:
            self.app.notify(refusal, severity="warning", timeout=6.0)
            return
        self.app.push_screen(
            modals.TextInputModal("Send a message to the machine (poke):", "message…"),
            self._on_poke,
        )

    def _on_poke(self, message: str | None) -> None:
        """Write the poke, re-reading the gate since the wait can close while the modal is open."""
        if message is None:
            return
        refusal = machine_verb_refusal(self._root, self._root.name, "poke")
        if refusal:
            self.app.notify(refusal, severity="warning", timeout=6.0)
            return
        try:
            self._journal.poke(message or None)
        except OSError as exc:
            self.app.notify(f"could not write the poke: {exc}", severity="warning", timeout=6.0)
            return
        self.app.notify("poked", timeout=3.0)

    def _flush_pending(self) -> None:
        """Write the accumulated thinking and answer text as one dim line."""
        text = self._pending.strip()
        self._pending = ""
        if text:
            self.query_one("#mw-log", textual_widgets.RichLog).write(
                rich_text.Text(f"  {text}", style="dim")
            )

    def _poll(self) -> None:
        """Fold the journal and repaint the header, the markers, the log, the prompts."""
        if self._ended:
            return
        try:
            events = self._journal.read()
            ms = fold_machine(self._spec, events)
        except JournalError as exc:
            # Shown and kept polling: an append may heal or end the run.
            self.query_one("#mw-head", textual_widgets.Static).update(
                rich_text.Text(f"journal unreadable: {exc}")
            )
            return

        # One probe per poll feeds the header, the footer and the keys; a park or a worker death
        # flips a verb with no MachineEnd, so the footer follows every verb's edge.
        self._execution_fold.refresh(self._root)
        probes = probe_instance(self._root, ms, execution=self._execution_fold.execution())
        self._set_refusals(probes.refusals(self._root.name, ms))
        # A parked instance reads "waiting", so a paused machine never looks busy.
        if ms.ended is not None:
            status = f"ended: {ms.ended.status} ({ms.ended.reason})"
        else:
            status = f"{probes.status_word(ms)} · {ms.current}"
        # A machine runs unattended against the USD ceiling, so the header carries its spend.
        spend, _in_flight = machine_spend(events, self._root, alive=ipc.worker_is_alive(self._root))
        cost = budget.format_usd(spend.usd, partial=spend.partial)
        self.query_one("#mw-head", textual_widgets.Static).update(
            rich_text.Text(
                f"machine: {ms.machine}   {status}"
                f"   transitions: {len(ms.transitions)}   spend: {cost}"
            )
        )
        table = self.query_one("#mw-states", textual_widgets.DataTable)
        for s in ms.states:
            mark = s.mark
            table.update_cell(s.name, "mark", mark)

        # Ended before the log renders, so the final agent state shows no live thinking line.
        if ms.ended is not None and not self._ended:
            self._ended = True
        live = not self._ended and self._steerable()

        log = self.query_one("#mw-log", textual_widgets.RichLog)
        for t in self._cursor.new_transitions(ms):
            self._flush_pending()
            log.write(rich_text.Text(t.line, style="bold"))

        newest, switched = self._cursor.advance_log(self._root)
        if switched:
            self._flush_pending()
            if newest is not None:
                log.write(
                    rich_text.Text(f"-- agent state: {newest.parent.name} --", style="cyan bold")
                )
        self._render_log_lines(log, live=live)
        self._flush_pending()

        self._dispatch_notifications(ms)
        self._dispatch_prompts(live=live)

    def _dispatch_notifications(self, ms: MachineState) -> None:
        """Notify each new machine.notify in-app and on the desktop, and the end once."""
        for n in self._cursor.new_notifications(ms):
            sev: SeverityLevel = (
                "warning" if n.level == "warn" else "error" if n.level == "error" else "information"
            )
            self.app.notify(n.message, title=f"{ms.machine} · {n.state}", severity=sev, timeout=8.0)
            notify.desktop_notify(f"agent6: {ms.machine}", n.message)
        ended = ms.ended
        if ended is not None and not self._end_notified:
            self._end_notified = True
            self.app.notify(
                ended.reason,
                title=f"{ms.machine} {ended.status}",
                severity="information" if ended.status == "ok" else "error",
                timeout=8.0,
            )
            notify.desktop_notify(f"agent6: {ms.machine} {ended.status}", ended.reason)

    def _dispatch_prompts(self, *, live: bool) -> None:
        """Dock the agent state's open approval and pop a modal per pending question.

        Only while the machine runs: a parked or ended instance's fold still carries an
        unanswered prompt, but nothing would poll the answer.
        """
        state_log = self._execution_fold.log
        if not live or state_log is None:
            self.sync_approval(None)
            return
        self.sync_approval(self.open_approval(self._execution_fold.state))
        prompts = self._prompts
        if prompts is not None:
            prompts.dispatch(state_log.parent, self._execution_fold.state)

    def approval_dir(self) -> pathlib.Path:
        """Return the newest agent state's dir."""
        state_log = self._execution_fold.log
        return state_log.parent if state_log is not None else self._root

    def approval_live(self) -> bool:
        """Return whether an answer reaches the worker, read fresh at the answer."""
        return self._answerable()

    def _render_log_lines(self, log: textual_widgets.RichLog, *, live: bool) -> None:
        """Render the state log's new complete lines: deltas accumulate, discrete events write."""
        batch: list[dict[str, object]] = []
        for raw in self._cursor.read_log_lines():
            try:
                evt = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(evt, dict):
                batch.append(evt)
        # The last unclosed role.call is the only turn that can be in flight.
        open_call = -1
        for i, evt in enumerate(batch):
            if evt.get("type") == "role.call":
                open_call = i
            elif evt.get("type") == "role.result":
                open_call = -1
        for i, evt in enumerate(batch):
            etype = evt.get("type")
            if etype in ("role.thinking_delta", "role.text_delta"):
                self._pending += str(evt.get("text", ""))
                continue
            discrete = _discrete_log_line(evt, in_flight=live and i == open_call)
            if discrete is not None:
                self._flush_pending()
                log.write(discrete)


def machine_detail_text(path: pathlib.Path) -> str:
    """Return a parsed machine as text: name, initial, states, validation and the graph.

    Args:
        path: The machine file.

    Returns:
        The text, or the load error itself so a half-written file never crashes the page.
    """
    try:
        spec = load_machine(path)
    except (MachineError, OSError) as exc:
        return f"failed to load {path.name}:\n\n{exc}"
    lines = [
        f"machine: {spec.machine}",
        f"initial: {spec.initial}",
        "",
        f"states ({len(spec.states)}):",
    ]
    # `state.kind` is the user word the watch screen and the web detail show.
    lines.extend(f"  {name}  ({state.kind})" for name, state in spec.states.items())
    problems = validate_semantics(spec)
    lines.append("")
    if problems:
        lines.append(f"semantics: {len(problems)} problem(s)")
        lines.extend(f"  - {p}" for p in problems)
    else:
        lines.append("semantics: OK  (`agent6 machine check` also lints the bundle's scripts)")
    lines += ["", "graph (mermaid):", render_mermaid(spec)]
    return "\n".join(lines)


class MachineDetailScreen(screen.Screen[None]):
    """The read-only view of one parsed machine."""

    CSS = """
    MachineDetailScreen { background: $surface; }
    #machine-detail-title {
        dock: top; height: 1; padding: 0 1; background: $panel; text-style: bold;
    }
    #machine-detail-body { height: 1fr; padding: 0 1; }
    #machine-detail-body Static { pointer: text; }  /* selectable text: I-beam */
    """

    BINDINGS: ClassVar = [
        textual_binding.Binding("escape", "close", "Back", key_display="Esc/q"),
        textual_binding.Binding("q", "close", "Back", show=False),
    ]

    def __init__(self, path: pathlib.Path) -> None:
        super().__init__()
        self._path = path

    def compose(self) -> textual_app.ComposeResult:
        """Yield the title, the scrollable text and the footer."""
        yield textual_widgets.Static(f"machine · {self._path.name}", id="machine-detail-title")
        with containers.VerticalScroll(id="machine-detail-body"):
            yield textual_widgets.Static(
                rich_text.Text(machine_detail_text(self._path))
            )  # plain Text: no markup parsing
        yield textual_widgets.Footer()

    def on_mount(self) -> None:
        """Focus the text for keyboard scrolling."""
        self.query_one("#machine-detail-body", containers.VerticalScroll).focus()

    def action_close(self) -> None:
        """Close the view."""
        self.dismiss()


class CreateMachineModal(screen.ModalScreen[str]):
    """Ask for the task a machine is authored from; the result is the text, "" on cancel."""

    CSS = (
        forms.FORM_CSS
        + """
    CreateMachineModal { align: center middle; }
    #create-box {
        width: 80%; max-width: 100; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    /* The new-task composer, so a machine is described in the same box a run
       is: multi-line, Enter drafts, Ctrl-J adds a line. */
    #create-input { margin-top: 1; }
    #create-hint { color: $text-muted; padding-top: 1; }
    """
    )

    BINDINGS: ClassVar = [textual_binding.Binding("escape", "cancel", "Cancel", show=False)]

    def compose(self) -> textual_app.ComposeResult:
        """Yield the box: the heading, the composer and the hint."""
        with containers.Container(id="create-box"):
            text = rich_text.Text()
            text.append("Create a machine\n\n", style="bold")
            text.append("Describe the loop, and agent6 drafts a .asm.toml in this repo.")
            yield textual_widgets.Static(text)
            yield composer.SteerInput(id="create-input")
            yield textual_widgets.Static(
                "e.g. nightly: pull, run tests, open an issue on failure", id="create-hint"
            )

    def on_mount(self) -> None:
        """Put the composer in draft mode and focus it."""
        bar = self.query_one("#create-input", composer.SteerInput)
        bar.set_mode(mode="draft")
        bar.focus()

    def on_steer_input_submitted(self, message: composer.SteerInput.Submitted) -> None:
        """Return the task text."""
        self.dismiss(message.text.strip())

    def action_cancel(self) -> None:
        """Return "" for a cancel."""
        self.dismiss("")


class MachinesScreen(screen_chrome.ScreenChrome, screen.Screen[None]):
    """The list of authored machines and their instances, with view, run, watch and create."""

    CSS = (
        theme.PALETTE_CSS
        + """
    MachinesScreen { layers: base dropdown; }
    #machines { height: 1fr; border: round $primary; background: $surface; }
    #machines:focus { border: round $accent; }
    #machines > .datatable--header { background: $panel; color: $foreground; text-style: bold; }
    #machines > .datatable--cursor { background: transparent; color: $foreground; }
    #machines:focus > .datatable--cursor {
        background: $primary 40%; color: $text; text-style: bold;
    }
    """
    )
    MENUS: ClassVar = (
        menubar.Menu(
            "File",
            (
                menubar.MenuItem("Back", "close"),
                menubar.MenuItem("Quit", "quit"),
            ),
        ),
        menubar.Menu(
            "Machines",
            (
                menubar.MenuItem("View", "view"),
                menubar.MenuItem("Run", "run"),
                menubar.MenuItem("Watch", "watch"),
                menubar.MenuItem("Create…", "create"),
                menubar.MenuItem("Refresh", "refresh"),
            ),
        ),
        menubar.Menu(
            "View",
            (
                menubar.MenuItem("Theme…", "choose_theme"),
                menubar.MenuItem("Copy method…", "choose_copy_method"),
            ),
        ),
        menubar.Menu(
            "Help",
            (
                menubar.MenuItem("Keys & actions", "help"),
                menubar.MenuItem("Command palette", "command_palette"),
            ),
        ),
    )
    FOOTER: ClassVar = (
        ("view", "View"),
        ("run", "Run"),
        ("watch", "Watch"),
        ("create", "Create"),
        ("refresh", "Refresh"),
        ("help", "Help"),
        ("close", "Back"),
    )
    BINDINGS: ClassVar = menubar.menu_bindings("machines", MENUS, footer=FOOTER)
    COMMANDS: ClassVar = screen.Screen.COMMANDS | {screen_chrome.MenuCommands}
    HELP_TITLE: ClassVar = "agent6 machines — keys & actions"
    HELP_HINTS: ClassVar = ("Enter opens the selected machine",)

    def __init__(
        self,
        agent6_dir: pathlib.Path,
        repo_cwd: pathlib.Path,
        config_path: pathlib.Path | None = None,
    ) -> None:
        super().__init__()
        self.agent6_dir = agent6_dir  # the per-repo state dir; drafts live under it
        self.repo_cwd = repo_cwd
        self.config_path = config_path
        self._machines: list[listing.MachineRow] = []

    def compose(self) -> textual_app.ComposeResult:
        """Yield the menu bar, the table and the footer."""
        yield menubar.MenuBar(self.MENUS)
        yield textual_widgets.DataTable(id="machines")
        yield textual_widgets.Footer()

    def on_mount(self) -> None:
        """Set the table's columns and load the rows."""
        table = self.query_one("#machines", textual_widgets.DataTable)
        table.cursor_type = "row"
        table.add_columns("machine", "status", "state", "updated", "states", "spec", "file")
        self._reload()

    def _reload(self) -> None:
        """Fill the table with the rows `agent6 machine` lists and count them in the title."""
        table = self.query_one("#machines", textual_widgets.DataTable)
        table.clear()
        self._machines = listing.machine_rows(self.repo_cwd, self.agent6_dir)
        for row in self._machines:
            status = (
                rich_text.Text(
                    format.status_label(row.status, row.reason),
                    style=theme.status_style(row.status),
                )
                if row.status
                else rich_text.Text("-")  # a file no instance ran
            )
            table.add_row(
                rich_text.Text(row.name),
                status,
                row.current or "-",
                format.format_when(row.mtime) if row.mtime else "-",
                row.states,
                row.spec,
                rich_text.Text(row.file.name if row.file is not None else "-"),
            )
        table.show_cursor = table.row_count > 0
        # An empty table alone reads as still loading, so the title says none exist.
        n = len(self._machines)
        tally = (
            "no machines yet (c creates one)" if not n else f"{n} machine{'' if n == 1 else 's'}"
        )
        self.app.sub_title = f"machines · {self.repo_cwd.name} · {tally}"

    def _selected_row(self) -> listing.MachineRow | None:
        """Return the row under the cursor."""
        table = self.query_one("#machines", textual_widgets.DataTable)
        if self._machines and 0 <= table.cursor_row < len(self._machines):
            return self._machines[table.cursor_row]
        return None

    def _selected(self) -> pathlib.Path | None:
        """Return the selected row's authored file; None when the instance's file is gone."""
        row = self._selected_row()
        return row.file if row is not None else None

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Return None to dim a row action with no fitting row; False would hide the key.

        View and Run need an authored file, Watch an instance.
        """
        del parameters
        if action in ("view", "run"):
            return True if self._selected() is not None else None
        if action == "watch":
            row = self._selected_row()
            return True if row is not None and row.status else None
        return True

    def on_data_table_row_highlighted(
        self, _event: textual_widgets.DataTable.RowHighlighted
    ) -> None:
        """Re-ask check_action for the new selection."""
        self.refresh_bindings()

    def on_data_table_row_selected(self, _event: textual_widgets.DataTable.RowSelected) -> None:
        """Open the parsed view on Enter, which the table consumes before any binding."""
        self.action_view()

    def action_view(self) -> None:
        """Open the selected machine's parsed view."""
        path = self._selected()
        if path is not None:
            self.app.push_screen(MachineDetailScreen(path))

    def action_run(self) -> None:
        """Confirm, then run the selected machine detached and watch it."""
        path = self._selected()
        if path is None:
            return
        self.app.push_screen(
            modals.ConfirmModal(
                f"Run machine {path.name}?",
                "Runs `agent6 machine run` (detached) and opens the live watch view: "
                "the state overview, each transition, and the agent state's reasoning. "
                "The run keeps going if you close the view.",
                confirm_label="Run",
            ),
            self._on_run_confirm(path),
        )

    def _on_run_confirm(self, path: pathlib.Path) -> Callable[[bool | None], None]:
        """Return the confirm's callback, which loads the spec and spawns the run."""

        def cb(confirmed: bool | None) -> None:
            if not confirmed:
                return
            try:
                spec = load_machine(path)
            except MachineError as exc:
                self.app.notify(f"cannot load {path.name}: {exc}", severity="error", timeout=8.0)
                return
            self._spawn_machine_run(
                self.app, path, layout.machines_root(self.agent6_dir) / spec.machine
            )

        return cb

    @textual.work(thread=True)
    def _spawn_machine_run(
        self, app: textual_app.App[object], path: pathlib.Path, instance: pathlib.Path
    ) -> None:
        """Spawn `machine run` detached, off the UI thread, and report back.

        Started means the child wrote its pid as the instance's worker.pid, right after
        taking the machine lock; a refusal exits before that and its stderr lands here.

        Args:
            app: The app, bound on the UI thread; a screen dismissed mid-spawn has no parent.
            path: The machine file.
            instance: The instance dir the child claims.
        """
        err = spawn.spawn_and_confirm(
            [*spawn.agent6_argv(self.config_path), "machine", "run", str(path)],
            self.repo_cwd,
            started=lambda pid: ipc.read_worker_pid(instance) == pid,
        )
        app.call_from_thread(self._machine_run_started, path, err)

    def _machine_run_started(self, path: pathlib.Path, err: str) -> None:
        """Open the watch on the started run, or show the refusal."""
        if err:
            self.app.notify(err, severity="error", timeout=8.0)
            return
        self._open_watch(path)

    def action_watch(self) -> None:
        """Open the watch on the selected instance, from the file or the source it recorded."""
        row = self._selected_row()
        if row is None or not row.status:
            return
        if row.file is not None:
            self._open_watch(row.file)
            return
        instance = layout.machines_root(self.agent6_dir) / row.name
        self._open_watch(instance / "machine.asm.toml")

    def _open_watch(self, path: pathlib.Path) -> None:
        """Push the watch screen for the machine at the path."""
        try:
            spec = load_machine(path)
        except MachineError as exc:
            self.app.notify(f"cannot load {path.name}: {exc}", severity="error", timeout=8.0)
            return
        instance = layout.machines_root(self.agent6_dir) / spec.machine
        self.app.push_screen(MachineWatchScreen(instance, spec))

    def action_create(self) -> None:
        """Ask for a task and author a machine from it."""
        self.app.push_screen(CreateMachineModal(), self._on_create)

    def _on_create(self, task: str | None) -> None:
        """Spawn the create for a non-empty task."""
        if task:
            self._spawn_machine_create(self.app, task)

    @textual.work(thread=True)
    def _spawn_machine_create(self, app: textual_app.App[object], task: str) -> None:
        """Spawn `machine create` detached, off the UI thread, and report the draft dir.

        The hub then opens the dashboard on the draft, so the authoring run is
        watchable live; the create keeps running detached.

        Args:
            app: The app, bound on the UI thread; a screen dismissed mid-spawn has no parent.
            task: What the machine should do.
        """
        draft_dir, error = spawn.spawn_and_locate(
            [*spawn.agent6_argv(self.config_path), "machine", "create", "--", task],
            self.repo_cwd,
            before=set(_list_drafts(self.agent6_dir)),
            list_dirs=lambda: _list_drafts(self.agent6_dir),
        )
        app.call_from_thread(self._machine_created, draft_dir, error)

    def _machine_created(self, draft_dir: pathlib.Path | None, error: str) -> None:
        """Hand the draft dir to the hub loop, which opens the dashboard on it."""
        if draft_dir is not None:
            self.app.exit(draft_dir)
        else:
            self.app.notify(
                error or "Could not start machine create.", severity="error", timeout=8.0
            )

    def action_refresh(self) -> None:
        """Reload the rows."""
        self._reload()

    def action_quit(self) -> None:
        """Quit the app."""
        self.app.exit()

    def action_close(self) -> None:
        """Return to the hub."""
        self.dismiss()


class _MachineWatchApp(theme.PlainNotify, theme.MuxPointerShapes, textual_app.App[None]):
    """The one-screen host `agent6 attach <machine> --tui` runs the watch view in."""

    CSS = (
        theme.PALETTE_CSS
        + """
    * { scrollbar-size-vertical: 1; scrollbar-size-horizontal: 1; }  /* match the other apps */
    /* A footer that does not fit clips (textual's default); the 1-row widget has no
       room for the scrollbar the universal rule gives it, which would replace every hint. */
    Footer { scrollbar-size-vertical: 0; scrollbar-size-horizontal: 0; }
    Input, TextArea { pointer: text; }
    """
    )

    def __init__(self, instance_dir: pathlib.Path, spec: MachineSpec) -> None:
        super().__init__()
        self._instance = instance_dir
        self._spec = spec

    def on_mount(self) -> None:
        """Apply the saved theme and push the watch screen."""
        theme.setup_theme(self)
        self.push_screen(MachineWatchScreen(self._instance, self._spec))


def run_machine_watch_tui(instance_dir: pathlib.Path, spec: MachineSpec) -> int:
    """Return the exit code of the watch view run over an instance."""
    return _MachineWatchApp(instance_dir, spec).run() or 0
