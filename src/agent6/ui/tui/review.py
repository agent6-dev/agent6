"""Inspect review findings and record operator actions in the TUI."""

from __future__ import annotations

import asyncio
import pathlib
import uuid
from typing import ClassVar, Literal

import pydantic
from rich import text
from textual import app, containers, screen, widgets

from agent6 import review_records
from agent6.app import review_actions
from agent6.sessions import process
from agent6.ui import keymap, spawn
from agent6.ui.tui import _diff_pane, forms, menubar, modals, screen_chrome
from agent6.viewmodel import review


class DispositionModal(screen.ModalScreen[review_records.FindingAction]):
    """Collect an operator's resolution or reopening reason."""

    DEFAULT_CSS = """
    DispositionModal { align: center middle; }
    #review-form { width: 80%; height: auto; max-height: 90%; padding: 1 2;
        border: round $accent; background: $surface; }
    #review-reason { height: 6; }
    #review-form Horizontal { height: auto; }
    #review-error { height: auto; color: $error; }
    """
    BINDINGS: ClassVar = [
        (key, action, keymap.REVIEW_TEXT["submit" if action == "submit" else "cancel"])
        for action, key in keymap.SCREEN_KEYS["finding disposition"].items()
    ]

    def __init__(self, action: Literal["resolve", "reopen"], version: int) -> None:
        super().__init__()
        self._action: Literal["resolve", "reopen"] = action
        self._version = version

    def compose(self) -> app.ComposeResult:
        """Yield the disposition, reason and submit controls."""
        with containers.VerticalScroll(id="review-form"):
            yield widgets.Static(keymap.REVIEW_TEXT[self._action])
            if self._action == "resolve":
                yield forms.Picker(
                    keymap.REVIEW_DISPOSITIONS,
                    value="fixed",
                    allow_blank=False,
                    id="review-disposition",
                )
                yield widgets.Input(
                    placeholder=keymap.REVIEW_TEXT["duplicate"], id="review-duplicate"
                )
            yield widgets.Label(keymap.REVIEW_TEXT["reason"])
            yield widgets.TextArea(id="review-reason")
            yield widgets.Static("", id="review-error", markup=False)
            with containers.Horizontal():
                yield widgets.Button(keymap.REVIEW_TEXT["submit"], id="review-submit")
                yield widgets.Button(keymap.REVIEW_TEXT["cancel"], id="review-cancel")

    def on_mount(self) -> None:
        """Focus the required reason."""
        self.query_one("#review-reason", widgets.TextArea).focus()

    def action_submit(self) -> None:
        """Validate the form before returning the versioned action."""
        reason = self.query_one("#review-reason", widgets.TextArea).text
        try:
            resolution = None
            if self._action == "resolve":
                resolution = review_records.FindingResolution.model_validate(
                    dict(
                        disposition=self.query_one("#review-disposition", widgets.Select).value,
                        reason=reason,
                        duplicate_of=self.query_one("#review-duplicate", widgets.Input).value
                        or None,
                    )
                )
            action = review_records.FindingAction(
                request_id=uuid.uuid4().hex,
                expected_version=self._version,
                action=self._action,
                resolution=resolution,
                reason=reason if self._action == "reopen" else None,
            )
        except pydantic.ValidationError as exc:
            self.query_one("#review-error", widgets.Static).update(str(exc))
            return
        self.dismiss(action)

    def on_button_pressed(self, event: widgets.Button.Pressed) -> None:
        """Submit or cancel the form."""
        if event.button.id == "review-submit":
            self.action_submit()
        else:
            self.action_close()

    def action_close(self) -> None:
        """Cancel without recording an action."""
        self.dismiss(None)


class FindingsScreen(screen_chrome.ScreenChrome, screen.Screen[None]):
    """Inspect stored findings and send operator actions through the shared service."""

    CSS = """
    #findings { height: 30%; border: round $primary; }
    #review-summary, #review-result { height: auto; }
    #finding-panes { height: 1fr; }
    #finding-context, #review-diff { width: 1fr; border: round $primary; padding: 0 1; }
    #finding-detail { height: auto; }
    """
    MENUS: ClassVar = (
        menubar.Menu(
            keymap.FINDINGS,
            (
                *(
                    menubar.MenuItem(keymap.REVIEW_TEXT[a], a)
                    for a in ("fix", "resolve", "reopen", "refresh")
                ),
                menubar.MenuItem(keymap.REVIEW_TEXT["back"], "close"),
            ),
        ),
        menubar.Menu(
            keymap.REVIEW_TEXT["help"],
            (
                menubar.MenuItem(keymap.REVIEW_TEXT["keys"], "help"),
                menubar.MenuItem(keymap.REVIEW_TEXT["palette"], "command_palette"),
            ),
        ),
    )
    FOOTER: ClassVar = (
        *((a, keymap.REVIEW_TEXT[a]) for a in ("fix", "resolve", "reopen", "refresh")),
        ("close", keymap.REVIEW_TEXT["back"]),
    )
    BINDINGS: ClassVar = menubar.menu_bindings("findings", MENUS, footer=FOOTER)
    COMMANDS: ClassVar = screen.Screen.COMMANDS | {screen_chrome.MenuCommands}
    HELP_HINTS: ClassVar = keymap.REVIEW_HINTS

    def __init__(
        self,
        session_dir: pathlib.Path,
        cwd: pathlib.Path,
        config_paths: tuple[pathlib.Path, ...] = (),
    ) -> None:
        super().__init__()
        self._session_dir = session_dir
        self._cwd = cwd
        self._config_paths = config_paths
        self._view = review.ReviewView()
        self._busy = False

    def compose(self) -> app.ComposeResult:
        """Yield the findings table, selected report and read-only diff."""
        yield menubar.MenuBar(self.MENUS)
        yield widgets.Static("", id="review-summary", markup=False)
        yield widgets.DataTable(id="findings", cursor_type="row")
        with containers.Horizontal(id="finding-panes"):
            with forms.ScrollPane(id="finding-context"):
                yield widgets.Static("", id="finding-detail", markup=False)
            yield _diff_pane.DiffPane(self._session_dir, id="review-diff")
        yield widgets.Static("", id="review-result", markup=False)
        yield widgets.Footer()

    def on_mount(self) -> None:
        """Read the journal and focus the findings table."""
        self.app.sub_title = f"{keymap.FINDINGS} · {self._session_dir.name}"
        self.query_one("#findings", widgets.DataTable).add_columns(*keymap.REVIEW_COLUMNS)
        self.action_refresh()
        self.query_one("#findings", widgets.DataTable).focus()
        self.set_interval(2.0, self._poll)

    def _poll(self) -> None:
        if self.app.screen is self and not self._busy:
            self.action_refresh()

    def _selected(self) -> review.FindingView | None:
        row = self.query_one("#findings", widgets.DataTable).cursor_row
        return self._view.findings[row] if 0 <= row < len(self._view.findings) else None

    def action_refresh(self) -> None:
        """Reload evidence while keeping the selected finding."""
        try:
            view = review_actions.read_review(self._session_dir)
        except (review_actions.ReviewActionError, OSError) as exc:
            self.query_one("#review-result", widgets.Static).update(str(exc))
            return
        selected = self._selected()
        table = self.query_one("#findings", widgets.DataTable)
        if view != self._view:
            self._view = view
            table.clear()
            for f in view.findings:
                report = f.observations[-1].finding
                table.add_row(
                    *(
                        text.Text(s)
                        for s in (
                            f.finding_id,
                            f.status,
                            report.severity,
                            str(report.location),
                            report.title,
                        )
                    )
                )
            if selected is not None:
                row = next(
                    (i for i, f in enumerate(view.findings) if f.finding_id == selected.finding_id),
                    0,
                )
                table.move_cursor(row=row)
        counts = view.open_counts
        self.query_one("#review-summary", widgets.Static).update(
            keymap.REVIEW_TEXT["summary"].format(
                block=counts.block, warn=counts.warn, nit=counts.nit, completion=view.completion
            )
            + (f"\n{view.legacy_notice}" if view.legacy_notice else "")
        )
        self._show_selected()
        self.refresh_bindings()

    def on_data_table_row_highlighted(self, _event: widgets.DataTable.RowHighlighted) -> None:
        """Preview the row selected by the keyboard or pointer."""
        self._show_selected()
        self.refresh_bindings()

    def _show_selected(self) -> None:
        finding = self._selected()
        detail = self.query_one("#finding-detail", widgets.Static)
        if finding is None:
            detail.update(keymap.REVIEW_TEXT["empty"])
            return
        observation = finding.observations[-1]
        report = observation.finding
        body = keymap.REVIEW_TEXT["detail"].format(
            finding_id=finding.finding_id,
            status=finding.status,
            severity=report.severity,
            location=report.location,
            title=report.title,
            detail=report.detail,
            fix_state=finding.fix_state,
        )
        for action in finding.actions:
            reason = (
                action.resolution.reason
                if action.resolution
                else action.reason or action.error or ""
            )
            label = (
                action.resolution.disposition
                if action.resolution
                else action.type.removeprefix("loop.review.")
            )
            body += "\n" + keymap.REVIEW_TEXT["history"].format(action=label, reason=reason)
        panel = next(p for p in self._view.panels if p.panel_id == observation.panel_id)
        if panel.result is not None:
            body += "\n" + keymap.REVIEW_TEXT["panel"].format(
                panel_id=panel.panel_id,
                block_on=panel.result.block_on,
                blocked=panel.result.blocked,
                disarmed=panel.result.disarmed,
            )
        detail.update(body)
        self.query_one(_diff_pane.DiffPane).show_review(self._cwd, panel.target, report.location)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Offer only actions that fit the selected finding's state.

        Args:
            action: The action name.
            parameters: The action arguments, unused.

        Returns:
            True for an available action, otherwise None to grey it out.
        """
        if action not in {"fix", "resolve", "reopen"}:
            return True
        finding = self._selected()
        if finding is None or self._busy:
            return None
        return True if (finding.status == "resolved") == (action == "reopen") else None

    def action_fix(self) -> None:
        """Steer a live worker or confirm a resume to fix the selected finding."""
        finding = self._selected()
        if finding is None or self._busy:
            return
        live = process.worker_is_alive(self._session_dir)
        request = review_records.FindingAction(
            request_id=uuid.uuid4().hex,
            expected_version=self._view.version,
            action="fix",
            resume=not live,
        )
        if live:
            self._submit(finding.finding_id, request)
        else:
            self.app.push_screen(
                modals.ConfirmModal(
                    keymap.REVIEW_TEXT["resume"],
                    keymap.REVIEW_TEXT["confirm"].format(finding_id=finding.finding_id),
                    confirm_label=keymap.REVIEW_TEXT["resume"],
                ),
                lambda confirmed: self._submit(finding.finding_id, request) if confirmed else None,
            )

    def action_resolve(self) -> None:
        """Ask for a disposition and reason."""
        self._disposition("resolve")

    def action_reopen(self) -> None:
        """Ask why the finding should be reopened."""
        self._disposition("reopen")

    def _disposition(self, action: Literal["resolve", "reopen"]) -> None:
        finding = self._selected()
        if finding is not None and not self._busy:
            self.app.push_screen(
                DispositionModal(action, self._view.version),
                lambda request: self._submit(finding.finding_id, request) if request else None,
            )

    def _submit(self, finding_id: str, request: review_records.FindingAction) -> None:
        if not self._busy:
            self._busy = True
            self.run_worker(self._apply(finding_id, request))

    async def _apply(self, finding_id: str, request: review_records.FindingAction) -> None:
        try:
            result = await asyncio.to_thread(
                review_actions.act,
                self._session_dir,
                finding_id,
                request,
                source="tui",
                wait_s=0,
                resume=lambda steer: spawn.spawn_detached_resume(
                    self._cwd,
                    self._session_dir.name,
                    steer=steer,
                    config_paths=self._config_paths,
                ),
            )
            message = keymap.REVIEW_TEXT["result"].format(
                message=result.message, request_id=result.request_id
            )
        except (review_actions.ReviewActionError, OSError) as exc:
            message = str(exc)
        finally:
            self._busy = False
        self.query_one("#review-result", widgets.Static).update(message)
        self.action_refresh()

    def action_close(self) -> None:
        """Return to the dashboard or hub."""
        if not self.close_open_list():
            self.app.pop_screen()
