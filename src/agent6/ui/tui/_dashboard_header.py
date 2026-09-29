# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The dashboard's header: the status, the task count, the cost, the budget
and the context on line one; the role and the task on line two; then the
manifest's lines as they appear (lineage, branch, pins, serving, a lane's
compare), each read once and cached as long as it can hold. Every line ends
in an ellipsis rather than wraps, so a long model id never pushes the status
onto a line of its own."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

from rich.markup import escape
from rich.text import Text
from textual.widgets import Static

from agent6.graph.order import DONE_STATUSES
from agent6.sessions.ipc import listening_ports
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.types import SESSION_KINDS
from agent6.ui.tui.theme import status_style
from agent6.viewmodel import manifest_branches, manifest_header, session_compare
from agent6.viewmodel.format import format_compare, format_usd, spinner_frame, status_label
from agent6.viewmodel.listing import task_snippet
from agent6.viewmodel.state import SessionState

if TYPE_CHECKING:
    from agent6.ui.tui.app import Agent6TUI


class RunHeader(Static):
    """The `#top` lines; `refresh_lines` repaints them from the app's fold
    (`s`) and the state the details are as of (`ds`, `as_of`: the selected
    step's, or the live one)."""

    def __init__(self) -> None:
        super().__init__("", id="top")
        self._compare_line: str | None = None  # cached fan-out compare header (terminal state)
        self._branch_line: str | None = None  # cached branch header
        self._branch_finished = False  # the run state the cached line was read under
        self._branch_recheck_at = 0.0  # a finished run re-reads every few seconds
        self._lineage_line: str | None = None  # cached fork lineage (never changes)
        self._start_role_line: str | None = None  # the manifest's driver, before any call

    @property
    def _tui(self) -> Agent6TUI:
        return cast("Agent6TUI", self.app)  # only the dashboard, on Agent6TUI, composes this

    def refresh_lines(self, s: SessionState, ds: SessionState, as_of: str, *, active: bool) -> None:
        """*active*: a model call is in flight, which is what earns the
        spinner and the seconds since the last event."""
        tui = self._tui
        role = s.last_role
        # A spinner + seconds since the last event belongs only to a model call
        # awaiting its result. A live worker before or between calls is not
        # evidence that a model is working.
        beat = ""
        if active and role is not None:
            spinner = spinner_frame(tui.spin)
            beat = f" {spinner} {tui.seconds_since_event()}s"
        role_line = f"{role.role} / {role.model}{beat}" if role else self._start_role()
        finished = self._end_label()
        # tasks and cost are both as-of the selected step; ctx is live.
        done_n = sum(1 for t in ds.tasks if t.status in DONE_STATUSES)
        step = f"tasks: {done_n}/{len(ds.tasks)}" if ds.tasks else "tasks: —"
        cost = f"[b]{format_usd(ds.budget.usd_total, partial=ds.budget.usd_partial)}[/]"
        # Consumption of the binding ledger: this execution's metered spend vs its
        # usd_cap (resume re-arms the cap while usd_total stays cumulative),
        # plus the unmetered-token fraction when that ledger has traffic.
        budget = ""
        if ds.budget.usd_cap > 0:
            execution_usd = ds.budget.usd_total - ds.budget.usd_prior_executions
            budget = f"   budget: {min(execution_usd / ds.budget.usd_cap, 1.0):.0%}"
        if ds.budget.tokens_unmetered and ds.budget.tokens_fallback_cap > 0:
            unmet = min(ds.budget.tokens_unmetered / ds.budget.tokens_fallback_cap, 1.0)
            budget += f"   unmetered: {unmet:.0%}"
        if ds.budget.plan_used_percent > 0:
            budget += f"   plan: {ds.budget.plan_used_percent:g}%"
            if ds.budget.plan_cap > 0:
                budget += f" (run {ds.budget.plan_consumed:g}/{ds.budget.plan_cap:g}pt)"
        pct = tui.context_pct()
        ctx = f"   ctx: {pct}%" if pct is not None else ""
        # The status leads line 1, where the eye lands; the role, model and task
        # share line 2. Every line ends in an ellipsis rather than wrap, so a long
        # model id never pushes the status onto a line of its own.
        status = f"{finished}   " if finished else ""
        task = escape(task_snippet(s.user_task or tui.fallback_task, max_chars=120))
        header = Text.from_markup(
            f"[b]agent6[/]  {status}{step}   cost: {cost}{budget}{as_of}{ctx}\n"
            f"role: {escape(role_line)} · task: {task}"
            f"{escape(self._lineage_top())}{escape(self._branch_top())}"
            f"{escape(self._pins_top(s))}{escape(self._serving_top())}"
            f"{escape(self._compare_top())}"
        )
        lines = header.split("\n")
        if width := self.content_size.width:
            for line in lines:
                line.truncate(width, overflow="ellipsis")
        self.update(Text("\n").join(lines))

    def _end_label(self) -> str:
        """The top-line status label, from the dir decision (status_for_session_dir,
        the same word the hub row shows), in the word's shared colour; empty
        while running (the heartbeat line carries live activity)."""
        word, reason = self._tui.dir_status
        if word == "running":
            return ""
        return f"[b {status_style(word)}]{escape(status_label(word, reason))}[/]"

    def _compare_top(self) -> str:
        """The fan-out compare outcome for the header's task line (empty for a
        non-lane run). Read from the manifest once it appears (a lane is stamped
        post-import, by which point it is finished) and cached: it never changes."""
        if self._compare_line is not None:
            return self._compare_line
        formatted = format_compare(session_compare(self._tui.session_dir))
        if formatted is None:
            return ""  # not stamped (yet); don't cache: a live lane may get stamped later
        headline, rationale = formatted
        rat = f" — {rationale[:100]}" if rationale else ""
        self._compare_line = f"\ncompare: {headline}{rat}"
        return self._compare_line

    def _start_role(self) -> str:
        """The role line before the first model call: the role and model the
        manifest says drives the run, read once the manifest exists (a
        launching run has none for a moment). A manifest naming no driver
        reads "(unknown)", once."""
        if self._start_role_line is None:
            try:
                m = read_manifest(self._tui.session_dir)
            except ManifestError:
                return "(unknown)"
            driver = m.models.driver
            if driver is None or m.mode not in SESSION_KINDS:
                self._start_role_line = "(unknown)"
            else:
                self._start_role_line = f"{SESSION_KINDS[m.mode].role} / {driver.model}"
        return self._start_role_line

    def _lineage_top(self) -> str:
        """Where a forked run came from, for the header (the web header's and
        `sessions show`'s line); read once, it never changes."""
        if self._lineage_line is None:
            lineage = manifest_header(self._tui.session_dir).get("forked_from", "")
            self._lineage_line = f"\nforked from: {lineage}" if lineage else ""
        return self._lineage_line

    def _branch_top(self) -> str:
        """Where the run's work lives, for the header: the run branch and the
        base a merge lands on, or the branch merged (the web header's line and
        `sessions show`'s `changes:`). Read from the manifest once it names a
        branch and cached while the run lives; a finished run re-reads it every
        few seconds, since the auto-merge lands after session.end while a held
        screen keeps repainting, and a resume in place (finished again False)
        drops the cache, since an execution committing past the stamp unmakes the
        merge."""
        finished = self._tui.state.finished
        if finished != self._branch_finished:
            self._branch_finished = finished
            self._branch_line = None
        now = time.monotonic()
        if self._branch_line is not None and (not finished or now < self._branch_recheck_at):
            return self._branch_line
        line = manifest_branches(self._tui.session_dir, repo=Path.cwd()).get("branch_line", "")
        if not line:
            return ""  # no manifest yet (a launching run); don't cache
        self._branch_line = f"\nbranch: {line}"
        self._branch_recheck_at = now + 5.0
        return self._branch_line

    @staticmethod
    def _pins_top(s: SessionState) -> str:
        """The operator's pinned instructions in force, for the header (the web
        header's and `sessions show`'s line)."""
        return f"\npins: {' | '.join(s.pins)}" if s.pins else ""

    def _serving_top(self) -> str:
        """What the run is serving, for the header: the ports its network
        listens on and the `agent6 forward` line that reaches one (the web
        header's and `sessions show`'s line). A live probe: "" once the
        network is gone."""
        ports = listening_ports(self._tui.session_dir)
        if not ports:
            return ""
        listed = ", ".join(str(p) for p in ports)
        return f"\nserving: {listed} · agent6 forward {self._tui.session_dir.name} {ports[0]}"
