# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The dashboard's header.

The status, the task count, the cost, the budget and the context on line one;
the role and the task on line two; then the manifest's lines as they appear
(lineage, branch, pins, serving, a lane's compare), each cached as long as it
can hold. Every line ends in an ellipsis rather than wraps, so a long model id
never pushes the status onto a line of its own.
"""

from __future__ import annotations

import pathlib
import time
from typing import TYPE_CHECKING, cast

from rich import markup, text
from textual import widgets

from agent6 import budget as agent6_budget
from agent6 import kinds
from agent6.graph import order
from agent6.sessions import manifest, process
from agent6.ui.tui import theme
from agent6.viewmodel import (
    SessionState,
    comparison,
    format,
    manifest_branches,
    manifest_header,
    policy,
)
from agent6.viewmodel.session import status as session_status

if TYPE_CHECKING:
    from agent6.ui.tui import app


class RunHeader(widgets.Static):
    """The `#top` lines, repainted by `refresh_lines`."""

    def __init__(self) -> None:
        """Create the header with every cached line empty."""
        super().__init__("", id="top")
        self._compare_line: str | None = None  # a lane's compare header, once stamped
        self._branch_line: str | None = None
        self._branch_finished = False  # the run state the branch line was read under
        self._branch_recheck_at = 0.0  # a finished run re-reads every few seconds
        self._lineage_line: str | None = None  # the fork lineage, which never changes
        self._start_role_line: str | None = None  # the manifest's driver, before any call

    @property
    def _tui(self) -> app.Agent6TUI:
        """The host app; only the dashboard, on `Agent6TUI`, composes this."""
        return cast("app.Agent6TUI", self.app)

    def refresh_lines(
        self,
        s: SessionState,
        ds: SessionState,
        as_of: str,
        *,
        active: bool,
    ) -> None:
        """Repaint the header.

        Args:
            s: The app's fold.
            ds: The state the details are as of: the selected commit's, or the live one.
            as_of: The suffix naming the selected commit, or "".
            active: A model call is in flight, which earns the spinner and the seconds
                since the last event; a live worker between calls is not a working model.
        """
        tui = self._tui
        call = s.last_model_call
        beat = ""
        if active and call is not None:
            spinner = format.spinner_frame(tui.spin)
            beat = f" {spinner} {tui.seconds_since_event()}s"
        role_line = f"{call.role} / {call.model}{beat}" if call else self._start_role()
        finished = self._end_label()
        # Tasks and cost are as of the selected commit; ctx is live.
        done_n = sum(1 for t in ds.tasks if t.status in order.DONE_STATUSES)
        tasks = f"tasks: {done_n}/{len(ds.tasks)}" if ds.tasks else "tasks: —"
        b = ds.budget
        cell = session_status.usage_cell(
            b.usd_so_far,
            partial=b.session.usd_partial,
            plan_observed_points=b.session.plan_observed_points,
            plan_used_percent=b.account.used_percent if b.account else 0.0,
        ) or agent6_budget.format_usd(0.0)
        segments = ["[b]agent6[/]"]
        if finished:
            segments.append(finished)
        segments.extend((tasks, f"usage: [b]{cell}[/]"))
        # This execution's metered spend against its cap; the session figure stays cumulative.
        run = b.execution
        if run.usd_cap is not None and run.usd_cap > 0:
            segments.append(f"budget: {min(run.usd / run.usd_cap, 1.0):.0%}")
        if (
            run.unmetered_tokens
            and run.tokens_fallback_cap is not None
            and run.tokens_fallback_cap > 0
        ):
            unmet = min(run.unmetered_tokens / run.tokens_fallback_cap, 1.0)
            segments.append(f"unmetered: {unmet:.0%}")
        if b.account is not None and b.account.used_percent > 0:
            plan = "plan: " + format.plan_readout(
                b.account.used_percent,
                b.account.resets_at,
                now=time.time(),
                observed_at=b.account.observed_at,
            )
            if run.plan_cap is not None and run.plan_cap > 0:
                plan += f" (moved {run.plan_observed_points:g}/{run.plan_cap:g}pt this run)"
            segments.append(plan)
        if as_of:
            segments.append(as_of.strip())
        pct = tui.context_used_percent()
        if pct is not None:
            segments.append(f"ctx: {pct}%")
        lead = f"role: {role_line} · task: "
        task = markup.escape(
            tui.task_label().line(max_chars=120).replace("\n", "\n" + " " * len(lead))
        )
        launch = policy.session_policy(
            tui.session_dir, targets=ds.targets, environment=ds.environment
        )
        header = text.Text.from_markup(
            "  ".join(segments) + "\n"
            f"role: {markup.escape(role_line)} · task: {task}"
            + "".join(f"\n{markup.escape(line)}" for line in (*launch.models, *launch.environment))
            + f"{markup.escape(self._lineage_top())}{markup.escape(self._branch_top())}"
            f"{markup.escape(self._pins_top(s))}{markup.escape(self._serving_top())}"
            f"{markup.escape(self._compare_top())}"
        )
        lines = header.split("\n")
        if width := self.content_size.width:
            for line in lines:
                line.truncate(width, overflow="ellipsis")
        self.update(text.Text("\n").join(lines))

    def _end_label(self) -> str:
        """Return the status label in its colour: the hub row's word; "" while running."""
        word, _reason = self._tui.dir_status
        if word == "running":
            return ""
        return f"[b {theme.status_style(word)}]{markup.escape(self._tui.status_label)}[/]"

    def _compare_top(self) -> str:
        """Return a lane's compare line, cached once stamped; "" for a non-lane run."""
        if self._compare_line is not None:
            return self._compare_line
        formatted = format.format_compare(
            comparison.session_placement(self._tui.session_dir),
            comparison.session_comparison(self._tui.session_dir),
        )
        if formatted is None:
            return ""  # not cached: a live lane is stamped after its import
        headline, rationale = formatted
        rat = f": {rationale[:100]}" if rationale else ""
        self._compare_line = f"\ncompare: {headline}{rat}"
        return self._compare_line

    def _start_role(self) -> str:
        """Return the role line before the first model call, from the manifest's driver.

        A launching run has no manifest for a moment, so nothing is cached until it exists.

        Returns:
            The line as `role / model`, or "(unknown)" without a manifest or a driver.
        """
        if self._start_role_line is None:
            try:
                m = manifest.read_manifest(self._tui.session_dir)
            except manifest.ManifestError:
                return "(unknown)"
            driver = m.models.driver
            if driver is None or m.mode not in kinds.SESSION_KINDS:
                self._start_role_line = "(unknown)"
            else:
                self._start_role_line = f"{kinds.SESSION_KINDS[m.mode].role} / {driver.model}"
        return self._start_role_line

    def _lineage_top(self) -> str:
        """Return the fork lineage line, read once; the web header and `sessions show` share it."""
        if self._lineage_line is None:
            lineage = manifest_header(self._tui.session_dir).get("forked_from", "")
            self._lineage_line = f"\nforked from: {lineage}" if lineage else ""
        return self._lineage_line

    def _branch_top(self) -> str:
        """Return the branch line: the run branch and its base, or the branch merged.

        Cached while the run lives. A finished run re-reads it every few seconds,
        since the auto-merge lands after the end while a held screen keeps
        repainting; a resume in place drops the cache, since a commit past the stamp
        unmakes the merge.

        Returns:
            The line, starting with a newline; "" while the run has no manifest or no branch.
        """
        finished = self._tui.state.finished
        if finished != self._branch_finished:
            self._branch_finished = finished
            self._branch_line = None
        now = time.monotonic()
        if self._branch_line is not None and (not finished or now < self._branch_recheck_at):
            return self._branch_line
        line = manifest_branches(self._tui.session_dir, repo=pathlib.Path.cwd()).get(
            "branch_line", ""
        )
        if not line:
            return ""  # not cached: a launching run has no manifest yet
        self._branch_line = f"\nbranch: {line}"
        self._branch_recheck_at = now + 5.0
        return self._branch_line

    @staticmethod
    def _pins_top(s: SessionState) -> str:
        """Return the pinned instructions line."""
        return f"\npins: {' | '.join(s.pins)}" if s.pins else ""

    def _serving_top(self) -> str:
        """Return the serving line: the listening ports and the `agent6 forward` line.

        Returns:
            The line, starting with a newline; "" while nothing listens.
        """
        ports = process.listening_ports(self._tui.session_dir)
        if not ports:
            return ""
        listed = ", ".join(str(p) for p in ports)
        return f"\nserving: {listed} · agent6 forward {self._tui.session_dir.name} {ports[0]}"
