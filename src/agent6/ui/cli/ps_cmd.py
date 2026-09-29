# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 ps`: live agent6 sessions across every repository on this machine.

`sessions list` is per-repo; this walks the whole state base so a detached run is
findable from anywhere, with the directory to cd to and the id to attach.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from agent6.paths import repo_root_of_id, state_base
from agent6.sessions.ipc import frontend_is_live, read_worker_pid, worker_is_alive
from agent6.sessions.layout import SESSION_BUCKETS
from agent6.ui.cli._common import home_contracted
from agent6.viewmodel.format import lane_count, lane_id_cell, status_label
from agent6.viewmodel.listing import ListingRow, SessionSummary, nested_rows, summarize_session_dir
from agent6.viewmodel.machine_state import summarize_machine_dir


@dataclass(frozen=True, slots=True)
class _Row:
    """One live session or machine instance.

    Attributes:
        directory: The checkout, or None when the id does not lead back to one.
        repo_id: The state dir's name for the repo.
        id: The session id or machine name.
        mode: The session mode, or "machine".
        status: The status word.
        pid: The worker pid, when readable.
        attached: A front end is live on the session.
        coordinator: For a lane, the live coordinator row it nests under.
        lanes: The fan-out's live lanes, nested.
    """

    directory: str | None
    repo_id: str
    id: str
    mode: str
    status: str
    pid: int | None
    attached: bool
    coordinator: str = ""
    lanes: tuple[_Row, ...] = ()

    def cells(self, id_cell: str) -> tuple[str, ...]:
        """Return the table cells, with the id cell as given."""
        where = self.directory if self.directory is not None else f"? ({self.repo_id})"
        return (
            where,
            id_cell,
            self.mode,
            self.status,
            str(self.pid) if self.pid is not None else "?",
            "attached" if self.attached else "",
        )


@dataclass(frozen=True, slots=True)
class _Live:
    """The live sessions (rows and summaries, by real dir) and machine rows.

    Attributes:
        rows: The session rows by their real directory.
        summaries: The session summaries by their real directory.
        machines: The machine rows.
    """

    rows: dict[Path, _Row]
    summaries: dict[Path, SessionSummary]
    machines: list[_Row]

    def nested(self) -> list[_Row]:
        """Return the session rows with a fan-out's live lanes under it, then the machines."""
        summaries_by_repo: dict[str, list[SessionSummary]] = {}
        rows_by_repo: dict[str, dict[str, _Row]] = {}
        for real, own in self.rows.items():
            summaries_by_repo.setdefault(own.repo_id, []).append(self.summaries[real])
            rows_by_repo.setdefault(own.repo_id, {})[own.id] = own

        def tree(row: ListingRow, own_rows: dict[str, _Row]) -> _Row:
            """Return the row with its lanes nested, from the listing fold's tree."""
            own = own_rows[row.summary.session_id]
            return replace(own, lanes=tuple(tree(lane, own_rows) for lane in row.lanes))

        session_rows: list[tuple[float, _Row]] = []
        for repo_id, summaries in summaries_by_repo.items():
            own_rows = rows_by_repo[repo_id]
            session_rows.extend((row.mtime, tree(row, own_rows)) for row in nested_rows(summaries))
        session_rows.sort(key=lambda item: item[0], reverse=True)

        return [*(row for _mtime, row in session_rows), *self.machines]


def _live_rows() -> _Live:
    """Return every live session and machine instance under the state base, one row each."""
    base = state_base()
    # Keyed on the real dir: a lane is linked under its coordinator's repo too, and the link wins.
    rows_by_dir: dict[Path, _Row] = {}
    summaries_by_dir: dict[Path, SessionSummary] = {}
    rows: list[_Row] = []
    if base.is_dir():
        for repo_dir in sorted(base.iterdir()):
            root = repo_root_of_id(repo_dir.name)
            # An elided-hash id is not reversible to a path; the cell says so.
            where = home_contracted(str(root)) if root is not None else None
            for bucket in SESSION_BUCKETS:
                bucket_path = repo_dir / "sessions" / bucket
                if not bucket_path.is_dir():
                    continue
                for sdir in sorted(bucket_path.iterdir()):
                    real = sdir.resolve()
                    if (
                        not sdir.is_dir()
                        or not worker_is_alive(sdir)
                        or (real in rows_by_dir and not sdir.is_symlink())
                    ):
                        continue
                    summary = summarize_session_dir(sdir)
                    summaries_by_dir[real] = summary
                    rows_by_dir[real] = _Row(
                        where,
                        repo_dir.name,
                        sdir.name,
                        summary.mode,
                        status_label(summary.status, summary.reason),
                        read_worker_pid(sdir),
                        frontend_is_live(sdir),
                        coordinator=summary.coordinator,
                    )
            # A machine instance is a live session too: its worker.pid sits at the instance root.
            machines = repo_dir / "machines"
            if machines.is_dir():
                for mdir in sorted(machines.iterdir()):
                    if not mdir.is_dir() or not worker_is_alive(mdir):
                        continue
                    machine = summarize_machine_dir(mdir)
                    rows.append(
                        _Row(
                            where,
                            repo_dir.name,
                            mdir.name,
                            "machine",
                            status_label(machine.status, machine.reason),
                            read_worker_pid(mdir),
                            False,
                        )
                    )
    return _Live(
        rows=rows_by_dir,
        summaries=summaries_by_dir,
        machines=rows,
    )


def cmd_ps(*, as_json: bool = False, lanes: bool = False) -> int:
    """Print one row per live session: directory, id, mode, status, pid, front end.

    Liveness is the worker-pid rule every listing uses (a foreign-owned or reused pid reads
    dead). A fan-out's live lanes nest under its row: folded into a count, or listed
    indented with `lanes`; the JSON row always nests them.

    Args:
        as_json: Print the rows as JSON.
        lanes: List each fan-out's lanes.

    Returns:
        The exit code, 0.
    """
    rows = _live_rows().nested()
    if as_json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
        return 0
    if not rows:
        print("no live agent6 sessions.")
        return 0
    headers = ("directory", "id", "mode", "status", "pid", "front-end")
    cells: list[tuple[str, ...]] = []

    def emit(r: _Row, depth: int) -> None:
        """Append the row's cells, then its lanes' when listing them."""
        if depth:
            cells.append(r.cells(lane_id_cell(r.id, depth)))
        else:
            folded = f" ({lane_count(len(r.lanes))})" if r.lanes and not lanes else ""
            cells.append(r.cells(r.id + folded))
        if lanes:
            for lane in r.lanes:
                emit(lane, depth + 1)

    for r in rows:
        emit(r, 0)
    widths = [max(len(headers[i]), *(len(c[i]) for c in cells)) for i in range(len(headers))]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip())
    for c in cells:
        print("  ".join(v.ljust(widths[i]) for i, v in enumerate(c)).rstrip())
    print(
        "\nattach with: cd <directory> && agent6 attach <id>"
        "  (a machine: agent6 machine status <id>;"
        " ? = directory not recoverable from the id)"
    )
    return 0
