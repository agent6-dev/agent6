# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Export session journals: fold, encode and deliver, in one operator-started process.

`export` reads the repository's journals, or one session's, folds each through
`viewmodel.telemetry`, posts the spans, log records and the metrics snapshot to the OTLP
endpoint and writes the Prometheus textfile, as `[telemetry]` selects. It never reaches into
a run: the journal stays the record, and a destination that fails is reported, not retried
forever. A cursor per endpoint, session and signal keeps a re-run from re-sending delivered
spans and log records; the metrics reading is always the whole cumulative snapshot, dated
from the earliest journal event so a re-run reports the same window.
"""

from __future__ import annotations

import dataclasses
import datetime
import email.utils
import hashlib
import json
import pathlib
import random
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import httpx2

from agent6 import __version__, paths, portable
from agent6.config import TelemetryConfig
from agent6.sessions import layout as sessions_layout
from agent6.sessions import process
from agent6.ui.telemetry import _otlp, _prometheus
from agent6.viewmodel import (
    listing,
    tail,
    telemetry,
    telemetry_machine,
    telemetry_metrics,
    telemetry_records,
)

BATCH = 256  # records per request
FLUSH_S = 5.0  # a follow posts what it holds at least this often while events arrive
TIMEOUT_S = 5.0
ATTEMPTS = 3
RETRY_STATUSES = frozenset({429, 502, 503, 504})
POLL_S = 0.5
CURSORS_DIR = "telemetry"
TEXTFILE_MODE = 0o644  # node_exporter runs as its own user; the metrics carry no content
Post = Callable[[str, dict[str, str], bytes], httpx2.Response]


class ExportError(Exception):
    """The export could not start: no such session, or `--follow` without one."""


@dataclasses.dataclass(frozen=True, slots=True)
class ExportReport:
    """What one export delivered.

    Attributes:
        sessions: Journals read.
        spans: Spans posted.
        logs: Log records posted.
        metric_posts: Metrics requests posted.
        prometheus_file: The textfile written, "" when none was selected.
        notes: What went wrong: a destination refusing, a partial success, a retry budget spent.
    """

    sessions: int
    spans: int
    logs: int
    metric_posts: int
    prometheus_file: str
    notes: tuple[str, ...]


def default_post(url: str, headers: dict[str, str], content: bytes) -> httpx2.Response:
    """POST one body: TLS verified, no proxy from the environment, no redirect, a bounded timeout.

    Args:
        url: The signal's URL.
        headers: The request headers.
        content: The JSON body.

    Returns:
        The response.
    """
    with httpx2.Client(
        trust_env=False, timeout=httpx2.Timeout(TIMEOUT_S), follow_redirects=False
    ) as client:
        return client.post(url, headers=headers, content=content)


def retry_after_s(value: str | None) -> float | None:
    """Return a `Retry-After` header's delay in seconds, from either of its forms.

    Args:
        value: The header value: seconds, or an HTTP date.

    Returns:
        The non-negative delay, or None when the header is absent or unreadable.
    """
    if value is None:
        return None
    text = value.strip()
    if text.isdigit():
        return float(text)
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.UTC)
    return max(0.0, (when - datetime.datetime.now(datetime.UTC)).total_seconds())


def _backoff(attempt: int, retry_after: str | None) -> float:
    waited = retry_after_s(retry_after)
    if waited is not None:
        return min(30.0, waited)
    return min(30.0, 2.0**attempt + random.uniform(0.0, 1.0))  # noqa: S311  # jitter, not a secret


class _Destination:
    """One OTLP endpoint: bounded retries, and quiet after a terminal refusal."""

    def __init__(
        self,
        endpoint: str,
        headers: Mapping[str, str],
        post: Post,
        sleep: Callable[[float], None],
    ) -> None:
        self._base = endpoint.rstrip("/")
        self._headers = {"Content-Type": "application/json", **headers}
        self._post = post
        self._sleep = sleep
        self.disabled = ""
        self.notes: list[str] = []

    def send(self, path: str, body: dict[str, Any]) -> bool:
        """Post one body, retrying what the collector may take later.

        Args:
            path: The signal path under the endpoint.
            body: The request body.

        Returns:
            Whether the collector accepted the request.
        """
        if self.disabled:
            return False
        content = json.dumps(body, separators=(",", ":")).encode("utf-8")
        failure = ""
        for attempt in range(ATTEMPTS):
            retry_after: str | None = None
            try:
                response = self._post(self._base + path, self._headers, content)
            except httpx2.HTTPError as exc:
                failure = type(exc).__name__
            else:
                if response.status_code == 200:
                    self._partial_success(path, response)
                    return True
                failure = f"HTTP {response.status_code}"
                if response.status_code not in RETRY_STATUSES:
                    self.disabled = f"{path}: {failure}; the endpoint is off for this export"
                    self.notes.append(self.disabled)
                    return False
                retry_after = response.headers.get("Retry-After")
            if attempt + 1 < ATTEMPTS:
                self._sleep(_backoff(attempt, retry_after))
        self.disabled = f"{path}: {failure} after {ATTEMPTS} attempts; the endpoint is off"
        self.notes.append(self.disabled)
        return False

    def _partial_success(self, path: str, response: httpx2.Response) -> None:
        try:
            reply = response.json()
        except ValueError:
            return
        partial = reply.get("partialSuccess") if isinstance(reply, dict) else None
        if not isinstance(partial, dict):
            return
        rejected = sum(
            int(value)
            for key, value in partial.items()
            if key.startswith("rejected") and isinstance(value, (int, str)) and str(value).isdigit()
        )
        if rejected:
            self.notes.append(f"{path}: the collector rejected {rejected} records")


@dataclasses.dataclass(slots=True)
class _Cursor:
    """The journal offset one signal was delivered through, per endpoint and session.

    `closed` records that the close's spans, the work a dead worker never ended, went out at
    that offset too, so a re-run of an unchanged journal sends nothing. `interrupted` keeps
    those span ids from being delivered again when a resume ends them in the journal.
    """

    path: pathlib.Path
    delivered: int
    closed: bool
    interrupted: set[str] = dataclasses.field(default_factory=set)

    @classmethod
    def read(cls, state_dir: pathlib.Path, endpoint: str, session_id: str, signal: str) -> _Cursor:
        """Return the cursor for the triple, at 0 when none was written."""
        key = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16]
        path = state_dir / CURSORS_DIR / key / f"{session_id}.{signal}.offset"
        interrupted: set[str] = set()
        try:
            words = path.read_text(encoding="utf-8").split()
            delivered = int(words[0]) if words else 0
            closed = len(words) > 1 and words[1] == "closed"
            for word in words[1:]:
                if word.startswith("interrupted="):
                    interrupted.update(word.removeprefix("interrupted=").split(","))
        except (OSError, ValueError):
            delivered, closed = 0, False
        return cls(path, delivered, closed, interrupted)

    def pending(self, position: int, *, closing: bool) -> bool:
        """Return whether records completed at `position` are still to deliver."""
        if position > self.delivered:
            return True
        return position == self.delivered and closing and not self.closed

    def advance(self, offset: int, *, closed: bool, interrupted: Iterable[str] = ()) -> None:
        """Record that everything through `offset` reached the destination."""
        if offset < self.delivered or (offset == self.delivered and self.closed):
            return
        self.delivered, self.closed = offset, closed
        self.interrupted.update(interrupted)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        suffix = f" interrupted={','.join(sorted(self.interrupted))}" if self.interrupted else ""
        portable.atomic_write(self.path, f"{offset}{' closed' if closed else ''}{suffix}\n")


class _Session:
    """One journal's fold, cursors and delivery."""

    def __init__(
        self,
        session_dir: pathlib.Path,
        config: TelemetryConfig,
        destination: _Destination | None,
        state_dir: pathlib.Path,
        resource: telemetry_records.Attributes,
        *,
        machine: _Session | None = None,
        seq: int = 0,
    ) -> None:
        self.session_id = (
            session_dir.name
            if machine is None
            else f"machine:{machine.session_id}:{session_dir.name}"
        )
        self.machine = machine
        self.state_dir = state_dir
        self.workers: dict[int, _Session] = {}
        self.is_machine = session_dir.parent == sessions_layout.machines_root(state_dir)
        self.session_dir = session_dir
        self.journal = session_dir / (
            "journal.jsonl" if self.is_machine else sessions_layout.LOGS_NAME
        )
        self.config = config
        self.destination = destination
        self.resource = resource
        parent = (
            telemetry_machine.state_parent(machine.session_id, seq) if machine is not None else None
        )
        self.fold = telemetry.TelemetryFold(self.session_id, capture=config.capture, parent=parent)
        cursor_id = f"machine:{self.session_id}" if self.is_machine else self.session_id
        self.cursors: dict[str, _Cursor] = {}
        if destination is not None:
            for signal, selected in (("traces", config.traces), ("logs", config.logs)):
                if selected == "otlp":
                    self.cursors[signal] = _Cursor.read(
                        state_dir, config.endpoint, cursor_id, signal
                    )
        self.spans: list[telemetry_records.Span] = []
        self.interrupted: set[str] = set()
        self.logs: list[telemetry_records.LogRecord] = []
        self.position = 0
        self.closing = False
        self.sent_spans = 0
        self.sent_logs = 0
        self.last_flush = time.monotonic()

    def journals(self) -> list[_Session]:
        """Return this journal and the worker journals retained under a machine instance."""
        if self.is_machine:
            for log in sorted((self.session_dir / "states").glob(f"*/{sessions_layout.LOGS_NAME}")):
                head, separator, _state = log.parent.name.partition("-")
                if not separator or not head.isdigit():
                    continue
                seq = int(head)
                if seq not in self.workers:
                    self.workers[seq] = _Session(
                        log.parent,
                        self.config,
                        self.destination,
                        self.state_dir,
                        self.resource,
                        machine=self,
                        seq=seq,
                    )
        return [self, *self.workers.values()]

    def metrics(self) -> telemetry_metrics.MetricsSnapshot:
        """Return this session's metrics, with its machine workers counted once."""
        return self.fold.metrics(
            workers={seq: worker.fold.metrics() for seq, worker in self.workers.items()}
        )

    def read(self, *, closing: bool) -> Iterable[telemetry_records.Emitted]:
        """Fold unread journal events and yield what each event completed.

        The close's spans, which end work the journal never ended, come only when no worker
        holds the session: a live run finishes them itself, and sending them now would send
        the same ids twice. The cursor remembers spans delivered at a close, so a later
        resume can end them in the fold without delivering them again.

        Args:
            closing: This is the export's final drain, not a follow's poll.

        Yields:
            What each event completed, then what the close ended.
        """
        self.closing = False
        events = tail.tail_events(
            self.journal,
            follow=False,
            start_at=self.position,
            on_position=self._heard,
        )
        for event in events:
            yield self.fold.feed(event)
        owner = self.machine.session_dir if self.machine is not None else self.session_dir
        if closing and not process.worker_is_alive(owner):
            self.closing = True
            yield self.fold.close()

    def _heard(self, offset: int) -> None:
        self.position = offset

    def collect(self, emitted: telemetry_records.Emitted) -> None:
        """Queue what an event completed, for the selected signals and past their cursors."""
        traces = self.cursors.get("traces")
        if traces is not None and traces.pending(self.position, closing=self.closing):
            spans = [s for s in emitted.spans if s.span_id not in traces.interrupted]
            self.spans.extend(spans)
            if self.closing:
                self.interrupted.update(s.span_id for s in spans)
        logs = self.cursors.get("logs")
        if logs is not None and logs.pending(self.position, closing=self.closing):
            self.logs.extend(emitted.logs)

    def due(self, *, follow: bool) -> bool:
        """Return whether a request is due: a full batch, or a follow's flush interval."""
        if len(self.spans) >= BATCH or len(self.logs) >= BATCH:
            return True
        return follow and time.monotonic() - self.last_flush >= FLUSH_S

    def flush(self) -> None:
        """Post the queued records; a signal's cursor moves only past what its endpoint took."""
        self.last_flush = time.monotonic()
        if self.destination is None:
            return
        for signal, records in (("traces", self.spans), ("logs", self.logs)):
            cursor = self.cursors.get(signal)
            if cursor is None:
                continue
            if records:
                body = (
                    _otlp.traces_body(self.spans, self.resource)
                    if signal == "traces"
                    else _otlp.logs_body(self.logs, self.resource)
                )
                if not self.destination.send(_otlp.path_for(signal), body):
                    continue  # the destination is off now; its cursor stays where it was
                if signal == "traces":
                    self.sent_spans += len(records)
                else:
                    self.sent_logs += len(records)
            if not self.destination.disabled:
                cursor.advance(
                    self.position,
                    closed=self.closing,
                    interrupted=self.interrupted if signal == "traces" else (),
                )
        self.spans = []
        self.interrupted.clear()
        self.logs = []


def _session_dirs(cwd: pathlib.Path, session_id: str) -> list[pathlib.Path]:
    state_dir = paths.state_dir(cwd)
    machines = [
        p.parent
        for p in sessions_layout.machines_root(state_dir).glob("*/journal.jsonl")
        if p.is_file()
    ]
    if session_id:
        matches = [s.session_dir for s in sessions_layout.session_matches(state_dir, session_id)]
        matches.extend(d for d in machines if d.name.startswith(session_id))
        exact = [d for d in matches if d.name == session_id]
        matches = exact or matches
        if len(matches) != 1:
            raise ExportError(
                f"no session {session_id!r} in this repository, or the id is ambiguous"
            )
        return matches
    dirs = listing.session_dirs(state_dir, sessions_layout.SESSION_BUCKETS)
    dirs = [d for d in dirs if (d / sessions_layout.LOGS_NAME).is_file()]
    return sorted(
        [*dirs, *machines],
        key=lambda d: (
            (d / ("journal.jsonl" if d in machines else sessions_layout.LOGS_NAME)).stat().st_mtime
        ),
    )


def export(
    cwd: pathlib.Path,
    config: TelemetryConfig,
    *,
    session_id: str = "",
    follow: bool = False,
    headers: Mapping[str, str] | None = None,
    post: Post = default_post,
    sleep: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] | None = None,
) -> ExportReport:
    """Export the repository's journals, or one session's, as `[telemetry]` selects.

    Args:
        cwd: The repository.
        config: The `[telemetry]` table.
        session_id: One session to export; "" for every session with a journal.
        follow: Keep reading the session's journal as it grows; needs `session_id`.
        headers: The resolved header values for the endpoint.
        post: The HTTP POST; the tests inject one.
        sleep: The retry pause; the tests inject one.
        should_stop: Polled while following; once true, the export drains and returns.

    Returns:
        The report.

    Raises:
        ExportError: The session does not exist, or `follow` came without a session.
    """
    if follow and not session_id:
        raise ExportError("--follow needs a session id; the one-shot export reads every session")
    state_dir = paths.state_dir(cwd)
    repo = hashlib.sha256(paths.repo_id(paths.project_root(cwd)).encode("utf-8")).hexdigest()
    resource: telemetry_records.Attributes = (
        ("service.name", "agent6"),
        ("service.version", __version__),
        ("agent6.repo", repo[:16]),
        # A single session and the whole repository are different cumulative series.
        ("agent6.export.scope", session_id or "repository"),
    )
    destination = _Destination(config.endpoint, headers or {}, post, sleep) if config.otlp else None
    sink = _MetricsSink(config, destination, resource)
    sessions = [
        _Session(d, config, destination, state_dir, resource)
        for d in _session_dirs(cwd, session_id)
    ]
    while True:
        stopping = not follow or (should_stop is not None and should_stop())
        for root in sessions:
            for session in root.journals():
                for emitted in session.read(closing=stopping):
                    session.collect(emitted)
                    if session.due(follow=follow):
                        session.flush()
                        if follow:
                            sink.publish(sessions)
                if stopping or session.due(follow=follow):
                    session.flush()
                    if follow and not stopping:
                        sink.publish(sessions)
        if stopping:
            break
        time.sleep(POLL_S)
    sink.publish(sessions)
    return ExportReport(
        sessions=len(sessions),
        spans=sum(s.sent_spans for root in sessions for s in root.journals()),
        logs=sum(s.sent_logs for root in sessions for s in root.journals()),
        metric_posts=sink.posts,
        prometheus_file=sink.written,
        notes=tuple(destination.notes) if destination is not None else (),
    )


class _MetricsSink:
    """Where the merged metrics reading goes: the endpoint, the textfile, or nowhere."""

    def __init__(
        self,
        config: TelemetryConfig,
        destination: _Destination | None,
        resource: telemetry_records.Attributes,
    ) -> None:
        self._config = config
        self._destination = destination
        self._resource = resource
        self.posts = 0
        self.written = ""

    def publish(self, sessions: list[_Session]) -> None:
        """Post or write the sessions' merged reading, dated from their earliest event."""
        if self._config.metrics == "none":
            return
        snapshot = telemetry_metrics.merge_snapshots(s.metrics() for s in sessions)
        if self._config.metrics == "otlp" and self._destination is not None:
            firsts = [s.fold.first_ns for s in sessions if s.fold.first_ns is not None]
            now_ns = time.time_ns()
            start_ns = min(firsts) if firsts else now_ns
            body = _otlp.metrics_body(snapshot, self._resource, start_ns=start_ns, now_ns=now_ns)
            if self._destination.send(_otlp.METRICS_PATH, body):
                self.posts += 1
        if self._config.metrics == "prometheus":
            target = pathlib.Path(self._config.prometheus_file)
            # The mode rides the temp file into the rename; nothing touches the destination path.
            portable.atomic_write(target, _prometheus.exposition(snapshot), mode=TEXTFILE_MODE)
            self.written = self._config.prometheus_file
