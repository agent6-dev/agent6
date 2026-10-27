# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 telemetry export`: journals in, OTLP requests and a Prometheus textfile out."""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import stat
import threading
from typing import Any

import httpx2
import pytest

from agent6 import paths
from agent6.config import TelemetryConfig
from agent6.sessions import process
from agent6.ui.cli import main
from agent6.ui.telemetry import export
from agent6.viewmodel import telemetry_records
from tests.unit import test_telemetry_fold as fold_tests
from tests.unit import test_telemetry_machine as machine_tests


@dataclasses.dataclass
class _Collector:
    """A fake OTLP endpoint: records every request and answers from a script of statuses."""

    statuses: list[int] = dataclasses.field(default_factory=list)
    requests: list[tuple[str, dict[str, str], dict[str, Any]]] = dataclasses.field(
        default_factory=list
    )
    slept: list[float] = dataclasses.field(default_factory=list)
    raise_first: int = 0  # transport failures before the first answer

    def post(self, url: str, headers: dict[str, str], content: bytes) -> httpx2.Response:
        self.requests.append((url, headers, json.loads(content)))
        if self.raise_first:
            self.raise_first -= 1
            raise httpx2.ConnectError("refused", request=httpx2.Request("POST", url))
        status = self.statuses.pop(0) if self.statuses else 200
        reply_headers = {"Retry-After": "0"} if status == 503 else {}
        return httpx2.Response(
            status,
            request=httpx2.Request("POST", url),
            content=b"{}" if status == 200 else b"",
            headers=reply_headers,
        )

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

    def paths(self) -> list[str]:
        return [url.rsplit("/v1/", 1)[-1] for url, _headers, _body in self.requests]


def _write_session(repo: pathlib.Path, session_id: str, rows: list[dict[str, Any]]) -> int:
    d = paths.state_dir(repo) / "sessions" / "runs" / session_id
    d.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    (d / "logs.jsonl").write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _config(**changes: Any) -> TelemetryConfig:
    base: dict[str, Any] = {
        "traces": "otlp",
        "metrics": "otlp",
        "logs": "otlp",
        "endpoint": "http://127.0.0.1:4318",
        "headers": ("Authorization",),
    }
    return TelemetryConfig(**{**base, **changes})


def _cursors(repo: pathlib.Path) -> dict[str, int]:
    root = paths.state_dir(repo) / "telemetry"
    return {c.name: int(c.read_text().split()[0]) for c in sorted(root.rglob("*.offset"))}


@pytest.fixture
def repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    root = tmp_path / "repo"
    root.mkdir()
    return root


def _resumed() -> list[dict[str, Any]]:
    ts = fold_tests._ts  # pyright: ignore[reportPrivateUsage]
    return [
        *fold_tests._journal(),  # pyright: ignore[reportPrivateUsage]
        {"type": "loop.resume.start", "ts": ts(100), "execution_id": "e2", "turn": 2},
        {"type": "session.end", "ts": ts(120), "reason": "finish_session", "verified": "passed"},
    ]


def test_an_export_posts_every_signal_and_a_rerun_resends_only_the_metrics(
    repo: pathlib.Path,
) -> None:
    """Spans and logs go once, under per-signal cursors; the cumulative metrics go every time."""
    first = fold_tests._journal()  # pyright: ignore[reportPrivateUsage]
    size_a = _write_session(repo, "run-a", first)
    size_b = _write_session(repo, "run-b", _resumed())
    collector = _Collector()
    report = export.export(
        repo,
        _config(),
        headers={"Authorization": "Bearer t"},
        post=collector.post,
        sleep=collector.sleep,
    )
    assert collector.paths() == ["traces", "logs", "traces", "logs", "metrics"]
    url, headers, body = collector.requests[0]
    assert url == "http://127.0.0.1:4318/v1/traces"
    assert headers == {"Content-Type": "application/json", "Authorization": "Bearer t"}
    resource = {a["key"]: a["value"] for a in body["resourceSpans"][0]["resource"]["attributes"]}
    assert resource["service.name"] == {"stringValue": "agent6"}
    assert resource["agent6.export.scope"] == {"stringValue": "repository"}
    assert len(resource["agent6.repo"]["stringValue"]) == 16
    assert str(repo) not in json.dumps(collector.requests)
    spans_a = len(fold_tests._run(first)[1])  # pyright: ignore[reportPrivateUsage]
    spans_b = len(fold_tests._run(_resumed())[1])  # pyright: ignore[reportPrivateUsage]
    assert (report.sessions, report.spans, report.metric_posts) == (2, spans_a + spans_b, 1)
    assert report.logs > 0 and report.notes == ()
    metrics = collector.requests[-1][2]["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    sessions_total = next(m for m in metrics if m["name"] == "agent6_sessions_total")
    (point,) = sessions_total["sum"]["dataPoints"]
    assert point["asDouble"] == 2.0
    # The cumulative window starts at the earliest journal event, the same on every run.
    first_ns = telemetry_records.event_ns(fold_tests._ts(0))  # pyright: ignore[reportPrivateUsage]
    assert point["startTimeUnixNano"] == str(first_ns)
    assert _cursors(repo) == {
        "run-a.logs.offset": size_a,
        "run-a.traces.offset": size_a,
        "run-b.logs.offset": size_b,
        "run-b.traces.offset": size_b,
    }

    again = _Collector()
    report = export.export(
        repo, _config(), headers={"Authorization": "Bearer t"}, post=again.post, sleep=again.sleep
    )
    assert again.paths() == ["metrics"] and (report.spans, report.logs) == (0, 0)
    start_again = again.requests[0][2]["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]
    assert start_again["sum"]["dataPoints"][0]["startTimeUnixNano"] == str(first_ns)


def test_a_signal_enabled_later_still_gets_the_history(repo: pathlib.Path) -> None:
    """A metrics-only run moves no span or log cursor, so traces turned on afterwards backfill."""
    _write_session(repo, "run-a", fold_tests._journal())  # pyright: ignore[reportPrivateUsage]
    only_metrics = _Collector()
    export.export(repo, _config(traces="none", logs="none"), post=only_metrics.post)
    assert only_metrics.paths() == ["metrics"] and _cursors(repo) == {}
    traces_too = _Collector()
    report = export.export(repo, _config(logs="none"), post=traces_too.post)
    assert traces_too.paths() == ["traces", "metrics"] and report.spans > 0
    assert list(_cursors(repo)) == ["run-a.traces.offset"]
    logs_too = _Collector()
    report = export.export(repo, _config(), post=logs_too.post)
    assert logs_too.paths() == ["logs", "metrics"] and report.logs > 0


def test_a_retryable_status_is_retried_and_a_refusal_ends_the_destination(
    repo: pathlib.Path,
) -> None:
    """A 503 and a dropped connection are retried; a 401 turns the endpoint off, cursors held."""
    _write_session(repo, "run-a", fold_tests._journal())  # pyright: ignore[reportPrivateUsage]
    retried = _Collector(statuses=[503, 200], raise_first=1)
    report = export.export(repo, _config(), post=retried.post, sleep=retried.sleep)
    assert retried.paths()[:3] == ["traces", "traces", "traces"]
    assert len(retried.slept) == 2 and retried.slept[1] == 0.0
    assert report.notes == () and report.spans > 0

    refused = _Collector(statuses=[401])
    for cursor in (paths.state_dir(repo) / "telemetry").rglob("*.offset"):
        cursor.unlink()
    report = export.export(repo, _config(), post=refused.post, sleep=refused.sleep)
    assert len(refused.requests) == 1 and refused.slept == []
    assert report.spans == 0 and report.metric_posts == 0
    assert len(report.notes) == 1 and "HTTP 401" in report.notes[0] and "off" in report.notes[0]
    assert _cursors(repo) == {}

    down = _Collector(raise_first=3)
    report = export.export(repo, _config(), post=down.post, sleep=down.sleep)
    assert "ConnectError after 3 attempts" in report.notes[0] and _cursors(repo) == {}


def test_a_failed_last_batch_keeps_its_cursor_for_the_next_run(repo: pathlib.Path) -> None:
    """When the batch that ends the journal is refused, nothing marks it delivered."""
    ts = fold_tests._ts  # pyright: ignore[reportPrivateUsage]
    rows: list[dict[str, Any]] = [{"type": "session.start", "ts": ts(0), "execution_id": "e"}]
    for i in range(export.BATCH):
        rows.append({"type": "tool.call", "ts": ts(i), "name": "read_file", "call_id": i})
        rows.append(
            {"type": "tool.result", "ts": ts(i), "name": "read_file", "ok": True, "call_id": i}
        )
    rows.append({"type": "session.end", "ts": ts(999), "reason": "finish_session"})
    _write_session(repo, "run-a", rows)
    refused = _Collector(statuses=[200, 401])
    report = export.export(repo, _config(logs="none", metrics="none"), post=refused.post)
    assert refused.paths() == ["traces", "traces"]
    assert report.spans == export.BATCH and len(report.notes) == 1
    resent = _Collector()
    report = export.export(repo, _config(logs="none", metrics="none"), post=resent.post)
    assert resent.paths() == ["traces"] and report.spans == 1  # the execution span


def test_the_prometheus_textfile_is_readable_and_nothing_is_dialled(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    _write_session(repo, "run-a", fold_tests._journal())  # pyright: ignore[reportPrivateUsage]
    out = tmp_path / "node" / "agent6.prom"
    out.parent.mkdir()

    def never(url: str, headers: dict[str, str], content: bytes) -> httpx2.Response:
        raise AssertionError(f"dialled {url}")

    config = TelemetryConfig(metrics="prometheus", prometheus_file=str(out))
    report = export.export(repo, config, post=never)
    assert report.prometheus_file == str(out) and report.metric_posts == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# HELP agent6_sessions_total ")
    assert 'agent6_verify_attempts_total{outcome="failed"} 1' in text
    assert [p.name for p in out.parent.iterdir()] == ["agent6.prom"]
    assert stat.S_IMODE(out.stat().st_mode) == export.TEXTFILE_MODE


def test_follow_posts_as_the_journal_grows_and_drains_when_told_to_stop(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A follow delivers while the run writes, not only at its end; a stop drains the rest."""
    monkeypatch.setattr(export, "FLUSH_S", 0.0)
    monkeypatch.setattr(export, "POLL_S", 0.01)
    ts = fold_tests._ts  # pyright: ignore[reportPrivateUsage]
    _write_session(repo, "run-a", [{"type": "session.start", "ts": ts(0), "execution_id": "e"}])
    journal = paths.state_dir(repo) / "sessions" / "runs" / "run-a" / "logs.jsonl"
    collector = _Collector()
    stop = threading.Event()
    first_post = threading.Event()
    real_post = collector.post

    def post(url: str, headers: dict[str, str], content: bytes) -> httpx2.Response:
        first_post.set()
        return real_post(url, headers, content)

    def writer() -> None:
        # The first pair is posted before the rest is written, so the follow delivers live.
        with journal.open("a", encoding="utf-8") as fh:
            for i in range(3):
                call = {"type": "tool.call", "ts": ts(i + 1), "name": "f", "call_id": i}
                result = {
                    "type": "tool.result",
                    "ts": ts(i + 1),
                    "name": "f",
                    "ok": True,
                    "call_id": i,
                }
                fh.write(json.dumps(call) + "\n" + json.dumps(result) + "\n")
                fh.flush()
                if i == 0:
                    assert first_post.wait(timeout=5), "nothing was posted for the first pair"
            end = {"type": "session.end", "ts": ts(9), "reason": "finish_session"}
            fh.write(json.dumps(end) + "\n")
        stop.set()

    thread = threading.Thread(target=writer)
    thread.start()
    with pytest.raises(export.ExportError, match="needs a session id"):
        export.export(repo, _config(), follow=True, post=collector.post)
    report = export.export(
        repo,
        _config(logs="none"),
        session_id="run-a",
        follow=True,
        post=post,
        sleep=collector.sleep,
        should_stop=stop.is_set,
    )
    thread.join(timeout=5)
    assert report.sessions == 1 and report.spans == 4  # three tools and the execution
    assert collector.paths().count("metrics") >= 2  # published with a flush and at the end
    with pytest.raises(export.ExportError, match="no session 'run-z'"):
        export.export(repo, _config(), session_id="run-z", post=collector.post)


def test_a_live_runs_unfinished_work_is_not_sent_as_incomplete(repo: pathlib.Path) -> None:
    """Under a live worker the open spans wait for their end; a dead worker's are sent closed."""
    ts = fold_tests._ts  # pyright: ignore[reportPrivateUsage]
    rows = [
        {"type": "session.start", "ts": ts(0), "execution_id": "e"},
        {"type": "tool.call", "ts": ts(1), "name": "f", "call_id": 1},
    ]
    _write_session(repo, "run-a", rows)
    session_dir = paths.state_dir(repo) / "sessions" / "runs" / "run-a"
    process.write_worker_pid(session_dir, os.getpid())
    try:
        live = _Collector()
        report = export.export(repo, _config(logs="none", metrics="none"), post=live.post)
    finally:
        process.clear_worker_pid(session_dir)
    assert report.spans == 0 and live.paths() == []
    dead = _Collector()
    report = export.export(repo, _config(logs="none", metrics="none"), post=dead.post)
    assert report.spans == 2 and dead.paths() == ["traces"]
    spans = dead.requests[0][2]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    incomplete = {"key": "agent6.incomplete", "value": {"boolValue": True}}
    assert all(incomplete in s["attributes"] for s in spans)


def _spans(collector: _Collector) -> list[dict[str, Any]]:
    return [
        span
        for url, _, body in collector.requests
        if url.endswith("/traces")
        for span in body["resourceSpans"][0]["scopeSpans"][0]["spans"]
    ]


@pytest.mark.parametrize("batch", [1, export.BATCH])
@pytest.mark.parametrize("accepted", [True, False])
def test_a_resumed_crash_delivers_each_span_once_per_endpoint(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, batch: int, accepted: bool
) -> None:
    """A delivered close is not replayed at resume; a refused close gets the journaled end."""
    monkeypatch.setattr(export, "BATCH", batch)
    ts = fold_tests._ts  # pyright: ignore[reportPrivateUsage]
    crashed = fold_tests._journal()[:3]  # pyright: ignore[reportPrivateUsage]
    _write_session(repo, "run-a", crashed)
    config = _config(logs="none", metrics="none")
    first = _Collector(statuses=[] if accepted else [401])
    report = export.export(repo, config, post=first.post)
    assert report.spans == (3 if accepted else 0)
    delivered = _spans(first) if accepted else []
    if accepted:
        unchanged = _Collector()
        assert export.export(repo, config, post=unchanged.post).spans == 0
    journal = paths.state_dir(repo) / "sessions" / "runs" / "run-a" / "logs.jsonl"
    with journal.open("a") as fh:
        fh.write(
            json.dumps(
                {"type": "loop.resume.start", "ts": ts(100), "execution_id": "e2", "turn": 2}
            )
            + "\n"
        )
        fh.write(
            json.dumps(
                {
                    "type": "session.end",
                    "ts": ts(120),
                    "reason": "finish_session",
                    "verified": "passed",
                }
            )
            + "\n"
        )
    resumed = _Collector()
    export.export(repo, config, post=resumed.post)
    delivered.extend(_spans(resumed))
    ids = [s["spanId"] for s in delivered]
    assert len(ids) == len(set(ids)) == 5
    interrupted = [
        s for s in delivered if s["startTimeUnixNano"] == str(telemetry_records.event_ns(ts(0)))
    ]
    assert len(interrupted) == 1
    end = ts(1) if accepted else ts(100)
    assert interrupted[0]["endTimeUnixNano"] == str(telemetry_records.event_ns(end))
    again = _Collector()
    assert export.export(repo, config, post=again.post).spans == 0
    elsewhere = _Collector()
    other = config.model_copy(update={"endpoint": "http://127.0.0.1:4319"})
    assert export.export(repo, other, post=elsewhere.post).spans == 5
    assert len({s["spanId"] for s in _spans(elsewhere)}) == 5


@pytest.mark.parametrize("closed", [False, True])
def test_legacy_cursors_keep_their_offset_and_close_marker(
    repo: pathlib.Path, closed: bool
) -> None:
    """An offset with an optional closed word needs no additional cursor fields."""
    import hashlib

    rows = fold_tests._journal()[:3]  # pyright: ignore[reportPrivateUsage]
    size = _write_session(repo, "run-a", rows)
    config = _config(logs="none", metrics="none")
    key = hashlib.sha256(config.endpoint.encode()).hexdigest()[:16]
    cursor = paths.state_dir(repo) / "telemetry" / key / "run-a.traces.offset"
    cursor.parent.mkdir(parents=True)
    cursor.write_text(f"{size}{' closed' if closed else ''}\n")
    collector = _Collector()
    assert export.export(repo, config, post=collector.post).spans == (0 if closed else 3)
    assert cursor.read_text().split()[:2] == [str(size), "closed"]


@pytest.mark.parametrize("selected", ["", "demo"])
def test_machine_exports_parent_workers_and_do_not_double_count_spend(
    repo: pathlib.Path, selected: str
) -> None:
    """Machine discovery includes worker journals under their state, with separate cursors."""
    root = paths.state_dir(repo) / "machines" / "demo"
    state = root / "states" / "0003-review"
    state.mkdir(parents=True)
    (root / "journal.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in machine_tests.journal())
    )
    (state / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in machine_tests.worker()))
    collector = _Collector()
    report = export.export(repo, _config(), session_id=selected, post=collector.post)
    assert report.spans == 7
    spans = [
        s
        for url, _, body in collector.requests
        if url.endswith("/traces")
        for s in body["resourceSpans"][0]["scopeSpans"][0]["spans"]
    ]
    review = next(
        s
        for s in spans
        if {"key": "agent6.machine.state", "value": {"stringValue": "review"}} in s["attributes"]
    )
    execution = next(s for s in spans if s["name"] == "agent6.execution")
    call = next(s for s in spans if s["name"] == "agent6.model_call")
    assert execution["parentSpanId"] == review["spanId"]
    assert call["parentSpanId"] == execution["spanId"]
    assert execution["traceId"] == call["traceId"] == review["traceId"]
    metrics = collector.requests[-1][2]["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    cost = next(m for m in metrics if m["name"] == "agent6_cost_usd_total")
    assert sum(p["asDouble"] for p in cost["sum"]["dataPoints"]) == 0.25
    tokens = next(m for m in metrics if m["name"] == "agent6_tokens_total")
    assert sum(p["asDouble"] for p in tokens["sum"]["dataPoints"]) == 1200
    again = _Collector()
    report = export.export(repo, _config(), session_id=selected, post=again.post)
    assert (report.spans, report.logs) == (0, 0)
    assert again.paths() == ["metrics"]


def test_machine_follow_discovers_workers_before_the_next_transition(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker's records go out while the supervisor journal is idle."""
    monkeypatch.setattr(export, "FLUSH_S", 0.0)
    monkeypatch.setattr(export, "POLL_S", 0.0)
    root = paths.state_dir(repo) / "machines" / "demo"
    root.mkdir(parents=True)
    journal = root / "journal.jsonl"
    rows = machine_tests.journal()
    journal.write_text("".join(json.dumps(e) + "\n" for e in rows[:4]))
    collector = _Collector()
    polls = 0

    def stop() -> bool:
        nonlocal polls
        polls += 1
        if polls == 2:
            state = root / "states" / "0003-review"
            state.mkdir(parents=True)
            (state / "logs.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in machine_tests.worker())
            )
        if polls == 3:
            assert "agent6.model_call" in json.dumps(collector.requests)
            with journal.open("a") as fh:
                fh.write("".join(json.dumps(e) + "\n" for e in rows[4:]))
            return True
        return False

    process.write_worker_pid(root, os.getpid())
    try:
        report = export.export(
            repo, _config(), session_id="demo", follow=True, post=collector.post, should_stop=stop
        )
    finally:
        process.clear_worker_pid(root)
    assert report.spans == 7


def test_retry_after_reads_seconds_and_http_dates() -> None:
    assert export.retry_after_s("7") == 7.0
    soon = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=30)
    delay = export.retry_after_s(soon.strftime("%a, %d %b %Y %H:%M:%S GMT"))
    assert delay is not None and 25.0 <= delay <= 30.0
    assert export.retry_after_s("later") is None and export.retry_after_s(None) is None


def test_a_machine_overlay_may_not_select_an_export() -> None:
    """An authored machine picks neither host egress nor content capture: operator-only."""
    from agent6.machine import schema

    refusal = schema.protected_overlay_key_error("telemetry.endpoint")
    assert refusal is not None and "telemetry" in refusal and "operator-only" in refusal
    assert schema.protected_overlay_key_error("telemetry.capture", "review") is not None


def test_the_cli_refuses_an_unselected_export_and_names_a_missing_header_secret(
    repo: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command says what to set, where a header's value is missing, and what it delivered."""
    monkeypatch.chdir(repo)
    _write_session(repo, "run-a", fold_tests._journal())  # pyright: ignore[reportPrivateUsage]
    nothing = tmp_path / "nothing.toml"
    nothing.write_text("[telemetry]\n", encoding="utf-8")
    assert main(["--config", str(nothing), "telemetry", "export"]) == 2
    assert "nothing is selected" in capsys.readouterr().err
    secretless = tmp_path / "otlp.toml"
    secretless.write_text(
        '[telemetry]\ntraces = "otlp"\nendpoint = "https://collector.example"\n'
        'headers = ["Authorization"]\n',
        encoding="utf-8",
    )
    assert main(["--config", str(secretless), "telemetry", "export"]) == 2
    assert "[telemetry.headers]" in capsys.readouterr().err
    out = tmp_path / "agent6.prom"
    textfile = tmp_path / "prom.toml"
    textfile.write_text(
        f'[telemetry]\nmetrics = "prometheus"\nprometheus_file = "{out}"\n', encoding="utf-8"
    )
    assert main(["--config", str(textfile), "telemetry", "export"]) == 0
    assert f"metrics written to {out}" in capsys.readouterr().out
    assert out.read_text(encoding="utf-8").startswith("# HELP ")
