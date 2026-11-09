from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Mapping

import pytest

from agent6.sessions import event_log

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _emitted_types(source: str, path: str = "") -> set[str]:
    """Collect literal journal types, including the writers' forwarding shapes."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            index = 0
            prefix = ""
            if name == "emit_session_start":
                index = 2
            elif name == "_emit" and path.endswith("app/review_actions.py"):
                index = 2
                prefix = "loop.review."
            elif name not in {"emit", "_emit", "emit_start", "_emit_start", "_journal"}:
                for kw in node.keywords:
                    if (
                        name in {"Nudge", "Stop", "Refusal"}
                        and kw.arg == "event"
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)
                    ):
                        found.add(kw.value.value)
                continue
            if len(node.args) > index:
                arg = node.args[index]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.add(prefix + arg.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "STEER_VERBS"
            and isinstance(node.value, ast.Dict)
        ):
            for value in node.value.values:
                assert isinstance(value, ast.Tuple)
                event = value.elts[1]
                assert isinstance(event, ast.Constant) and isinstance(event.value, str)
                found.add(event.value)
    return found


def test_inventory_follows_forwarders_not_unrelated_strings() -> None:
    source = """
sink.emit("direct", text="not.an.event")
_emit(events, request, "finding.reopened", reason="why")
process.emit_session_start(events, directory, "session.start")
self._journal("approval.answer", approved=True)
advice.Nudge("notice", event="loop.no_progress.nudge")
STEER_VERBS: dict = {"exit": ("exit", "loop.steer.exited", "log text")}
"""
    assert _emitted_types(source, "app/review_actions.py") == {
        "direct",
        "loop.review.finding.reopened",
        "session.start",
        "approval.answer",
        "loop.no_progress.nudge",
        "loop.steer.exited",
    }


def test_registry_matches_writer_inventory() -> None:
    """A new writer or an orphaned registry entry requires an explicit schema edit."""
    from agent6.sessions import event_schema

    emitted: set[str] = set()
    for path in (_ROOT / "src" / "agent6").rglob("*.py"):
        emitted |= _emitted_types(path.read_text(encoding="utf-8"), path.as_posix())
    registered = set(event_schema.REGISTRY)
    assert not emitted - registered, f"unregistered writers: {sorted(emitted - registered)}"
    assert not registered - emitted, f"no writer: {sorted(registered - emitted)}"
    assert [spec.type for spec in event_schema.EVENTS] == sorted(registered)
    for spec in event_schema.EVENTS:
        assert event_schema.spec_for(spec.type) is spec
        assert spec.meaning
        assert len({field.name for field in spec.fields}) == len(spec.fields)
        assert all(field.meaning for field in spec.fields)
    assert event_schema.spec_for("not.registered") is None


def test_provider_request_registry_documents_call_and_seat_identity() -> None:
    """The request schema includes its parent call and optional panel seat identity."""
    from agent6.sessions import event_schema

    fields = {field.name: field for field in event_schema.REGISTRY["provider.request"].fields}
    assert {"model_call_id", "panel_id", "seat_index"} <= fields.keys()
    assert fields["model_call_id"].kind == "str" and fields["model_call_id"].required
    assert fields["panel_id"].kind == "str" and not fields["panel_id"].required
    assert fields["seat_index"].kind == "int" and not fields["seat_index"].required


def test_session_lifecycle_names_keep_creation_resume_and_end_distinct() -> None:
    """The registry names session creation, resumed executions and execution ends separately."""
    from agent6.sessions import event_schema
    from agent6.viewmodel import events

    for name in ("session.start", "session.resume", "session.stop"):
        assert event_schema.spec_for(name) is not None, name
    assert event_schema.spec_for("loop.resume.start") is None
    assert frozenset({"session.resume", "session.start"}) == events.SESSION_START_EVENTS
    assert isinstance(events.parse_event({"type": "loop.resume.start"}), events.RawEvent)


@pytest.mark.parametrize(
    ("event_type", "fields", "message"),
    [
        ("not.registered", {}, "unregistered"),
        ("loop.note", {}, "missing required field.*text"),
        ("loop.note", {"text": "ok", "typo": 1}, "unknown field.*typo"),
        ("loop.note", {"text": 3}, "text.*str"),
        ("loop.note", {"text": None}, "text.*str"),
        ("loop.compact.gist.call", {"files": True}, "files.*int"),
        ("budget.update", {"usd_total": True}, "usd_total.*float"),
        (
            "session.stop",
            {"execution_id": "execution", "reason": "done", "turns": 1, "verification": 1},
            "verification.*str",
        ),
        ("loop.pin.restored", {"pins": "bad", "count": 1}, "pins.*list"),
        ("graph.update", {"nodes": [], "cursor": None}, "nodes.*object"),
        (
            "loop.review.skipped",
            {"panel_id": "p", "execution_id": "e", "reason": "r", "ts": "fake"},
            "unknown",
        ),
    ],
)
def test_checker_rejects_invalid_writes(
    tmp_path: pathlib.Path, event_type: str, fields: dict[str, object], message: str
) -> None:
    """The autouse checker rejects invalid payloads before the sink appends them."""
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    with pytest.raises(pytest.fail.Exception, match=message):
        sink.emit(event_type, **fields)
    assert not sink.path.exists()


def test_real_sink_accepts_registered_fields(tmp_path: pathlib.Path) -> None:
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    sink.emit("loop.note", text="ready")
    sink.emit("graph.update", nodes={}, cursor=None)
    sink.emit("loop.pin.restored", pins=("keep tests",), count=1)
    sink.emit("budget.update", usd_total=0)
    rows = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert rows[0]["text"] == "ready"
    assert rows[1]["cursor"] is None
    assert rows[2]["pins"] == ["keep tests"]
    assert rows[3]["usd_total"] == 0


def test_production_sink_keeps_unchecked_writes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production does not validate journal payloads."""
    monkeypatch.setattr(event_log, "CHECK", None, raising=False)
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    sink.emit("future.event", new_field=3)
    assert json.loads(sink.path.read_text())["new_field"] == 3


def test_history_stays_tolerant() -> None:
    from agent6.viewmodel import events

    assert isinstance(events.parse_event({"type": "future.event"}), events.RawEvent)
    assert isinstance(events.parse_event({"type": "session.start"}), events.SessionStart)


def test_a_background_write_fails_even_if_the_thread_catches_it(tmp_path: pathlib.Path) -> None:
    """A thread cannot turn a checker violation into a warning or a swallowed exception."""
    (tmp_path / "conftest.py").write_text('pytest_plugins = ("tests.conftest",)\n')
    (tmp_path / "test_write.py").write_text("""
from __future__ import annotations
import threading
from agent6.sessions import event_log


def test_write(tmp_path):
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    rejected = []
    def write():
        try:
            sink.emit("loop.note")
        except BaseException as exc:
            rejected.append(exc)
    thread = threading.Thread(target=write)
    thread.start()
    thread.join()
    assert len(rejected) == 1
    assert not sink.path.exists()
""")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(_ROOT), str(_ROOT / "src")))},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "background journal writes" in result.stdout
    assert "loop.note: missing required field: text" in result.stdout


def test_sink_checks_before_writing_or_notifying(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected payload never reaches the journal or its listeners."""
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    received: list[dict[str, object]] = []
    sink.subscribe(received.append)

    def reject(event_type: str, fields: Mapping[str, object]) -> None:
        assert event_type == "session.stranded"
        assert fields == {}
        pytest.fail("rejected event")

    monkeypatch.setattr(event_log, "CHECK", reject, raising=False)
    with pytest.raises(pytest.fail.Exception, match="rejected event"):
        sink.emit("session.stranded")
    assert not sink.path.exists()
    assert received == []
