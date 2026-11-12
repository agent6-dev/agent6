from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import pytest

from agent6 import paths, portable
from agent6.harness import loop_conversation, loop_snapshot
from agent6.harness.compaction import rules as compaction_rules
from agent6.providers import provider_types
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import history_transcripts, sessions_transcript
from agent6.viewmodel import transcript_render

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _reference(value: Any, directory: pathlib.Path) -> dict[str, str]:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{digest}.json").write_text(text)
    return {"$object": digest}


def test_a_bare_transcript_directory_keeps_its_objects_inside(tmp_path: pathlib.Path) -> None:
    """A review directory owns its objects instead of spilling them into its parent."""
    from agent6.sessions import objects

    reviews = tmp_path / "reviews"
    sink = provider_types.TranscriptSink(reviews, objects_dir=reviews / "objects")
    body = {"messages": [{"role": "user", "content": "review this"}]}
    attempt = sink.open(request_headers={}, request_body=body)
    assert attempt.path is not None
    assert (reviews / "objects").is_dir()
    attempt.response_status = 200
    attempt.write("completed")
    record = objects.materialize(json.loads(attempt.path.read_text()), reviews / "objects")
    assert record["request"]["body"] == body
    assert not (tmp_path / "objects").exists()


@pytest.mark.parametrize("field", ["messages", "input"])
def test_objects_are_sanitized_canonical_and_reused_on_retry(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Requests publish sanitized objects before their records and retries write none again."""
    secret = "sk-private-credential-123456789"
    message = {"role": "user", "content": secret}
    headers = {"authorization": f"Bearer {secret}"}
    body = {field: [message], "system": "sys", "instructions": "inst", "tools": [], "model": "m"}
    writes: list[pathlib.Path] = []
    real_write = portable.atomic_write

    def write(path: pathlib.Path, data: str | bytes) -> None:
        if path.parent.name == "transcripts":
            record = json.loads(data)
            for key in ("system", "instructions", "tools"):
                digest = record["request"]["body"][key]["$object"]
                assert (tmp_path / "objects" / f"{digest}.json").is_file()
        writes.append(path)
        real_write(path, data)

    monkeypatch.setattr(portable, "atomic_write", write)
    sink = provider_types.TranscriptSink(tmp_path / "transcripts", objects_dir=tmp_path / "objects")
    first = sink.open(request_headers=headers, request_body=body)
    assert first.path is not None
    raw = json.loads(first.path.read_text())
    ref = raw["request"]["body"][field][0]
    clean = {"role": "user", "content": "<REDACTED>"}
    canonical = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    assert ref == {"$object": hashlib.sha256(canonical.encode()).hexdigest()}
    assert (tmp_path / "objects" / f"{ref['$object']}.json").read_text() == canonical
    before = [p for p in writes if p.parent.name == "objects"]
    assert len(before) == 4
    first.write("failed")
    sink.open(request_headers=headers, request_body=body)
    assert [p for p in writes if p.parent.name == "objects"] == before
    assert raw["request"]["body"]["model"] == "m"
    assert body[field] == [message]
    assert all(secret not in p.read_text() for p in tmp_path.rglob("*.json"))


@pytest.mark.parametrize("referenced", [False, True])
@pytest.mark.parametrize("field", ["messages", "input"])
def test_transcript_readers_materialize_before_search_and_seq_window(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    referenced: bool,
    field: str,
) -> None:
    """Old and referenced records expose the same content through every transcript reader."""
    monkeypatch.chdir(tmp_path)
    layout = sessions_layout.SessionLayout(paths.state_dir(tmp_path), "objects-test")
    layout.ensure()
    layout.logs_path.write_text('{"type": "session.start"}\n')
    message = {"role": "user", "content": "operator needle"}
    record = {"seq": 2, "request": {"body": {field: [message]}}, "response": {"body": {}}}
    stored = json.loads(json.dumps(record))
    if referenced:
        stored["request"]["body"][field] = [_reference(message, layout.session_dir / "objects")]
    (layout.transcripts_dir / "0002.json").write_text(json.dumps(stored))
    assert transcript_render.load_transcripts(layout.transcripts_dir) == [record]
    (searched,) = history_transcripts.search_text([layout.session_dir])
    assert "operator needle" in searched.text and not searched.error
    assert (
        sessions_transcript.cmd_transcript(
            layout.session_id, as_json=True, no_thinking=False, tools="both", seq="2"
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == [record]


def test_missing_object_names_hash_and_record_in_every_reader(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing object is an explicit error, even outside a requested sequence window."""
    from agent6.sessions import objects

    monkeypatch.chdir(tmp_path)
    layout = sessions_layout.SessionLayout(paths.state_dir(tmp_path), "missing-test")
    layout.ensure()
    layout.logs_path.write_text('{"type": "session.start"}\n')
    digest = "a" * 64
    record = {"seq": 1, "request": {"body": {"messages": [{"$object": digest}]}}}
    path = layout.transcripts_dir / "0001.json"
    path.write_text(json.dumps(record))
    with pytest.raises(objects.ObjectStoreError, match=f"{digest}.*0001.json"):
        objects.materialize(record, layout.session_dir / "objects", record_path=path)
    with pytest.raises(objects.ObjectStoreError, match=digest):
        transcript_render.load_transcripts(layout.transcripts_dir)
    (searched,) = history_transcripts.search_text([layout.session_dir])
    assert digest in searched.error and str(path) in searched.error
    assert (
        sessions_transcript.cmd_transcript(
            layout.session_id, as_json=True, no_thinking=False, tools="both", seq="2"
        )
        == 2
    )
    assert digest in capsys.readouterr().err


def test_bench_cost_reader_uses_materialized_records(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exported-state bench reader resolves objects through the shared decoder."""
    import glob

    from agent6.sessions import objects

    script = (_ROOT / "bench" / "swebench" / "in_container.sh").read_text()
    code = script.split(" > /out/cost.json", 1)[1].split("\n", 1)[1].split("\nPYEOF", 1)[0]
    path = provider_types.TranscriptSink(
        tmp_path / "transcripts", objects_dir=tmp_path / "objects"
    ).record(
        url="https://example.com/api",
        request_headers={},
        request_body={"messages": [{"role": "user", "content": "task"}]},
        response_status=200,
        response_body={"usage": {"cost": 0.25}},
    )

    def exported_paths(_: str) -> list[str]:
        return [str(path)]

    monkeypatch.setattr(glob, "glob", exported_paths)
    exec(compile(code, "bench cost reader", "exec"), {})
    assert json.loads(capsys.readouterr().out) == {"billed_usd": 0.25, "calls": 1}
    next((tmp_path / "objects").glob("*.json")).unlink()
    with pytest.raises(objects.ObjectStoreError, match=str(path)):
        exec(compile(code, "bench cost reader", "exec"), {})


def test_forty_compacting_calls_store_each_distinct_value_once(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Tier-1 rewrites add objects for changed messages, not for their unchanged suffix."""
    from agent6.sessions import objects

    sink = provider_types.TranscriptSink(tmp_path / "transcripts", objects_dir=tmp_path / "objects")
    conversation = loop_conversation.Conversation()
    conversation.notice("task")
    values: set[str] = set()
    full_bytes = 0
    elisions = 0
    for turn in range(1, 41):
        assistant = conversation.assistant(
            [
                {
                    "type": "tool_use",
                    "id": str(turn),
                    "name": "read_file",
                    "input": {"path": f"{turn}.txt"},
                }
            ]
        )
        conversation.results(
            [
                loop_conversation.ToolResultItem(
                    str(turn), f"result {turn}: " + "x" * 8000, assistant.tool_uses[0]
                )
            ]
        )
        stats = compaction_rules.compact_old_tool_results(
            conversation, max_total_bytes=60000, keep_recent=4
        )
        elisions += len(stats.elided_calls)
        messages = conversation.to_wire()
        body: dict[str, Any] = {"messages": messages, "system": "system prompt " * 100, "tools": []}
        for value in [*messages, body["system"], body["tools"]]:
            values.add(json.dumps(value, sort_keys=True, separators=(",", ":")))
        path = sink.record(
            request_headers={}, request_body=body, response_status=200, response_body={}
        )
        full = objects.materialize(json.loads(path.read_text()), tmp_path / "objects")
        full_bytes += len(json.dumps(full, indent=2, sort_keys=True).encode())
        snapshot = loop_snapshot.SessionSnapshot(
            system=body["system"],
            messages=messages,
            tool_calls=turn,
            next_turn=turn,
            root_task_id=None,
            original_task="task",
            verify_command=(),
        )
        checkpoint = tmp_path / "checkpoints" / f"{turn:04d}.json"
        loop_snapshot.write_checkpoint(checkpoint, snapshot)
        assert loop_snapshot.load_session_snapshot(checkpoint) == snapshot
        full_bytes += len(snapshot.model_dump_json().encode())
    assert elisions > 0
    files = list((tmp_path / "objects").glob("*.json"))
    assert len(files) == len(values)
    assert {p.read_text() for p in files} == values
    stored_bytes = sum(p.stat().st_size for p in tmp_path.rglob("*.json"))
    ratio = stored_bytes / full_bytes
    assert ratio < 0.2
    with capsys.disabled():
        print(
            f"40-call storage: {len(files)} objects, "
            f"{stored_bytes}/{full_bytes} bytes = {ratio:.4%}"
        )
