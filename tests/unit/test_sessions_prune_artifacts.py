from __future__ import annotations

import datetime
import json
import pathlib

import pytest

from agent6 import paths
from agent6.harness import loop_snapshot
from agent6.providers import provider_types
from agent6.sessions import layout as sessions_layout
from agent6.sessions import lock
from agent6.ui.cli import main


def _session(
    repo: pathlib.Path, name: str, *, age: int, bucket: str = "runs"
) -> sessions_layout.SessionLayout:
    layout = sessions_layout.SessionLayout(paths.state_dir(repo), name, subdir=bucket)
    layout.ensure()
    end = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=age)
    events = [
        {
            "type": "session.start",
            "ts": (end - datetime.timedelta(hours=1)).isoformat(),
            "mode": "run",
        },
        {
            "type": "session.end",
            "ts": end.isoformat(),
            "reason": "finish_session",
            "all_passed": True,
        },
    ]
    layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in events))
    layout.manifest_path.write_text(
        json.dumps({"session_id": name, "mode": "run", "user_task": "task"})
    )
    snapshot = loop_snapshot.SessionSnapshot(
        system="system",
        messages=[{"role": "user", "content": "hello"}],
        tool_calls=0,
        next_turn=1,
        root_task_id=None,
        original_task="task",
        verify_command=(),
    )
    (layout.session_dir / "loop_state.json").write_text(snapshot.model_dump_json())
    loop_snapshot.write_checkpoint(layout.checkpoint_path(1), snapshot)
    (layout.session_dir / "receipt.json").write_text('{"kept": true}')
    (layout.graph_dir / "node.json").write_text('{"id": "0001"}')
    layout.journal_path.write_text('{"graph_version": 1}\n')
    sink = provider_types.TranscriptSink(layout.transcripts_dir, objects_dir=layout.objects_dir)
    for _ in range(2):
        sink.record(
            request_headers={},
            request_body={"messages": snapshot.messages},
            response_status=200,
            response_body={},
        )
    return layout


def test_prune_artifacts_keeps_recovery_and_records_counts_and_sequence(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Explicit age pruning deletes the three stores, not recovery, graph or journal state."""
    monkeypatch.chdir(tmp_path)
    layout = _session(tmp_path, "old-run", age=10)
    names = ("transcripts", "checkpoints", "objects")
    counts = {name: len(list((layout.session_dir / name).glob("*.json"))) for name in names}
    kept = {
        p: p.read_bytes()
        for p in layout.session_dir.rglob("*")
        if p.is_file() and p.parent.name not in names
    }
    before = datetime.datetime.now(datetime.UTC)
    assert main(["sessions", "prune", "--older-than", "7"]) == 0
    assert all(not (layout.session_dir / name).exists() for name in names)
    assert {p: p.read_bytes() for p in kept} == kept
    receipt = json.loads((layout.session_dir / "pruned.json").read_text())
    assert receipt["counts"] == counts
    assert receipt["last_transcript_seq"] == 2
    assert datetime.datetime.fromisoformat(receipt["pruned_at"]) >= before
    out = capsys.readouterr().out
    assert layout.session_id in out and all(name in out for name in names)
    stamp = receipt["pruned_at"]
    assert main(["sessions", "transcript", layout.session_id]) == 2
    assert stamp in capsys.readouterr().err
    assert main(["sessions", "transcript"]) == 2
    assert stamp in capsys.readouterr().err
    assert main(["fork", layout.session_id, "--at-turn", "1", "--no-run"]) == 2
    assert stamp in capsys.readouterr().err
    assert loop_snapshot.load_session_snapshot(layout.session_dir / "loop_state.json").messages
    resumed = provider_types.TranscriptSink(layout.transcripts_dir, objects_dir=layout.objects_dir)
    attempt = resumed.open(
        request_headers={}, request_body={"messages": [{"role": "user", "content": "again"}]}
    )
    assert attempt.seq == 3
    assert main(["sessions", "transcript", layout.session_id, "--json", "--seq", "3"]) == 0
    captured = capsys.readouterr()
    assert stamp in captured.err and json.loads(captured.out)[0]["seq"] == 3
    assert main(["sessions", "prune", "--older-than", "7"]) == 0
    assert (
        provider_types.TranscriptSink(layout.transcripts_dir, objects_dir=layout.objects_dir)
        .open(request_headers={}, request_body={})
        .seq
        == 4
    )


def test_prune_uses_last_execution_end_across_buckets_and_respects_writer_lock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recent resume, unfinished execution or held worker lock protects old artifacts."""
    monkeypatch.chdir(tmp_path)
    old = _session(tmp_path, "old-ask", age=10, bucket="asks")
    recent = _session(tmp_path, "recent-run", age=1)
    resumed = _session(tmp_path, "resumed-run", age=10)
    active = _session(tmp_path, "active-run", age=10)
    held = _session(tmp_path, "held-run", age=10)
    for layout in (resumed, active):
        with layout.logs_path.open("a") as log:
            log.write(
                json.dumps(
                    {
                        "type": "session.resume",
                        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                    }
                )
                + "\n"
            )
    with resumed.logs_path.open("a") as log:
        log.write(
            json.dumps(
                {"type": "session.end", "ts": datetime.datetime.now(datetime.UTC).isoformat()}
            )
            + "\n"
        )
    fd = lock.acquire_single_writer(held.session_dir)
    assert fd is not None
    try:
        assert main(["sessions", "prune", "--older-than", "7"]) == 0
    finally:
        lock.release_single_writer(fd)
    assert not old.transcripts_dir.exists()
    for layout in (recent, resumed, active, held):
        assert layout.transcripts_dir.is_dir()
        assert not (layout.session_dir / "pruned.json").exists()


def test_prune_rejects_negative_age(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A negative age cannot turn a retention request into deletion of every finished run."""
    monkeypatch.chdir(tmp_path)
    layout = _session(tmp_path, "recent-run", age=1)
    assert main(["sessions", "prune", "--older-than", "-1"]) == 2
    assert layout.transcripts_dir.is_dir()
