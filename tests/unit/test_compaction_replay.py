from __future__ import annotations

import json
import pathlib
import subprocess
import sys

from bench import compaction_replay


def test_size_replay_leaves_headroom_and_honours_recent_results() -> None:
    """Stepped drops retain fewer characters and rewrite fewer requests."""
    frames = [
        tuple(compaction_replay.Result(str(i), 1000, 100) for i in range(n)) for n in range(1, 13)
    ]
    old = compaction_replay.replay(frames, bound=6000, stepped=False)
    new = compaction_replay.replay(frames, bound=6000, stepped=True)
    assert old.drops == 6
    assert new.drops == 2
    assert new.mean_chars < old.mean_chars


def test_size_replay_resets_at_recorded_restarts() -> None:
    frames = [
        tuple(compaction_replay.Result(str(i), 1000, 100) for i in range(7)),
        (compaction_replay.Result("new", 500, 100),),
    ]
    result = compaction_replay.replay(frames, bound=6000, stepped=True)
    assert result.drops == 0  # the initial unseen batch is never elided
    assert result.mean_chars == 3750


def test_transcripts_supply_full_sizes_not_recorded_elisions(tmp_path: pathlib.Path) -> None:
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    for seq, output in enumerate(("x" * 1000, "<elided by context compaction: marker>")):
        (transcripts / f"{seq}.json").write_text(
            json.dumps(
                {
                    "seat": "worker",
                    "request": {
                        "body": {
                            "input": [
                                {
                                    "type": "function_call",
                                    "call_id": "one",
                                    "name": "read_file",
                                    "arguments": '{"path": "a.py"}',
                                },
                                {
                                    "type": "function_call_output",
                                    "call_id": "one",
                                    "output": output,
                                },
                            ]
                        }
                    },
                }
            )
        )
    frames = compaction_replay.read_frames(tmp_path)
    assert [frame[0].chars for frame in frames] == [1000, 1000]


def test_cli_refuses_missing_transcript_objects(tmp_path: pathlib.Path) -> None:
    transcripts = tmp_path / "session" / "transcripts"
    transcripts.mkdir(parents=True)
    (transcripts / "one.json").write_text(
        json.dumps({"seat": "worker", "request": {"body": {"input": [{"$object": "a" * 64}]}}})
    )
    proc = subprocess.run(
        [sys.executable, "bench/compaction_replay.py", str(tmp_path), "--drop-at-chars", "6000"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "unavailable" in proc.stdout
    assert "cannot read object" in proc.stderr
