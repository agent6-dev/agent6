from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import Any


def test_settings_count_only_journal_text_before_the_cut(tmp_path: pathlib.Path) -> None:
    """No checkout, manifest, object store or provider is needed to count a journal."""
    rows: list[dict[str, Any]] = [
        dict(type="session.start", user_task="Goal"),
        dict(type="execution.settings", settings=[{"key": "sandbox.hide_paths", "value": "x"}]),
        dict(type="steer.injected", text="Label"),
    ]
    for i in range(1, 6):
        rows.extend(
            [
                dict(type="budget.turn", turn=i),
                dict(type="tool.call", name="read_file", call_id=i, args_preview={"path": "a"}),
                dict(type="tool.result", name="read_file", call_id=i, summary="x" * 25000),
            ]
        )
    journal = tmp_path / "logs.jsonl"
    journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cmd = [sys.executable, "-m", "bench.compaction_offline", str(journal), "--turn", "5"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    arms = {r["arm"]: r for line in proc.stdout.splitlines() if (r := json.loads(line))}
    assert len(arms) == 6
    assert arms["baseline"]["quote_chars"] == 9
    assert arms["no-operator"]["quote_chars"] == 0
    assert arms["chars-20000"]["kept_chars"] == arms["turns-1"]["kept_chars"]
    assert arms["baseline"]["kept_chars"] > arms["turns-1"]["kept_chars"]
    assert arms["pointers"]["pointer_chars"] > 0
    assert arms["pointers"]["kept_chars"] < arms["baseline"]["kept_chars"] - 70000
    for arm in arms.values():
        assert 100000 < arm["source_chars"] < 101000
        assert arm["kept_chars"] + arm["dropped_chars"] == arm["source_chars"]
    assert "xxxxx" not in proc.stdout and list(tmp_path.iterdir()) == [journal]
    journal.write_text(journal.read_text() + '{"type":"budget.turn","turn":5}\n')
    duplicate = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert duplicate.returncode != 0 and "ambiguous" in duplicate.stderr and not duplicate.stdout
