from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from typing import Any

import pytest
from bench import compaction_arms, compaction_messages, compaction_slot, compaction_small


def test_all_settings_change_the_expected_first_request(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Six long scripted tool results distinguish the eleven treatments at the wire."""
    for name in ("CONFIG", "STATE", "CACHE", "DATA"):
        monkeypatch.setenv(f"XDG_{name}_HOME", str(tmp_path / name.lower()))
    profile = tmp_path / "config/agent6/config.toml"
    profile.parent.mkdir(parents=True)
    profile.write_text('[sandbox]\nrun_commands="allow"\n')
    seed = compaction_small.prepare(tmp_path / "dummy")
    path = seed.session / "loop_state.json"
    snap = json.loads(path.read_bytes())
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "origin": "operator", "text": "Earlier steer: amber-741."}
            ],
        }
    ]
    for i in range(6):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": f"reason-{i}", "signature": f"sig-{i}"},
                        {
                            "type": "tool_use",
                            "id": f"read-{i}",
                            "name": "read_file",
                            "input": {"path": "pairs.py"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"read-{i}",
                            "content": f"evidence-{i} " + "x" * 19000,
                        }
                    ],
                },
            ]
        )
    path.write_text(json.dumps(snap | {"messages": messages}))
    seed = dataclasses.replace(seed, digest=hashlib.sha256(path.read_bytes()).hexdigest())
    first: dict[str, str] = {}
    calls: list[str] = []

    def reply(body: dict[str, Any]) -> str | tuple[str, dict[str, Any]]:
        if not body.get("tools") or "Write the summary now." in str(body["messages"][-1]):
            return "A handoff."
        calls.append(json.dumps(body["messages"]))
        return "finish_session", {"summary": "Done.\nfield-amber-741"}

    with compaction_messages.scripted(reply) as url:
        for arm in (*compaction_arms.KEEP, *compaction_arms.COLD):
            calls.clear()
            row = dict(
                arm=arm,
                task="replay",
                model="scripted/claude-sonnet-4-5",
                trial=str(tmp_path / arm),
            )
            with compaction_slot.restored(seed, tmp_path / arm) as session:
                compaction_slot.resume(
                    seed,
                    session,
                    row,
                    gate=("python3", "-c", 'print("fixture ok")'),
                    scripted_url=url,
                )
            first[arm] = calls[0]
    assert "evidence-5" in first["baseline"] and "evidence-0" not in first["baseline"]
    assert "amber-741." in first["baseline"] and "amber-741." not in first["no-operator"]
    assert "evidence-4" in first["baseline"] and "evidence-4" not in first["chars-20000"]
    assert "evidence-4" not in first["turns-1"] and "evidence-5" in first["turns-1"]
    assert "amber-741." in first["operator-only"] and "evidence-4" not in first["operator-only"]
    assert "evidence-5" in first["operator-only"]  # The last result has no subsequent reader.
    assert (
        "evidence-5" not in first["pointers"]
        and "<elided by context compaction" in first["pointers"]
    )
    assert "evidence-0" in first["resend"] and "evidence-0" not in first["drop"]
    assert "reason-3" in first["drop"] and "reason-3" not in first["drop-thinking-2"]
    assert "sig-4" in first["drop-thinking-2"] and "sig-5" in first["drop-thinking-2"]
    assert "A handoff." in first["summarize"] and "A handoff." not in first["truncate"]
    assert "earlier history was left out" in first["truncate"]
