# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for the pure web payload builders (no HTTP)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent6.ui.web import model


def _run(cwd: Path, run_id: str, events: list[dict[str, object]]) -> Path:
    d = model.runs_root(cwd) / run_id
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return d


def test_run_summary_captures_cost_and_status(tmp_path: Path) -> None:
    _run(
        tmp_path,
        "r1",
        [
            {"type": "run.start", "mode": "run", "user_task": "the task"},
            {"type": "budget.update", "usd_total": 0.0123},
            {"type": "run.end", "all_passed": True},
        ],
    )
    (s,) = model.hub_payload(tmp_path)["runs"]
    assert s["mode"] == "run"
    assert s["task"] == "the task"
    assert s["status"] == "passed"
    assert s["usd"] == 0.0123


def test_run_summary_survives_torn_utf8_tail(tmp_path: Path) -> None:
    # A live writer can leave the log's last line torn mid multibyte UTF-8
    # sequence; the hub summary must fold the complete lines, not raise.
    d = model.runs_root(tmp_path) / "torn"
    d.mkdir(parents=True)
    full = json.dumps({"type": "role.text_delta", "text": "café"}, ensure_ascii=False).encode()
    cut = full.rindex(b"\xc3\xa9") + 1  # keep only the first byte of the é
    head = json.dumps({"type": "run.start", "mode": "run", "user_task": "torn tail"}).encode()
    (d / "logs.jsonl").write_bytes(head + b"\n" + full[:cut])
    (s,) = model.hub_payload(tmp_path)["runs"]
    assert s["task"] == "torn tail"


def test_conversation_payload_folds_the_event_log(tmp_path: Path) -> None:
    # Items come from the shared TranscriptFold + item_lines renderer: a tool's
    # multi-line result is clipped to its first line + a "+N more lines" note,
    # with the full rendering carried separately for per-item expansion.
    dump = "3 validation errors for ApplyEditInput\npath\n  Field required"
    d = _run(
        tmp_path,
        "r2",
        [
            {"type": "run.start", "user_task": "x"},
            {"type": "tool.call", "name": "apply_edit", "args": {"path": "a.py"}},
            {"type": "tool.result", "name": "apply_edit", "ok": False, "summary": dump},
        ],
    )
    payload = model.conversation_payload(d)
    assert payload["run_id"] == "r2"
    (item,) = payload["items"]
    assert item["kind"] == "tool"
    flat = "".join(text for line in item["lines"] for text, _style in line)
    assert "(+2 more lines)" in flat
    assert "Field required" not in flat  # clipped in the collapsed rendering
    full = "".join(text for line in item["full"] for text, _style in line)
    assert "Field required" in full  # the expanded rendering carries it


def test_conversation_payload_empty_without_log(tmp_path: Path) -> None:
    d = model.runs_root(tmp_path) / "r2b"
    d.mkdir(parents=True)
    assert model.conversation_payload(d) == {"run_id": "r2b", "items": []}


def test_machine_conversation_payload_uses_newest_state_log(tmp_path: Path) -> None:
    md = model.machines_root(tmp_path) / "m2"
    (md / "states" / "0001-work").mkdir(parents=True)
    (md / "states" / "0001-work" / "logs.jsonl").write_text(
        json.dumps({"type": "loop.steer.injected", "text": "hello"}) + "\n", encoding="utf-8"
    )
    payload = model.machine_conversation_payload(md)
    assert payload["state_dir"] == "0001-work"
    (item,) = payload["items"]
    assert item["kind"] == "operator"
    assert model.machine_conversation_payload(model.machines_root(tmp_path) / "nope") == {
        "state_dir": "",
        "items": [],
    }


def test_reasoning_snapshot_empty_without_state_log(tmp_path: Path) -> None:
    # A machine dir with no states/ subtree has no agent reasoning to fold.
    md = model.machines_root(tmp_path) / "m1"
    md.mkdir(parents=True)
    assert model.machine_reasoning_snapshot(md) == {}


def test_run_dir_for_rejects_traversal(tmp_path: Path) -> None:
    _run(tmp_path, "good-run", [{"type": "run.start"}])
    assert model.run_dir_for(tmp_path, "good-run") is not None
    for bad in ("..", ".", "", "../good-run", "a/b", "..\\x"):
        assert model.run_dir_for(tmp_path, bad) is None


def test_machine_dir_for_rejects_traversal(tmp_path: Path) -> None:
    (model.machines_root(tmp_path) / "m1").mkdir(parents=True)
    assert model.machine_dir_for(tmp_path, "m1") is not None
    for bad in ("..", "../m1", "a/b", ""):
        assert model.machine_dir_for(tmp_path, bad) is None


def test_hub_payload_shape(tmp_path: Path) -> None:
    _run(tmp_path, "r3", [{"type": "run.start", "mode": "plan"}])
    hub = model.hub_payload(tmp_path)
    assert [r["id"] for r in hub["runs"]] == ["r3"]
    assert hub["machines"] == []


def test_hub_payload_lists_machine_drafts(tmp_path: Path) -> None:
    draft = model.state_dir_for(tmp_path) / "machine-drafts" / "breezy-fern-AB12CD"
    draft.mkdir(parents=True)
    (draft / "logs.jsonl").write_text(
        json.dumps({"type": "run.start", "mode": "run", "user_task": "author a triage machine"})
        + "\n",
        encoding="utf-8",
    )
    hub = model.hub_payload(tmp_path)
    (s,) = hub["drafts"]
    assert s["id"] == "breezy-fern-AB12CD"
    assert s["task"] == "author a triage machine"


def test_hub_and_lookup_skip_husk_run_dirs(tmp_path: Path) -> None:
    # A husk (neither manifest nor logs) is not listed, and must not shadow a
    # real ask of the same id when resolving #/run/<id>.
    (model.runs_root(tmp_path) / "echo-fern-AA11BB").mkdir(parents=True)
    ask = model.asks_root(tmp_path) / "echo-fern-AA11BB"
    ask.mkdir(parents=True)
    (ask / "logs.jsonl").write_text(
        json.dumps({"type": "run.start", "mode": "ask", "user_task": "q"}) + "\n",
        encoding="utf-8",
    )
    hub = model.hub_payload(tmp_path)
    assert [r["mode"] for r in hub["runs"]] == ["ask"]
    assert model.run_dir_for(tmp_path, "echo-fern-AA11BB") == ask


def test_config_suggestions_providers_and_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # models.<role>.provider offers the configured provider names;
    # models.<role>.model the role's provider's model ids via the same
    # cache-first listing the TUI config page and CLI completion use.
    cfg_home = Path(os.environ["AGENT6_CONFIG_HOME"])
    (cfg_home / "config.toml").write_text(
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        '[models.worker]\nprovider = "openrouter"\nmodel = "kimi"\n',
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def _fake_list(provider: str, entry: object, api_key: object) -> list[str]:
        seen["provider"] = provider
        return ["kimi", "qwen3"]

    monkeypatch.setattr(model, "list_models", _fake_list)
    assert model.config_suggestions(tmp_path, "models.worker.provider") == ["openrouter"]
    assert model.config_suggestions(tmp_path, "models.worker.model") == ["kimi", "qwen3"]
    assert seen["provider"] == "openrouter"
    # unknown keys / roles suggest nothing
    assert model.config_suggestions(tmp_path, "web.port") == []
    assert model.config_suggestions(tmp_path, "models.nosuch.model") == []
