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
    assert s["usd_partial"] is False
    assert s["label"] == "passed"  # the one shared human label, rendered verbatim


def test_run_summary_carries_the_partial_cost_marker(tmp_path: Path) -> None:
    # The hub row renders the same lower-bound marker as the run page.
    _run(
        tmp_path,
        "r1p",
        [
            {"type": "run.start", "mode": "run", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.0123, "usd_partial": True},
            {"type": "run.end", "all_passed": True},
        ],
    )
    (s,) = model.hub_payload(tmp_path)["runs"]
    assert s["usd_partial"] is True


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


def test_run_snapshot_embeds_the_compare_outcome(tmp_path: Path) -> None:
    # A fan-out lane's manifest carries the compare block; the run snapshot
    # embeds it so the page header can render rank/winner/rationale.
    d = _run(tmp_path, "lane1", [{"type": "run.start", "user_task": "x"}])
    (d / "manifest.json").write_text(
        json.dumps(
            {"compare": {"group": "fan", "rank": 1, "of": 2, "winner": True,
                         "ranked_by": "judge", "rationale": "cleanest diff"}}
        ),
        encoding="utf-8",
    )  # fmt: skip
    snap = model.run_snapshot(d)
    assert snap["compare"]["winner"] is True and snap["compare"]["rank"] == 1
    assert snap["compare"]["rationale"] == "cleanest diff"
    # A run with no compare block carries no `compare` key (non-lane runs).
    plain = _run(tmp_path, "plain", [{"type": "run.start", "user_task": "y"}])
    assert "compare" not in model.run_snapshot(plain)


def test_run_snapshot_falls_back_to_the_manifest_task(tmp_path: Path) -> None:
    """The fold sets user_task only from run.start, so a parked/created/forked
    run's page said "task: (none)" while the hub row (summarize_run_dir's
    manifest fallback) showed the real work. The snapshot now carries the
    manifest's task as fallback_task, on both the one-shot and SSE paths."""
    d = model.runs_root(tmp_path) / "parked1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"mode": "run", "user_task": "queued work", "parked_task": "queued work"}),
        encoding="utf-8",
    )
    snap = model.run_snapshot(d)
    assert snap.get("user_task", "") == ""  # the fold has no run.start
    assert snap["fallback_task"] == "queued work"


def test_hub_marks_the_fan_out_winner(tmp_path: Path) -> None:
    d = _run(tmp_path, "lane-win", [{"type": "run.start", "mode": "run", "user_task": "t"}])
    (d / "manifest.json").write_text(
        json.dumps({"compare": {"rank": 1, "of": 2, "winner": True}}), encoding="utf-8"
    )
    (s,) = model.hub_payload(tmp_path)["runs"]
    assert s["winner"] is True


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


TINY_MACHINE = """
machine = "tiny"
version = 1
initial = "route"

[budget]
max_transitions = 10

[states.route]
kind = "branch"
when = [{ else = true, goto = "done" }]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""


def test_machine_snapshot_carries_the_dir_status_word(tmp_path: Path) -> None:
    """The machine wire payload stamps `status` (machine_word_for_dir), so a
    client can gate Steer and the prompt boxes on liveness -- with only
    `ended` it cannot tell a parked machine from a running one."""
    md = model.machines_root(tmp_path) / "m3"
    md.mkdir(parents=True)
    (md / "machine.asm.toml").write_text(TINY_MACHINE, encoding="utf-8")
    (md / "journal.jsonl").write_text("", encoding="utf-8")
    assert model.machine_snapshot(md)["status"] == "stopped"  # no wait, no worker
    (md / "wait.json").write_text('{"state":"route","wake_epoch":9999999999.0}\n', encoding="utf-8")
    assert model.machine_snapshot(md)["status"] == "waiting"  # parked


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


def test_config_suggestions_parallel_models_pseudo_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The /parallel composer autocomplete: the worker's configured model plus the
    # worker provider's cached listing, cache-only so it never blocks.
    cfg_home = Path(os.environ["AGENT6_CONFIG_HOME"])
    (cfg_home / "config.toml").write_text(
        '[providers.openrouter]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        '[models.worker]\nprovider = "openrouter"\nmodel = "role-only-model"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    cache = tmp_path / "cache" / "models"
    cache.mkdir(parents=True)
    (cache / "openrouter.json").write_text(
        json.dumps({"models": ["moonshotai/kimi-k2.6", "z-ai/glm-4.6"]}), encoding="utf-8"
    )
    out = model.config_suggestions(tmp_path, "parallel.models")
    assert out == ["moonshotai/kimi-k2.6", "role-only-model", "z-ai/glm-4.6"]


def test_parallel_models_suggestions_scoped_to_worker_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Lanes inherit the WORKER provider (only the model is overridden per lane),
    # so the suggestions offer only models the lanes can actually run: a sibling
    # provider's cached catalog is excluded.
    cfg_home = Path(os.environ["AGENT6_CONFIG_HOME"])
    (cfg_home / "config.toml").write_text(
        '[providers.w]\napi_format = "openai"\nbase_url = "https://w.example/v1"\n'
        '[providers.s]\napi_format = "openai"\nbase_url = "https://s.example/v1"\n'
        '[models.worker]\nprovider = "w"\nmodel = "w/base-model"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    cache = tmp_path / "cache" / "models"
    cache.mkdir(parents=True)
    (cache / "w.json").write_text(json.dumps({"models": ["w/model-a"]}), encoding="utf-8")
    (cache / "s.json").write_text(json.dumps({"models": ["s/only-model"]}), encoding="utf-8")
    assert model.config_suggestions(tmp_path, "parallel.models") == ["w/base-model", "w/model-a"]


def test_run_snapshot_labels_a_parked_submission(tmp_path: Path) -> None:
    """A parked run (the busy-checkout refusal saved the task) has no events by
    construction, so the event fold alone reads it as "running" while the hub row
    says parked. The run page must not disagree with the hub about the same run."""
    d = model.runs_root(tmp_path) / "parked1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"run_id": "parked1", "parked_task": "do the thing"}), encoding="utf-8"
    )
    assert model.run_snapshot(d)["status_label"] == "parked · resume to start"
    (hub_row,) = model.hub_payload(tmp_path)["runs"]
    assert hub_row["status"] == "parked"  # the two surfaces lead with one word


def test_run_snapshot_labels_a_dead_worker_stale(tmp_path: Path) -> None:
    """A run whose recorded worker is gone and that never logged run.end folds to
    "running". The hub calls it stale off the same pid probe; the one-shot payload
    the page first paints from has to say so too, not only the SSE frame."""
    d = _run(tmp_path, "crashed1", [{"type": "run.start", "mode": "run", "user_task": "t"}])
    (d / "worker.pid").write_text("999999 12345678", encoding="utf-8")  # dead pid
    assert model.run_snapshot(d)["status_label"] == "stale"


def test_run_snapshot_labels_waiting_starting_created(tmp_path: Path) -> None:
    """The run page speaks EVERY listing word, not just parked/stale: blocked
    on an operator answer reads "waiting · needs answer" (it read "running"
    and sent the operator off to wait on the model while the run waited on
    THEM), a live pre-run.start worker "starting", a never-started dir
    "created"."""
    import os

    from agent6.runs.ipc import write_worker_pid

    d = _run(
        tmp_path,
        "blocked1",
        [
            {"type": "run.start", "mode": "run", "user_task": "t"},
            {"type": "approval.prompt", "id": "approval-1", "prompt": "rm -rf?"},
        ],
    )
    write_worker_pid(d, os.getpid())
    assert model.run_snapshot(d)["status_label"] == "waiting · needs answer"

    e = model.runs_root(tmp_path) / "fresh1"
    e.mkdir(parents=True)
    (e / "manifest.json").write_text(json.dumps({"run_id": "fresh1"}), encoding="utf-8")
    write_worker_pid(e, os.getpid())
    assert model.run_snapshot(e)["status_label"] == "starting"
    (e / "worker.pid").unlink()
    assert model.run_snapshot(e)["status_label"] == "created"


def test_run_snapshot_leaves_a_finished_run_alone(tmp_path: Path) -> None:
    """The dir-derived relabels never touch a run that ended on its own terms."""
    d = _run(
        tmp_path,
        "done1",
        [
            {"type": "run.start", "mode": "run", "user_task": "t"},
            {"type": "run.end", "reason": "finish_run", "iterations": 1, "all_passed": True},
        ],
    )
    (d / "worker.pid").write_text("999999 12345678", encoding="utf-8")
    assert model.run_snapshot(d)["status_label"] == "passed"


def test_run_snapshot_marks_a_parked_run_not_live(tmp_path: Path) -> None:
    """The page keys its composer and Stop/Compact buttons on liveness. The fold
    calls every unfinished run "running", so a parked run offered a steer
    composer and a Stop button that both dead-ended while resume -- the one
    action that works -- was unreachable. `live` is the dir-aware answer."""
    import os

    from agent6.runs.ipc import write_worker_pid

    parked = model.runs_root(tmp_path) / "parked2"
    parked.mkdir(parents=True)
    (parked / "manifest.json").write_text(
        json.dumps({"run_id": "parked2", "parked_task": "do the thing"}), encoding="utf-8"
    )
    assert model.run_snapshot(parked)["live"] is False

    crashed = _run(tmp_path, "crashed2", [{"type": "run.start", "mode": "run", "user_task": "t"}])
    (crashed / "worker.pid").write_text("999999999", encoding="utf-8")
    assert model.run_snapshot(crashed)["live"] is False

    alive = _run(tmp_path, "alive2", [{"type": "run.start", "mode": "run", "user_task": "t"}])
    write_worker_pid(alive, os.getpid())
    assert model.run_snapshot(alive)["live"] is True

    done = _run(
        tmp_path,
        "done2",
        [
            {"type": "run.start", "mode": "run", "user_task": "t"},
            {"type": "run.end", "reason": "finish_run", "iterations": 1, "all_passed": True},
        ],
    )
    assert model.run_snapshot(done)["live"] is False


def test_the_states_that_offer_resume_are_not_live(tmp_path: Path) -> None:
    """The composer's resume-takeover poll waits for the run to come alive. It
    polled `finished === false`, which is ALREADY true for the parked and stale
    runs it routes into resume mode, so takeover was declared on the first poll
    and a resume that died on spawn was reported as success -- the spawn's
    stderr goes to DEVNULL, so nothing else could surface it. `live` is what
    separates the two, and both these states must read false."""
    parked = model.runs_root(tmp_path) / "parked-live"
    parked.mkdir(parents=True)
    (parked / "manifest.json").write_text(
        json.dumps({"run_id": "parked-live", "parked_task": "t"}), encoding="utf-8"
    )
    stale = _run(tmp_path, "stale-live", [{"type": "run.start", "mode": "run", "user_task": "t"}])
    (stale / "worker.pid").write_text("999999 12345678", encoding="utf-8")  # dead pid

    for d in (parked, stale):
        snap = model.run_snapshot(d)
        assert snap["finished"] is False, d.name  # why the old poll fired at once
        assert snap["live"] is False, d.name  # ...and what actually distinguishes them
