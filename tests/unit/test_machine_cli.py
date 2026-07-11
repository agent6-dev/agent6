# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI tests for Phase 4 machine ergonomics: status, poke, run --exit-on-wait."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.machine import MachineJournal
from agent6.ui.cli import main

WAITER_DELAYED = """
machine = "waiter_delayed"
version = 1
initial = "poll"

[budget]
max_usd = 1.0
max_transitions = 100

[vars.operator]
secs = { type = "int", value = 3600 }

[states.poll]
kind = "wait"
every_secs = "{{ secs }}"
on = { tick = "done", signal = "woken" }

[states.done]
kind = "terminal"
status = "ok"
reason = "ticked"

[states.woken]
kind = "terminal"
status = "ok"
reason = "signalled"
"""


def _write_machine(tmp_path: Path) -> Path:
    f = tmp_path / "waiter.asm.toml"
    f.write_text(WAITER_DELAYED, encoding="utf-8")
    return f


def test_run_exit_on_wait_yields_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    code = main(["machine", "run", str(f), "--exit-on-wait"])
    assert code == 0
    out = capsys.readouterr().out
    assert "WAITING" in out
    # The wait was armed and persisted.
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    pending = MachineJournal(root).read_pending_wait()
    assert pending is not None
    assert pending.state == "poll"


def test_status_reports_waiting_state_and_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()  # drop run output
    code = main(["machine", "status", "waiter_delayed"])
    assert code == 0
    out = capsys.readouterr().out
    assert "waiter_delayed" in out
    assert "next wake:" in out
    assert "spend: $0.0000" in out


def test_status_missing_instance_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["machine", "status", "nope"])
    assert code == 1
    assert "no machine instance" in capsys.readouterr().err


# A no-I/O machine that reaches a terminal immediately (branch -> terminal), so
# `agent6 watch` on it takes the finished path (overview + end) without blocking
# in the follow loop and without needing a model or the jail.
TINY = """
machine = "tiny"
version = 1
initial = "route"

[budget]
max_transitions = 10

[vars.code]
n = { type = "int", default = 0 }

[states.route]
kind = "branch"
when = [
  { if = "n == 0", goto = "done" },
  { else = true, goto = "done" },
]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""


def test_watch_finished_instance_shows_overview_and_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    assert main(["machine", "run", str(f)]) == 0
    capsys.readouterr()  # drop run output
    # A finished instance has a journaled MachineEnd, so the unified `agent6 watch`
    # (which routes a machine name to the machine follower) prints the overview +
    # the final state and returns instead of entering the (blocking) follow loop.
    code = main(["watch", "tiny"])
    assert code == 0
    out = capsys.readouterr().out
    assert "machine: tiny" in out
    assert "> done" in out  # current state marked
    assert ". route" in out  # a visited state marked
    assert "OK: ended in 'done'" in out


def test_poke_drops_signal_for_waiting_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    code = main(["machine", "poke", "waiter_delayed"])
    assert code == 0
    assert "poked" in capsys.readouterr().out
    # The signal is now pending for the next take_signal().
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    assert MachineJournal(root).take_signal() == (True, None)


def test_poke_carries_data_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    assert main(["machine", "poke", "waiter_delayed", "--data", '{"cmd": "go"}']) == 0
    root = resolved_state_dir(tmp_path) / "machines" / "waiter_delayed"
    assert MachineJournal(root).take_signal() == (True, {"cmd": "go"})
    # --message wraps a plain string.
    assert main(["machine", "poke", "waiter_delayed", "--message", "hello"]) == 0
    assert MachineJournal(root).take_signal() == (True, "hello")


def test_poke_rejects_invalid_json_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    f = _write_machine(tmp_path)
    assert main(["machine", "run", str(f), "--exit-on-wait"]) == 0
    capsys.readouterr()
    assert main(["machine", "poke", "waiter_delayed", "--data", "{not json}"]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_poke_missing_instance_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["machine", "poke", "nope"])
    assert code == 1
    assert "no machine instance" in capsys.readouterr().err


def test_tail_state_log_tolerates_torn_utf8_tail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `machine watch` polls the per-state log; a tail torn mid multibyte UTF-8
    # sequence must be held back (offset at the partial start), not raise, and
    # print once completed.
    import json

    from agent6.ui.cli.machine_cmds import _tail_state_log  # pyright: ignore[reportPrivateUsage]

    log = tmp_path / "logs.jsonl"
    full = json.dumps({"type": "loop.note", "text": "café"}, ensure_ascii=False).encode()
    cut = full.rindex(b"\xc3\xa9") + 1  # keep only the first byte of the é
    head = json.dumps({"type": "run.start", "ts": 1000.0}).encode() + b"\n"
    log.write_bytes(head + full[:cut])
    offset, anchor = _tail_state_log(log, 0, None)
    assert offset == len(head)  # complete line consumed; torn tail held back
    assert anchor == 1000.0
    assert "run.start" in capsys.readouterr().out
    with log.open("ab") as fh:
        fh.write(full[cut:] + b"\n")
    offset, _ = _tail_state_log(log, offset, anchor)
    assert offset == len(head) + len(full) + 1
    assert "café" in capsys.readouterr().out


AGENT_MACHINE_HARD = """
machine = "hard-usd"
version = 1
initial = "judge"

[budget]
max_usd = 1.0
max_transitions = 10

[schemas.r]
ok = "bool"

[vars.agent]
out = { type = "r", default = {} }

[states.judge]
kind = "agent"
prompt = "judge"
output_schema = "r"
capture = { finish_json = "out" }
timeout_secs = 60
on = { ok = "done", failed = "done", budget_exhausted = "done", timeout = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "done"
"""


def _hard_usd_cfg(model: str):
    from agent6.config import Config

    return Config.model_validate(
        {
            "providers": {"p": {"api_format": "openai", "base_url": "http://localhost:1"}},
            "models": {"worker": {"provider": "p", "model": model}},
        }
    )


def _load_spec(tmp_path: Path, body: str):
    from agent6.machine import load_machine

    f = tmp_path / "m.asm.toml"
    f.write_text(body, encoding="utf-8")
    return load_machine(f)


def test_hard_usd_preflight_refuses_unpriced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [budget].max_usd is a hard promise; an unpriced inherited worker model
    # cannot honor it, so machine run must refuse up front.
    from agent6.ui.cli.machine_cmds import (
        _hard_usd_preflight_error,  # pyright: ignore[reportPrivateUsage]
    )

    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "empty-cache"))
    spec = _load_spec(tmp_path, AGENT_MACHINE_HARD)
    err = _hard_usd_preflight_error(spec, _hard_usd_cfg("nobody/unpriced"))
    assert err is not None
    assert "max_usd" in err and "nobody/unpriced" in err and "judge" in err


def test_hard_usd_preflight_passes_best_effort_and_priced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json as _json

    from agent6.ui.cli.machine_cmds import (
        _hard_usd_preflight_error,  # pyright: ignore[reportPrivateUsage]
    )

    # best_effort_usd_limit never refuses, priced or not.
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "empty-cache"))
    soft = AGENT_MACHINE_HARD.replace("max_usd = 1.0", "best_effort_usd_limit = 1.0")
    spec = _load_spec(tmp_path, soft)
    assert _hard_usd_preflight_error(spec, _hard_usd_cfg("nobody/unpriced")) is None

    # max_usd passes once the model has price data.
    cache = tmp_path / "cache"
    (cache / "models").mkdir(parents=True)
    (cache / "models" / "p.json").write_text(
        _json.dumps({"models": ["priced/m"], "pricing": {"priced/m": [1.0, 2.0]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(cache))
    spec_hard = _load_spec(tmp_path, AGENT_MACHINE_HARD)
    assert _hard_usd_preflight_error(spec_hard, _hard_usd_cfg("priced/m")) is None


def test_hard_usd_preflight_checks_per_state_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A soft machine budget with a hard per-state max_usd still requires price
    # data for that state's model.
    from agent6.ui.cli.machine_cmds import (
        _hard_usd_preflight_error,  # pyright: ignore[reportPrivateUsage]
    )

    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "empty-cache"))
    body = AGENT_MACHINE_HARD.replace("max_usd = 1.0", "best_effort_usd_limit = 1.0").replace(
        'kind = "agent"', 'kind = "agent"\nmax_usd = 0.5\nmodel = "pinned/unpriced"', 1
    )
    spec = _load_spec(tmp_path, body)
    err = _hard_usd_preflight_error(spec, _hard_usd_cfg("priced/m"))
    assert err is not None and "pinned/unpriced" in err
