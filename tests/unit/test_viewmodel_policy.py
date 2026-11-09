# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One fold supplies the run's policy facts to every surface."""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import evidence, kinds
from agent6.viewmodel import session_log, session_policy


def _manifest(session_dir: pathlib.Path, **over: object) -> None:
    data: dict[str, object] = {
        "version": 3,
        "mode": "run",
        "session_id": "r",
        "models": {"driver": {"provider": "anthropic", "model": "claude-x"}},
        "policy": {"run_commands": "ask", "isolation": "full", "network": "session"},
        "harness": {"gate": {"command": ["uv", "run", "pytest"], "source": "configured"}},
    }
    data.update(over)
    (session_dir / "manifest.json").write_text(json.dumps(data), encoding="utf-8")


def test_the_line_carries_what_an_operator_needs(tmp_path: pathlib.Path) -> None:
    """Model, sandbox, command setting and the gate, without an interrupt or a config read."""
    _manifest(tmp_path)
    line = session_policy(tmp_path).line()
    assert line == (
        "claude-x · isolation full · network session · run_commands ask"
        " · gate uv run pytest (configured)"
    )


def test_the_gate_says_whose_it_is(tmp_path: pathlib.Path) -> None:
    """An inferred gate came from a file the model can edit; a configured one did not.

    A surface that hides the difference hides the only thing that makes "passed" mean something.
    """
    _manifest(tmp_path, harness={"gate": {"command": ["make", "test"], "source": "inferred"}})
    assert session_policy(tmp_path).gate() == "gate make test (inferred)"


def test_a_gateless_run_says_so(tmp_path: pathlib.Path) -> None:
    _manifest(tmp_path, harness={"gate": {}})
    assert session_policy(tmp_path).gate() == "no verify gate"


def test_a_run_whose_gate_is_not_resolved_yet_claims_none(tmp_path: pathlib.Path) -> None:
    """A run parked before its execution may still infer a gate; "no verify gate" is a claim."""
    _manifest(tmp_path, harness={"gate": {"availability": "pending"}})
    assert session_policy(tmp_path).gate() == "verify gate not settled yet"


def test_a_withheld_gate_is_named_not_denied(tmp_path: pathlib.Path) -> None:
    """A run that may not run commands still has a gate.

    "no verify gate" is what a repository with none gets; a withheld one names its command.
    """
    _manifest(
        tmp_path,
        harness={
            "gate": {
                "command": ["python3", "-m", "pytest", "-q"],
                "source": "configured",
                "availability": "withheld",
            }
        },
    )
    assert session_policy(tmp_path).gate() == "gate withheld: python3 -m pytest -q"


def test_an_unreadable_dir_reports_nothing_rather_than_guessing(tmp_path: pathlib.Path) -> None:
    assert session_policy(tmp_path).line() == ""


def test_an_asks_line_names_no_gate(tmp_path: pathlib.Path) -> None:
    """An ask's policy line carries no gate; a configured gate still rides its manifest."""
    _manifest(tmp_path, mode="ask")
    assert (
        session_policy(tmp_path).line()
        == "claude-x · isolation full · network session · run_commands ask"
    )


def test_a_configured_word_is_not_a_network(tmp_path: pathlib.Path) -> None:
    """A manifest an older agent6 wrote holds `auto`; the header names no network it would guess."""
    _manifest(tmp_path, policy={"run_commands": "ask", "isolation": "full", "network": "auto"})
    assert session_policy(tmp_path).short() == "isolation full · run_commands ask"
    _manifest(tmp_path, policy={"run_commands": "ask", "isolation": "full"})
    assert session_policy(tmp_path).short() == "isolation full · run_commands ask"


def _target_event(target_id: str, execution_id: str, home: str = "/home/op") -> dict[str, object]:
    stamp = evidence.ExecutionTargetEvidence(
        target_id=target_id,
        execution_id=execution_id,
        target=kinds.ExecutionTargetStamp(
            transport="local",
            host_root=pathlib.Path("/work"),
            command_root=pathlib.Path("/work"),
            user="1000",
            home=home,
            environment=kinds.ExecutionEnvironmentPolicy(inherit=False, variable_names=()),
        ),
    )
    return {
        "type": "execution.target",
        "ts": 1.0,
        **evidence.TARGET.dump_python(stamp, mode="json"),
    }


def _journal(session_dir: pathlib.Path, *rows: dict[str, object]) -> None:
    with (session_dir / "logs.jsonl").open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_the_environment_comes_from_the_journal_tail_not_a_replay(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The current target is read backward from the tail; a replay of the journal costs seconds."""
    _manifest(tmp_path)
    filler: dict[str, object] = {"type": "role.text_delta", "ts": 1.0, "text": "x" * 64}
    _journal(tmp_path, *([filler] * 2000))
    _journal(tmp_path, _target_event("t1", "e1"), {"type": "session.start", "execution_id": "e1"})

    def no_replay(logs: pathlib.Path) -> session_log.LogScan:
        raise AssertionError("the policy line replayed the whole journal")

    monkeypatch.setattr(session_log, "scan_session_log", no_replay)
    assert session_policy(tmp_path).environment
    assert "Environment: local" in session_policy(tmp_path).line()
    _journal(tmp_path, {"type": "session.stop", "execution_id": "e1", "reason": "finished"})
    assert session_policy(tmp_path).environment
    _journal(tmp_path, {"type": "session.resume", "execution_id": "e2"})
    assert session_policy(tmp_path).environment == ()


@pytest.mark.parametrize("damage", ["conflicting", "unbound", "malformed"])
def test_the_tail_read_agrees_with_the_full_fold(tmp_path: pathlib.Path, damage: str) -> None:
    """A damaged later record of the same target clears the environment in both readers."""
    _manifest(tmp_path)
    good = _target_event("t1", "e1")
    _journal(tmp_path, good, {"type": "session.start", "execution_id": "e1"})
    later = _target_event("t1", "e1", home="/elsewhere" if damage == "conflicting" else "/home/op")
    if damage == "unbound":
        later.pop("execution_id")
    if damage == "malformed":
        later["target"] = None
    _journal(tmp_path, later)
    logs = tmp_path / "logs.jsonl"
    assert session_log.current_target(logs) is None
    assert session_log.scan_session_log(logs).current_target is None
    assert session_policy(tmp_path).environment == ()
