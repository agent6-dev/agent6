# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Byte pin for the machine-agent subprocess IPC.

`MachineAgentRequest` owns `request.json`, `AgentExecResult` owns `result.json`, and the argv
contract is frozen; the files are transient per invocation, so the pin is a same-version one.
"""

from __future__ import annotations

import pathlib

from agent6 import git_ops
from agent6.app import machine_agent
from agent6.machine import AgentExecResult, AgentRequest, schema

_REQUEST = machine_agent.MachineAgentRequest(
    cwd=pathlib.Path("/work/repo"),
    root=pathlib.Path("/work/repo"),
    overlay={"budget": {"max_tokens_fallback": 9000}},
    isolation="strict",
    transcript_dir=pathlib.Path("/state/machines/m/i/transcripts"),
    events_log=pathlib.Path("/state/machines/m/i/states/0002-review/logs.jsonl"),
    protect_paths=(pathlib.Path("/work/repo/m.asm.toml"),),
    commit_identity=git_ops.CommitIdentity(name="Machine Bot", email="bot@example.com"),
    request=AgentRequest(
        prompt="review the queue",
        timeout_s=600.0,
        model="claude-x",
        provider="anthropic",
        effort="low",
        temperature=0.2,
        max_usd=1.5,
        max_tokens_fallback=200000,
        mode="run",
        state_name="review",
        step_seq=2,
        output_schema="verdict",
        schemas={
            "verdict": {
                "ok": schema.FieldSpec(type="bool"),
                "note": schema.FieldSpec(type="str", optional=True),
            }
        },
    ),
)

_REQUEST_BYTES = (
    '{"cwd":"/work/repo","root":"/work/repo",'
    '"overlay":{"budget":{"max_tokens_fallback":9000}},"isolation":"strict",'
    '"transcript_dir":"/state/machines/m/i/transcripts",'
    '"events_log":"/state/machines/m/i/states/0002-review/logs.jsonl",'
    '"protect_paths":["/work/repo/m.asm.toml"],'
    '"commit_identity":{"name":"Machine Bot","email":"bot@example.com","trailer":null},'
    '"request":{"prompt":"review the queue","timeout_s":600.0,"model":"claude-x",'
    '"provider":"anthropic","effort":"low","temperature":0.2,"max_usd":1.5,'
    '"max_tokens_fallback":200000,"mode":"run",'
    '"state_name":"review","step_seq":2,"output_schema":"verdict",'
    '"schemas":{"verdict":{"ok":{"type":"bool","optional":false,"enum":null},'
    '"note":{"type":"str","optional":true,"enum":null}}}}}'
)

_RESULT = AgentExecResult(
    reason="finish_session",
    payload={"ok": True, "notes": "queue drained"},
    usd=0.0588752,
    input_tokens=66084,
    output_tokens=838,
)

_RESULT_BYTES = (
    '{"reason":"finish_session","payload":{"ok":true,"notes":"queue drained"},'
    '"usd":0.0588752,"usd_partial":false,"input_tokens":66084,"output_tokens":838}'
)


def test_request_serializes_to_pinned_bytes() -> None:
    assert _REQUEST.model_dump_json() == _REQUEST_BYTES


def test_request_bytes_validate_to_same_object() -> None:
    assert machine_agent.MachineAgentRequest.model_validate_json(_REQUEST_BYTES) == _REQUEST


def test_result_serializes_to_pinned_bytes() -> None:
    assert _RESULT.model_dump_json() == _RESULT_BYTES


def test_result_bytes_validate_to_same_object() -> None:
    assert AgentExecResult.model_validate_json(_RESULT_BYTES) == _RESULT


def test_result_payload_with_a_lone_surrogate_still_serializes() -> None:
    """A model's finish_session arguments reach the payload through json.loads, surrogates too.

    An unscrubbed payload killed the subprocess before its result write, and the host routed a
    successful state to on.failed.
    """
    import json

    res = AgentExecResult(reason="finish_session", payload={"note": "bad \ud800 tail"})
    text = res.model_dump_json()
    assert json.loads(text)["payload"]["note"].startswith("bad ")
    assert "\ud800" not in json.loads(text)["payload"]["note"]


def test_defaulted_request_omits_nothing() -> None:
    # Optional envelope fields serialize explicitly, never key-drop, so the reader needs no default.
    minimal = machine_agent.MachineAgentRequest(
        cwd=pathlib.Path("/w"),
        root=pathlib.Path("/w"),
        overlay={},
        isolation="none",
        transcript_dir=pathlib.Path("/t"),
        request=AgentRequest(prompt="p", timeout_s=1.0),
    )
    assert minimal.model_dump_json() == (
        '{"cwd":"/w","root":"/w","overlay":{},"isolation":"none","transcript_dir":"/t",'
        '"events_log":null,"protect_paths":[],"commit_identity":null,'
        '"request":{"prompt":"p","timeout_s":1.0,"model":null,"provider":null,'
        '"effort":null,"temperature":null,"max_usd":null,'
        '"max_tokens_fallback":null,"mode":"agent","state_name":"","step_seq":0,'
        '"output_schema":null,"schemas":{}}}'
    )
