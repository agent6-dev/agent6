# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""finish_session result coercion: a stringified JSON object still lands as the
structured finish payload (weak models routinely stringify it; a machine
agent state's whole cycle used to fail on shape)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent6.config import Config
from agent6.harness._chain import RunChain
from agent6.harness._conversation import AssistantTurn
from agent6.harness._finish_gates import FinishCall
from agent6.harness.loop import (
    Harness,
    TurnState,
)


def _wf(**kw: Any) -> Harness:
    kw.setdefault("state_dir", Path("/tmp/state"))
    return Harness(
        chain=RunChain(Path("/tmp")),
        config=Config.model_validate({}),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _m: None,
        **kw,
    )


def _capture(tool_input: dict[str, Any]) -> FinishCall:
    wf = _wf()
    turn = TurnState(iteration=1, resp=MagicMock(), assistant=AssistantTurn((), ()))
    wf._capture_finish(turn, "finish_session", tool_input)  # pyright: ignore[reportPrivateUsage]
    assert turn.finish is not None
    return turn.finish


def test_finish_result_object_passes_through() -> None:
    finish = _capture({"summary": "s", "result": {"found": True}})
    assert finish.payload == {"found": True}


def test_finish_result_stringified_object_is_coerced() -> None:
    finish = _capture({"summary": "s", "result": '{"found": true, "file": "a.py"}'})
    assert finish.payload == {"found": True, "file": "a.py"}


def test_finish_result_garbage_string_stays_none() -> None:
    finish = _capture({"summary": "s", "result": "not json"})
    assert finish.payload is None


def test_finish_result_stringified_non_object_stays_none() -> None:
    finish = _capture({"summary": "s", "result": '["a", "b"]'})
    assert finish.payload is None
