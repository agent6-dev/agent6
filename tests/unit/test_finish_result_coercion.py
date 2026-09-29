# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""finish_session result coercion: a stringified JSON object lands as the structured payload."""

from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

from agent6.config import Config
from agent6.harness import _chain, _conversation, _finish_gates, _loop_state, loop


def _wf(**kw: Any) -> loop.Harness:
    kw.setdefault("state_dir", pathlib.Path("/tmp/state"))
    return loop.Harness(
        chain=_chain.RunChain(pathlib.Path("/tmp")),
        config=Config.model_validate({}),
        provider=mock.MagicMock(),
        dispatcher=mock.MagicMock(),
        logger=lambda _m: None,
        **kw,
    )


def _capture(tool_input: dict[str, Any]) -> _finish_gates.FinishCall:
    wf = _wf()
    turn = _loop_state.TurnState(
        iteration=1, resp=mock.MagicMock(), assistant=_conversation.AssistantTurn((), ())
    )
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
