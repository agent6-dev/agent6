# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Golden compatibility pin for the logs.jsonl folds.

logs.jsonl is append-only history: every run dir ever written must keep folding identically. A
frozen fixture of real-shaped event bytes (every state-folded family, the transcript-only families,
loop telemetry, unknown types, adversarial edge cases, then malformed lines the tail layer drops)
goes through the production read path and folds two ways, byte-for-byte against committed
expectations.

Regenerate the expectations only with a deliberate, reviewed behaviour change:

uv run python tests/unit/test_fold_golden.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from agent6.viewmodel import fold_session, fold_transcript, session_state_as_dict, tail_events

_DATA = Path(__file__).parent / "data"
_FIXTURE = _DATA / "golden_session_logs.jsonl"
_STATE = _DATA / "golden_session_state.json"
_TRANSCRIPT = _DATA / "golden_transcript.json"


def _wire(obj: object) -> object:
    """The JSON wire form a viewer receives (tuples become lists): the pin is what crosses."""
    return json.loads(json.dumps(obj, ensure_ascii=False))


def _folded_state() -> object:
    return _wire(session_state_as_dict(fold_session(tail_events(_FIXTURE, follow=False))))


def _folded_transcript() -> object:
    items = fold_transcript(list(tail_events(_FIXTURE, follow=False)))
    return _wire([dataclasses.asdict(i) for i in items])


def test_tail_drops_malformed_lines_but_keeps_every_object() -> None:
    # 40 JSON objects; the 6 trailing malformed lines are dropped by the tail layer before the fold.
    events = list(tail_events(_FIXTURE, follow=False))
    assert len(events) == 40
    assert all(isinstance(e, dict) for e in events)


def test_run_state_fold_is_byte_identical_to_golden() -> None:
    assert _folded_state() == json.loads(_STATE.read_text(encoding="utf-8"))


def test_transcript_fold_is_byte_identical_to_golden() -> None:
    assert _folded_transcript() == json.loads(_TRANSCRIPT.read_text(encoding="utf-8"))


def _regenerate() -> None:
    """Rewrite the committed expectations from the current code. Manual, guarded."""
    _STATE.write_text(
        json.dumps(_folded_state(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _TRANSCRIPT.write_text(
        json.dumps(_folded_transcript(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    _regenerate()
    print("regenerated golden expectations")
