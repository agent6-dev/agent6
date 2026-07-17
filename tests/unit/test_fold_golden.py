# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Golden compatibility pin for the logs.jsonl folds.

logs.jsonl is append-only history: every run dir ever written must keep folding
identically. This reads a frozen fixture of real-shaped event bytes (all 19
state-folded families + the transcript-only families + loop.* telemetry +
unknown types + adversarial edge cases, followed by malformed / non-object lines
the tail layer must silently drop) through the exact production read path
(`tail_events`), folds it two ways, and asserts the output byte-for-byte against
committed expectations.

The typed event core (viewmodel.events) reshapes how `apply_event` reads these
bytes; this test is the proof it reshapes nothing an external viewer can see.
Regenerate the expectations only with a deliberate, reviewed behaviour change:

    uv run python tests/unit/test_fold_golden.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from agent6.viewmodel import fold_run, fold_transcript, run_state_as_dict, tail_events

_DATA = Path(__file__).parent / "data"
_FIXTURE = _DATA / "golden_run_logs.jsonl"
_STATE = _DATA / "golden_run_state.json"
_TRANSCRIPT = _DATA / "golden_transcript.json"


def _wire(obj: object) -> object:
    """The JSON wire form a viewer actually receives (tuples become lists), so the
    comparison pins what crosses the boundary, not Python container identity."""
    return json.loads(json.dumps(obj, ensure_ascii=False))


def _folded_state() -> object:
    return _wire(run_state_as_dict(fold_run(tail_events(_FIXTURE, follow=False))))


def _folded_transcript() -> object:
    items = fold_transcript(list(tail_events(_FIXTURE, follow=False)))
    return _wire([dataclasses.asdict(i) for i in items])


def test_tail_drops_malformed_lines_but_keeps_every_object() -> None:
    # 40 JSON objects in the fixture; the 6 trailing malformed / non-object lines
    # are dropped by the tail layer, so the fold never sees them (it cannot crash
    # on bytes it is never handed).
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
