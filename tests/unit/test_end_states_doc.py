# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The end-state table in docs/usage.md is rendered from the code's own list of end reasons.

One row per `SessionEndReason`: its headline from `listing.status_word`, its exit code from
`finalize.session_exit_code`. A new reason, a moved label or a moved exit code turns this red
until the page moves with it. Print the table to paste:

uv run python tests/unit/test_end_states_doc.py
"""

from __future__ import annotations

import pathlib
from typing import get_args

from agent6.app import finalize
from agent6.harness import _snapshot
from agent6.viewmodel import format, listing

_USAGE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "usage.md"

# A gate state an end can stand over: its name, then `all_passed`, `gate_red`, `gate_timed_out`
# and the `verified` word the loop writes for it.
_Gate = tuple[str, bool | None, bool, bool, _snapshot.Verification]
_GREEN: _Gate = ("gate green", True, False, False, "passed")
_NO_GATE: _Gate = ("no gate", None, False, False, "not_applicable")
_RED: _Gate = ("gate red", False, True, False, "failed")
_TIMED_OUT: _Gate = ("gate timed out", False, True, True, "failed")
_UNVERIFIED: _Gate = ("unverified", False, False, False, "unverified")
_NOT_GREEN = (_RED, _TIMED_OUT, _UNVERIFIED)
_ANY_GATE = (_GREEN, _NO_GATE, *_NOT_GREEN)

# The ends built `completed=True`, each with the gate states it can stand over. Every other
# end is unfinished: no gate state moves its headline or its exit code.
_COMPLETED: dict[str, tuple[_Gate, ...]] = {
    "finish_session": _ANY_GATE,
    "finish_planning": (_NO_GATE,),
    "answered": (_NO_GATE,),
    "silent_finish": _ANY_GATE,
    "metric_plateau": _ANY_GATE,
    "verify_settled": (_GREEN,),
    "settled": (*_NOT_GREEN, _NO_GATE),
    "interactive_stop": (_UNVERIFIED,),
    "ask_repl_empty": (_NO_GATE,),
    "gate_stale": _NOT_GREEN,
    "gate_red_at_base": (_RED,),
    "lanes_ranked": (_GREEN, _NO_GATE),
}
# Exit codes another owner decides: `_execution` for the operator's undo and detach, the
# fan-out for its own verdict.
_OTHER_EXITS = {"undone": 0, "detached": 0, "no_lane_passed": finalize.EXIT_VERIFY_FAILED}
# Ends that write no session.end, so no surface shows a headline for them.
_NO_HEADLINE = frozenset({"detached", "ask_repl_empty"})


def _row(reason: _snapshot.SessionEndReason) -> str:
    """Return one end reason's table row."""
    gates = _COMPLETED.get(reason, (_UNVERIFIED,))
    labels: list[str] = []
    exits: dict[int, list[str]] = {}
    for name, all_passed, gate_red, gate_timed_out, verified in gates:
        word, detail = listing.status_word(
            finished=True,
            all_passed=all_passed,
            end_reason=reason,
            gate_red=gate_red,
            gate_timed_out=gate_timed_out,
        )
        label = format.status_label(word, detail)
        if label not in labels:
            labels.append(label)
        result = _snapshot.SessionResult(
            completed=reason in _COMPLETED,
            reason=reason,
            summary="",
            iterations=1,
            tool_calls=0,
            verified=verified,
        )
        code = _OTHER_EXITS.get(reason, finalize.session_exit_code(result))
        exits.setdefault(code, []).append(name)
    headline = (
        "none: no end is recorded"
        if reason in _NO_HEADLINE
        else ", ".join(f"`{label}`" for label in labels)
    )
    codes = (
        f"`{next(iter(exits))}`"
        if len(exits) == 1
        else ", ".join(f"`{code}` ({', '.join(names)})" for code, names in exits.items())
    )
    return f"| `{reason}` | {headline} | {codes} |"


def end_states_table() -> str:
    """Return the table docs/usage.md holds, one row per end reason in declaration order."""
    rows = [_row(reason) for reason in get_args(_snapshot.SessionEndReason)]
    return "\n".join(["| End reason | Headline | Exit code |", "|---|---|---|", *rows])


def test_the_end_state_table_matches_the_code() -> None:
    """The usage page's end-state table is the one the code renders.

    A label or an exit code that moves without its row would leave the page naming an end the
    surfaces no longer show.
    """
    assert set(_COMPLETED) | set(_OTHER_EXITS) | _NO_HEADLINE <= set(
        get_args(_snapshot.SessionEndReason)
    )
    assert end_states_table() in _USAGE.read_text(encoding="utf-8"), (
        "docs/usage.md is stale; paste the table printed by:"
        " uv run python tests/unit/test_end_states_doc.py"
    )


def test_only_a_finish_call_over_a_green_gate_is_headlined_passed() -> None:
    """A harness-declared end over a green gate must not read passed."""
    passed = [
        reason
        for reason in get_args(_snapshot.SessionEndReason)
        if "`passed`" in _row(reason).split("|")[2]
    ]
    assert passed == ["finish_session"]


if __name__ == "__main__":
    print(end_states_table())
