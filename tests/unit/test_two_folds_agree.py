# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The two folds over one session log agree on the facts they share.

`viewmodel.session_log.scan_session_log` folds `logs.jsonl` for the listings and
`viewmodel.state.fold_session` folds the same log for the viewers; the lifecycle and the
usage fold once, in `viewmodel.session_core`, and each keeps its own projection. This folds
every fixture both ways and pins the two projections against the core they share.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from agent6.viewmodel import session_log, status_facts
from agent6.viewmodel import state as viewmodel_state

_GOLDEN = pathlib.Path(__file__).parent / "data" / "golden_session_logs.jsonl"

_START: dict[str, Any] = {
    "ts": "2026-07-14T10:00:00+00:00",
    "type": "session.start",
    "session_id": "two-folds-AAAA11",
    "user_task": "fix the clock",
    "mode": "run",
}
_END_PASSED: dict[str, Any] = {
    "ts": "2026-07-14T10:01:00+00:00",
    "type": "session.stop",
    "reason": "finish_session",
    "verification": "passed",
}
_SHAPES: dict[str, list[dict[str, Any]]] = {
    "started": [_START],
    "passed": [_START, _END_PASSED],
    "failed": [
        _START,
        {
            "ts": "2026-07-14T10:00:30+00:00",
            "type": "budget.update",
            "input_total": 18,
            "output_total": 2194,
            "cache_read_total": 42486,
            "cache_creation_total": 22617,
            "usd_total": 0.25,
        },
        {
            "ts": "2026-07-14T10:01:00+00:00",
            "type": "session.stop",
            "reason": "provider_error",
            "verification": "failed",
        },
    ],
    "usd_then_tokens_only": [
        _START,
        {"ts": "2026-07-14T10:00:30+00:00", "type": "budget.update", "usd_total": 0.25},
        {
            "ts": "2026-07-14T10:00:40+00:00",
            "type": "budget.update",
            "input_total": 18,
            "output_total": 2194,
        },
        _END_PASSED,
    ],
    "plan_then_usd_only": [
        _START,
        {
            "ts": "2026-07-14T10:00:30+00:00",
            "type": "budget.update",
            "usd_total": 0.0,
            "plan_observed_points": 2.5,
            "plan_cap": 6.0,
        },
        {"ts": "2026-07-14T10:00:40+00:00", "type": "budget.update", "usd_total": 0.1},
        _END_PASSED,
    ],
    "plan_capped": [
        _START,
        {
            "ts": "2026-07-14T10:00:30+00:00",
            "type": "budget.update",
            "input_total": 18,
            "output_total": 2194,
            "usd_total": 0.0,
            "plan_used_percent": 13.0,
            "plan_observed_points": 2.5,
            "plan_cap": 6.0,
        },
        {
            "ts": "2026-07-14T10:01:00+00:00",
            "type": "session.stop",
            "reason": "budget_exhausted",
            "verification": "failed",
        },
    ],
    "ungated": [_START, {**_END_PASSED, "verification": "not_applicable"}],
    "container": [
        _START,
        {"ts": "2026-07-14T10:00:10+00:00", "type": "environment.starting", "command": ["up"]},
        {
            "ts": "2026-07-14T10:00:40+00:00",
            "type": "environment.acquired",
            "lifecycle_ref": "k",
            "role": "owner",
            "reason": "started by this run; owned",
            "started": True,
            "start_s": 30.0,
        },
        {
            "ts": "2026-07-14T10:01:00+00:00",
            "type": "execution.environment",
            "outcome": "stopped",
            "reason": "owned; stopped",
        },
        _END_PASSED,
    ],
    "resumed": [
        _START,
        {
            "ts": "2026-07-14T10:00:30+00:00",
            "type": "budget.update",
            "usd_total": 0.25,
            "plan_observed_points": 2.5,
            "plan_cap": 6.0,
        },
        _END_PASSED,
        {"ts": "2026-07-14T10:02:00+00:00", "type": "session.resume", "turn": 3},
        {
            "ts": "2026-07-14T10:02:30+00:00",
            "type": "budget.update",
            "usd_total": 0.10,
            "usd_partial": True,
        },
        {**_END_PASSED, "ts": "2026-07-14T10:03:00+00:00", "reason": "steer_abort"},
    ],
    "resumed_after_red": [
        _START,
        {
            "ts": "2026-07-14T10:00:30+00:00",
            "type": "verify.end",
            "cmd": ["pytest"],
            "exit_code": 1,
        },
        {**_END_PASSED, "verification": "failed"},
        {"ts": "2026-07-14T10:02:00+00:00", "type": "session.resume", "turn": 3},
        {**_END_PASSED, "ts": "2026-07-14T10:03:00+00:00", "verification": "failed"},
    ],
    "blocked": [
        _START,
        {"ts": "2026-07-14T10:00:30+00:00", "type": "question.prompt", "id": "question-1"},
    ],
    "asked_unseen_then_seen": [
        _START,
        {"ts": "2026-07-14T10:00:30+00:00", "type": "question.prompt", "id": "question-1"},
        {
            "ts": "2026-07-14T10:00:31+00:00",
            "type": "question.answer",
            "id": "question-1",
            "answers": [""],
            "source": "headless-default",
            "unseen": True,
        },
        {"ts": "2026-07-14T10:01:00+00:00", "type": "session.resume"},
        {"ts": "2026-07-14T10:01:30+00:00", "type": "question.prompt", "id": "question-1"},
        {
            "ts": "2026-07-14T10:01:40+00:00",
            "type": "question.answer",
            "id": "question-1",
            "answers": ["b"],
            "source": "frontend",
            "unseen": False,
        },
        {**_END_PASSED, "ts": "2026-07-14T10:02:00+00:00"},
    ],
    "answered": [
        _START,
        {"ts": "2026-07-14T10:00:30+00:00", "type": "approval.prompt", "id": "approval-1"},
        {
            "ts": "2026-07-14T10:00:40+00:00",
            "type": "approval.answer",
            "id": "approval-1",
            "approved": True,
            "source": "stdin",
        },
    ],
}


def _shared_facts(
    events: list[dict[str, Any]], tmp_path: pathlib.Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    log = tmp_path / "logs.jsonl"
    log.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    scan = session_log.scan_session_log(log)
    state = viewmodel_state.fold_session(events)
    listing: dict[str, Any] = {
        "task": scan.task_label.task,
        "finished": scan.finished,
        "stop_reason": scan.stop_reason if scan.finished else "",
        "verification": scan.verification if scan.finished else None,
        "verify_scoped": scan.verify_scoped,
        "gate_red": scan.status_facts().gate_red,
        "gate_timed_out": scan.status_facts().gate_timed_out,
        "cost_usd": scan.cost_usd,
        "usd_partial": scan.usd_partial,
        "tokens": (
            scan.input_tokens,
            scan.output_tokens,
            scan.cache_read_tokens,
            scan.cache_creation_tokens,
        ),
        "plan": (scan.plan_observed_points, scan.plan_prior_executions, scan.plan_cap),
        "pins": scan.pins,
        "blocked": scan.operator_blocked,
        "unattended": scan.unattended_questions,
        "environment": scan.environment,
    }
    viewer: dict[str, Any] = {
        "task": state.task_label.task,
        "finished": state.finished,
        "stop_reason": state.stop_reason if state.finished else "",
        "verification": state.verification if state.finished else None,
        "verify_scoped": state.verify_scoped,
        "gate_red": status_facts(state).gate_red,
        "gate_timed_out": status_facts(state).gate_timed_out,
        "cost_usd": state.budget.session.usd if scan.cost_usd is not None else None,
        "usd_partial": state.budget.session.usd_partial,
        "tokens": (
            state.budget.execution.input_tokens,
            state.budget.execution.output_tokens,
            state.budget.execution.cache_read_tokens,
            state.budget.execution.cache_creation_tokens,
        ),
        "plan": (
            state.budget.session.plan_observed_points,
            state.budget.session.plan_observed_points - state.budget.execution.plan_observed_points,
            state.budget.execution.plan_cap or 0.0,
        ),
        "pins": state.pins,
        "blocked": any(not p.answered for p in (*state.approvals, *state.questions)),
        "unattended": state.unattended_questions,
        "environment": state.environment,
    }
    life, usage = state.core.lifecycle, state.core.budget
    core: dict[str, Any] = {
        **viewer,
        "task": life.task_label.task,
        "finished": life.finished,
        "stop_reason": life.stop_reason if life.finished else "",
        "verification": life.verification if life.finished else None,
        "verify_scoped": life.verify_scoped,
        "gate_red": life.last_verify_rc is not None and life.last_verify_rc != 0,
        "gate_timed_out": life.verify_timeout_s > 0,
        "cost_usd": usage.session.usd if scan.cost_usd is not None else None,
        "usd_partial": usage.session.usd_partial,
        "tokens": (
            usage.execution.input_tokens,
            usage.execution.output_tokens,
            usage.execution.cache_read_tokens,
            usage.execution.cache_creation_tokens,
        ),
        "plan": (
            usage.session.plan_observed_points,
            usage.session.plan_observed_points - usage.execution.plan_observed_points,
            usage.execution.plan_cap or 0.0,
        ),
        "pins": life.pins,
        "unattended": life.unattended_questions,
        "environment": life.environment,
    }
    return listing, viewer, core


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_the_listing_and_the_viewer_fold_agree(shape: str, tmp_path: pathlib.Path) -> None:
    listing, viewer, core = _shared_facts(_SHAPES[shape], tmp_path)
    assert listing == viewer == core, shape


def test_the_viewer_fold_counts_unseen_questions_per_execution() -> None:
    """The viewers' count of questions nobody saw starts again at a resume, as the listing's."""
    shape = _SHAPES["asked_unseen_then_seen"]
    resume = next(i for i, e in enumerate(shape) if e["type"] == "session.resume")
    assert viewmodel_state.fold_session(shape[:resume]).unattended_questions == 1
    assert viewmodel_state.fold_session(shape).unattended_questions == 0


def test_the_folds_agree_on_the_golden_log(tmp_path: pathlib.Path) -> None:
    from agent6.viewmodel import tail

    listing, viewer, core = _shared_facts(list(tail.tail_events(_GOLDEN, follow=False)), tmp_path)
    assert listing == viewer == core


def _verify_end(exit_code: int, **fields: Any) -> dict[str, Any]:
    return {
        "ts": "2026-07-14T10:00:50+00:00",
        "type": "verify.end",
        "cmd": ["pytest"],
        "exit_code": exit_code,
        "duration_s": 300.0,
        **fields,
    }


_END_RED: dict[str, Any] = {
    "ts": "2026-07-14T10:01:00+00:00",
    "type": "session.stop",
    "reason": "finish_session",
    "verification": "failed",
}


@pytest.mark.parametrize("recorded", [False, True])
def test_exit_124_without_a_timeout_fact_stays_red(tmp_path: pathlib.Path, recorded: bool) -> None:
    """A configured deadline alone does not make exit 124 a timeout in any view."""
    from agent6 import evidence
    from agent6.viewmodel import transcript

    end = _verify_end(124, timeout_s=300.0)
    if recorded:
        end["outcome"] = "failed"
    events = [_START, end, _END_RED]
    listing, viewer, core = _shared_facts(events, tmp_path)
    assert listing == viewer == core
    assert listing["gate_timed_out"] is False
    assert evidence.outcome(end) == "failed"
    done = next(it for it in transcript.fold_transcript(events) if it.kind == "done")
    assert done.name == "finished · gate red"


def test_both_folds_tell_a_timed_out_gate_from_a_red_one(tmp_path: pathlib.Path) -> None:
    """A verify killed at `verify_timeout_s` read "gate red", as if the suite had failed."""
    from agent6.viewmodel import transcript

    timed_out = [_START, _verify_end(124, timeout_s=300.0, outcome="timed_out"), _END_RED]
    listing, viewer, core = _shared_facts(timed_out, tmp_path)
    assert listing == viewer == core
    assert listing["gate_timed_out"] is True and listing["gate_red"] is True
    done = next(it for it in transcript.fold_transcript(timed_out) if it.kind == "done")
    assert done.name == "finished · gate timed out"
    # An exit without a timeout fact is a red gate.
    for red in (_verify_end(124), _verify_end(124, timeout_s=0.0), _verify_end(1, timeout_s=300.0)):
        listing, viewer, core = _shared_facts([_START, red, _END_RED], tmp_path)
        assert listing == viewer == core
        assert listing["gate_timed_out"] is False and listing["gate_red"] is True
    # A later verdict, or a new execution, replaces the timeout.
    for later in (_verify_end(1, timeout_s=300.0), {**_START, "type": "session.resume"}):
        events = [_START, _verify_end(124, timeout_s=300.0, outcome="timed_out"), later, _END_RED]
        listing, viewer, core = _shared_facts(events, tmp_path)
        assert listing == viewer and listing["gate_timed_out"] is False


def test_the_folds_agree_that_a_queued_question_blocks_nothing(tmp_path: pathlib.Path) -> None:
    """Both folds carry a queued question past a resume and block only on a waiting one."""
    prompt = {
        "ts": "2026-07-14T10:00:01+00:00",
        "type": "question.prompt",
        "id": "question-1",
        "questions": [{"question": "a?", "options": []}],
        "call_id": 1,
        "wait": False,
    }
    rows = [
        {"ts": "2026-07-14T10:00:00+00:00", "type": "session.start", "user_task": "t"},
        prompt,
        {"ts": "2026-07-14T10:00:02+00:00", "type": "session.resume", "user_task": "t"},
    ]
    waiting = {**prompt, "ts": "2026-07-14T10:00:03+00:00", "id": "question-2", "wait": True}
    for events, blocked in ((rows, False), ([*rows, waiting], True)):
        log = tmp_path / "logs.jsonl"
        log.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        scan = session_log.scan_session_log(log).status_facts()
        viewer = status_facts(viewmodel_state.fold_session(events))
        assert (scan.operator_blocked, scan.blocked_kind) == (
            blocked,
            "question" if blocked else "",
        )
        assert (viewer.operator_blocked, viewer.blocked_kind) == (blocked, scan.blocked_kind)
        assert scan.pending_questions == viewer.pending_questions == (2 if blocked else 1)
