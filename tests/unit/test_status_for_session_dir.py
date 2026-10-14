# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One status decision for any run with a dir.

``status_for_session_dir`` is the single place a run's status word is decided from
its dir; listings (``summarize_session_dir``) and every header feed it facts from
either producer -- the tolerant scanner (``LogScan``) or the typed fold
(``SessionState``). The matrix pins the word per dir state for BOTH producers and
for the listing, so no two surfaces can disagree about any non-``session.end``
state (parked/created/starting/stale/waiting each shipped a real
surface-disagreement bug before this existed).
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from agent6.sessions import ipc
from agent6.viewmodel import format, listing, transcript
from agent6.viewmodel import state as viewmodel_state

LIVE = os.getpid()
DEAD = 999999999


def _mk(
    tmp_path: pathlib.Path,
    name: str,
    events: list[dict[str, object]] | None,
    *,
    parked: str = "",
    pid: int | None = None,
) -> pathlib.Path:
    d = tmp_path / name
    d.mkdir()
    manifest: dict[str, object] = {"mode": "run", "session_id": name, "user_task": "t"}
    if parked:
        manifest["parked_task"] = parked
        manifest["parked_reason"] = "checkout busy"
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if events is not None:
        (d / "logs.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
        )
    if pid is not None:
        (d / "worker.pid").write_text(str(pid), encoding="utf-8")
    return d


_START: dict[str, object] = {"type": "session.start", "mode": "run", "user_task": "t"}
_TOOL: dict[str, object] = {"type": "tool.call", "name": "grep", "args": {}}
_APPROVAL: dict[str, object] = {"type": "approval.prompt", "id": "approval-1", "prompt": "ok?"}
_QUESTION: dict[str, object] = {
    "type": "question.prompt",
    "id": "q-1",
    "questions": [{"question": "x?"}],
}


_ANSWER: dict[str, object] = {"type": "approval.answer", "id": "approval-1", "approved": True}
_RESUME: dict[str, object] = {"type": "loop.resume.start", "iteration": 2}


def _end(reason: str, all_passed: bool = False) -> dict[str, object]:
    return {"type": "session.end", "reason": reason, "all_passed": all_passed}


# (name, events (None = no logs.jsonl), parked_task, pid, expected word, expected reason)
MATRIX: list[tuple[str, list[dict[str, object]] | None, str, int | None, str, str]] = [
    ("created", None, "", None, "created", ""),
    ("parked", None, "fix it", None, "parked", "checkout busy"),
    # A parked run being resumed: a live worker in its pre-session.start preflight
    # window reads "starting" even while the manifest still carries parked_task
    # (resume clears it only once under way).
    ("parked-resuming", None, "fix it", LIVE, "starting", ""),
    ("starting", [], "", LIVE, "starting", ""),
    ("running", [_START, _TOOL], "", LIVE, "running", ""),
    ("waiting-approval", [_START, _APPROVAL], "", LIVE, "waiting", "needs answer"),
    ("waiting-question", [_START, _QUESTION], "", LIVE, "waiting", "needs answer"),
    # The start question about uncommitted changes comes before session.start.
    ("waiting-before-start", [_QUESTION], "", LIVE, "waiting", "needs answer"),
    # ... and a worker that died on it is not "waiting" for anyone.
    ("died-on-start-question", [_QUESTION], "", DEAD, "stale", "died launching"),
    (
        "answered-runs-on",
        [_START, _APPROVAL, _ANSWER],
        "",
        LIVE,
        "running",
        "",
    ),
    # A crash while waiting, then a resume: the new execution re-prompts with fresh
    # ids, so the orphaned prompt must not read "waiting" (or duplicate) forever.
    (
        "resumed-past-orphaned-prompt",
        [_START, _APPROVAL, _RESUME, _TOOL],
        "",
        LIVE,
        "running",
        "",
    ),
    # A prompt is still pending when a LATER event lands: Ctrl-C in the
    # launching terminal emits session.steer_requested from the signal handler
    # while the approval waits. "Blocked" is about the unanswered prompt, not
    # about which event happened to be last.
    (
        "waiting-with-a-later-event",
        [_START, _APPROVAL, {"type": "session.steer_requested", "source": "sigint"}],
        "",
        LIVE,
        "waiting",
        "needs answer",
    ),
    # A FORK is driven by resume(), so its fresh log never carries a session.start:
    # keying "started" on that alone left every forked run "starting" while alive
    # and "created" (the never-started word) once it died.
    ("forked-live", [_RESUME, _TOOL], "", LIVE, "running", ""),
    ("forked-dead", [_RESUME, _TOOL], "", DEAD, "stale", ""),
    ("forked-waiting", [_RESUME, _APPROVAL], "", LIVE, "waiting", "needs answer"),
    ("stale", [_START, _TOOL], "", DEAD, "stale", ""),
    ("stale-beats-waiting", [_START, _APPROVAL], "", DEAD, "stale", ""),
    # A worker killed during PREFLIGHT (its pid file survives a kill; no
    # session.start ever lands) used to read "created" -- the fork --no-run
    # word -- beside the real dollars its verify-inference call spent.
    (
        "killed-in-preflight",
        [{"type": "budget.update", "usd_total": 0.02}],
        "",
        DEAD,
        "stale",
        "died launching",
    ),
    ("killed-in-preflight-no-log", None, "", DEAD, "stale", "died launching"),
    ("passed", [_START, _end("finish_session", True)], "", None, "passed", ""),
    (
        "passed-scoped",
        [_START, {**_end("finish_session", True), "scoped": True}],
        "",
        None,
        "passed",
        "scoped gate",
    ),
    (
        "finished-gate-red",
        [_START, {"type": "verify.end", "cmd": ["pytest"], "exit_code": 1}, _end("finish_session")],
        "",
        None,
        "finished",
        "gate red",
    ),
    ("finished-unverified", [_START, _end("finish_session")], "", None, "finished", "unverified"),
    (
        "finished-gateless",
        [_START, {**_end("finish_session"), "all_passed": None}],
        "",
        None,
        "finished",
        "",
    ),
    ("settled", [_START, _end("settled")], "", None, "finished", "unverified"),
    ("failed", [_START, _end("provider_error")], "", None, "failed", "provider_error"),
    ("stopped", [_START, _end("steer_abort")], "", None, "stopped", ""),
    ("planned", [_START, _end("finish_planning")], "", None, "planned", ""),
    ("answered", [_START, _end("answered")], "", None, "answered", ""),
]


@pytest.mark.parametrize(
    ("name", "events", "parked", "pid", "word", "reason"),
    MATRIX,
    ids=[row[0] for row in MATRIX],  # LIVE is this process's pid: never in an id
)
def test_both_fact_producers_and_the_listing_agree(
    tmp_path: pathlib.Path,
    name: str,
    events: list[dict[str, object]] | None,
    parked: str,
    pid: int | None,
    word: str,
    reason: str,
) -> None:
    d = _mk(tmp_path, name, events, parked=parked, pid=pid)
    logs = d / "logs.jsonl"
    scan = listing.scan_session_log(logs) if logs.is_file() else listing.LogScan()
    fold = viewmodel_state.fold_session([] if events is None else events)
    assert listing.status_for_session_dir(d, scan.status_facts()) == (word, reason)
    assert listing.status_for_session_dir(d, viewmodel_state.status_facts(fold)) == (word, reason)
    summary = listing.summarize_session_dir(d)
    assert (summary.status, summary.reason) == (word, reason)


@pytest.mark.parametrize(
    ("reason", "detail"),
    [("silent_finish", "no finish call"), ("metric_plateau", "metric plateau")],
)
@pytest.mark.parametrize(
    ("all_passed", "verify_exit", "gate_detail"),
    [
        (True, 0, ""),
        (None, None, ""),
        (False, 1, "gate red"),
        (False, 124, "gate timed out"),
        (False, 0, "unverified"),
    ],
    ids=["green", "no-gate", "red", "timed-out", "stale-green"],
)
@pytest.mark.parametrize("scoped", [False, True])
def test_harness_end_labels_agree_across_the_scan_and_both_folds(
    tmp_path: pathlib.Path,
    reason: str,
    detail: str,
    all_passed: bool | None,
    verify_exit: int | None,
    gate_detail: str,
    scoped: bool,
) -> None:
    """A silent or plateau end must not become passed in any fold or hide a not-green gate."""
    events: list[dict[str, object]] = [_START]
    if verify_exit is not None:
        events.append(
            {"type": "verify.end", "cmd": ["pytest"], "exit_code": verify_exit, "timeout_s": 300.0}
        )
    events.append(
        {"type": "session.end", "reason": reason, "all_passed": all_passed, "scoped": scoped}
    )
    d = _mk(tmp_path, reason, events)
    scan = listing.scan_session_log(d / "logs.jsonl")
    fold = viewmodel_state.fold_session(events)
    expected = ("finished", gate_detail or detail)
    assert listing.status_for_session_dir(d, scan.status_facts()) == expected
    assert listing.status_for_session_dir(d, viewmodel_state.status_facts(fold)) == expected
    summary = listing.summarize_session_dir(d)
    assert (summary.status, summary.reason) == expected
    label = f"finished · {gate_detail or detail}"
    assert format.listing_status_label(summary.mode, summary.status, summary.reason) == label
    done = next(it for it in transcript.fold_transcript(events) if it.kind == "done")
    assert done.name == label


@pytest.mark.parametrize(
    ("events", "pid", "now", "word", "reason"),
    [
        ([_START, _TOOL], LIVE, False, "running", "stopping after this step"),
        ([_START, _TOOL], LIVE, True, "running", ""),
        ([_START, _TOOL], DEAD, False, "stale", ""),
        ([_START, _end("steer_abort")], LIVE, False, "stopped", ""),
        ([], LIVE, False, "starting", "stopping after this step"),
        ([_START, _QUESTION], LIVE, False, "waiting", "needs answer · stopping after this step"),
    ],
    ids=["running", "immediate", "dead", "ended", "starting", "waiting"],
)
def test_pending_step_stop_labels_agree_in_the_listing_and_typed_fold(
    tmp_path: pathlib.Path,
    events: list[dict[str, object]],
    pid: int,
    now: bool,
    word: str,
    reason: str,
) -> None:
    """A pending after-step stop is visible on live runs in both status fact producers."""
    d = _mk(tmp_path, "stop-AAAAAA", events, pid=pid)
    assert ipc.request_stop(d, now=now)
    fold = viewmodel_state.fold_session(events)
    scan = listing.scan_session_log(d / "logs.jsonl")
    assert listing.status_for_session_dir(d, scan.status_facts()) == (word, reason)
    assert listing.status_for_session_dir(d, viewmodel_state.status_facts(fold)) == (word, reason)
    row = listing.summarize_session_dir(d)
    assert (row.status, row.reason) == (word, reason)
    assert viewmodel_state.session_state_as_dict(fold, d)["status_label"] == format.status_label(
        word, reason
    )
    assert listing.summary_row(row)["label"] == format.status_label(word, reason)


def test_a_verify_cut_short_leaves_no_verdict_in_either_producer(tmp_path: pathlib.Path) -> None:
    """A run killed in a re-verify carries no gate verdict: the red before it was an older tree's.

    The hub read the earlier red while the run page read none.
    """
    events: list[dict[str, object]] = [
        _START,
        {"type": "verify.start", "cmd": ["true"]},
        {"type": "verify.end", "cmd": ["true"], "exit_code": 1, "duration_s": 1.0},
        {"type": "verify.start", "cmd": ["true"]},
    ]
    d = _mk(tmp_path, "cut-short", events)
    scan_facts = listing.scan_session_log(d / "logs.jsonl").status_facts()
    fold_facts = viewmodel_state.status_facts(viewmodel_state.fold_session(events))
    assert (scan_facts.gate_red, fold_facts.gate_red) == (False, False)
    assert listing.status_for_session_dir(d, scan_facts) == listing.status_for_session_dir(
        d, fold_facts
    )


def test_clearing_a_pending_step_stop_clears_its_label(tmp_path: pathlib.Path) -> None:
    """The stop detail follows the marker, not a cached event or a prior listing."""
    events = [_START, _TOOL]
    d = _mk(tmp_path, "clear-AAAAAA", events, pid=LIVE)
    fold = viewmodel_state.fold_session(events)
    assert ipc.request_stop(d)
    assert viewmodel_state.session_state_as_dict(fold, d)["status_label"] == (
        "running · stopping after this step"
    )
    ipc.clear_stop_request(d)
    assert listing.summarize_session_dir(d).reason == ""
    assert viewmodel_state.session_state_as_dict(fold, d)["status_label"] == "running"


@pytest.mark.parametrize(("all_passed", "word"), [(True, "passed"), (None, "finished")])
def test_a_finish_whose_edits_got_no_commit_reads_not_committed(
    tmp_path: pathlib.Path, all_passed: bool | None, word: str
) -> None:
    """A run whose commits all failed listed as plain "passed" while it exited 5.

    The end path journals the fact, both readers carry it, and the next execution starts
    without it.
    """
    end: dict[str, object] = {
        "type": "session.end",
        "reason": "finish_session",
        "all_passed": all_passed,
    }
    events: list[dict[str, object]] = [_START, end, {"type": "session.stranded"}]
    d = _mk(tmp_path, "strand-AAAAAA", events)
    scan = listing.scan_session_log(d / "logs.jsonl")
    fold = viewmodel_state.fold_session(events)
    for facts in (scan.status_facts(), viewmodel_state.status_facts(fold)):
        assert listing.status_for_session_dir(d, facts) == (word, "not committed")
    row = listing.summarize_session_dir(d)
    assert format.status_label(row.status, row.reason) == f"{word} · not committed"

    resumed = [*events, _RESUME, end]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in resumed), encoding="utf-8")
    scan = listing.scan_session_log(d / "logs.jsonl")
    fold = viewmodel_state.fold_session(resumed)
    for facts in (scan.status_facts(), viewmodel_state.status_facts(fold)):
        assert listing.status_for_session_dir(d, facts) == (word, "")


def test_resume_clears_orphaned_pending_prompts() -> None:
    """An execution boundary invalidates unanswered prompts.

    The resumed execution re-asks with restarted ids, so a held-over pending entry would
    mislabel the run "waiting" and duplicate when the same-id prompt arrives.
    """
    events = [
        _START,
        _APPROVAL,
        _QUESTION,
        {"type": "loop.resume.start", "iteration": 2},
        _APPROVAL,
    ]
    state = viewmodel_state.fold_session(events)
    assert len(state.pending_approvals) == 1  # the new execution's, not the orphan + a dup
    assert state.pending_questions == ()


def test_waiting_names_the_prompt_kind_and_age_in_both_producers(tmp_path: pathlib.Path) -> None:
    """The waiting status names what the run waits on and for how long.

    With a prompt ts, both fact producers word it "approval 5m" / "question 5m" (oldest unanswered
    prompt); a ts-less log keeps the generic wording.
    """
    import datetime
    import os

    ts = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=5)).isoformat()
    prompts: tuple[tuple[str, dict[str, object]], ...] = (
        ("approval", {"type": "approval.prompt", "id": "approval-1", "prompt": "p", "ts": ts}),
        ("question", {"type": "question.prompt", "id": "question-1", "questions": [], "ts": ts}),
    )
    for kind, ev in prompts:
        events: list[dict[str, object]] = [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            ev,
        ]
        d = _mk(tmp_path, f"waiting-{kind}", events, parked="", pid=os.getpid())
        scan = listing.scan_session_log(d / "logs.jsonl")
        fold = viewmodel_state.fold_session(events)
        expect = ("waiting", f"{kind} 5m")
        assert listing.status_for_session_dir(d, scan.status_facts()) == expect
        assert listing.status_for_session_dir(d, viewmodel_state.status_facts(fold)) == expect

    no_ts: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "approval.prompt", "id": "approval-1", "prompt": "p"},
    ]
    d = _mk(tmp_path, "waiting-no-ts", no_ts, parked="", pid=os.getpid())
    assert listing.status_for_session_dir(
        d, listing.scan_session_log(d / "logs.jsonl").status_facts()
    ) == (
        "waiting",
        "needs answer",
    )
