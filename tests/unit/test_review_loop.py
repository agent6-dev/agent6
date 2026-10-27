# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""In-loop review-panel wiring.

The panel fires at before_finish, grounds against the run diff, and gates the finish only under a
gating block_on (no network).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Literal
from unittest import mock

import pytest

from agent6 import event_log, review_records
from agent6.harness import _chain, _context, _conversation, _panel, _reviewer, loop
from agent6.tools import results
from agent6.vcs import git_read
from tests.unit import test_review_diff_collect
from tests.unit.test_review_gate import (
    _exec,  # pyright: ignore[reportPrivateUsage]
    _finish_tool_use,  # pyright: ignore[reportPrivateUsage]
    _resp,  # pyright: ignore[reportPrivateUsage]
    _resp_with_tool_use,  # pyright: ignore[reportPrivateUsage]
    _verify_pass_tool_use,  # pyright: ignore[reportPrivateUsage]
    _wf,  # pyright: ignore[reportPrivateUsage]
)

# A diff touching foo.py new line 11, so a "foo.py:11" citation grounds.
_DIFF = """\
--- a/foo.py
+++ b/foo.py
@@ -10,1 +10,2 @@ def f():
     x = 1
+    API_KEY = "sk-secret"
"""
_BLOCK = (
    '{"verdict":"block","summary":"leak","findings":[{"category":"security",'
    '"severity":"block","location":{"path":"foo.py","side":"head","start_line":11,"end_line":11},'
    '"title":"hardcoded key","detail":"x"}]}'
)
_PASS = '{"verdict":"pass","summary":"ok","findings":[]}'
_BLOCK_AND_NEW = json.dumps(
    {
        "verdict": "block",
        "summary": "leak and data loss",
        "findings": [
            *json.loads(_BLOCK)["findings"],
            {
                "category": "data-loss",
                "severity": "block",
                "location": {"path": "foo.py", "side": "head", "start_line": 11, "end_line": 11},
                "title": "overwrites data",
                "detail": "x",
            },
        ],
    }
)
_NONGATING = (  # grounded but a non-block-eligible category -> downgraded, never gates
    '{"verdict":"block","summary":"style","findings":[{"category":"style",'
    '"severity":"block","location":{"path":"foo.py","side":"head","start_line":11,"end_line":11},'
    '"title":"rename it","detail":"x"}]}'
)


@pytest.fixture
def immutable_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give scheduling tests a fixed diff and blob without masking the real-repository test."""
    target = review_records.ReviewTarget(
        "base", "base-tree", "tip", "head-tree", "tip", (), "revisions"
    )
    monkeypatch.setattr(
        _context, "review_rules", mock.Mock(return_value=review_records.ReviewRules())
    )
    monkeypatch.setattr(_chain.RunChain, "review_target", mock.Mock(return_value=target))
    monkeypatch.setattr(_chain.RunChain, "diff_since_base", mock.Mock(return_value=_DIFF))
    monkeypatch.setattr(git_read, "diff_range", mock.Mock(return_value=_DIFF))
    monkeypatch.setattr(git_read, "tree_entry", mock.Mock(return_value="100644 blob cited-blob"))


def _seat(provider: Any, focus: str = "security", model: str = "m1") -> _reviewer.ReviewSeat:
    return _reviewer.ReviewSeat(focus=focus, model=model, provider=provider)


def _disp() -> mock.MagicMock:
    d = mock.MagicMock()
    d.dispatch.return_value = results.RawResult(
        {"ok": True}
    )  # JSON-serializable (finish_session is dispatched)
    return d


def _begin() -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": "TASK:\ngo\n\nBegin."}]}]


def _drive(wf: loop.Harness, messages: list[dict[str, Any]]) -> Any:
    conversation = _conversation.Conversation.from_wire(messages)
    result = wf._drive_loop(  # pyright: ignore[reportPrivateUsage]
        system="S",
        conversation=conversation,
        tool_calls=0,
        start_turn=1,
        root_task_id=None,
        original_task="t",
    )
    # The callers' fixtures still read the final history via *messages*.
    messages[:] = conversation.to_wire()
    return result


def test_has_reviewer_true_with_seats() -> None:
    wf = _wf(review=_reviewer.ReviewSettings(reviewers=[_seat(mock.MagicMock())]))
    assert wf.reviewer.available() is True
    assert _wf(review=_reviewer.ReviewSettings(reviewers=[])).reviewer.available() is False


@pytest.mark.usefixtures("immutable_target")
def test_panel_blocks_finish_under_veto_then_accepts() -> None:
    """A grounded security block under veto revokes the finish until the seat passes.

    The first `finish_session` is revoked; once the seat passes, the second is accepted.
    """
    worker = mock.MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("f1", _finish_tool_use("a", "done")),
        _resp_with_tool_use("f2", _finish_tool_use("b", "done")),
    ]
    seat_provider = mock.MagicMock()
    seat_provider.call.side_effect = [_resp(_BLOCK), _resp(_PASS)]
    wf = _wf(
        provider=worker,
        dispatcher=_disp(),
        review=_reviewer.ReviewSettings(
            reviewers=[_seat(seat_provider)], block_on="any", trigger="before_finish"
        ),
        base_sha="b",
    )
    messages = _begin()
    result = _drive(wf, messages)
    assert result.reason == "finish_session" and result.turns == 2
    assert seat_provider.call.call_count == 2  # panel ran at each finish attempt
    # the first (blocked) finish injected the findings for the worker to address
    injected = "".join(
        b.get("text", "") for m in messages for b in (m.get("content") or []) if isinstance(b, dict)
    )
    assert "rejected your finish_session" in injected and "hardcoded key" in injected


@pytest.mark.parametrize(
    ("second_panel", "rejections"), [(_BLOCK, 1), (_BLOCK_AND_NEW, 2)], ids=["duplicate", "new"]
)
@pytest.mark.usefixtures("immutable_target")
def test_panels_count_and_inject_each_finding_once(
    tmp_path: pathlib.Path, second_panel: str, rejections: int
) -> None:
    """A repeat neither rejects nor injects again, while a new finding still does both."""
    worker = mock.MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use(f"f{i}", _finish_tool_use(str(i), "done")) for i in range(4)
    ]
    seat_provider = mock.MagicMock()
    seat_provider.call.side_effect = [
        _resp(_BLOCK),
        _resp(second_panel),
        _resp(second_panel),
        _resp(_PASS),
    ]
    snapshot = tmp_path / "loop_state.json"
    events = event_log.EventSink(tmp_path / "logs.jsonl")
    wf = _wf(
        provider=worker,
        dispatcher=_disp(),
        review=_reviewer.ReviewSettings(
            reviewers=[_seat(seat_provider)], block_on="any", trigger="before_finish"
        ),
        base_sha="b",
        resume_state_path=snapshot,
        events=events,
    )
    messages = _begin()
    result = _drive(wf, messages)
    assert json.loads(snapshot.read_text())["review_rejections_total"] == rejections
    assert result.reason == "finish_session" and result.turns == rejections + 1
    feedback = [
        b["text"]
        for m in messages
        for b in m.get("content", [])
        if b.get("type") == "text" and b["text"].startswith("[review]")
    ]
    assert len(feedback) == rejections
    assert "".join(feedback).count("hardcoded key") == 1
    assert "".join(feedback).count("overwrites data") == rejections - 1

    panels = [
        json.loads(line)
        for line in events.path.read_text().splitlines()
        if json.loads(line)["type"] == "loop.review.panel"
    ]
    observations = panels[0].get("findings")
    assert isinstance(observations, list) and observations
    assert observations[0]["finding"]["title"] == "hardcoded key"
    assert panels[1]["findings"][0]["finding_id"] == observations[0]["finding_id"]

    # A resume reads already-shown findings from the journal, not from process memory.
    worker.call.side_effect = [
        _resp_with_tool_use(f"resumed{i}", _finish_tool_use(f"r{i}", "done")) for i in range(3)
    ]
    seat_provider.call.reset_mock()
    seat_provider.call.side_effect = [_resp(_BLOCK), _resp(_BLOCK), _resp(_PASS)]
    resumed = wf.resume()
    assert json.loads(snapshot.read_text())["review_rejections_total"] == rejections
    assert resumed.reason == "finish_session" and resumed.turns == rejections + 2
    prompt = seat_provider.call.call_args_list[0].kwargs["messages"][0]["content"]
    assert "ALREADY RAISED" in prompt
    rows = [json.loads(line) for line in events.path.read_text().splitlines()]
    starts = [row for row in rows if row["type"] == "loop.review.start"]
    assert len({row["panel_id"] for row in starts}) == len(starts)
    resume_start = next(row for row in rows if row["type"] == "loop.resume.start")
    assert starts[-1]["execution_id"] == resume_start["execution_id"]
    assert starts[-1]["execution_id"] != starts[0]["execution_id"]


def test_resumes_match_findings_by_the_cited_git_blob(tmp_path: pathlib.Path) -> None:
    """Keep a finding known across unrelated edits, but count it again on a changed blob."""
    git = test_review_diff_collect._git
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    source = root / "foo.py"
    prefix = "x = 1\n" * 10
    source.write_text(prefix, encoding="utf-8")
    git(root, "add", "foo.py")
    git(root, "commit", "-qm", "base")
    base = git(root, "rev-parse", "HEAD").strip()
    source.write_text(prefix + 'API_KEY = "sk-secret"\n', encoding="utf-8")

    worker = mock.MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use(f"f{i}", _finish_tool_use(str(i), "done")) for i in range(5)
    ]
    seat_provider = mock.MagicMock()
    seat_provider.call.return_value = _resp(_BLOCK)
    snapshot = tmp_path / "session" / "loop_state.json"
    events = event_log.EventSink(snapshot.with_name("logs.jsonl"))

    def harness() -> loop.Harness:
        return _wf(
            root,
            ref="refs/agent6/test/head",
            fallback_parent=base,
            base_sha=base,
            provider=worker,
            dispatcher=_disp(),
            review=_reviewer.ReviewSettings(
                reviewers=[_seat(seat_provider)], block_on="any", trigger="before_finish"
            ),
            resume_state_path=snapshot,
            events=events,
        )

    wf = harness()
    first_tip = wf.chain.commit("add key")
    first_blob = git(root, "rev-parse", f"{first_tip}:foo.py").strip()
    result = _drive(wf, _begin())
    assert json.loads(snapshot.read_text())["review_rejections_total"] == 1
    assert result.reason == "finish_session" and result.turns == 2

    (root / "other.txt").write_text("unrelated edit\n", encoding="utf-8")
    second_tip = wf.chain.commit("edit another file")
    assert second_tip != first_tip
    assert git(root, "rev-parse", f"{second_tip}:foo.py").strip() == first_blob
    resumed = harness().resume()
    assert json.loads(snapshot.read_text())["review_rejections_total"] == 1
    assert resumed.reason == "finish_session" and resumed.turns == 3

    source.write_text(prefix + 'API_KEY = "sk-replaced"\n', encoding="utf-8")
    third_tip = wf.chain.commit("change cited blob")
    assert git(root, "rev-parse", f"{third_tip}:foo.py").strip() != first_blob
    changed = harness().resume()
    assert json.loads(snapshot.read_text())["review_rejections_total"] == 2
    assert changed.reason == "finish_session" and changed.turns == 5

    panels = [
        row
        for line in events.path.read_text().splitlines()
        if (row := json.loads(line))["type"] == "loop.review.panel"
    ]
    assert len(panels) == 5
    assert all(len(panel["findings"]) == 1 for panel in panels)
    ids = [panel["findings"][0]["finding_id"] for panel in panels]
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] == ids[4] and ids[3] != ids[0]
    assert panels[0]["findings"][0]["finding"] == panels[3]["findings"][0]["finding"]
    assert [panel["n_block"] for panel in panels] == [1, 0, 0, 1, 0]
    assert [panel["blocked"] for panel in panels] == [True, False, False, True, False]
    assert [panel["injected_finding_ids"] for panel in panels] == [[ids[0]], [], [], [ids[3]], []]
    assert [panel["target"]["head_commit"] for panel in panels] == [
        first_tip,
        first_tip,
        second_tip,
        third_tip,
        third_tip,
    ]


@pytest.mark.usefixtures("immutable_target")
def test_second_panel_prompt_names_prior_finding() -> None:
    """The second seat call receives the finding injected after the first panel."""
    worker = mock.MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("f1", _finish_tool_use("a", "done")),
        _resp_with_tool_use("f2", _finish_tool_use("b", "done")),
    ]
    seat_provider = mock.MagicMock()
    seat_provider.call.side_effect = [_resp(_BLOCK), _resp(_PASS)]
    wf = _wf(
        provider=worker,
        dispatcher=_disp(),
        review=_reviewer.ReviewSettings(
            reviewers=[_seat(seat_provider)], block_on="any", trigger="before_finish"
        ),
        base_sha="b",
    )
    _drive(wf, _begin())
    prompt = seat_provider.call.call_args_list[1].kwargs["messages"][0]["content"]
    assert "ALREADY RAISED (do not repeat): foo.py:11 security" in prompt


@pytest.mark.parametrize("trigger", ["periodic", "on_verify_fail"])
@pytest.mark.usefixtures("immutable_target")
def test_observe_only_panels_do_not_repeat_findings(
    trigger: Literal["periodic", "on_verify_fail"],
) -> None:
    """Advisory panels carry their injected findings to the next seat call."""
    worker = mock.MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use("t1", _verify_pass_tool_use("v1")),
        _resp_with_tool_use("t2", _verify_pass_tool_use("v2")),
        _resp_with_tool_use("done", _finish_tool_use("f", "done")),
    ]
    dispatcher = _disp()
    dispatcher.dispatch.side_effect = [
        _exec(1, stderr="test x failed"),
        _exec(1, stderr="test x failed"),
        results.RawResult({"ok": True}),
    ]
    seat_provider = mock.MagicMock()
    seat_provider.call.side_effect = [_resp(_BLOCK), _resp(_BLOCK), _resp(_PASS)]
    wf = _wf(
        provider=worker,
        dispatcher=dispatcher,
        review=_reviewer.ReviewSettings(
            reviewers=[_seat(seat_provider)], trigger=trigger, interval_turns=1
        ),
        base_sha="b",
    )
    messages = _begin()
    result = _drive(wf, messages)
    assert result.reason == "finish_session" and result.turns == 3
    feedback = "".join(
        b["text"]
        for m in messages
        for b in m.get("content", [])
        if b.get("type") == "text" and b["text"].startswith("[review]")
    )
    assert feedback.count("hardcoded key") == 1
    prompt = seat_provider.call.call_args_list[1].kwargs["messages"][0]["content"]
    assert "ALREADY RAISED (do not repeat): foo.py:11 security" in prompt


@pytest.mark.usefixtures("immutable_target")
def test_panel_skipped_when_budget_fraction_low() -> None:
    """Below `min_remaining_budget_fraction` the panel is skipped and the finish stands.

    Reviewing costs most when budget is scarcest. The seat that WOULD block is never called, and
    finish_session is accepted. This is the only behavioural test of the floor.
    """
    worker = mock.MagicMock()
    worker.call.return_value = _resp_with_tool_use("f1", _finish_tool_use("a", "done"))
    seat_provider = mock.MagicMock()
    seat_provider.call.side_effect = [_resp(_BLOCK)]  # would block IF the panel ran
    wf = _wf(
        provider=worker,
        dispatcher=_disp(),
        review=_reviewer.ReviewSettings(
            reviewers=[_seat(seat_provider)], block_on="any", trigger="before_finish"
        ),
        base_sha="b",
    )  # min_remaining_budget_fraction defaults to 0.25
    with mock.patch.object(loop.Harness, "_budget_fraction_remaining", return_value=0.10):
        result = _drive(wf, _begin())
    assert seat_provider.call.call_count == 0  # panel skipped, not run
    assert result.reason == "finish_session" and result.turns == 1


@pytest.mark.usefixtures("immutable_target")
def test_panel_advisory_does_not_block_finish() -> None:
    """The SAME grounded block under advisory does not gate -> finish on turn 1."""
    worker = mock.MagicMock()
    worker.call.side_effect = [_resp_with_tool_use("f1", _finish_tool_use("a", "done"))]
    seat_provider = mock.MagicMock()
    seat_provider.call.return_value = _resp(_BLOCK)
    wf = _wf(
        provider=worker,
        dispatcher=_disp(),
        review=_reviewer.ReviewSettings(
            reviewers=[_seat(seat_provider)], block_on="none", trigger="before_finish"
        ),
        base_sha="b",
    )
    result = _drive(wf, _begin())
    assert result.reason == "finish_session" and result.turns == 1
    assert seat_provider.call.call_count == 1  # panel still ran (events), just didn't gate


@pytest.mark.usefixtures("immutable_target")
def test_panel_does_not_block_on_nongating_category_even_under_veto() -> None:
    """A grounded 'style' block never stalls a finish, even under veto.

    The aggregator downgrades it, so the finish is accepted on the first turn.
    """
    worker = mock.MagicMock()
    worker.call.side_effect = [_resp_with_tool_use("f1", _finish_tool_use("a", "done"))]
    seat_provider = mock.MagicMock()
    seat_provider.call.return_value = _resp(_NONGATING)
    wf = _wf(
        provider=worker,
        dispatcher=_disp(),
        review=_reviewer.ReviewSettings(
            reviewers=[_seat(seat_provider)], block_on="any", trigger="before_finish"
        ),
        base_sha="b",
    )
    result = _drive(wf, _begin())
    assert result.reason == "finish_session" and result.turns == 1


@pytest.mark.parametrize("cap", [2, 4])
@pytest.mark.usefixtures("immutable_target")
def test_the_configured_cap_counts_rejected_finishes(cap: int) -> None:
    """The one counter is the configured cap: the finish after the cap's rejection stands.

    A persistently blocking panel cannot stall the run forever, and nothing else lets an end
    through early: the default cap of 4 means four rejections, not two.
    """
    worker = mock.MagicMock()
    # Each panel finds a distinct grounded block while the worker keeps trying to finish.
    worker.call.side_effect = [
        _resp_with_tool_use(f"f{i}", _finish_tool_use(str(i), "done")) for i in range(8)
    ]
    seat_provider = mock.MagicMock()
    seat_provider.call.side_effect = [
        _resp(_BLOCK.replace('"security"', f'"{category}"'))
        for category in (
            "security",
            "sandbox-bypass",
            "off-topic-edit",
            "data-loss",
            "verify-uncovered-correctness",
        )
    ]
    wf = _wf(
        provider=worker,
        dispatcher=_disp(),
        review=_reviewer.ReviewSettings(
            reviewers=[_seat(seat_provider)],
            block_on="any",
            trigger="before_finish",
            max_review_rejections=cap,
        ),
        base_sha="b",
    )
    result = _drive(wf, _begin())
    assert result.reason == "finish_session" and result.turns == cap + 1


@pytest.mark.usefixtures("immutable_target")
def test_in_loop_panel_all_abstain_names_the_abstention() -> None:
    """The in-loop panel names an all-abstain panel's abstention, never 'No blocking findings.'.

    The in-loop panel spends the run's budget and feeds the critique to the model; it uses the
    shared `panel_is_inconclusive` and `inconclusive_note` owner, and the gate still lets the finish
    through, since a panel never deadlocks a run.
    """
    from agent6.harness import _loop_state, _panel
    from agent6.harness import _reviewer as review_mod

    abstain = review_records.ReviewVerdict(
        seat_index=0, focus="s", model_route="m", verdict="abstain", error="output hit the cap"
    )
    res = _panel.PanelResult(
        panel_id="p",
        block_on="none",
        blocked=False,
        merged_findings=(),
        per_seat=(abstain, abstain, abstain),
        n_block=0,
        n_abstain=3,
    )
    wf = _wf(
        provider=mock.MagicMock(),
        dispatcher=_disp(),
        review=_reviewer.ReviewSettings(reviewers=[_seat(mock.MagicMock())], block_on="none"),
        base_sha="b",
    )
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    with (
        mock.patch.object(review_mod, "run_panel", return_value=res),
    ):
        out = wf.reviewer.critique(state, trigger="periodic", turn=1)
    assert out is not None
    assert "inconclusive" in out.text and "abstained" in out.text
    assert "No blocking findings" not in out.text
    assert out.satisfied is True  # never deadlocks the run


@pytest.mark.usefixtures("immutable_target")
def test_findings_the_notice_cut_are_not_remembered_as_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finding the notice cut was remembered as injected, so later panels dropped it unseen."""
    monkeypatch.setattr(_panel, "REVIEW_NOTICE_BYTES", 1_000)
    big = json.dumps(
        {
            "verdict": "block",
            "summary": "three",
            "findings": [
                {
                    "category": category,
                    "severity": "block",
                    "location": {
                        "path": "foo.py",
                        "side": "head",
                        "start_line": 11,
                        "end_line": 11,
                    },
                    "title": f"finding {i}",
                    "detail": "d" * 500,
                }
                for i, category in enumerate(("security", "data-loss", "sandbox-bypass"), 1)
            ],
        }
    )
    worker = mock.MagicMock()
    worker.call.side_effect = [
        _resp_with_tool_use(f"f{i}", _finish_tool_use(str(i), "done")) for i in range(4)
    ]
    seat_provider = mock.MagicMock()
    seat_provider.call.side_effect = [_resp(big), _resp(big), _resp(big), _resp(_PASS)]
    wf = _wf(
        provider=worker,
        dispatcher=_disp(),
        review=_reviewer.ReviewSettings(
            reviewers=[_seat(seat_provider)], block_on="any", trigger="before_finish"
        ),
        base_sha="b",
    )
    messages = _begin()
    result = _drive(wf, messages)
    notices = [
        b["text"]
        for m in messages
        for b in m.get("content", [])
        if b.get("type") == "text" and b["text"].startswith("[review]")
    ]
    assert result.reason == "finish_session" and result.turns == 4
    shown = [[f"finding {i}" for i in (1, 2, 3) if f"finding {i}" in n] for n in notices]
    assert sorted(title for titles in shown for title in titles) == [
        "finding 1",
        "finding 2",
        "finding 3",
    ]
    assert [len(titles) for titles in shown] == [1, 1, 1]
    assert "[2 more findings not shown]" in notices[0]
    assert "[1 more finding not shown]" in notices[1]
    assert "not shown" not in notices[2]
    category = {"finding 1": "security", "finding 2": "data-loss", "finding 3": "sandbox-bypass"}
    prompt = seat_provider.call.call_args_list[1].kwargs["messages"][0]["content"]
    assert f"foo.py:11 {category[shown[0][0]]}" in prompt
    assert all(f"foo.py:11 {category[t[0]]}" not in prompt for t in shown[1:])
