# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for the grounded review-panel aggregator (the false-block defense).

These lock in the property that got the pre-0.0.4 reviewer.py deleted once it was
fixed here: a reviewer can only GATE (block a finish) when its objection is
grounded in the actual diff AND in a category we allow to block. Taste, test
gaps, and uncited claims are mechanically downgraded and can never stall a run.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, Literal
from unittest import mock

from agent6 import review_records
from agent6.app import _setup, providers
from agent6.config import layer
from agent6.harness import _panel as harness__panel
from agent6.harness import _reviewer as harness__reviewer
from agent6.vcs import review
from tests.unit import test_review_diff_collect

# A real unified diff touching foo.py new-lines 10..14 and creating bar.py 1..2.
SAMPLE_DIFF = """\
--- a/foo.py
+++ b/foo.py
@@ -10,3 +10,5 @@ def f():
     x = 1
+    y = 2
+    z = 3
     return x
--- /dev/null
+++ b/bar.py
@@ -0,0 +1,2 @@
+import os
+VALUE = 1
"""


def _ctx(**kw: Any) -> harness__panel.ReviewContext:
    return harness__panel.ReviewContext(diff=SAMPLE_DIFF, **kw)


def _seat(
    model: str, *findings: review_records.Finding, seat: str = "s", error: str | None = None
) -> review_records.ReviewVerdict:
    verdict = (
        "abstain" if error else "block" if any(f.severity == "block" for f in findings) else "pass"
    )
    return review_records.ReviewVerdict(
        seat_index=0, focus=seat, model_route=model, verdict=verdict, findings=findings, error=error
    )


def _location(
    path: str,
    start: int | None = None,
    end: int | None = None,
    *,
    side: Literal["base", "head"] = "head",
) -> review_records.ReviewLocation:
    return review_records.ReviewLocation(
        path=path, side=side, start_line=start, end_line=start if end is None else end
    )


def _block(category: str, location: review_records.ReviewLocation) -> review_records.Finding:
    return review_records.Finding(category=category, severity="block", location=location, title="x")


def _agg(
    seats: list[review_records.ReviewVerdict],
    *,
    block_on: harness__panel.ReviewBlockOn = "any",
    min_blocking_models: int = 2,
    ctx: harness__panel.ReviewContext | None = None,
) -> harness__panel.PanelResult:
    return harness__panel.aggregate_verdicts(
        seats,
        ctx or _ctx(),
        block_on=block_on,
        min_blocking_models=min_blocking_models,
        panel_id="p",
    )


# --- diff grounding primitives ------------------------------------------------


def test_diff_hunks_parses_paths_and_both_sides_of_each_hunk() -> None:
    hunks = review.diff_hunks(SAMPLE_DIFF)
    # A created file has no base-side location.
    assert hunks[("foo.py", "head")] == [review.Hunk(old=(10, 12), new=(10, 14))]
    assert hunks[("bar.py", "head")] == [review.Hunk(old=None, new=(1, 2))]


def test_is_grounded_line_in_range_path_only_and_misses() -> None:
    ranges = review.diff_hunks(SAMPLE_DIFF)
    assert harness__panel.is_grounded(_location("foo.py", 11), ranges)  # inside 10..14
    assert harness__panel.is_grounded(_location("foo.py"), ranges)  # path-only, file touched
    assert not harness__panel.is_grounded(
        _location("foo.py", 99), ranges
    )  # outside the touched range
    assert not harness__panel.is_grounded(_location("other.py", 1), ranges)  # file not in the diff


def test_a_colon_in_the_path_is_not_a_line_separator() -> None:
    hunks = review.diff_hunks("--- a/src/a:b.py\n+++ b/src/a:b.py\n@@ -1 +1 @@\n-x\n+y\n")
    assert harness__panel.is_grounded(_location("src/a:b.py", 1), hunks)
    assert not harness__panel.is_grounded(_location("src/a", 1), hunks)


# --- executable grounding in aggregation --------------------------------------


def test_grounded_security_block_gates_under_veto() -> None:
    res = _agg([_seat("m1", _block("security", _location("foo.py", 11)))], block_on="any")
    assert res.blocked is True and res.n_block == 1
    assert res.merged_findings[0].severity == "block"


def test_ungrounded_block_is_not_reported_and_does_not_gate() -> None:
    # A finding at a line absent from the reviewed tree is not reportable.
    res = _agg([_seat("m1", _block("security", _location("foo.py", 99)))], block_on="any")
    assert res.blocked is False and res.n_block == 0
    assert res.merged_findings == ()


def test_non_gating_category_block_is_downgraded() -> None:
    # a "test-gap" can never block even if grounded in the diff.
    res = _agg([_seat("m1", _block("test-gap", _location("foo.py", 11)))], block_on="any")
    assert res.blocked is False
    assert res.merged_findings[0].severity == "warn"


def test_verify_uncovered_requires_verify_passed() -> None:
    f = _block("verify-uncovered-correctness", _location("foo.py", 11))
    # verify failed (or unknown) -> the claim is incoherent -> downgraded.
    assert _agg([_seat("m1", f)], ctx=_ctx(verify_ok=False)).blocked is False
    assert _agg([_seat("m1", f)], ctx=_ctx(verify_ok=None)).blocked is False
    # verify passed -> a grounded uncovered-correctness block may gate.
    assert _agg([_seat("m1", f)], ctx=_ctx(verify_ok=True)).blocked is True


# --- block_on policies --------------------------------------------------------


def test_advisory_never_blocks_even_with_grounded_security_block() -> None:
    res = _agg([_seat("m1", _block("security", _location("foo.py", 11)))], block_on="none")
    assert res.blocked is False
    assert res.merged_findings[0].severity == "block"  # still reported, just not gating


def test_quorum_counts_distinct_models_not_seats() -> None:
    g = lambda: _block("security", _location("foo.py", 11))  # noqa: E731
    # two blocking seats but the SAME model -> counts as one -> min_blocking_models(2) unmet.
    same = [_seat("m1", g(), seat="a"), _seat("m1", g(), seat="b")]
    assert _agg(same, block_on="model_quorum", min_blocking_models=2).blocked is False
    assert _agg(same, block_on="model_quorum", min_blocking_models=2).n_block == 1
    # distinct models -> min_blocking_models met.
    diff_models = [_seat("m1", g(), seat="a"), _seat("m2", g(), seat="b")]
    assert _agg(diff_models, block_on="model_quorum", min_blocking_models=2).blocked is True


def test_all_requires_every_non_abstaining_seat_to_block() -> None:
    blk = _seat("m1", _block("security", _location("foo.py", 11)), seat="a")
    passing = _seat("m2", seat="b")  # no findings -> pass
    assert _agg([blk, passing], block_on="all").blocked is False
    assert _agg(
        [blk, _seat("m2", _block("security", _location("bar.py", 1)), seat="b")], block_on="all"
    ).blocked


def test_all_lone_blocker_with_mostly_errored_panel_does_not_gate() -> None:
    # FINDING 1 regression: a 5-seat panel where 4 seats abstained (provider
    # error) and ONE seat blocks must NOT gate under "all" -- "all" means the
    # panel that actually reviewed unanimously agreed, and one vote is not a
    # min_blocking_models of five. Previously this blocked because abstentions were filtered
    # out before the all(...) check, so all(...) ran over the single survivor.
    blk = _seat("m1", _block("security", _location("foo.py", 11)), seat="a")
    errs = [_seat(f"m{i}", seat=f"s{i}", error="provider timeout") for i in range(2, 6)]
    res = _agg([blk, *errs], block_on="all")
    assert res.n_abstain == 4 and res.n_block == 1
    assert res.blocked is False  # no majority min_blocking_models responded
    # the same lone grounded block DOES gate under veto (one block is enough there).
    assert _agg([blk, *errs], block_on="any").blocked is True


def test_abstain_does_not_count_as_pass_or_block() -> None:
    blk = _seat("m1", _block("security", _location("foo.py", 11)), seat="a")
    err = _seat("m2", seat="b", error="provider timeout")
    err2 = _seat("m3", seat="c", error="unparseable")
    # "all" must NOT gate when only a minority responded: a lone blocker with
    # everyone else abstaining is not unanimous agreement of the panel.
    res = _agg([blk, err], block_on="all")  # 1 of 2 responded -> no majority min_blocking_models
    assert res.n_abstain == 1
    assert res.blocked is False
    # even more lopsided: one block, two abstentions -> still no gate.
    res3 = _agg([blk, err, err2], block_on="all")
    assert res3.n_abstain == 2 and res3.blocked is False
    # "all" means every configured seat, so an abstention prevents a gate.
    blk2 = _seat("m2", _block("security", _location("bar.py", 1)), seat="b")
    res4 = _agg([blk, blk2, err2], block_on="all")
    assert res4.n_abstain == 1 and res4.blocked is False
    # under veto, a single grounded block still gates regardless of abstentions.
    assert _agg([blk, err], block_on="any").blocked is True
    # an all-abstain panel never blocks
    only_err = _agg([err], block_on="any")
    assert only_err.blocked is False and only_err.n_abstain == 1


# --- dedup / rendering --------------------------------------------------------


def test_dedup_across_seats_and_against_prior_findings() -> None:
    f = _block("security", _location("foo.py", 11))
    res = _agg([_seat("m1", f, seat="a"), _seat("m2", f, seat="b")], block_on="none")
    assert len(res.merged_findings) == 1  # same (path, category) merged
    prior = (f,)
    res2 = harness__panel.aggregate_verdicts(
        [_seat("m1", f)],
        _ctx(prior_findings=prior),
        block_on="none",
        min_blocking_models=2,
        panel_id="p",
    )
    assert res2.merged_findings == ()  # already injected -> not re-surfaced


def test_prior_deduped_block_does_not_count_toward_the_gate() -> None:
    # A seat whose only surviving block dedups away against prior_findings must
    # not gate: otherwise blocked=True ships with merged_findings=() and the
    # worker is rejected while being told "No blocking findings.".
    f = _block("security", _location("foo.py", 11))
    prior = (f,)
    for block_on in ("any", "model_quorum", "all"):
        res = harness__panel.aggregate_verdicts(
            [_seat("m1", f)],
            _ctx(prior_findings=prior),
            block_on=block_on,
            min_blocking_models=1,
            panel_id="p",
        )
        assert res.merged_findings == ()
        assert res.n_block == 0 and res.blocked is False, block_on
    # A NEW grounded block alongside the deduped one still gates.
    new = _block("data-loss", _location("bar.py", 1))
    res2 = harness__panel.aggregate_verdicts(
        [_seat("m1", f, new)],
        _ctx(prior_findings=prior),
        block_on="any",
        min_blocking_models=2,
        panel_id="p",
    )
    assert res2.blocked is True and res2.n_block == 1
    assert [x.category for x in res2.merged_findings] == ["data-loss"]


def test_two_defects_at_the_same_location_remain_distinct() -> None:
    """Keep different finding content at the same line, category and severity."""
    first = _block("security", _location("foo.py", 11))
    second = first.model_copy(update={"title": "a different defect"})
    result = _agg([_seat("m1", first, second)])
    assert result.merged_findings == (first, second)


def test_normalized_exact_repeat_is_not_reinjected() -> None:
    """Ignore surrounding whitespace without combining different finding text."""
    first = _block("security", _location("foo.py", 11))
    repeat = first.model_copy(update={"title": " x \n"})
    result = _agg([_seat("m1", repeat)], ctx=_ctx(prior_findings=(first,)))
    assert result.merged_findings == ()
    assert not result.blocked


def test_render_findings_formats_and_empty() -> None:
    assert harness__panel.render_findings(()) == ""
    out = harness__panel.render_findings(
        (
            review_records.Finding(
                category="security",
                severity="block",
                location=_location("foo.py", 11),
                title="leak",
                detail="fix it",
            ),
        )
    )
    assert "[block:security]" in out and "foo.py:11" in out and "leak" in out and "fix it" in out


def test_render_findings_distinguishes_base_and_head_locations() -> None:
    """Name the base side when both sides cite the same path and line."""
    findings = (
        _block("security", _location("foo.py", 11, side="base")),
        _block("security", _location("foo.py", 11)),
    )
    assert harness__panel.render_findings(findings) == (
        "- [block:security] (foo.py:11 (base)) x\n- [block:security] (foo.py:11) x"
    )


def test_a_pass_verdict_cannot_contribute_a_blocking_vote() -> None:
    finding = _block("security", _location("foo.py", 11))
    contradictory = review_records.ReviewVerdict(
        seat_index=0, focus="s", model_route="m1", verdict="pass", findings=(finding,)
    )
    res = _agg([contradictory], block_on="any")
    assert res.blocked is False and res.n_block == 0


# --- diff-parsing edge cases (regressions fixed in the pre-squash review) ------


def test_added_line_starting_like_a_header_is_not_a_file_header() -> None:
    # An added line whose CONTENT begins with "++ b/evil.py" renders as
    # "+++ b/evil.py"; it must not be mistaken for a +++ header (only a +++ that
    # follows a --- is one). A LATER hunk follows so the misparse would actually
    # re-attribute a range to "evil.py" if the prev_minus guard were dropped.
    diff = (
        "--- a/foo.py\n+++ b/foo.py\n"
        "@@ -1,2 +1,3 @@\n keep\n+++ b/evil.py\n+real = 1\n"
        "@@ -10,2 +10,3 @@\n ctx\n+added\n more\n"
    )
    ranges = review.diff_hunks(diff)
    assert ("evil.py", "head") not in ranges
    assert ranges[("foo.py", "head")] == [
        review.Hunk(old=(1, 2), new=(1, 3)),
        review.Hunk(old=(10, 11), new=(10, 12)),
    ]


def test_deleted_line_starting_like_a_header_is_not_a_file_header() -> None:
    # A DELETED line whose CONTENT begins with "-- " renders as "--- ..." and must
    # not be mistaken for a "--- " file header (the symmetric "+++ " side was
    # already guarded; the "--- " side was not). Otherwise it clobbers the path
    # and every LATER hunk's range is mis-attributed, so a grounded block citing a
    # real line in a later hunk is silently downgraded to a warning.
    diff = (
        "--- a/schema.sql\n+++ b/schema.sql\n"
        "@@ -10,3 +10,2 @@\n CREATE TABLE t (\n--- legacy column note\n   id INT\n"
        "@@ -50,2 +50,3 @@\n cols\n+  api_key TEXT\n more\n"
    )
    ranges = review.diff_hunks(diff)
    assert ("legacy column note", "base") not in ranges  # the deletion was not read as a header
    assert harness__panel.is_grounded(
        _location("schema.sql", 51), ranges
    )  # the later hunk still grounds


def test_in_place_modification_grounds_old_side_lines() -> None:
    # A hunk that deletes lines from a kept (not renamed) file: a block citing
    # the deleted code at its OLD line number grounds, so a review of deleted
    # code can gate.
    diff = "--- a/mod.py\n+++ b/mod.py\n@@ -100,5 +50,2 @@\n ctx\n-gone1\n-gone2\n-gone3\n ctx2\n"
    ranges = review.diff_hunks(diff)
    assert [h.old for h in ranges[("mod.py", "base")]] == [
        (100, 104)
    ]  # old side of the in-place hunk
    assert harness__panel.is_grounded(_location("mod.py", 103, side="base"), ranges)
    res = _agg(
        [_seat("m1", _block("data-loss", _location("mod.py", 103, side="base")))],
        block_on="any",
        ctx=harness__panel.ReviewContext(diff=diff),
    )
    assert res.blocked is True and res.n_block == 1
    assert res.merged_findings[0].severity == "block"


def test_pure_deletion_grounds_on_the_old_path() -> None:
    # A file deleted entirely (post-image /dev/null) still grounds a citation
    # of the deleted file, so a data-loss block on it can gate.
    diff = "--- a/gone.py\n+++ /dev/null\n@@ -1,3 +0,0 @@\n-a\n-b\n-c\n"
    ranges = review.diff_hunks(diff)
    assert ranges[("gone.py", "base")] == [review.Hunk(old=(1, 3), new=None)]
    assert harness__panel.is_grounded(_location("gone.py", 2, side="base"), ranges)


def test_grounding_accepts_a_single_line_and_an_inclusive_range() -> None:
    ranges = review.diff_hunks(SAMPLE_DIFF)
    assert harness__panel.is_grounded(_location("foo.py", 11), ranges)
    assert harness__panel.is_grounded(
        _location("foo.py", 11, 13), ranges
    )  # a range fully inside 10..14


def test_grounding_range_overlap_not_just_start_line() -> None:
    # foo.py changed lines 10..14. A range whose START line is unchanged but whose
    # INTERIOR overlaps the touched range must still ground (FINDING 2 regression:
    # previously only the start line was checked, so this was wrongly ungrounded).
    ranges = review.diff_hunks(SAMPLE_DIFF)
    assert harness__panel.is_grounded(
        _location("foo.py", 8, 12), ranges
    )  # start 8 untouched, but 10..12 overlap
    assert harness__panel.is_grounded(
        _location("foo.py", 13, 20), ranges
    )  # end 20 untouched, but 13..14 overlap
    assert harness__panel.is_grounded(
        _location("foo.py", 1, 99), ranges
    )  # span fully contains the touched range
    assert harness__panel.is_grounded(
        _location("foo.py", 14, 30), ranges
    )  # touches only the last changed line
    assert not harness__panel.is_grounded(
        _location("foo.py", 1, 9), ranges
    )  # entirely before the touched range
    assert not harness__panel.is_grounded(
        _location("foo.py", 15, 30), ranges
    )  # entirely after the touched range


def test_a_citation_under_a_real_top_level_a_or_b_dir_grounds() -> None:
    """A location keeps a real top-level `a/` or `b/` directory in its path."""
    diff = "--- a/a/foo.py\n+++ b/a/foo.py\n@@ -1,2 +1,2 @@\n x\n-y\n+z\n"
    ranges = review.diff_hunks(diff)
    assert list(ranges) == [("a/foo.py", "base"), ("a/foo.py", "head")]
    assert harness__panel.is_grounded(
        _location("a/foo.py", 2), ranges
    ) and harness__panel.is_grounded(_location("a/foo.py"), ranges)
    assert not harness__panel.is_grounded(_location("foo.py", 2), ranges)  # a different file
    res = _agg(
        [_seat("m1", _block("security", _location("a/foo.py", 2)))],
        ctx=harness__panel.ReviewContext(diff=diff),
    )
    assert res.blocked is True and res.merged_findings[0].severity == "block"
    # Repository paths are literal, not diff-header paths.
    assert not harness__panel.is_grounded(_location("b/foo.py", 11), review.diff_hunks(SAMPLE_DIFF))


def test_range_block_with_unchanged_start_still_gates() -> None:
    # End-to-end: a grounded block citing a real range whose start line is
    # unchanged must keep blocking (not be silently downgraded to warn).
    res = _agg([_seat("m1", _block("security", _location("foo.py", 8, 12)))], block_on="any")
    assert res.blocked is True and res.n_block == 1
    assert res.merged_findings[0].severity == "block"


def test_all_abstain_panel_prints_inconclusive_not_pass(
    monkeypatch: Any, capsys: Any, tmp_path: pathlib.Path
) -> None:
    """An all-abstain panel prints INCONCLUSIVE, never PASS.

    Three seats, three abstains and real dollars spent produce no review; "0 blocking" is not a
    verdict. `run_panel` already short-circuits on an all-abstain panel; the printed verdict is the
    CLI's, so this pins the CLI.
    """
    from agent6 import budget
    from agent6.config import Config
    from agent6.providers import TranscriptSink
    from agent6.ui.cli import review_cmds

    class _Seat:
        focus = "security"
        scope = "diff"
        model = "test"
        temperature = 0.0

    abstain = review_records.ReviewVerdict(
        seat_index=0,
        focus="security",
        model_route="moonshotai/kimi-k3",
        verdict="abstain",
        error="unparseable reviewer output",
    )
    res = harness__panel.PanelResult(
        panel_id="cli",
        block_on="none",
        blocked=False,
        merged_findings=(),
        per_seat=(abstain, abstain, abstain),
        n_block=0,
        n_abstain=3,
    )

    def _fake_seats(*_a: object, **_k: object) -> list[_Seat]:
        return [_Seat(), _Seat(), _Seat()]

    def _fake_panel(*_a: object, **_k: object) -> harness__panel.PanelResult:
        return res

    monkeypatch.setattr(providers, "build_review_seats", _fake_seats)
    monkeypatch.setattr(harness__reviewer, "run_panel", _fake_panel)
    rc = review_cmds._run_review_panel(  # pyright: ignore[reportPrivateUsage]
        Config(),
        git="git",
        root=pathlib.Path.cwd(),
        base="main",
        head="topic",
        label="main..topic",
        diff="d",
        rules=review_records.ReviewRules(),
        target=review_records.ReviewTarget("b", "bt", "h", "ht", "h", (), "revisions"),
        journal=mock.MagicMock(),
        panel_id="cli",
        reviewers=3,
        focus="security,correctness,tests",
        transcript_sink=TranscriptSink(tmp_path),
        reviews_dir=tmp_path / "reviews",
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1),
    )
    out, err = capsys.readouterr()
    assert "main..topic" in err
    assert "VERDICT: PASS" not in out
    assert "INCONCLUSIVE" in out
    assert "abstained" in out  # the why is on the verdict line, not buried in stderr
    assert rc == 1  # a panel that reviewed nothing is not a success
    (record_path,) = tuple(tmp_path.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["response"]["body"]["stdout"] == out


def test_review_exit_code_is_consistent_across_verdicts(
    monkeypatch: Any, capsys: Any, tmp_path: pathlib.Path
) -> None:
    """The exit code carries the verdict consistently: PASS 0, INCONCLUSIVE 1, BLOCK 2.

    A BLOCK left at 0 lets a CI gate pass a security block while failing on 'nothing reviewed'.
    """
    from agent6 import budget
    from agent6.config import Config
    from agent6.ui.cli import review_cmds

    class _Seat:
        focus = "security"
        scope = "diff"
        model = "test"
        temperature = 0.0

    block = _seat("m1", _block("security", _location("foo.py", 11)))  # a real gating verdict
    passing = _seat("m2", seat="s2")

    def run(
        per_seat: tuple[review_records.ReviewVerdict, ...],
        *,
        blocked: bool,
        findings: tuple[Any, ...],
    ) -> int:
        res = harness__panel.PanelResult(
            panel_id="cli",
            block_on="any",
            blocked=blocked,
            merged_findings=findings,
            per_seat=per_seat,
            n_block=1 if blocked else 0,
            n_abstain=sum(1 for v in per_seat if v.error),
        )

        def _seats(*_a: object, **_k: object) -> list[_Seat]:
            return [_Seat()]

        def _panel(*_a: object, **_k: object) -> harness__panel.PanelResult:
            return res

        monkeypatch.setattr(providers, "build_review_seats", _seats)
        monkeypatch.setattr(harness__reviewer, "run_panel", _panel)
        rc = review_cmds._run_review_panel(  # pyright: ignore[reportPrivateUsage]
            Config(),
            git="git",
            root=pathlib.Path.cwd(),
            base="",
            head="HEAD",
            label="working tree vs HEAD",
            diff="d",
            rules=review_records.ReviewRules(),
            target=review_records.ReviewTarget("b", "bt", "h", "ht", "h", (), "revisions"),
            journal=mock.MagicMock(),
            panel_id="cli",
            reviewers=1,
            focus="security",
            transcript_sink=mock.MagicMock(),
            reviews_dir=tmp_path,
            budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1),
        )
        capsys.readouterr()
        return rc

    assert run((passing,), blocked=False, findings=()) == 0  # PASS
    assert run((block,), blocked=True, findings=block.findings) == 4  # BLOCK, run's 4
    abstain = _seat("m3", seat="s3", error="starved")
    assert run((abstain,), blocked=False, findings=()) == 1  # INCONCLUSIVE


def test_output_cap_truncated_case_folds_both_spellings() -> None:
    from agent6.providers import ProviderResponse, output_cap_truncated

    def r(stop: str) -> ProviderResponse:
        return ProviderResponse(
            text="",
            tool_uses=(),
            stop_reason=stop,
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )

    assert output_cap_truncated(r("length"))
    assert output_cap_truncated(r("max_tokens"))  # anthropic spelling
    assert output_cap_truncated(r("MAX_TOKENS"))  # upper-casing gateway
    assert not output_cap_truncated(r("end_turn"))
    assert not output_cap_truncated(r("tool_use"))


def test_panel_is_inconclusive_owner() -> None:
    abstain = review_records.ReviewVerdict(
        seat_index=0, focus="s", model_route="m", verdict="abstain", error="starved"
    )
    passing = review_records.ReviewVerdict(
        seat_index=0, focus="s2", model_route="m2", verdict="pass"
    )
    all_abstain = harness__panel.PanelResult(
        panel_id="p",
        block_on="none",
        blocked=False,
        merged_findings=(),
        per_seat=(abstain, abstain),
        n_block=0,
        n_abstain=2,
    )
    assert harness__panel.panel_is_inconclusive(all_abstain) is True
    mixed = harness__panel.PanelResult(
        panel_id="p",
        block_on="none",
        blocked=False,
        merged_findings=(),
        per_seat=(abstain, passing),
        n_block=0,
        n_abstain=1,
    )
    assert harness__panel.panel_is_inconclusive(mixed) is False
    empty = harness__panel.PanelResult(
        panel_id="p",
        block_on="none",
        blocked=False,
        merged_findings=(),
        per_seat=(),
        n_block=0,
        n_abstain=0,
    )
    assert harness__panel.panel_is_inconclusive(empty) is False  # nothing to be inconclusive about


def test_the_panel_refuses_unreadable_base_rules(
    monkeypatch: Any, tmp_path: pathlib.Path, capsys: Any
) -> None:
    from agent6 import errors
    from agent6.config import Config
    from agent6.ui.cli import review_cmds
    from agent6.vcs import git_files
    from tests.unit import test_review_saved

    test_review_saved._repo_with_a_change(tmp_path)
    (tmp_path / "REVIEW.md").write_text("rule")
    test_review_diff_collect._git(tmp_path, "add", "REVIEW.md")
    test_review_diff_collect._git(tmp_path, "commit", "-qm", "rules")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(layer, "load_effective", mock.Mock(return_value=mock.Mock(config=Config())))
    monkeypatch.setattr(
        git_files, "blob_at", mock.Mock(side_effect=errors.GitError("unreadable object"))
    )
    panel = mock.Mock()
    monkeypatch.setattr(review_cmds, "_run_review_panel", panel)
    assert review_cmds._cmd_review((), base="", head="HEAD", paths=(), reviewers=1) == 2
    assert "REVIEW.md: unreadable object" in capsys.readouterr().err
    panel.assert_not_called()


def test_the_panel_reviews_under_the_freeform_reviews_label(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """One label for both reviews.

    The panel's header and transcript named the range alone, dropping the `-- paths` the freeform
    review's label carries.
    """
    import types

    from agent6.config import Config
    from agent6.ui.cli import review_cmds

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    seen: dict[str, str] = {}

    def _fake_panel(_cfg: Config, **kwargs: Any) -> int:
        seen["label"] = kwargs["label"]
        return 0

    def _fake_effective(*_a: object, **_k: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(config=Config())

    def _runnable(_self: Config, _role: str) -> None:
        return None

    def _no_key_error(_cfg: Config) -> None:
        return None

    monkeypatch.setattr(layer, "load_effective", _fake_effective)
    monkeypatch.setattr(Config, "require_runnable", _runnable)
    monkeypatch.setattr(_setup, "check_provider_keys", _no_key_error)
    monkeypatch.setattr(review_cmds, "_run_review_panel", _fake_panel)
    test_review_diff_collect._git(tmp_path, "init", "-q", "-b", "main")
    test_review_diff_collect._git(tmp_path, "config", "user.name", "t")
    test_review_diff_collect._git(tmp_path, "config", "user.email", "t@t")
    test_review_diff_collect._git(tmp_path, "commit", "--allow-empty", "-qm", "init")
    start = test_review_diff_collect._git(tmp_path, "rev-parse", "HEAD").strip()
    (tmp_path / "src").mkdir()
    (tmp_path / "src/x.py").write_text("x = 1\n")
    test_review_diff_collect._git(tmp_path, "checkout", "-qb", "topic")
    test_review_diff_collect._git(tmp_path, "add", "src/x.py")
    test_review_diff_collect._git(tmp_path, "commit", "-qm", "change")
    rc = review_cmds._cmd_review(  # pyright: ignore[reportPrivateUsage]
        (), base="main", head="HEAD", paths=("src/x.py",), reviewers=1
    )
    assert rc == 0
    assert seen["label"] == f"main...HEAD -- src/x.py (from {start})"
    captured = capsys.readouterr()
    assert f"[agent6] reviewing: main...HEAD -- src/x.py (from {start})\n" in captured.err
    (tmp_path / "src/x.py").write_text("x = 2\n")
    rc = review_cmds._cmd_review(  # pyright: ignore[reportPrivateUsage]
        (), base="", head="HEAD", paths=("src/x.py",), reviewers=1
    )
    assert rc == 0
    assert seen["label"] == "working tree vs HEAD -- src/x.py"


def test_diff_touched_ranges_records_a_file_touched_without_hunks() -> None:
    """A file touched without hunks is recorded with no ranges: grounded by path, not by line.

    A binary change, a pure rename and a mode flip carry no hunks.
    """
    diff = (
        "diff --git a/img.png b/img.png\n"
        "index 1111111..2222222 100644\n"
        "Binary files a/img.png and b/img.png differ\n"
        "diff --git a/old.py b/new.py\n"
        "similarity index 100%\n"
        "rename from old.py\n"
        "rename to new.py\n"
        "diff --git a/run.sh b/run.sh\n"
        "old mode 100644\n"
        "new mode 100755\n"
    )
    ranges = review.diff_hunks(diff)
    assert ranges == {
        ("img.png", "base"): [],
        ("img.png", "head"): [],
        ("old.py", "base"): [],
        ("new.py", "head"): [],
        ("run.sh", "base"): [],
        ("run.sh", "head"): [],
    }
    assert harness__panel.is_grounded(_location("img.png"), ranges) and harness__panel.is_grounded(
        _location("new.py"), ranges
    )
    assert not harness__panel.is_grounded(_location("img.png", 3), ranges)


def test_the_seat_prompt_says_verify_was_not_run_without_a_result() -> None:
    """The seat prompt says verify was not run when there is no result.

    `agent6 review` runs no verify command and the loop has none when none is configured; "none
    configured" is wrong for a review of a repo that has one.
    """
    from agent6.harness import _review_seat  # pyright: ignore[reportPrivateUsage]

    prompt = _review_seat.build_user_message(harness__panel.ReviewContext(task="t"))
    assert "VERIFY: not run." in prompt and "none configured" not in prompt


def test_prior_findings_notice_distinguishes_base_and_head_locations() -> None:
    """Keep the prior findings' sides visible at the same path and line."""
    from agent6.harness import _review_seat  # pyright: ignore[reportPrivateUsage]

    prior = (
        _block("security", _location("foo.py", 11, side="base")),
        _block("security", _location("foo.py", 11)),
    )
    prompt = _review_seat.build_user_message(_ctx(prior_findings=prior))
    assert "ALREADY RAISED (do not repeat): foo.py:11 (base) security; foo.py:11 security" in prompt


def test_the_seat_prompt_carries_the_whole_large_diff() -> None:
    from agent6.harness import _review_seat  # pyright: ignore[reportPrivateUsage]

    diff = "start\n" + "x" * 200_000 + "\nend"
    prompt = _review_seat.build_user_message(harness__panel.ReviewContext(task="t", diff=diff))
    assert prompt.endswith(f"DIFF:\n{diff}")


# A second hunk in foo.py, well away from the first.
TWO_HUNKS_DIFF = SAMPLE_DIFF.replace(
    "--- /dev/null\n",
    "@@ -40,2 +42,3 @@ def g():\n     a = 1\n+    b = 2\n     return a\n--- /dev/null\n",
    1,
)


def test_two_findings_in_different_hunks_of_one_file_both_survive() -> None:
    """The dedup key was (path, category).

    A second finding in another hunk of the same file was dropped as a duplicate of the first, and
    the report never said it existed. The key carries the hunk.
    """
    ctx = harness__panel.ReviewContext(diff=TWO_HUNKS_DIFF)
    seat = _seat(
        "m1",
        _block("security", _location("foo.py", 11)),
        _block("security", _location("foo.py", 43)),
    )
    res = _agg([seat], ctx=ctx)
    assert [str(f.location) for f in res.merged_findings] == ["foo.py:11", "foo.py:43"]


def test_distinct_locations_in_one_hunk_remain_distinct() -> None:
    """Keep uncertain matches separate, including nearby lines in the same hunk."""
    a = _block("security", _location("foo.py", 11))
    b = _block("security", _location("foo.py", 13))
    res = _agg([_seat("m1", a), _seat("m2", b, seat="t")])
    assert len(res.merged_findings) == 2
    res = _agg([_seat("m1", b)], ctx=_ctx(prior_findings=(a,)))
    assert res.merged_findings == (b,) and res.blocked
    res = _agg([_seat("m1", a, _block("security", _location("foo.py")))])
    assert len(res.merged_findings) == 2


def test_side_and_location_distinguish_findings_in_overlapping_hunks() -> None:
    """Keep findings on different sides separate even when their line ranges overlap."""
    diff = (
        "--- a/foo.py\n+++ b/foo.py\n"
        "@@ -40,3 +42,3 @@ def f():\n     a = 1\n-    b = 2\n+    b = 3\n     return a\n"
        "--- a/bar.py\n+++ b/bar.py\n"
        "@@ -10,50 +10,2 @@ def g():\n     x = 1\n" + "-    gone\n" * 48 + "     return x\n"
        "@@ -70,3 +23,4 @@ def h():\n     y = 1\n+    z = 2\n     return y\n"
    )
    ctx = harness__panel.ReviewContext(diff=diff)
    pre, post = (
        _block("security", _location("foo.py", 40, side="base")),
        _block("security", _location("foo.py", 42)),
    )
    res = _agg([_seat("m1", post)], ctx=dataclasses.replace(ctx, prior_findings=(pre,)))
    assert res.merged_findings == (post,) and res.blocked
    seat = _seat(
        "m1",
        _block("security", _location("bar.py", 23)),
        _block("security", _location("bar.py", 55, side="base")),
    )
    res = _agg([seat], ctx=ctx)
    assert [str(f.location) for f in res.merged_findings] == ["bar.py:23", "bar.py:55 (base)"]


def test_a_renames_two_names_do_not_ground_each_others_lines() -> None:
    """A rename's two names do not ground each other's lines.

    One hunk filed under both the pre-image and the post-image path lets each name ground the
    other's line numbers (`old.py:201`, `new.py:11`), and the gate fails open. Each name carries its
    own side.
    """
    diff = "--- a/old.py\n+++ b/new.py\n@@ -10,3 +200,3 @@\n a\n-b\n+B\n c\n"
    hunks = review.diff_hunks(diff)
    assert hunks == {
        ("new.py", "head"): [review.Hunk(old=None, new=(200, 202))],
        ("old.py", "base"): [review.Hunk(old=(10, 12), new=None)],
    }
    assert harness__panel.is_grounded(
        _location("old.py", 11, side="base"), hunks
    ) and harness__panel.is_grounded(_location("new.py", 201), hunks)
    assert not harness__panel.is_grounded(
        _location("old.py", 201, side="base"), hunks
    ) and not harness__panel.is_grounded(_location("new.py", 11), hunks)
    # A pure deletion inside a kept file still grounds the post-image line it
    # sits at, and a pure insertion the pre-image one.
    kept = review.diff_hunks("--- a/k.py\n+++ b/k.py\n@@ -10,5 +9,0 @@\n-a\n-b\n-c\n-d\n-e\n")
    assert kept[("k.py", "head")] == [review.Hunk(old=(10, 14), new=(9, 9))]
    assert harness__panel.is_grounded(_location("k.py", 9), kept) and harness__panel.is_grounded(
        _location("k.py", 12, side="base"), kept
    )


def test_file_level_locations_keep_the_rename_side() -> None:
    hunks = review.diff_hunks(
        "diff --git a/old.py b/new.py\n"
        "similarity index 100%\nrename from old.py\nrename to new.py\n"
    )
    assert harness__panel.is_grounded(_location("old.py", side="base"), hunks)
    assert harness__panel.is_grounded(_location("new.py"), hunks)
    assert not harness__panel.is_grounded(_location("old.py"), hunks)
    assert not harness__panel.is_grounded(_location("new.py", side="base"), hunks)


def test_file_level_locations_exclude_absent_sides() -> None:
    hunks = review.diff_hunks(
        "diff --git a/new.bin b/new.bin\nnew file mode 100644\n"
        "Binary files /dev/null and b/new.bin differ\n"
        "diff --git a/old.bin b/old.bin\ndeleted file mode 100644\n"
        "Binary files a/old.bin and /dev/null differ\n"
    )
    assert harness__panel.is_grounded(_location("new.bin"), hunks)
    assert harness__panel.is_grounded(_location("old.bin", side="base"), hunks)
    assert not harness__panel.is_grounded(_location("new.bin", side="base"), hunks)
    assert not harness__panel.is_grounded(_location("old.bin"), hunks)


def test_type_changed_path_keeps_both_sides_hunks() -> None:
    """Keep the deleted symlink's base hunk when its regular-file addition follows."""
    diff = (
        "diff --git a/config b/config\n"
        "deleted file mode 120000\n"
        "index 1de5659..0000000\n"
        "--- a/config\n+++ /dev/null\n"
        "@@ -1 +0,0 @@\n-config.default\n\\ No newline at end of file\n"
        "diff --git a/config b/config\n"
        "new file mode 100644\n"
        "index 0000000..3b5d5c3\n"
        "--- /dev/null\n+++ b/config\n"
        "@@ -0,0 +1,2 @@\n+enabled = true\n+timeout = 30\n"
    )
    hunks = review.diff_hunks(diff)
    assert hunks == {
        ("config", "base"): [review.Hunk(old=(1, 1), new=None)],
        ("config", "head"): [review.Hunk(old=None, new=(1, 2))],
    }
    assert harness__panel.is_grounded(_location("config", 1, side="base"), hunks)
    assert harness__panel.is_grounded(_location("config", 2), hunks)
    assert not harness__panel.is_grounded(_location("config", 2, side="base"), hunks)
    finding = _block("data-loss", _location("config", 1, side="base"))
    result = _agg([_seat("m1", finding)], ctx=harness__panel.ReviewContext(diff=diff))
    assert result.merged_findings == (finding,)
    assert result.blocked and result.n_block == 1


def test_grounding_uses_only_the_cited_side() -> None:
    """A head citation cannot borrow an old line number from the same file."""
    hunks = review.diff_hunks("--- a/mod.py\n+++ b/mod.py\n@@ -100,3 +10,2 @@\n a\n-b\n c\n")
    base = review_records.ReviewLocation(path="mod.py", side="base", start_line=101, end_line=101)
    head = review_records.ReviewLocation(path="mod.py", side="head", start_line=101, end_line=101)
    assert harness__panel.is_grounded(base, hunks)
    assert not harness__panel.is_grounded(head, hunks)


def test_a_review_notice_is_cut_head_first_at_a_character_boundary() -> None:
    """A review notice is cut head first at a character boundary.

    The cut keeps the head (the findings lead), lands inside the byte budget, and names the bytes it
    dropped.
    """
    assert harness__panel.review_notice("short") == "[review]\nshort"
    text = "\u6f22" * 2_000  # three bytes a character
    head, marker = harness__panel.review_notice(text).removeprefix("[review]\n").rsplit("\n", 1)
    assert head == "\u6f22" * (harness__panel.REVIEW_NOTICE_BYTES // 3)
    assert marker == f"[review: {6_000 - len(head.encode())} more bytes cut]"
