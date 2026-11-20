from __future__ import annotations

from agent6 import budget, token_usage


def test_a_continued_child_charges_only_its_new_usage() -> None:
    """Cumulative child readings do not charge its stopped execution twice."""
    tracker = budget.BudgetTracker(max_usd=10, max_tokens_fallback=1000, max_plan_points=-1)
    prior = budget.UsageSettlement(
        usd=2, usd_partial=False, tokens=100, model_tokens=token_usage.TokenCounts(90, 10)
    )
    tracker.settle_child("child", prior)
    tracker.child_execution("child", "child:1", prior)
    later = budget.UsageSettlement(
        usd=3, usd_partial=False, tokens=150, model_tokens=token_usage.TokenCounts(120, 30)
    )
    tracker.note_running("child", later)
    assert tracker.available().usd == 7
    assert tracker.available().tokens == 850
    tracker.settle_child("child", later)
    tracker.settle_child("child", later)
    assert tracker.available().usd == 7
    assert tracker.available().tokens == 850
    assert tracker.snapshot().settled_children == frozenset({"child", "child:1"})
    assert "Children: in=120 out=30" in tracker.format_summary()
    tracker.child_execution("child", "child:2", later)
    tracker.settle_child("child", budget.UsageSettlement(usd=4, usd_partial=False, tokens=200))
    assert tracker.available().usd == 6
    assert tracker.available().tokens == 800
    assert "Children: in=unreported out=unreported" in tracker.format_summary()


def test_recovery_charges_only_the_unaccounted_child_execution() -> None:
    """An earlier settlement key cannot suppress a continued child's new charge."""
    tracker = budget.BudgetTracker(max_usd=10, max_tokens_fallback=1000, max_plan_points=-1)
    prior = budget.UsageSettlement(usd=2, usd_partial=False, tokens=100)
    tracker.mark_settled(["child"])
    tracker.child_execution("child", "child:1", prior)
    tracker.settle_child("child", budget.UsageSettlement(usd=3, usd_partial=False, tokens=150))
    assert tracker.available().usd == 9
    assert tracker.available().tokens == 950
    replay = budget.BudgetTracker(max_usd=10, max_tokens_fallback=1000, max_plan_points=-1)
    replay.mark_settled(["child", "child:1"])
    replay.child_execution("child", "child:1", prior)
    replay.settle_child("child", budget.UsageSettlement(usd=3, usd_partial=False, tokens=150))
    assert replay.available().usd == 10
