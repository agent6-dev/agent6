# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 sessions compare`: advisory verify+judge ranking across already-run candidates.

Real tmp git repos + fabricated run state (manifest.json + logs.jsonl), same fabrication pattern as
test_cli_runs_merge.py (branches) and test_parallel_orchestrator.py (`_write_fake_run`). The judge
path is driven with a fake provider (no network).
"""

from __future__ import annotations

import io
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, cast

import pytest

from agent6 import budget as agent6_budget
from agent6 import paths
from agent6.config import Config
from agent6.harness import judge as harness_judge
from agent6.providers import Provider, ProviderError
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import _compare as compare_mod
from agent6.ui.cli import main
from agent6.viewmodel import format


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_repo(repo: pathlib.Path) -> str:
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return _git(repo, "rev-parse", "HEAD")


def _setup_run(
    repo: pathlib.Path,
    session_id: str,
    *,
    base_sha: str,
    commits: list[tuple[str, str, str]],
    task: str = "implement the thing",
    status: str = "passed",
    cost: float = 0.05,
    manifest_extra: dict[str, Any] | None = None,
) -> None:
    """Cut agent6/<session_id> off base_sha with commits and write the run's state fixture.

    Writes manifest.json and logs.jsonl, then returns the checkout to where it was;
    *manifest_extra* merges extra manifest fields (a fan-out lane's lineage, a compare stamp).
    """
    branch = f"agent6/{session_id}"
    current = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _git(repo, "checkout", "-q", base_sha)
    _git(repo, "checkout", "-q", "-b", branch)
    for name, content, msg in commits:
        (repo / name).write_text(content, encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", msg)
    _git(repo, "checkout", "-q", current)

    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id=session_id)
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": session_id,
                "base_sha": base_sha,
                "base_branch": "main",
                "run_branch": branch,
                "user_task": task,
                **(manifest_extra or {}),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    events: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": task},
        {"type": "budget.update", "usd_total": cost},
    ]
    if status == "passed":
        events.append({"type": "session.end", "reason": "finish_session", "all_passed": True})
    elif status == "failed":
        events.append({"type": "session.end", "reason": "provider_error", "all_passed": False})
    layout.logs_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


@pytest.fixture
def repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    # Run state is isolated by the autouse `_isolate_state` fixture (conftest.py)
    # to a tmp dir OUTSIDE this one; nesting XDG_STATE_HOME under tmp_path here
    # would put untracked run state inside the repo's own working tree, where a
    # second run's `git add -A` sweeps it onto that run's branch and a later
    # checkout back to main deletes it as "not in this branch's tree".
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_compare_needs_at_least_two_ids(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")])
    rc = main(["sessions", "compare", "run-AAAA11"])
    assert rc == 2
    assert "2 or more run ids, or one --parallel fan-out id" in capsys.readouterr().err


def test_compare_of_a_fanout_id_compares_its_lanes(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fan-out id given alone names every lane carrying it, in lane order.

    It read as one run, too few, and `sessions show` called it ambiguous.
    """
    base = _init_repo(repo)
    for lane, cost in ((1, 0.09), (2, 0.01)):
        _setup_run(
            repo,
            f"fan-{lane}",
            base_sha=base,
            commits=[(f"{lane}.txt", "x\n", f"add {lane}")],
            cost=cost,
            manifest_extra={"parallel": {"group": "fan", "lane": lane, "coordinator": "fan"}},
        )
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")])
    assert main(["sessions", "compare", "fan"]) == 0
    out = capsys.readouterr().out
    assert "comparing 2 runs" in out
    assert "fan-1" in out and "fan-2" in out and "run-AAAA11" not in out


def test_compare_unknown_id_errors_loudly(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")])
    rc = main(["sessions", "compare", "run-AAAA11", "nonexistent"])
    assert rc == 2
    assert "no session matches" in capsys.readouterr().err


def test_compare_ambiguous_id_errors_loudly(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _init_repo(repo)
    _setup_run(repo, "run-DUPXX1", base_sha=base, commits=[("a.txt", "a\n", "add a")])
    _setup_run(repo, "run-DUPXX2", base_sha=base, commits=[("b.txt", "b\n", "add b")])
    _setup_run(repo, "run-CCCC33", base_sha=base, commits=[("c.txt", "c\n", "add c")])
    rc = main(["sessions", "compare", "run-DUP", "run-CCCC33"])
    assert rc == 2
    assert "ambiguous" in capsys.readouterr().err


def test_compare_rejects_duplicate_id(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")])
    _setup_run(repo, "run-BBBB22", base_sha=base, commits=[("b.txt", "b\n", "add b")])
    rc = main(["sessions", "compare", "run-AAAA11", "run-AAAA11"])
    assert rc == 2
    assert "more than once" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Mechanical path (no reviewer model configured)
# ---------------------------------------------------------------------------


def test_compare_prefix_resolution_and_mechanical_ranking(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _init_repo(repo)
    # Cheaper lane fails verify; the other passes -- verify-pass must win despite
    # costing more (mechanical_ranking: verify-pass first, then lower cost).
    _setup_run(
        repo,
        "run-AAAA11",
        base_sha=base,
        commits=[("a.txt", "a\n", "add a")],
        status="failed",
        cost=0.01,
    )
    _setup_run(
        repo,
        "run-BBBB22",
        base_sha=base,
        commits=[("b.txt", "b\n", "add b")],
        status="passed",
        cost=0.09,
    )
    rc = main(["sessions", "compare", "run-AAAA", "run-BBBB22"])  # unique prefix + exact id
    assert rc == 0
    out = capsys.readouterr().out
    assert "ranked candidates" in out
    assert out.index("run-BBBB22") < out.index("run-AAAA11")
    assert "agent6 sessions merge run-BBBB22" in out
    assert "no reviewer model configured" in out
    # Candidate spend is totaled; no judge ran, so no judge figure.
    assert "total: candidates $0.10" in out and "+ judge" not in out


def test_compare_row_of_a_merged_run_says_so(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A candidate already merged is not offered `sessions merge` again; its row names the base."""
    base = _init_repo(repo)
    _setup_run(
        repo,
        "run-MMMM55",
        base_sha=base,
        commits=[("m.txt", "m\n", "add m")],
        status="passed",
        cost=0.01,
        manifest_extra={"merged": {"into": "main", "sha": "abc", "tip": "abc"}},
    )
    _setup_run(
        repo,
        "run-NNNN66",
        base_sha=base,
        commits=[("n.txt", "n\n", "add n")],
        status="passed",
        cost=0.02,
    )
    assert main(["sessions", "compare", "run-MMMM55", "run-NNNN66"]) == 0
    out = capsys.readouterr().out
    merged_row = next(line for line in out.splitlines() if "run-MMMM55" in line)
    assert merged_row.endswith("merged into main")
    other_row = next(line for line in out.splitlines() if "run-NNNN66" in line)
    assert other_row.endswith("merge with: agent6 sessions merge run-NNNN66")


def test_compare_rows_and_total_format_cost_the_same_way(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Per-row and total costs render through the one cost formatter.

    Hand-formatting the rows at four decimals put '$1.5000' above a '$1.52' total.
    """
    base = _init_repo(repo)
    _setup_run(
        repo,
        "run-CCCC33",
        base_sha=base,
        commits=[("c.txt", "c\n", "add c")],
        status="passed",
        cost=1.50,
    )
    _setup_run(
        repo,
        "run-DDDD44",
        base_sha=base,
        commits=[("d.txt", "d\n", "add d")],
        status="passed",
        cost=0.02,
    )
    assert main(["sessions", "compare", "run-CCCC33", "run-DDDD44"]) == 0
    out = capsys.readouterr().out
    assert "$1.5000" not in out
    assert "$1.50" in out
    assert "total: candidates $1.52" in out


def test_compare_excludes_a_run_that_never_finished(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that died before session.end is dropped from a hand-picked comparison, and said so.

    Its truncated spend is the lowest, so mechanical ranking floated it to first place with a
    merge suggestion; the fan-out already excludes such lanes.
    """
    base = _init_repo(repo)
    _setup_run(
        repo,
        "run-EEEE55",
        base_sha=base,
        commits=[("e.txt", "e\n", "add e")],
        status="failed",  # a real verdict, and the more expensive run
        cost=0.09,
    )
    _setup_run(
        repo,
        "run-FFFF66",
        base_sha=base,
        commits=[("f.txt", "f\n", "add f")],
        status="crashed",  # no session.end
        cost=0.01,
    )
    # A recorded-but-dead worker pid is what makes an unfinished run read "stale".
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="run-FFFF66")
    (layout.session_dir / "worker.pid").write_text("999999999", encoding="utf-8")

    assert main(["sessions", "compare", "run-EEEE55", "run-FFFF66"]) == 0
    out = capsys.readouterr().out
    assert "1. run-EEEE55" in out
    assert "1. run-FFFF66" not in out
    assert "agent6 sessions merge run-FFFF66" not in out
    assert "run-FFFF66" in out and "stale" in out  # named as excluded, not hidden


def test_compare_excludes_a_run_that_is_still_live(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live run is dropped from a comparison like a died one.

    Its verify reads None and its spend is truncated to the lowest, so ranking floated the
    half-done run to first place with a merge suggestion for a branch still moving.
    """
    import os

    base = _init_repo(repo)
    _setup_run(
        repo,
        "run-GGGG77",
        base_sha=base,
        commits=[("g.txt", "g\n", "add g")],
        status="failed",
        cost=0.09,
    )
    _setup_run(
        repo,
        "run-HHHH88",
        base_sha=base,
        commits=[("h.txt", "h\n", "add h")],
        status="crashed",  # no session.end...
        cost=0.01,
    )
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="run-HHHH88")
    (layout.session_dir / "worker.pid").write_text(
        str(os.getpid()), encoding="utf-8"
    )  # ...but LIVE

    assert main(["sessions", "compare", "run-GGGG77", "run-HHHH88"]) == 0
    out = capsys.readouterr().out
    assert "1. run-GGGG77" in out
    assert "1. run-HHHH88" not in out
    assert "agent6 sessions merge run-HHHH88" not in out
    assert "run-HHHH88 is still running" in out  # named as excluded, with why


def test_compare_is_read_only(repo: pathlib.Path) -> None:
    """Never merges, never writes to the run's own branch/manifest."""
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")])
    _setup_run(repo, "run-BBBB22", base_sha=base, commits=[("b.txt", "b\n", "add b")])
    head_before = _git(repo, "rev-parse", "main")
    manifest_before = (
        sessions_layout.SessionLayout(
            state_dir=paths.state_dir(repo), session_id="run-AAAA11"
        ).manifest_path
    ).read_text(encoding="utf-8")
    rc = main(["sessions", "compare", "run-AAAA11", "run-BBBB22"])
    assert rc == 0
    assert _git(repo, "rev-parse", "main") == head_before
    assert (
        sessions_layout.SessionLayout(
            state_dir=paths.state_dir(repo), session_id="run-AAAA11"
        ).manifest_path
    ).read_text(encoding="utf-8") == manifest_before


# ---------------------------------------------------------------------------
# Judge path (fake provider, no network)
# ---------------------------------------------------------------------------


def _write_reviewer_config(repo: pathlib.Path) -> None:
    p = paths.repo_config_path(repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        '[providers.anthropic]\napi_format = "anthropic"\napi_key_env = "FAKE_KEY_NOT_SET"\n\n'
        '[models.reviewer]\nprovider = "anthropic"\nmodel = "reviewer-default"\n',
        encoding="utf-8",
    )


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeProvider:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0

    def call(self, **_kw: Any) -> Any:
        self.calls += 1
        return _Resp(self._texts.pop(0))


def _stub_builder(provider: object) -> Any:
    """Stand in for `_build_role_provider` so the judge path needs no API key or network.

    Returns *provider* regardless of the arguments: any object with the fake `.call()` shape,
    cast to `Provider` for the caller.
    """

    def _build(*_a: Any, **_k: Any) -> Provider:
        return cast(Provider, provider)

    return _build


class _CostingFakeProvider(_FakeProvider):
    """A fake provider that bills each call into the BudgetTracker its builder received."""

    budget: agent6_budget.BudgetTracker | None = None

    def call(self, **kw: Any) -> Any:
        assert self.budget is not None
        self.budget.record(
            model="reviewer-default",
            input_tokens=1000,
            output_tokens=100,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            cost_usd=0.0102,
        )
        return super().call(**kw)


def _costing_stub_builder(provider: _CostingFakeProvider) -> Any:
    """`_stub_builder`, but hands the provider the budget it must bill into."""

    def _build(*_a: Any, **kw: Any) -> Provider:
        provider.budget = kw["budget"]
        return cast(Provider, provider)

    return _build


def test_compare_uses_judge_when_reviewer_configured(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")], cost=0.10)
    _setup_run(repo, "run-BBBB22", base_sha=base, commits=[("b.txt", "b\n", "add b")], cost=0.02)
    _write_reviewer_config(repo)
    verdict = '{"ranking": ["run-BBBB22", "run-AAAA11"], "rationale": "b is cleaner"}'
    provider = _FakeProvider([verdict])
    monkeypatch.setattr(compare_mod, "build_role_provider", _stub_builder(provider))

    rc = main(["sessions", "compare", "run-AAAA11", "run-BBBB22"])

    assert rc == 0
    out = capsys.readouterr().out
    assert out.index("run-BBBB22") < out.index("run-AAAA11")
    assert "judge: b is cleaner" in out
    assert "no reviewer model configured" not in out
    assert provider.calls == 1


def test_compare_total_line_accounts_the_judge_calls_own_spend(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The judge call's spend lands on the ranked report's total line."""
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")], cost=0.10)
    _setup_run(repo, "run-BBBB22", base_sha=base, commits=[("b.txt", "b\n", "add b")], cost=0.02)
    _write_reviewer_config(repo)
    verdict = '{"ranking": ["run-BBBB22", "run-AAAA11"], "rationale": "b is cleaner"}'
    provider = _CostingFakeProvider([verdict])
    monkeypatch.setattr(compare_mod, "build_role_provider", _costing_stub_builder(provider))

    rc = main(["sessions", "compare", "run-AAAA11", "run-BBBB22"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "total: candidates $0.12 + judge $0.01 = $0.13" in out


def _lane_extra(*, winner: bool, rank: int) -> dict[str, Any]:
    return {
        "parallel": {"group": "fan", "lane": rank, "coordinator": "fan"},
        "compare": {"rank": rank, "of": 2, "winner": winner, "ranked_by": "judge"},
    }


def test_compare_discloses_a_fresh_verdict_that_contradicts_the_stamp(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-judging a fan-out's lanes never rewrites the stamp, and a flipped winner is said."""
    base = _init_repo(repo)
    _setup_run(
        repo,
        "run-AAAA11",
        base_sha=base,
        commits=[("a.txt", "a\n", "add a")],
        manifest_extra=_lane_extra(winner=False, rank=2),
    )
    _setup_run(
        repo,
        "run-BBBB22",
        base_sha=base,
        commits=[("b.txt", "b\n", "add b")],
        manifest_extra=_lane_extra(winner=True, rank=1),
    )
    _write_reviewer_config(repo)
    # The fresh judge flips the order: stamped winner run-BBBB22 now ranks last.
    verdict = '{"ranking": ["run-AAAA11", "run-BBBB22"], "rationale": "a is cleaner"}'
    monkeypatch.setattr(compare_mod, "build_role_provider", _stub_builder(_FakeProvider([verdict])))

    rc = main(["sessions", "compare", "run-AAAA11", "run-BBBB22"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "note: the recorded fan-out verdict picked run-BBBB22" in out
    assert "nothing was re-stamped" in out


def test_compare_stays_quiet_when_the_fresh_verdict_agrees_with_the_stamp(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _init_repo(repo)
    _setup_run(
        repo,
        "run-AAAA11",
        base_sha=base,
        commits=[("a.txt", "a\n", "add a")],
        manifest_extra=_lane_extra(winner=False, rank=2),
    )
    _setup_run(
        repo,
        "run-BBBB22",
        base_sha=base,
        commits=[("b.txt", "b\n", "add b")],
        manifest_extra=_lane_extra(winner=True, rank=1),
    )
    _write_reviewer_config(repo)
    verdict = '{"ranking": ["run-BBBB22", "run-AAAA11"], "rationale": "b still wins"}'
    monkeypatch.setattr(compare_mod, "build_role_provider", _stub_builder(_FakeProvider([verdict])))

    rc = main(["sessions", "compare", "run-AAAA11", "run-BBBB22"])

    assert rc == 0
    assert "note:" not in capsys.readouterr().out


def test_failed_judge_announces_what_its_attempts_still_spent(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two malformed judge replies fall back to the mechanical ranking with their spend reported.

    The degradation line carries the spend and the mechanical outcome stamps it.
    """
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")], cost=0.10)
    _setup_run(repo, "run-BBBB22", base_sha=base, commits=[("b.txt", "b\n", "add b")], cost=0.02)
    _write_reviewer_config(repo)
    provider = _CostingFakeProvider(["not json at all", "still not json"])
    monkeypatch.setattr(compare_mod, "build_role_provider", _costing_stub_builder(provider))

    rc = main(["sessions", "compare", "run-AAAA11", "run-BBBB22"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "judge failed" in captured.err
    assert "judge spend $0.02" in captured.err  # two attempts billed 0.0102 each
    assert "total: candidates $0.12 + judge $0.02 = $0.14" in captured.out


class _UnpricedFakeProvider(_FakeProvider):
    """Bill usage with no reported cost under an unpriced model name.

    The shape that makes estimate_usd return (0.0, unknown=True).
    """

    budget: agent6_budget.BudgetTracker | None = None

    def call(self, **kw: Any) -> Any:
        assert self.budget is not None
        self.budget.record(
            model="unpriced-mystery-model",
            input_tokens=1000,
            output_tokens=100,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )
        return super().call(**kw)


def test_unpriced_judge_spend_reads_as_a_lower_bound_not_nothing(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unpriced reviewer with no reported cost renders as the ~ lower bound, never hidden."""
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")], cost=0.10)
    _setup_run(repo, "run-BBBB22", base_sha=base, commits=[("b.txt", "b\n", "add b")], cost=0.02)
    _write_reviewer_config(repo)
    verdict = '{"ranking": ["run-BBBB22", "run-AAAA11"], "rationale": "b is cleaner"}'
    provider = _UnpricedFakeProvider([verdict])

    def _build(*_a: Any, **kw: Any) -> Provider:
        provider.budget = kw["budget"]
        return cast(Provider, provider)

    monkeypatch.setattr(compare_mod, "build_role_provider", _build)

    rc = main(["sessions", "compare", "run-AAAA11", "run-BBBB22"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "+ judge ~$0.0000 = ~$0.12" in out  # marked lower bound, not hidden


def test_compare_falls_back_to_mechanical_on_judge_error(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A configured reviewer that never yields a verdict falls back to the mechanical ranking.

    Two malformed replies raise JudgeError; `rank` then ranks as `--parallel`'s auto-compare.
    """
    base = _init_repo(repo)
    _setup_run(
        repo,
        "run-AAAA11",
        base_sha=base,
        commits=[("a.txt", "a\n", "add a")],
        status="failed",
        cost=0.01,
    )
    _setup_run(
        repo,
        "run-BBBB22",
        base_sha=base,
        commits=[("b.txt", "b\n", "add b")],
        status="passed",
        cost=0.09,
    )
    _write_reviewer_config(repo)
    provider = _FakeProvider(["not json at all", "still not json"])
    monkeypatch.setattr(compare_mod, "build_role_provider", _stub_builder(provider))

    rc = main(["sessions", "compare", "run-AAAA11", "run-BBBB22"])

    assert rc == 0
    captured = capsys.readouterr()
    out, err = captured.out, captured.err
    assert provider.calls == 2  # the judge retried once, then gave up
    # Mechanical fallback: verify-pass wins despite costing more.
    assert out.index("run-BBBB22") < out.index("run-AAAA11")
    assert "judge:" not in out
    # The degradation is announced (not silent), so a mechanical table isn't
    # mistaken for a judged one. Same `rank` path feeds `--parallel`'s auto-compare.
    assert "judge failed" in err and "ranked mechanically" in err


# ---------------------------------------------------------------------------
# "judging..." feedback while the judge call is in flight
# ---------------------------------------------------------------------------


def _reviewer_cfg() -> Config:
    return Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {"reviewer": {"provider": "o", "model": "reviewer-1"}},
        }
    )


def _two_candidates() -> list[harness_judge.CandidateBrief]:
    return [
        harness_judge.CandidateBrief(
            session_id="run-AAAA11", task="t", diff="", verify_ok=True, cost_usd=0.1
        ),
        harness_judge.CandidateBrief(
            session_id="run-BBBB22", task="t", diff="", verify_ok=True, cost_usd=0.2
        ),
    ]


_VERDICT = '{"ranking": ["run-BBBB22", "run-AAAA11"], "rationale": "b is cleaner"}'

# The one spinner-frame owner every surface shares (imported at top).
_SPINNER_GLYPHS = format.SPINNER_FRAMES

# The run stream's heartbeat tick (`_console_view._HEARTBEAT_TICK_S`).
_HEARTBEAT_TICK_S = 0.5


class _FakeTTYOut(io.StringIO):
    """A tty-like stdout stand-in: isatty() True so the judging status animates."""

    def isatty(self) -> bool:
        return True


class _SlowFakeProvider:
    """A fake provider whose `.call()` sleeps first, and can raise instead of responding.

    The sleep gives a real terminal's spinner time to tick; the raise exercises the
    judge-failure cleanup path.
    """

    def __init__(
        self, *, sleep_s: float, text: str = "", raise_exc: Exception | None = None
    ) -> None:
        self._sleep_s = sleep_s
        self._text = text
        self._raise = raise_exc
        self.calls = 0

    def call(self, **_kw: Any) -> Any:
        self.calls += 1
        time.sleep(self._sleep_s)
        if self._raise is not None:
            raise self._raise
        return _Resp(self._text)


def test_rank_plain_judging_line_on_non_tty(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Piped or detached, one truthful line surrounds the judge call and no frame animates."""
    provider = _FakeProvider([_VERDICT])
    monkeypatch.setattr(compare_mod, "build_role_provider", _stub_builder(provider))

    compare_mod.rank(_reviewer_cfg(), _two_candidates(), transcript_dir=tmp_path)

    assert capsys.readouterr().out == "judging...\n"


def test_rank_animates_the_judging_status_on_a_tty(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real terminal spins the run stream's glyphs and cadence, then clears the line."""
    fake = _FakeTTYOut()
    monkeypatch.setattr(sys, "stdout", fake)
    provider = _SlowFakeProvider(sleep_s=_HEARTBEAT_TICK_S * 2.4, text=_VERDICT)
    monkeypatch.setattr(compare_mod, "build_role_provider", _stub_builder(provider))

    compare_mod.rank(_reviewer_cfg(), _two_candidates(), transcript_dir=tmp_path)

    text = fake.getvalue()
    assert any(glyph in text for glyph in _SPINNER_GLYPHS)
    assert "judging..." in text
    assert text.endswith("\r\x1b[2K")  # cleared before control returns to the caller
    frames = text.split("\r\x1b[2K")
    assert len({f for f in frames if f}) >= 2  # ticked through more than one frame


def test_rank_clears_the_judging_status_even_when_the_judge_call_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeTTYOut()
    monkeypatch.setattr(sys, "stdout", fake)
    provider = _SlowFakeProvider(sleep_s=_HEARTBEAT_TICK_S * 1.2, raise_exc=ProviderError("down"))
    monkeypatch.setattr(compare_mod, "build_role_provider", _stub_builder(provider))

    outcome = compare_mod.rank(_reviewer_cfg(), _two_candidates(), transcript_dir=tmp_path)

    assert outcome.ranked_by == "mechanical"  # judge failed -> fell back
    assert fake.getvalue().endswith("\r\x1b[2K")  # no leftover spinner droppings


def test_rank_mechanical_path_prints_no_judging_line(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no reviewer configured the mechanical fallback is instant and shows no status."""
    outcome = compare_mod.rank(Config(), _two_candidates(), transcript_dir=tmp_path)

    assert outcome.ranked_by == "mechanical"
    assert capsys.readouterr().out == ""


def test_parallel_and_runs_compare_share_one_rank_implementation() -> None:
    """The fan-out auto-compare and `sessions compare` route through the one core in `app.compare`.

    The CLI side only injects the console spinner and the reviewer-provider wiring.
    """
    from agent6.app import compare as app_compare
    from agent6.app import parallel
    from agent6.ui.cli import sessions_compare

    # The fan-out's auto-compare calls the core directly, through the module.
    assert parallel.app_compare is app_compare
    # `sessions compare` goes through the CLI wrapper, which delegates to that core.
    assert sessions_compare.rank is compare_mod.rank


def test_compare_reads_a_pruned_runs_change_from_the_recorded_merge(
    repo: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """After `prune --delete-squashed` a candidate's diff comes from the recorded merge."""
    base = _init_repo(repo)
    _setup_run(repo, "run-PPPP77", base_sha=base, commits=[("p.txt", "p\n", "add p")])
    tip = _git(repo, "rev-parse", "agent6/run-PPPP77")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--squash", "agent6/run-PPPP77")
    _git(repo, "commit", "-q", "-m", "squash p")
    landed = _git(repo, "rev-parse", "HEAD")
    _git(repo, "branch", "-D", "agent6/run-PPPP77")
    state = paths.state_dir(repo)
    manifest = sessions_layout.SessionLayout(
        state_dir=state, session_id="run-PPPP77", subdir="runs"
    ).manifest_path
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["merged"] = {"into": "main", "sha": landed, "tip": tip}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    _setup_run(repo, "run-QQQQ88", base_sha=base, commits=[("q.txt", "q\n", "add q")])
    assert main(["sessions", "compare", "run-PPPP77", "run-QQQQ88"]) == 0
    out = capsys.readouterr().out
    assert "read from the recorded merge" in out
    assert "empty diff" not in out.lower()


def _stamped_fanout(repo: pathlib.Path) -> None:
    """Two lanes of one fan-out, stamped with the verdict its auto-compare recorded."""
    base = _init_repo(repo)
    _setup_run(
        repo,
        "run-AAAA11",
        base_sha=base,
        commits=[("a.txt", "a\n", "add a")],
        manifest_extra=_lane_extra(winner=False, rank=2),
    )
    _setup_run(
        repo,
        "run-BBBB22",
        base_sha=base,
        commits=[("b.txt", "b\n", "add b")],
        manifest_extra=_lane_extra(winner=True, rank=1),
    )
    _write_reviewer_config(repo)


def test_compare_of_a_fanout_id_prints_the_recorded_verdict_without_judging(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Asking a fan-out for its comparison reads the stamp, never a fresh judge call."""
    _stamped_fanout(repo)
    judge = _FakeProvider(['{"ranking": ["run-AAAA11", "run-BBBB22"], "rationale": "flip"}'])
    monkeypatch.setattr(compare_mod, "build_role_provider", _stub_builder(judge))

    rc = main(["sessions", "compare", "fan"])

    assert rc == 0
    assert judge.calls == 0, "the recorded verdict is on disk; asking for it costs nothing"
    out = capsys.readouterr().out
    assert "recorded verdict" in out
    assert out.index("run-BBBB22") < out.index("run-AAAA11"), "the stamped order, best first"
    assert "--rejudge" in out, "the way to a fresh opinion is named"


def test_rejudge_on_a_fanout_id_spends_a_fresh_judge_call(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stamped_fanout(repo)
    judge = _FakeProvider(['{"ranking": ["run-AAAA11", "run-BBBB22"], "rationale": "flip"}'])
    monkeypatch.setattr(compare_mod, "build_role_provider", _stub_builder(judge))

    rc = main(["sessions", "compare", "fan", "--rejudge"])

    assert rc == 0
    assert judge.calls == 1
    out = capsys.readouterr().out
    assert out.index("run-AAAA11") < out.index("run-BBBB22"), "the fresh ranking"
    assert "note: the recorded fan-out verdict picked run-BBBB22" in out


def test_a_fanout_with_no_recorded_verdict_is_judged(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An auto-compare that never ran leaves no stamp, so the lanes are ranked instead."""
    base = _init_repo(repo)
    for sid, name in (("run-AAAA11", "a"), ("run-BBBB22", "b")):
        _setup_run(
            repo,
            sid,
            base_sha=base,
            commits=[(f"{name}.txt", f"{name}\n", f"add {name}")],
            manifest_extra={"parallel": {"group": "fan", "lane": 1, "coordinator": "fan"}},
        )
    _write_reviewer_config(repo)
    judge = _FakeProvider(['{"ranking": ["run-BBBB22", "run-AAAA11"], "rationale": "b wins"}'])
    monkeypatch.setattr(compare_mod, "build_role_provider", _stub_builder(judge))

    rc = main(["sessions", "compare", "fan"])

    assert rc == 0
    assert judge.calls == 1
    assert "b wins" in capsys.readouterr().out
