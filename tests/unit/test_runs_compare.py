# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 runs compare`: advisory verify+judge ranking across
already-run candidates. Real tmp git repos + fabricated run state (manifest.json
+ logs.jsonl), same fabrication pattern as test_cli_runs_merge.py (branches) and
test_parallel_orchestrator.py (`_write_fake_run`). The judge path is driven with
a fake provider (no network)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

from agent6.budget import BudgetTracker
from agent6.config import Config
from agent6.config.layer import repo_config_path_for, resolved_state_dir
from agent6.providers import Provider, ProviderError
from agent6.runs.layout import RunLayout
from agent6.ui.cli import _compare as compare_mod
from agent6.ui.cli import main
from agent6.workflows.judge import CandidateBrief


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_repo(repo: Path) -> str:
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return _git(repo, "rev-parse", "HEAD")


def _setup_run(
    repo: Path,
    run_id: str,
    *,
    base_sha: str,
    commits: list[tuple[str, str, str]],
    task: str = "implement the thing",
    status: str = "passed",
    cost: float = 0.05,
    manifest_extra: dict[str, Any] | None = None,
) -> None:
    """Cut agent6/<run_id> off base_sha with *commits*, write manifest.json +
    logs.jsonl (the run-branch + run-state fixture `runs compare` reads), and
    return the checkout to where it was. *manifest_extra* merges extra manifest
    fields (e.g. a fan-out lane's parallel_id + compare stamp)."""
    branch = f"agent6/{run_id}"
    current = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _git(repo, "checkout", "-q", base_sha)
    _git(repo, "checkout", "-q", "-b", branch)
    for name, content, msg in commits:
        (repo / name).write_text(content, encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", msg)
    _git(repo, "checkout", "-q", current)

    layout = RunLayout(state_dir=resolved_state_dir(repo), run_id=run_id)
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "run_id": run_id,
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
        {"type": "run.start", "mode": "run", "user_task": task},
        {"type": "budget.update", "usd_total": cost},
    ]
    if status == "passed":
        events.append({"type": "run.end", "reason": "finish_run", "all_passed": True})
    elif status == "failed":
        events.append({"type": "run.end", "reason": "provider_error", "all_passed": False})
    layout.logs_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Run state is isolated by the autouse `_isolate_state` fixture (conftest.py)
    # to a tmp dir OUTSIDE this one; nesting AGENT6_STATE_HOME under tmp_path here
    # would put untracked run state inside the repo's own working tree, where a
    # second run's `git add -A` sweeps it onto that run's branch and a later
    # checkout back to main deletes it as "not in this branch's tree".
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_compare_needs_at_least_two_ids(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")])
    rc = main(["runs", "compare", "run-AAAA11"])
    assert rc == 2
    assert "at least 2" in capsys.readouterr().err


def test_compare_unknown_id_errors_loudly(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")])
    rc = main(["runs", "compare", "run-AAAA11", "nonexistent"])
    assert rc == 2
    assert "no run matches" in capsys.readouterr().err


def test_compare_ambiguous_id_errors_loudly(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _init_repo(repo)
    _setup_run(repo, "run-DUPXX1", base_sha=base, commits=[("a.txt", "a\n", "add a")])
    _setup_run(repo, "run-DUPXX2", base_sha=base, commits=[("b.txt", "b\n", "add b")])
    _setup_run(repo, "run-CCCC33", base_sha=base, commits=[("c.txt", "c\n", "add c")])
    rc = main(["runs", "compare", "run-DUP", "run-CCCC33"])
    assert rc == 2
    assert "ambiguous" in capsys.readouterr().err


def test_compare_rejects_duplicate_id(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")])
    _setup_run(repo, "run-BBBB22", base_sha=base, commits=[("b.txt", "b\n", "add b")])
    rc = main(["runs", "compare", "run-AAAA11", "run-AAAA11"])
    assert rc == 2
    assert "more than once" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Mechanical path (no reviewer model configured)
# ---------------------------------------------------------------------------


def test_compare_prefix_resolution_and_mechanical_ranking(
    repo: Path, capsys: pytest.CaptureFixture[str]
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
    rc = main(["runs", "compare", "run-AAAA", "run-BBBB22"])  # unique prefix + exact id
    assert rc == 0
    out = capsys.readouterr().out
    assert "ranked candidates" in out
    assert out.index("run-BBBB22") < out.index("run-AAAA11")
    assert "agent6 runs merge run-BBBB22" in out
    assert "no reviewer model configured" in out
    # Candidate spend is totaled; no judge ran, so no judge figure.
    assert "total: candidates $0.1000" in out and "+ judge" not in out


def test_compare_is_read_only(repo: Path) -> None:
    """Never merges, never writes to the run's own branch/manifest."""
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")])
    _setup_run(repo, "run-BBBB22", base_sha=base, commits=[("b.txt", "b\n", "add b")])
    head_before = _git(repo, "rev-parse", "main")
    manifest_before = (
        RunLayout(state_dir=resolved_state_dir(repo), run_id="run-AAAA11").manifest_path
    ).read_text(encoding="utf-8")
    rc = main(["runs", "compare", "run-AAAA11", "run-BBBB22"])
    assert rc == 0
    assert _git(repo, "rev-parse", "main") == head_before
    assert (
        RunLayout(state_dir=resolved_state_dir(repo), run_id="run-AAAA11").manifest_path
    ).read_text(encoding="utf-8") == manifest_before


# ---------------------------------------------------------------------------
# Judge path (fake provider, no network)
# ---------------------------------------------------------------------------


def _write_reviewer_config(repo: Path) -> None:
    p = repo_config_path_for(repo)
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
    """Stand-in for `_build_role_provider` so the judge path needs no API key or
    network; returns *provider* regardless of the (cfg, role, ...) it's called
    with. *provider* is any object with the fake `.call()` shape (`_FakeProvider`,
    `_SlowFakeProvider`), cast to `Provider` for the caller."""

    def _build(*_a: Any, **_k: Any) -> Provider:
        return cast(Provider, provider)

    return _build


class _CostingFakeProvider(_FakeProvider):
    """A fake provider that also bills each call into the BudgetTracker its
    builder received, the way a real provider records usage."""

    budget: BudgetTracker | None = None

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
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")], cost=0.10)
    _setup_run(repo, "run-BBBB22", base_sha=base, commits=[("b.txt", "b\n", "add b")], cost=0.02)
    _write_reviewer_config(repo)
    verdict = '{"ranking": ["run-BBBB22", "run-AAAA11"], "rationale": "b is cleaner"}'
    provider = _FakeProvider([verdict])
    monkeypatch.setattr(compare_mod, "build_role_provider", _stub_builder(provider))

    rc = main(["runs", "compare", "run-AAAA11", "run-BBBB22"])

    assert rc == 0
    out = capsys.readouterr().out
    assert out.index("run-BBBB22") < out.index("run-AAAA11")
    assert "judge: b is cleaner" in out
    assert "no reviewer model configured" not in out
    assert provider.calls == 1


def test_compare_total_line_accounts_the_judge_calls_own_spend(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The judge call is real money: the ranked report's total line carries it
    (candidates + judge = grand total), so judging spend is never invisible."""
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")], cost=0.10)
    _setup_run(repo, "run-BBBB22", base_sha=base, commits=[("b.txt", "b\n", "add b")], cost=0.02)
    _write_reviewer_config(repo)
    verdict = '{"ranking": ["run-BBBB22", "run-AAAA11"], "rationale": "b is cleaner"}'
    provider = _CostingFakeProvider([verdict])
    monkeypatch.setattr(compare_mod, "build_role_provider", _costing_stub_builder(provider))

    rc = main(["runs", "compare", "run-AAAA11", "run-BBBB22"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "total: candidates $0.1200 + judge $0.0102 = $0.1302" in out


def _lane_extra(*, winner: bool, rank: int) -> dict[str, Any]:
    return {
        "parallel_id": "fan",
        "lane": rank,
        "compare": {"rank": rank, "of": 2, "winner": winner, "ranked_by": "judge"},
    }


def test_compare_discloses_a_fresh_verdict_that_contradicts_the_stamp(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-judging one fan-out's own lanes can flip the winner; the recorded
    stamp (the listings' star) is never rewritten, so the clash must be said
    out loud, not left for the operator to trip over in `runs list`."""
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

    rc = main(["runs", "compare", "run-AAAA11", "run-BBBB22"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "note: the recorded fan-out verdict picked run-BBBB22" in out
    assert "nothing was re-stamped" in out


def test_compare_stays_quiet_when_the_fresh_verdict_agrees_with_the_stamp(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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

    rc = main(["runs", "compare", "run-AAAA11", "run-BBBB22"])

    assert rc == 0
    assert "note:" not in capsys.readouterr().out


def test_failed_judge_announces_what_its_attempts_still_spent(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two malformed replies fall back to the mechanical ranking, but both
    attempts billed; the degradation line must carry that spend and the
    mechanical outcome must stamp it (never-invisible is the whole point)."""
    base = _init_repo(repo)
    _setup_run(repo, "run-AAAA11", base_sha=base, commits=[("a.txt", "a\n", "add a")], cost=0.10)
    _setup_run(repo, "run-BBBB22", base_sha=base, commits=[("b.txt", "b\n", "add b")], cost=0.02)
    _write_reviewer_config(repo)
    provider = _CostingFakeProvider(["not json at all", "still not json"])
    monkeypatch.setattr(compare_mod, "build_role_provider", _costing_stub_builder(provider))

    rc = main(["runs", "compare", "run-AAAA11", "run-BBBB22"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "judge failed" in captured.err
    assert "judge spend $0.0204" in captured.err  # two attempts billed 0.0102 each
    assert "total: candidates $0.1200 + judge $0.0204 = $0.1404" in captured.out


class _UnpricedFakeProvider(_FakeProvider):
    """Bills usage with NO reported cost under an unpriced model name, the
    shape that makes estimate_usd return (0.0, unknown=True)."""

    budget: BudgetTracker | None = None

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
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unpriced reviewer with no reported cost estimates $0.0000 with the
    unknown flag; suppressing the judge figure entirely would make judging
    spend invisible again, so it renders as the ~ lower bound instead."""
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

    rc = main(["runs", "compare", "run-AAAA11", "run-BBBB22"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "+ judge ~$0.0000 = ~$0.1200" in out  # marked lower bound, not hidden


def test_compare_falls_back_to_mechanical_on_judge_error(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reviewer model is configured but the judge never produces a valid
    verdict (two malformed replies -> JudgeError): `rank` falls back to the
    mechanical ranking, same as `--parallel`'s auto-compare."""
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

    rc = main(["runs", "compare", "run-AAAA11", "run-BBBB22"])

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


def _two_candidates() -> list[CandidateBrief]:
    return [
        CandidateBrief(run_id="run-AAAA11", task="t", diff="", verify_ok=True, cost_usd=0.1),
        CandidateBrief(run_id="run-BBBB22", task="t", diff="", verify_ok=True, cost_usd=0.2),
    ]


_VERDICT = '{"ranking": ["run-BBBB22", "run-AAAA11"], "rationale": "b is cleaner"}'

# Same glyph set as `_console_view._SPINNER` (also duplicated in ui/tui/app.py and
# ui/web/page.py) -- proves the judging indicator reuses the CLI's one animation
# rather than inventing a second one.
_SPINNER_GLYPHS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
# The run stream's heartbeat tick (`_console_view._HEARTBEAT_TICK_S`).
_HEARTBEAT_TICK_S = 0.5


class _FakeTTYOut(io.StringIO):
    """A tty-like stdout stand-in: isatty() True so the judging status animates."""

    def isatty(self) -> bool:
        return True


class _SlowFakeProvider:
    """Like `_FakeProvider`, but `.call()` sleeps first so a real terminal's
    spinner gets time to tick during the (fake) judge call -- and can raise
    instead of responding, to exercise the judge-failure cleanup path."""

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Piped/detached (not a terminal, the default under capsys): one truthful
    line around the judge call, no animation frames."""
    provider = _FakeProvider([_VERDICT])
    monkeypatch.setattr(compare_mod, "build_role_provider", _stub_builder(provider))

    compare_mod.rank(_reviewer_cfg(), _two_candidates(), transcript_dir=tmp_path)

    assert capsys.readouterr().out == "judging...\n"


def test_rank_animates_the_judging_status_on_a_tty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real terminal spins the SAME glyphs/cadence as the run stream's
    provider-call heartbeat, then clears the line before the ranked table."""
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeTTYOut()
    monkeypatch.setattr(sys, "stdout", fake)
    provider = _SlowFakeProvider(sleep_s=_HEARTBEAT_TICK_S * 1.2, raise_exc=ProviderError("down"))
    monkeypatch.setattr(compare_mod, "build_role_provider", _stub_builder(provider))

    outcome = compare_mod.rank(_reviewer_cfg(), _two_candidates(), transcript_dir=tmp_path)

    assert outcome.ranked_by == "mechanical"  # judge failed -> fell back
    assert fake.getvalue().endswith("\r\x1b[2K")  # no leftover spinner droppings


def test_rank_mechanical_path_prints_no_judging_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No reviewer configured -> the mechanical fallback is instant; nothing to
    show a status for."""
    outcome = compare_mod.rank(Config(), _two_candidates(), transcript_dir=tmp_path)

    assert outcome.ranked_by == "mechanical"
    assert capsys.readouterr().out == ""


def test_parallel_and_runs_compare_share_one_rank_implementation() -> None:
    """No second spinner/rank implementation to drift: the fan-out auto-compare
    and `runs compare` both route through the ONE core in `app.compare`; the CLI
    side only injects the console spinner + reviewer-provider wiring."""
    from agent6.app import compare as app_compare
    from agent6.app import parallel
    from agent6.ui.cli import runs_cmds

    # The fan-out's auto-compare calls the core directly.
    assert getattr(parallel, "rank") is app_compare.rank  # noqa: B009
    # `runs compare` goes through the CLI wrapper, which delegates to that core.
    assert runs_cmds.rank is compare_mod.rank
