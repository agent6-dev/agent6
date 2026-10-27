# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 review` saves its rendered review under `<state-dir>/reviews/`.

Beside the provider transcripts, so a later session working on a module can read its review there.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import types
from typing import Any
from unittest import mock

import pytest

from agent6 import event_log, paths, review_records
from agent6.app import _setup, providers
from agent6.config import Config, layer
from agent6.harness import _reviewer, code_review
from agent6.ui.cli import review_cmds
from tests.unit import test_review_diff_collect, test_review_gate


def _repo_with_a_change(repo: pathlib.Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (repo / "lib.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    git("add", "lib.py")
    git("commit", "-qm", "A")
    (repo / "lib.py").write_text("def f(x):\n    return x + 1\n", encoding="utf-8")


def _reviewer_config() -> Config:
    return Config.model_validate(
        {
            "providers": {"local": {"api_format": "openai", "base_url": "https://example.test/v1"}},
            "models": {"reviewer": {"provider": "local", "model": "reviewer"}},
        }
    )


@pytest.fixture
def repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    _repo_with_a_change(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    cfg = _reviewer_config()

    def loaded(*_a: object, **_k: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(config=cfg)

    monkeypatch.setattr(layer, "load_effective", loaded)
    monkeypatch.setattr(_setup, "check_provider_keys", mock.MagicMock(return_value=None))
    return tmp_path


def _saved_reviews(repo: pathlib.Path) -> list[pathlib.Path]:
    return sorted((paths.state_dir(repo) / "reviews").glob("*-review.md"))


def test_the_freeform_review_is_saved_and_its_path_named(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(providers, "build_role_provider", mock.MagicMock())
    monkeypatch.setattr(
        code_review, "review_diff", mock.MagicMock(return_value="LGTM with nits\n- [nit] x")
    )

    rc = review_cmds._cmd_review((), base="", head="HEAD", paths=())  # pyright: ignore[reportPrivateUsage]

    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "LGTM with nits\n- [nit] x\n"
    saved = _saved_reviews(repo)
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8") == (
        "# review: working tree vs HEAD\n\nLGTM with nits\n- [nit] x\n"
    )
    assert f"review saved: {saved[0]}" in out.err


def test_a_review_over_budget_prints_the_ledger_reason_once(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The budget error already names its ledger; the command adds no second prefix.

    A review at its plan cap printed "budget exhausted: plan budget exhausted: ...".
    """
    from agent6 import budget as agent6_budget

    monkeypatch.setattr(providers, "build_role_provider", mock.MagicMock())
    monkeypatch.setattr(
        code_review,
        "review_diff",
        mock.MagicMock(
            side_effect=agent6_budget.BudgetExceededError("plan budget exhausted: 3.0 of 3pt")
        ),
    )

    rc = review_cmds._cmd_review((), base="", head="HEAD", paths=())  # pyright: ignore[reportPrivateUsage]

    err = capsys.readouterr().err
    assert rc == 3
    assert "plan budget exhausted: 3.0 of 3pt" in err
    assert err.count("budget exhausted") == 1


class _PassingSeatProvider:
    def call(self, **_kwargs: Any) -> Any:
        from agent6.providers import ProviderResponse

        return ProviderResponse(
            text='{"verdict":"pass","summary":"clean","findings":[]}',
            tool_uses=(),
            stop_reason="end_turn",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )


def test_the_panel_verdict_is_saved_too(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seat = _reviewer.ReviewSeat(
        focus="correctness",
        provider=_PassingSeatProvider(),  # type: ignore[arg-type]
        model="reviewer",
    )
    monkeypatch.setattr(providers, "build_review_seats", mock.MagicMock(return_value=[seat]))

    rc = review_cmds._cmd_review((), base="", head="HEAD", paths=(), reviewers=1)  # pyright: ignore[reportPrivateUsage]

    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "VERDICT: PASS\n"
    saved = _saved_reviews(repo)
    assert len(saved) == 1
    assert (
        saved[0].read_text(encoding="utf-8") == "# review: working tree vs HEAD\n\nVERDICT: PASS\n"
    )
    assert f"review saved: {saved[0]}" in out.err


@pytest.mark.parametrize("nested", [False, True], ids=["root", "subdirectory"])
@pytest.mark.parametrize("committed", [False, True], ids=["worktree", "revisions"])
@pytest.mark.parametrize("hidden", [False, True], ids=["ordinary", "hidden-ignore-revs"])
def test_standalone_panel_records_line_origins_against_its_base(
    repo: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    committed: bool,
    nested: bool,
    hidden: bool,
) -> None:
    """Classify every saved finding against the reviewed base, not the head tree."""
    base = test_review_diff_collect._git(repo, "rev-parse", "HEAD").strip()
    if committed:
        test_review_diff_collect._git(repo, "commit", "-qam", "change")
    if hidden:
        secret = repo / "secret.env"
        secret.write_text("SECRET-ENV-LINE-4242\n")
        test_review_diff_collect._git(repo, "config", "blame.ignoreRevsFile", "secret.env")
        cfg = _reviewer_config()
        cfg = cfg.model_copy(
            update={"sandbox": cfg.sandbox.model_copy(update={"hide_paths": (str(secret),)})}
        )
        monkeypatch.setattr(
            layer, "load_effective", mock.Mock(return_value=types.SimpleNamespace(config=cfg))
        )
    cwd = repo / "nested" if nested else repo
    cwd.mkdir(exist_ok=True)
    monkeypatch.chdir(cwd)
    provider = mock.Mock()
    provider.call.return_value = test_review_gate._resp(
        json.dumps(
            dict(
                verdict="block",
                findings=[
                    dict(
                        category="correctness",
                        severity="block",
                        title=f"Defect {line}",
                        location=dict(path="lib.py", side="head", start_line=line, end_line=line),
                    )
                    for line in (1, 2)
                ],
            )
        )
    )
    monkeypatch.setattr(
        providers,
        "build_review_seats",
        mock.Mock(return_value=[_reviewer.ReviewSeat("correctness", "model", provider)]),
    )
    assert (
        review_cmds._cmd_review(
            (), base=base if committed else "", head="HEAD", paths=(), reviewers=1
        )
        == 0
    )
    (journal,) = (paths.state_dir(cwd) / "reviews").glob("*.jsonl")
    assert "SECRET-ENV-LINE-4242" not in journal.read_text()
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    (record,) = [
        review_records.ReviewPanel.model_validate(event_log.payload(row))
        for row in rows
        if row["type"] == "loop.review.panel"
    ]
    assert record.target.base_commit == base
    assert record.target.base_tree != record.target.head_tree
    assert [f.line_origin.kind for f in record.findings] == [
        "unknown" if hidden else "pre_existing",
        "changed",
    ]
    origin = record.findings[0].line_origin
    if hidden:
        assert "blame.ignoreRevsFile" in origin.reason and "refused" in origin.reason
        assert "sandbox.hide_paths" in origin.reason
        assert origin.base_location is None and not origin.commits
    else:
        assert origin.base_location == review_records.ReviewLocation(
            path="lib.py", side="base", start_line=1, end_line=1
        )
        assert origin.commits == (base,)
    assert record.findings[1].line_origin.base_location is None
    assert not record.findings[1].line_origin.commits


@pytest.mark.parametrize("budget_failure", [False, True])
def test_failed_freeform_review_records_a_terminal_reason(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, budget_failure: bool
) -> None:
    from agent6 import budget
    from agent6.providers import ProviderError
    from agent6.viewmodel import scan_session_log

    provider = mock.Mock()
    provider.call.side_effect = (
        budget.BudgetExceededError("plan budget exhausted")
        if budget_failure
        else ProviderError("offline")
    )
    monkeypatch.setattr(providers, "build_role_provider", mock.Mock(return_value=provider))
    rc = review_cmds._cmd_review((), base="", head="HEAD", paths=())
    assert rc == (3 if budget_failure else 2)
    (journal,) = (paths.state_dir(repo) / "reviews").glob("*.jsonl")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert rows[-1]["type"] == "loop.review.skipped"
    assert rows[-1]["reason"]
    assert rows[-1]["panel_id"] == rows[0]["panel_id"]
    assert scan_session_log(journal).review.completion == "skipped"


def test_two_reviews_in_one_second_keep_both(tmp_path: pathlib.Path) -> None:
    """Two reviews in one second keep both.

    Two reviews of one repo in the same second (a CLI review beside a TUI one) choose the same name;
    the name is claimed with an exclusive create, so a file that appears between the check and the
    write is never replaced.
    """
    first = review_cmds.save_review(tmp_path, label="a", body="one")
    second = review_cmds.save_review(tmp_path, label="b", body="two")
    assert first != second
    taken = tmp_path / "20260101T000000Z-review.md"
    taken.write_text("# review: theirs\n\nkept\n", encoding="utf-8")
    with mock.patch("agent6.ui.cli.review_cmds.time.strftime", return_value="20260101T000000Z"):
        mine = review_cmds.save_review(tmp_path, label="c", body="three")
    assert mine == tmp_path / "20260101T000000Z-2-review.md"
    assert taken.read_text(encoding="utf-8") == "# review: theirs\n\nkept\n"
    assert {p.read_text(encoding="utf-8") for p in (first, second)} == {
        "# review: a\n\none\n",
        "# review: b\n\ntwo\n",
    }
