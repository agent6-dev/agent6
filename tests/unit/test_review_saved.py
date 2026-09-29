# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 review` saves its rendered review under `<state-dir>/reviews/`.

Beside the provider transcripts, so a later session working on a module can read its review there.
"""

from __future__ import annotations

import pathlib
import subprocess
import types
from typing import Any
from unittest import mock

import pytest

from agent6 import paths
from agent6.config import Config
from agent6.ui.cli import review_cmds


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

    monkeypatch.setattr(review_cmds, "load_effective", loaded)
    monkeypatch.setattr(review_cmds, "check_provider_keys", mock.MagicMock(return_value=None))
    return tmp_path


def _saved_reviews(repo: pathlib.Path) -> list[pathlib.Path]:
    return sorted((paths.state_dir(repo) / "reviews").glob("*-review.md"))


def test_the_freeform_review_is_saved_and_its_path_named(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(review_cmds, "build_role_provider", mock.MagicMock())
    monkeypatch.setattr(
        review_cmds, "code_review", mock.MagicMock(return_value="LGTM with nits\n- [nit] x")
    )

    rc = review_cmds._cmd_review(None, base="", head="HEAD", paths=())  # pyright: ignore[reportPrivateUsage]

    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "LGTM with nits\n- [nit] x\n"
    saved = _saved_reviews(repo)
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8") == (
        "# review: working tree vs HEAD\n\nLGTM with nits\n- [nit] x\n"
    )
    assert f"review saved: {saved[0]}" in out.err


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
    from agent6.harness import _reviewer

    seat = _reviewer.ReviewSeat(
        persona="correctness",
        provider=_PassingSeatProvider(),  # type: ignore[arg-type]
        model="reviewer",
    )
    monkeypatch.setattr(review_cmds, "build_review_seats", mock.MagicMock(return_value=[seat]))

    rc = review_cmds._cmd_review(None, base="", head="HEAD", paths=(), reviewers=1)  # pyright: ignore[reportPrivateUsage]

    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "VERDICT: PASS\n"
    saved = _saved_reviews(repo)
    assert len(saved) == 1
    assert (
        saved[0].read_text(encoding="utf-8") == "# review: working tree vs HEAD\n\nVERDICT: PASS\n"
    )
    assert f"review saved: {saved[0]}" in out.err


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
