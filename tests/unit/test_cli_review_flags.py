# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 review` says which of its flags it cannot honour."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.config import ConfigError
from agent6.ui.cli import cli_main


def test_negative_reviewer_count_is_rejected_by_the_parser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["review", "--reviewers", "-1"])
    assert exc.value.code == 2
    assert "non-negative" in capsys.readouterr().err


def test_head_without_base_is_rejected_before_config_load(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def should_not_load(*_a: object, **_k: object) -> object:
        raise AssertionError("config loaded")

    monkeypatch.setattr("agent6.ui.cli.review_cmds.load_effective", should_not_load)
    assert cli_main(["review", "--head", "topic"]) == 2
    assert "--head requires --base" in capsys.readouterr().err


def test_an_empty_range_is_reported_before_provider_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid empty range is a successful no-work review, even when no model
    is configured; the note names the range and paths, on stderr like every
    other status line, so a script reading stdout for a verdict sees none."""
    from types import SimpleNamespace

    from agent6.config import Config
    from agent6.ui.cli import review_cmds

    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "init",
        ],
        check=True,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        review_cmds,
        "load_effective",
        MagicMock(return_value=SimpleNamespace(config=Config())),
    )
    monkeypatch.setattr(
        review_cmds,
        "check_provider_keys",
        MagicMock(side_effect=AssertionError("provider keys checked")),
    )
    monkeypatch.setattr(
        review_cmds,
        "build_review_seats",
        MagicMock(side_effect=AssertionError("seat built")),
    )

    rc = review_cmds._cmd_review(  # pyright: ignore[reportPrivateUsage]
        None,
        base="HEAD",
        head="HEAD",
        paths=("src/x.py",),
        reviewers=1,
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""
    assert captured.err == "(no diff to review: HEAD..HEAD -- src/x.py)\n"


def test_personas_without_reviewers_is_said_to_be_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--personas` is read only under `--reviewers N`: alone it ran the single
    freeform review with the named seats silently dropped. The sibling
    `model` command prints a note for a flag it cannot use; so does this."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)

    def stop(*_a: object, **_k: object) -> object:
        raise ConfigError("stop here")

    monkeypatch.setattr("agent6.ui.cli.review_cmds.load_effective", stop)
    rc = cli_main(["review", "--personas", "security,tests"])
    err = capsys.readouterr().err
    assert rc == 2 and "stop here" in err
    assert "note: --personas ignored (no --reviewers N" in err

    rc = cli_main(["review", "--personas", "security,tests", "--reviewers", "2"])
    assert rc == 2 and "--personas ignored" not in capsys.readouterr().err


def test_personas_under_configured_seats_is_said_to_be_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`[review].seats` names the roster outright, as the flag's help says;
    the flag beside it was dropped in silence."""
    from types import SimpleNamespace

    from agent6.config import Config

    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    cfg = Config.model_validate({"review": {"seats": ["security@openrouter/some-model"]}})

    def loaded(*_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(config=cfg)

    monkeypatch.setattr("agent6.ui.cli.review_cmds.load_effective", loaded)
    rc = cli_main(["review", "--personas", "tests", "--reviewers", "2"])
    err = capsys.readouterr().err
    assert rc == 2 and "note: --personas ignored ([review].seats names the roster)." in err


def _panel_config(*, with_reviewer: bool) -> Any:
    from agent6.config import Config

    data: dict[str, Any] = {
        "providers": {"local": {"api_format": "openai", "base_url": "https://example.test/v1"}}
    }
    if with_reviewer:
        data["models"] = {"reviewer": {"provider": "local", "model": "reviewer"}}
    return Config.model_validate(data)


def _two_commits(repo: Path) -> tuple[str, str]:
    """Commit A defines `f(x)` with a caller `f(1)`; commit B widens the
    signature and updates the caller. Returns (A, B), checked out at A."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (repo / "lib.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    (repo / "caller.py").write_text("from lib import f\n\nprint(f(1))\n", encoding="utf-8")
    git("add", "lib.py", "caller.py")
    git("commit", "-qm", "A")
    a = git("rev-parse", "HEAD")
    (repo / "lib.py").write_text("def f(x, y):\n    return x + y\n", encoding="utf-8")
    (repo / "caller.py").write_text("from lib import f\n\nprint(f(1, 2))\n", encoding="utf-8")
    git("add", "lib.py", "caller.py")
    git("commit", "-qm", "B")
    b = git("rev-parse", "HEAD")
    git("checkout", "-q", a)
    return a, b


class _FixedReviewProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.last_user = ""

    def call(self, **kwargs: Any) -> Any:
        from agent6.providers import ProviderResponse

        self.calls += 1
        self.last_user = str(kwargs["messages"][0]["content"])
        return ProviderResponse(
            text='{"verdict":"pass","summary":"clean","findings":[]}',
            tool_uses=(),
            stop_reason="end_turn",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )


def test_an_arbitrary_range_uses_the_selected_heads_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reviewer's recent-history context came from the checkout's HEAD
    whatever --head named; it is the reviewed head's log."""
    from types import SimpleNamespace

    from agent6.ui.cli import review_cmds

    base, head = _two_commits(tmp_path)
    (tmp_path / "checkout-only.txt").write_text("not in reviewed head\n", encoding="utf-8")
    subprocess.run(["git", "add", "checkout-only.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "checkout-only"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    provider = _FixedReviewProvider()
    monkeypatch.setattr(
        review_cmds,
        "load_effective",
        MagicMock(return_value=SimpleNamespace(config=_panel_config(with_reviewer=True))),
    )
    monkeypatch.setattr(review_cmds, "check_provider_keys", MagicMock(return_value=None))
    monkeypatch.setattr(review_cmds, "build_role_provider", MagicMock(return_value=provider))

    rc = review_cmds._cmd_review(  # pyright: ignore[reportPrivateUsage]
        None, base=base, head=head, paths=()
    )

    recent_log = provider.last_user.split("\n\nDIFF:", 1)[0]
    assert rc == 0
    assert any(line.endswith(" B") for line in recent_log.splitlines())
    assert "checkout-only" not in recent_log
    assert "reviewing:" in capsys.readouterr().err


def test_an_unknown_pinned_provider_is_named_without_a_reviewer_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A seat pinning a provider with no [providers.<name>] block was named
    only once a reviewer route existed; every seat's provider is checked
    before any seat is built."""
    from types import SimpleNamespace

    from agent6.ui.cli import review_cmds

    base, head = _two_commits(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(
        review_cmds,
        "load_effective",
        MagicMock(return_value=SimpleNamespace(config=_panel_config(with_reviewer=False))),
    )
    monkeypatch.setattr(
        review_cmds, "check_provider_keys", MagicMock(return_value="unrelated missing key")
    )

    rc = review_cmds._cmd_review(  # pyright: ignore[reportPrivateUsage]
        None,
        base=base,
        head=head,
        paths=(),
        reviewers=1,
        personas="security@missing/model",
    )

    err = capsys.readouterr().err
    assert rc == 2
    assert "provider 'missing'" in err
    assert "unrelated" not in err


def test_a_fully_pinned_panel_needs_no_reviewer_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A panel whose every seat pins a model has no use for [models.reviewer],
    which it used to require."""
    from types import SimpleNamespace

    from agent6.app import providers as provider_builders
    from agent6.ui.cli import review_cmds

    base, head = _two_commits(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    provider = _FixedReviewProvider()
    monkeypatch.setattr(
        review_cmds,
        "load_effective",
        MagicMock(return_value=SimpleNamespace(config=_panel_config(with_reviewer=False))),
    )
    monkeypatch.setattr(review_cmds, "check_provider_keys", MagicMock(return_value=None))
    monkeypatch.setattr(provider_builders, "_provider_from_entry", MagicMock(return_value=provider))

    rc = review_cmds._cmd_review(  # pyright: ignore[reportPrivateUsage]
        None,
        base=base,
        head=head,
        paths=(),
        reviewers=1,
        personas="security@local/model",
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert provider.calls == 1
    assert captured.out == "VERDICT: PASS\n"
    assert "1 seats (security)" in captured.err


def _explore_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, base: str, head: str
) -> tuple[int, dict[str, Any]]:
    """`agent6 review --reviewers 1` under `review.tier = "explore"` with one
    fake seat whose panel reads `caller.py` the way the explore prompt tells it
    to; returns the exit code and what the seat read."""
    from types import SimpleNamespace

    from agent6.config import Config
    from agent6.ui.cli import review_cmds
    from agent6.workflows._panel import PanelResult
    from agent6.workflows._review import ReviewSeat

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    cfg = Config.model_validate({"review": {"tier": "explore"}})
    seen: dict[str, Any] = {}

    def _fake_effective(*_a: object, **_k: object) -> SimpleNamespace:
        return SimpleNamespace(config=cfg)

    def _runnable(_self: Config, _role: str) -> None:
        return None

    def _no_key_error(*_a: object, **_k: object) -> None:
        return None

    def _fake_seats(_cfg: Config, **_k: Any) -> list[ReviewSeat]:
        return [
            ReviewSeat(
                persona="correctness",
                model="fake",
                provider=None,  # pyright: ignore[reportArgumentType]
                tier="explore",
            )
        ]

    def _fake_panel(_seats: Any, _ctx: Any, **kw: Any) -> PanelResult:
        seen["read_file"] = kw["dispatch"]("read_file", {"path": "caller.py"}).content
        return PanelResult(
            panel_id="cli",
            decision="advisory",
            blocked=False,
            merged_findings=(),
            per_seat=(),
            n_block=0,
            n_abstain=0,
        )

    monkeypatch.setattr(review_cmds, "load_effective", _fake_effective)
    monkeypatch.setattr(Config, "require_runnable", _runnable)
    monkeypatch.setattr(review_cmds, "check_provider_keys", _no_key_error)
    monkeypatch.setattr(review_cmds, "build_review_seats", _fake_seats)
    monkeypatch.setattr(review_cmds, "run_panel", _fake_panel)
    rc = review_cmds._cmd_review(  # pyright: ignore[reportPrivateUsage]
        None, base=base, head=head, paths=(), reviewers=1
    )
    return rc, seen


def test_explore_tier_gates_on_the_head_being_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`review.tier = "explore"` hands a seat read-only tools over the CHECKOUT,
    so reviewing a `--head` that is not checked out fed it file contents the
    diff contradicts: the diff said the caller became `f(1, 2)` while
    `read_file` returned the old `f(1)`, the false break the explore prompt
    tells a seat to BLOCK on. Both directions: the ordinary `--head HEAD` on
    the checked-out commit still runs its seat."""
    base, head = _two_commits(tmp_path)
    rc, seen = _explore_review(tmp_path, monkeypatch, base=base, head=head)
    err = capsys.readouterr().err
    assert seen == {}, f"an explore seat read the checkout, not --head: {seen}"
    assert rc == 2
    assert "--head" in err and "explore" in err

    subprocess.run(["git", "checkout", "-q", head], cwd=tmp_path, check=True)
    (tmp_path / "scratch.log").write_text("build output\n", encoding="utf-8")  # untracked
    rc, seen = _explore_review(tmp_path, monkeypatch, base=base, head="HEAD")
    assert rc == 0
    assert seen == {"read_file": "from lib import f\n\nprint(f(1, 2))\n"}


def test_explore_tier_refuses_a_dirty_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default `--head HEAD` names the checked-out commit whatever the tree
    holds, so an uncommitted edit to a file the `base..HEAD` diff describes fed
    a seat the same false break a wrong `--head` does."""
    base, head = _two_commits(tmp_path)
    subprocess.run(["git", "checkout", "-q", head], cwd=tmp_path, check=True)
    (tmp_path / "caller.py").write_text("from lib import f\n\nprint(f(1))\n", encoding="utf-8")
    rc, seen = _explore_review(tmp_path, monkeypatch, base=base, head="HEAD")
    err = capsys.readouterr().err
    assert seen == {}, f"an explore seat read a dirty tree the diff contradicts: {seen}"
    assert rc == 2
    assert "--head" in err and "explore" in err


def test_a_panel_with_an_unpinned_seat_still_requires_the_reviewer_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving the route requirement under the single-reviewer path left the
    panel with none: an unpinned seat fell through to the provider builder's
    bare "no model configured" where `require_runnable` names the remedy."""
    from types import SimpleNamespace

    from agent6.config import ConfigError
    from agent6.ui.cli import review_cmds

    base, head = _two_commits(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(
        review_cmds,
        "load_effective",
        MagicMock(return_value=SimpleNamespace(config=_panel_config(with_reviewer=False))),
    )
    monkeypatch.setattr(review_cmds, "check_provider_keys", MagicMock(return_value=None))

    with pytest.raises(ConfigError, match="reviewer"):
        review_cmds._cmd_review(  # pyright: ignore[reportPrivateUsage]
            None, base=base, head=head, paths=(), reviewers=2, personas="security"
        )


@pytest.mark.parametrize("head", ["rel", ""])
def test_the_recent_log_survives_a_head_that_is_also_a_path_or_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], head: str
) -> None:
    """`git log <head>` failed on a ref that is also a path (ambiguous) and
    on the empty head `--base` alone leaves, so the review ran with no recent
    history at all; the rev is named as a rev and defaults to HEAD."""
    from types import SimpleNamespace

    from agent6.ui.cli import review_cmds

    base, b = _two_commits(tmp_path)
    subprocess.run(["git", "-C", str(tmp_path), "branch", "rel", b], check=True)
    (tmp_path / "rel").write_text("a path named like the branch\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "rel"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-qm",
            "rel-file",
        ],
        check=True,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    provider = _FixedReviewProvider()
    monkeypatch.setattr(
        review_cmds,
        "load_effective",
        MagicMock(return_value=SimpleNamespace(config=_panel_config(with_reviewer=True))),
    )
    monkeypatch.setattr(review_cmds, "check_provider_keys", MagicMock(return_value=None))
    monkeypatch.setattr(review_cmds, "build_role_provider", MagicMock(return_value=provider))

    rc = review_cmds._cmd_review(  # pyright: ignore[reportPrivateUsage]
        None, base=base, head=head, paths=()
    )

    assert rc == 0
    assert "RECENT COMMITS:" in provider.last_user.split("\n\nDIFF:", 1)[0]


def test_personas_a_configured_roster_ignores_are_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With `[review].seats` naming the roster the command announces
    `--personas` ignored, then parsed them for the key check: a malformed
    spec crashed the command and a well-formed one refused the run."""
    from types import SimpleNamespace

    from agent6.config import Config
    from agent6.ui.cli import review_cmds

    base, head = _two_commits(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    cfg = Config.model_validate(
        {
            "providers": {"local": {"api_format": "openai", "base_url": "https://example.test/v1"}},
            "review": {"seats": ["security@local/m"]},
        }
    )
    seen: dict[str, Any] = {}

    def _keys(_cfg: Config, extra_providers: Any = ()) -> str:
        seen["extra"] = list(extra_providers)
        return "stop here"

    monkeypatch.setattr(
        review_cmds, "load_effective", MagicMock(return_value=SimpleNamespace(config=cfg))
    )
    monkeypatch.setattr(review_cmds, "build_review_seats", MagicMock(return_value=[]))
    monkeypatch.setattr(review_cmds, "check_provider_keys", _keys)

    rc = review_cmds._cmd_review(  # pyright: ignore[reportPrivateUsage]
        None, base=base, head=head, paths=(), reviewers=1, personas="tests@oops"
    )

    assert rc == 2
    assert seen["extra"] == []
    assert "stop here" in capsys.readouterr().err


def test_a_tracked_file_named_head_does_not_break_the_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`git diff HEAD` with no pathspec is ambiguous in a repo tracking a file
    named HEAD, so the review failed before it read a line."""
    from types import SimpleNamespace

    from agent6.ui.cli import review_cmds

    _two_commits(tmp_path)
    (tmp_path / "HEAD").write_text("h\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "HEAD"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.email=t@example.com", "commit", "-qm", "HEAD"],
        check=True,
    )
    (tmp_path / "caller.py").write_text("print(1)\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    provider = _FixedReviewProvider()
    monkeypatch.setattr(
        review_cmds,
        "load_effective",
        MagicMock(return_value=SimpleNamespace(config=_panel_config(with_reviewer=True))),
    )
    monkeypatch.setattr(review_cmds, "check_provider_keys", MagicMock(return_value=None))
    monkeypatch.setattr(review_cmds, "build_role_provider", MagicMock(return_value=provider))

    rc = review_cmds._cmd_review(None, base="", head="", paths=())  # pyright: ignore[reportPrivateUsage]

    assert rc == 0, capsys.readouterr().err
    assert "print(1)" in provider.last_user
