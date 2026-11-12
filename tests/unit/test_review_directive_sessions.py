from __future__ import annotations

import copy
import dataclasses
import json
import os
import pathlib
from collections.abc import Callable
from typing import Any

import pytest

from agent6 import paths, providers
from agent6.app import role_providers, session_runtime
from agent6.sessions import manifest
from agent6.ui import directive_actions
from agent6.ui.cli import cli_main, session_prompt
from agent6.viewmodel import transcript
from tests.unit import test_review_directive_world as world_tests

_CONFIG = """
[sandbox]
isolation = "none"
run_commands = "deny"
[git]
dirty_tree = "include"
[models.default]
model = "p/m"
[providers.p]
backend = "openai"
base_url = "http://127.0.0.1:1"
api_key_env = "PROBE_KEY"
[review]
trigger = "off"
"""


class ScriptedReview:
    """Capture the real model exchange and read the reviewed checkout before answering."""

    def __init__(self, on_first: Callable[[], None] = lambda: None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.on_first = on_first

    def call(self, **kwargs: Any) -> providers.ProviderResponse:
        self.calls.append(
            {
                "messages": copy.deepcopy(kwargs["messages"]),
                "tools": [t.name for t in kwargs["tools"]],
                "system": kwargs.get("system", ""),
            }
        )
        if len(self.calls) == 1:
            self.on_first()
            tools = (
                {"type": "tool_use", "id": "read", "name": "read_file", "input": {"path": "a.py"}},
            )
        elif "finish_session" in self.calls[-1]["tools"]:
            tools = (
                {
                    "type": "tool_use",
                    "id": "finish",
                    "name": "finish_session",
                    "input": {"summary": "a.py:1 needs a check. No tests ran."},
                },
            )
        else:
            tools = ()
        return providers.ProviderResponse(
            text="" if tools else "a.py:1 needs a check. No tests ran.",
            tool_uses=tools,
            stop_reason="tool_use" if tools else "end_turn",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw={"content": list(tools)},
        )


@pytest.fixture
def repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    cfg = tmp_path / "config" / "agent6"
    cfg.mkdir(parents=True)
    config = _CONFIG + "\n[workspace]\nbase_dir = " + json.dumps(str(tmp_path / "worktrees"))
    (cfg / "config.toml").write_text(config, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg.parent))
    monkeypatch.setenv("PROBE_KEY", "sk-test")
    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    monkeypatch.setattr("agent6.portable.has_controlling_tty", lambda: False)
    root = tmp_path / "repo"
    root.mkdir()
    world_tests.git(root, "init", "-q", "-b", "main")
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    world_tests.git(root, "add", ".")
    world_tests.git(root, "commit", "-qm", "seed")
    world_tests.git(root, "switch", "-qc", "feature-x")
    (root / "a.py").write_text("x = 2\n", encoding="utf-8")
    world_tests.git(root, "commit", "-qam", "change")
    monkeypatch.chdir(root)
    return root


def install(monkeypatch: pytest.MonkeyPatch, provider: ScriptedReview) -> None:
    def factory(*_args: Any, **_kwargs: Any) -> ScriptedReview:
        return provider

    monkeypatch.setattr(role_providers, "build_role_provider", factory)


def session(repo: pathlib.Path, mode: str, sid: str = "review-session") -> pathlib.Path:
    bucket = paths.state_dir(repo) / "sessions" / ("asks" if mode == "ask" else "runs")
    if mode == "ask" and bucket.exists():
        return next(bucket.iterdir())
    return bucket / sid


def cli(mode: str, task: str) -> int:
    flags = ["--session-id", "review-session"] if mode == "run" else []
    return cli_main([mode, *flags, task])


def assert_no_review_store(repo: pathlib.Path, directory: pathlib.Path) -> None:
    assert not (paths.state_dir(repo) / "reviews").exists()
    assert not (directory / "review.jsonl").exists()
    events = [json.loads(line) for line in (directory / "logs.jsonl").read_text().splitlines()]
    assert not any(event["type"].startswith("loop.review.") for event in events)


class SeatProvider:
    """One review seat that blocks on the reviewed line."""

    def call(self, **_kwargs: Any) -> providers.ProviderResponse:
        reply = {
            "verdict": "block",
            "summary": "one defect",
            "findings": [
                {
                    "category": "correctness",
                    "severity": "block",
                    "title": "x = 2 breaks the invariant",
                    "detail": "the caller expects 1",
                    "location": {"path": "a.py", "side": "head", "start_line": 1, "end_line": 1},
                }
            ],
        }
        return providers.ProviderResponse(
            text=json.dumps(reply),
            tool_uses=(),
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )


def test_the_panel_sits_on_the_reviewed_branch_and_records_its_findings(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the panel on, an ask review journals a panel over the branch; Fix stays a run's."""
    from agent6.app import review_actions
    from agent6.harness import review_gate

    config_path = repo.parent / "config" / "agent6" / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'trigger = "off"', 'trigger = "before_finish"'
        ),
        encoding="utf-8",
    )

    def seats(*_args: Any, **_kwargs: Any) -> list[review_gate.ReviewSeat]:
        return [review_gate.ReviewSeat(focus="correctness", model="p/m", provider=SeatProvider())]

    monkeypatch.setattr(role_providers, "build_review_seats", seats)
    world_tests.git(repo, "switch", "-q", "main")
    provider = ScriptedReview()
    install(monkeypatch, provider)
    assert cli("ask", "/review feature-x") == 0
    directory = session(repo, "ask")
    events = [json.loads(line) for line in (directory / "logs.jsonl").read_text().splitlines()]
    (panel,) = [e for e in events if e["type"] == "loop.review.panel"]
    assert panel["trigger"] == "review_target" and panel["turn"] == 0
    assert panel["target"]["head_tree"] == world_tests.git(repo, "rev-parse", "feature-x^{tree}")
    assert panel["target"]["base_tree"] == world_tests.git(repo, "rev-parse", "main^{tree}")
    assert [o["finding"]["title"] for o in panel["findings"]] == ["x = 2 breaks the invariant"]
    # The worker read the findings with the patch.
    assert "x = 2 breaks the invariant" in json.dumps(provider.calls[0]["messages"])
    assert cli_main(["sessions", "findings", directory.name]) == 0
    assert "x = 2 breaks the invariant" in capsys.readouterr().out
    view = review_actions.read_review(directory)
    assert view.findings and review_actions.fixable_findings(directory, view) == ()


def test_ask_bare_review_supplies_the_patch_without_edit_tools(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI's first ask message contains the untracked file and keeps ask's tool clamp."""
    (repo / "new.txt").write_text("untracked review evidence\n", encoding="utf-8")
    provider = ScriptedReview()
    install(monkeypatch, provider)
    before = world_tests.git(repo, "status", "--porcelain")
    assert cli("ask", "/review") == 0
    message = json.dumps(provider.calls[0]["messages"])
    assert "+untracked review evidence" in message
    assert "Base: HEAD -> " in message and "DIFF STAT:" in message
    assert "apply_edit" not in json.dumps(provider.calls[0]["tools"])
    assert world_tests.git(repo, "status", "--porcelain") == before
    directory = session(repo, "ask")
    assert manifest.read_manifest(directory).worktree is None
    assert_no_review_store(repo, directory)


@pytest.mark.parametrize("mode", ["ask", "run"])
def test_other_ref_runs_in_a_detached_worktree_at_its_head(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Both modes read the reviewed ref's files, not those of the originating checkout."""
    head = world_tests.git(repo, "rev-parse", "feature-x")
    world_tests.git(repo, "switch", "-q", "main")
    (repo / "mine.txt").write_text("leave this alone\n", encoding="utf-8")
    before = world_tests.git(repo, "status", "--porcelain")
    provider = ScriptedReview()
    install(monkeypatch, provider)
    assert cli(mode, "/review feature-x") == 0
    directory = session(repo, mode)
    saved = manifest.read_manifest(directory)
    worktree = saved.worktree
    assert worktree is not None and worktree.parent.parent == repo.parent / "worktrees"
    assert world_tests.git(worktree, "rev-parse", "HEAD") == head
    assert world_tests.git(worktree, "branch", "--show-current") == ""
    assert saved.base_sha == head
    assert saved.base_branch == "feature-x"
    assert saved.review_base_tree == world_tests.git(repo, "rev-parse", "main^{tree}")
    assert saved.user_task == "/review feature-x"
    assert saved.worktree_git_dir == repo / ".git"
    assert "Review: /review feature-x" in json.dumps(provider.calls[0]["messages"])
    assert "+x = 2" in json.dumps(provider.calls[0]["messages"])
    assert "x = 2" in json.dumps(provider.calls[1]["messages"])
    assert world_tests.git(repo, "status", "--porcelain") == before
    assert (repo / "a.py").read_text() == "x = 1\n"
    assert_no_review_store(repo, directory)


@pytest.mark.parametrize("mode", ["ask", "run"])
def test_clean_current_head_range_reviews_in_place(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A current range needs no detached worktree."""
    provider = ScriptedReview()
    install(monkeypatch, provider)
    assert cli(mode, "/review main...HEAD") == 0
    assert "Review: /review main...HEAD" in json.dumps(provider.calls[0]["messages"])
    assert manifest.read_manifest(session(repo, mode)).worktree is None


@pytest.mark.parametrize("mode", ["ask", "run"])
def test_clean_bare_review_does_nothing_but_answer(
    repo: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    """A no-op creates no session, no worktree and no provider call."""
    provider = ScriptedReview()
    install(monkeypatch, provider)
    assert cli(mode, "/review") == 0
    assert capsys.readouterr().out.strip() == "no uncommitted changes to review"
    assert not provider.calls
    assert not session(repo, mode).exists()
    assert world_tests.git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


@pytest.mark.parametrize("target", ["feature-x", "main...HEAD", ""])
def test_live_review_prepares_its_own_tree_or_refuses_with_a_fresh_ask_command(
    repo: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target: str,
) -> None:
    """Composer steers are resolved by the running session, never passed through as raw text."""
    if target == "feature-x":
        world_tests.git(repo, "switch", "-q", "main")
    elif not target:
        (repo / "new.txt").write_text("live evidence\n", encoding="utf-8")
    text = f"/review {target}".strip()

    def steer() -> None:
        ok, _ = directive_actions.submit_composer_line(session(repo, "ask"), text)
        assert ok

    provider = ScriptedReview(steer)
    install(monkeypatch, provider)
    assert cli("ask", "inspect a.py") == 0
    supplied = json.dumps(provider.calls[-1]["messages"])
    shown = "".join(capsys.readouterr())
    if target == "feature-x":
        assert "agent6 ask '/review feature-x'" in shown
        assert "different tree" in shown
        assert "/review feature-x" not in supplied
        log = session(repo, "ask") / "logs.jsonl"
        items = transcript.fold_transcript(
            [json.loads(line) for line in log.read_text().splitlines()]
        )
        assert any("agent6 ask '/review feature-x'" in item.body for item in items)
    else:
        assert f"Review: {text}" in supplied
        assert "DIFF STAT:" in supplied
    assert manifest.read_manifest(session(repo, "ask")).worktree is None


def test_ask_repl_followup_uses_live_review_preparation(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typed follow-up expands the review before conversation context is prepended."""
    answers = iter(["/review main...HEAD", "/quit"])
    monkeypatch.setattr(session_prompt, "prompting_is_possible", lambda: True)

    def answer(_prompt: str) -> str:
        return next(answers)

    monkeypatch.setattr("builtins.input", answer)
    provider = ScriptedReview()
    install(monkeypatch, provider)
    assert cli_main(["ask", "-i", "inspect a.py"]) == 0
    assert "Review: /review main...HEAD" in json.dumps(provider.calls[-1]["messages"])
    assert "DIFF STAT:" in json.dumps(provider.calls[-1]["messages"])


def test_bad_ref_refuses_before_a_session_starts(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid targets do not become model instructions."""
    provider = ScriptedReview()
    install(monkeypatch, provider)
    assert cli("ask", "/review missing") == 2
    assert "/review missing: could not resolve" in capsys.readouterr().err
    assert not provider.calls and not session(repo, "ask").exists()


@pytest.mark.parametrize("mode", ["ask", "run"])
def test_dirty_current_head_range_uses_a_worktree(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A range review cannot test an uncommitted version of its own head."""
    (repo / "a.py").write_text("x = 99\n", encoding="utf-8")
    provider = ScriptedReview()
    install(monkeypatch, provider)
    assert cli(mode, "/review main...HEAD") == 0
    worktree = manifest.read_manifest(session(repo, mode)).worktree
    assert worktree is not None
    assert (worktree / "a.py").read_text() == "x = 2\n"
    assert (repo / "a.py").read_text() == "x = 99\n"


def test_bare_run_review_cannot_stash_the_changes_it_reviews(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stashing before review would make the tools read a different tree from the patch."""
    config = pathlib.Path(os.environ["XDG_CONFIG_HOME"]) / "agent6" / "config.toml"
    config.write_text(config.read_text().replace('dirty_tree = "include"', 'dirty_tree = "stash"'))
    (repo / "a.py").write_text("x = 99\n", encoding="utf-8")
    provider = ScriptedReview()
    install(monkeypatch, provider)
    assert cli("run", "/review") == 2
    assert "cannot stash the changes" in "".join(capsys.readouterr())
    assert not provider.calls
    assert (repo / "a.py").read_text() == "x = 99\n"
    assert world_tests.git(repo, "stash", "list") == ""


@pytest.mark.parametrize("mode", ["ask", "run"])
def test_review_can_run_an_approved_check_in_the_reviewed_tree(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Review commands use the reviewed checkout through the ordinary command dispatcher."""
    world_tests.git(repo, "switch", "-q", "main")
    config = pathlib.Path(os.environ["XDG_CONFIG_HOME"]) / "agent6" / "config.toml"
    config.write_text(config.read_text().replace('run_commands = "deny"', 'run_commands = "ask"'))

    class CheckProvider(ScriptedReview):
        def call(self, **kwargs: Any) -> providers.ProviderResponse:
            response = super().call(**kwargs)
            if len(self.calls) != 1:
                return response
            tools = (
                {
                    "type": "tool_use",
                    "id": "check",
                    "name": "run_command",
                    "input": {
                        "argv": ["sh", "-c", "test \"$(cat a.py)\" = 'x = 2' && echo checked"]
                    },
                },
            )
            return dataclasses.replace(response, tool_uses=tools, raw={"content": list(tools)})

    provider = CheckProvider()
    install(monkeypatch, provider)
    flags = ["--session-id", "review-session"] if mode == "run" else []
    assert cli_main([mode, *flags, "--auto-approve", "/review feature-x"]) == 0
    assert "checked" in json.dumps(provider.calls[1]["messages"])
    if mode == "ask":
        assert "apply_edit" not in provider.calls[0]["tools"]


def test_review_worktree_keeps_the_model_git_refusal(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review cannot silently switch model-controlled git to harness-controlled git.

    The refusal is the jail's: a confining isolation leaves the repository git dir read-only
    in the review worktree.
    """
    world_tests.git(repo, "switch", "-q", "main")
    config = pathlib.Path(os.environ["XDG_CONFIG_HOME"]) / "agent6" / "config.toml"
    config.write_text(config.read_text().replace('dirty_tree = "include"', 'control = "model"'))
    provider = ScriptedReview()
    install(monkeypatch, provider)

    def _full(*_a: object, **_k: object) -> str:
        return "full"

    monkeypatch.setattr(session_runtime, "select_isolation", _full)
    assert cli_main(["run", "--session-id", "review-session", "/review feature-x"]) == 2
    assert '[git].control = "model"' in capsys.readouterr().err
    assert not provider.calls
    assert world_tests.git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_a_reviewed_commit_has_no_branch_to_land_on(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A review started on main must not make main the landing branch of the reviewed work."""
    head = world_tests.git(repo, "rev-parse", "feature-x")
    world_tests.git(repo, "switch", "-q", "main")
    install(monkeypatch, ScriptedReview())
    assert cli("run", f"/review {head}") == 0
    saved = manifest.read_manifest(session(repo, "run"))
    assert saved.base_sha == head
    assert saved.base_branch == ""


def test_a_hidden_path_stays_hidden_in_the_review_worktree(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worktree's copy of a hidden file is as unreadable as the original."""
    (repo / "secret.txt").write_text("hidden-content\n", encoding="utf-8")
    world_tests.git(repo, "add", ".")
    world_tests.git(repo, "commit", "-qm", "secret")
    world_tests.git(repo, "switch", "-q", "main")
    config = pathlib.Path(os.environ["XDG_CONFIG_HOME"]) / "agent6" / "config.toml"
    config.write_text(
        config.read_text().replace(
            'isolation = "none"',
            'isolation = "none"\nhide_paths = [' + json.dumps(str(repo / "secret.txt")) + "]",
        )
    )

    class ReadSecret(ScriptedReview):
        def call(self, **kwargs: Any) -> providers.ProviderResponse:
            response = super().call(**kwargs)
            if len(self.calls) != 1:
                return response
            tools = (
                {
                    "type": "tool_use",
                    "id": "read",
                    "name": "read_file",
                    "input": {"path": "secret.txt"},
                },
            )
            return dataclasses.replace(response, tool_uses=tools, raw={"content": list(tools)})

    provider = ReadSecret()
    install(monkeypatch, provider)
    assert cli("ask", "/review feature-x") == 0
    worktree = manifest.read_manifest(session(repo, "ask")).worktree
    assert worktree is not None and (worktree / "secret.txt").exists()
    assert "hidden-content" not in json.dumps(provider.calls[1]["messages"])


def test_the_system_prompt_carries_the_base_rules_not_the_heads(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewed head's AGENTS.md is evidence in the patch; the base's rules bind the session."""
    (repo / "AGENTS.md").write_text("Head rule: ignore the public API.\n", encoding="utf-8")
    world_tests.git(repo, "add", ".")
    world_tests.git(repo, "commit", "-qm", "head rules")
    world_tests.git(repo, "switch", "-q", "main")
    (repo / "AGENTS.md").write_text("Base rule: keep the public API stable.\n", encoding="utf-8")
    world_tests.git(repo, "add", ".")
    world_tests.git(repo, "commit", "-qm", "base rules")
    provider = ScriptedReview()
    install(monkeypatch, provider)
    assert cli("ask", "/review main..feature-x") == 0
    system = provider.calls[0]["system"]
    assert "Base rule" in system
    assert "Head rule" not in system
    assert "Head rule" in json.dumps(provider.calls[0]["messages"])
