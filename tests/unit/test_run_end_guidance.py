from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any

import pytest

from agent6 import budget, errors, paths
from agent6.app import finalize, reporter, role_providers
from agent6.harness import loop_snapshot
from agent6.providers import ProviderResponse
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import cli_main
from agent6.vcs import git_read


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    _git(checkout, "init", "-q", "-b", "main")
    (checkout / "seed.txt").write_text("seed\n", encoding="utf-8")
    (checkout / ".gitignore").write_text("output/\n", encoding="utf-8")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "seed")
    return checkout


@pytest.fixture
def layout(tmp_path: pathlib.Path, repo: pathlib.Path) -> sessions_layout.SessionLayout:
    result = sessions_layout.SessionLayout(state_dir=tmp_path / "state", session_id="guidance")
    result.ensure()
    result.manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": result.session_id,
                "mode": "run",
                "base_sha": _git(repo, "rev-parse", "HEAD"),
                "base_branch": "main",
                "harness": {"gate": {"command": ["true"], "source": "configured"}},
            }
        ),
        encoding="utf-8",
    )
    return result


def _end(
    repo: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    *,
    policy: str,
    completed: bool = True,
    hidden: frozenset[str] = frozenset(),
) -> str:
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        git_control="model" if policy == "model" else "agent6",
        policy={"commit_per_turn": policy != "no_commit"},
    )
    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = loop_snapshot.SessionResult(
        completed=completed,
        reason="finish_session" if completed else "provider_error",
        summary="done" if completed else "provider failed",
        turns=2,
        tool_calls=1,
        verification="unverified",
    )
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "session_id": layout.session_id, "user_task": "t"})
        + "\n"
        + json.dumps(
            {"type": "session.stop", "reason": result.reason, "verification": result.verification}
        )
        + "\n",
        encoding="utf-8",
    )
    lines: list[str] = []
    finalize.print_session_end(
        result,
        layout=layout,
        cwd=repo,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1),
        console_stream=False,
        reporter=reporter.Reporter(out=lines.append, err=lines.append),
        hidden=hidden,
    )
    return "\n".join(lines)


@pytest.mark.parametrize("policy", ["model", "no_commit", "failed_commit"])
@pytest.mark.parametrize("path", ["seed.txt", "new.txt"])
@pytest.mark.parametrize("completed", [True, False])
def test_visible_uncommitted_work_gets_verification_guidance(
    repo: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    policy: str,
    path: str,
    completed: bool,
) -> None:
    """Neither missing chain commits nor untracked files mean a run left no work."""
    (repo / path).write_text("work\n", encoding="utf-8")
    out = _end(repo, layout, policy=policy, completed=completed)
    assert "nothing verified the final tree" in out
    if completed:
        assert 'resume and run the gate:  agent6 resume guidance --steer "verify"' in out
    else:
        assert "resume with:  agent6 resume guidance" in out and "--steer" not in out
    assert "changes merged" not in out and "changes are on" not in out
    if not completed:
        assert "failed · provider error" in out
    elif policy == "model":
        assert "git was model-controlled" in out
        expected = (
            "tracked files with uncommitted changes: 1"
            if path == "seed.txt"
            else "new untracked files: 1"
        )
        assert expected in out
        assert "status/log/diff" in out
    elif policy == "no_commit":
        assert "commit_per_turn = false" in out and "WARNING" not in out
    else:
        assert "WARNING: the run finished with no commit" in out


@pytest.mark.parametrize("policy", ["model", "no_commit", "failed_commit"])
@pytest.mark.parametrize("dirt", ["none", "preexisting", "hidden", "ignored"])
def test_excluded_only_work_stays_quiet(
    repo: pathlib.Path, layout: sessions_layout.SessionLayout, policy: str, dirt: str
) -> None:
    """The status check uses stored untracked-at-start paths and hidden directory exclusions."""
    hidden = frozenset[str]()
    if dirt != "none":
        folder = "output" if dirt == "ignored" else "notes"
        (repo / folder).mkdir()
        (repo / folder / "file.txt").write_text("not run work\n", encoding="utf-8")
        if dirt == "preexisting":
            sessions_layout.write_untracked_at_start(layout.session_dir, {"notes/file.txt"})
            (repo / folder / "file.txt").write_text("edited but still excluded\n", encoding="utf-8")
        elif dirt == "hidden":
            hidden = frozenset({"notes"})
    out = _end(repo, layout, policy=policy, hidden=hidden)
    assert "nothing verified" not in out and "--steer" not in out
    assert "git was model-controlled" not in out and "WARNING" not in out


@pytest.mark.parametrize("policy", ["model", "failed_commit"])
def test_committed_work_keeps_verification_guidance(
    repo: pathlib.Path, layout: sessions_layout.SessionLayout, policy: str
) -> None:
    """A clean status cannot hide changed model HEAD or the harness chain's work."""
    (repo / "seed.txt").write_text("committed work\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "work")
    if policy != "model":
        _git(repo, "update-ref", git_read.chain_ref_for(layout.session_id), "HEAD")
        _git(repo, "checkout", "--detach", "HEAD~")
    out = _end(repo, layout, policy=policy)
    assert "nothing verified the final tree" in out and '--steer "verify"' in out


@pytest.mark.parametrize("policy", ["model", "no_commit", "failed_commit"])
def test_unreadable_status_does_not_hide_verification_guidance(
    repo: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    """Failure to read Git is not evidence that the run has nothing to verify."""

    def unavailable(*_args: object, **_kwargs: object) -> git_read.GitStatus:
        raise errors.GitError("status unavailable")

    monkeypatch.setattr(git_read, "status", unavailable)
    out = _end(repo, layout, policy=policy)
    assert "nothing verified the final tree" in out


@pytest.mark.parametrize("policy", ["model", "no_commit", "failed_commit"])
def test_an_unknown_base_cannot_establish_that_no_work_was_left(
    repo: pathlib.Path, layout: sessions_layout.SessionLayout, policy: str
) -> None:
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    manifest.pop("base_sha")
    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "nothing verified the final tree" in _end(repo, layout, policy=policy)


class _Scripted:
    def __init__(self, path: str) -> None:
        self.path = path
        self.calls = 0

    def call(self, **_kwargs: Any) -> ProviderResponse:
        self.calls += 1
        tool = (
            {
                "name": "apply_edit",
                "input": {"path": self.path, "kind": "overwrite", "new_string": "work\n"},
            }
            if self.calls == 1
            else {"name": "finish_session", "input": {"summary": "done"}}
        )
        tools = ({"type": "tool_use", "id": str(self.calls), **tool},)
        return ProviderResponse(
            text="",
            tool_uses=tools,
            stop_reason="tool_use",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw={"content": list(tools)},
        )


@pytest.mark.parametrize(
    ("policy", "path"),
    [
        ("model", "new.txt"),
        ("no_commit", "seed.txt"),
        ("failed_commit", "seed.txt"),
        ("failed_commit", "new.txt"),
    ],
)
def test_cli_records_unverified_work_and_prints_its_guidance(
    tmp_path: pathlib.Path,
    repo: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    policy: str,
    path: str,
) -> None:
    """Commands must be allowed for an unrun configured gate to record unverified, not N/A."""
    if policy == "failed_commit":
        ref_lock = repo / ".git" / (git_read.chain_ref_for("guidance") + ".lock")
        ref_lock.parent.mkdir(parents=True)
        ref_lock.touch()
    cfg = tmp_path / "config" / "agent6"
    cfg.mkdir(parents=True)
    cfg.joinpath("config.toml").write_text(
        '[sandbox]\nisolation = "none"\nrun_commands = "allow"\n'
        '[models.default]\nmodel = "p/m"\n[models.worker]\nmodel = "p/m"\n'
        '[providers.p]\nbackend = "openai"\nbase_url = "http://127.0.0.1:1"\n'
        'api_key_env = "PROBE_KEY"\n[review]\ntrigger = "off"\n'
        '[harness]\nverify_command = ["true"]\nverify_at_start = false\nverify_when = "never"\n'
        + {
            "model": '[git]\ncontrol = "model"\n',
            "no_commit": "[git]\ncommit_per_turn = false\n",
            "failed_commit": "",
        }[policy],
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg.parent))
    monkeypatch.setenv("PROBE_KEY", "test-key")
    monkeypatch.chdir(repo)
    provider = _Scripted(path)
    monkeypatch.setattr(role_providers, "build_role_provider", lambda *_args, **_kwargs: provider)
    assert cli_main(["run", "--session-id", "guidance", "write work"]) == (
        5 if policy == "failed_commit" else 0
    )
    out = capsys.readouterr().out
    session_dir = paths.state_dir(repo) / "sessions" / "runs" / "guidance"
    events = [json.loads(line) for line in (session_dir / "logs.jsonl").read_text().splitlines()]
    stop = next(event for event in events if event["type"] == "session.stop")
    assert stop["verification"] == "unverified" and stop["reason"] == "finish_session"
    assert not any(event["type"] == "verify.end" for event in events)
    assert _git(repo, "status", "--porcelain") == (
        "?? new.txt" if path == "new.txt" else "M seed.txt"
    )
    if policy == "failed_commit":
        assert "commit failed" in out
        assert git_read.chain_tip(repo, git_read.chain_ref_for("guidance")) is None
    assert ("not committed" if policy == "failed_commit" else "finished · unverified") in out
    assert "nothing verified the final tree" in out and '--steer "verify"' in out
    assert "changes merged" not in out
    if policy == "model":
        assert "git was model-controlled" in out and "new untracked files: 1" in out
