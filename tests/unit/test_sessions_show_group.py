# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`sessions show` lists a parent's children and names a child's parent."""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import paths
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import main


def _session(repo: pathlib.Path, session_id: str, manifest: dict[str, object]) -> None:
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id=session_id)
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps({"version": 3, "session_id": session_id, "mode": "run", **manifest}),
        encoding="utf-8",
    )
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "session.stop", "verification": "passed", "reason": "finish_session"})
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def group(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _session(
        repo,
        "fan",
        {
            "comparison": {
                "group": "fan",
                "ranking": ["fan-l2", "fan-l1"],
                "ranked_by": "judge",
                "rationale": "covers the edge case",
                "judge_cost_usd": 0.0123,
                "judge_cost_partial": True,
            },
        },
    )
    for child in (1, 2):
        _session(
            repo,
            f"fan-l{child}",
            {
                "models": {
                    "driver": {"provider": "openai", "model": f"child-{child}"},
                    "driver_from_flag": True,
                },
                "parallel": {"group": "fan", "number": child, "parent": "fan"},
                "placement": {"rank": 3 - child, "of": 2, "winner": child == 2, "group": "fan"},
            },
        )
    return repo


def test_show_names_the_driving_route_and_its_model_flag(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _session(
        repo,
        "flagged",
        {
            "models": {
                "driver": {"provider": "openrouter", "model": "moonshotai/kimi-k2.6"},
                "driver_from_flag": True,
            }
        },
    )

    assert main(["sessions", "show", "flagged"]) == 0
    assert "model:      openrouter/moonshotai/kimi-k2.6 (from --model)" in capsys.readouterr().out
    assert main(["sessions", "show", "flagged", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model"] == "openrouter/moonshotai/kimi-k2.6"
    assert data["model_from_flag"] is True


def test_show_names_the_preset_the_run_continues_under(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _session(repo, "preset", {"harness": {"preset": "deep_review", "preset_from_flag": True}})

    assert main(["sessions", "show", "preset"]) == 0
    assert "preset:     deep_review" in capsys.readouterr().out
    assert main(["sessions", "show", "preset", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["preset"] == "deep_review"


def test_show_labels_the_session_status(
    group: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Text and JSON call the session's lifecycle result status."""
    assert main(["sessions", "show", "fan-l1"]) == 0
    out = capsys.readouterr().out
    assert "status:     passed" in out
    assert "state:" not in out
    assert main(["sessions", "show", "fan-l1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


def test_show_on_a_parent_lists_its_children(
    group: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sessions", "show", "fan"]) == 0
    out = capsys.readouterr().out
    assert "children:      2\n" in out
    children = [ln for ln in out.splitlines() if "fan-l" in ln]
    assert len(children) == 2 and "fan-l1" in children[0] and "fan-l2" in children[1]
    assert children[1].split()[:2] == ["rank", "1/2"] and "passed" in children[1]
    assert main(["sessions", "show", "fan", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [ln["session_id"] for ln in data["children"]] == ["fan-l1", "fan-l2"]
    assert data["children"][1]["winner"] is True


def test_show_on_a_parent_lists_each_children_route(
    group: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sessions", "show", "fan"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "fan-l" in ln]
    assert "openai/child-1" in lines[0]
    assert "openai/child-2" in lines[1]

    assert main(["sessions", "show", "fan", "--json"]) == 0
    children = json.loads(capsys.readouterr().out)["children"]
    assert [child["model"] for child in children] == ["openai/child-1", "openai/child-2"]
    assert all(child["model_from_flag"] is True for child in children)


def test_stale_parent_does_not_offer_resume(
    group: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale group parent has no loop for a resume command to continue."""
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(group), session_id="fan")
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )

    assert main(["sessions", "show", "fan"]) == 0

    out = capsys.readouterr().out
    assert "status:     stale" in out
    assert "resume with:" not in out


def test_show_keeps_a_failed_setups_output_and_network_diagnostic(
    group: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A child's setup failure remains readable when the clone and child streams are gone."""
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(group), session_id="fan-l1")
    problem = (
        "the setup command failed (exit 1): uv sync --offline\n"
        "    no cached package\n"
        "  the command ran with no route off this machine (sandbox.network), so it has to"
        " work offline. See docs/security.md."
    )
    events = [
        {"type": "session.start", "mode": "run"},
        {"type": "worktree.setup", "command": ["uv", "sync", "--offline"], "exit_code": 1},
        {
            "type": "session.stop",
            "reason": "setup_command_failed",
            "verification": "unverified",
            "summary": problem,
        },
    ]
    layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    assert main(["sessions", "show", "fan-l1"]) == 0
    shown = capsys.readouterr().out
    assert "failed · setup command failed" in shown
    assert "ended:      the setup command failed (exit 1): uv sync --offline\n" in shown
    assert "setup:      no cached package\n" in shown and "(sandbox.network)" in shown
    assert main(["sessions", "show", "fan-l1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["setup_output"] == problem


def test_show_on_a_child_names_its_parent(
    group: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sessions", "show", "fan-l1"]) == 0
    out = capsys.readouterr().out
    assert "child of:    fan (child 1 of group fan)" in out
    assert main(["sessions", "show", "fan-l1", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["parallel"] == {
        "group": "fan",
        "number": 1,
        "parent": "fan",
        "config_notes": [],
    }
    assert data["children"] == []


def test_show_marks_a_child_unmerged_like_the_listing(
    group: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A child whose branch holds commits its base lacks reads unmerged in the group view."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(group), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    (group / "a.txt").write_text("a\n", encoding="utf-8")
    git("add", "a.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("branch", "agent6/fan-l1", base)
    git("checkout", "-q", "agent6/fan-l1")
    (group / "b.txt").write_text("b\n", encoding="utf-8")
    git("add", "b.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "child work")
    git("checkout", "-q", "main")
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(group), session_id="fan-l1")
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    manifest.update({"base_sha": base, "base_branch": "main", "run_branch": "agent6/fan-l1"})
    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert main(["sessions", "show", "fan"]) == 0
    out = capsys.readouterr().out
    (line,) = [ln for ln in out.splitlines() if "fan-l1" in ln]
    assert "unmerged" in line
    assert main(["sessions", "show", "fan", "--json"]) == 0
    children = json.loads(capsys.readouterr().out)["children"]
    assert children[0]["session_id"] == "fan-l1" and children[0]["unmerged"] is True


def test_show_usage_carries_the_cached_tokens_and_the_listings_cost_cell(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The usage line carries the cached input side and the listing's usage cell.

    It kept only the uncached `in=` (a 500k-token run read as `in=18`) and spelled a clean $0
    as `usage $0.0000` where the listing blanks it.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _session(repo, "cached", {})
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="cached")
    budget = {
        "type": "budget.update",
        "input_total": 18,
        "output_total": 2194,
        "cache_read_total": 42486,
        "cache_write_total": 22617,
        "usd_total": 0.0,
    }
    lines = layout.logs_path.read_text(encoding="utf-8").splitlines()
    layout.logs_path.write_text(
        "\n".join([lines[0], json.dumps(budget), *lines[1:]]) + "\n", encoding="utf-8"
    )
    assert main(["sessions", "show", "cached"]) == 0
    usage = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("usage:"))
    assert usage.split(None, 1)[1] == "in=18 out=2194 cache_r=42486 cache_w=22617"
    layout.logs_path.write_text(
        "\n".join([lines[0], json.dumps({**budget, "usd_total": 0.5}), *lines[1:]]) + "\n",
        encoding="utf-8",
    )
    assert main(["sessions", "show", "cached"]) == 0
    usage = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("usage:"))
    assert usage.endswith("cache_w=22617  usage $0.50")


def test_show_usage_and_json_carry_the_plan_points(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A plan-metered run shows the points this execution consumed outside the live views.

    The usage line and the JSON carry them against [budget].max_plan_points; a dollar-metered run
    shows none.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _session(repo, "capped", {})
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="capped")
    budget = {
        "type": "budget.update",
        "input_total": 18,
        "output_total": 2194,
        "usd_total": 0.0,
        "plan_used_percent": 13.0,
        "plan_observed_points": 2.5,
        "plan_cap": 6.0,
    }
    lines = layout.logs_path.read_text(encoding="utf-8").splitlines()
    layout.logs_path.write_text(
        "\n".join([lines[0], json.dumps(budget), *lines[1:]]) + "\n", encoding="utf-8"
    )
    assert main(["sessions", "show", "capped"]) == 0
    usage = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("usage:"))
    assert usage.split(None, 1)[1] == "in=18 out=2194 plan moved=2.5/6pt  usage 2.5pt"
    assert main(["sessions", "show", "capped", "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    budget_json = obj["budget"]
    assert budget_json["session"]["plan_observed_points"] == 2.5
    assert budget_json["execution"]["plan_cap"] == 6.0
    layout.logs_path.write_text(
        "\n".join([lines[0], json.dumps({**budget, "plan_cap": 0.0}), *lines[1:]]) + "\n",
        encoding="utf-8",
    )
    assert main(["sessions", "show", "capped"]) == 0
    usage = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("usage:"))
    assert "plan" not in usage


def test_show_names_the_session_that_seeded_a_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions show` prints a `--from` seed as `seeded from` and carries it in the JSON."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _session(repo, "seeded", {"source_session_id": "seed-src-AAAAAA"})
    assert main(["sessions", "show", "seeded"]) == 0
    assert "seeded from: seed-src-AAAAAA" in capsys.readouterr().out
    assert main(["sessions", "show", "seeded", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["source_session_id"] == "seed-src-AAAAAA"


def test_show_names_the_questions_nobody_answered(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions show` names a question that waited unanswered in the transcript.

    Asked while no operator was attached, it got empty answers and the run went on.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _session(repo, "asked", {})
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="asked")
    lines = layout.logs_path.read_text(encoding="utf-8").splitlines()
    asked = [
        {"type": "question.prompt", "id": "q1", "questions": [{"question": "Which?"}]},
        {
            "type": "question.answer",
            "id": "q1",
            "answers": [""],
            "source": "headless-default",
            "unseen": True,
        },
    ]
    layout.logs_path.write_text(
        "\n".join([lines[0], *(json.dumps(e) for e in asked), *lines[1:]]) + "\n", encoding="utf-8"
    )
    assert main(["sessions", "show", "asked"]) == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("questions:"))
    assert (
        line.split(None, 1)[1]
        == "1 unanswered (nobody was attached): agent6 sessions transcript asked"
    )
    assert main(["sessions", "show", "asked", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["unattended_questions"] == 1


def test_show_drops_the_questions_line_after_a_resume(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions show` counts the questions nobody saw in the current execution.

    A run resumed after such a question kept its `questions:` line and its JSON count.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _session(repo, "asked", {})
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="asked")
    lines = layout.logs_path.read_text(encoding="utf-8").splitlines()
    asked = [
        {"type": "question.prompt", "id": "q1", "questions": [{"question": "Which?"}]},
        {
            "type": "question.answer",
            "id": "q1",
            "answers": [""],
            "source": "headless-default",
            "unseen": True,
        },
    ]
    resumed = [
        {"type": "session.resume", "turn": 3},
        {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
    ]
    layout.logs_path.write_text(
        "\n".join([lines[0], *(json.dumps(e) for e in asked), *lines[1:]])
        + "\n"
        + "".join(json.dumps(e) + "\n" for e in resumed),
        encoding="utf-8",
    )
    assert main(["sessions", "show", "asked"]) == 0
    assert "questions:" not in capsys.readouterr().out
    assert main(["sessions", "show", "asked", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["unattended_questions"] == 0


def test_show_names_a_failed_end_once_beside_the_unanswered_count(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The state line of a failed run carries the unanswered count, and its reason once.

    The count was dropped beside any other detail; joined on, the raw end reason must not
    follow the label that already spells it.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _session(repo, "capped", {})
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="capped")
    events: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "question.prompt", "id": "q1", "questions": [{"question": "Which?"}]},
        {
            "type": "question.answer",
            "id": "q1",
            "answers": [""],
            "source": "headless-default",
            "unseen": True,
        },
        {"type": "session.stop", "verification": "unverified", "reason": "max_turns"},
    ]
    layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    assert main(["sessions", "show", "capped"]) == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("status:"))
    assert line == "status:     failed · max turns · 1 question unanswered"


def test_show_offers_plan_show_only_once_the_plan_is_written(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A plan stopped before it wrote plan.md gets no `plan show` line it cannot honour."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(repo), session_id="plan-AAAA11", subdir="plans"
    )
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps({"version": 3, "session_id": "plan-AAAA11", "mode": "plan"}), encoding="utf-8"
    )
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "plan", "user_task": "t"})
        + "\n"
        + json.dumps(
            {"type": "session.stop", "verification": "unverified", "reason": "steer_abort"}
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["sessions", "show", "plan-AAAA11"]) == 0
    assert "plan show" not in capsys.readouterr().out

    (layout.session_dir / "plan.md").write_text("# Plan\n", encoding="utf-8")
    assert main(["sessions", "show", "plan-AAAA11"]) == 0
    assert "plan:       agent6 plan show plan-AAAA11" in capsys.readouterr().out


def _log(repo: pathlib.Path, session_id: str, events: list[dict[str, object]]) -> None:
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id=session_id)
    layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def _stopped_short(repo: pathlib.Path, session_id: str, reason: str) -> None:
    _log(
        repo,
        session_id,
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.stop", "verification": "unverified", "reason": reason},
        ],
    )
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id=session_id)
    (layout.session_dir / "loop_state.json").write_text("{}", encoding="utf-8")  # a turn ran


@pytest.mark.parametrize("reason", ["steer_abort", "budget_exhausted"])
def test_show_offers_resume_for_a_run_that_stopped_short(
    group: pathlib.Path, capsys: pytest.CaptureFixture[str], reason: str
) -> None:
    """A stopped or failed run prints the resume command, as a stale one does."""
    _session(group, "solo-1", {})
    _stopped_short(group, "solo-1", reason)

    assert main(["sessions", "show", "solo-1"]) == 0

    assert "resume with:  agent6 resume solo-1" in capsys.readouterr().out


def test_show_offers_fork_for_a_child_that_stopped_short(
    group: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stopped child names `fork`: the resume command it printed was one resume refuses."""
    _stopped_short(group, "fan-l1", "steer_abort")

    assert main(["sessions", "show", "fan-l1"]) == 0

    out = capsys.readouterr().out
    assert "resume with:" not in out
    assert (
        "resume:     a child of the group fan, whose work is on its own branch; `agent6 fork" in out
    )
    assert 'fan-l1 --steer "..."` continues it in a worktree of its own' in out


def test_show_json_and_the_listing_name_the_stop_reason_alike(
    group: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both JSON forms carry `stop_reason`; the listing's `reason` stays the label's detail."""
    assert main(["sessions", "show", "fan-l1", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["stop_reason"] == "finish_session" and "reason" not in shown

    assert main(["sessions", "list", "--json"]) == 0
    children = {r["session_id"]: r for r in json.loads(capsys.readouterr().out)[0]["children"]}
    assert (children["fan-l1"]["stop_reason"], children["fan-l1"]["status_detail"]) == (
        "finish_session",
        "",
    )


def test_show_nests_attempts_directly_under_the_verify_field(
    group: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verification attempts stay with their summary instead of trailing the other fields."""
    _log(
        group,
        "fan-l1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "verify.end", "exit_code": 1, "duration_s": 2.5},
            {"type": "verify.end", "exit_code": 0, "duration_s": 1.2},
            {"type": "pin.added", "text": "Keep the regression checks"},
            {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
        ],
    )
    assert main(["sessions", "show", "fan-l1"]) == 0
    out = capsys.readouterr().out
    assert (
        "verify:     green on the final tree\n"
        "            verify: ✗ exit 1 · 2.5s\n"
        "            verify: ✓ pass · 1.2s\n"
    ) in out
    assert out.count("verify:") == 3
    assert out.index("verify: ✓ pass") < out.index("pins:       Keep the regression checks")


def test_show_names_the_start_gates_verdict_beside_the_final_one(
    group: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The verify line says what the gate made of the tree before the run's first turn."""
    _log(
        group,
        "fan-l1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "verify.baseline", "ok": False, "turn": 0},
            {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
        ],
    )

    assert main(["sessions", "show", "fan-l1"]) == 0
    assert "verify:     green on the final tree (red before the run)" in capsys.readouterr().out

    assert main(["sessions", "show", "fan-l1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["baseline_ok"] is False


def test_status_json_keeps_comparison_on_its_owner(
    group: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Child status names the group; parent status carries its ranking and judge cost."""
    assert main(["sessions", "show", "fan-l2", "--json"]) == 0
    child = json.loads(capsys.readouterr().out)
    assert child["placement"] == {"group": "fan", "rank": 1, "of": 2, "winner": True}
    assert child["comparison"] is None
    assert main(["sessions", "show", "fan", "--json"]) == 0
    parent = json.loads(capsys.readouterr().out)
    assert parent["placement"] is None
    assert parent["comparison"] == {
        "group": "fan",
        "ranking": ["fan-l2", "fan-l1"],
        "ranked_by": "judge",
        "rationale": "covers the edge case",
        "judge_cost_usd": 0.0123,
        "judge_cost_partial": True,
    }
    assert main(["sessions", "show", "fan-l2"]) == 0
    assert "judge: covers the edge case" in capsys.readouterr().out


@pytest.fixture
def ordinary_parent(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A run that used `/parallel` once: no group stamp, two children by lineage, one record."""
    from agent6.sessions import child_dispatch

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    _session(repo, "co", {})
    for child in (1, 2):
        _session(
            repo,
            f"co-p1-l{child}",
            {
                "models": {"driver": {"provider": "openai", "model": f"child-{child}"}},
                "parallel": {"group": "co-p1", "number": child, "parent": "co"},
            },
        )
    parent = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="co")
    child_dispatch.write_record(
        parent.session_dir,
        child_dispatch.DispatchRecord(
            identity="co-p1",
            source="operator",
            owner="exec-1",
            turn=2,
            base="a" * 40,
            segments=(child_dispatch.SegmentRecord(task="audit", spec="2", children=2),),
            children=(
                child_dispatch.ChildRecord(
                    session_id="co-p1-l1",
                    number=1,
                    task="audit",
                    join=child_dispatch.JoinRecord(
                        branch="agent6/co-p1-l1", status="joined", sha="b" * 40
                    ),
                    usage=child_dispatch.UsageRecord(usd=0.5),
                    settled=True,
                ),
                child_dispatch.ChildRecord(
                    session_id="co-p1-l2",
                    number=2,
                    task="audit",
                    join=child_dispatch.JoinRecord(branch="agent6/co-p1-l2", status="no_changes"),
                    usage=child_dispatch.UsageRecord(usd=0.25),
                    settled=True,
                ),
            ),
            state="complete",
            message="[parallel] group co-p1 complete",
            delivered=True,
        ),
        now=1.0,
    )
    return repo


def test_show_on_an_ordinary_parent_lists_its_children_with_their_joins(
    ordinary_parent: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `/parallel` run is no group: no group line, its children listed with their outcomes."""
    assert main(["sessions", "show", "co"]) == 0
    out = capsys.readouterr().out
    assert "group:" not in out
    assert "children:      2" in out
    assert "child 1" in out and "co-p1-l1" in out and "openai/child-1" in out
    assert "joined at bbbbbbbbbbbb" in out
    assert "child 2" in out and "no changes" in out


def test_show_on_a_child_of_an_ordinary_parent_names_it(
    ordinary_parent: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sessions", "show", "co-p1-l1"]) == 0
    assert "child of:    co (child 1 of group co-p1)" in capsys.readouterr().out
