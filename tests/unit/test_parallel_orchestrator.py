# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 run --parallel` fan-out orchestrator (ui/cli/parallel.py).

Driven with a fake spawner that, for each LaneSpec, really clones the origin,
commits on the lane's `agent6/<id>` branch, and fabricates a finished run dir
(manifest.json + logs.jsonl) -- so the orchestrator's clone-independent behavior
(symlink live view, import, lineage stamp, ranked report, resilience to a failed
lane) is exercised on real tmp git repos without spawning real runs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from agent6.app import parallel
from agent6.app.parallel import (
    LaneRuntime,
    ParallelError,
    build_lane_specs,
    run_parallel,
)
from agent6.config import Config
from agent6.directive import DirectiveError
from agent6.git_ops import branch_exists, commit_all, create_branch
from agent6.ui.cli import parallel as parallel_cmd
from agent6.ui.cli.parallel import lane_runtime
from agent6.workflows.subrun import LaneResult, LaneSpec, LaneTask, clone_workspace


def _git(repo: Path, *args: str) -> None:
    import subprocess

    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")


def _write_fake_run(run_dir: Path, task: str, *, status: str, cost: float) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"version": 2, "run_id": run_dir.name, "mode": "run", "user_task": task}) + "\n",
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
    else:  # "finished": a deliberate finish without all-passed
        events.append({"type": "run.end", "reason": "finish_run", "all_passed": False})
    (run_dir / "logs.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )


class _FakeSpawner:
    """A synchronous stand-in for the bridge spawner: clone, commit on the lane
    branch, and fabricate a finished run dir. Records what it observed so a test
    can assert the orchestrator's symlink-then-replace behavior."""

    def __init__(
        self,
        origin: Path,
        origin_state: Path,
        state_root: Path,
        *,
        fail: set[int] | None = None,
        status_by_lane: dict[int, str] | None = None,
        cost_by_lane: dict[int, float] | None = None,
        pid_lanes: set[int] | None = None,
    ) -> None:
        self.origin = origin
        self.origin_state = origin_state
        self.state_root = state_root
        self.fail = fail or set()
        self.status_by_lane = status_by_lane or {}
        self.cost_by_lane = cost_by_lane or {}
        # Lanes whose fabricated run dir carries a LIVE worker.pid (this test
        # process), simulating the teardown window where run.end is already in
        # logs.jsonl but the lane process has not yet cleared its pid.
        self.pid_lanes = pid_lanes or set()
        self.prior_link_was_symlink: dict[int, bool] = {}
        self.tasks: list[str] = []

    def __call__(self, spec: LaneSpec, task: str) -> LaneResult:
        self.tasks.append(task)
        branch = f"agent6/{spec.run_id}"
        if spec.lane > 1:  # observe the previous lane's live symlink
            prefix = spec.run_id.rsplit("-l", 1)[0]
            prior = self.origin_state / "runs" / f"{prefix}-l{spec.lane - 1}"
            self.prior_link_was_symlink[spec.lane] = prior.is_symlink()
        if spec.lane in self.fail:
            return LaneResult(
                spec=spec, run_dir=spec.workdir, branch=branch, ok=False, error="boom"
            )
        clone_workspace(self.origin, spec.workdir)
        create_branch(spec.workdir, branch)
        (spec.workdir / f"lane{spec.lane}.txt").write_text(f"lane {spec.lane}\n", encoding="utf-8")
        commit_all(spec.workdir, f"lane {spec.lane} work")
        run_dir = self.state_root / f"lane{spec.lane}" / "runs" / spec.run_id
        _write_fake_run(
            run_dir,
            task,
            status=self.status_by_lane.get(spec.lane, "passed"),
            cost=self.cost_by_lane.get(spec.lane, 0.05),
        )
        if spec.lane in self.pid_lanes:
            (run_dir / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
        return LaneResult(spec=spec, run_dir=run_dir, branch=branch, ok=True, error="")


@pytest.fixture
def origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    o = tmp_path / "origin"
    _init_repo(o)
    return o


@pytest.fixture
def runtime() -> LaneRuntime:
    """The real front-end LaneRuntime the pipeline drives (detached process spawn
    + run-dir bridge + reviewer/judging wiring). Tests faking one primitive use
    `dataclasses.replace` on it (e.g. a fake `spawn` or `worker_is_alive`)."""
    return lane_runtime()


def _specs(tmp_path: Path, cfg: Config, fanout_id: str, spec: str) -> list[LaneSpec]:
    return build_lane_specs(
        spec, cfg=cfg, fanout_id=fanout_id, workdir_root=tmp_path / "work" / fanout_id
    )


# ---------------------------------------------------------------------------
# Lane planning
# ---------------------------------------------------------------------------


def test_build_lane_specs_int_layout(tmp_path: Path) -> None:
    lanes = _specs(tmp_path, Config(), "fan", "3")
    assert [(ln.lane, ln.run_id, ln.model) for ln in lanes] == [
        (1, "fan-l1", None),
        (2, "fan-l2", None),
        (3, "fan-l3", None),
    ]
    assert lanes[0].workdir == tmp_path / "work" / "fan" / "lane-1"


def test_build_lane_specs_model_list(tmp_path: Path) -> None:
    lanes = _specs(tmp_path, Config(), "fan", "kimi,glm")
    assert [(ln.lane, ln.model) for ln in lanes] == [(1, "kimi"), (2, "glm")]


def test_build_lane_specs_over_cap_refused(tmp_path: Path) -> None:
    cfg = Config.model_validate({"parallel": {"max_lanes": 2}})
    with pytest.raises(ParallelError, match="max_lanes"):
        _specs(tmp_path, cfg, "fan", "5")


def test_build_lane_specs_rejects_zero(tmp_path: Path) -> None:
    # Spec-shape errors come from the shared grammar (parse_spec); the cap error
    # stays a ParallelError (a ui/cli concern that reads [parallel].max_lanes).
    with pytest.raises(DirectiveError):
        _specs(tmp_path, Config(), "fan", "0")


# ---------------------------------------------------------------------------
# Pre-spawn model validation (B3): a bogus model refuses before any clone.
# ---------------------------------------------------------------------------


def _write_models_cache(cache_home: Path, provider: str, models: list[str]) -> None:
    p = cache_home / "models" / f"{provider}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"models": models}), encoding="utf-8")


def _provider_cfg(model: str = "moonshotai/kimi-k2.6") -> Config:
    return Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {"worker": {"provider": "o", "model": model}},
        }
    )


def test_dispatch_parallel_refuses_unknown_model_before_any_clone(
    origin: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    _write_models_cache(tmp_path / "cache", "o", ["moonshotai/kimi-k2.6", "z-ai/glm-4.6"])

    def _boom(*_a: object, **_k: object) -> int:
        raise AssertionError("run_parallel must not be reached on a refusal")

    monkeypatch.setattr(parallel_cmd, "run_parallel", _boom)
    rc = parallel_cmd.dispatch_parallel(
        _provider_cfg(), "fix the bug", "moonshotai/kimi-k2.7", cwd=origin
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "REFUSING" in err
    assert "unknown model 'moonshotai/kimi-k2.7'" in err
    assert "closest: moonshotai/kimi-k2.6" in err


def test_dispatch_parallel_unknown_model_no_cache_warns_and_proceeds(
    origin: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))  # empty: no snapshot

    reached: list[str] = []

    def _fake_run(task: str, lanes: object, **_k: object) -> int:
        reached.append(task)
        return 0

    monkeypatch.setattr(parallel_cmd, "run_parallel", _fake_run)
    rc = parallel_cmd.dispatch_parallel(_provider_cfg(), "fix the bug", "made-up/model", cwd=origin)
    assert rc == 0
    assert reached == ["fix the bug"]  # not blocked offline
    assert "WARNING" in capsys.readouterr().err


def test_dispatch_parallel_forwards_auto_approve_to_run_parallel(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI's `--auto-approve` must reach `run_parallel`, same as --max-usd."""
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    captured: list[object] = []

    def _fake_run(task: str, lanes: object, **kw: object) -> int:
        captured.append(kw.get("auto_approve"))
        return 0

    monkeypatch.setattr(parallel_cmd, "run_parallel", _fake_run)
    parallel_cmd.dispatch_parallel(
        _provider_cfg(), "fix the bug", "made-up/model", cwd=origin, auto_approve=True
    )
    parallel_cmd.dispatch_parallel(_provider_cfg(), "fix the bug", "made-up/model", cwd=origin)

    assert captured == [True, False]


def test_coordinator_dispatch_refuses_unknown_model(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """The ui-built group dispatcher validates before cloning: an unknown model
    raises, and the loop's group-failure feedback (its `except Exception`) carries
    the message to the coordinator -- so workflows needs no models dependency."""
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    _write_models_cache(tmp_path / "cache", "o", ["moonshotai/kimi-k2.6"])
    origin_state = tmp_path / "ostate"
    origin_state.mkdir()

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("clone must not happen before validation")

    monkeypatch.setattr(parallel, "clone_workspace", _boom)
    dispatch = parallel.build_lane_spawner(
        _provider_cfg(), origin, origin_state, coordinator_run_id="coord", runtime=runtime
    )
    with pytest.raises(ParallelError, match=r"unknown model 'moonshotai/kimi-k2\.7'"):
        dispatch([LaneTask(task="do it", model="moonshotai/kimi-k2.7")], "p1")


def test_bridge_spawner_argv_ends_options_before_task(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    """The lane spawner puts every flag before `--` and the task after it, so a
    task that looks like a flag can never be parsed as one (matches web/TUI). The
    agent6 executable is folded into the injected `spawn`, so the argv it receives
    starts at the subcommand."""
    captured: list[list[str]] = []

    def fake_spawn(argv: list[str], workdir: Path, **_k: object) -> tuple[Path, str]:
        captured.append(list(argv))
        return workdir, ""

    cfg = Config()
    spec = LaneSpec(lane=1, run_id="fan-l1", workdir=tmp_path / "work" / "lane-1", model=None)
    parallel.bridge_spawner(
        spec, "--allow-root pwn", cfg=cfg, origin=origin, max_usd=2.0,
        runtime=replace(runtime, spawn=fake_spawn),
    )  # fmt: skip

    argv = captured[-1]
    assert argv[:1] == ["run"]  # the exe is prepended inside the injected spawn
    dd = argv.index("--")
    assert {"--run-id", "--config", "--max-usd"} <= set(argv[:dd])  # flags precede `--`
    assert argv[dd + 1 :] == ["--allow-root pwn"]  # task is the sole element after


def test_bridge_spawner_argv_includes_auto_approve_when_set(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    """A coordinator/fan-out started with --auto-approve must forward it to the
    lane, or the lane sits on run_commands=ask with nothing to answer it."""
    captured: list[list[str]] = []

    def fake_spawn(argv: list[str], workdir: Path, **_k: object) -> tuple[Path, str]:
        captured.append(list(argv))
        return workdir, ""

    cfg = Config()
    spec = LaneSpec(lane=1, run_id="fan-l1", workdir=tmp_path / "work" / "lane-1", model=None)
    parallel.bridge_spawner(
        spec, "do it", cfg=cfg, origin=origin, max_usd=None, auto_approve=True,
        runtime=replace(runtime, spawn=fake_spawn),
    )  # fmt: skip

    argv = captured[-1]
    dd = argv.index("--")
    assert "--auto-approve" in argv[:dd]  # precedes the `--` separator, like --max-usd


def test_bridge_spawner_argv_omits_auto_approve_by_default(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    captured: list[list[str]] = []

    def fake_spawn(argv: list[str], workdir: Path, **_k: object) -> tuple[Path, str]:
        captured.append(list(argv))
        return workdir, ""

    cfg = Config()
    spec = LaneSpec(lane=1, run_id="fan-l1", workdir=tmp_path / "work" / "lane-1", model=None)
    parallel.bridge_spawner(
        spec, "do it", cfg=cfg, origin=origin, max_usd=None,
        runtime=replace(runtime, spawn=fake_spawn),
    )  # fmt: skip

    assert "--auto-approve" not in captured[-1]


def test_run_lane_to_completion_forwards_auto_approve_to_the_default_spawner(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """When no *spawner* is injected, `run_lane_to_completion` builds the real
    bridge spawner itself (the coordinator's path); auto_approve must reach it
    exactly like max_usd already does."""
    captured: list[dict[str, object]] = []

    def fake_bridge(spec: LaneSpec, task: str, **kw: object) -> LaneResult:
        captured.append(kw)
        return LaneResult(
            spec=spec, run_dir=spec.workdir, branch="agent6/x", ok=False, error="stub"
        )

    monkeypatch.setattr(parallel, "bridge_spawner", fake_bridge)
    cfg = Config()
    spec = LaneSpec(lane=1, run_id="co-p1-l1", workdir=tmp_path / "work" / "co-p1-l1", model=None)

    parallel.run_lane_to_completion(
        spec,
        "do it",
        cfg=cfg,
        origin=origin,
        origin_state=tmp_path / "ostate",
        group="p1",
        runtime=runtime,
        auto_approve=True,
    )

    assert captured[-1]["auto_approve"] is True


# ---------------------------------------------------------------------------
# Pending-ask probe: a "running" lane blocked on an approval/question
# ---------------------------------------------------------------------------


def _lane_logs(run_dir: Path, *events: Mapping[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )


def test_pending_prompt_reads_last_prompt_answer_event(tmp_path: Path) -> None:
    start = {"type": "run.start", "mode": "run", "user_task": "t"}
    q = tmp_path / "q"
    _lane_logs(q, start, {"type": "question.prompt", "id": "question-1"})
    assert parallel._pending_prompt(q) == "a question"  # pyright: ignore[reportPrivateUsage]

    a = tmp_path / "a"
    _lane_logs(a, start, {"type": "approval.prompt", "id": "approval-1"})
    assert parallel._pending_prompt(a) == "approval"  # pyright: ignore[reportPrivateUsage]

    # answered -> not waiting; and a dir with no prompt events -> "".
    answered = tmp_path / "answered"
    _lane_logs(
        answered,
        start,
        {"type": "question.prompt", "id": "question-1"},
        {"type": "question.answer", "id": "question-1", "answers": ["yes"]},
    )
    assert parallel._pending_prompt(answered) == ""  # pyright: ignore[reportPrivateUsage]
    plain = tmp_path / "plain"
    _lane_logs(plain, start, {"type": "tool.call", "name": "read_file"})
    assert parallel._pending_prompt(plain) == ""  # pyright: ignore[reportPrivateUsage]


def test_await_lanes_status_line_flags_a_waiting_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime: LaneRuntime,
) -> None:
    """A lane the fold still calls "running" but which is blocked on an
    unanswered question shows the "waiting on ... (answer via the web or TUI
    hub)" note, so the operator knows to open a front-end."""
    from agent6.viewmodel import RunSummary

    lane = tmp_path / "lane"
    _lane_logs(
        lane,
        {"type": "run.start", "mode": "run", "user_task": "t"},
        {"type": "question.prompt", "id": "question-1"},
    )
    spec = LaneSpec(lane=1, run_id="fan-l1", workdir=tmp_path / "wd", model=None)
    res = LaneResult(spec=spec, run_dir=lane, branch="agent6/fan-l1", ok=True, error="")

    statuses = iter(["running", "failed"])  # waiting first poll, terminal next

    def fake_summary(rd: Path) -> RunSummary:
        return RunSummary(
            run_id=rd.name, mode="run", task="t", status=next(statuses),
            reason="", cost_usd=0.0, mtime=0.0,
        )  # fmt: skip

    def fake_worker_is_alive(_run_dir: Path) -> bool:
        return False

    def fake_sleep(*_args: object) -> None:
        return None

    monkeypatch.setattr(parallel, "summarize_run_dir", fake_summary)
    monkeypatch.setattr(parallel.time, "sleep", fake_sleep)
    rt = replace(runtime, worker_is_alive=fake_worker_is_alive)

    assert parallel._await_lanes([res], runtime=rt) is False  # pyright: ignore[reportPrivateUsage]
    assert "waiting on a question (answer via the web or TUI hub)" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_run_parallel_imports_branches_and_stamps_lineage(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "2")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    rc = run_parallel(
        "do the task",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    assert rc == 0
    # Both lane branches landed in the origin.
    assert branch_exists(origin, "agent6/fan-l1")
    assert branch_exists(origin, "agent6/fan-l2")
    # The live symlink was replaced by the real imported dir.
    imported = origin_state / "runs" / "fan-l1"
    assert imported.is_dir() and not imported.is_symlink()
    # Lineage was stamped post-import.
    manifest = json.loads((imported / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parallel_id"] == "fan"
    assert manifest["lane"] == 1
    assert (
        json.loads(
            (origin_state / "runs" / "fan-l2" / "manifest.json").read_text(encoding="utf-8")
        )["lane"]
        == 2
    )


def test_run_parallel_forwards_auto_approve_to_the_default_spawner(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """`run --parallel --auto-approve` must reach the lane's own default (real)
    bridge spawner, same plumbing as --max-usd."""
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "1")
    captured: list[dict[str, object]] = []

    def fake_bridge(spec: LaneSpec, task: str, **kw: object) -> LaneResult:
        captured.append(kw)
        return LaneResult(spec=spec, run_dir=spec.workdir, branch="agent6/x", ok=False, error="s")

    monkeypatch.setattr(parallel, "bridge_spawner", fake_bridge)

    run_parallel(
        "t", lanes, cfg=cfg, origin=origin, origin_state=origin_state,
        runtime=runtime, fanout_id="fan", auto_approve=True,
    )  # fmt: skip

    assert captured[-1]["auto_approve"] is True


def test_compare_outcome_stamped_into_each_lane_manifest(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    """The fan-out's auto-compare stamps a `compare` block into EVERY imported
    lane's manifest (winner + loser), recording rank/of/winner and, with no
    reviewer configured, ranked_by="mechanical" with an empty rationale."""
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()  # no reviewer -> mechanical ranking
    lanes = _specs(tmp_path, cfg, "fan", "2")
    # Lane 2 passes verify, lane 1 fails -> lane 2 wins (rank 1) mechanically.
    spawner = _FakeSpawner(
        origin, origin_state, tmp_path / "lane-state", status_by_lane={1: "failed", 2: "passed"}
    )

    run_parallel(
        "t", lanes, cfg=cfg, origin=origin, origin_state=origin_state,
        runtime=runtime, spawner=spawner, fanout_id="fan",
    )  # fmt: skip

    m1 = json.loads((origin_state / "runs" / "fan-l1" / "manifest.json").read_text("utf-8"))
    m2 = json.loads((origin_state / "runs" / "fan-l2" / "manifest.json").read_text("utf-8"))
    assert m2["compare"] == {
        "group": "fan", "rank": 1, "of": 2, "winner": True,
        "ranked_by": "mechanical", "rationale": "",
    }  # fmt: skip
    assert m1["compare"] == {
        "group": "fan", "rank": 2, "of": 2, "winner": False,
        "ranked_by": "mechanical", "rationale": "",
    }  # fmt: skip
    # The lineage stamp is untouched by the compare stamp (shared rewrite merges).
    assert m1["parallel_id"] == "fan" and m1["lane"] == 1


def test_compare_stamp_records_judge_rationale_truncated(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """When the judge ranks (not the mechanical fallback), every lane records
    ranked_by="judge" and the SAME rationale, truncated to bound the manifest."""
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "2")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")
    long_rationale = "x" * 3000

    def fake_rank(*_a: object, **_k: object) -> tuple[tuple[str, ...], str, str]:
        return ("fan-l1", "fan-l2"), long_rationale, "judge"

    monkeypatch.setattr(parallel, "_rank", fake_rank)

    run_parallel(
        "t", lanes, cfg=cfg, origin=origin, origin_state=origin_state,
        runtime=runtime, spawner=spawner, fanout_id="fan",
    )  # fmt: skip

    m1 = json.loads((origin_state / "runs" / "fan-l1" / "manifest.json").read_text("utf-8"))
    m2 = json.loads((origin_state / "runs" / "fan-l2" / "manifest.json").read_text("utf-8"))
    assert m1["compare"]["ranked_by"] == "judge" and m1["compare"]["winner"] is True
    assert m1["compare"]["rank"] == 1 and m2["compare"]["rank"] == 2
    assert len(m1["compare"]["rationale"]) == 2000  # truncated ~2000
    assert m2["compare"]["rationale"] == m1["compare"]["rationale"]  # same group rationale


def test_run_parallel_symlink_appears_before_import(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "2")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    # While lane 2 spawned, lane 1 was already visible as a live symlink...
    assert spawner.prior_link_was_symlink[2] is True
    # ...and after completion every lane is a real dir, no symlink left behind.
    for i in (1, 2):
        link = origin_state / "runs" / f"fan-l{i}"
        assert link.is_dir() and not link.is_symlink()


def test_failed_lane_does_not_stop_others(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "3")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state", fail={2})

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    assert rc == 0  # lanes 1 and 3 still produced candidates
    assert branch_exists(origin, "agent6/fan-l1")
    assert branch_exists(origin, "agent6/fan-l3")
    assert not branch_exists(origin, "agent6/fan-l2")
    assert (origin_state / "runs" / "fan-l1").is_dir()
    assert not (origin_state / "runs" / "fan-l2").exists()


def test_report_ranks_passing_lane_first(
    origin: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], runtime: LaneRuntime
) -> None:
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()  # no reviewer model -> mechanical ranking
    lanes = _specs(tmp_path, cfg, "fan", "2")
    # Lane 1 fails verify but is cheaper; lane 2 passes. Verify-pass wins.
    spawner = _FakeSpawner(
        origin,
        origin_state,
        tmp_path / "lane-state",
        status_by_lane={1: "failed", 2: "passed"},
        cost_by_lane={1: 0.01, 2: 0.09},
    )

    run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    out = capsys.readouterr().out
    assert "ranked candidates" in out
    # The passing lane ranks first despite costing more.
    assert out.index("fan-l2") < out.index("fan-l1")
    assert "agent6 runs merge fan-l2" in out


def test_run_parallel_all_failed_returns_1(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "2")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state", fail={1, 2})

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )
    assert rc == 1


def test_lineage_stamp_oserror_does_not_abort_import_loop(
    origin: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime: LaneRuntime,
) -> None:
    """An atomic_write OSError while stamping lineage (disk full / read-only
    mount) must not abort the import loop mid-way: each lane's import stands, the
    degradation prints, and the remaining lanes still import + report."""
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "2")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    def boom(_path: Path, _data: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(parallel, "atomic_write", boom)

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    assert rc == 0  # both lanes still imported despite the stamp failure
    assert branch_exists(origin, "agent6/fan-l1")
    assert branch_exists(origin, "agent6/fan-l2")
    assert (origin_state / "runs" / "fan-l1").is_dir()
    assert (origin_state / "runs" / "fan-l2").is_dir()
    err = capsys.readouterr().err
    assert "lineage" in err and "disk full" in err


def test_ctrl_c_during_spawn_loop_stops_imports_and_reports(
    origin: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime: LaneRuntime,
) -> None:
    """A KeyboardInterrupt while still spawning (before the await) routes into the
    same stop-grace + import-what-exists + report path: the already-started lane
    is imported, the run exits 130, and lanes never spawned are simply absent."""
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "3")
    base = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    def interrupting_spawner(spec: LaneSpec, task: str) -> LaneResult:
        if spec.lane == 2:  # interrupt AFTER lane 1 has started
            raise KeyboardInterrupt
        return base(spec, task)

    monkeypatch.setattr(parallel, "_POLL_INTERVAL_S", 0.01)

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=interrupting_spawner,
        fanout_id="fan",
    )

    assert rc == 130
    # Lane 1 (started before the interrupt) was stopped + imported...
    assert branch_exists(origin, "agent6/fan-l1")
    assert (origin_state / "runs" / "fan-l1").is_dir()
    # ...lanes 2 and 3 never produced a candidate.
    assert not branch_exists(origin, "agent6/fan-l2")
    assert not branch_exists(origin, "agent6/fan-l3")
    assert "interrupted; stopping lanes" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Teardown race + cleanup safety
# ---------------------------------------------------------------------------


def test_await_waits_for_worker_pid_to_clear(
    origin: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime: LaneRuntime,
) -> None:
    """run.end lands in logs.jsonl BEFORE the lane's teardown clears worker.pid.
    The await gate must keep waiting through that window (terminal = non-running
    status AND pid cleared/dead); importing inside it would misread the lane as
    still running and cleanup would destroy its only copy."""
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "1")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state", pid_lanes={1})

    # Clear the live pid only on the SECOND status poll of the lane's run dir,
    # so poll 1 exercises the race window deterministically.
    real_summarize = parallel.summarize_run_dir
    polls = {"n": 0}

    def summarize_then_clear_pid(run_dir: Path) -> object:
        summary = real_summarize(run_dir)
        if run_dir.name == "fan-l1":
            polls["n"] += 1
            if polls["n"] >= 2:
                (run_dir / "worker.pid").unlink(missing_ok=True)
        return summary

    monkeypatch.setattr(parallel, "summarize_run_dir", summarize_then_clear_pid)
    monkeypatch.setattr(parallel, "_POLL_INTERVAL_S", 0.01)

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    assert rc == 0
    assert polls["n"] >= 2  # the gate really held through the live-pid poll
    assert branch_exists(origin, "agent6/fan-l1")
    imported = origin_state / "runs" / "fan-l1"
    assert imported.is_dir() and not imported.is_symlink()
    assert "failed lanes" not in capsys.readouterr().out


def test_cleanup_preserves_unimported_lane(
    origin: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], runtime: LaneRuntime
) -> None:
    """A lane whose import is refused keeps its clone, run state, and live
    symlink (the clone holds the only copy of its branch), and the report names
    what was kept. Imported lanes are still cleaned up."""
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "2")
    # Pre-existing branch in the origin makes lane 1's import refuse.
    create_branch(origin, "agent6/fan-l1")
    _git(origin, "checkout", "main")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    assert rc == 0  # lane 2 still imported
    # Lane 1 kept: clone (with its branch), fabricated run state, live symlink.
    assert lanes[0].workdir.is_dir()
    assert branch_exists(lanes[0].workdir, "agent6/fan-l1")
    assert (tmp_path / "lane-state" / "lane1" / "runs" / "fan-l1").is_dir()
    assert (origin_state / "runs" / "fan-l1").is_symlink()
    # Lane 2 imported and cleaned: real dir in origin state, clone gone.
    assert (origin_state / "runs" / "fan-l2").is_dir()
    assert not (origin_state / "runs" / "fan-l2").is_symlink()
    assert not lanes[1].workdir.exists()
    # The report names the kept clone so the operator can act on it.
    out = capsys.readouterr().out
    assert str(lanes[0].workdir) in out


def test_await_uses_real_run_dir_not_symlink(
    origin: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime: LaneRuntime,
) -> None:
    """The symlink is a view for the hub, not the source of truth: with symlink
    creation failing entirely, the lane is still awaited on its REAL run dir
    (its true status is observed, not '?') and imported."""
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    lanes = _specs(tmp_path, cfg, "fan", "1")
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")

    def _no_symlink(*_a: object, **_k: object) -> None:
        return None

    monkeypatch.setattr(parallel, "_symlink_lane", _no_symlink)

    rc = run_parallel(
        "t",
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=runtime,
        spawner=spawner,
        fanout_id="fan",
    )

    assert rc == 0
    assert branch_exists(origin, "agent6/fan-l1")
    assert (origin_state / "runs" / "fan-l1").is_dir()
    # The lane's real terminal status was observed (not the missing-link "?").
    assert "lane 1 [fan-l1]: passed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Coordinator plumbing: run_lane_to_completion + the group spawner
# ---------------------------------------------------------------------------


def test_run_lane_to_completion_imports_and_stamps(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """One lane fully: spawn (fake), symlink it live into the origin's runs/,
    await to terminal, import its branch + run dir into the origin, and stamp
    `<group>` lineage. The live symlink is visible while the lane runs (so a hub
    can see + answer it) and is replaced by the real dir after import."""
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state")
    spec = LaneSpec(lane=1, run_id="co-p1-l1", workdir=tmp_path / "work" / "co-p1-l1", model=None)

    # The await polls summarize_run_dir; observe the origin link state then -- it
    # must be a live symlink while the lane is still running.
    link = origin_state / "runs" / "co-p1-l1"
    real_summarize = parallel.summarize_run_dir
    seen: dict[str, bool] = {}

    def observe(run_dir: Path) -> object:
        seen.setdefault("symlink_during_life", link.is_symlink())
        return real_summarize(run_dir)

    monkeypatch.setattr(parallel, "summarize_run_dir", observe)

    res = parallel.run_lane_to_completion(
        spec,
        "do it",
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        group="p1",
        runtime=runtime,
        spawner=spawner,
        poll_interval_s=0.01,
    )

    assert res.ok
    assert seen["symlink_during_life"] is True  # a hub could see + answer the lane
    assert branch_exists(origin, "agent6/co-p1-l1")
    imported = origin_state / "runs" / "co-p1-l1"
    assert imported.is_dir() and not imported.is_symlink()  # replaced by the real dir
    assert res.run_dir == imported
    manifest = json.loads((imported / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parallel_id"] == "p1"
    assert manifest["lane"] == 1


def test_run_lane_to_completion_failed_spawn_imports_nothing(
    origin: Path, tmp_path: Path, runtime: LaneRuntime
) -> None:
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    spawner = _FakeSpawner(origin, origin_state, tmp_path / "lane-state", fail={1})
    spec = LaneSpec(lane=1, run_id="co-p1-l1", workdir=tmp_path / "work" / "co-p1-l1", model=None)

    res = parallel.run_lane_to_completion(
        spec,
        "do it",
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        group="p1",
        runtime=runtime,
        spawner=spawner,
    )

    assert not res.ok and res.error == "boom"
    assert not branch_exists(origin, "agent6/co-p1-l1")


def test_build_lane_spawner_builds_specs_and_preserves_order(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """The group dispatcher names lanes `<coord>-<group>-l<i>`, puts them under a
    per-group workdir, and returns results in dispatch order despite the pool."""
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    seen: list[tuple[int, str, str, str, str]] = []

    def fake_rltc(spec: LaneSpec, task: str, **kw: object) -> LaneResult:
        seen.append((spec.lane, spec.run_id, task, str(kw["group"]), str(spec.workdir)))
        return LaneResult(
            spec=spec, run_dir=spec.workdir, branch=f"agent6/{spec.run_id}", ok=True, error=""
        )

    monkeypatch.setattr(parallel, "run_lane_to_completion", fake_rltc)
    dispatch = parallel.build_lane_spawner(
        cfg, origin, origin_state, coordinator_run_id="co", runtime=runtime
    )
    lanes = [LaneTask(task="task a", model="kimi"), LaneTask(task="task b", model=None)]
    results = dispatch(lanes, "p2")

    assert [r.spec.run_id for r in results] == ["co-p2-l1", "co-p2-l2"]
    assert [r.spec.model for r in results] == ["kimi", None]  # per-lane model threaded through
    assert sorted(s[0] for s in seen) == [1, 2]  # every lane ran once
    assert all(group == "p2" for (_l, _r, _t, group, _w) in seen)
    assert all(f"{os.sep}p2{os.sep}lane-" in workdir for (*_rest, workdir) in seen)


def test_build_lane_spawner_forwards_auto_approve(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    from agent6.config.layer import resolved_state_dir

    origin_state = resolved_state_dir(origin)
    cfg = Config()
    seen: list[object] = []

    def fake_rltc(spec: LaneSpec, task: str, **kw: object) -> LaneResult:
        seen.append(kw["auto_approve"])
        return LaneResult(spec=spec, run_dir=spec.workdir, branch="agent6/x", ok=True, error="")

    monkeypatch.setattr(parallel, "run_lane_to_completion", fake_rltc)
    dispatch = parallel.build_lane_spawner(
        cfg, origin, origin_state, coordinator_run_id="co", runtime=runtime, auto_approve=True
    )
    dispatch([LaneTask(task="task a", model=None)], "p3")

    assert seen == [True]


def test_build_coordinator_spawner_forwards_auto_approve(
    origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: LaneRuntime
) -> None:
    """A coordinator started with --auto-approve dispatches lanes that inherit
    it; one started without does not (build_coordinator_spawner -> build_lane_
    spawner, same param as max_usd)."""
    origin_state = tmp_path / "ostate"
    origin_state.mkdir()
    cfg = Config()
    captured: list[object] = []

    def fake_build_lane_spawner(*_a: object, **kw: object) -> object:
        captured.append(kw.get("auto_approve"))
        return "dispatcher"

    monkeypatch.setattr(parallel, "build_lane_spawner", fake_build_lane_spawner)

    parallel.build_coordinator_spawner(
        cfg, origin, origin_state, mode="run", run_id="co", runtime=runtime, auto_approve=True
    )
    parallel.build_coordinator_spawner(
        cfg, origin, origin_state, mode="run", run_id="co", runtime=runtime
    )

    assert captured == [True, False]
