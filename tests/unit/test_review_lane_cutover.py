from __future__ import annotations

import dataclasses
import pathlib

import pytest

from agent6 import budget, kinds, lane_records, review_records
from agent6.app import role_providers
from agent6.harness import parallel_dispatch, review_gate, review_seat, toolset
from agent6.sessions import completion
from tests import review_traces, subordinate_env


def test_seat_resolution_constructs_no_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("roster resolution constructed a live provider")

    monkeypatch.setattr(role_providers, "_provider_from_entry", forbidden)
    resolve = getattr(role_providers, "resolve_review_seats", None)
    assert resolve is not None, "review seats need pure settings resolution"
    seats = resolve(subordinate_env.config())
    assert len(seats) == 1 and seats[0].focus == "security"
    assert seats[0].settings == kinds.ModelSettings(kinds.ModelRoute("p", "base"), None, 0.0)
    assert "provider" not in {f.name for f in dataclasses.fields(seats[0])}


def test_old_panel_execution_is_deleted() -> None:
    for module, names in (
        (review_seat, ("structured_review", "repo_review", "ReviewDispatch")),
        (review_gate, ("_run_seats_concurrently",)),
        (toolset, ("READONLY_REVIEW_TOOLS", "build_readonly_review_tools")),
        (role_providers, ("build_review_seats",)),
    ):
        for name in names:
            assert not hasattr(module, name), f"obsolete panel executor: {module.__name__}.{name}"


def test_a_roster_runs_in_batches_each_under_what_the_run_has_left(
    short_repo: pathlib.Path,
) -> None:
    """Each batch's seats start under the run's remaining allowance, the earlier spend out.

    The roster's seats each held an even fifth from the start, whatever the others used.
    """
    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    directory = subordinate_env.create_record(repo, "parent")
    meter = budget.BudgetTracker(max_usd=10, max_tokens_fallback=1000, max_plan_points=20)
    executor = subordinate_env.dispatcher(repo, directory, subordinate_env.config(), meter)
    executor.max_lanes = 2
    executor.clock = lambda: 1000.0
    observed: list[tuple[int, float]] = []
    deadlines: list[float | None] = []

    def spawn(
        lanes: list[lane_records.LaneTask], group: str, *, at: str | None = None
    ) -> list[lane_records.LaneResult]:
        assert lanes[0].allowance is not None
        assert all(lane.allowance == lanes[0].allowance for lane in lanes)
        observed.append((len(lanes), lanes[0].allowance.usd))
        deadlines.extend(lane.limits.deadline_at if lane.limits else None for lane in lanes)
        spent = budget.UsageSettlement(usd=1, usd_partial=False, tokens=0)
        for i in range(1, len(lanes) + 1):
            # The wait's watch counts a seat's spend while it runs.
            meter.note_running(f"parent-{group}-l{i}", spent)
        return [
            lane_records.LaneResult(
                lane_records.LaneSpec(i, f"parent-{group}-l{i}", repo, lane.settings),
                directory,
                "",
                False,
                "scripted failure",
                usage=spent,
            )
            for i, lane in enumerate(lanes, 1)
        ]

    executor.lane_spawner = spawn
    contract = kinds.ConversationContract("no_tools", "s", "o", 1, 1500)
    requests = [lane_records.LaneTask(str(i), "reviewer", contract=contract) for i in range(5)]
    result = executor.dispatch_panel(
        requests,
        panel_id="roster",
        at=subordinate_env.git(repo, "rev-parse", "HEAD"),
        concurrency=3,
    )
    assert result.dispatched and len(result.lanes) == 5
    assert observed == [(2, 10), (2, 8), (1, 6)]
    assert meter.estimate_usd()[0] == 5 and meter.snapshot().usd_running_lanes == 0
    assert [j.detail for j in result.lanes] == ["scripted failure"] * 5
    # A later batch waits for the earlier ones: its window starts where theirs ends.
    window = parallel_dispatch.MODEL_LANE_DEADLINE_S
    assert deadlines == [1000 + window] * 2 + [1000 + 2 * window] * 2 + [1000 + 3 * window]


@pytest.mark.parametrize("scope", ["diff", "repo"])
def test_real_panel_runs_bound_read_only_sessions(short_repo: pathlib.Path, scope: str) -> None:
    from agent6.harness import review_panel
    from agent6.viewmodel import receipt

    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    directory = subordinate_env.create_record(repo, "parent")
    with review_traces.server("tools", scope) as (url, calls):
        cfg = subordinate_env.config(url).model_copy(
            update={"review": subordinate_env.config().review.model_copy(update={"scope": scope})}
        )
        meter = budget.BudgetTracker(max_usd=10, max_tokens_fallback=1000, max_plan_points=20)
        executor = subordinate_env.dispatcher(repo, directory, cfg, meter)
        (repo / "README.md").write_text("base\nchanged\n")
        target = executor.chain.review_target()
        assert target is not None
        target = executor.chain.retain_review_target(target, "parent")
        result = review_gate.run_panel(
            role_providers.resolve_review_seats(cfg),
            review_panel.ReviewContext(
                task="Review changes", diff=subordinate_env.git(repo, "diff")
            ),
            target=target,
            executor=executor,
            block_on="any",
            min_blocking_models=2,
            panel_id="test",
            repo_max_turns=3,
        )
    assert result.n_abstain == 0 and result.per_seat[0].verdict == "pass"
    assert len(calls) == (1 if scope == "diff" else 3)
    assert all(c["max_tokens"] == (1500 if scope == "diff" else 2000) for c in calls)
    names = {t["function"]["name"] for t in calls[0].get("tools", [])}
    assert names == (
        set()
        if scope == "diff"
        else {"read_file", "list_dir", "outline", "find_definition", "find_references"}
    )
    child = receipt.read_receipt(directory, repo).children[0]
    record = completion.read_completion(directory.parent.parent / "read_only" / child.session_id)
    assert record is not None and record.review_binding is not None
    assert record.review_binding.panel_id == "test"
    assert result.per_seat[0].session_id == child.session_id
    assert (repo / "README.md").read_text() == "base\nchanged\n"


@pytest.mark.parametrize("failure", ["malformed", "missing", "provider", "deadline", "exit"])
def test_failed_artifacts_keep_the_actual_diagnostic(
    short_repo: pathlib.Path, failure: str
) -> None:
    from agent6.harness import review_panel

    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    directory = subordinate_env.create_record(repo, "parent")
    executor = subordinate_env.dispatcher(
        repo,
        directory,
        subordinate_env.config(),
        budget.BudgetTracker(max_usd=10, max_tokens_fallback=1000, max_plan_points=20),
    )
    (repo / "README.md").write_text("base\nchanged\n")
    target = executor.chain.review_target()
    assert target is not None
    target = executor.chain.retain_review_target(target, "parent")

    def spawn(
        lanes: list[lane_records.LaneTask], group: str, *, at: str | None = None
    ) -> list[lane_records.LaneResult]:
        sid = f"parent-{group}-l1"
        seat_dir = subordinate_env.create_record(repo, sid, parent="parent", mode="read_only")
        contract = lanes[0].contract
        assert contract is not None
        if failure == "malformed":
            (seat_dir / completion.COMPLETION_NAME).write_text("not json")
        elif failure != "missing":
            completion.write_completion(
                seat_dir,
                completion.CompletionRecord(
                    text=review_traces.PASS,
                    session_id=sid,
                    review_binding=contract.review_binding,
                    stop_reason="provider_error" if failure == "provider" else "answered",
                    calls=1,
                    output_tokens=20,
                    diagnostics=("provider: service unavailable",),
                ),
            )
        return [
            lane_records.LaneResult(
                lane_records.LaneSpec(1, sid, repo, lanes[0].settings),
                seat_dir,
                "",
                False,
                "lane deadline exceeded"
                if failure == "deadline"
                else "child exited 1"
                if failure == "exit"
                else "no result",
                usage=budget.UsageSettlement(usd=0, usd_partial=False, tokens=0),
            )
        ]

    executor.lane_spawner = spawn
    result = review_gate.run_panel(
        role_providers.resolve_review_seats(subordinate_env.config()),
        review_panel.ReviewContext(task="Review changes", diff=subordinate_env.git(repo, "diff")),
        target=target,
        executor=executor,
        block_on="any",
        min_blocking_models=2,
        panel_id="failure",
        repo_max_turns=3,
    )
    assert result.n_abstain == 1 and not result.blocked
    expected = {
        "malformed": "malformed completion",
        "missing": "no result",
        "provider": "service unavailable",
        "deadline": "deadline exceeded",
        "exit": "child exited 1",
    }[failure]
    assert expected in (result.per_seat[0].error or "")


@pytest.mark.parametrize("wrong", ["panel", "seat", "target", "session", "unbound"])
def test_artifact_provenance_is_checked_before_parsing(wrong: str) -> None:
    target = review_records.ReviewTarget("b", "bt", "h", "ht", "h", (), "revisions")
    binding_type = getattr(review_records, "ReviewBinding", None)
    assert binding_type is not None, "completion evidence needs a parent-owned binding"
    binding = binding_type("panel", 0, target)
    changed = {
        "panel": dataclasses.replace(binding, panel_id="other"),
        "seat": dataclasses.replace(binding, seat_index=1),
        "target": dataclasses.replace(
            binding, target=dataclasses.replace(target, head_tree="other")
        ),
        "session": binding,
        "unbound": None,
    }[wrong]
    record = completion.CompletionRecord(
        text=review_traces.PASS,
        stop_reason="answered",
        calls=1,
        output_tokens=20,
        session_id="other" if wrong == "session" else "seat-session",
        review_binding=changed,
    )
    verdict = review_seat.verdict_from_completion(
        record,
        binding=binding,
        session_id="seat-session",
        focus="security",
        model_route="p/base",
        scope="diff",
        max_calls=1,
    )
    assert verdict.verdict == "abstain" and "binding" in (verdict.error or "")


def test_worker_child_schedules_its_panel_without_certifying_parent(
    short_repo: pathlib.Path,
) -> None:
    from agent6 import directive
    from agent6.harness import loop_conversation, loop_state
    from agent6.sessions import lane_dispatch
    from agent6.viewmodel import receipt

    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    directory = subordinate_env.create_record(repo, "parent")
    with review_traces.server("worker", "diff") as (url, calls):
        cfg = subordinate_env.config(url)
        cfg = cfg.model_copy(update={"review": cfg.review.model_copy(update={"when": "finish"})})
        executor = subordinate_env.dispatcher(
            repo,
            directory,
            cfg,
            budget.BudgetTracker(max_usd=10, max_tokens_fallback=10000, max_plan_points=20),
        )
        result = executor.dispatch(
            loop_conversation.Conversation(),
            1,
            loop_state.LoopState(original_task="parent", tool_calls=0),
            [directive.Segment(task="Change README and finish", spec="1")],
            limits=lane_records.LaneLimits(max_turns=5),
        )
    assert result.dispatched and result.lanes[0].status == "joined", result.message
    worker = directory.parent / result.lanes[0].session_id
    panels = receipt.read_receipt(worker, repo).review.panels
    assert len(panels) == 1 and panels[0].completion == "complete"
    leaf = receipt.read_receipt(worker, repo).children[0]
    leaf_dir = directory.parent.parent / "read_only" / leaf.session_id
    rendered = receipt.render(receipt.read_receipt(worker, repo))
    assert f"Seat 1 (security): {leaf.session_id}" in rendered
    assert completion.read_completion(leaf_dir) is not None
    assert lane_dispatch.read_records(leaf_dir) == ()
    assert receipt.read_receipt(directory, repo).review.panels == ()
    assert len(calls) == 3


def test_repo_seat_reads_retained_target_not_head_or_live_tree(short_repo: pathlib.Path) -> None:
    from agent6.harness import review_panel

    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    directory = subordinate_env.create_record(repo, "parent")
    with review_traces.server("tools", "repo") as (url, calls):
        cfg = subordinate_env.config(url)
        cfg = cfg.model_copy(update={"review": cfg.review.model_copy(update={"scope": "repo"})})
        executor = subordinate_env.dispatcher(
            repo,
            directory,
            cfg,
            budget.BudgetTracker(max_usd=10, max_tokens_fallback=1000, max_plan_points=20),
        )
        (repo / "README.md").write_text("retained target bytes\n")
        target = executor.chain.review_target()
        assert target is not None
        target = executor.chain.retain_review_target(target, "parent")
        diff = subordinate_env.git(repo, "diff")
        (repo / "README.md").write_text("later live bytes\n")
        assert subordinate_env.git(repo, "show", "HEAD:README.md") == "base"
        result = review_gate.run_panel(
            role_providers.resolve_review_seats(cfg),
            review_panel.ReviewContext(task="Review changes", diff=diff),
            target=target,
            executor=executor,
            block_on="any",
            min_blocking_models=2,
            panel_id="retained",
            repo_max_turns=3,
        )
    assert result.n_abstain == 0
    tool_results = [m["content"] for m in calls[1]["messages"] if m["role"] == "tool"]
    assert any("retained target bytes" in text for text in tool_results), tool_results
    assert all("later live bytes" not in text for text in tool_results)
    assert (repo / "README.md").read_text() == "later live bytes\n"


def test_configured_top_level_ask_retains_panel_authority(short_repo: pathlib.Path) -> None:
    from agent6.app import lane_launch

    authority = lane_launch.dispatch_authority(
        subordinate_env.config(), mode="ask", session_id="review", root=short_repo
    )
    assert authority.panels and not authority.work


def test_ask_review_uses_real_seat_sessions(
    short_repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6 import paths
    from agent6.ui.cli import cli_main
    from agent6.viewmodel import receipt

    repo = short_repo / "origin"
    subordinate_env.init_repo(repo)
    (repo / "README.md").write_text("base\nchanged\n")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("AGENT6_ALLOW_ROOT", "1")
    with review_traces.server("pass", "diff") as (url, calls):
        config = short_repo / "cfg.toml"
        config.write_text(f'''[providers.p]
backend = "openai"
base_url = "{url}"
[models.default]
model = "p/base"
[sandbox]
isolation = "none"
protect_git = "off"
run_commands = "deny"
[review]
when = "verify_fail"
[prompt]
decompose = "off"
[harness]
verify_infer = false
verify_when = "never"
''')
        rc = cli_main(["--config", str(config), "ask", "/review"])
    assert rc == 0
    (parent,) = (paths.state_dir(repo) / "sessions" / "asks").iterdir()
    view = receipt.read_receipt(parent, repo)
    assert len(view.children) == 1 and len(view.review.panels) == 1
    assert view.review.panels[0].seats[0].session_id == view.children[0].session_id
    assert len(calls) == 2
