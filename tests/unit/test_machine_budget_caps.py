"""Machine spend limits survive config resolution and the agent request boundary."""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Mapping
from unittest import mock

import pytest

from agent6 import paths
from agent6.app import preflight, reporter, session_setup
from agent6.app.machine import agent, run
from agent6.config import model
from agent6.machine import contracts

_MACHINE = """machine = "caps"
version = 1
initial = "judge"
[budget]
max_transitions = 5
[schemas.r]
ok = "bool"
[vars.agent]
out = { type = "r", default = {} }
[states.judge]
kind = "agent"
prompt = "judge"
output_schema = "r"
capture = { finish_json = "out" }
timeout_s = 60
on = { ok = "done", failed = "done", budget_exhausted = "done", timeout = "done" }
[states.done]
kind = "terminal"
status = "ok"
reason = "done"
"""


@pytest.mark.parametrize(
    "config_cap,state_cap,machine_cap,source,expected",
    [
        (1.0, None, 25.0, "state", 1.0),
        (0.0, None, 25.0, "state", 0.0),
        (-1.0, None, 25.0, "state", 25.0),
        (30.0, None, 25.0, "operator", 25.0),
        (1.0, 5.0, 25.0, "operator", 1.0),
        (10.0, 2.0, 25.0, "operator", 2.0),
        (-1.0, 2.0, None, "operator", 2.0),
        (-1.0, None, None, "operator", -1.0),
        (1.0, None, None, "operator", 1.0),
    ],
)
def test_preflight_and_runtime_preserve_every_spend_cap(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    config_cap: float,
    state_cap: float | None,
    machine_cap: float | None,
    source: str,
    expected: float,
) -> None:
    """The machine remainder cannot widen a state's cap or turn zero into paid calls."""
    monkeypatch.chdir(tmp_path)
    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        '[providers.p]\nbackend = "openai"\nbase_url = "http://127.0.0.1:9"\n'
        '[models.default]\nmodel = "p/unpriced-test-model"\n'
        + (f"[budget]\nmax_usd = {config_cap}\n" if source == "operator" else ""),
        encoding="utf-8",
    )
    body = _MACHINE
    if machine_cap is not None:
        body = body.replace("[budget]", f"[budget]\nmax_usd = {machine_cap}")
    if state_cap is not None:
        body = body.replace('kind = "agent"', f'kind = "agent"\nmax_usd = {state_cap}')
    if source == "state":
        body += f"\n[states.judge.config.budget]\nmax_usd = {config_cap}\n"
    path = tmp_path / "caps.asm.toml"
    path.write_text(body, encoding="utf-8")
    requests: list[contracts.AgentRequest] = []
    runtime_caps: list[float] = []

    class BudgetCapturedError(Exception):
        pass

    def capture_budget(cfg: model.Config) -> None:
        runtime_caps.append(cfg.budget.max_usd)
        raise BudgetCapturedError

    def build(
        overlays: Mapping[str, agent.StateOverlay],
        cwd: pathlib.Path,
        transcript_dir: pathlib.Path,
        *_args: object,
        **_kwargs: object,
    ) -> Callable[[contracts.AgentRequest, pathlib.Path | None], contracts.AgentExecResult]:
        def invoke(
            request: contracts.AgentRequest, _events_log: pathlib.Path | None
        ) -> contracts.AgentExecResult:
            requests.append(request)
            state = overlays[request.state_name]
            envelope = agent.MachineAgentRequest(
                cwd=cwd,
                root=cwd,
                overlay=state.overlay,
                isolation=state.isolation,
                transcript_dir=transcript_dir,
                request=request,
            )
            decoded = agent.MachineAgentRequest.model_validate_json(envelope.model_dump_json())
            with pytest.raises(BudgetCapturedError):
                agent.run_one(decoded)
            return contracts.AgentExecResult(reason="finish_session", payload={"ok": True})

        return invoke

    preflight_spy = mock.Mock(wraps=preflight.budget_preflight)
    monkeypatch.setattr(preflight, "budget_preflight", preflight_spy)
    monkeypatch.setattr(session_setup, "budget_tracker", capture_budget)
    monkeypatch.setattr(agent, "build_machine_agent_runner", build)
    said: list[str] = []
    assert (
        run.run_machine(
            path, reporter.Reporter(out=said.append, err=said.append), disable_sandbox=True
        )
        == 0
    ), said
    assert runtime_caps == [expected]
    assert [call.args[0].budget.max_usd for call in preflight_spy.call_args_list] == [expected]
    assert len(requests) == 1 and requests[0].max_usd == state_cap
    assert requests[0].remaining_usd == machine_cap


@pytest.mark.parametrize(
    "config_cap,state_cap,remaining,expected",
    [
        (1.0, 5.0, 25.0, 1.0),
        (0.0, 5.0, 25.0, 0.0),
        (-1.0, None, 0.0, 0.0),
        (5.0, 0.0, 25.0, 0.0),
        (5.0, -1.0, None, 5.0),
        (-1.0, None, None, -1.0),
        (-1.0, 2.0, 0.5, 0.5),
    ],
)
def test_spend_clamp_honors_zero_and_unlimited(
    config_cap: float, state_cap: float | None, remaining: float | None, expected: float
) -> None:
    """Zero refuses metered calls, while an unlimited cap never widens another bound."""
    cfg = model.Config.model_validate({"budget": {"max_usd": config_cap}})
    assert (
        cfg.with_machine_agent_overrides(max_usd=state_cap, remaining_usd=remaining).budget.max_usd
        == expected
    )
