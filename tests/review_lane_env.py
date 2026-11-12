from __future__ import annotations

import concurrent.futures
import dataclasses
import pathlib
import tempfile
from collections.abc import Sequence
from typing import Any, Protocol

from agent6 import budget, kinds, lane_records, media, review_records
from agent6.config import Config
from agent6.harness import loop, operator_bridge, parallel_dispatch, provider_call, run_chain
from agent6.harness.review import gate as review_gate
from agent6.harness.review import panel as review_panel
from agent6.providers import ProviderResponse
from agent6.sessions import event_log, journal
from agent6.tools import dispatch as tools_dispatch
from agent6.tools import results
from tests import child_env

_run_panel = review_gate.run_panel
# What a parent whose command policy is `deny` hands a seat.
READING_TOOLS = ("find_definition", "find_references", "list_dir", "outline", "read_file")
_providers: dict[int, Any] = {}


class Dispatch(Protocol):
    def __call__(
        self, name: str, raw: dict[str, Any], /, *, origin: kinds.ToolCallOrigin | None = None
    ) -> results.ToolResult: ...


def seat(
    focus: str,
    model: str,
    provider: Any,
    *,
    temperature: float | None = 0.0,
    media_input: media.InputCapabilities | None = None,
) -> review_gate.ReviewSeat:
    entry, sep, name = model.partition("/")
    result = review_gate.ReviewSeat(
        focus,
        kinds.ModelSettings(
            kinds.ModelRoute(entry if sep else "", name if sep else entry), None, temperature
        ),
    )
    _providers[id(result)] = (provider, media_input)
    return result


class ResponseAdapter:
    """Give older scripted policy providers the complete common response shape."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def call(self, **kwargs: Any) -> ProviderResponse:
        resp = self.provider.call(**kwargs)
        if isinstance(resp, ProviderResponse):
            return resp
        tools = tuple(getattr(resp, "tool_uses", ()))
        raw = dict(getattr(resp, "raw", {}))
        if not raw.get("content"):
            raw["content"] = ([{"type": "text", "text": resp.text}] if resp.text else []) + [
                {"type": "tool_use", **t} for t in tools
            ]
        return ProviderResponse(
            text=resp.text,
            tool_uses=tools,
            stop_reason=getattr(resp, "stop_reason", "end_turn"),
            input_tokens=10,
            output_tokens=getattr(resp, "output_tokens", 20),
            cache_read_tokens=0,
            cache_write_tokens=0,
            raw=raw,
        )


def run_panel(
    seats: Sequence[review_gate.ReviewSeat],
    ctx: review_panel.ReviewContext,
    *,
    dispatch: Dispatch | None = None,
    **kwargs: Any,
) -> review_panel.PanelResult:
    """Drive policy fixtures through real requests, records and the bounded Harness."""
    recovery = kwargs.pop("recovery", None)
    emit = kwargs.pop("emit", None)
    kwargs.pop("execution_id", None)
    kwargs.setdefault("tools", READING_TOOLS)
    kwargs.pop("media_rejections", None)
    kwargs.pop("executor", None)
    with tempfile.TemporaryDirectory(prefix="review-policy-") as tmp:
        root = pathlib.Path(tmp)
        repo = root / "repo"
        child_env.init_repo(repo)
        directory = child_env.create_record(repo, "parent")
        (repo / "foo.py").write_text("x\n" * 20)
        target = kwargs.pop("target", None) or review_records.ReviewTarget(
            "base", "base-tree", "head", "head-tree", "snapshot", (), "revisions"
        )
        cfg = Config.model_validate({"sandbox": {"isolation": "none", "protect_git": "off"}})

        def spawn(
            lanes: list[lane_records.LaneTask], group: str, *, at: str | None = None
        ) -> list[lane_records.LaneResult]:
            def one(pair: tuple[int, lane_records.LaneTask]) -> lane_records.LaneResult:
                index, request = pair
                assert request.contract is not None and request.contract.review_binding is not None
                provider, capabilities = _providers[
                    id(seats[request.contract.review_binding.seat_index])
                ]
                sid = f"parent-{group}-l{index}"
                child = directory.parent.parent / "reviewers" / sid
                child.mkdir(parents=True)
                dispatcher = tools_dispatch.ToolDispatcher(
                    root=repo, config=cfg, isolation="none", mode="reviewer", session_dir=child
                )
                if dispatch is not None:
                    dispatcher.dispatch = dispatch  # type: ignore[method-assign]
                harness = loop.Harness(
                    chain=run_chain.RunChain(repo, ref=None, fallback_parent=None, per_turn=False),
                    config=cfg,
                    provider=ResponseAdapter(provider),
                    dispatcher=dispatcher,  # type: ignore[arg-type]
                    logger=lambda _m: None,
                    events=event_log.EventSink(child / "logs.jsonl"),
                    resume_state_path=child / "loop_state.json",
                    bridge=operator_bridge.OperatorBridge(),
                    call=provider_call.CallSettings(retry_count=0),
                    mode="reviewer",
                    contract=request.contract,
                    media_input=capabilities
                    or (
                        recovery.capabilities if recovery is not None else media.InputCapabilities()
                    ),
                )
                if recovery is not None:
                    harness.media_rejections = {harness.media_input.route: recovery.unsupported}
                if emit is not None:

                    def media_event(row: dict[str, Any]) -> None:
                        if row.get("type") == "media.refused":
                            emit(
                                journal.MediaRefused(
                                    provider_tool_call_id=row["provider_tool_call_id"],
                                    reason=row["reason"],
                                )
                            )

                    assert harness.events is not None
                    harness.events.subscribe(media_event)
                harness.run(request.task)
                return lane_records.LaneResult(
                    lane_records.LaneSpec(index, sid, repo, request.settings, mode="reviewer"),
                    child,
                    "",
                    True,
                    "",
                    usage=budget.UsageSettlement(usd=0, usd_partial=False, tokens=0),
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as pool:
                return list(pool.map(one, enumerate(lanes, 1)))

        executor = parallel_dispatch.ParallelDispatcher(
            chain=run_chain.RunChain(repo, ref=None, fallback_parent=None, per_turn=False),
            curator=None,
            max_lanes=4,
            lane_spawner=spawn,
            save_snapshot=lambda *_a, **_k: None,
            log=lambda _m: None,
            emit=lambda *_a, **_k: None,
            emit_graph_snapshot=lambda: None,
            session_id="parent",
            session_dir=directory,
            authority=lane_records.DispatchAuthority(panels=True),
        )
        return _run_panel(seats, ctx, target=target, executor=executor, **kwargs)


def single_verdict(
    provider: Any,
    ctx: review_panel.ReviewContext,
    *,
    dispatch: Dispatch | None = None,
    **kwargs: Any,
) -> review_records.ReviewVerdict:
    index = kwargs.pop("seat_index")
    focus, route = kwargs.pop("focus"), kwargs.pop("model_route")
    resolved = seat(focus, route, provider, temperature=kwargs.pop("temperature", 0.0))
    turns = kwargs.pop("max_turns", 20)
    kwargs.pop("max_tokens", None)
    result = run_panel(
        [resolved],
        ctx,
        block_on="none",
        min_blocking_models=2,
        panel_id=kwargs.pop("panel_id", None) or "test",
        max_turns=turns,
        dispatch=dispatch,
        **kwargs,
    )
    return dataclasses.replace(result.per_seat[0], seat_index=index)
