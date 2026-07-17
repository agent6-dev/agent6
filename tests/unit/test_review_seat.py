# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the review seat call + run_panel (with fake providers, no network)."""

from __future__ import annotations

from typing import Any, cast

from agent6.providers import Provider, ProviderError
from agent6.workflows._panel import ReviewContext
from agent6.workflows._review import (
    Seat,
    _coerce_findings,  # pyright: ignore[reportPrivateUsage]
    _extract_json,  # pyright: ignore[reportPrivateUsage]
    run_panel,
    structured_review,
)

SAMPLE_DIFF = """\
--- a/foo.py
+++ b/foo.py
@@ -10,2 +10,3 @@ def f():
     x = 1
+    token = "sk-secret"
     return x
"""

_BLOCK_JSON = (
    '{"verdict":"block","summary":"leak","findings":[{"category":"security",'
    '"severity":"block","file_line":"foo.py:11","title":"secret leak","detail":"hardcoded"}]}'
)


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeProvider:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def call(self, **kw: Any) -> Any:
        self.calls += 1
        return _Resp(self._text)


class _ErrProvider:
    def call(self, **kw: Any) -> Any:
        raise ProviderError("boom")


def _prov(text: str) -> Provider:
    return cast(Provider, _FakeProvider(text))


def _ctx() -> ReviewContext:
    return ReviewContext(task="add auth", diff=SAMPLE_DIFF, verify_ok=True)


def test_structured_review_parses_clean_json() -> None:
    v = structured_review(_prov(_BLOCK_JSON), _ctx(), seat="security", model="m1")
    assert v.error is None and v.verdict == "block"
    assert v.findings[0].category == "security" and v.findings[0].file_line == "foo.py:11"


def test_structured_review_parses_fenced_json_with_prose() -> None:
    text = f"Here is my review:\n```json\n{_BLOCK_JSON}\n```\nDone."
    v = structured_review(_prov(text), _ctx(), seat="s", model="m1")
    assert v.error is None and v.verdict == "block" and len(v.findings) == 1


def test_structured_review_junk_output_abstains() -> None:
    v = structured_review(_prov("I could not produce JSON, sorry."), _ctx(), seat="s", model="m1")
    assert v.error is not None and v.verdict == "pass"  # abstain, never a false pass-as-real


def test_structured_review_provider_error_abstains() -> None:
    v = structured_review(cast(Provider, _ErrProvider()), _ctx(), seat="s", model="m1")
    assert v.error is not None and "provider" in v.error


def test_coerce_findings_normalizes_bad_category_and_severity() -> None:
    raw = [
        {"category": "bogus", "severity": "critical", "file_line": "a:1", "title": "t"},
        "not a dict",
        {"category": "security", "severity": "block", "file_line": "b:2", "title": "ok"},
    ]
    fs = _coerce_findings(raw)
    assert len(fs) == 2
    assert fs[0].category == "other" and fs[0].severity == "warn"  # normalized
    assert fs[1].category == "security" and fs[1].severity == "block"


def test_run_panel_distinct_models_quorum_blocks() -> None:
    seats = [
        Seat(persona="security", model="m1", provider=_prov(_BLOCK_JSON)),
        Seat(persona="correctness", model="m2", provider=_prov(_BLOCK_JSON)),
    ]
    res = run_panel(seats, _ctx(), decision="quorum", quorum=2, panel_id="p")
    assert res.blocked is True and res.n_block == 2


def test_run_panel_advisory_never_blocks() -> None:
    seats = [Seat(persona="security", model="m1", provider=_prov(_BLOCK_JSON))]
    res = run_panel(seats, _ctx(), decision="advisory", quorum=2, panel_id="p")
    assert res.blocked is False
    assert (
        res.merged_findings and res.merged_findings[0].severity == "block"
    )  # reported, not gating


def test_run_panel_concurrent_preserves_order_and_aggregates() -> None:
    # 3 seats, distinct models, concurrency=3: results stay in seat order and the
    # grounded quorum still blocks (thread pool must not change the verdict).
    seats = [
        Seat(persona="security", model="m1", provider=_prov(_BLOCK_JSON)),
        Seat(persona="correctness", model="m2", provider=_prov(_BLOCK_JSON)),
        Seat(persona="edge", model="m3", provider=_prov(_BLOCK_JSON)),
    ]
    res = run_panel(seats, _ctx(), decision="quorum", quorum=2, panel_id="p", concurrency=3)
    assert [v.seat for v in res.per_seat] == ["security", "correctness", "edge"]
    assert res.blocked is True and res.n_block == 3


def test_parse_seat_spec_forms() -> None:
    from agent6.workflows._review import parse_seat_spec

    assert parse_seat_spec("security@openrouter/moonshotai/kimi-k2") == (
        "security",
        "openrouter",
        "moonshotai/kimi-k2",  # model keeps its own slashes
    )
    assert parse_seat_spec("correctness") == ("correctness", "", "")  # bare persona
    assert parse_seat_spec("@anthropic/claude-opus-4-8") == ("", "anthropic", "claude-opus-4-8")


# --- model_override on configured review_seats (the `review --model X` flag) ---


def _cfg_with_seats(seats: tuple[str, ...]) -> Any:
    from agent6.config import Config

    return Config.model_validate(
        {
            "providers": {"anthropic": {"api_format": "anthropic", "api_key_env": "FAKE_KEY"}},
            "models": {"reviewer": {"provider": "anthropic", "model": "reviewer-default"}},
            "review": {"trigger": "before_finish", "seats": list(seats)},
        }
    )


def _stub_seat_provider(*_a: Any, **_k: Any) -> Provider:
    """Stand-in for `_provider_from_entry` / `_build_role_provider` so seat
    construction needs no API key or network; the test asserts on the seat label."""
    return _prov("{}")


def test_build_review_seats_model_override_overrides_pinned_seat(monkeypatch: Any) -> None:
    # `review --model X` must override each configured seat's pinned model while
    # keeping its provider routing -- otherwise the flag silently does nothing.
    from agent6.app import providers as prov_mod

    monkeypatch.setattr(prov_mod, "_provider_from_entry", _stub_seat_provider)
    cfg = _cfg_with_seats(
        ("security@anthropic/claude-opus-4-8", "correctness@anthropic/some-model")
    )

    seats = prov_mod.build_review_seats(
        cfg,
        transcript_sink=cast(Any, None),
        budget=cast(Any, None),
        n=1,
        model_override="claude-haiku-override",
    )
    assert [s.model for s in seats] == [
        "anthropic/claude-haiku-override",
        "anthropic/claude-haiku-override",
    ]  # provider kept, model overridden
    assert [s.persona for s in seats] == ["security", "correctness"]


def test_build_review_seats_no_override_keeps_pinned_models(monkeypatch: Any) -> None:
    from agent6.app import providers as prov_mod

    monkeypatch.setattr(prov_mod, "_provider_from_entry", _stub_seat_provider)
    cfg = _cfg_with_seats(("security@anthropic/claude-opus-4-8",))

    seats = prov_mod.build_review_seats(
        cfg, transcript_sink=cast(Any, None), budget=cast(Any, None), n=1
    )
    assert seats[0].model == "anthropic/claude-opus-4-8"  # unchanged when no --model


def test_build_review_seats_model_override_on_bare_persona_seat(monkeypatch: Any) -> None:
    # A bare-persona seat routes via the reviewer role; --model must override that
    # role's model too (mirroring the non-configured simple-form branch).
    from agent6.app import providers as prov_mod

    monkeypatch.setattr(prov_mod, "build_role_provider", _stub_seat_provider)
    cfg = _cfg_with_seats(("correctness",))

    seats = prov_mod.build_review_seats(
        cfg,
        transcript_sink=cast(Any, None),
        budget=cast(Any, None),
        n=1,
        model_override="claude-haiku-override",
    )
    assert seats[0].model == "claude-haiku-override"  # reviewer-default overridden


# --- explore tier (read-only tool-using reviewer) -----------------------------


class _ExploreResp:
    def __init__(
        self, text: str = "", tool_uses: tuple[Any, ...] = (), raw: dict[str, Any] | None = None
    ) -> None:
        self.text = text
        self.tool_uses = list(tool_uses)
        self.raw = raw or {"content": []}


class _ExploreProvider:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def call(self, **kw: Any) -> Any:
        self.calls += 1
        return self._responses.pop(0)


def test_explore_review_uses_tools_then_verdicts() -> None:
    from agent6.workflows._review import explore_review

    tu = {"name": "grep", "id": "t1", "input": {"pattern": "read_doc"}}
    provider = _ExploreProvider(
        [
            _ExploreResp(tool_uses=(tu,), raw={"content": [{"type": "tool_use", **tu}]}),
            _ExploreResp(text=_BLOCK_JSON),  # final verdict, no tool calls
        ]
    )
    dispatched: list[str] = []

    def dispatch(name: str, inp: dict[str, Any]) -> dict[str, Any]:
        dispatched.append(name)
        return {"matches": ["caller.py:9: read_doc(x)"]}

    v = explore_review(
        cast(Provider, provider), _ctx(), seat="security", model="m1", tools=[], dispatch=dispatch
    )
    assert provider.calls == 2 and dispatched == ["grep"]  # investigated, then judged
    assert v.error is None and v.verdict == "block" and v.findings[0].category == "security"


def test_explore_review_abstains_when_no_verdict_in_budget() -> None:
    from agent6.workflows._review import explore_review

    tu = {"name": "list_dir", "id": "t1", "input": {"path": "."}}
    # provider keeps calling tools, never emits a verdict -> abstain after max_iters
    provider = _ExploreProvider(
        [
            _ExploreResp(tool_uses=(tu,), raw={"content": [{"type": "tool_use", **tu}]})
            for _ in range(10)
        ]
    )
    v = explore_review(
        cast(Provider, provider),
        _ctx(),
        seat="s",
        model="m1",
        tools=[],
        dispatch=lambda n, i: {"entries": []},
        max_iters=3,
    )
    assert v.error is not None and provider.calls == 3  # bounded, abstains (never false-pass)


def test_explore_review_skips_dispatch_on_final_iteration() -> None:
    from agent6.workflows._review import explore_review

    # The last allowed model call returns tool_uses and no verdict: the seat is
    # about to abstain, and no model call follows to consume the results, so the
    # final round's tools must not be executed (pure waste).
    tu = {"name": "grep", "id": "t1", "input": {"pattern": "x"}}
    provider = _ExploreProvider(
        [
            _ExploreResp(tool_uses=(tu,), raw={"content": [{"type": "tool_use", **tu}]})
            for _ in range(3)
        ]
    )
    dispatched: list[str] = []

    def dispatch(name: str, inp: dict[str, Any]) -> dict[str, Any]:
        dispatched.append(name)
        return {"ok": True}

    v = explore_review(
        cast(Provider, provider),
        _ctx(),
        seat="s",
        model="m1",
        tools=[],
        dispatch=dispatch,
        max_iters=3,
    )
    assert v.error == "explore: no verdict within max_iters"
    assert provider.calls == 3  # every model call still happens
    assert dispatched == ["grep", "grep"]  # rounds 1-2 only; final round skipped


def test_run_panel_routes_explore_tier_seats() -> None:
    tu = {"name": "grep", "id": "t1", "input": {"pattern": "x"}}
    prov = _ExploreProvider(
        [
            _ExploreResp(tool_uses=(tu,), raw={"content": [{"type": "tool_use", **tu}]}),
            _ExploreResp(text=_BLOCK_JSON),
        ]
    )
    seat = Seat(persona="security", model="m1", provider=cast(Provider, prov), tier="explore")
    res = run_panel(
        [seat],
        _ctx(),
        decision="veto",
        quorum=2,
        panel_id="p",
        tools=[],
        dispatch=lambda n, i: {"ok": True},
    )
    assert prov.calls == 2 and res.blocked is True  # explore seat ran the tool loop + blocked


def test_extract_json_prefers_the_verdict_object_over_a_stray_preamble() -> None:
    # A reasoning model may emit a throwaway object before the real verdict, or
    # wrap it in a fence with prose. Prefer the LAST object carrying verdict/findings.
    text = (
        'I will think first {"note": "scratch"} and here is my answer:\n'
        '```json\n{"verdict": "pass", "summary": "ok", "findings": []}\n```\n'
    )
    obj = _extract_json(text)
    assert obj is not None and obj["verdict"] == "pass" and obj["summary"] == "ok"


def test_extract_json_ignores_braces_inside_strings() -> None:
    obj = _extract_json('{"verdict": "block", "summary": "a } brace { in text", "findings": []}')
    assert obj is not None and obj["verdict"] == "block"


def test_explore_review_honors_verdict_alongside_tool_use_on_last_iter() -> None:
    from agent6.workflows._review import explore_review

    # On the FINAL allowed iteration the model emits a tool_use AND a verdict in
    # the same turn; the verdict must be honored (not wasted into an abstain).
    tu = {"name": "grep", "id": "t1", "input": {"pattern": "x"}}
    provider = _ExploreProvider(
        [_ExploreResp(text=_BLOCK_JSON, tool_uses=(tu,), raw={"content": []})]
    )
    v = explore_review(
        cast(Provider, provider),
        _ctx(),
        seat="s",
        model="m1",
        tools=[],
        dispatch=lambda n, i: {"ok": True},
        max_iters=1,
    )
    assert v.error is None and v.verdict == "block"
