# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the reviewer call + run_panel (with fake providers, no network)."""

from __future__ import annotations

import itertools
import json
import time
from typing import Any, cast
from unittest import mock

import pytest

from agent6 import kinds
from agent6.harness import llm_json
from agent6.harness.review import gate as review_gate
from agent6.harness.review import panel as review_panel
from agent6.harness.review import verdict as review_verdict
from agent6.providers import Provider, ProviderError
from agent6.sessions import journal
from agent6.tools import results
from tests import review_lane_env

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
    '"severity":"block","location":{"path":"foo.py","side":"head","start_line":11,"end_line":11},'
    '"title":"secret leak","detail":"hardcoded"}]}'
)


class _Resp:
    def __init__(
        self,
        text: str,
        *,
        stop_reason: str = "end_turn",
        output_tokens: int = 10,
        thinking: str = "",
    ) -> None:
        self.text = text
        self.stop_reason = stop_reason
        self.output_tokens = output_tokens
        # As the providers build it: reasoning rides as a thinking block in raw.
        self.raw: dict[str, Any] = (
            {"content": [{"type": "thinking", "thinking": thinking}]} if thinking else {}
        )


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


def _ctx() -> review_panel.ReviewContext:
    return review_panel.ReviewContext(task="add auth", diff=SAMPLE_DIFF, verification="passed")


def test_structured_review_parses_clean_json() -> None:
    v = review_lane_env.single_verdict(
        _prov(_BLOCK_JSON), _ctx(), reviewer_index=0, focus="security", model_route="m1"
    )
    assert v.error is None and v.verdict == "block"
    assert v.findings[0].category == "security" and str(v.findings[0].location) == "foo.py:11"


def test_structured_review_parses_fenced_json_with_prose() -> None:
    text = f"Here is my review:\n```json\n{_BLOCK_JSON}\n```\nDone."
    v = review_lane_env.single_verdict(
        _prov(text), _ctx(), reviewer_index=0, focus="s", model_route="m1"
    )
    assert v.error is None and v.verdict == "block" and len(v.findings) == 1


def test_structured_review_junk_output_abstains() -> None:
    v = review_lane_env.single_verdict(
        _prov("I could not produce JSON, sorry."),
        _ctx(),
        reviewer_index=0,
        focus="s",
        model_route="m1",
    )
    assert v.error is not None and v.verdict == "abstain"  # abstain, never a false pass-as-real


@pytest.mark.parametrize("text", ["{}", '{"findings": []}', '{"verdict": "maybe"}'])
def test_structured_review_without_a_verdict_abstains(text: str) -> None:
    v = review_lane_env.single_verdict(
        _prov(text), _ctx(), reviewer_index=0, focus="s", model_route="m1"
    )
    assert v.error == "invalid reviewer verdict"


def test_structured_review_provider_error_abstains() -> None:
    v = review_lane_env.single_verdict(
        cast(Provider, _ErrProvider()), _ctx(), reviewer_index=0, focus="s", model_route="m1"
    )
    assert v.error is not None and "provider" in v.error


def test_structured_review_starved_output_names_the_cap() -> None:
    """A reasoning model can spend the whole output cap before emitting any content (kimi-k3.

    Finish_reason=length, 0 content chars, ~5.8k reasoning chars, every reviewer abstained).
    "unparseable reviewer output" blamed the parser for the provider's truncation and hid the one
    actionable fact.
    """
    starved = _FakeProvider("")
    starved_resp = _Resp("", stop_reason="length", output_tokens=4500)
    starved.call = lambda **_kw: starved_resp  # type: ignore[method-assign]
    v = review_lane_env.single_verdict(
        cast(Provider, starved), _ctx(), reviewer_index=0, focus="s", model_route="kimi-k3"
    )
    assert v.error is not None and v.verdict == "abstain"
    assert "unparseable" not in v.error
    assert "hit the cap" in v.error and "stop_reason=length" in v.error
    assert "4500" in v.error

    # Truncated mid-answer (partial JSON) names the truncation, not the parser.
    cut = _FakeProvider("")
    cut_resp = _Resp('{"verdict": "blo', stop_reason="length", output_tokens=1500)
    cut.call = lambda **_kw: cut_resp  # type: ignore[method-assign]
    v2 = review_lane_env.single_verdict(
        cast(Provider, cut), _ctx(), reviewer_index=0, focus="s", model_route="m1"
    )
    assert v2.error is not None
    assert "before the verdict JSON completed" in v2.error


def test_an_empty_reviewer_response_says_it_returned_nothing() -> None:
    """An empty reviewer response says it returned nothing.

    Observed live: the upstream answered finish_reason=error with a null body after the model spent
    16,801 tokens in the reasoning channel. There was nothing to parse, and "unparseable reviewer
    output" blames the parser for it.
    """
    empty = _FakeProvider("")
    empty_resp = _Resp("", stop_reason="error", output_tokens=16801)
    empty.call = lambda **_kw: empty_resp  # type: ignore[method-assign]

    v = review_lane_env.single_verdict(
        cast(Provider, empty), _ctx(), reviewer_index=0, focus="s", model_route="kimi-k2.6"
    )

    assert v.error is not None and v.verdict == "abstain"
    assert "unparseable" not in v.error
    assert "returned no content" in v.error
    assert "stop_reason=error" in v.error and "16801" in v.error

    # The sibling shape, observed on the same model: a clean stop, no content,
    # and the whole answer spent in the reasoning channel. Reported as bare
    # "no content" it read like the upstream error above, and the operator had
    # no way to tell a broken call from a model that only ever thinks.
    thinker = _FakeProvider("")
    thought = _Resp("", stop_reason="stop", output_tokens=7160, thinking="t" * 30_175)
    thinker.call = lambda **_kw: thought  # type: ignore[method-assign]

    v2 = review_lane_env.single_verdict(
        cast(Provider, thinker), _ctx(), reviewer_index=0, focus="s", model_route="kimi-k2.6"
    )

    assert v2.error is not None
    assert "30,175 chars of it in the reasoning channel" in v2.error


def test_coerce_findings_normalizes_bad_category_and_severity() -> None:
    raw = [
        {
            "category": "bogus",
            "severity": "critical",
            "location": {"path": "a", "side": "head", "start_line": 1, "end_line": 1},
            "title": "t",
        },
        "not a dict",
        {
            "category": "security",
            "severity": "block",
            "location": {"path": "b", "side": "head", "start_line": 2, "end_line": 2},
            "title": "ok",
        },
    ]
    fs = review_verdict.coerce_findings(raw)
    assert len(fs) == 2
    assert fs[0].category == "other" and fs[0].severity == "warn"  # normalized
    assert fs[1].category == "security" and fs[1].severity == "block"


@pytest.mark.parametrize(
    "location",
    [
        None,
        {"path": "../secret", "side": "head"},
        {"path": "foo.py", "side": "head", "start_line": 11},
        {"path": "foo.py", "side": "head", "start_line": 11, "end_line": 0},
    ],
)
def test_invalid_locations_abstain_instead_of_passing(location: object) -> None:
    payload = json.loads(_BLOCK_JSON)
    payload["findings"][0]["location"] = location
    verdict = review_lane_env.single_verdict(
        _prov(json.dumps(payload)), _ctx(), reviewer_index=3, focus="security", model_route="m1"
    )
    assert verdict.verdict == "abstain" and not verdict.findings
    assert verdict.error is not None and "invalid reviewer findings" in verdict.error
    assert verdict.reviewer_index == 3


def test_provider_findings_keep_the_title_and_detail_caps() -> None:
    payload = json.loads(_BLOCK_JSON)
    payload["findings"][0].update(title="t" * 201, detail="d" * 1001)
    verdict = review_lane_env.single_verdict(
        _prov(json.dumps(payload)), _ctx(), reviewer_index=0, focus="security", model_route="m1"
    )
    assert verdict.error is None
    assert verdict.findings[0].title == "t" * 200
    assert verdict.findings[0].detail == "d" * 1000


def test_finding_text_is_one_line() -> None:
    (finding,) = review_verdict.coerce_findings(
        [
            {
                "category": "security",
                "severity": "warn",
                "location": {"path": "foo.py", "side": "head", "start_line": 11, "end_line": 11},
                "title": "first\n- [block:security] invented",
                "detail": "one\ntwo",
            }
        ]
    )
    assert str(finding.location) == "foo.py:11"
    assert finding.title == "first - [block:security] invented"
    assert finding.detail == "one two"


def test_run_panel_distinct_models_quorum_blocks() -> None:
    reviewers = [
        review_lane_env.reviewer(focus="security", model="m1", provider=_prov(_BLOCK_JSON)),
        review_lane_env.reviewer(focus="correctness", model="m2", provider=_prov(_BLOCK_JSON)),
    ]
    res = review_lane_env.run_panel(
        reviewers,
        _ctx(),
        block_on="model_quorum",
        min_blocking_models=2,
        panel_id="p",
        max_turns=20,
    )
    assert res.blocked is True and res.n_block == 2
    assert [v.reviewer_index for v in res.per_reviewer] == [0, 1]
    assert [v.model_route for v in res.per_reviewer] == ["m1", "m2"]


def test_run_panel_advisory_never_blocks() -> None:
    reviewers = [
        review_lane_env.reviewer(focus="security", model="m1", provider=_prov(_BLOCK_JSON))
    ]
    res = review_lane_env.run_panel(
        reviewers, _ctx(), block_on="none", min_blocking_models=2, panel_id="p", max_turns=20
    )
    assert res.blocked is False
    assert (
        res.merged_findings and res.merged_findings[0].severity == "block"
    )  # reported, not gating


def test_run_panel_concurrent_preserves_order_and_aggregates() -> None:
    # 3 reviewers, distinct models, concurrency=3: results stay in reviewer order and the
    # grounded min_blocking_models still blocks (thread pool must not change the verdict).
    reviewers = [
        review_lane_env.reviewer(focus="security", model="m1", provider=_prov(_BLOCK_JSON)),
        review_lane_env.reviewer(focus="correctness", model="m2", provider=_prov(_BLOCK_JSON)),
        review_lane_env.reviewer(focus="edge", model="m3", provider=_prov(_BLOCK_JSON)),
    ]
    res = review_lane_env.run_panel(
        reviewers,
        _ctx(),
        block_on="model_quorum",
        min_blocking_models=2,
        panel_id="p",
        max_turns=20,
        concurrency=3,
    )
    assert [v.reviewer_index for v in res.per_reviewer] == [0, 1, 2]
    assert [v.focus for v in res.per_reviewer] == ["security", "correctness", "edge"]
    assert res.blocked is True and res.n_block == 3


# A routed reviewer leaves pinned reviewers on their configured models.


def _cfg_with_reviewers(reviewers: tuple[dict[str, str], ...]) -> Any:
    from agent6.config import Config

    return Config.model_validate(
        {
            "providers": {"anthropic": {"backend": "anthropic", "api_key_env": "FAKE_KEY"}},
            "models": {
                "default": {"model": "anthropic/default-model"},
                "reviewer": {"model": "anthropic/reviewer-default"},
            },
            "review": {"when": "finish", "reviewers": list(reviewers)},
        }
    )


def _stub_seat_provider(*_a: Any, **_k: Any) -> Provider:
    """Stand in for `_provider_from_entry` and `_build_role_provider` so a reviewer needs no key.

    The test asserts on the reviewer label.
    """
    return _prov("{}")


def test_a_routed_reviewer_leaves_pinned_reviewers_alone(monkeypatch: Any) -> None:
    """A routed reviewer leaves pinned reviewers alone.

    A reviewer pinned to a provider and model in `[review].reviewers` keeps that route.
    """
    from agent6.app import role_providers

    monkeypatch.setattr(role_providers, "_provider_from_entry", _stub_seat_provider)
    cfg = _cfg_with_reviewers(
        (
            {"focus": "security", "model": "anthropic/claude-opus-4-8"},
            {"focus": "correctness", "model": "anthropic/some-model"},
        )
    )
    cfg = cfg.with_model_route("reviewer", cfg.model_route("reviewer", "claude-haiku-override"))

    reviewers = role_providers.resolve_reviewers(cfg)
    assert [s.model for s in reviewers] == [
        "anthropic/claude-opus-4-8",
        "anthropic/some-model",
    ]
    assert [s.focus for s in reviewers] == ["security", "correctness"]


def test_build_reviewers_no_override_keeps_pinned_models(monkeypatch: Any) -> None:
    from agent6.app import role_providers

    monkeypatch.setattr(role_providers, "_provider_from_entry", _stub_seat_provider)
    cfg = _cfg_with_reviewers(({"focus": "security", "model": "anthropic/claude-opus-4-8"},))

    reviewers = role_providers.resolve_reviewers(cfg)
    assert reviewers[0].model == "anthropic/claude-opus-4-8"  # unchanged when no --model


def test_a_bare_focus_reviewer_follows_the_routed_reviewer(monkeypatch: Any) -> None:
    """A bare-focus reviewer inherits the reviewer role's route."""
    from agent6.app import role_providers

    monkeypatch.setattr(role_providers, "_provider_from_entry", _stub_seat_provider)
    cfg = _cfg_with_reviewers(({"focus": "correctness"},))
    cfg = cfg.with_model_route("reviewer", cfg.model_route("reviewer", "claude-haiku-override"))

    reviewers = role_providers.resolve_reviewers(cfg)
    assert reviewers[0].model == "anthropic/claude-haiku-override"


def test_bare_and_pinned_routes_to_the_same_model_count_once(monkeypatch: Any) -> None:
    from agent6.app import role_providers

    def blocking_provider(*_a: Any, **_k: Any) -> Provider:
        return _prov(_BLOCK_JSON)

    monkeypatch.setattr(role_providers, "_provider_from_entry", blocking_provider)
    cfg = _cfg_with_reviewers(
        (
            {"focus": "correctness"},
            {"focus": "security", "model": "anthropic/reviewer-default"},
        )
    )
    reviewers = role_providers.resolve_reviewers(cfg)
    assert [reviewer.model for reviewer in reviewers] == [
        "anthropic/reviewer-default",
        "anthropic/reviewer-default",
    ]
    reviewers = [review_lane_env.reviewer(s.focus, s.model, _prov(_BLOCK_JSON)) for s in reviewers]
    result = review_lane_env.run_panel(
        reviewers,
        _ctx(),
        block_on="model_quorum",
        min_blocking_models=2,
        panel_id="p",
        max_turns=20,
    )
    assert result.n_block == 1 and result.blocked is False


# --- the reviewer's tool loop --------------------------------------------------


class _ReviewerResp:
    def __init__(
        self, text: str = "", tool_uses: tuple[Any, ...] = (), raw: dict[str, Any] | None = None
    ) -> None:
        self.text = text
        self.tool_uses = list(tool_uses)
        self.raw = raw or {"content": []}


class _ReviewerProvider:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def call(self, **kw: Any) -> Any:
        self.calls += 1
        return self._responses.pop(0)


@pytest.mark.parametrize("max_turns", [1, 2, 20])
def test_reviewer_first_request_names_the_call_budget(max_turns: int) -> None:
    """The reviewer knows its total call budget and that the last call must be its verdict."""
    provider = mock.MagicMock()
    provider.call.return_value = _ReviewerResp(text=_BLOCK_JSON)
    verdict = review_lane_env.single_verdict(
        provider,
        _ctx(),
        reviewer_index=0,
        focus="security",
        model_route="m1",
        dispatch=mock.MagicMock(),
        max_turns=max_turns,
    )
    assert verdict.error is None
    system = provider.call.call_args_list[0].kwargs["system"]
    assert f"at most {max_turns} replies" in system
    assert "by the last reply at the latest, reply with ONLY the JSON verdict" in system


def test_reviewer_returns_a_verdict_on_the_sixth_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five tool rounds leave the sixth call for a verdict, however long they took."""
    clock = itertools.count(0, 100.0)
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    tu = {"name": "find_references", "id": "t1", "input": {"symbol": "read_doc"}}
    provider = _ReviewerProvider(
        [
            _ReviewerResp(tool_uses=(tu,), raw={"content": [{"type": "tool_use", **tu}]})
            for _ in range(5)
        ]
        + [_ReviewerResp(text=_BLOCK_JSON)]
    )
    dispatched: list[str] = []

    def dispatch(name: str, inp: dict[str, Any], *, origin: object = None) -> results.ToolResult:
        dispatched.append(name)
        return results.RawResult({"matches": ["caller.py:9: read_doc(x)"]})

    v = review_lane_env.single_verdict(
        cast(Provider, provider),
        _ctx(),
        reviewer_index=0,
        focus="security",
        model_route="m1",
        dispatch=dispatch,
        max_turns=6,
    )
    assert v.error is None and v.verdict == "block" and v.findings[0].category == "security"
    assert provider.calls == 6
    assert dispatched == ["find_references"] * 5


def test_reviewer_abstains_after_six_calls_without_a_verdict() -> None:
    """A reviewer that keeps requesting tools abstains after six provider calls."""
    tu = {"name": "list_dir", "id": "t1", "input": {"path": "."}}
    provider = _ReviewerProvider(
        [
            _ReviewerResp(tool_uses=(tu,), raw={"content": [{"type": "tool_use", **tu}]})
            for _ in range(10)
        ]
    )
    v = review_lane_env.single_verdict(
        cast(Provider, provider),
        _ctx(),
        reviewer_index=0,
        focus="s",
        model_route="m1",
        dispatch=lambda n, i, origin=None: results.RawResult({"entries": []}),
        max_turns=6,
    )
    assert v.error == "no verdict within 6 turns"
    assert provider.calls == 6


def test_reviewer_skips_dispatch_on_final_turn() -> None:
    # The last allowed model call returns tool_uses and no verdict: the reviewer is
    # about to abstain, and no model call follows to consume the results, so the
    # final round's tools must not be executed (pure waste).
    tu = {"name": "find_references", "id": "t1", "input": {"symbol": "x"}}
    provider = _ReviewerProvider(
        [
            _ReviewerResp(tool_uses=(tu,), raw={"content": [{"type": "tool_use", **tu}]})
            for _ in range(3)
        ]
    )
    dispatched: list[str] = []

    def dispatch(name: str, inp: dict[str, Any], *, origin: object = None) -> results.ToolResult:
        dispatched.append(name)
        return results.RawResult({"ok": True})

    v = review_lane_env.single_verdict(
        cast(Provider, provider),
        _ctx(),
        reviewer_index=0,
        focus="s",
        model_route="m1",
        dispatch=dispatch,
        max_turns=3,
    )
    assert v.error == "no verdict within 3 turns"
    assert provider.calls == 3  # every model call still happens
    assert dispatched == ["find_references", "find_references"]  # rounds 1-2; final skipped


def test_run_panel_runs_a_reviewer_through_its_tool_loop() -> None:
    tu = {"name": "find_references", "id": "t1", "input": {"symbol": "x"}}
    prov = _ReviewerProvider(
        [
            _ReviewerResp(tool_uses=(tu,), raw={"content": [{"type": "tool_use", **tu}]}),
            _ReviewerResp(text=_BLOCK_JSON),
        ]
    )
    reviewer = review_lane_env.reviewer(focus="security", model="m1", provider=cast(Provider, prov))
    res = review_lane_env.run_panel(
        [reviewer],
        _ctx(),
        block_on="any",
        min_blocking_models=2,
        panel_id="p",
        max_turns=2,
        dispatch=lambda n, i, origin=None: results.RawResult({"ok": True}),
    )
    assert prov.calls == 2 and res.blocked is True  # the reviewer ran the tool loop and blocked


def test_extract_json_prefers_the_verdict_object_over_a_stray_preamble() -> None:
    # A reasoning model may emit a throwaway object before the real verdict, or
    # wrap it in a fence with prose. Prefer the LAST object carrying verdict/findings.
    text = (
        'I will think first {"note": "scratch"} and here is my answer:\n'
        '```json\n{"verdict": "pass", "summary": "ok", "findings": []}\n```\n'
    )
    obj = llm_json.extract_json(text, prefer=("verdict", "findings"))
    assert obj is not None and obj["verdict"] == "pass" and obj["summary"] == "ok"


def test_extract_json_ignores_braces_inside_strings() -> None:
    obj = llm_json.extract_json(
        '{"verdict": "block", "summary": "a } brace { in text", "findings": []}',
        prefer=("verdict", "findings"),
    )
    assert obj is not None and obj["verdict"] == "block"


def test_reviewer_honors_verdict_alongside_tool_use_on_last_turn() -> None:
    # On the FINAL allowed turn the model emits a tool_use AND a verdict in
    # the same turn; the verdict must be honored (not wasted into an abstain).
    tu = {"name": "find_references", "id": "t1", "input": {"symbol": "x"}}
    provider = _ReviewerProvider(
        [_ReviewerResp(text=_BLOCK_JSON, tool_uses=(tu,), raw={"content": []})]
    )
    v = review_lane_env.single_verdict(
        cast(Provider, provider),
        _ctx(),
        reviewer_index=0,
        focus="s",
        model_route="m1",
        dispatch=lambda n, i, origin=None: results.RawResult({"ok": True}),
        max_turns=1,
    )
    assert v.error is None and v.verdict == "block"


def test_run_panel_executor_crash_retains_diagnostics() -> None:
    """An executor failure retains every reviewer as an abstention with its diagnostic."""

    class _BoomProvider:
        def call(self, **kw: Any) -> Any:
            raise RuntimeError("unexpected reviewer crash")

    reviewers = [
        review_lane_env.reviewer(focus="a", model="m", provider=_prov(_BLOCK_JSON)),
        review_lane_env.reviewer(focus="b", model="m", provider=cast(Provider, _BoomProvider())),
    ]
    result = review_lane_env.run_panel(
        reviewers,
        _ctx(),
        block_on="none",
        min_blocking_models=2,
        panel_id="p",
        max_turns=20,
        concurrency=2,
    )
    assert result.n_abstain == 2
    assert all("unexpected reviewer crash" in (v.error or "") for v in result.per_reviewer)


def test_run_panel_concurrency_limit_is_honored() -> None:
    """min(concurrency, len(reviewers)) reviewers run at once; the rest queue."""
    import threading
    import time

    lock = threading.Lock()
    active = 0
    peak = 0

    class _SlowProvider:
        def call(self, **kw: Any) -> Any:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return _Resp(_BLOCK_JSON)

    reviewers = [
        review_lane_env.reviewer(focus=f"s{i}", model="m", provider=cast(Provider, _SlowProvider()))
        for i in range(6)
    ]
    res = review_lane_env.run_panel(
        reviewers,
        _ctx(),
        block_on="none",
        min_blocking_models=2,
        panel_id="p",
        max_turns=20,
        concurrency=2,
    )
    assert len(res.per_reviewer) == 6
    assert peak <= 2


def test_empty_roster_builds_one_security_reviewer_on_the_reviewer_route(monkeypatch: Any) -> None:
    """An empty roster still builds the in-loop panel's default reviewer."""
    from agent6.app import role_providers

    monkeypatch.setattr(role_providers, "_provider_from_entry", _stub_seat_provider)
    reviewers = role_providers.resolve_reviewers(_cfg_with_reviewers(()))
    assert [(s.focus, s.model) for s in reviewers] == [
        ("security", "anthropic/reviewer-default"),
    ]


@pytest.mark.parametrize("extra", ["  " + "x" * 1100, None, ["a"]])
def test_an_unknown_reply_key_cannot_discard_a_valid_block(extra: object) -> None:
    """A key the reply format does not name never removes a valid verdict and its findings."""
    reply = json.loads(_BLOCK_JSON)
    reply["risk"] = extra
    verdict = review_verdict.verdict_from_obj(reply, 0, "security", "provider/model")
    assert verdict.verdict == "block" and len(verdict.findings) == 1


def test_reviewer_preserves_image_result() -> None:
    """A reviewer retains image blocks instead of flattening or clipping them."""
    tu = {"name": "read_file", "id": "t1", "input": {"path": "page.png"}}
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _ReviewerResp(tool_uses=(tu,), raw={"content": [{"type": "tool_use", **tu}]}),
        _ReviewerResp(text=_BLOCK_JSON),
    ]
    image = kinds.MediaAttachment(b"x" * 90000, "image/png", "page.png")
    verdict = review_lane_env.single_verdict(
        provider,
        _ctx(),
        reviewer_index=0,
        focus="ui",
        model_route="p/m",
        dispatch=lambda name, inp, origin=None: results.ReadMediaResult(image),
        max_turns=2,
    )
    assert verdict.error is None
    reply = provider.call.call_args_list[1].kwargs["messages"][2]["content"][0]
    assert reply["tool_use_id"] == "t1"
    assert reply["content"][1]["source"]["data"] == "eHh4" * 30000


def test_reviewer_uses_shared_media_recovery_without_rerunning_tools() -> None:
    """A reviewer retries its delivery rather than repeating read_file."""
    from agent6 import kinds, media
    from agent6.harness import provider_call
    from agent6.providers import provider_types
    from agent6.tools import results

    tu = {"name": "read_file", "id": "t1", "input": {"path": "page.png"}}
    provider = mock.MagicMock()
    provider.call.side_effect = [
        _ReviewerResp(tool_uses=(tu,), raw={"content": [{"type": "tool_use", **tu}]}),
        provider_types.MediaInputError(modality="image", reason="unsupported"),
        _ReviewerResp(text=_BLOCK_JSON),
    ]
    dispatch = mock.Mock(
        return_value=results.ReadMediaResult(kinds.MediaAttachment(b"x", "image/png", "page.png"))
    )
    emit = mock.Mock()
    verdict = review_lane_env.single_verdict(
        provider,
        _ctx(),
        reviewer_index=0,
        focus="ui",
        model_route="p/m",
        dispatch=dispatch,
        max_turns=3,
        recovery=provider_call.MediaRecovery(media.InputCapabilities(kinds.ModelRoute("p", "m"))),
        emit=emit,
    )
    assert verdict.error is None
    assert provider.call.call_count == 3 and dispatch.call_count == 1
    text = provider.call.call_args.kwargs["messages"][-1]["content"][0]["content"]
    assert text == (
        "Cannot deliver image 'page.png' to p/m: image input is unsupported. "
        "Describe the page through its DOM or accessibility tree, or read the test output."
    )
    assert emit.call_args.args[0].provider_tool_call_id == "t1"


def test_panel_media_refusals_follow_the_reviewer_not_the_shared_dispatch() -> None:
    """A text-only reviewer refuses locally while a vision one receives the same tool result."""
    from agent6 import kinds, media
    from agent6.tools import results

    tu = {"name": "read_file", "id": "t1", "input": {"path": "page.png"}}
    reviewers: list[review_gate.Reviewer] = []
    role_providers: list[Any] = []
    cases: tuple[tuple[str, media.Support], ...] = (
        ("text", "unsupported"),
        ("vision", "supported"),
    )
    for model, support in cases:
        provider = mock.MagicMock()
        provider.call.side_effect = [
            _ReviewerResp(tool_uses=(tu,), raw={"content": [{"type": "tool_use", **tu}]}),
            _ReviewerResp(text=_BLOCK_JSON),
        ]
        role_providers.append(provider)
        reviewers.append(
            review_lane_env.reviewer(
                focus="ui",
                model=f"p/{model}",
                provider=provider,
                media_input=media.InputCapabilities(kinds.ModelRoute("p", model), image=support),
            )
        )
    dispatch = mock.Mock(
        return_value=results.ReadMediaResult(kinds.MediaAttachment(b"x", "image/png", "page.png"))
    )
    review_lane_env.run_panel(
        reviewers,
        _ctx(),
        block_on="none",
        min_blocking_models=1,
        panel_id="p",
        max_turns=2,
        dispatch=dispatch,
    )
    replies = [
        p.call.call_args.kwargs["messages"][-1]["content"][0]["content"] for p in role_providers
    ]
    assert replies[0].startswith("Cannot deliver image 'page.png' to p/text:")
    assert replies[1][1]["type"] == "image"
    assert dispatch.call_count == 2


@pytest.mark.parametrize("cross_turn", [False, True])
def test_reviewer_records_attachment_limit_refusals(cross_turn: bool) -> None:
    """Turn and request limit refusals keep their tool-call identity in the journal."""
    from agent6 import kinds, media
    from agent6.tools import results

    def request(*ids: str) -> _ReviewerResp:
        calls = tuple(
            {"id": tid, "name": "read_file", "input": {"path": tid + ".png"}} for tid in ids
        )
        return _ReviewerResp(
            tool_uses=calls, raw={"content": [{"type": "tool_use", **c} for c in calls]}
        )

    provider = mock.MagicMock()
    provider.call.side_effect = [
        request("one"),
        *([request("two", "three")] if cross_turn else []),
        _ReviewerResp(text=_BLOCK_JSON),
    ]
    data = b"x" * (media.IMAGE_MAX_BYTES + (not cross_turn))
    dispatch = mock.Mock(
        return_value=results.ReadMediaResult(kinds.MediaAttachment(data, "image/png", "page.png"))
    )
    emit = mock.Mock()
    verdict = review_lane_env.single_verdict(
        provider,
        _ctx(),
        reviewer_index=0,
        focus="ui",
        model_route="p/m",
        dispatch=dispatch,
        max_turns=3 if cross_turn else 2,
        emit=emit,
    )
    assert verdict.error is None
    emit.assert_called_once()
    refused = emit.call_args.args[0]
    assert isinstance(refused, journal.MediaRefused)
    assert refused.provider_tool_call_id == "one"
    assert ("4 MiB" if cross_turn else "2 MiB") in refused.reason


@pytest.fixture(autouse=True)
def panel_lanes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review_gate, "run_panel", review_lane_env.run_panel)
