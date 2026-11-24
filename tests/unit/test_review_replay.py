from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any
from unittest import mock

import httpx2
import pytest
from bench.review import recordings, replay, scoring

from agent6 import kinds, providers
from agent6.prompts import review
from agent6.providers import transport
from tests import review_replay_env as env


def finding(category: str, title: str, detail: str, severity: str = "block") -> dict[str, Any]:
    return {
        "category": category,
        "title": title,
        "detail": detail,
        "location": {"path": "schema.py", "side": "head", "start_line": 7, "end_line": 7},
        "severity": severity,
    }


def answer(*findings: dict[str, Any]) -> str:
    return json.dumps(
        {"verdict": "block" if findings else "pass", "summary": "", "findings": findings}
    )


@pytest.fixture
def responses_config(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "provider.toml"
    config.write_text(
        '[models.default]\nmodel = "p/other"\neffort = "low"\n[providers.p]\n'
        'backend = "responses"\nbase_url = "https://scripted.invalid/v1"\nauth_style = "none"\n'
    )
    return config


def test_cli_replays_one_call_per_arm_and_reports_only_comparable_pairs(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    responses_config: pathlib.Path,
) -> None:
    """Score the public command using actual provider request bodies but no network."""
    state = tmp_path / "state"
    second = "beta\tcopy\nrow\x01\u202e"
    for sid in ("alpha", second):
        env.record_review(state, sid, env.responses_body())

    reader = finding("security", "Metadata exposed", "An unauthenticated client can read the row.")
    bug = finding("data-loss", "Loses zero", "Zero becomes null.")
    warning = finding("style", "Wrap line", "Long line.", "warn")
    both = answer(reader, warning, bug)
    changed = {**reader, "title": "Row visible", "detail": "Reader: an API client."}
    tool = {"type": "function_call", "call_id": "x", "name": "run_command", "arguments": "{}"}
    # Odd pairs: recorded then tree. Even pairs: tree then recorded.
    script = iter(
        [
            both,
            answer(changed),
            answer(),
            both,
            [tool],
            both,
            answer(),
            answer(),
            "{broken",
            answer(),
            "capped",
            answer(),
        ]
    )
    requests: list[dict[str, Any]] = []

    def post(
        url: str, *, headers: dict[str, str], content: bytes, timeout: float
    ) -> httpx2.Response:
        assert url == "https://scripted.invalid/v1/responses"
        requests.append(json.loads(content))
        item = next(script)
        output = (
            item
            if isinstance(item, list)
            else [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": item}],
                }
            ]
        )
        reply: dict[str, Any] = {
            "status": "completed",
            "output": output,
            "usage": {"input_tokens": 18, "output_tokens": 9},
        }
        if item == "capped":
            reply.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        return httpx2.Response(200, json=reply)

    monkeypatch.setattr(transport, "http_post", post)
    assert (
        replay.main(
            [
                str(state),
                "--config",
                str(responses_config),
                "--model",
                "p/selected",
                "--effort",
                "high",
                "-n",
                "3",
            ]
        )
        == 0
    )
    out, table = capsys.readouterr()
    rows = [json.loads(line) for line in out.splitlines()]
    assert len(rows) == len(requests) == 12
    for sent, row in zip(requests, rows, strict=True):
        assert sent["instructions"] == (
            env.SYSTEM
            if row["arm"] == "recorded"
            else review.panel_prompt("security", max_turns=7, commands=True)
        )
        for field in ("input", "tools", "max_output_tokens", "temperature"):
            assert sent[field] == env.responses_body()[field]
        assert row["system_sha256"] == hashlib.sha256(sent["instructions"].encode()).hexdigest()
        assert sent["model"] == "selected" and sent["reasoning"]["effort"] == "high"
        assert row["model"] == "p/selected" and row["requested_effort"] == "high"
    assert [r["repeat"] for r in rows] == [1, 1, 2, 2, 3, 3] * 2
    assert {r["session"] for r in rows[:6]} == {"alpha"}
    assert {r["session"] for r in rows[6:]} == {second}
    assert [r["outcome"] for r in rows[4:6]] == ["asked for a tool", "block"]
    assert rows[4]["tools_requested"] == ["run_command"]
    assert rows[8]["outcome"] == "invalid reply" and rows[10]["outcome"] == "truncated"
    blocks = rows[0]["blocks"]
    assert [(b["category"], b["names_reader"]) for b in blocks] == [
        ("security", True),
        ("data-loss", False),
    ]
    assert blocks[0]["reader_evidence"] and blocks[0]["detail"] == reader["detail"]
    assert "alpha\trecorded\tblock=2, asked for a tool=1" in table
    assert "alpha\ttree\tblock=2, pass=1" in table
    assert "data-loss=2, security=2" in table
    assert "compared pairs\tverdict changes\tblocks kept\tblocks lost" in table
    label = json.dumps(second)
    assert second not in table and f"{label}\trecorded\t" in table and f"{label}\ttree\t" in table
    assert "alpha\t2/3\t1\t1\t3" in table and f"{label}\t1/3\t0\t0\t0" in table
    assert "Reviews whose verdict changed: 1/2; blocks kept: 1; blocks lost: 3" in table


def test_no_recordings_fail_before_calling_the_provider(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert replay.main([str(tmp_path), "--model", "p/model", "--effort", "default"]) != 0
    out, err = capsys.readouterr()
    assert not out and "no finished panel reviewers" in err


@pytest.mark.parametrize(
    "part",
    ["sessions", "reviewers", "child", "completion", "transcripts", "opening", "objects", "object"],
)
def test_a_link_in_copied_state_stops_replay_before_any_call(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    responses_config: pathlib.Path,
    part: str,
) -> None:
    """A later review cannot draw on files outside the selected copy."""
    state = tmp_path / "copy"
    env.record_review(state, "first", env.responses_body())
    child = env.record_review(state, "second", env.responses_body())
    opening = sorted(child.transcripts_dir.glob("*.json"))[1]
    digest = json.loads(opening.read_text())["request"]["body"]["instructions"]["$object"]
    path = {
        "sessions": state / "sessions",
        "reviewers": child.session_dir.parent,
        "child": child.session_dir,
        "completion": child.session_dir / "completion.json",
        "transcripts": child.transcripts_dir,
        "opening": opening,
        "objects": child.objects_dir,
        "object": child.objects_dir / f"{digest}.json",
    }[part]
    source = path.rename(tmp_path / "fixture-source")
    path.symlink_to(source, target_is_directory=source.is_dir())
    monkeypatch.setattr(
        transport, "http_post", mock.Mock(side_effect=AssertionError("unexpected call"))
    )

    code = replay.main(
        [str(state), "--config", str(responses_config), "--model", "p/m", "--effort", "default"]
    )
    out, err = capsys.readouterr()
    assert code == 1 and not out
    assert "symlink" in err and str(path.relative_to(state)) in err


def test_cli_refuses_a_reasoning_override_before_spending(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    responses_config: pathlib.Path,
) -> None:
    """A Responses body override would send low while the experiment requests high."""
    env.record_review(tmp_path, "first", env.responses_body())
    with responses_config.open("a") as file:
        file.write('[providers.p.extra_body.reasoning]\neffort = "low"\n')
    sent: list[str] = []

    def post(url: str, *, content: bytes, **kwargs: Any) -> httpx2.Response:
        sent.append(json.loads(content)["reasoning"]["effort"])
        message = {"type": "message", "content": [{"type": "output_text", "text": answer()}]}
        return httpx2.Response(
            200,
            json={
                "status": "completed",
                "output": [message],
                "usage": {"input_tokens": 18, "output_tokens": 9},
            },
        )

    monkeypatch.setattr(transport, "http_post", post)
    code = replay.main(
        [str(tmp_path), "--config", str(responses_config), "--model", "p/m", "--effort", "high"]
    )
    out, err = capsys.readouterr()
    assert not sent, f"sent {sent}; calls: {out}"
    assert code == 1 and not out
    assert "extra_body.reasoning" in err and "--effort" in err


def test_a_provider_failure_leaves_a_call_record_not_a_pass(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env.record_review(tmp_path, "first", env.responses_body())
    provider = mock.Mock(spec=providers.Provider)
    provider.call.side_effect = providers.ProviderError("service unavailable")
    settings = kinds.ModelSettings(kinds.ModelRoute("p", "chosen"))
    with pytest.raises(providers.ProviderError, match="service unavailable"):
        replay.replay(recordings.read_reviews(tmp_path), provider, settings=settings, repeats=2)
    row = json.loads(capsys.readouterr().out)
    assert (row["arm"], row["repeat"], row["verdict"]) == ("recorded", 1, None)
    assert row["outcome"] == "call failed" and row["error"] == "service unavailable"
    assert provider.call.call_count == 1


def test_cli_displays_a_gateway_error_as_text(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    responses_config: pathlib.Path,
) -> None:
    env.record_review(tmp_path, "first", env.responses_body())
    failure = mock.Mock(return_value=httpx2.Response(400, text="bad\x01 request"))
    monkeypatch.setattr(transport, "http_post", failure)
    assert (
        replay.main(
            [str(tmp_path), "--config", str(responses_config), "--model", "p/m", "--effort", "high"]
        )
        == 1
    )
    out, err = capsys.readouterr()
    assert json.loads(out)["outcome"] == "call failed"
    assert "bad request" in err and "\x01" not in err


@pytest.mark.parametrize(
    "reader",
    [
        "A caller can read it",
        "Reader: unknown",
        "Reader: `unknown`",
        "Attacker: ` none `",
        "Consumer: `unspecified`",
        "No reader named",
    ],
)
def test_generic_reader_mentions_do_not_name_one(reader: str) -> None:
    assert scoring.reader_evidence(reader) is None


def test_a_code_span_can_name_a_reader() -> None:
    assert scoring.reader_evidence("Reader: `build bot`") == "Reader: `build bot`"
