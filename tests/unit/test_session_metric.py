"""The operator's metric is one session's command, not a task or a config setting."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import directive
from agent6.sessions import ipc
from agent6.sessions import layout as sessions_layout
from agent6.tools import result_format
from agent6.ui import directive_actions
from agent6.ui.cli import parser


def test_run_metric_preserves_capture_and_command_arguments() -> None:
    spec = (
        r"""minimize 'cost -- ms: (\d+)' target -0.123456789 -- python3 'two words.py' target 8"""
    )
    args = parser.build_parser().parse_args(["run", "make it faster", "--metric", spec])
    assert args.metric.goal == "minimize"
    assert args.metric.pattern == r"cost -- ms: (\d+)"
    assert args.metric.target == -0.123456789
    assert args.metric.command == ("python3", "two words.py", "target", "8")
    assert directive.parse_metric_spec(args.metric.text()) == args.metric
    assert not hasattr(parser.build_parser().parse_args(["resume", "id"]), "metric")


@pytest.mark.parametrize(
    "spec",
    [
        "up -- true",
        "measure -- true",
        "maximize target -- true",
        "minimize target low -- true",
        "minimize target NaN -- true",
        "maximize target 1e999 -- true",
        "maximize target 8 target 9 -- true",
        "minimize",
        "maximize --",
        "maximize '(' -- true",
        "maximize 'no group' -- true",
        "maximize '(x{4294967296})' -- true",
        "minimize unquoted(1) -- true",
        "maximize -- 'broken",
        "minimize -- 'cost\rcommand'",
        "minimize -- cost\udcff",
    ],
)
def test_invalid_metric_is_not_queued_or_steered(tmp_path: pathlib.Path, spec: str) -> None:
    session = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="one").session_dir
    session.mkdir(parents=True)
    did, said = directive_actions.submit_composer_line(session, f"/metric {spec}", repo=tmp_path)
    assert not did, said
    assert not ipc.drain_requests(session)
    assert not ipc.steer_request_pending(session)


def test_only_a_leading_composer_directive_sets_the_metric(tmp_path: pathlib.Path) -> None:
    session = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="one").session_dir
    session.mkdir(parents=True)
    assert directive_actions.act_on_directive(session, "please /metric off") is None
    assert directive.steer_problem("/metric maximize -- score")
    did, said = directive_actions.submit_composer_line(session, "/metric", repo=tmp_path)
    assert did and "off" in said
    assert not ipc.drain_requests(session)
    assert not ipc.steer_request_pending(session)


@pytest.mark.parametrize(
    ("stdout", "stderr", "score"),
    [
        ("n=3, elapsed: -.25e+2 ms", "", -25.0),
        ("4.7", "best +1.5E-3", 0.0015),
        ("\033[31m42\033[0m", "", 42.0),
        ("no number", "", None),
        ("iteration 4: loss NaN", "", None),
        ("iteration 4: loss -INFINITY", "", None),
        ("[agent6-jail] output capture failed: historical diagnostic\nfinal: 22", "", 22.0),
    ],
)
def test_metric_without_a_pattern_reads_the_last_number(
    stdout: str, stderr: str, score: float | None
) -> None:
    assert result_format.parse_metric_score(stdout, stderr, pattern=None) == score


def test_live_metric_is_visible_on_adoption_and_off_resets_the_guard(
    tmp_path: pathlib.Path,
) -> None:
    from agent6.graph import curator as graph_curator
    from agent6.harness import loop_conversation, loop_state, loop_steer, metric_rules
    from agent6.sessions import event_log
    from tests.unit.test_task_queue_drain import _workflow  # pyright: ignore[reportPrivateUsage]

    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="one")
    wf = _workflow(graph_curator.GraphCurator(layout), event_log.EventSink(layout.logs_path))
    layout.ensure()
    state = loop_state.LoopState(original_task="cost", tool_calls=0)
    for text in ("maximize -- score target 8", "minimize target 0 -- elapsed", "off"):
        state.metric.history.append(
            metric_rules.MetricSample(label="unrelated", score=9, returncode=0)
        )
        did, said = directive_actions.submit_composer_line(
            layout.session_dir, f"/metric {text}", repo=tmp_path
        )
        assert did and "next" in said
        assert loop_steer.take_requests(wf, state, loop_conversation.Conversation())
        assert state.metric.history == []
        assert directive_actions.act_on_directive(layout.session_dir, "/metric") == (
            True,
            f"metric: {text}",
        )
        assert wf.metrics.active == (text != "off")
        if text != "off":
            assert "three times" in wf.metrics.notice() and "plateau" in wf.metrics.notice()
