from __future__ import annotations

import json
import pathlib
from unittest import mock

import pytest

from agent6 import event_log
from agent6.harness import _chain, _loop_state, _reviewer
from tests.unit import test_review_gate, test_review_line_origin
from tests.unit.test_review_diff_collect import _git


@pytest.mark.parametrize("unsafe", ["absolute", "traversal", "symlink"])
def test_line_origin_refuses_unsafe_blame_configuration(
    tmp_path: pathlib.Path, unsafe: str
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    base = test_review_line_origin._commit(root, b"old\ncontext\n")
    head = test_review_line_origin._commit(root, b"new\ncontext\n")
    outside = tmp_path / "outside"
    outside.write_text(base + "\n")
    (root / "ignore").symlink_to(outside)
    value = (
        str(outside)
        if unsafe == "absolute"
        else "../outside"
        if unsafe == "traversal"
        else "ignore"
    )
    _git(root, "config", "blame.ignoreRevsFile", value)
    origin = test_review_line_origin._observe(
        root, test_review_line_origin._target(root, base, head), 2
    ).line_origin
    assert origin.kind == "unknown"
    assert "blame.ignoreRevsFile" in origin.reason and "refused" in origin.reason
    assert outside.read_text() == base + "\n"


def test_panel_classifies_immutable_bytes_before_its_single_record(tmp_path: pathlib.Path) -> None:
    _git(tmp_path, "init", "-q")
    base = test_review_line_origin._commit(tmp_path, b"old\ncontext\n")
    chain = _chain.RunChain(
        tmp_path, ref="refs/agent6/test/head", fallback_parent=base, base_sha=base
    )
    (tmp_path / "code.py").write_text("new\ncontext\n")
    tip = chain.commit("edit")
    index = (tmp_path / ".git" / "index").read_bytes()
    refs = _git(tmp_path, "show-ref")
    sink = event_log.EventSink(tmp_path / "journal" / "logs.jsonl")
    rows: list[dict[str, object]] = []
    sink.subscribe(rows.append)
    provider = mock.MagicMock()

    def seat(**_kwargs: object) -> object:
        (tmp_path / "code.py").write_text("workspace\nnot context\n")
        return test_review_gate._resp(
            json.dumps(
                dict(
                    verdict="block",
                    summary="two defects",
                    findings=[
                        dict(
                            category="security",
                            severity="block",
                            title=f"Defect {line}",
                            location=dict(
                                path="code.py", side="head", start_line=line, end_line=line
                            ),
                        )
                        for line in (1, 2)
                    ],
                )
            )
        )

    provider.call.side_effect = seat
    reviewer = _reviewer.Reviewer(
        settings=_reviewer.ReviewSettings(
            reviewers=[_reviewer.ReviewSeat("security", "model", provider)], block_on="any"
        ),
        chain=chain,
        hide=(),
        review_tools=lambda: ([], mock.MagicMock()),
        budget_remaining=lambda: None,
        log=lambda _s: None,
        emit=sink.emit,
    )
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    result = reviewer.critique(state, trigger="before_finish", turn=1)
    assert result is not None and not result.satisfied
    panels = [row for row in rows if row["type"] == "loop.review.panel"]
    assert len(panels) == 1
    record = state.review_panels[0]
    assert record.target.snapshot_commit == tip
    assert [f.line_origin.kind for f in record.findings] == ["changed", "pre_existing"]
    assert all(f.finding.severity == "block" for f in record.findings)
    assert not any(row["type"] == "loop.review.line_origin" for row in rows)
    assert (tmp_path / ".git" / "index").read_bytes() == index
    assert _git(tmp_path, "show-ref") == refs
    assert (tmp_path / "code.py").read_text() == "workspace\nnot context\n"
