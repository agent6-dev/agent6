from __future__ import annotations

import dataclasses
import json
import pathlib
from unittest import mock

import pytest

from agent6 import review_records
from agent6.harness import loop_state, repo_context, run_chain
from agent6.harness.review import gate as review_gate
from agent6.sessions import event_log
from agent6.vcs import git_files, git_read, review
from tests import review_helpers, review_lane_env
from tests.unit import (
    test_review_gate,
    test_review_line_origin,
)


@pytest.mark.parametrize("side", ["base", "head"])
@pytest.mark.parametrize("missing_blob", [False, True], ids=["present", "unreadable"])
def test_line_origin_refuses_hidden_rename_evidence(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, side: str, missing_blob: bool
) -> None:
    """Hidden rename candidates must not be read or recorded as line-origin evidence."""
    review_helpers.git(tmp_path, "init", "-q")
    base = test_review_line_origin._commit(
        tmp_path, b"first unchanged line\nsecond unchanged line\n", "old.txt"
    )
    review_helpers.git(tmp_path, "mv", "old.txt", "new.txt")
    (tmp_path / "new.txt").write_text("first unchanged line\nsecond unchanged line\nadded\n")
    review_helpers.git(tmp_path, "commit", "-qam", "rename")
    assert review_helpers.git(tmp_path, "diff", "-M", "--name-status", "HEAD~1", "HEAD").startswith(
        "R"
    )
    hidden = "old.txt" if side == "head" else "new.txt"
    visible = "new.txt" if side == "head" else "old.txt"
    hide = (str(tmp_path / hidden),)
    head = review_helpers.git(tmp_path, "rev-parse", "HEAD").strip()
    target = dataclasses.replace(
        test_review_line_origin._target(tmp_path, base, head), paths=(visible,)
    )
    if missing_blob:
        tree = target.base_tree if hidden == "old.txt" else target.head_tree
        blob = review_helpers.git(tmp_path, "rev-parse", f"{tree}:{hidden}").strip()
        (tmp_path / ".git/objects" / blob[:2] / blob[2:]).unlink()
    blobs = mock.Mock(wraps=git_files.blob_at)
    monkeypatch.setattr(git_files, "blob_at", blobs)

    origin = review.classify_line_origin(
        tmp_path,
        target,
        review_records.ReviewLocation(
            path=visible, side="head" if side == "head" else "base", start_line=1, end_line=1
        ),
        hide=hide,
    )

    assert origin.kind == "unknown", origin
    assert "sandbox.hide_paths" in origin.reason
    assert hidden not in repr(origin)
    assert origin.base_location is None and not origin.commits
    blobs.assert_not_called()


@pytest.mark.parametrize("hidden", [False, True], ids=["visible", "hidden"])
def test_in_loop_subdirectory_panel_uses_repository_relative_evidence(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, hidden: bool
) -> None:
    """Subdirectory panels must check citations and hidden rename sources at the repo root."""
    review_helpers.git(tmp_path, "init", "-q")
    nested = tmp_path / "sub"
    nested.mkdir()
    old_name, new_name = "sub/secret.py", "sub/new.py"
    base = test_review_line_origin._commit(
        tmp_path, b"first unchanged line\nsecond unchanged line\n", old_name
    )
    monkeypatch.chdir(nested)
    chain = run_chain.RunChain(
        pathlib.Path.cwd(), ref="refs/agent6/test/head", fallback_parent=base, base_sha=base
    )
    (tmp_path / old_name).rename(tmp_path / new_name)
    (tmp_path / new_name).write_text("first unchanged line\nsecond unchanged line\nadded\n")
    tip = chain.commit("rename")
    assert chain.root == nested and tip
    hide = (str(tmp_path / old_name),) if hidden else ()
    provider = mock.MagicMock()
    provider.call.return_value = test_review_gate._resp(
        json.dumps(
            dict(
                verdict="block",
                findings=[
                    dict(
                        category="security",
                        severity="block",
                        title="Unchanged citation",
                        location=dict(path=new_name, side="head", start_line=1, end_line=1),
                    )
                ],
            )
        )
    )
    sink = event_log.EventSink(tmp_path / "journal" / "logs.jsonl")
    blobs = mock.Mock(wraps=git_files.blob_at)
    monkeypatch.setattr(git_files, "blob_at", blobs)
    reviewer = review_gate.Reviewer(
        settings=review_gate.ReviewSettings(
            reviewers=[review_lane_env.seat("security", "model", provider)], block_on="any"
        ),
        chain=chain,
        hide=hide,
        executor=mock.MagicMock,
        budget_remaining=lambda: None,
        log=lambda _s: None,
        emit=sink.emit,
    )
    state = loop_state.LoopState(original_task="t", tool_calls=0)

    result = reviewer.critique(state, trigger="finish", turn=1)

    assert result is not None and not result.satisfied
    rows = [json.loads(line) for line in sink.path.read_text().splitlines()]
    panels = [
        review_records.ReviewPanel.model_validate(event_log.payload(row))
        for row in rows
        if row["type"] == "review.panel"
    ]
    assert len(panels) == 1 and state.review_panels == (panels[0],)
    record = panels[0]
    assert record.target.snapshot_commit == tip and len(record.findings) == 1
    observation = record.findings[0]
    origin = observation.line_origin
    if hidden:
        assert origin.kind == "unknown"
        assert origin.reason == "sandbox.hide_paths prevents complete line-origin evidence."
        assert old_name not in repr(origin)
        assert origin.base_location is None and not origin.commits
        blobs.assert_not_called()
    else:
        assert origin.kind == "pre_existing", origin
        assert origin.base_location == review_records.ReviewLocation(
            path=old_name, side="base", start_line=1, end_line=1
        )
        assert origin.commits == (base,)
    for root in (tmp_path, nested):
        assert origin == review.classify_line_origin(
            root, record.target, observation.finding.location, hide=hide
        )


@pytest.mark.parametrize("unsafe", ["absolute", "traversal", "symlink"])
def test_line_origin_refuses_unsafe_blame_configuration(
    tmp_path: pathlib.Path, unsafe: str
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    review_helpers.git(root, "init", "-q")
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
    review_helpers.git(root, "config", "blame.ignoreRevsFile", value)
    origin = test_review_line_origin._observe(
        root, test_review_line_origin._target(root, base, head), 2
    ).line_origin
    assert origin.kind == "unknown"
    assert "blame.ignoreRevsFile" in origin.reason and "refused" in origin.reason
    assert outside.read_text() == base + "\n"


@pytest.mark.parametrize("source", ["config", "shallow", "rules"])
def test_review_reasons_do_not_quote_hidden_git_inputs(tmp_path: pathlib.Path, source: str) -> None:
    """Git diagnostics can quote hidden config values or invalid shallow-history lines."""
    review_helpers.git(tmp_path, "init", "-q")
    base = test_review_line_origin._commit(tmp_path, b"old\ncontext\n")
    head = test_review_line_origin._commit(tmp_path, b"new\ncontext\n")
    target = test_review_line_origin._target(tmp_path, base, head)
    secret = tmp_path / "secret.env"
    if source == "shallow":
        secret.write_text("SECRET-ENV-LINE-4242\n")
        (tmp_path / ".git/shallow").symlink_to(secret)
    else:
        secret.write_text(
            "[core]\nabbrev = SECRET-ENV-LINE-4242\n"
            if source == "rules"
            else "[diff]\nalgorithm = SECRET-ENV-LINE-4242\n"
        )
        review_helpers.git(tmp_path, "config", "include.path", "../secret.env")
    if source == "rules":
        rules = repo_context.review_rules(tmp_path, target.base_tree, (str(secret),))
        assert not rules.complete
        reason = rules.reason
    else:
        origin = review.classify_line_origin(
            tmp_path,
            target,
            review_records.ReviewLocation(path="code.py", side="head", start_line=2, end_line=2),
            hide=(str(secret),),
        )
        assert origin.kind == "unknown"
        reason = origin.reason
    assert reason and "SECRET-ENV-LINE-4242" not in reason
    assert "git " in reason and "failed" in reason and "exit " in reason


@pytest.mark.parametrize(
    "refusal",
    ["ordinary", "hidden", "absolute", "traversal", "missing", "directory", "symlink", "oversize"],
)
def test_panel_classifies_immutable_bytes_before_its_single_record(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, refusal: str
) -> None:
    review_helpers.git(tmp_path, "init", "-q")
    base = test_review_line_origin._commit(tmp_path, b"old\ncontext\n")
    secret = tmp_path / "secret.env"
    hide = (str(secret),) if refusal != "ordinary" else ()
    chain = run_chain.RunChain(
        tmp_path,
        ref="refs/agent6/test/head",
        fallback_parent=base,
        base_sha=base,
        hidden=git_read.hidden_in_repo(tmp_path, hide),
    )
    (tmp_path / "code.py").write_text("new\ncontext\n")
    tip = chain.commit("edit")
    if refusal == "hidden":
        secret.write_text("SECRET-ENV-LINE-4242\n")
        review_helpers.git(tmp_path, "config", "blame.ignoreRevsFile", "secret.env")
    elif refusal != "ordinary":
        value = "SECRET-CONFIG-4242\x1b]52;c;c2VjcmV0\x07"
        if refusal == "absolute":
            value = str(tmp_path / value)
        elif refusal == "traversal":
            value = "../" + value
        elif refusal == "directory":
            (tmp_path / value).mkdir()
        elif refusal == "symlink":
            (tmp_path / value).symlink_to("code.py")
        elif refusal == "oversize":
            (tmp_path / value).write_bytes(b"x" * (1024 * 1024 + 1))
        review_helpers.git(tmp_path, "config", "--file", str(secret), "blame.ignoreRevsFile", value)
        review_helpers.git(tmp_path, "config", "include.path", "../secret.env")
    opened = mock.Mock(wraps=git_files._open_regular)
    monkeypatch.setattr(git_files, "_open_regular", opened)
    # The config files above are the operator's: untracked when the run started.
    chain = dataclasses.replace(chain, untracked_at_start=git_read.untracked_paths(tmp_path))
    index = (tmp_path / ".git" / "index").read_bytes()
    refs = review_helpers.git(tmp_path, "show-ref")
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
    reviewer = review_gate.Reviewer(
        settings=review_gate.ReviewSettings(
            reviewers=[review_lane_env.seat("security", "model", provider)], block_on="any"
        ),
        chain=chain,
        hide=hide,
        executor=mock.MagicMock,
        budget_remaining=lambda: None,
        log=lambda _s: None,
        emit=sink.emit,
    )
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    result = reviewer.critique(state, trigger="finish", turn=1)
    assert result is not None and not result.satisfied
    panels = [row for row in rows if row["type"] == "review.panel"]
    assert len(panels) == 1
    record = state.review_panels[0]
    assert record.target.snapshot_commit == tip
    assert "SECRET-ENV-LINE-4242" not in sink.path.read_text()
    assert "SECRET-CONFIG-4242" not in sink.path.read_text()
    if refusal in ("ordinary", "hidden", "absolute", "traversal"):
        opened.assert_not_called()
    assert [f.line_origin.kind for f in record.findings] == [
        "changed",
        "pre_existing" if refusal == "ordinary" else "unknown",
    ]
    if refusal != "ordinary":
        reason = record.findings[1].line_origin.reason
        assert "blame.ignoreRevsFile" in reason and "refused" in reason
        if refusal == "hidden":
            assert "sandbox.hide_paths" in reason
        else:
            assert ("1 MiB" if refusal == "oversize" else "regular file") in reason
    assert all(f.finding.severity == "block" for f in record.findings)
    assert not any(row["type"] == "review.line_origin" for row in rows)
    assert (tmp_path / ".git" / "index").read_bytes() == index
    assert review_helpers.git(tmp_path, "show-ref") == refs
    assert (tmp_path / "code.py").read_text() == "workspace\nnot context\n"


@pytest.fixture(autouse=True)
def panel_lanes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review_gate, "run_panel", review_lane_env.run_panel)
