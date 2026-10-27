from __future__ import annotations

import dataclasses
import pathlib
from typing import Literal

import pytest

from agent6 import review_records
from agent6.harness import _review_history
from agent6.vcs import git_files, git_read
from tests.unit.test_review_diff_collect import _git


def _commit(root: pathlib.Path, text: bytes, name: str = "code.py") -> str:
    (root / name).write_bytes(text)
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "edit")
    return _git(root, "rev-parse", "HEAD").strip()


def _target(root: pathlib.Path, base: str | None, head: str) -> review_records.ReviewTarget:
    return review_records.ReviewTarget(
        base,
        git_read.tree_sha(root, base) if base else git_read.EMPTY_TREE,
        head,
        git_read.tree_sha(root, head),
        head,
        (),
        "revisions",
    )


def _observe(
    root: pathlib.Path,
    target: review_records.ReviewTarget,
    start: int | None,
    end: int | None = None,
    *,
    path: str = "code.py",
    side: Literal["base", "head"] = "head",
) -> review_records.FindingObservation:
    finding = review_records.Finding(
        category="security",
        severity="block",
        title="Defect beside an edit",
        location=review_records.ReviewLocation(
            path=path, side=side, start_line=start, end_line=end if end is not None else start
        ),
    )
    verdict = review_records.ReviewVerdict(0, "security", "model", "block", (finding,))
    return _review_history.observations("panel", (verdict,), {}, root=root, target=target)[0]


@pytest.mark.parametrize(
    ("before", "after", "side", "start", "end", "kind", "base_line"),
    [
        (b"old\ncontext\n", b"new\ncontext\n", "head", 2, 2, "pre_existing", 2),
        (b"context\n", b"added\ncontext\n", "head", 2, 2, "pre_existing", 1),
        (b"old\ncontext\n", b"context\n", "head", 1, 1, "pre_existing", 2),
        (b"old\ncontext\n", b"new\ncontext\n", "head", 1, 1, "changed", None),
        (b"old\ncontext\n", b"context\n", "base", 1, 1, "changed", None),
        (b"old\ncontext\n", b"new\ncontext\n", "head", 1, 2, "changed", None),
        (b"old\ncontext\n", b"new\ncontext\n", "base", 1, 2, "changed", None),
        (b"old\ncontext\n", b"new\ncontext\n", "base", 2, 2, "pre_existing", 2),
        (b"first\ndeleted\nlast\n", b"first\nlast\n", "head", 1, 2, "unknown", None),
        (b"old\ncontext\r\n", b"new\ncontext\n", "head", 2, 2, "changed", None),
        (b"old\ncontext\n", b"new\ncontext", "head", 2, 2, "changed", None),
    ],
    ids=[
        "context",
        "insert-shift",
        "delete-shift",
        "replace",
        "delete",
        "mixed-head",
        "mixed-base",
        "base-context",
        "noncontiguous",
        "line-ending",
        "final-newline",
    ],
)
def test_observation_classifies_bytes_not_hunk_membership(
    tmp_path: pathlib.Path,
    before: bytes,
    after: bytes,
    side: Literal["base", "head"],
    start: int,
    end: int,
    kind: str,
    base_line: int | None,
) -> None:
    _git(tmp_path, "init", "-q")
    base = _commit(tmp_path, before)
    head = _commit(tmp_path, after)
    observation = _observe(tmp_path, _target(tmp_path, base, head), start, end, side=side)
    origin = observation.line_origin
    assert origin.kind == kind
    assert origin.reason
    assert observation.finding.severity == "block"
    if base_line is not None:
        assert origin.base_location == review_records.ReviewLocation(
            path="code.py", side="base", start_line=base_line, end_line=base_line
        )
        assert origin.commits == (base,)
    else:
        assert origin.base_location is None and not origin.commits


def test_pure_rename_preserves_an_exact_range(tmp_path: pathlib.Path) -> None:
    _git(tmp_path, "init", "-q")
    base = _commit(tmp_path, b"first\nsecond\n")
    _git(tmp_path, "mv", "code.py", "renamed.py")
    _git(tmp_path, "commit", "-qm", "rename")
    head = _git(tmp_path, "rev-parse", "HEAD").strip()
    origin = _observe(tmp_path, _target(tmp_path, base, head), 1, 2, path="renamed.py").line_origin
    assert origin.kind == "pre_existing"
    assert origin.base_location == review_records.ReviewLocation(
        path="code.py", side="base", start_line=1, end_line=2
    )
    assert origin.commits == (base,)


@pytest.mark.parametrize("evidence", ["file-only", "binary", "symlink", "history", "tree", "range"])
def test_unavailable_or_unsupported_evidence_is_unknown(
    tmp_path: pathlib.Path, evidence: str
) -> None:
    _git(tmp_path, "init", "-q")
    base = _commit(tmp_path, b"old\ncontext\n")
    head = _commit(tmp_path, b"new\ncontext\n")
    target = _target(tmp_path, base, head)
    if evidence == "binary":
        head = _commit(tmp_path, b"new\0\ncontext\n")
        target = _target(tmp_path, base, head)
    elif evidence == "symlink":
        (tmp_path / "code.py").unlink()
        (tmp_path / "code.py").symlink_to("outside")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-qm", "link")
        target = _target(tmp_path, base, _git(tmp_path, "rev-parse", "HEAD").strip())
    elif evidence == "history":
        target = dataclasses.replace(target, base_commit="f" * 40)
    elif evidence == "tree":
        target = dataclasses.replace(target, base_tree="f" * 40)
    start = None if evidence == "file-only" else 99 if evidence == "range" else 2
    origin = _observe(tmp_path, target, start).line_origin
    assert origin.kind == "unknown" and origin.reason


def test_root_commit_additions_are_changed(tmp_path: pathlib.Path) -> None:
    _git(tmp_path, "init", "-q")
    head = _commit(tmp_path, b"first\n")
    assert _observe(tmp_path, _target(tmp_path, None, head), 1).line_origin.kind == "changed"


def test_blame_is_bounded_and_ignore_revs_are_honoured(tmp_path: pathlib.Path) -> None:
    _git(tmp_path, "init", "-q")
    first = _commit(tmp_path, b"one\ntwo\nthree\n")
    second = _commit(tmp_path, b"one\nTWO\nthree\n")
    (tmp_path / ".git-blame-ignore-revs").write_text(second + "\n")
    _git(tmp_path, "config", "blame.ignoreRevsFile", ".git-blame-ignore-revs")
    rows = git_files.blame_lines(
        tmp_path, second, "code.py", b"one\nTWO\nthree\n", start_line=2, lines=1
    )
    assert [(row.line, row.sha) for row in rows] == [(2, first)]
    head = _commit(tmp_path, b"changed\nTWO\nthree\n")
    assert _observe(tmp_path, _target(tmp_path, second, head), 2).line_origin.commits == (first,)


@pytest.mark.parametrize("author", ["t", "other"], ids=["same", "different"])
@pytest.mark.parametrize("date", ["2000-01-01T00:00:00+0000", "2030-01-01T00:00:00+0000"])
def test_author_and_time_do_not_classify_lines(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, author: str, date: str
) -> None:
    _git(tmp_path, "init", "-q")
    base = _commit(tmp_path, b"old\ncontext\n")
    monkeypatch.setenv("GIT_AUTHOR_NAME", author)
    monkeypatch.setenv("GIT_AUTHOR_DATE", date)
    monkeypatch.setenv("GIT_COMMITTER_DATE", date)
    head = _commit(tmp_path, b"new\ncontext\n")
    target = _target(tmp_path, base, head)
    assert _observe(tmp_path, target, 2).line_origin.kind == "pre_existing"
    assert _observe(tmp_path, target, 1).line_origin.kind == "changed"
