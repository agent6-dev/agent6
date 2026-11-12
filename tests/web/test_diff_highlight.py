from __future__ import annotations

import pathlib
from typing import Any

import pytest

from agent6.ui.web import git_model, highlight, model, review_routes
from agent6.vcs.git import diff as git_diff
from agent6.vcs.git import history as git_history
from agent6.vcs.git import read as git_read
from tests.unit import test_git_history


@pytest.fixture(params=["git", "run", "review"])
def producer(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _payload(repo: pathlib.Path, producer: str, base: str, head: str) -> dict[str, Any]:
    if producer == "git":
        return git_model.diff_payload(repo, base, head, "code.py")
    if producer == "review":
        status, payload = review_routes._diff(  # pyright: ignore[reportPrivateUsage]
            repo, base, head, git_history.changed_files(repo, base, head), "code.py", False
        )
        assert status == 200
        return payload
    payload, reason = model._diff_payload(  # pyright: ignore[reportPrivateUsage]
        repo, head, base=base, cumulative=False, miss="missing"
    )
    assert payload is not None, reason
    return payload


def _revisions(repo: pathlib.Path, old: bytes, new: bytes) -> tuple[str, str]:
    test_git_history._git(repo, "init", "-q")
    (repo / "code.py").write_bytes(old)
    base = test_git_history._commit(repo, "base")
    (repo / "code.py").write_bytes(new)
    return base, test_git_history._commit(repo, "head")


@pytest.mark.needs_grammar(language="python")
def test_each_diff_producer_highlights_both_blob_sides(
    tmp_path: pathlib.Path, producer: str
) -> None:
    """Diff spans use each complete blob, not patch-relative line numbers or checkout bytes."""
    base, head = _revisions(
        tmp_path, b"# old\ndef old():\n    return 1\n", b"def new():\n    return 2\n"
    )
    (tmp_path / "code.py").write_text("pass\n")
    payload = _payload(tmp_path, producer, base, head)
    assert (0, 3, "keyword") in payload["old_spans"]["code.py"][1]
    assert (0, 3, "keyword") in payload["new_spans"]["code.py"][0]
    assert len(payload["old_spans"]["code.py"]) == 3
    assert len(payload["new_spans"]["code.py"]) == 2


@pytest.mark.parametrize("plain", [b"\x00def binary():\n", b"#" * 129])
@pytest.mark.parametrize("side", ["old", "new"])
def test_each_diff_producer_leaves_binary_or_oversized_sides_plain(
    tmp_path: pathlib.Path,
    producer: str,
    plain: bytes,
    side: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binary and oversized input sides keep their patches but carry no spans."""
    monkeypatch.setattr(highlight, "HIGHLIGHT_MAX_BYTES", 128)
    code = b"def f():\n    return 1\n"
    base, head = _revisions(
        tmp_path, plain if side == "old" else code, plain if side == "new" else code
    )
    payload = _payload(tmp_path, producer, base, head)
    assert payload[f"{side}_spans"]["code.py"] == []
    assert payload["patch"]


def test_each_diff_producer_preserves_the_parsers_side_and_utf16_spans(
    tmp_path: pathlib.Path, producer: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The payload parses each complete side once and keeps the parser's UTF-16 columns."""
    old, new = 's = "😀"\npass\n', 's = "é"\npass\n'
    computed = {
        old: [[(4, 8, "string")], [(0, 4, "keyword")]],
        new: [[(4, 7, "string")], [(0, 4, "keyword")]],
    }

    def parse(text: str, language: str) -> list[list[highlight.Span]]:
        assert language == "python"
        return computed.pop(text)

    monkeypatch.setattr(highlight, "_compute", parse)
    base, head = _revisions(tmp_path, old.encode(), new.encode())
    payload = _payload(tmp_path, producer, base, head)
    assert payload["old_spans"]["code.py"] == [[(4, 8, "string")], [(0, 4, "keyword")]]
    assert payload["new_spans"]["code.py"] == [[(4, 7, "string")], [(0, 4, "keyword")]]


@pytest.mark.needs_grammar(language="python")
def test_worktree_diff_highlights_the_disk_side(tmp_path: pathlib.Path) -> None:
    """Working-tree highlights describe disk content rather than the index or HEAD."""
    base, _ = _revisions(tmp_path, b"pass\n", b"def committed():\n    pass\n")
    (tmp_path / "code.py").write_text("# disk\nclass Disk:\n    pass\n")
    payload = git_model.diff_payload(tmp_path, base, "WORKTREE", "code.py")
    assert (0, 4, "keyword") in payload["old_spans"]["code.py"][0]
    assert (0, 5, "keyword") in payload["new_spans"]["code.py"][1]


@pytest.mark.needs_grammar(language="python")
def test_run_diff_spans_cover_renames_multiple_files_and_cumulative_base(
    tmp_path: pathlib.Path,
) -> None:
    """A chain uses its base, while each renamed side keeps its own path and language."""
    base, _ = _revisions(tmp_path, b"pass\n", b"def f():\n    return 1\n")
    test_git_history._git(tmp_path, "mv", "code.py", "renamed.txt")
    (tmp_path / "second.py").write_text("class Second:\n    pass\n")
    head = test_git_history._commit(tmp_path, "rename and add")
    payload, reason = model._diff_payload(  # pyright: ignore[reportPrivateUsage]
        tmp_path, head, base=base, cumulative=True, miss="missing"
    )
    assert payload is not None, reason
    assert (0, 4, "keyword") in payload["old_spans"]["code.py"][0]
    assert payload["new_spans"]["renamed.txt"] == []
    assert (0, 5, "keyword") in payload["new_spans"]["second.py"][0]
    root = _payload(tmp_path, "run", git_read.EMPTY_TREE, base)
    assert root["old_spans"]["code.py"] == []
    assert (0, 4, "keyword") in root["new_spans"]["code.py"][0]


def test_converted_worktree_side_stays_plain(tmp_path: pathlib.Path) -> None:
    """CRLF conversion cannot attach raw-file offsets to normalized diff text."""
    test_git_history._git(tmp_path, "init", "-q")
    (tmp_path / ".gitattributes").write_text("*.py text eol=crlf\n")
    (tmp_path / "code.py").write_bytes(b"def old():\r\n    pass\r\n")
    base = test_git_history._commit(tmp_path, "base")
    (tmp_path / "code.py").write_bytes(b"def new():\r\n    pass\r\n")
    payload = git_model.diff_payload(tmp_path, base, "WORKTREE", "code.py")
    assert "+def new():\n" in payload["patch"]
    assert payload["new_spans"]["code.py"] == []


def test_full_context_without_text_hunks_highlights_the_captured_bytes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pure rename's displayed file rows use the diff's inputs, not a later disk read."""
    text = "pass\n"
    test_git_history._git(tmp_path, "init", "-q")
    (tmp_path / "code.py").write_text(text)
    base = test_git_history._commit(tmp_path, "base")
    test_git_history._git(tmp_path, "mv", "code.py", "renamed.py")
    (tmp_path / "renamed.py").write_text(text)
    original_read = git_diff.read

    def read(*args: Any, **kwargs: Any) -> git_diff.FileDiff:
        diff = original_read(*args, **kwargs)
        (tmp_path / "renamed.py").write_text("class Later:\n    pass\n")
        return diff

    monkeypatch.setattr(git_diff, "read", read)

    def parse(source: str, language: str) -> list[list[highlight.Span]]:
        assert (source, language) == (text, "python")
        return [[(0, 4, "keyword")]]

    monkeypatch.setattr(highlight, "_compute", parse)
    payload = git_model.diff_payload(
        tmp_path, base, "WORKTREE", "renamed.py", "code.py", full_context=True
    )
    assert "rename from code.py" in payload["patch"] and "@@" not in payload["patch"]
    assert payload["context"] == text
    assert payload["old_spans"]["code.py"] == [[(0, 4, "keyword")]]
    assert payload["new_spans"]["renamed.py"] == [[(0, 4, "keyword")]]


@pytest.mark.needs_grammar(language="python")
def test_cut_patch_highlights_only_displayed_absolute_lines(tmp_path: pathlib.Path) -> None:
    """A cut hunk retains absolute indexes and multiline syntax from outside the hunk."""
    from agent6.ui.web import diff_highlight

    text = '"""\n' + "hidden\n" * 10 + 'def not_code():\n"""\n'
    patch = "@@ -12,2 +12,2 @@\n-def not_code():\n+def still_not_code():\n"
    rows = diff_highlight.side_spans(patch, text.encode(), "code.py", "old")
    assert all(row == [] for row in rows[:11])
    assert rows[11] == [(0, 15, "string")]
    assert rows[12] == []


@pytest.mark.parametrize("producer", ["git", "run", "review"])
def test_diff_parse_timeout_leaves_the_patch_available(
    tmp_path: pathlib.Path, producer: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parser that misses its deadline cannot suppress or delay the patch indefinitely."""
    import threading

    started = threading.Event()
    release = threading.Event()

    def blocked(_text: str, _language: str) -> list[list[highlight.Span]]:
        started.set()
        release.wait(5)
        return []

    base, head = _revisions(tmp_path, b"pass\n", b"def f():\n    pass\n")
    monkeypatch.setattr(highlight, "_compute", blocked)
    monkeypatch.setattr(highlight, "TIMEOUT_S", 0.01)
    try:
        payload = _payload(tmp_path, producer, base, head)
        assert started.is_set()
        assert payload["old_spans"]["code.py"] == payload["new_spans"]["code.py"] == []
        assert "+def f():" in payload["patch"]
    finally:
        release.set()
