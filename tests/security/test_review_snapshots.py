from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
from unittest import mock

import pytest

from agent6 import errors, event_log, paths, review_records
from agent6.app import _setup, providers
from agent6.config import layer
from agent6.harness import _chain, _context, _loop_state, _reviewer, code_review
from agent6.ui.cli import review_cmds
from agent6.vcs import git_files, git_read, review
from tests.unit import test_review_gate, test_review_line_origin, test_review_saved
from tests.unit.test_review_diff_collect import _git


@pytest.mark.parametrize("policy", ["hidden-file", "hidden-directory", "unselected"])
@pytest.mark.parametrize("tracked", [False, True], ids=["untracked", "tracked"])
def test_review_never_writes_hidden_or_unselected_worktree_bytes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    policy: str,
    tracked: bool,
) -> None:
    """Filtering the diff must also keep excluded worktree bytes out of Git objects."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("old\n")
    private = tmp_path / "private"
    private.mkdir()
    secret = private / "secret.txt"
    if tracked:
        secret.write_text("old private content\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "a.py").write_text("new\n")
    secret.write_text("unpublished secret\n")
    blob = _git(tmp_path, "hash-object", str(secret)).strip()
    index = (tmp_path / ".git/index").read_bytes()
    cfg = test_review_saved._reviewer_config()
    hidden = (
        ()
        if policy == "unselected"
        else (str(private if policy.endswith("directory") else secret),)
    )
    cfg = cfg.model_copy(update={"sandbox": cfg.sandbox.model_copy(update={"hide_paths": hidden})})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(layer, "load_effective", mock.Mock(return_value=mock.Mock(config=cfg)))
    monkeypatch.setattr(_setup, "check_provider_keys", mock.Mock(return_value=None))
    monkeypatch.setattr(providers, "build_role_provider", mock.Mock())
    provider_call = mock.Mock(return_value="VERDICT: LGTM")
    monkeypatch.setattr(code_review, "review_diff", provider_call)
    assert (
        review_cmds._cmd_review(
            (), base="", head="HEAD", paths=("a.py",) if policy == "unselected" else ()
        )
        == 0
    )
    assert "unpublished secret" not in provider_call.call_args.kwargs["diff"]
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", blob], cwd=tmp_path, capture_output=True, check=False
        ).returncode
        != 0
    )
    (journal,) = (paths.state_dir(tmp_path) / "reviews").glob("*.jsonl")
    start = json.loads(journal.read_text().splitlines()[0])
    tree = start["target"]["head_tree"]
    stored = subprocess.run(
        ["git", "cat-file", "-p", f"{tree}:private/secret.txt"],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert stored.stdout == ("old private content\n" if tracked else "")
    assert (tmp_path / ".git/index").read_bytes() == index
    assert ("hidden paths" in capsys.readouterr().err) == (policy != "unselected")


@pytest.mark.parametrize("policy", ["hidden", "unselected"])
def test_staged_edits_outside_the_selection_take_the_base_entries(
    tmp_path: pathlib.Path, policy: str
) -> None:
    """A staged edit under a hidden or unselected path stayed in the recorded head tree."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("keep\nold\n")
    private = tmp_path / "private"
    private.mkdir()
    secret = private / "secret.txt"
    secret.write_text("base\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "a.py").write_text("keep\nnew\n")
    secret.write_text("STAGED-SECRET-4242\n")
    _git(tmp_path, "add", "private/secret.txt")
    hide = (str(private),) if policy == "hidden" else ()
    compared = review.comparison(
        tmp_path, base="", head="HEAD", paths=() if hide else ("a.py",), hide=hide
    )
    entry = "ls-tree", "-r", "{}", "private/secret.txt"
    assert _git(tmp_path, *(a.format(compared.target.head_tree) for a in entry)) == _git(
        tmp_path, *(a.format(compared.target.base_tree) for a in entry)
    )
    assert "+new" in compared.diff and "STAGED-SECRET" not in compared.diff
    origin = review.classify_line_origin(
        tmp_path,
        compared.target,
        review_records.ReviewLocation(path="a.py", side="head", start_line=1, end_line=1),
        hide=hide,
    )
    assert origin.kind == "pre_existing", origin


@pytest.mark.parametrize("policy", ["hidden", "unselected", "renamed"])
def test_staged_deletions_outside_the_selection_keep_the_base_entries(
    tmp_path: pathlib.Path, policy: str
) -> None:
    """A path removed from the index outside the selection had no entry to restore."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("keep\nold\n")
    private = tmp_path / "private"
    private.mkdir()
    (private / "secret.txt").write_text("BASE-SECRET\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    if policy == "renamed":
        _git(tmp_path, "mv", "a.py", "b.py")
        (tmp_path / "b.py").write_text("keep\nnew\n")
        paths, hide, restored = ("b.py",), (), "a.py"
    else:
        (tmp_path / "a.py").write_text("keep\nnew\n")
        _git(tmp_path, "rm", "-q", "--cached", "private/secret.txt")
        hide = (str(private),) if policy == "hidden" else ()
        paths, restored = (() if hide else ("a.py",)), "private/secret.txt"
    compared = review.comparison(tmp_path, base="", head="HEAD", paths=paths, hide=hide)
    entry = "ls-tree", "-r", "{}", restored
    assert _git(tmp_path, *(a.format(compared.target.head_tree) for a in entry)) == _git(
        tmp_path, *(a.format(compared.target.base_tree) for a in entry)
    )
    assert "+new" in compared.diff and "BASE-SECRET" not in compared.diff


def test_a_hidden_name_in_a_quoted_hunkless_entry_refuses_the_review(
    tmp_path: pathlib.Path,
) -> None:
    """A mode-only change to `ü.txt` has a quoted header and no hunk; its name reached the seat."""
    _git(tmp_path, "init", "-q")
    for name in ("a.py", "ü.txt"):
        (tmp_path / name).write_text("old\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "a.py").write_text("new\n")
    (tmp_path / "ü.txt").chmod(0o755)
    _git(tmp_path, "commit", "-qam", "head")
    with pytest.raises(errors.GitError, match="cannot enforce"):
        review.comparison(
            tmp_path, base="HEAD~1", head="HEAD", paths=(), hide=(str(tmp_path / "Ü.txt"),)
        )


def test_hidden_names_git_cannot_fold_never_reach_the_diff(tmp_path: pathlib.Path) -> None:
    """Python folds `Ü` onto `ü` and git's `icase` under `LC_ALL=C` does not.

    A worktree review stages by agent6's own filter, so the hidden file stays at its base entry;
    a revision review, whose diff git filters alone, refuses when a hidden name reaches it.
    """
    _git(tmp_path, "init", "-q")
    for name in ("a.py", "ü.txt"):
        (tmp_path / name).write_text("old\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    for name in ("a.py", "ü.txt"):
        (tmp_path / name).write_text("new\n")
    hide = (str(tmp_path / "Ü.txt"),)
    compared = review.comparison(tmp_path, base="", head="HEAD", paths=(), hide=hide)
    assert "+new" in compared.diff and "ü.txt" not in compared.diff
    assert _git(tmp_path, "ls-tree", compared.target.head_tree, "ü.txt") == _git(
        tmp_path, "ls-tree", compared.target.base_tree, "ü.txt"
    )
    _git(tmp_path, "commit", "-qam", "head")
    with pytest.raises(errors.GitError, match="cannot enforce"):
        review.comparison(tmp_path, base="HEAD~1", head="HEAD", paths=(), hide=hide)


def test_top_level_exclusion_survives_review_from_an_empty_subdirectory(
    tmp_path: pathlib.Path,
) -> None:
    """An exclusion must keep bytes out of the snapshot even when its cwd has no files."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("old\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "a.py").write_text("new\n")
    secret = tmp_path / "secret.txt"
    secret.write_text("unpublished secret\n")
    blob = _git(tmp_path, "hash-object", str(secret)).strip()
    subdir = tmp_path / "empty"
    subdir.mkdir()
    index = (tmp_path / ".git/index").read_bytes()

    compared = review.comparison(
        subdir, base="", head="HEAD", paths=(":/", ":(top,exclude)secret.txt")
    )

    assert "+new" in compared.diff and "secret" not in compared.diff
    blob_written = (
        subprocess.run(
            ["git", "cat-file", "-e", blob], cwd=tmp_path, capture_output=True, check=False
        ).returncode
        == 0
    )
    assert {
        "in_snapshot": bool(
            _git(tmp_path, "ls-tree", compared.target.head_tree, "--", "secret.txt")
        ),
        "blob_written": blob_written,
    } == {"in_snapshot": False, "blob_written": False}
    assert (tmp_path / ".git/index").read_bytes() == index


@pytest.mark.parametrize("linked", [False, True], ids=["main", "linked"])
def test_untracked_ignored_file_stays_deleted_without_writing_its_secret(
    tmp_path: pathlib.Path, linked: bool
) -> None:
    """An index deletion must not resurrect an ignored file or store its new bytes."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("old\n")
    (tmp_path / "private.txt").write_text("old private content\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    root = tmp_path
    if linked:
        root = tmp_path / "linked"
        _git(tmp_path, "worktree", "add", "--detach", str(root))
    _git(root, "rm", "--cached", "private.txt")
    (root / ".gitignore").write_text("private.txt\n")
    secret = root / "private.txt"
    secret.write_text("unpublished secret\n")
    (root / "a.py").write_text("new\n")
    blob = _git(root, "hash-object", str(secret)).strip()
    index_path = pathlib.Path(
        _git(root, "rev-parse", "--path-format=absolute", "--git-path", "index").strip()
    )
    index = index_path.read_bytes()

    compared = review.comparison(root, base="", head="HEAD", paths=())

    blob_written = (
        subprocess.run(
            ["git", "cat-file", "-e", blob], cwd=root, capture_output=True, check=False
        ).returncode
        == 0
    )
    assert {
        "secret_in_diff": "unpublished secret" in compared.diff,
        "blob_written": blob_written,
    } == {
        "secret_in_diff": False,
        "blob_written": False,
    }
    assert _git(root, "ls-tree", compared.target.head_tree, "--", "private.txt") == ""
    assert "deleted file mode" in compared.diff and "-old private content" in compared.diff
    assert "+new" in compared.diff
    assert index_path.read_bytes() == index
    assert secret.read_text() == "unpublished secret\n"


@pytest.mark.parametrize("policy", ["hidden", "unselected"])
def test_excluded_child_cannot_be_staged_through_a_tracked_ancestor(
    tmp_path: pathlib.Path, policy: str
) -> None:
    """A literal Git pathspec for a tracked file can also select its replacement directory."""
    _git(tmp_path, "init", "-q")
    parent = tmp_path / "private"
    parent.write_text("old file\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    parent.unlink()
    parent.mkdir()
    secret = parent / "secret"
    secret.write_text("unpublished secret\n")
    blob = _git(tmp_path, "hash-object", str(secret)).strip()
    compared = review.comparison(
        tmp_path,
        base="",
        head="HEAD",
        paths=("private", ":(exclude)private/secret") if policy == "unselected" else (),
        hide=(str(secret),) if policy == "hidden" else (),
    )
    assert "unpublished secret" not in compared.diff
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", blob], cwd=tmp_path, capture_output=True, check=False
        ).returncode
        != 0
    )


@pytest.mark.parametrize("tracked", [False, True], ids=["untracked", "tracked"])
@pytest.mark.parametrize("selected", [("a.py",), ()], ids=["selected-file", "all-paths"])
def test_hidden_ignored_paths_do_not_refuse_visible_changes(
    tmp_path: pathlib.Path, tracked: bool, selected: tuple[str, ...]
) -> None:
    """Hidden ignored files neither block visible changes nor enter the snapshot."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("old\n")
    secret = tmp_path / "secret.txt"
    if tracked:
        secret.write_text("old secret\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    if tracked:
        _git(tmp_path, "rm", "--cached", "secret.txt")
    (tmp_path / ".gitignore").write_text("secret.txt\n")
    (tmp_path / "a.py").write_text("new\n")
    secret.write_text("unpublished secret\n")
    blob = _git(tmp_path, "hash-object", str(secret)).strip()
    index = (tmp_path / ".git/index").read_bytes()
    compared = review.comparison(
        tmp_path, base="", head="HEAD", paths=selected, hide=(str(secret),)
    )
    assert "+new" in compared.diff
    assert all(name != "secret.txt" for name, _side in review.diff_hunks(compared.diff))
    assert "unpublished secret" not in compared.diff
    assert (tmp_path / ".git/index").read_bytes() == index
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", blob], cwd=tmp_path, capture_output=True, check=False
        ).returncode
        != 0
    )


@pytest.mark.parametrize("name", ["private", "[private]*?", "é"])
def test_root_exclusions_preserve_tracked_ignored_children(
    tmp_path: pathlib.Path, name: str
) -> None:
    """Root exclusions cover ignored tracked children and escape Git pattern characters."""
    _git(tmp_path, "init", "-q")
    private = tmp_path / name
    private.mkdir()
    secret = private / "child"
    secret.write_text("old private content\n")
    (tmp_path / "a.py").write_text("old\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    (tmp_path / ".gitignore").write_text("*\n")
    secret.write_text("unpublished secret\n")
    (tmp_path / "a.py").write_text("new\n")
    blob = _git(tmp_path, "hash-object", str(secret)).strip()
    index = (tmp_path / ".git/index").read_bytes()
    compared = review.comparison(tmp_path, base="", head="HEAD", paths=(), hide=(str(private),))
    assert "+new" in compared.diff and "private content" not in compared.diff
    assert "unpublished secret" not in compared.diff
    assert (
        _git(tmp_path, "show", f"{compared.target.head_tree}:{name}/child")
        == "old private content\n"
    )
    assert (tmp_path / ".git/index").read_bytes() == index
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", blob], cwd=tmp_path, capture_output=True, check=False
        ).returncode
        != 0
    )


def test_revision_review_excludes_hidden_deletions_and_modifications(
    tmp_path: pathlib.Path,
) -> None:
    """Hidden paths on either tree stay out of the diff and recorded coverage."""
    _git(tmp_path, "init", "-q")
    private = tmp_path / "private"
    private.mkdir()
    for name in ("a.py", "private/changed", "private/deleted"):
        (tmp_path / name).write_text("old\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "a.py").write_text("new\n")
    (private / "changed").write_text("secret\n")
    (private / "deleted").unlink()
    _git(tmp_path, "commit", "-qam", "head")
    compared = review.comparison(
        tmp_path, base="HEAD~1", head="HEAD", paths=(), hide=(str(private),)
    )
    assert compared.target.paths == ("a.py",)
    assert "private/" not in compared.diff
    assert compared.hidden_paths == ("private/changed", "private/deleted")


@pytest.mark.parametrize("side", ["base", "head"])
@pytest.mark.parametrize("missing_blob", [False, True], ids=["present", "unreadable"])
def test_line_origin_refuses_hidden_rename_evidence(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, side: str, missing_blob: bool
) -> None:
    """Hidden rename candidates must not be read or recorded as line-origin evidence."""
    _git(tmp_path, "init", "-q")
    test_review_line_origin._commit(
        tmp_path, b"first unchanged line\nsecond unchanged line\n", "old.txt"
    )
    _git(tmp_path, "mv", "old.txt", "new.txt")
    (tmp_path / "new.txt").write_text("first unchanged line\nsecond unchanged line\nadded\n")
    _git(tmp_path, "commit", "-qam", "rename")
    assert _git(tmp_path, "diff", "-M", "--name-status", "HEAD~1", "HEAD").startswith("R")
    hidden = "old.txt" if side == "head" else "new.txt"
    visible = "new.txt" if side == "head" else "old.txt"
    hide = (str(tmp_path / hidden),)
    compared = review.comparison(tmp_path, base="HEAD~1", head="HEAD", paths=(visible,), hide=hide)
    if missing_blob:
        tree = compared.target.base_tree if hidden == "old.txt" else compared.target.head_tree
        blob = _git(tmp_path, "rev-parse", f"{tree}:{hidden}").strip()
        (tmp_path / ".git/objects" / blob[:2] / blob[2:]).unlink()
    blobs = mock.Mock(wraps=git_files.blob_at)
    monkeypatch.setattr(git_files, "blob_at", blobs)

    origin = review.classify_line_origin(
        tmp_path,
        compared.target,
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
    _git(tmp_path, "init", "-q")
    nested = tmp_path / "sub"
    nested.mkdir()
    old_name, new_name = "sub/secret.py", "sub/new.py"
    base = test_review_line_origin._commit(
        tmp_path, b"first unchanged line\nsecond unchanged line\n", old_name
    )
    monkeypatch.chdir(nested)
    chain = _chain.RunChain(
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
    reviewer = _reviewer.Reviewer(
        settings=_reviewer.ReviewSettings(
            reviewers=[_reviewer.ReviewSeat("security", "model", provider)], block_on="any"
        ),
        chain=chain,
        hide=hide,
        review_tools=lambda: ([], mock.MagicMock()),
        budget_remaining=lambda: None,
        log=lambda _s: None,
        emit=sink.emit,
    )
    state = _loop_state.LoopState(original_task="t", tool_calls=0)

    result = reviewer.critique(state, trigger="before_finish", turn=1)

    assert result is not None and not result.satisfied
    rows = [json.loads(line) for line in sink.path.read_text().splitlines()]
    panels = [
        review_records.ReviewPanel.model_validate(event_log.payload(row))
        for row in rows
        if row["type"] == "loop.review.panel"
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


@pytest.mark.parametrize("source", ["config", "shallow", "rules"])
def test_review_reasons_do_not_quote_hidden_git_inputs(tmp_path: pathlib.Path, source: str) -> None:
    """Git diagnostics can quote hidden config values or invalid shallow-history lines."""
    _git(tmp_path, "init", "-q")
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
        _git(tmp_path, "config", "include.path", "../secret.env")
    if source == "rules":
        rules = _context.review_rules(tmp_path, target.base_tree, (str(secret),))
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
    _git(tmp_path, "init", "-q")
    base = test_review_line_origin._commit(tmp_path, b"old\ncontext\n")
    secret = tmp_path / "secret.env"
    hide = (str(secret),) if refusal != "ordinary" else ()
    chain = _chain.RunChain(
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
        _git(tmp_path, "config", "blame.ignoreRevsFile", "secret.env")
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
        _git(tmp_path, "config", "--file", str(secret), "blame.ignoreRevsFile", value)
        _git(tmp_path, "config", "include.path", "../secret.env")
    opened = mock.Mock(wraps=git_files._open_regular)
    monkeypatch.setattr(git_files, "_open_regular", opened)
    # The config files above are the operator's: untracked when the run started.
    chain = dataclasses.replace(chain, untracked_at_start=git_read.untracked_paths(tmp_path))
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
        hide=hide,
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
    assert not any(row["type"] == "loop.review.line_origin" for row in rows)
    assert (tmp_path / ".git" / "index").read_bytes() == index
    assert _git(tmp_path, "show-ref") == refs
    assert (tmp_path / "code.py").read_text() == "workspace\nnot context\n"
