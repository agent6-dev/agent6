from __future__ import annotations

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _doc(path: str, name: str = "") -> str:
    tree = ast.parse((_ROOT / "src/agent6" / path).read_text(encoding="utf-8"))
    node = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        tree,
    )
    if name:
        assert isinstance(node, ast.FunctionDef), name
    return " ".join((ast.get_docstring(node) or "").split())


@pytest.mark.parametrize("name", ["_credential_step", "_cmd_connect"])
def test_connect_flow_docstrings_include_chatgpt(name: str) -> None:
    assert "flow:" in _doc("ui/cli/connect.py", name)
    assert "ChatGPT or a configured issuer" in _doc("ui/cli/connect.py", name)


@pytest.mark.parametrize("name", ["_sign_in", "_credential_step", "_cmd_connect"])
def test_connect_verification_docstrings_exclude_chatgpt(name: str) -> None:
    assert "configured-issuer grant" in _doc("ui/cli/connect.py", name)


def test_pruning_docstring_names_unreachable_commit_recovery() -> None:
    text = _doc("ui/cli/sessions_merge.py", "_prune_branch")
    assert "tip sha, restorable until gc prunes the unreachable commit" in text
    assert "reflog" not in text


def test_btw_docstrings_describe_detached_launch_without_namespace_escape() -> None:
    assert "spawning it detached" in _doc("ui/btw.py")
    assert "It spawns as a detached `agent6` process." in _doc("app/btw.py")
    assert "namespace" not in _doc("ui/btw.py", "direct_launch")
    assert "unlike a confined lane" not in (_ROOT / "src/agent6/ui/cli/run.py").read_text()


def test_save_review_owns_rendered_reviews_not_every_file() -> None:
    text = _doc("ui/cli/sessions_review.py")
    assert "`save_review` writes the rendered reviews" in text
    assert "one writer" not in text


def test_review_lookup_docstring_names_the_operator() -> None:
    assert "The operator finds a session's review" in _doc(
        "ui/cli/sessions_review.py", "save_review"
    )


def test_web_docstring_distinguishes_cli_and_config_bind_opt_ins() -> None:
    text = _doc("ui/cli/web_cmds.py", "_cmd_web")
    assert "A non-loopback `[web].host` needs `[web].allow_non_loopback = true`" in text
    assert "`--allow-non-loopback` opts in a non-loopback `--host`" in text


def test_key_prompt_docstring_describes_terminal_masking() -> None:
    text = _doc("ui/cli/connect.py", "_prompt_api_key")
    assert "On a terminal each keystroke shows as `*`" in text
    assert "off one, a summary prints the key's length" in text
    assert "Python 3.14" not in text


def test_resume_blocker_docstring_does_not_promise_resume_will_start() -> None:
    text = _doc("viewmodel/listing.py", "resume_blocker")
    assert "session's state cannot be resumed at all" in text
    assert "no such state blocker applies" in text
    assert "would work" not in text
    assert "can start" not in text


def test_setup_docstring_only_promises_to_skip_a_successful_setup() -> None:
    text = _doc("app/workspace_setup.py", "session_needs_setup")
    assert "setup that left the workspace ready" in text
    assert "a resume after one skips the command" in text


def test_steering_docstring_includes_turn_boundaries() -> None:
    assert "Apply steering at turn and execution boundaries." in _doc("harness/loop_steer.py")


def test_lane_import_docstring_does_not_invent_a_failed_starts_symlink() -> None:
    text = _doc("app/parallel_results.py", "import_lanes")
    assert "keeps what it created" in text
    assert "still running or whose import was refused" in text


def test_graph_docstring_names_fork_locking_and_full_isolation() -> None:
    text = _doc("graph/__init__.py")
    assert "per-mutation flock also serialises fork's copy" in text
    assert "the jail masks under `full`" in text
    assert "operator-CLI" not in text
