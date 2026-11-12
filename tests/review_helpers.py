from __future__ import annotations

import dataclasses
import pathlib
import shutil
import subprocess
from typing import Any

import pydantic

from agent6 import paths, review_records
from agent6.sessions import event_log, journal, layout
from agent6.vcs import review


def git(root: pathlib.Path, *args: str) -> str:
    """Run Git in a test repository and return its stdout."""
    git = shutil.which("git")
    assert git is not None
    out = subprocess.run(
        [git, "-c", "commit.gpgsign=false", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def reviewed_session(cwd: pathlib.Path) -> layout.SessionLayout:
    """Create a session with two grounded findings in its real state directory."""
    saved = layout.SessionLayout(paths.state_dir(cwd), "review-run")
    saved.ensure()
    target = review_records.ReviewTarget("base", "bt", "head", "ht", "head", (), "revisions")
    findings = tuple(
        review_records.FindingObservation(
            name,
            "panel",
            review_records.Finding(
                category="correctness",
                severity="block",
                title=f"Defect {name}",
                detail="Reproduce with an empty input.",
                location=review_records.ReviewLocation(
                    path="parser.py", side="head", start_line=3, end_line=4
                ),
            ),
            (0,),
        )
        for name in ("f1", "f2")
    )
    panel = review_records.ReviewPanel(
        panel_id="panel",
        execution_id="execution",
        turn=1,
        trigger="finish",
        target=target,
        findings=findings,
        injected_finding_ids=("f1", "f2"),
        block_on="none",
        raw_blocked=False,
        blocked=False,
        disarmed=False,
        n_block=1,
        n_abstain=0,
        completion="complete",
    )
    sink = event_log.EventSink(saved.logs_path)
    sink.emit(journal.ReviewPanel(**panel.model_dump(mode="json")))
    return saved


def reviewed_repo(cwd: pathlib.Path) -> layout.SessionLayout:
    """Record a real rename, deletion and file-level finding over immutable Git trees."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-qb", "main")
    (cwd / "AGENTS.md").write_text("Keep the base rule. <b>literal</b>\n")
    (cwd / "old.txt").write_text("first\nsecond\nthird\nfourth\n")
    (cwd / "gone.txt").write_text("deleted\n")
    (cwd / "context.txt").write_text("unchanged context\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    git("mv", "old.txt", "new.txt")
    git("rm", "gone.txt", "AGENTS.md")
    (cwd / "new.txt").write_text("first\nsecond\nthird\nfourth\nadded\n")
    git("add", ".")
    git("commit", "-qm", "head")
    head = git("rev-parse", "HEAD")
    target = review_records.ReviewTarget(
        base,
        git("rev-parse", base + "^{tree}"),
        head,
        git("rev-parse", head + "^{tree}"),
        head,
        (),
        "revisions",
    )
    saved = reviewed_session(cwd)
    saved.logs_path.unlink()
    findings = tuple(
        review_records.FindingObservation(
            name,
            "panel",
            review_records.Finding(
                category="correctness",
                severity="block",
                title="<img src=x onerror=alert(1)>",
                detail="<script>hostile()</script>",
                location=review_records.ReviewLocation.model_validate(
                    dict(path=path, side=side, start_line=line, end_line=line)
                ),
            ),
            (0,),
        )
        for name, path, side, line in (
            ("rename", "old.txt", "base", 2),
            ("delete", "gone.txt", "base", 1),
            ("head", "new.txt", "head", 5),
            ("file", "context.txt", "head", None),
        )
    )
    findings = tuple(
        dataclasses.replace(
            observation,
            line_origin=review.classify_line_origin(
                cwd, target, observation.finding.location, hide=()
            ),
        )
        for observation in findings
    )
    sink = event_log.EventSink(saved.logs_path)
    sink.emit(
        journal.SessionStart(
            session_id=saved.session_id,
            mode="run",
            user_task="Review parser",
            execution_id="execution",
        )
    )
    sink.emit(
        journal.ReviewStart(
            panel_id="panel",
            execution_id="execution",
            turn=1,
            trigger="finish",
            target=dataclasses.asdict(target),
            seats=[],
            rules_complete=True,
            rules_reason=None,
            rule_sources=[
                dict(
                    path="AGENTS.md",
                    blob_id=git("rev-parse", base + ":AGENTS.md"),
                    supplied_bytes=35,
                )
            ],
        )
    )
    sink.emit(
        journal.ReviewPanel(
            panel_id="panel",
            execution_id="execution",
            turn=1,
            trigger="finish",
            target=dataclasses.asdict(target),
            findings=pydantic.TypeAdapter(
                tuple[review_records.FindingObservation, ...]
            ).dump_python(findings, mode="json"),
            injected_finding_ids=[f.finding_id for f in findings],
            block_on="none",
            raw_blocked=False,
            blocked=False,
            disarmed=False,
            n_block=1,
            n_abstain=0,
            completion="complete",
        )
    )
    sink.emit(
        journal.SessionStop(
            execution_id="execution",
            reason="finish_session",
            turns=1,
            verification="passed",
        )
    )
    return saved


def panel_rows(
    *, block: int = 0, warn: int = 0, abstain: int = 0, head: str = "ht"
) -> list[dict[str, Any]]:
    """Return a complete panel's start and completion rows with the given open findings."""
    target = review_records.ReviewTarget("b", "bt", "h", head, "h", (), "revisions")
    observations = []
    for index, severity in enumerate(["block"] * block + ["warn"] * warn):
        finding = review_records.Finding(
            category="correctness",
            severity=severity,  # type: ignore[arg-type]
            title=f"f{index}",
            location=review_records.ReviewLocation(
                path="a.py", side="head", start_line=1, end_line=1
            ),
        )
        observations.append(review_records.FindingObservation(f"f{index}", "p", finding, (0,)))
    record = review_records.ReviewPanel(
        panel_id="p",
        execution_id="e",
        turn=1,
        trigger="finish",
        target=target,
        findings=tuple(observations),
        injected_finding_ids=(),
        block_on="any",
        raw_blocked=block > 0,
        blocked=block > 0,
        disarmed=False,
        n_block=block,
        n_abstain=abstain,
        completion="complete",
    )
    return [
        {
            "type": "review.start",
            "ts": "2026-10-25T10:00:00+00:00",
            "panel_id": "p",
            "execution_id": "e",
            "target": dataclasses.asdict(target),
            "seats": [],
        },
        {
            "type": "review.panel",
            "ts": "2026-10-25T10:00:05+00:00",
            **record.model_dump(mode="json"),
        },
    ]


def verify_rows(
    *, exit_code: int = 0, scoped: bool = False, tree: str = "ht"
) -> list[dict[str, Any]]:
    """Return a verification attempt that observed `tree` on both sides."""
    return [
        {"type": "verify.start", "verify_id": "v", "command": ["pytest"], "tree_before": tree},
        {
            "type": "verify.end",
            "verify_id": "v",
            "command": ["pytest"],
            "exit_code": exit_code,
            "tree_before": tree,
            "tree_after": tree,
            "scoped": scoped,
            "timeout_s": 5.0 if exit_code == 124 else 0.0,
            "timed_out": exit_code == 124,
        },
    ]


START: dict[str, Any] = {
    "type": "session.start",
    "ts": "2026-10-25T09:00:00+00:00",
    "mode": "run",
    "user_task": "t",
}
END: dict[str, Any] = {
    "type": "session.stop",
    "ts": "2026-10-25T11:00:00+00:00",
    "reason": "finish_session",
    "verification": "passed",
    "tree": "ht",
    "final_verify_id": "v",
}
