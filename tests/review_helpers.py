from __future__ import annotations

import dataclasses
import pathlib
import subprocess

import pydantic

from agent6 import event_log, paths, review_records
from agent6.sessions import layout
from agent6.vcs import review


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
        trigger="before_finish",
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
    sink.emit("loop.review.panel", **panel.model_dump(mode="json"))
    return saved


def reviewed_repo(cwd: pathlib.Path) -> layout.SessionLayout:
    """Record a real rename, deletion and file-level finding over immutable Git trees."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-qb", "main")
    (cwd / "old.txt").write_text("first\nsecond\nthird\nfourth\n")
    (cwd / "gone.txt").write_text("deleted\n")
    (cwd / "context.txt").write_text("unchanged context\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    git("mv", "old.txt", "new.txt")
    git("rm", "gone.txt")
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
            line_origin=review.classify_line_origin(cwd, target, observation.finding.location),
        )
        for observation in findings
    )
    sink = event_log.EventSink(saved.logs_path)
    sink.emit("session.start", session_id=saved.session_id, mode="run", task="Review parser")
    sink.emit(
        "loop.review.panel",
        panel_id="panel",
        execution_id="execution",
        turn=1,
        trigger="before_finish",
        target=dataclasses.asdict(target),
        findings=pydantic.TypeAdapter(tuple[review_records.FindingObservation, ...]).dump_python(
            findings, mode="json"
        ),
        injected_finding_ids=[f.finding_id for f in findings],
        block_on="none",
        raw_blocked=False,
        blocked=False,
        disarmed=False,
        n_block=1,
        n_abstain=0,
        completion="complete",
    )
    sink.emit("session.end", reason="finish_session", turns=1)
    return saved
