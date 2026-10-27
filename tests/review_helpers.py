from __future__ import annotations

import pathlib

from agent6 import event_log, paths, review_records
from agent6.sessions import layout


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
