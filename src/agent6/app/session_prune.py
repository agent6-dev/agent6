"""Prune aged session artifacts under the same lock that excludes concurrent resumes."""

from __future__ import annotations

import datetime
import pathlib
import shutil

from agent6 import portable
from agent6.app import reporter as app_reporter
from agent6.sessions import layout, lock, process, retention
from agent6.viewmodel import session_log


def prune_artifacts(
    state_dir: pathlib.Path,
    older_than_days: int,
    *,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> int:
    """Remove retained artifacts of executions that ended before the age cutoff.

    Args:
        state_dir: The repository's state directory.
        older_than_days: The nonnegative age in days.
        reporter: Receives one line per pruned session and any refusal.

    Returns:
        Zero on success, two when an age or filesystem operation is refused.
    """
    if older_than_days < 0:
        reporter.error("--older-than must be a nonnegative number of days")
        return 2
    now = datetime.datetime.now(datetime.UTC)
    cutoff = now.timestamp() - older_than_days * 86400
    failed = False
    for bucket in layout.SESSION_BUCKETS:
        for session_dir in sorted(layout.bucket_dir(state_dir, bucket).glob("*")):
            if not session_dir.is_dir() or session_dir.is_symlink():
                continue
            fd = lock.acquire_single_writer(session_dir)
            if fd is None:
                continue
            try:
                scan = session_log.scan_session_log(session_dir / layout.LOGS_NAME)
                if (
                    process.worker_is_alive(session_dir)
                    or not scan.finished
                    or scan.end_ep is None
                    or scan.end_ep >= cutoff
                    or scan.evidence_notes
                ):
                    continue
                directories = ("transcripts", "checkpoints", "objects")
                if not any((session_dir / name).exists() for name in directories):
                    continue
                record = retention.Pruned(
                    pruned_at=now,
                    counts={
                        name: sum(p.is_file() for p in (session_dir / name).rglob("*"))
                        for name in directories
                    },
                    last_transcript_seq=retention.last_sequence(session_dir / "transcripts"),
                )
                # Persist the watermark before deletion so an interrupted prune cannot reuse seqs.
                portable.atomic_write(session_dir / "pruned.json", record.model_dump_json(indent=2))
                for name in directories:
                    directory = session_dir / name
                    if directory.exists():
                        shutil.rmtree(directory)
                counts = ", ".join(f"{name}/ ({record.counts[name]} files)" for name in directories)
                reporter.out(f"[agent6] {session_dir.name}: removed {counts}")
            except (OSError, retention.RetentionError) as exc:
                failed = True
                reporter.error(f"could not prune {session_dir.name}: {exc}")
            finally:
                lock.release_single_writer(fd)
    return 2 if failed else 0
