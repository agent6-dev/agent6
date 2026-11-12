from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import tempfile
import time

from agent6 import paths
from agent6.sessions import manifest
from agent6.ui.cli import common
from agent6.viewmodel import listing


def read_chars() -> int:
    return int(pathlib.Path("/proc/self/io").read_text().split("rchar: ")[1].splitlines()[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure session ordering over synthetic journals."
    )
    parser.add_argument("--sessions", type=int, default=100)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="session-ordering-", dir=pathlib.Path.cwd()) as temp:
        root = pathlib.Path(temp)
        repo = root / "repo"
        repo.mkdir()
        os.environ["XDG_STATE_HOME"] = str(root / "state")
        state = paths.state_dir(repo)
        event = {"ts": "2026-01-01T00:00:00+00:00", "type": "model.text_delta", "text": ""}
        event["text"] = "x" * (111 - len(json.dumps(event, separators=(",", ":"))))
        line = (json.dumps(event, separators=(",", ":")) + "\n").encode()
        assert len(line) == 112
        body = line * args.events
        total = 0
        for i in range(args.sessions):
            session = state / "sessions" / "runs" / f"s{i:04}"
            session.mkdir(parents=True)
            (session / "manifest.json").write_text(
                manifest.SessionManifest(session_id=session.name).model_dump_json()
            )
            log = session / "logs.jsonl"
            log.write_bytes(
                body + (json.dumps({"type": "session.stop", "ts": 1800000000 + i}) + "\n").encode()
            )
            os.utime(log, (2000000000 - i, 2000000000 - i))
            total += log.stat().st_size
        print(
            json.dumps(
                {
                    "sessions": args.sessions,
                    "events_per_session": args.events + 1,
                    "journal_bytes": total,
                }
            )
        )
        for operation in ("newest", "ordering"):
            samples = []
            for _ in range(args.repeats):
                before = read_chars()
                start = time.perf_counter()
                if operation == "newest":
                    found = common.resolve_or_newest_layout(repo, "")
                    assert found is not None
                    newest = found.session_id
                else:
                    newest = listing.session_dirs(state)[0].name
                elapsed = time.perf_counter() - start
                consumed = read_chars() - before
                assert newest == f"s{args.sessions - 1:04}"
                samples.append(elapsed)
                print(
                    json.dumps({"operation": operation, "seconds": elapsed, "read_chars": consumed})
                )
            print(
                json.dumps(
                    {
                        "operation": operation,
                        "median_s": statistics.median(samples),
                        "stdev_s": statistics.stdev(samples),
                    }
                )
            )


if __name__ == "__main__":
    main()
