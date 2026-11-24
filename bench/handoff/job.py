"""The six independent reports in dispatch order."""

from __future__ import annotations

FILES = ("ash.txt", "beech.txt", "cedar.txt", "elm.txt", "fir.txt", "oak.txt")
TITLES = tuple(f"Report the code in {name}." for name in FILES)
CHILDREN = 2
PIN = f"Parent: keep {CHILDREN} children running on distinct listed tasks while work remains."
INSTRUCTION = {
    "text": "Copy each listed title verbatim into parallel's task argument. Do not name a task id in the dispatch.",
    "id": "Give parallel the listed task_id. Do not copy the task text into the dispatch.",
}


def prompt(arm: str) -> str:
    """Return the parent brief with only the dispatch instruction varied."""
    return (
        "Collect the six file reports. List the queued tasks. Dispatch one child per task in "
        f"list order using the default worker. {INSTRUCTION[arm]} "
        "Leave the files unchanged. Wait for each report. Mark its original task passed "
        "if it still needs acceptance. Finish with the six codes."
    )
