"""Repair torn final lines without rewriting committed journal bytes."""

from __future__ import annotations

import logging
import os
import pathlib


def heal_torn_tail(path: pathlib.Path) -> None:
    """Truncate a journal's unfinished last line in place and log the bytes removed.

    The caller holds its writer lock across this repair and the following append.

    Args:
        path: The journal file; a missing file needs no repair.
    """
    try:
        fh = path.open("r+b")
    except FileNotFoundError:
        return
    with fh:
        size = fh.seek(0, os.SEEK_END)
        if not size:
            return
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) == b"\n":
            return
        end = size
        while end:
            start = max(0, end - (1 << 20))
            fh.seek(start)
            cut = fh.read(end - start).rfind(b"\n")
            if cut >= 0:
                end = start + cut + 1
                break
            end = start
        # Truncating in place keeps committed bytes visible to readers holding the file open.
        fh.truncate(end)
    logging.getLogger(__name__).warning(
        "Dropped %d bytes from torn journal tail at %s", size - end, path
    )
