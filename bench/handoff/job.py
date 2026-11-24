"""The six independent reports in dispatch order."""

from __future__ import annotations

FILES = ("ash.txt", "beech.txt", "cedar.txt", "elm.txt", "fir.txt", "oak.txt")
TITLES = tuple(f"Report the code in {name}." for name in FILES)
CHILDREN = 2
