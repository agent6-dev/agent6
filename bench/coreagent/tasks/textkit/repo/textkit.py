"""textkit — small text-processing library. See spec.md.

Implement each function. Stub bodies raise NotImplementedError so the test
suite fails until you do.
"""

from __future__ import annotations


def normalize_whitespace(s: str) -> str:
    raise NotImplementedError


def word_count(s: str) -> int:
    raise NotImplementedError


def most_common_words(s: str, n: int) -> list[tuple[str, int]]:
    raise NotImplementedError


def wrap_text(s: str, width: int) -> list[str]:
    raise NotImplementedError


def to_snake_case(name: str) -> str:
    raise NotImplementedError
