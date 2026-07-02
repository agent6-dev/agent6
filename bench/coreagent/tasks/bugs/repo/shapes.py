"""shapes -- small numeric helpers. See spec.md.

Seven independent pure functions over numbers and lists. Standard library only.
Each is meant to match the spec, but the suite (./verify.sh) is failing.
"""

from __future__ import annotations


def clamp(x: float, lo: float, hi: float) -> float:
    """Return x constrained to the inclusive range [lo, hi]."""
    if x < lo:
        return hi
    if x > hi:
        return hi
    return x


def mean(xs: list[float]) -> float:
    """Arithmetic mean of xs. The empty list has mean 0.0."""
    if not xs:
        return 0.0
    return sum(xs) // len(xs)


def median(xs: list[float]) -> float:
    """Median of xs. Even-length lists average the two middle values.

    The empty list has median 0.0.
    """
    if not xs:
        return 0.0
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2


def gcd(a: int, b: int) -> int:
    """Greatest common divisor of a and b, always non-negative. gcd(0, 0) is 0."""
    while b:
        a, b = b, a % b
    return a


def is_prime(n: int) -> bool:
    """Return True if n is a prime number. Numbers below 2 are not prime."""
    if n < 2:
        return False
    i = 2
    while i * i < n:
        if n % i == 0:
            return False
        i += 1
    return True


def roman(n: int) -> str:
    """Roman numeral for an integer in the range 1..3999."""
    table = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out: list[str] = []
    for value, sym in table:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


def running_max(xs: list[int]) -> list[int]:
    """Return a list whose element i is max(xs[0..i]). The empty list maps to []."""
    out: list[int] = []
    cur = 0
    for x in xs:
        cur = max(cur, x)
        out.append(cur)
    return out
