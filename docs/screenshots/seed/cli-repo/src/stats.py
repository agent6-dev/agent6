"""Small statistics helpers."""


def mean(xs):
    """Arithmetic mean of a non-empty list of numbers."""
    return sum(xs) / len(xs)


def median(xs):
    """Median of a non-empty list of numbers."""
    s = sorted(xs)
    n = len(s)
    return s[n // 2]
