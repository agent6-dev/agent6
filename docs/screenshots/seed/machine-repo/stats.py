# A small stats helper with one deliberate bug for the code-fixer machine to
# find and fix: median is wrong for even-length inputs (it returns the upper of
# the two middle elements instead of their average).


def median(xs):
    ordered = sorted(xs)
    n = len(ordered)
    return ordered[n // 2]


def mean(xs):
    return sum(xs) / len(xs)
