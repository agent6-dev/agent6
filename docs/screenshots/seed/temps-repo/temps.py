"""Temperature conversions."""


def c_to_f(c):
    return c * 9 / 5 + 32


def f_to_c(f):
    return f - 32 * 5 / 9


def kelvin_to_c(k):
    return k - 273.15
