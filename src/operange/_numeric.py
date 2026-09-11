"""Directed bounds for arithmetic on the exact values of declared floats."""

from decimal import Context, Decimal, localcontext
from fractions import Fraction
from math import inf, nextafter

from .primitives import finite


def rounded(value, *, upper):
    value = Fraction(value)
    number = finite(float(value), "arithmetic bound")
    needs_rounding = Fraction(number) < value if upper else Fraction(number) > value
    if needs_rounding:
        number = nextafter(number, inf if upper else -inf)
    return finite(number, "arithmetic bound")


def round_up(value):
    return rounded(value, upper=True)


def round_down(value):
    return rounded(value, upper=False)


def exact_dot(a, b):
    return sum((Fraction(x) * Fraction(y) for x, y in zip(a, b)), Fraction(0))


def normalized_score(space, coefficients, point):
    return sum(
        (
            Fraction(coefficients.get(c.name, 0))
            * (Fraction(point[c.name]) - Fraction(c.nominal))
            / Fraction(c.scale)
            for c in space.coordinates
        ),
        Fraction(0),
    )


def exact_normalized_bounds(box):
    return tuple(
        (
            (Fraction(p.lower) - Fraction(p.nominal)) / Fraction(p.scale),
            (Fraction(p.upper) - Fraction(p.nominal)) / Fraction(p.scale),
        )
        for p in box.scalar_parameters
    )


def sqrt_upper(value):
    """A finite upper bound, checked exactly after an approximate square root."""
    value = Fraction(value)
    if value < 0:
        raise ValueError("square root requires a nonnegative value")
    with localcontext(Context(prec=80)):
        candidate = finite(
            float((Decimal(value.numerator) / Decimal(value.denominator)).sqrt()),
            "square root bound",
        )
    while Fraction(candidate) ** 2 < value:
        candidate = finite(nextafter(candidate, inf), "square root bound")
    return candidate
