"""Independent closed-form checks for this specific constant-COP model.

These functions do not call an optimizer or consume compiled LP matrices.
They are not a general process-model certification backend.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence


def capacity(cop: float, source: float, power: float, derating: float) -> float:
    return min(cop * power, cop / (cop - 1) * source * (1 - derating))


def affine_threshold_radius(
    nominal_slack: float,
    margin: float,
    slopes: Sequence[float],
    scales: Sequence[float],
    headrooms: Sequence[float],
) -> float | None:
    """First radius where a decreasing affine slack reaches ``-margin``.

    At radius r, coordinate i can increase by min(scale[i]*r, headroom[i]).
    This yields a piecewise-linear envelope with finitely many breakpoints.
    Fraction arithmetic avoids cancellation in the breakpoint comparisons;
    it represents the supplied floating coefficients, not exact plant physics.
    """

    def rational(value: float) -> Fraction:
        return Fraction(str(value))

    needed = rational(nominal_slack) + rational(margin)
    if needed <= 0:
        return 0.0
    terms = [
        (rational(slope), rational(scale), rational(room))
        for slope, scale, room in zip(slopes, scales, headrooms)
        if slope > 0 and room > 0
    ]
    previous = Fraction(0)
    gain = Fraction(0)
    for endpoint in sorted({room / scale for _, scale, room in terms}):
        rate = sum(
            (slope * scale for slope, scale, room in terms if room / scale > previous),
            Fraction(0),
        )
        next_gain = gain + rate * (endpoint - previous)
        if next_gain >= needed:
            return float(previous + (needed - gain) / rate)
        previous, gain = endpoint, next_gain
    return None
