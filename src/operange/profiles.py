"""Supplied continuous-time profiles with explicit interpolation and tails."""

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction

from .contract_types import Record, nonempty
from .primitives import finite


@dataclass(frozen=True)
class PiecewiseLinearProfile(Record):
    """Values at elapsed seconds, linear between knots, zero before time zero.

    The value at zero is the right-hand value (an initial jump is allowed).
    After the last knot the final value is held. These are declared equations,
    not an assertion that an underlying measured waveform has no other peaks.
    """

    times_seconds: tuple[float, ...]
    values: tuple[float, ...]
    unit: str
    provenance: str

    def _validate(self):
        nonempty(self.unit, "profile unit")
        nonempty(self.provenance, "profile provenance")
        if (
            len(self.times_seconds) < 2
            or len(self.values) != len(self.times_seconds)
            or self.times_seconds[0] != 0
            or any(b <= a for a, b in zip(self.times_seconds, self.times_seconds[1:]))
        ):
            raise ValueError(
                "profile needs matching values and increasing times starting at zero"
            )

    def _at(self, time: Fraction, *, left=False) -> Fraction:
        """Exact interpolation of the declared binary floating-point inputs."""
        if time < 0 or (time == 0 and left):
            return Fraction(0)
        times = tuple(Fraction(t) for t in self.times_seconds)
        if time >= times[-1]:
            return Fraction(self.values[-1])
        i = bisect_right(times, time) - 1
        a, b = Fraction(self.values[i]), Fraction(self.values[i + 1])
        return a + (b - a) * (time - times[i]) / (times[i + 1] - times[i])

    def value_at(self, elapsed_seconds: float) -> float:
        return finite(
            float(self._at(Fraction(finite(elapsed_seconds, "elapsed time")))),
            "profile value",
        )
