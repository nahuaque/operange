"""Severity objectives over named physical coordinates."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, runtime_checkable

from .contract_types import Record
from .domains import ParameterSpace
from .primitives import finite


@runtime_checkable
class Distance(Protocol):
    @property
    def space(self) -> ParameterSpace: ...

    def measure(self, realization, baseline=None) -> float: ...

    def to_manifest(self) -> dict: ...


@dataclass(frozen=True)
class NormalizedLInf(Record):
    """Maximum absolute scaled deviation. Baselines need not be domain members."""

    space: ParameterSpace

    def _validate(self):
        self.space.normalize(self.space.nominal)

    def measure(self, realization, baseline=None):
        values = self.space.validate(realization)
        origin = (
            self.space.nominal if baseline is None else self.space.validate(baseline)
        )
        return finite(
            max(
                abs(values[c.name] - origin[c.name]) / c.scale
                for c in self.space.coordinates
            ),
            "distance",
        )

    def to_manifest(self):
        return {
            "metric": "normalized_linf",
            "coordinates": list(self.space.names),
            "units": [c.unit for c in self.space.coordinates],
            "origins": [c.nominal for c in self.space.coordinates],
            "scales": [c.scale for c in self.space.coordinates],
            "unit": "1",
        }


@dataclass(frozen=True)
class NormalizedL2(NormalizedLInf):
    """Euclidean norm of explicitly scaled deviations, rounded outward.

    Scales and origins belong to this metric, independently of the domain.
    The returned float bounds the exact Euclidean distance from above.
    """

    def measure(self, realization, baseline=None):
        from ._numeric import sqrt_upper

        values = self.space.validate(realization)
        origin = (
            self.space.nominal if baseline is None else self.space.validate(baseline)
        )
        return sqrt_upper(
            sum(
                (
                    (Fraction(values[c.name]) - Fraction(origin[c.name]))
                    / Fraction(c.scale)
                )
                ** 2
                for c in self.space.coordinates
            )
        )

    def to_manifest(self):
        return {**super().to_manifest(), "metric": "normalized_l2"}
