"""Small immutable contracts; no symbolic modeling framework is required."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from .vector import VectorParameter


def finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def nonnegative(value: float, name: str) -> float:
    value = finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class Parameter:
    """A bounded scalar with an explicit engineering distance scale.

    Units are labels, not a conversion engine. A model adapter must validate
    the units it supports. ``scale`` has the same units as the parameter.
    """

    name: str
    unit: str
    nominal: float
    lower: float
    upper: float
    scale: float
    provenance: str

    def __post_init__(self) -> None:
        for name in ("name", "unit", "provenance"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("nominal", "lower", "upper", "scale"):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if not self.lower <= self.nominal <= self.upper:
            raise ValueError("bounds must be ordered and contain the nominal value")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        finite((self.upper - self.nominal) / self.scale, "upper normalized excursion")
        finite((self.nominal - self.lower) / self.scale, "lower normalized excursion")

    @property
    def coordinate(self):
        from .domains import Coordinate

        return Coordinate(
            self.name, self.unit, self.provenance, self.nominal, self.scale
        )


@dataclass(frozen=True)
class BoxSet:
    """A nonempty Cartesian box, with no implied probability distribution."""

    parameters: tuple[Parameter | VectorParameter, ...]

    def __post_init__(self) -> None:
        from .vector import VectorParameter

        object.__setattr__(self, "parameters", tuple(self.parameters))
        if not self.parameters or not all(
            isinstance(p, (Parameter, VectorParameter)) for p in self.parameters
        ):
            raise ValueError("a box needs at least one Parameter")
        if len({p.name for p in self.parameters}) != len(self.parameters):
            raise ValueError("parameter names must be unique")
        self.space

    @property
    def scalar_parameters(self):
        from .vector import VectorParameter

        return tuple(
            component
            for p in self.parameters
            for component in (
                p.scalar_parameters if isinstance(p, VectorParameter) else (p,)
            )
        )

    def parameter(self, name: str) -> Parameter:
        for parameter in self.scalar_parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)

    @property
    def nominal(self) -> dict[str, float]:
        return {p.name: p.nominal for p in self.scalar_parameters}

    @property
    def upper_corner(self) -> dict[str, float]:
        return {p.name: p.upper for p in self.scalar_parameters}

    def coordinates(self, realization: Mapping[str, float]) -> dict[str, float]:
        return self.space.validate(realization)

    def contains(self, realization: Mapping[str, float]) -> bool:
        values = self.coordinates(realization)
        return all(p.lower <= values[p.name] <= p.upper for p in self.scalar_parameters)

    def distance(self, realization: Mapping[str, float]) -> float:
        """Normalized L-infinity distance; also defined for out-of-box points."""
        return self.default_distance.measure(realization)

    @property
    def space(self):
        from .domains import ParameterSpace

        return ParameterSpace.from_parameters(self.parameters)

    @property
    def feasible_point(self):
        return self.nominal

    @property
    def default_distance(self):
        from .distance import NormalizedLInf

        return NormalizedLInf(self.space)

    @property
    def capabilities(self):
        from .domains import DomainCapabilities

        return DomainCapabilities(linear_optimization=True)

    def maximize_linear(self, coefficients):
        from ._basic_support import box_support

        return box_support(self, coefficients)

    def to_manifest(self):
        from dataclasses import asdict
        from .vector import VectorParameter

        return {
            "kind": "box",
            "parameters": [
                {"kind": "vector", **p.to_dict()}
                if isinstance(p, VectorParameter)
                else asdict(p)
                for p in sorted(self.parameters, key=lambda p: p.name)
            ],
        }

    @cached_property
    def ref(self):
        from .contract_types import reference

        return reference(self.to_manifest(), "uncertainty_set/v1")

    def membership(self, realization):
        from .contract_types import Evidence
        from .domains import MembershipCheck

        values = self.coordinates(realization)
        outside = tuple(
            p.name
            for p in self.scalar_parameters
            if not p.lower <= values[p.name] <= p.upper
        )
        return MembershipCheck(
            "outside" if outside else "inside",
            self.ref,
            values,
            (
                Evidence(
                    "membership",
                    "domain_membership",
                    "box_bounds",
                    "verified",
                    details={
                        "outside_coordinates": outside,
                        "comparison": "inclusive exact bounds",
                    },
                ),
            ),
        )


@dataclass(frozen=True)
class HeatRecoveryDesign:
    """Source capacity in MW thermal; compressor capacity in MW electrical."""

    source_capacity_mw: float = 6.0
    power_capacity_mw: float = 2.0

    def __post_init__(self) -> None:
        for name in ("source_capacity_mw", "power_capacity_mw"):
            object.__setattr__(self, name, nonnegative(getattr(self, name), name))


@dataclass(frozen=True)
class HeatDemand:
    """Required delivery = base_heat_mw * (1 + demand_increase) + reserve_mw."""

    base_heat_mw: float = 6.0
    reserve_mw: float = 0.0

    def __post_init__(self) -> None:
        for name in ("base_heat_mw", "reserve_mw"):
            object.__setattr__(self, name, nonnegative(getattr(self, name), name))
        if self.base_heat_mw == 0:
            raise ValueError("base_heat_mw must be positive")


@dataclass(frozen=True)
class RepairAction:
    """A replacement design under the SAME claim and disturbance box.

    Changing the demand requirement is a new claim, not a repair action.
    Consumers attach costs and selection policies to these engineering changes.
    """

    name: str
    design: HeatRecoveryDesign
    provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.design, HeatRecoveryDesign):
            raise ValueError("design must be a HeatRecoveryDesign")
        for name in ("name", "provenance"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must be a nonempty string")
