"""Named coordinates and uncertainty-domain interfaces, independent of physics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .contract_types import (
    Axis,
    ArtifactRef,
    Evidence,
    Record,
    freeze,
    nonempty,
    reference,
    unique,
)
from .primitives import finite


@dataclass(frozen=True)
class Coordinate(Record):
    """A scalar coordinate. A finite event set need not declare a nominal/scale."""

    name: str
    unit: str
    provenance: str
    nominal: float | None = None
    scale: float | None = None
    parameter: str | None = None
    axis: Axis | None = None
    label: str | None = None

    def _validate(self):
        for name in ("name", "unit", "provenance"):
            nonempty(getattr(self, name), name)
        if self.scale is not None and self.scale <= 0:
            raise ValueError("coordinate scale must be positive")
        metadata = (self.parameter, self.axis, self.label)
        if any(value is not None for value in metadata):
            from .vector import component_name

            if (
                any(value is None for value in metadata)
                or self.label not in self.axis.coordinates
            ):
                raise ValueError(
                    "vector coordinates need a parameter, axis and declared label"
                )
            nonempty(self.parameter, "parameter")
            if self.name != component_name(
                self.parameter, self.axis.axis_id, self.label
            ):
                raise ValueError("vector coordinate name disagrees with its metadata")

    def aligned_with(self, other):
        return (
            self.name == other.name
            and self.unit == other.unit
            and self.parameter == other.parameter
            and self.label == other.label
            and (
                self.axis is None
                and other.axis is None
                or self.axis is not None
                and other.axis is not None
                and self.axis.axis_id == other.axis.axis_id
                and set(self.axis.coordinates) == set(other.axis.coordinates)
            )
        )


@dataclass(frozen=True)
class ParameterSpace(Record):
    """Coordinate alignment and numerical scaling; it imposes no membership set."""

    coordinates: tuple[Coordinate, ...]

    def _validate(self):
        if not self.coordinates:
            raise ValueError("parameter space needs coordinates")
        unique(self.names, "coordinate names")
        axes = {}
        scalar_names = {c.name for c in self.coordinates if c.parameter is None}
        for c in self.coordinates:
            if c.parameter is not None:
                if c.parameter in scalar_names:
                    raise ValueError("scalar and vector parameter names collide")
                definition = (c.axis.axis_id, frozenset(c.axis.coordinates))
                if c.parameter in axes and axes[c.parameter] != definition:
                    raise ValueError("conflicting vector axis definitions")
                axes[c.parameter] = definition

    @classmethod
    def from_parameters(cls, parameters):
        from .primitives import Parameter
        from .vector import VectorParameter

        coordinates = []
        for parameter in parameters:
            if isinstance(parameter, VectorParameter):
                coordinates.extend(parameter.space.coordinates)
            elif isinstance(parameter, Parameter):
                coordinates.append(parameter.coordinate)
            else:
                raise ValueError("expected scalar or vector parameters")
        return cls(tuple(coordinates))

    @property
    def names(self):
        return tuple(c.name for c in self.coordinates)

    def coordinate(self, name):
        for coordinate in self.coordinates:
            if coordinate.name == name:
                return coordinate
        raise ValueError(f"unknown coordinate: {name}")

    def validate(self, realization):
        if not isinstance(realization, Mapping) or set(realization) != set(self.names):
            raise ValueError(
                "realization must have exactly the declared named coordinates"
            )
        return {name: finite(realization[name], name) for name in self.names}

    def check_alignment(self, other):
        if not isinstance(other, ParameterSpace) or set(self.names) != set(other.names):
            raise ValueError("coordinate names do not align")
        if any(not c.aligned_with(other.coordinate(c.name)) for c in self.coordinates):
            raise ValueError(
                "coordinate units or vector axes do not align; conversion must be explicit"
            )

    def flatten(self, realization):
        """Expand explicitly nested, label-keyed vectors into scalar coordinates."""
        keys = {c.parameter or c.name for c in self.coordinates}
        if not isinstance(realization, Mapping) or set(realization) != keys:
            raise ValueError("nested realization must match the parameter names")
        flat = {}
        for c in self.coordinates:
            if c.parameter is None:
                flat[c.name] = realization[c.name]
            else:
                vector = realization[c.parameter]
                labels = {
                    v.label for v in self.coordinates if v.parameter == c.parameter
                }
                if not isinstance(vector, Mapping) or set(vector) != labels:
                    raise ValueError("vector values must match their declared labels")
                flat[c.name] = vector[c.label]
        return self.validate(flat)

    def unflatten(self, realization):
        values = self.validate(realization)
        nested = {}
        for c in self.coordinates:
            if c.parameter is None:
                nested[c.name] = values[c.name]
            else:
                nested.setdefault(c.parameter, {})[c.label] = values[c.name]
        return nested

    def denormalize(self, realization):
        values = self.validate(realization)
        self.normalize(self.nominal)
        return {
            c.name: finite(c.nominal + c.scale * values[c.name], "physical coordinate")
            for c in self.coordinates
        }

    @property
    def nominal(self):
        if any(c.nominal is None for c in self.coordinates):
            raise ValueError("this coordinate space has no complete nominal point")
        return {c.name: c.nominal for c in self.coordinates}

    def normalize(self, realization):
        values = self.validate(realization)
        if any(c.nominal is None or c.scale is None for c in self.coordinates):
            raise ValueError(
                "normalization requires a nominal and scale for every coordinate"
            )
        return {
            c.name: finite(
                (values[c.name] - c.nominal) / c.scale, "normalized coordinate"
            )
            for c in self.coordinates
        }


@dataclass(frozen=True)
class DomainCapabilities(Record):
    """Domain operations only. Global process guarantees belong to an adapter."""

    membership: bool = True
    enumeration: bool = False
    sampling: bool = False
    linear_optimization: bool = False


@dataclass(frozen=True)
class MembershipCheck(Record):
    status: Literal["inside", "outside", "unknown"]
    domain_ref: ArtifactRef
    realization: Mapping[str, float]
    evidence: tuple[Evidence, ...]

    def _validate(self):
        for name, value in self.realization.items():
            finite(value, name)
        if self.status != "unknown" and not any(
            e.subject == "domain_membership" and e.outcome == "verified"
            for e in self.evidence
        ):
            raise ValueError("membership conclusion needs verified evidence")


@runtime_checkable
class UncertaintySet(Protocol):
    @property
    def space(self) -> ParameterSpace: ...

    @property
    def capabilities(self) -> DomainCapabilities: ...

    @property
    def ref(self) -> ArtifactRef: ...

    def to_manifest(self) -> dict: ...

    def membership(self, realization: Mapping[str, float]) -> MembershipCheck: ...


@dataclass(frozen=True)
class Scenario(Record):
    name: str
    values: Mapping[str, float]
    provenance: str

    def _validate(self):
        nonempty(self.name, "scenario name")
        nonempty(self.provenance, "scenario provenance")
        object.__setattr__(
            self,
            "values",
            freeze({name: finite(value, name) for name, value in self.values.items()}),
        )


@dataclass(frozen=True)
class FiniteSet(Record):
    """Named physical realizations; labels retain distinct event identities.

    No probabilities, observation timing, or nominal point are inferred.
    """

    space: ParameterSpace
    scenarios: tuple[Scenario, ...]

    def _validate(self):
        if not self.scenarios:
            raise ValueError("finite uncertainty set must be nonempty")
        unique(tuple(s.name for s in self.scenarios), "scenario names")
        object.__setattr__(
            self, "scenarios", tuple(sorted(self.scenarios, key=lambda s: s.name))
        )
        for scenario in self.scenarios:
            self.space.validate(scenario.values)

    @property
    def capabilities(self):
        normalized = all(
            c.nominal is not None and c.scale is not None
            for c in self.space.coordinates
        )
        return DomainCapabilities(enumeration=True, linear_optimization=normalized)

    def maximize_linear(self, coefficients):
        from ._basic_support import finite_support

        return finite_support(self, coefficients)

    def to_manifest(self):
        return {
            "kind": "finite_set",
            "space": self.space.to_dict(),
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

    @property
    def ref(self):
        return reference(self.to_manifest(), "uncertainty_set/v1")

    def members(self):
        return self.scenarios

    @property
    def feasible_point(self):
        return self.scenarios[0].values

    def scenario(self, name):
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        raise ValueError(f"unknown scenario: {name}")

    def membership(self, realization):
        values = self.space.validate(realization)
        matches = tuple(s.name for s in self.scenarios if dict(s.values) == values)
        return MembershipCheck(
            "inside" if matches else "outside",
            self.ref,
            values,
            (
                Evidence(
                    "membership",
                    "domain_membership",
                    "finite_enumeration",
                    "verified",
                    details={
                        "matching_scenarios": matches,
                        "comparison": "exact declared coordinate values",
                    },
                ),
            ),
        )

    def contains(self, realization):
        return self.membership(realization).status == "inside"
