"""Bounded vector parameters with explicit, named component alignment."""

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import quote

from .contract_types import Axis, Record, freeze, nonempty
from .primitives import Parameter, finite


def component_name(parameter, axis, label):
    return (
        f"{quote(parameter, safe='')}[{quote(axis, safe='')}={quote(label, safe='')}]"
    )


@dataclass(frozen=True)
class VectorParameter(Record):
    name: str
    unit: str
    axis: Axis
    nominal: Mapping[str, float]
    lower: float | Mapping[str, float]
    upper: float | Mapping[str, float]
    scale: float | Mapping[str, float]
    provenance: str

    def _validate(self):
        for name in ("name", "unit", "provenance"):
            nonempty(getattr(self, name), name)
        for name in ("nominal", "lower", "upper", "scale"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                if set(value) != set(self.axis.coordinates):
                    raise ValueError(f"{name} must match the vector's axis labels")
            else:
                value = dict.fromkeys(self.axis.coordinates, value)
            object.__setattr__(
                self,
                name,
                freeze(
                    {
                        label: finite(value[label], name)
                        for label in self.axis.coordinates
                    }
                ),
            )
        # Reuse scalar validation for bounds, scales and finite excursions.
        self.scalar_parameters

    def name_for(self, label):
        if label not in self.axis.coordinates:
            raise ValueError(f"unknown vector label: {label}")
        return component_name(self.name, self.axis.axis_id, label)

    @property
    def scalar_parameters(self):
        return tuple(
            Parameter(
                self.name_for(label),
                self.unit,
                self.nominal[label],
                self.lower[label],
                self.upper[label],
                self.scale[label],
                self.provenance,
            )
            for label in self.axis.coordinates
        )

    @property
    def space(self):
        from .domains import Coordinate, ParameterSpace

        return ParameterSpace(
            tuple(
                Coordinate(
                    self.name_for(label),
                    self.unit,
                    self.provenance,
                    self.nominal[label],
                    self.scale[label],
                    self.name,
                    self.axis,
                    label,
                )
                for label in self.axis.coordinates
            )
        )

    def flatten(self, values):
        if not isinstance(values, Mapping) or set(values) != set(self.axis.coordinates):
            raise ValueError("vector values must match the axis labels")
        return {
            self.name_for(label): finite(values[label], label)
            for label in self.axis.coordinates
        }

    def unflatten(self, values):
        flat = self.space.validate(values)
        return {label: flat[self.name_for(label)] for label in self.axis.coordinates}
