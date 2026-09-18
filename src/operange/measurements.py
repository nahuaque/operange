"""Declared sensor readings over true process inputs and bounded error inputs."""

from dataclasses import dataclass
from fractions import Fraction as F

from .affine import AffineTerm
from .contract_types import Record, nonempty, unique
from .domains import Coordinate, ParameterSpace
from .primitives import finite


@dataclass(frozen=True)
class AffineMeasurement(Record):
    """A sensor reading, rounded once after its exact affine sum."""

    name: str
    unit: str
    offset: float
    terms: tuple[AffineTerm, ...]
    provenance: str

    def _validate(self):
        for field in ("name", "unit", "provenance"):
            nonempty(getattr(self, field), field)
        unique(tuple(t.variable for t in self.terms), "measurement inputs")

    def read(self, realization):
        return finite(
            float(
                F(self.offset)
                + sum(
                    (
                        F(t.coefficient)
                        * F(finite(realization[t.variable], t.variable))
                        for t in self.terms
                    ),
                    F(0),
                )
            ),
            self.name,
        )


@dataclass(frozen=True)
class MeasurementModel(Record):
    """The complete observation interface for a static linear controller.

    Error bounds and dependence live in the claim's joint uncertainty domain.
    Reusing an error input across readings represents shared calibration error.
    Error inputs must not enter the physical model's output equations.
    """

    name: str
    readings: tuple[AffineMeasurement, ...]
    error_inputs: tuple[str, ...] = ()
    provenance: str = "Caller-declared measurement assumptions"

    def _validate(self):
        nonempty(self.name, "measurement model name")
        nonempty(self.provenance, "measurement provenance")
        if not self.readings or any(
            type(r) is not AffineMeasurement for r in self.readings
        ):
            raise ValueError("measurement model needs declarative affine readings")
        unique(tuple(r.name for r in self.readings), "measurement names")
        unique(self.error_inputs, "measurement error inputs")
        for n in self.error_inputs:
            nonempty(n, "measurement error input")

    @property
    def space(self):
        return ParameterSpace(
            tuple(Coordinate(r.name, r.unit, r.provenance) for r in self.readings)
        )

    def validate(self, model):
        inputs = set(model.input_space.names)
        reserved = (
            inputs | {c.name for c in model.variables} | {o.name for o in model.outputs}
        )
        if any(r.name in reserved for r in self.readings):
            raise ValueError(
                "measurement names must be distinct from physical inputs, controls and outputs"
            )
        if not set(self.error_inputs) <= inputs:
            raise ValueError("measurement errors must be declared uncertainty inputs")
        if any(t.variable in self.error_inputs for o in model.outputs for t in o.terms):
            raise ValueError(
                "measurement error inputs cannot change physical output equations"
            )
        for r in self.readings:
            for t in r.terms:
                if t.variable not in inputs:
                    raise ValueError(
                        "measurement terms must reference true or error inputs"
                    )
                if (
                    t.unit
                    != f"{r.unit}/{model.input_space.coordinate(t.variable).unit}"
                ):
                    raise ValueError(
                        "measurement coefficient units must match reading/input units"
                    )

    def observe(self, realization):
        """Compute all declared readings; never expose latent inputs as readings."""
        return {r.name: r.read(realization) for r in self.readings}
