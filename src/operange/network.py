"""Declarative steady-state networks of extensive material and energy rates."""

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Literal

from ._artifact_json import (
    check_identity,
    dumps,
    read_json,
    require_fields,
    require_schema,
)
from .affine import AffineOutput, AffineRequirement
from .contract_types import Record, nonempty, reference, unique
from .domains import ParameterSpace
from .measurements import MeasurementModel
from .objectives import ControlTrackingObjective, LinearObjective


@dataclass(frozen=True)
class ProcessStream(Record):
    """A directed rate: bounded unknown, fixed value, or true input coordinate.

    Units and commodity labels must match at conserving equipment; no property
    calculation or unit conversion is implied. Unknown rates require a finite
    upper bound. Fixed values use equal variable bounds in the compiled model.
    """

    name: str
    unit: str
    physical_kind: Literal["mass_flow", "thermal_power"]
    commodity: str
    lower: float = 0
    upper: float | None = None
    input: str | None = None
    value: float | None = None
    provenance: str = "Caller-declared stream"

    def _validate(self):
        for field in ("name", "unit", "commodity", "provenance"):
            nonempty(getattr(self, field), field)
        if self.lower < 0 or (self.upper is not None and self.upper < self.lower):
            raise ValueError("stream bounds must be nonnegative and ordered")
        if self.input is not None:
            nonempty(self.input, "stream input")
        if self.value is not None:
            if self.value < 0:
                raise ValueError("fixed stream value must be nonnegative")
            if self.input is not None or self.upper is not None or self.lower != 0:
                raise ValueError(
                    "fixed stream value cannot be combined with input or bounds"
                )
        elif self.input is None and self.upper is None:
            raise ValueError("unknown stream rate requires a finite upper bound")

    @property
    def quantity_id(self):
        return f"stream:{self.name}"


def _ports(item, inlets, outlets):
    nonempty(item.name, "equipment name")
    nonempty(item.provenance, "equipment provenance")
    for name in (*inlets, *outlets):
        nonempty(name, "stream port")
    unique((*inlets, *outlets), "equipment stream ports")


@dataclass(frozen=True)
class StreamSource(Record):
    """An external supply boundary; the stream declares its rate or capacity."""

    name: str
    outlet: str
    provenance: str = "Caller-declared supply boundary"
    kind: Literal["source"] = "source"

    def _validate(self):
        _ports(self, (), (self.outlet,))


@dataclass(frozen=True)
class StreamSink(Record):
    """Delivery >= demand input + minimum; optionally prohibit excess delivery.

    Delivery is a selectable service requirement. The no-excess constraint, if
    requested, is an operating limit and always applies.
    """

    name: str
    inlet: str
    demand: str | None = None
    minimum: float = 0
    allow_excess: bool = True
    tolerance: float = 1e-8
    residual_scale: float = 1
    provenance: str = "Caller-declared delivery boundary"
    kind: Literal["sink"] = "sink"

    def _validate(self):
        _ports(self, (self.inlet,), ())
        if self.demand is not None:
            nonempty(self.demand, "sink demand input")
        if self.minimum < 0 or self.tolerance < 0 or self.residual_scale <= 0:
            raise ValueError(
                "sink minimum/tolerance must be nonnegative and scale positive"
            )

    @property
    def requirement_id(self):
        return f"service:{self.name}"


@dataclass(frozen=True)
class Mixer(Record):
    """Combine compatible rates; this does not compute mixture properties."""

    name: str
    inlets: tuple[str, ...]
    outlet: str
    provenance: str = "Caller-declared mixer balance"
    kind: Literal["mixer"] = "mixer"

    def _validate(self):
        _ports(self, self.inlets, (self.outlet,))
        if len(self.inlets) < 2:
            raise ValueError("mixer needs at least two inlet streams")


@dataclass(frozen=True)
class UtilityHeader(Record):
    """Conserve compatible incoming and outgoing rates with free allocation."""

    name: str
    inlets: tuple[str, ...]
    outlets: tuple[str, ...]
    provenance: str = "Caller-declared utility header balance"
    kind: Literal["header"] = "header"

    def _validate(self):
        _ports(self, self.inlets, self.outlets)
        if not self.inlets or not self.outlets:
            raise ValueError("utility header needs incoming and outgoing streams")


@dataclass(frozen=True)
class FixedRatioSplitter(Record):
    """Conserve flow and split it in proportion to nonnegative declared weights.

    Weights are dimensionless ratios, not normalized fractions. Equations use
    cross multiplication without rounding a computed fraction.
    """

    name: str
    inlet: str
    outlets: tuple[str, ...]
    weights: tuple[float, ...]
    provenance: str = "Caller-declared fixed split ratios"
    kind: Literal["splitter"] = "splitter"

    def _validate(self):
        _ports(self, (self.inlet,), self.outlets)
        if len(self.outlets) < 2 or len(self.weights) != len(self.outlets):
            raise ValueError(
                "splitter needs at least two outlets and one weight per outlet"
            )
        if any(w < 0 for w in self.weights) or not any(w > 0 for w in self.weights):
            raise ValueError(
                "splitter weights must be nonnegative with a positive weight"
            )


@dataclass(frozen=True)
class Boiler(Record):
    """Fixed-efficiency conversion of fuel thermal power to steam thermal power.

    Both rates use the same unit. No steam mass, enthalpy, combustion composition,
    pressure, or temperature calculation is implied.
    """

    name: str
    fuel: str
    steam: str
    efficiency: float
    provenance: str = "Caller-declared fixed boiler efficiency"
    kind: Literal["boiler"] = "boiler"

    def _validate(self):
        _ports(self, (self.fuel,), (self.steam,))
        if not 0 < self.efficiency <= 1:
            raise ValueError("boiler efficiency must lie in (0, 1]")


@dataclass(frozen=True)
class StreamTotal(Record):
    """A named reporting sum, not an additional producer or consumer of flow."""

    name: str
    streams: tuple[str, ...]
    provenance: str = "Caller-declared stream total"

    def _validate(self):
        nonempty(self.name, "total name")
        nonempty(self.provenance, "total provenance")
        if not self.streams:
            raise ValueError("stream total needs at least one stream")
        for stream in self.streams:
            nonempty(stream, "total stream")
        unique(self.streams, "total streams")


@dataclass(frozen=True)
class ProcessNetwork(Record):
    """Authoring layer compiling to the existing declarative linear adapter.

    Every stream has exactly one producer and consumer. Physical balances are
    always-applied operating limits; sinks declare selectable service demands.
    Cycles are allowed: internal rates remain bounded algebraic variables.
    An explicit ``actuators`` tuple identifies commandable non-input streams;
    other rates become states. None preserves legacy all-control declarations.
    """

    name: str
    input_space: ParameterSpace
    streams: tuple[ProcessStream, ...]
    equipment: tuple[
        StreamSource | StreamSink | Mixer | UtilityHeader | FixedRatioSplitter | Boiler,
        ...,
    ]
    totals: tuple[StreamTotal, ...] = ()
    outputs: tuple[AffineOutput, ...] = ()
    requirements: tuple[AffineRequirement, ...] = ()
    operating_limits: tuple[AffineRequirement, ...] = ()
    objective: LinearObjective | ControlTrackingObjective | None = None
    measurements: MeasurementModel | None = None
    balance_tolerance: float = 1e-8
    balance_scale: float = 1
    solver_tolerance: float = 1e-9
    provenance: str = "Caller-declared steady-state process network"
    actuators: tuple[str, ...] | None = None

    schema_version: ClassVar[str] = "process_network/v1"

    def _validate(self):
        from ._network_compile import validate_network

        nonempty(self.name, "network name")
        nonempty(self.provenance, "network provenance")
        if self.balance_tolerance < 0 or self.balance_scale <= 0:
            raise ValueError("balance tolerance must be nonnegative and scale positive")
        if not 1e-10 <= self.solver_tolerance <= 1e-4:
            raise ValueError("solver_tolerance must lie between 1e-10 and 1e-4")
        validate_network(self)

    def compile(self):
        """Build physical equations and a LinearProcessAdapter without solving."""
        from ._network_compile import compile_network

        return compile_network(self)

    def to_dict(self):
        data = super().to_dict()
        if self.actuators is None:
            data.pop("actuators")
        return data

    def _body(self):
        return {"schema_version": self.schema_version, "network": self.to_dict()}

    @cached_property
    def ref(self):
        return reference(self._body(), self.schema_version)

    def to_json(self, *, indent=2):
        return dumps(
            {**self._body(), "network_id": self.ref.artifact_id},
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, document):
        """Load declarations without executing a solver or an equipment callback."""
        return cls.from_dict(read_json(document))

    @classmethod
    def from_dict(cls, data):
        require_fields(data, ("schema_version", "network", "network_id"))
        require_schema(data, cls.schema_version)
        try:
            network = cls(**data["network"])
        except TypeError as exc:
            raise ValueError("invalid process network fields") from exc
        check_identity(data["network_id"], network.ref.artifact_id)
        return network
