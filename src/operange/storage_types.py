"""Two-period thermal-storage contracts, with explicit information timing."""

from __future__ import annotations

from dataclasses import dataclass

from .primitives import finite, nonnegative


def text_field(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class StorageScenario:
    """One event-period future; all recovered heat must be used or stored.

    The preparation observation is the ONLY information about the future
    available before charging. Exact event identity is revealed afterwards.
    """

    name: str
    demand_mw: float
    recovered_heat_mw: float
    provenance: str
    preparation_observation: str = "unrevealed"

    def __post_init__(self) -> None:
        for name in ("name", "provenance", "preparation_observation"):
            text_field(getattr(self, name), name)
        for name in ("demand_mw", "recovered_heat_mw"):
            object.__setattr__(self, name, nonnegative(getattr(self, name), name))


@dataclass(frozen=True)
class ScenarioTree:
    """A finite two-period tree. Observation labels partition shared histories.

    Period one is preparation; period two is the revealed event. There are no
    scenario probabilities, intermediate observations, or hidden time steps.
    """

    scenarios: tuple[StorageScenario, ...]
    preparation_hours: float = 1.0
    event_hours: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        if not self.scenarios or not all(
            isinstance(s, StorageScenario) for s in self.scenarios
        ):
            raise ValueError("tree requires at least one StorageScenario")
        if len({s.name for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("scenario names must be unique")
        for name in ("preparation_hours", "event_hours"):
            value = finite(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    @property
    def domain(self):
        """The finite physical futures; observation permissions are separate."""
        from .domains import Coordinate, FiniteSet, ParameterSpace, Scenario

        space = ParameterSpace(
            tuple(
                Coordinate(
                    name,
                    "MW",
                    "Event-period thermal power; period duration is in the model contract",
                )
                for name in ("demand", "recovered_heat")
            )
        )
        return FiniteSet(
            space,
            tuple(
                Scenario(
                    s.name,
                    {"demand": s.demand_mw, "recovered_heat": s.recovered_heat_mw},
                    s.provenance,
                )
                for s in self.scenarios
            ),
        )


@dataclass(frozen=True)
class StorageDesign:
    """Thermal MW/MWh only. Preparation can charge but cannot discharge."""

    capacity_mwh: float = 1.0
    charging_power_mw: float = 1.0
    discharging_power_mw: float = 1.0
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0
    loss_rate_per_hour: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "capacity_mwh",
            "charging_power_mw",
            "discharging_power_mw",
            "loss_rate_per_hour",
        ):
            object.__setattr__(self, name, nonnegative(getattr(self, name), name))
        for name in ("charge_efficiency", "discharge_efficiency"):
            value = finite(getattr(self, name), name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class StorageRequirement:
    initial_energy_mwh: float = 0.0
    terminal_min_mwh: float = 0.0
    terminal_max_mwh: float | None = None

    def __post_init__(self) -> None:
        for name in ("initial_energy_mwh", "terminal_min_mwh"):
            object.__setattr__(self, name, nonnegative(getattr(self, name), name))
        if self.terminal_max_mwh is not None:
            object.__setattr__(
                self,
                "terminal_max_mwh",
                nonnegative(self.terminal_max_mwh, "terminal_max_mwh"),
            )
            if self.terminal_max_mwh < self.terminal_min_mwh:
                raise ValueError("terminal bounds must be ordered")


@dataclass(frozen=True)
class StorageRepairAction:
    """Equipment change and/or an explicit ideal event-identity signal.

    Earlier revelation preserves the physical futures, but changes information
    permissions. It assumes a correct signal, not a noisy demand forecast.
    """

    name: str
    provenance: str
    design: StorageDesign | None = None
    reveal_before_preparation: bool = False

    def __post_init__(self) -> None:
        text_field(self.name, "name")
        text_field(self.provenance, "provenance")
        if self.design is not None and not isinstance(self.design, StorageDesign):
            raise ValueError("design must be a StorageDesign")
        if type(self.reveal_before_preparation) is not bool:
            raise ValueError("reveal_before_preparation must be a boolean")
