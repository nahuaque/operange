"""Steady-state sensible-heat targeting, separate from network feasibility."""

from dataclasses import dataclass
from typing import Literal

from .claim import (
    AdapterCapabilities,
    Capability,
    Claim,
    bind_contract,
    rejected_result,
)
from .contract_types import (
    ConstraintSpec,
    ProcessContract,
    QuantitySpec,
    Record,
    nonempty,
    snapshot,
    unique,
)
from .domains import Coordinate, FiniteSet, ParameterSpace
from .recourse import RecoursePolicy


@dataclass(frozen=True)
class SensibleHeatStream(Record):
    """Supplied stream endpoints in degC and constant heat-capacity flow in kW/K.

    Zero heat-capacity flow or equal endpoints represent zero duty. Phase
    changes, variable heat capacity and thermodynamic properties are not inferred.
    """

    name: str
    kind: Literal["hot", "cold"]
    supply_temperature_c: float
    target_temperature_c: float
    heat_capacity_flow_kw_per_k: float
    provenance: str

    def _validate(self):
        nonempty(self.name, "stream name")
        nonempty(self.provenance, "stream provenance")
        self.validate_conditions(
            self.supply_temperature_c,
            self.target_temperature_c,
            self.heat_capacity_flow_kw_per_k,
        )

    def validate_conditions(self, supply, target, capacity_flow):
        if min(supply, target) < -273.15 or capacity_flow < 0:
            raise ValueError(
                "stream temperatures must be physical and capacity flow nonnegative"
            )
        if (self.kind == "hot" and supply < target) or (
            self.kind == "cold" and supply > target
        ):
            raise ValueError(
                "stream endpoints contradict its declared hot/cold direction"
            )

    def input_name(self, field):
        if field not in (
            "supply_temperature_c",
            "target_temperature_c",
            "heat_capacity_flow_kw_per_k",
        ):
            raise ValueError("unknown sensible-heat stream input")
        return f"{self.name}:{field}"


@dataclass(frozen=True)
class HeatCascadeAdapter(Record):
    """Minimum utility targets at fixed positive delta-T-min, in kW and degC.

    Arbitrary heat matches and splitting are allowed independently per scenario.
    Utilities can heat/cool beyond all process temperatures. No exchanger
    topology, area, utility temperature restrictions or dynamic response is
    assessed. A passing target claim is not an installed-network certificate.
    """

    name: str
    streams: tuple[SensibleHeatStream, ...]
    delta_t_min_k: float
    hot_utility_capacity_kw: float
    cold_utility_capacity_kw: float
    tolerance_kw: float = 1e-8

    def _validate(self):
        nonempty(self.name, "model name")
        if not self.streams:
            raise ValueError("heat cascade needs at least one stream")
        unique(tuple(s.name for s in self.streams), "heat streams")
        if (
            self.delta_t_min_k <= 0
            or min(
                self.hot_utility_capacity_kw,
                self.cold_utility_capacity_kw,
                self.tolerance_kw,
            )
            < 0
        ):
            raise ValueError(
                "delta-T-min must be positive; capacities and tolerance nonnegative"
            )
        self.base_contract

    @property
    def input_space(self):
        return ParameterSpace(
            tuple(
                Coordinate(s.input_name(field), unit, s.provenance, getattr(s, field))
                for s in self.streams
                for field, unit in (
                    ("supply_temperature_c", "degC"),
                    ("target_temperature_c", "degC"),
                    ("heat_capacity_flow_kw_per_k", "kW/K"),
                )
            )
        )

    @property
    def recourse_policy(self):
        return RecoursePolicy("fixed", ())

    @property
    def base_contract(self):
        quantities = (
            tuple(
                QuantitySpec(c.name, c.unit, "supplied_stream_condition", "input")
                for c in self.input_space.coordinates
            )
            + tuple(
                QuantitySpec(name, "kW", "thermal_power_target", "output")
                for name in (
                    "minimum_hot_utility",
                    "minimum_cold_utility",
                    "maximum_heat_recovery",
                    "hot_stream_duty",
                    "cold_stream_duty",
                )
            )
            + tuple(
                QuantitySpec(name, "kW", "thermal_power_residual", "residual")
                for name in (
                    "energy_balance_residual",
                    "hot_utility_residual",
                    "cold_utility_residual",
                )
            )
        )
        return ProcessContract(
            self.name,
            {
                "kind": "sensible_heat_cascade",
                "scope": "thermodynamic_utility_targets",
                "streams": tuple(s.to_dict() for s in self.streams),
                "heat_capacity": "constant_within_each_stream",
                "heat_matches": "unrestricted_and_independent_per_scenario",
                "stream_splitting": "unrestricted",
                "utilities": "hot_above_and_cold_below_all_process_temperatures",
                "installed_network_feasibility": "not_assessed",
                "excluded": (
                    "phase_change",
                    "heat_losses",
                    "heat_pumps",
                    "storage",
                    "dynamics",
                ),
            },
            {
                "delta_t_min_k": self.delta_t_min_k,
                "hot_utility_capacity_kw": self.hot_utility_capacity_kw,
                "cold_utility_capacity_kw": self.cold_utility_capacity_kw,
            },
            {},
            {"state": "independent_steady_states", "network_controls": "not_modelled"},
            quantities,
            (
                ConstraintSpec(
                    "heat_balance",
                    "energy_balance_residual",
                    "equation",
                    "eq",
                    "hot duty + minimum hot utility - cold duty - minimum cold utility",
                    0,
                ),
                ConstraintSpec(
                    "hot_utility_target",
                    "hot_utility_residual",
                    "requirement",
                    "le",
                    "minimum hot utility - hot utility capacity (target screening only)",
                    self.tolerance_kw,
                ),
                ConstraintSpec(
                    "cold_utility_target",
                    "cold_utility_residual",
                    "requirement",
                    "le",
                    "minimum cold utility - cold utility capacity (target screening only)",
                    self.tolerance_kw,
                ),
            ),
            {
                "arithmetic": "exact_rationals_of_declared_floats",
                "residual_export": "rounded_toward_positive_infinity",
                "absolute_tolerance_kw": self.tolerance_kw,
                "pinch_detection": "exact_zero_cascade_including_terminal_and_interval_pinches",
            },
        )

    def describe(self, claim):
        return bind_contract(
            self.base_contract, claim.domain, claim.recourse, claim.requirements
        )

    def capabilities(self, claim):
        if (
            type(claim.domain) is not FiniteSet
            or claim.recourse != self.recourse_policy
        ):
            no = Capability(
                False,
                "Heat targeting supports an explicit FiniteSet and no network control rules only.",
            )
            return AdapterCapabilities(*(no for _ in range(5)))
        no = Capability(
            False,
            "Heat-cascade derivatives and nearest-failure searches are not implemented.",
        )
        return AdapterCapabilities(
            Capability(
                True,
                "Exact sensible-heat target at one declared steady-state scenario; no installed-network assessment.",
            ),
            no,
            Capability(
                True,
                "Complete finite enumeration of utility targets, not network feasibility or continuous uncertainty.",
            ),
            no,
            no,
        )

    def as_claim(self, domain, *, requirements=None, recourse=None):
        return Claim(
            self,
            domain,
            self.recourse_policy if recourse is None else recourse,
            requirements,
        )

    def run(self, claim, operation, realization, options):
        from ._pinch_results import audit, evaluate

        if options:
            return rejected_result(
                claim.contract,
                operation,
                {"realization": snapshot(realization), **snapshot(options)},
                "Unknown heat-cascade query options.",
                execution="invalid",
                code="invalid_options",
            )
        if operation == "evaluation":
            return evaluate(claim, realization)
        if operation == "audit":
            return audit(claim)
        return rejected_result(
            claim.contract, operation, {}, "Unsupported heat-cascade operation."
        )
