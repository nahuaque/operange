"""A deliberately bounded process adapter with independently checked LPs.

This model has fixed COP > 1, source/compressor capacities, a rectangular
disturbance set, and fully observed static recourse. All power values are MW.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

from . import _reference
from .linear import solve_lp as _solve_lp
from .primitives import (
    BoxSet,
    HeatDemand,
    HeatRecoveryDesign,
    Parameter,
    RepairAction,
    finite,
)
from .results import (
    BranchResult,
    ClaimAudit,
    OperationResult,
    RepairComparison,
    RepairResult,
    ThresholdResult,
)

COORDINATES = ("source_derating", "demand_increase")
MODEL_ID = "heat_recovery_constant_cop_v1"


@dataclass(frozen=True)
class HeatRecoveryClaim:
    """Audit delivered heat under a constant-COP model and static recourse.

    The inclusive pass rule permits the declared absolute numerical tolerance.
    Only this model's monotonicity justifies checking the box's upper corner.
    """

    name: str
    design: HeatRecoveryDesign
    uncertainty: BoxSet
    requirement: HeatDemand = field(default_factory=HeatDemand)
    cop: float = 4.0
    recourse: str = "fully_observed_static"
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        if not isinstance(self.design, HeatRecoveryDesign):
            raise ValueError("design must be a HeatRecoveryDesign")
        if not isinstance(self.requirement, HeatDemand):
            raise ValueError("requirement must be a HeatDemand")
        if not isinstance(self.uncertainty, BoxSet):
            raise ValueError("uncertainty must be a BoxSet")
        if any(type(p) is not Parameter for p in self.uncertainty.parameters):
            raise ValueError("the heat model requires scalar disturbance parameters")
        object.__setattr__(self, "cop", finite(self.cop, "cop"))
        object.__setattr__(self, "tolerance", finite(self.tolerance, "tolerance"))
        if self.cop <= 1:
            raise ValueError("cop must exceed one")
        if not 1e-10 <= self.tolerance <= 1e-4:
            raise ValueError("tolerance must lie between 1e-10 and 1e-4")
        if self.recourse != "fully_observed_static":
            raise ValueError("only fully_observed_static recourse is supported")
        if {p.name for p in self.uncertainty.parameters} != set(COORDINATES):
            raise ValueError(f"this model requires exactly {COORDINATES}")
        if any(p.unit != "fraction" for p in self.uncertainty.parameters):
            raise ValueError(
                "heat-model uncertainty coordinates require unit='fraction'"
            )
        f, delta = (self.uncertainty.parameter(name) for name in COORDINATES)
        if f.lower < 0 or f.upper > 1:
            raise ValueError("source_derating bounds must lie in [0, 1]")
        if delta.lower < -1:
            raise ValueError("demand_increase cannot be below -1")
        for value in (
            self.cop * self.design.power_capacity_mw,
            self.cop / (self.cop - 1) * self.design.source_capacity_mw,
            self.requirement.base_heat_mw * (1 + delta.upper)
            + self.requirement.reserve_mw,
        ):
            finite(value, "derived model coefficient")

    def evaluate(self, realization: Mapping[str, float]) -> OperationResult:
        values = self.uncertainty.coordinates(realization)
        if not self.uncertainty.contains(values):
            raise ValueError("realization is outside the declared uncertainty box")
        f = values["source_derating"]
        required = (
            self.requirement.base_heat_mw * (1 + values["demand_increase"])
            + self.requirement.reserve_mw
        )
        available = self.design.source_capacity_mw * (1 - f)
        # Variables are electrical input p, source heat s, delivered heat q.
        x, evidence = _solve_lp(
            [0, 0, -1],
            [(0, self.design.power_capacity_mw), (0, available), (0, None)],
            self.tolerance,
            equalities=[[-(self.cop - 1), 1, 0], [-self.cop, 0, 1]],
            rhs=[0, 0],
        )
        if x is not None:
            expected = _reference.capacity(
                self.cop,
                self.design.source_capacity_mw,
                self.design.power_capacity_mw,
                f,
            )
            error = abs(x[2] - expected)
            evidence = replace(evidence, reference_error=error)
            if error > self.tolerance:
                evidence = replace(
                    evidence,
                    status="unresolved",
                    message="analytical capacity check failed",
                )
                x = None
        if x is None:
            return OperationResult(
                "inconclusive", values, required, None, None, None, None, (), evidence
            )
        p, source, q = x
        slack = q - required
        binding = []
        if abs(p - self.design.power_capacity_mw) <= self.tolerance:
            binding.append("compressor_capacity")
        if abs(source - available) <= self.tolerance:
            binding.append("source_capacity")
        evidence = replace(
            evidence,
            message="LP optimum agrees with the analytical global capacity bound",
        )
        return OperationResult(
            "pass" if slack >= -self.tolerance else "fail",
            values,
            required,
            q,
            p,
            source,
            slack,
            tuple(binding),
            evidence,
        )

    def audit(self) -> ClaimAudit:
        nominal = self.evaluate(self.uncertainty.nominal)
        worst = self.evaluate(self.uncertainty.upper_corner)
        return ClaimAudit(self.name, worst.status, nominal, worst)

    def breaking_witness(self, *, shortfall_margin_mw: float = 0.01) -> ThresholdResult:
        margin = finite(shortfall_margin_mw, "shortfall_margin_mw")
        if margin <= 10 * self.tolerance:
            raise ValueError(
                "shortfall_margin_mw must exceed ten times the numerical tolerance"
            )
        return self._threshold_search(margin, kind="witness")

    def boundary_distance(self) -> ThresholdResult:
        """Distance to nonpositive delivery slack; equality is NOT a failure."""
        return self._threshold_search(0.0, kind="boundary")

    def _threshold_search(self, margin: float, *, kind: str) -> ThresholdResult:
        parameters = tuple(self.uncertainty.parameter(name) for name in COORDINATES)
        f, delta = parameters
        heat = self.requirement.base_heat_mw
        reserve = self.requirement.reserve_mw
        source_bound = self.cop / (self.cop - 1) * self.design.source_capacity_mw
        branches = []
        # min(electric_capacity, source_capacity) misses demand iff at least
        # one of these affine capacities misses it. Solve the exhaustive union.
        for limit, intercept, slope in (
            ("compressor_capacity", self.cop * self.design.power_capacity_mw, 0.0),
            ("source_capacity", source_bound, source_bound),
        ):
            inequalities = [
                [1, 0, -f.scale],
                [-1, 0, -f.scale],
                [0, 1, -delta.scale],
                [0, -1, -delta.scale],
                [-slope, -heat, 0],
            ]
            upper = [
                f.nominal,
                -f.nominal,
                delta.nominal,
                -delta.nominal,
                heat + reserve - intercept - margin,
            ]
            reference = _reference.affine_threshold_radius(
                intercept - slope * f.nominal - heat * (1 + delta.nominal) - reserve,
                margin,
                [slope, heat],
                [p.scale for p in parameters],
                [p.upper - p.nominal for p in parameters],
            )
            x, evidence = _solve_lp(
                [0, 0, 1],
                [(f.lower, f.upper), (delta.lower, delta.upper), (0, None)],
                self.tolerance,
                inequalities=inequalities,
                upper=upper,
            )
            point = None
            distance = None
            status = "unresolved"
            if x is None and evidence.status == "infeasible" and reference is None:
                status = "unreachable"
                evidence = replace(
                    evidence,
                    message="LP infeasibility agrees with the analytical box bound",
                )
            elif x is not None and reference is not None:
                # Only correct roundoff at box endpoints after checking raw residuals.
                point = {
                    p.name: min(p.upper, max(p.lower, value))
                    for p, value in zip(parameters, x[:2])
                }
                distance = self.uncertainty.distance(point)
                slack = (
                    intercept
                    - slope * point[f.name]
                    - heat * (1 + point[delta.name])
                    - reserve
                )
                error = max(abs(distance - reference), abs(x[2] - reference))
                evidence = replace(evidence, reference_error=error)
                if error <= self.tolerance and slack <= -margin + self.tolerance:
                    status = "optimal"
                    evidence = replace(
                        evidence,
                        message="LP optimum agrees with independent analytical distance",
                    )
                else:
                    point, distance = None, None
            if status == "unresolved":
                evidence = replace(
                    evidence,
                    status="unresolved",
                    message=f"threshold branch unresolved: {evidence.message}",
                )
            branches.append(
                BranchResult(limit, status, distance, point, reference, evidence)
            )
        resolved = all(row.status != "unresolved" for row in branches)
        candidates = [row for row in branches if row.status == "optimal"]
        if not candidates:
            return ThresholdResult(
                kind,
                "unreachable" if resolved else "inconclusive",
                margin,
                None,
                False,
                None,
                tuple(branches),
            )
        best = min(candidates, key=lambda row: row.distance)
        operation = self.evaluate(best.realization)
        verified = (
            operation.status != "inconclusive"
            and operation.slack_mw <= -margin + self.tolerance
        )
        minimum = resolved and verified
        status = "found" if minimum else "inconclusive"
        if minimum and best.distance == 0 and operation.status == "fail":
            status = "already_broken"
        return ThresholdResult(
            kind, status, margin, best.distance, minimum, operation, tuple(branches)
        )

    def compare_repairs(self, actions: Sequence[RepairAction]) -> RepairComparison:
        """Audit engineering changes in caller order without ranking or selection."""
        actions = tuple(actions)
        if not all(isinstance(action, RepairAction) for action in actions):
            raise ValueError("all repair candidates must be RepairAction objects")
        if len({action.name for action in actions}) != len(actions):
            raise ValueError("repair names must be unique")
        results = tuple(
            RepairResult(action, replace(self, design=action.design).audit())
            for action in actions
        )
        return RepairComparison(results)

    def with_repair(self, action: RepairAction) -> HeatRecoveryClaim:
        """Apply a candidate without changing the requirement or uncertainty."""
        if not isinstance(action, RepairAction):
            raise ValueError("action must be a RepairAction")
        return replace(self, design=action.design)

    def evaluate_result(self, realization):
        """Return a portable evaluation, separating physics and service checks."""
        return self.as_claim().evaluate_result(realization)

    def audit_result(self):
        """Return structured full-box robustness evidence."""
        return self.as_claim().audit_result()

    def breaking_result(self, *, shortfall_margin_mw=0.01):
        """Return witness validity separately from minimum-distance resolution."""
        return self.as_claim().breaking_result(shortfall_margin_mw=shortfall_margin_mw)

    def boundary_result(self):
        """Search nonpositive margin; equality alone does not refute the claim."""
        return self.as_claim().boundary_result()

    def sensitivity_result(self, realization, **options):
        """Request analytical optimized-response derivatives with named coordinates."""
        return self.as_claim().sensitivity_result(realization, **options)

    @property
    def recourse_policy(self):
        from .recourse import DecisionRule, RecoursePolicy

        return RecoursePolicy(
            "static",
            (DecisionRule("electrical_power", "MW", "operation", COORDINATES),),
        )

    def as_claim(self):
        """Bind this reference model through the common engineering DSL."""
        from .adapters import HeatRecoveryAdapter
        from .claim import Claim

        return Claim(
            HeatRecoveryAdapter(self),
            self.uncertainty,
            self.recourse_policy,
            distance=self.uncertainty.default_distance,
        )

    def freeze(self):
        """Freeze this small serializable model contract; no guarantee upgrade."""
        from .policy import FrozenHeatRecoveryContract

        return FrozenHeatRecoveryContract(self)
