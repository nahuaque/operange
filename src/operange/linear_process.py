"""Bounded linear operation with fully observed, single-stage adjustment."""

from dataclasses import dataclass, replace
from functools import cached_property

from .affine import AffineOutput, AffineProcessAdapter, AffineRequirement
from .claim import (
    AdapterCapabilities,
    Capability,
    Claim,
    bind_contract,
    rejected_result,
)
from .contract_types import ConstraintSpec, QuantitySpec, Record, nonempty
from .domains import FiniteSet, ParameterSpace
from .convex_hull import ConvexHullSet
from .primitives import BoxSet
from .objectives import ControlTrackingObjective, LinearObjective
from .measurements import MeasurementModel
from .operating import OperatingSpec, resolve_operating


@dataclass(frozen=True)
class LinearControl(Record):
    """A scalar control with finite physical bounds, enforced without tolerance."""

    name: str
    unit: str
    lower: float
    upper: float
    physical_kind: str = "declared_process_control"

    def _validate(self):
        for name in ("name", "unit", "physical_kind"):
            nonempty(getattr(self, name), name)
        if self.lower > self.upper:
            raise ValueError("control bounds must be ordered")

    @property
    def quantity(self):
        return QuantitySpec(self.name, self.unit, self.physical_kind, "control")


@dataclass(frozen=True)
class LinearState(Record):
    """A bounded algebraic variable, with no command or observation permission.

    A feasible completion satisfies the declared equations, limits and bounds.
    Declaring a state does not establish that those equations determine it uniquely.
    """

    name: str
    unit: str
    lower: float
    upper: float
    physical_kind: str = "algebraic_state"

    def _validate(self):
        for name in ("name", "unit", "physical_kind"):
            nonempty(getattr(self, name), name)
        if self.lower > self.upper:
            raise ValueError("state bounds must be ordered")

    @property
    def quantity(self):
        return QuantitySpec(self.name, self.unit, self.physical_kind, "state")


@dataclass(frozen=True)
class LinearProcessAdapter(Record):
    """Find controls satisfying equipment limits and selected requirements jointly.

    Outputs use the same physical affine expressions as AffineProcessAdapter.
    Operating limits use AffineRequirement declarations but always apply, even
    when a claim selects only some service requirements. An optional objective
    selects a preferred feasible dispatch; optimality is checked separately.
    """

    name: str
    input_space: ParameterSpace
    outputs: tuple[AffineOutput, ...]
    requirements: tuple[AffineRequirement, ...]
    controls: tuple[LinearControl, ...]
    operating_limits: tuple[AffineRequirement, ...] = ()
    solver_tolerance: float = 1e-9
    objective: LinearObjective | ControlTrackingObjective | None = None
    measurements: MeasurementModel | None = None
    states: tuple[LinearState, ...] = ()

    def _validate(self):
        if self.measurements is not None:
            if type(self.measurements) is not MeasurementModel:
                raise ValueError("measurements require a declarative MeasurementModel")
            self.measurements.validate(self)
        if not self.variables or not self.requirements:
            raise ValueError(
                "linear operation needs bounded variables and requirements"
            )
        if not 1e-10 <= self.solver_tolerance <= 1e-4:
            raise ValueError("solver_tolerance must lie between 1e-10 and 1e-4")
        # Reuse affine declaration/unit validation, then check generated bounds.
        self.base_contract

    def to_dict(self):
        data = super().to_dict()
        # Preserve existing model and frozen-controller identities when omitted.
        if self.objective is None:
            data.pop("objective")
        if self.measurements is None:
            data.pop("measurements")
        if not self.states:
            data.pop("states")
        return data

    @property
    def variables(self):
        """Mathematical unknowns; only ``controls`` are physical actuators."""
        return self.controls + self.states

    @property
    def observation_space(self):
        return (
            self.input_space if self.measurements is None else self.measurements.space
        )

    def observations(self, realization):
        return (
            realization
            if self.measurements is None
            else self.measurements.observe(realization)
        )

    @cached_property
    def _affine(self):
        return AffineProcessAdapter(
            self.name,
            self.input_space,
            self.outputs,
            self.operating_limits + self.requirements,
            tuple(c.quantity for c in self.controls),
            tuple(s.quantity for s in self.states),
        )

    def output(self, name):
        return self._affine.output(name)

    equation_id = staticmethod(AffineProcessAdapter.equation_id)
    residual_id = staticmethod(AffineProcessAdapter.residual_id)

    @staticmethod
    def bound_id(control, side):
        return f"{control.quantity.role}_{side}:{control.name}"

    @cached_property
    def base_contract(self):
        base = self._affine.base_contract
        limits = {r.name for r in self.operating_limits}
        constraints = [
            replace(c, kind="operating_limit") if c.constraint_id in limits else c
            for c in base.constraints
        ]
        quantities = list(base.quantities)
        for control in self.variables:
            for side, expression in (
                ("lower", f"lower - {control.quantity.role}"),
                ("upper", f"{control.quantity.role} - upper"),
            ):
                name = self.bound_id(control, side)
                quantities.append(
                    QuantitySpec(
                        self.residual_id(name),
                        control.unit,
                        control.physical_kind,
                        "residual",
                    )
                )
                constraints.append(
                    ConstraintSpec(
                        name,
                        self.residual_id(name),
                        "operating_limit",
                        "le",
                        expression,
                        0,
                    )
                )
        if isinstance(self.objective, LinearObjective):
            self.output(self.objective.output)
        elif isinstance(self.objective, ControlTrackingObjective):
            units = {c.name: c.unit for c in self.controls}
            for target in self.objective.targets:
                if target.control not in units or units[target.control] != target.unit:
                    raise ValueError(
                        "objective targets must name controls with matching units"
                    )
            quantities.append(
                QuantitySpec(
                    self.objective.quantity_id,
                    "1",
                    "normalized_squared_command_departure",
                    "output",
                )
            )
        return replace(
            base,
            model_id="linear_process/v1",
            model=self.to_dict(),
            operating={
                **(
                    {
                        "algebraic_states": "bounded feasible completion; no actuator permissions or uniqueness asserted"
                    }
                    if self.states
                    else {}
                ),
                "response": "optimized_dispatch"
                if self.objective is not None
                else "linear_feasibility",
                "feasibility_scope": "model_and_selected_requirements",
                "stage": "operation",
                "adjustable_observations": self.observation_space.names,
                "objective": self.objective.to_dict()
                if self.objective is not None
                else "find any jointly feasible actuator/state vector; no dispatch optimum asserted"
                if self.states
                else "find any feasible controls; no dispatch optimum asserted",
            },
            quantities=tuple(quantities),
            constraints=tuple(constraints),
            numerical_policy={
                **(
                    {
                        "absolute_tolerance": self.objective.tolerance,
                        "optimality": "verified primal and global objective bounds; distinct from feasibility",
                    }
                    if self.objective is not None
                    else {}
                ),
                "arithmetic": "exact_rationals_of_declared_floats",
                "solver_tolerance": self.solver_tolerance,
                "control_bound_tolerance": 0,
                **({"state_bound_tolerance": 0} if self.states else {}),
                "constraint_tolerances": {
                    r.name: r.tolerance
                    for r in self.operating_limits + self.requirements
                },
                "infeasibility": "verified nonnegative row combination over bounded actuators and algebraic states"
                if self.states
                else "verified nonnegative row combination over bounded controls",
            },
        )

    def describe(self, claim):
        return bind_contract(
            self.base_contract, claim.domain, claim.recourse, claim.requirements
        )

    def _permissions(self, claim):
        return (
            OperatingSpec(claim.recourse)
            .validate(self.controls, self.observation_space.names)
            .point_commands(
                self.input_space.names, measurements=self.measurements is not None
            )
        )

    def capabilities(self, claim):
        try:
            self._permissions(claim)
        except ValueError as exc:
            unsupported = Capability(False, str(exc))
            return AdapterCapabilities(*(unsupported for _ in range(5)))
        return AdapterCapabilities(
            Capability(True, "Joint linear feasibility at one verified domain member."),
            Capability(
                False,
                "Selected-command derivatives are not provided; use dispatch_sensitivity() for checked local objective-value slopes.",
            ),
            Capability(
                type(claim.domain) is FiniteSet
                or type(self) is LinearProcessAdapter
                and type(claim.domain) in (BoxSet, ConvexHullSet),
                "Complete finite enumeration, or continuous BoxSet/ConvexHullSet coverage "
                "by checked generators for the built-in fixed-coefficient linear model; "
                "fixed or fully observed adjustment only, subject to max_vertices.",
            ),
            Capability(
                False, "Boundary searches with adjustable controls are not provided."
            ),
            Capability(
                False,
                "Closest-breaking searches with adjustable controls are not provided.",
            ),
        )

    def as_claim(
        self,
        domain,
        *,
        recourse=None,
        operating=None,
        requirements=None,
        controller=None,
        distance=None,
    ):
        spec = resolve_operating(
            self.controls,
            self.observation_space.names,
            operating=operating,
            recourse=recourse,
            controller=controller,
        )
        recourse = spec.policy
        if spec.controller is not None:
            from .controllers import _ControllerAdapter

            adapter = _ControllerAdapter(self, spec.controller)
            adapter.validate_binding(recourse)
            return Claim(adapter, domain, recourse, requirements, distance)
        return Claim(self, domain, recourse, requirements, distance)

    def query_options(self, claim, operation):
        return (
            {"diagnose", "relief", "backend"}
            if operation == "evaluation"
            else {"backend", "max_vertices", "relief"}
            if operation == "audit" and type(claim.domain) in (BoxSet, ConvexHullSet)
            else {"backend", "relief"}
            if operation == "audit"
            else set()
        )

    def run(self, claim, operation, realization, options):
        from ._linear_process_results import audit_result, evaluate_result

        allowed = self.query_options(claim, operation)
        if set(options) - allowed:
            return rejected_result(
                claim.contract,
                operation,
                {"query": operation, "realization": realization, **options},
                "Unknown linear process query options.",
                code="invalid_query_options",
                execution="invalid",
            )
        from ._cvxpy_backend import BackendUnavailable, validate_backend

        try:
            if (
                isinstance(self.objective, ControlTrackingObjective)
                and options.get("backend", "scipy") == "scipy"
            ):
                return rejected_result(
                    claim.contract,
                    operation,
                    {"query": operation, "realization": realization, **options},
                    'Control tracking requires backend="cvxpy" and the operange[cvxpy] extra.',
                    code="unsupported_objective_backend",
                )
            validate_backend(options.get("backend", "scipy"))
        except (ValueError, BackendUnavailable) as exc:
            return rejected_result(
                claim.contract,
                operation,
                {"query": operation, "realization": realization, **options},
                str(exc),
                code="backend_unavailable"
                if isinstance(exc, BackendUnavailable)
                else "invalid_backend",
                execution="unsupported"
                if isinstance(exc, BackendUnavailable)
                else "invalid",
            )
        if operation == "evaluation":
            return evaluate_result(claim, realization, **options)
        if operation == "audit":
            return audit_result(claim, **options)
        return rejected_result(
            claim.contract,
            operation,
            {"query": operation},
            "Unsupported linear process query.",
        )
