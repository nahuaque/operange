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
from .recourse import DecisionRule, RecoursePolicy


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
class LinearProcessAdapter(Record):
    """Find controls satisfying equipment limits and selected requirements jointly.

    Outputs use the same physical affine expressions as AffineProcessAdapter.
    Operating limits use AffineRequirement declarations but always apply, even
    when a claim selects only some service requirements. No economic or dispatch
    objective is implied: the returned feasible control vector need not be unique.
    """

    name: str
    input_space: ParameterSpace
    outputs: tuple[AffineOutput, ...]
    requirements: tuple[AffineRequirement, ...]
    controls: tuple[LinearControl, ...]
    operating_limits: tuple[AffineRequirement, ...] = ()
    solver_tolerance: float = 1e-9

    def _validate(self):
        if not self.controls or not self.requirements:
            raise ValueError("linear operation needs bounded controls and requirements")
        if not 1e-10 <= self.solver_tolerance <= 1e-4:
            raise ValueError("solver_tolerance must lie between 1e-10 and 1e-4")
        # Reuse affine declaration/unit validation, then check generated bounds.
        self.base_contract

    @cached_property
    def _affine(self):
        return AffineProcessAdapter(
            self.name,
            self.input_space,
            self.outputs,
            self.operating_limits + self.requirements,
            tuple(c.quantity for c in self.controls),
        )

    def output(self, name):
        return self._affine.output(name)

    equation_id = staticmethod(AffineProcessAdapter.equation_id)
    residual_id = staticmethod(AffineProcessAdapter.residual_id)

    @staticmethod
    def bound_id(control, side):
        return f"control_{side}:{control.name}"

    @cached_property
    def base_contract(self):
        base = self._affine.base_contract
        limits = {r.name for r in self.operating_limits}
        constraints = [
            replace(c, kind="operating_limit") if c.constraint_id in limits else c
            for c in base.constraints
        ]
        quantities = list(base.quantities)
        for control in self.controls:
            for side, expression in (
                ("lower", "lower - control"),
                ("upper", "control - upper"),
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
        return replace(
            base,
            model_id="linear_process/v1",
            model=self.to_dict(),
            operating={
                "response": "linear_feasibility",
                "feasibility_scope": "model_and_selected_requirements",
                "stage": "operation",
                "adjustable_observations": self.input_space.names,
                "objective": "find any feasible controls; no dispatch optimum asserted",
            },
            quantities=tuple(quantities),
            constraints=tuple(constraints),
            numerical_policy={
                "arithmetic": "exact_rationals_of_declared_floats",
                "solver_tolerance": self.solver_tolerance,
                "control_bound_tolerance": 0,
                "constraint_tolerances": {
                    r.name: r.tolerance
                    for r in self.operating_limits + self.requirements
                },
                "infeasibility": "verified nonnegative row combination over bounded controls",
            },
        )

    def describe(self, claim):
        return bind_contract(
            self.base_contract, claim.domain, claim.recourse, claim.requirements
        )

    def _permissions(self, claim):
        policy = claim.recourse
        if policy.mode not in ("static", "fixed"):
            raise ValueError(
                "linear operation supports fixed or fully observed static recourse"
            )
        if {r.control for r in policy.rules} != {c.name for c in self.controls}:
            raise ValueError("linear operation needs one permission for every control")
        for rule in policy.rules:
            if rule.stage != "operation":
                raise ValueError(
                    "linear controls must use the single 'operation' stage"
                )
            if rule.fixed_value is None:
                if policy.mode == "fixed" or set(rule.observes) != set(
                    self.input_space.names
                ):
                    raise ValueError(
                        "each adjustable control must observe every process input"
                    )
        return {
            r.control: r.fixed_value for r in policy.rules if r.fixed_value is not None
        }

    def capabilities(self, claim):
        try:
            self._permissions(claim)
        except ValueError as exc:
            unsupported = Capability(False, str(exc))
            return AdapterCapabilities(*(unsupported for _ in range(5)))
        return AdapterCapabilities(
            Capability(True, "Joint linear feasibility at one verified domain member."),
            Capability(
                False, "Derivatives of a selected linear dispatch are not provided."
            ),
            Capability(
                type(claim.domain) is FiniteSet,
                "Complete finite enumeration with independently adjustable, fully observed controls; "
                "continuous-domain recourse audits are not provided.",
            ),
            Capability(
                False, "Boundary searches with adjustable controls are not provided."
            ),
            Capability(
                False,
                "Closest-breaking searches with adjustable controls are not provided.",
            ),
        )

    def as_claim(self, domain, *, recourse=None, requirements=None, controller=None):
        if controller is not None:
            from .controllers import _ControllerAdapter

            adapter = _ControllerAdapter(self, controller)
            recourse = (
                adapter.controller.recourse_policy if recourse is None else recourse
            )
            adapter.validate_binding(recourse)
            return Claim(adapter, domain, recourse, requirements)
        if recourse is None:
            recourse = RecoursePolicy(
                "static",
                tuple(
                    DecisionRule(c.name, c.unit, "operation", self.input_space.names)
                    for c in self.controls
                ),
            )
        return Claim(self, domain, recourse, requirements)

    def run(self, claim, operation, realization, options):
        from ._linear_process_results import audit_result, evaluate_result

        if options:
            return rejected_result(
                claim.contract,
                operation,
                {"query": operation, "realization": realization, **options},
                "Unknown linear process query options.",
                code="invalid_query_options",
                execution="invalid",
            )
        if operation == "evaluation":
            return evaluate_result(claim, realization)
        if operation == "audit":
            return audit_result(claim)
        return rejected_result(
            claim.contract,
            operation,
            {"query": operation},
            "Unsupported linear process query.",
        )
