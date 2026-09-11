"""Caller-declared affine process responses under fixed operating decisions."""

from dataclasses import dataclass
from fractions import Fraction
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
    unique,
)
from .domains import FiniteSet, ParameterSpace
from .distance import NormalizedLInf
from .polytope import PolytopeSet
from ._numeric import exact_dot
from .primitives import BoxSet, finite
from .recourse import RecoursePolicy


@dataclass(frozen=True)
class AffineTerm(Record):
    """A coefficient on a physical input/control, with explicit output/input units."""

    variable: str
    coefficient: float
    unit: str

    def _validate(self):
        nonempty(self.variable, "variable")
        nonempty(self.unit, "coefficient unit")


@dataclass(frozen=True)
class AffineOutput(Record):
    name: str
    unit: str
    physical_kind: str
    offset: float
    terms: tuple[AffineTerm, ...]
    provenance: str

    def _validate(self):
        for name in ("name", "unit", "physical_kind", "provenance"):
            nonempty(getattr(self, name), name)
        unique(tuple(t.variable for t in self.terms), "affine variables")

    def evaluate(self, values):
        return finite(float(self._exact_value(values)), self.name)

    def _exact_value(self, values):
        return Fraction(self.offset) + exact_dot(
            (t.coefficient for t in self.terms),
            (values[t.variable] for t in self.terms),
        )


@dataclass(frozen=True)
class AffineRequirement(Record):
    """Output <= or >= limit, all numeric fields in the output's physical unit.

    residual_scale converts the signed requirement residual to a dimensionless
    linear-support objective; it does not change the requirement or its tolerance.
    """

    name: str
    output: str
    limit: float
    relation: Literal["le", "ge"] = "le"
    tolerance: float = 1e-8
    residual_scale: float = 1.0

    def _validate(self):
        nonempty(self.name, "requirement name")
        nonempty(self.output, "requirement output")
        if self.tolerance < 0 or self.residual_scale <= 0:
            raise ValueError(
                "requirement tolerance must be nonnegative and residual_scale positive"
            )

    @property
    def sign(self):
        return 1 if self.relation == "le" else -1

    def residual(self, value):
        return finite(self.sign * (value - self.limit), "requirement residual")


@dataclass(frozen=True)
class AffineProcessAdapter(Record):
    name: str
    input_space: ParameterSpace
    outputs: tuple[AffineOutput, ...]
    requirements: tuple[AffineRequirement, ...]
    controls: tuple[QuantitySpec, ...] = ()

    def _validate(self):
        nonempty(self.name, "model name")
        if not self.outputs or not self.requirements:
            raise ValueError("affine model needs outputs and requirements")
        unique(tuple(r.name for r in self.requirements), "requirement names")
        variables = {c.name: c.unit for c in self.input_space.coordinates}
        for control in self.controls:
            if control.role != "control" or control.axes:
                raise ValueError("affine controls must be scalar control quantities")
            if control.quantity_id in variables:
                raise ValueError("input and control identities must be distinct")
            variables[control.quantity_id] = control.unit
        for output in self.outputs:
            for term in output.terms:
                if term.variable not in variables:
                    raise ValueError(
                        "affine terms must reference declared inputs or controls"
                    )
                if term.unit != f"{output.unit}/{variables[term.variable]}":
                    raise ValueError(
                        "affine coefficient units must match output/input units; conversion must be explicit"
                    )
        for requirement in self.requirements:
            self.output(requirement.output)
        # Validate generated quantity/constraint identities, including collisions.
        self.base_contract

    def output(self, name):
        for output in self.outputs:
            if output.name == name:
                return output
        raise ValueError(f"unknown affine output: {name}")

    @staticmethod
    def equation_id(output):
        return f"affine_definition:{output.name}"

    @staticmethod
    def residual_id(name):
        return f"residual:{name}"

    @property
    def base_contract(self):
        quantities = [
            QuantitySpec(c.name, c.unit, "declared_process_input", "input")
            for c in self.input_space.coordinates
        ]
        quantities.extend(self.controls)
        constraints = []
        for output in self.outputs:
            quantities.append(
                QuantitySpec(output.name, output.unit, output.physical_kind, "output")
            )
            name = self.equation_id(output)
            quantities.append(
                QuantitySpec(
                    self.residual_id(name),
                    output.unit,
                    output.physical_kind,
                    "residual",
                )
            )
            constraints.append(
                ConstraintSpec(
                    name,
                    self.residual_id(name),
                    "equation",
                    "eq",
                    "output - declared affine expression",
                    0,
                )
            )
        for requirement in self.requirements:
            output = self.output(requirement.output)
            quantities.append(
                QuantitySpec(
                    self.residual_id(requirement.name),
                    output.unit,
                    output.physical_kind,
                    "residual",
                )
            )
            constraints.append(
                ConstraintSpec(
                    requirement.name,
                    self.residual_id(requirement.name),
                    "requirement",
                    "le",
                    f"{requirement.sign} * ({requirement.output} - limit)",
                    requirement.tolerance,
                )
            )
        return ProcessContract(
            "affine_process/v1",
            self.to_dict(),
            {},
            {"kind": "unbound", "space": self.input_space.to_dict()},
            {
                "response": "fixed_affine",
                "feasibility_scope": "declared_affine_equations_only",
            },
            tuple(quantities),
            tuple(constraints),
            {
                "arithmetic": "exact_rational_evaluation_and_directed_support_bounds",
                "requirement_tolerances": {
                    r.name: r.tolerance for r in self.requirements
                },
            },
        )

    def describe(self, claim):
        return bind_contract(
            self.base_contract, claim.domain, claim.recourse, claim.requirements
        )

    def _fixed_values(self, claim):
        policy = claim.recourse
        if policy.mode != "fixed" or {r.control for r in policy.rules} != {
            c.quantity_id for c in self.controls
        }:
            raise ValueError(
                "affine operation requires a fixed value for every declared control and no additional controls"
            )
        if any(r.fixed_value is None or r.observes for r in policy.rules):
            raise ValueError(
                "every affine control must be fixed before uncertainty is observed"
            )
        return {r.control: r.fixed_value for r in policy.rules}

    def capabilities(self, claim):
        try:
            self._fixed_values(claim)
        except ValueError as exc:
            capability = Capability(False, str(exc))
            return AdapterCapabilities(*(capability for _ in range(5)))
        support = claim.domain.capabilities.linear_optimization and callable(
            getattr(claim.domain, "maximize_linear", None)
        )
        normalized = all(
            c.nominal is not None and c.scale is not None
            for c in claim.domain.space.coordinates
        )
        finite_domain = type(claim.domain) is FiniteSet
        search = Capability(
            type(claim.domain) in (BoxSet, PolytopeSet)
            and type(claim.distance) is NormalizedLInf,
            "Fixed affine threshold searches over a BoxSet or PolytopeSet require an explicit NormalizedLInf distance; "
            "each selected requirement is a separate search branch.",
        )
        return AdapterCapabilities(
            Capability(
                True,
                "Direct affine response under the declared fixed operation at a verified domain member.",
            ),
            Capability(
                True,
                "Analytical ambient Jacobian/directional derivatives of the fixed affine response.",
            ),
            Capability(
                finite_domain or (support and normalized),
                "Complete finite enumeration in physical coordinates."
                if finite_domain
                else "Requirement support bounds over the declared domain."
                if support and normalized
                else "Audit requires normalized linear support; membership alone is insufficient.",
            ),
            search,
            search,
        )

    def as_claim(self, domain, *, recourse=None, requirements=None, distance=None):
        if recourse is None:
            if self.controls:
                raise ValueError(
                    "supply fixed recourse for the declared affine controls"
                )
            recourse = RecoursePolicy("fixed", ())
        return Claim(self, domain, recourse, requirements, distance)

    def run(self, claim, operation, realization, options):
        from .affine_results import audit_result, evaluate_result, sensitivity_result

        allowed = (
            {
                "inputs",
                "outputs",
                "operator",
                "coordinate_space",
                "response",
                "side",
                "direction",
                "method",
                "perturbation_scope",
            }
            if operation == "sensitivity"
            else {"distance_tolerance"}
            if operation == "boundary"
            else {"violation_margins", "distance_tolerance"}
            if operation == "breaking"
            else set()
        )
        if set(options) - allowed:
            return rejected_result(
                claim.contract,
                operation,
                {"realization": realization, **options},
                "Unknown affine query options.",
                code="invalid_query_options",
                execution="invalid",
            )
        if operation == "evaluation":
            return evaluate_result(claim, realization)
        if operation == "sensitivity":
            return sensitivity_result(claim, realization, **options)
        if operation == "audit":
            return audit_result(claim)
        if operation in ("boundary", "breaking"):
            from ._affine_distance_results import threshold_result

            return threshold_result(claim, boundary=operation == "boundary", **options)
        return rejected_result(
            claim.contract,
            operation,
            {"realization": realization, **options},
            "Unsupported affine query.",
        )
