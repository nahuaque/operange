"""One fixed quadratic response over one ellipsoid: an optional-backend experiment.

Coefficient values have the response unit and act on the MODEL's normalized
coordinates. Domain normalization can differ and is transformed explicitly.
Only point evaluation and an upper-limit audit are in scope. No surrogate
approximation error or physical model validity is inferred.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum
from typing import Literal

import numpy as np
from scipy.linalg import cholesky

from ..claim import (
    AdapterCapabilities,
    Capability,
    Claim,
    bind_contract,
    rejected_result,
)
from ..contract_types import (
    ConstraintCheck,
    ConstraintSpec,
    Coverage,
    Diagnostic,
    Evidence,
    EvaluationPayload,
    Measurement,
    Membership,
    ProcessContract,
    QuantitySpec,
    QuantityValue,
    Record,
    RobustnessPayload,
    Witness,
    freeze,
    nonempty,
    snapshot,
)
from ..domains import ParameterSpace
from ..engineering_results import EvaluationResult, RobustnessResult
from ..geometries import EllipsoidSet
from ..primitives import finite
from ..recourse import RecoursePolicy
from ._quadratic_bounds import PreparedTrustRegion
from ._residopt_quadratic import BackendUnavailable, PreparedResidualSDP, load_backend


@dataclass(frozen=True)
class QuadraticModel(Record):
    name: str
    space: ParameterSpace
    unit: str
    physical_kind: str
    offset: float
    linear: Mapping[str, float]
    hessian: Mapping[str, Mapping[str, float]]
    limit: float
    provenance: str
    tolerance: float = 1e-6
    residual_scale: float = 1.0

    def _validate(self):
        for name in ("name", "unit", "physical_kind", "provenance"):
            nonempty(getattr(self, name), name)
        self.space.normalize(self.space.nominal)
        names = set(self.space.names)
        if (
            set(self.linear) != names
            or set(self.hessian) != names
            or any(
                not isinstance(row, Mapping) or set(row) != names
                for row in self.hessian.values()
            )
        ):
            raise ValueError("quadratic coefficients must name every model coordinate")
        linear = {n: finite(self.linear[n], n) for n in self.space.names}
        hessian = {
            a: {b: finite(self.hessian[a][b], "hessian") for b in self.space.names}
            for a in self.space.names
        }
        if any(hessian[a][b] != hessian[b][a] for a in names for b in names):
            raise ValueError("quadratic hessian must be symmetric")
        if self.tolerance < 0 or self.residual_scale <= 0:
            raise ValueError(
                "tolerance must be nonnegative and residual_scale positive"
            )
        object.__setattr__(self, "linear", freeze(linear))
        object.__setattr__(self, "hessian", freeze(hessian))

    def arrays(self):
        names = self.space.names
        return np.array([[self.hessian[a][b] for b in names] for a in names]), np.array(
            [self.linear[n] for n in names]
        )

    def evaluate(self, point):
        z = self.space.normalize(point)
        return finite(
            fsum(
                [
                    self.offset,
                    *(self.linear[n] * z[n] for n in z),
                    *(0.5 * self.hessian[a][b] * z[a] * z[b] for a in z for b in z),
                ]
            ),
            "quadratic response",
        )


def ball_problem(model, domain):
    """Return normalized H, b, constant residual and a map to physical points."""
    names = model.space.names
    H, linear = model.arrays()
    shape = np.array([[domain.shape[a][b] for b in names] for a in names])
    B = domain.radius * cholesky(shape, lower=True)
    domain_coordinates = [domain.space.coordinate(n) for n in names]
    delta = np.array(
        [
            (c.nominal - model.space.coordinate(c.name).nominal)
            / model.space.coordinate(c.name).scale
            for c in domain_coordinates
        ]
    )
    ratios = np.array(
        [c.scale / model.space.coordinate(c.name).scale for c in domain_coordinates]
    )
    transform = ratios[:, None] * B
    A = transform.T @ H @ transform / model.residual_scale
    A = 0.5 * (A + A.T)
    b = transform.T @ (linear + H @ delta) / model.residual_scale
    constant = (
        model.evaluate(domain.space.nominal) - model.limit
    ) / model.residual_scale
    if (
        not np.all(np.isfinite(A))
        or not np.all(np.isfinite(b))
        or not np.isfinite(constant)
    ):
        raise ValueError("quadratic normalization overflowed")

    def physical(y):
        z = B @ np.asarray(y)
        return domain.space.denormalize(dict(zip(names, (float(v) for v in z))))

    return A, b, constant, physical


@dataclass(frozen=True)
class QuadraticProcessAdapter(Record):
    model: QuadraticModel
    backend: Literal["trust_region", "residopt_sdp"] = "residopt_sdp"
    solver: str = "CLARABEL"
    bound_tolerance: float = 1e-7

    def _validate(self):
        nonempty(self.solver, "solver")
        if not 1e-10 <= self.bound_tolerance <= 1e-4:
            raise ValueError("bound_tolerance must lie between 1e-10 and 1e-4")
        self.base_contract

    @property
    def input_space(self):
        return self.model.space

    @property
    def base_contract(self):
        m = self.model
        quantities = [
            QuantitySpec(c.name, c.unit, "declared_process_input", "input")
            for c in m.space.coordinates
        ]
        quantities.extend(
            QuantitySpec(n, m.unit, m.physical_kind, role)
            for n, role in (
                ("response", "output"),
                ("definition_residual", "residual"),
                ("limit_residual", "residual"),
            )
        )
        return ProcessContract(
            "quadratic_process_experiment/v1",
            m.to_dict(),
            {},
            {"kind": "unbound", "space": m.space.to_dict()},
            {
                "response": "fixed_quadratic",
                "feasibility_scope": "declared_quadratic_equation_only",
                "model_validity": "no approximation error or physical validation inferred",
            },
            tuple(quantities),
            (
                ConstraintSpec(
                    "quadratic_definition",
                    "definition_residual",
                    "equation",
                    "eq",
                    "response - declared quadratic expression",
                    0,
                ),
                ConstraintSpec(
                    "response_limit",
                    "limit_residual",
                    "requirement",
                    "le",
                    "response - limit",
                    m.tolerance,
                ),
            ),
            {
                "arithmetic": "float64",
                "backend": self.backend,
                "solver": self.solver if self.backend == "residopt_sdp" else None,
                "normalized_bound_tolerance": self.bound_tolerance,
                "requirement_tolerance": m.tolerance,
            },
        )

    def describe(self, claim):
        return bind_contract(
            self.base_contract, claim.domain, claim.recourse, claim.requirements
        )

    def as_claim(self, domain):
        return Claim(self, domain, RecoursePolicy("fixed", ()))

    def capabilities(self, claim):
        fixed = claim.recourse == RecoursePolicy("fixed", ())
        audit = fixed and type(claim.domain) is EllipsoidSet and claim.domain.radius > 0
        no = Capability(
            False,
            "This experiment supports fixed point evaluation and an upper-limit audit only.",
        )
        return AdapterCapabilities(
            Capability(
                fixed, "Direct quadratic response with no operating adjustments."
            ),
            no,
            Capability(
                audit,
                "Single positive-radius ellipsoid under fixed operation; intersections, recourse and degenerate domains are unsupported.",
            ),
            no,
            no,
        )

    def run(self, claim, operation, realization, options):
        if options:
            return rejected_result(
                claim.contract,
                operation,
                {"realization": realization, **options},
                "Unknown experiment query options.",
                code="invalid_query_options",
                execution="invalid",
            )
        if operation == "evaluation":
            return _evaluate(claim, realization)
        if operation == "audit":
            return _audit(claim)
        return rejected_result(
            claim.contract, operation, {}, "Unsupported experiment query."
        )


def _evaluate(claim, point):
    m, contract = claim.adapter.model, claim.contract
    request = snapshot({"query": "evaluation", "realization": point})
    try:
        check = claim.domain.membership(point)
    except (ValueError, TypeError) as exc:
        return rejected_result(
            contract,
            "evaluation",
            request,
            str(exc),
            code="invalid_realization",
            execution="invalid",
        )
    proof = Evidence(
        "membership",
        "domain_membership",
        "declared_domain_membership",
        "unresolved" if check.status == "unknown" else "verified",
        details={"check": check.to_dict()},
    )
    member = Membership(
        check.status, evidence_refs=("membership",) if check.status != "unknown" else ()
    )
    if check.status != "inside":
        return EvaluationResult(
            contract,
            request,
            "invalid" if check.status == "outside" else "unresolved",
            EvaluationPayload(
                "not_evaluated" if check.status == "outside" else "unknown", member
            ),
            (proof,),
            (Diagnostic("domain_membership", "realization", check.status),),
        )
    try:
        value = m.evaluate(point)
        residual = finite(value - m.limit, "requirement residual")
    except (ValueError, OverflowError) as exc:
        return EvaluationResult(
            contract,
            request,
            "unresolved",
            EvaluationPayload("unknown", member),
            (proof,),
            (Diagnostic("quadratic_arithmetic", "response", str(exc)),),
        )
    response = Evidence(
        "response",
        "fixed_response",
        "direct_quadratic_evaluation",
        "verified",
        details={"normalization": m.space.to_dict()},
    )
    return EvaluationResult(
        contract,
        request,
        "completed",
        EvaluationPayload(
            "feasible",
            member,
            tuple(
                QuantityValue(n, v)
                for n, v in {
                    **check.realization,
                    "response": value,
                    "definition_residual": 0,
                    "limit_residual": residual,
                }.items()
            ),
            (
                ConstraintCheck(
                    "quadratic_definition",
                    "satisfied",
                    QuantityValue("definition_residual", 0),
                    ("response",),
                ),
                ConstraintCheck(
                    "response_limit",
                    "satisfied" if residual <= m.tolerance else "violated",
                    QuantityValue("limit_residual", residual),
                    ("response",),
                ),
            ),
            included_constraints=("quadratic_definition",),
        ),
        (proof, response),
    )


def _audit(claim):
    adapter, domain = claim.adapter, claim.domain
    m = adapter.model
    if adapter.backend == "residopt_sdp":
        try:
            load_backend()
        except BackendUnavailable as exc:
            return rejected_result(
                claim.contract,
                "audit",
                {"query": "audit"},
                str(exc),
                code="optional_backend_unavailable",
            )
    evaluations = {}

    def evaluate(point):
        result = _evaluate(claim, point)
        evaluations[result.result_id] = result
        return result

    evaluate(domain.feasible_point)
    upper, evidence = None, []
    details = {"backend": adapter.backend}
    try:
        A, b, constant, physical = ball_problem(m, domain)
        reference = PreparedTrustRegion(A).maximize(
            b, tolerance=adapter.bound_tolerance
        )
        details["independent_reference"] = {
            "lower": reference.lower,
            "upper": reference.upper,
            "point": reference.point,
            **reference.details,
        }
        point = domain._interior_candidate(
            physical(reference.point), domain.feasible_point
        )
        candidate = evaluate(point)
        if candidate.execution != "completed":
            raise RuntimeError("physical candidate could not be verified")
        attained = next(
            c.residual.value
            for c in candidate.payload.constraint_checks
            if c.constraint_ref == "response_limit"
        )
        # Verify normalization at the unmodified candidate as well as evaluating
        # the physical feasible witness after any roundoff shrink.
        transformation_error = abs(
            (m.evaluate(physical(reference.point)) - m.limit) / m.residual_scale
            - (constant + reference.lower)
        )
        if transformation_error > adapter.bound_tolerance:
            raise RuntimeError("physical and normalized quadratic responses disagree")
        selected_upper = reference.upper
        if adapter.backend == "residopt_sdp":
            selected_upper, backend_details = PreparedResidualSDP(
                A, solver=adapter.solver
            ).upper_bound(b, tolerance=adapter.bound_tolerance)
            details["residopt"] = backend_details
            if selected_upper < reference.lower - adapter.bound_tolerance:
                raise RuntimeError(
                    "SDP bound contradicts the independently evaluated candidate"
                )
            # Both methods must support a passing claim in this experiment.
            selected_upper = max(selected_upper, reference.upper)
        rounding = (
            128 * np.finfo(float).eps * max(1.0, abs(constant), abs(selected_upper))
        )
        upper = finite(
            m.residual_scale * fsum((constant, selected_upper, rounding)),
            "residual upper bound",
        )
        if attained > upper:
            upper = None
            raise RuntimeError("upper bound contradicts the physical witness")
        details.update(
            {
                "ball_hessian": A.tolist(),
                "ball_linear": b.tolist(),
                "constant_residual": constant,
                "normalization_error": transformation_error,
                "residual_scale": m.residual_scale,
                "evaluation_ref": candidate.ref.to_dict(),
            }
        )
        evidence.append(
            Evidence(
                "bound",
                "requirement_bound",
                "checked_quadratic_ellipsoid_bound",
                "verified",
                (
                    Measurement("attained_residual", attained, m.unit),
                    Measurement("residual_upper", upper, m.unit),
                ),
                (Measurement("requirement_tolerance", m.tolerance, m.unit),),
                details=details,
            )
        )
    except BackendUnavailable as exc:
        return rejected_result(
            claim.contract,
            "audit",
            {"query": "audit"},
            str(exc),
            code="optional_backend_unavailable",
        )
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        upper = None
        details["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        evidence.append(
            Evidence(
                "bound",
                "requirement_bound",
                "checked_quadratic_ellipsoid_bound",
                "unresolved",
                details=details,
            )
        )
    witness = None
    for evaluation in evaluations.values():
        if (
            evaluation.execution == "completed"
            and evaluation.payload.constraint_checks[-1].assessment == "violated"
        ):
            evidence.extend(
                (
                    Evidence(
                        "witness_membership",
                        "domain_membership",
                        "evaluated_domain_member",
                        "verified",
                        details={"evaluation_ref": evaluation.ref.to_dict()},
                    ),
                    Evidence(
                        "fixed_failure",
                        "recourse_infeasibility",
                        "unique_fixed_quadratic_response",
                        "verified",
                        details={
                            "evaluation_ref": evaluation.ref.to_dict(),
                            "scope": "no operating adjustments permitted",
                        },
                    ),
                )
            )
            witness = Witness(
                "fixed_policy_failure",
                (evaluation.request["realization"],),
                ("response_limit",),
                ("witness_membership",),
                ("fixed_failure",),
            )
            break
    verdict = (
        "fail"
        if witness
        else "pass"
        if upper is not None and upper <= m.tolerance
        else "inconclusive"
    )
    if upper is not None:
        evidence.append(
            Evidence(
                "coverage",
                "domain_coverage",
                "single_ellipsoid_quadratic_bound",
                "verified",
                details={"bound_evidence": "bound"},
            )
        )
    return RobustnessResult(
        claim.contract,
        {"query": "audit"},
        "unresolved" if verdict == "inconclusive" else "completed",
        RobustnessPayload(
            verdict,
            Coverage(
                "analytical_domain" if upper is not None else "partial",
                evaluated_support={
                    "realizations": [
                        e.request["realization"] for e in evaluations.values()
                    ]
                },
                unexplored_support={"reason": "unresolved quadratic requirement"}
                if verdict == "inconclusive"
                else {},
                evidence_refs=("coverage",) if upper is not None else (),
            ),
            tuple(e.ref for e in evaluations.values()),
            witness,
        ),
        tuple(evidence),
        ()
        if verdict != "inconclusive"
        else (
            Diagnostic(
                "quadratic_bound_unresolved",
                "audit",
                "No verified violating member or passing global bound was established.",
            ),
        ),
        tuple(evaluations.values()),
    )
