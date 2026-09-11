"""Existing engineering result families for fixed affine process operation."""

from fractions import Fraction

from ._geometry import LinearSupport, dot
from .claim import rejected_result
from .contract_types import (
    ConstraintCheck,
    Coverage,
    Derivative,
    Diagnostic,
    Evidence,
    EvaluationPayload,
    Measurement,
    Membership,
    QuantityValue,
    RobustnessPayload,
    SensitivityPayload,
    Witness,
    snapshot,
)
from .engineering_results import EvaluationResult, RobustnessResult, SensitivityResult
from .primitives import finite
from .domains import FiniteSet
from ._finite_audit import audit_finite
from ._numeric import normalized_score, round_down, round_up


def _requirements(claim):
    return tuple(r for r in claim.adapter.requirements if r.name in claim.requirements)


def evaluate_result(claim, realization, *, contract=None):
    model = claim.adapter
    contract = claim.contract if contract is None else contract
    request = snapshot({"query": "evaluation", "realization": realization})
    try:
        membership = claim.domain.membership(realization)
        point = claim.domain.space.validate(realization)
    except (ValueError, TypeError, KeyError) as exc:
        return rejected_result(
            contract,
            "evaluation",
            request,
            str(exc),
            code="invalid_realization",
            execution="invalid",
        )
    evidence = [
        Evidence(
            "membership",
            "domain_membership",
            "declared_domain_membership",
            "unresolved" if membership.status == "unknown" else "verified",
            details={"check": membership.to_dict()},
        )
    ]
    member = Membership(
        membership.status,
        evidence_refs=("membership",) if membership.status != "unknown" else (),
    )
    if membership.status != "inside":
        return EvaluationResult(
            contract,
            request,
            "invalid" if membership.status == "outside" else "unresolved",
            EvaluationPayload(
                "not_evaluated" if membership.status == "outside" else "unknown", member
            ),
            tuple(evidence),
            (
                Diagnostic(
                    "domain_membership",
                    "realization",
                    f"Domain membership is {membership.status}.",
                ),
            ),
        )
    try:
        fixed = model._fixed_values(claim)
        variables = {**point, **fixed}
        exact_outputs = {o.name: o._exact_value(variables) for o in model.outputs}
        outputs = {
            name: finite(float(value), name) for name, value in exact_outputs.items()
        }
        residuals = {
            r.name: round_up(r.sign * (exact_outputs[r.output] - Fraction(r.limit)))
            for r in _requirements(claim)
        }
    except (ValueError, OverflowError) as exc:
        return EvaluationResult(
            contract,
            request,
            "unresolved",
            EvaluationPayload("unknown", member),
            tuple(evidence),
            (Diagnostic("affine_arithmetic_unresolved", "response", str(exc)),),
        )
    evidence.append(
        Evidence(
            "response",
            "fixed_response",
            "direct_affine_evaluation",
            "verified",
            assumptions=(
                "caller-declared affine equations",
                "all declared controls held fixed",
            ),
            details={"fixed_controls": fixed, "response_values": outputs},
        )
    )
    values = [QuantityValue(n, v) for n, v in {**variables, **outputs}.items()]
    checks, equations = [], []
    for output in model.outputs:
        name = model.equation_id(output)
        equations.append(name)
        residual = QuantityValue(model.residual_id(name), 0)
        values.append(residual)
        checks.append(ConstraintCheck(name, "satisfied", residual, ("response",)))
    for requirement in _requirements(claim):
        value = residuals[requirement.name]
        residual = QuantityValue(model.residual_id(requirement.name), value)
        values.append(residual)
        checks.append(
            ConstraintCheck(
                requirement.name,
                "satisfied" if value <= requirement.tolerance else "violated",
                residual,
                ("response",),
            )
        )
    return EvaluationResult(
        contract,
        request,
        "completed",
        EvaluationPayload(
            "feasible",
            member,
            tuple(values),
            tuple(checks),
            included_constraints=tuple(equations),
        ),
        tuple(evidence),
    )


def sensitivity_result(
    claim,
    realization,
    *,
    inputs=None,
    outputs=None,
    operator="jacobian",
    coordinate_space="physical",
    response="fixed",
    side="two_sided",
    direction=None,
    method="analytical",
    perturbation_scope="ambient",
):
    model, contract = claim.adapter, claim.contract
    inputs = claim.domain.space.names if inputs is None else inputs
    outputs = tuple(o.name for o in model.outputs) if outputs is None else outputs
    request = snapshot(
        {
            "query": "sensitivity",
            "realization": realization,
            "inputs": inputs,
            "outputs": outputs,
            "operator": operator,
            "coordinate_space": coordinate_space,
            "response": response,
            "side": side,
            "direction": direction,
            "method": method,
            "perturbation_scope": perturbation_scope,
        }
    )
    if (
        response != "fixed"
        or method != "analytical"
        or operator not in ("jacobian", "directional")
        or perturbation_scope != "ambient"
    ):
        return rejected_result(
            contract,
            "sensitivity",
            request,
            "Only analytical ambient derivatives of the fixed affine map are supported.",
            code="unsupported_sensitivity",
        )
    try:
        for selected, known in (
            (inputs, claim.domain.space.names),
            (outputs, tuple(o.name for o in model.outputs)),
        ):
            if (
                not isinstance(selected, (tuple, list))
                or not selected
                or any(n not in known for n in selected)
                or len(set(selected)) != len(selected)
            ):
                raise ValueError(
                    "derivative coordinates must be distinct declared names"
                )
        if coordinate_space not in ("physical", "normalized") or side not in (
            "two_sided",
            "forward",
            "backward",
        ):
            raise ValueError("invalid derivative coordinate space or side")
        if operator == "directional":
            if not isinstance(direction, (tuple, list)) or len(direction) != len(
                inputs
            ):
                raise ValueError("direction needs one finite value per input")
            direction = tuple(finite(v, "direction") for v in direction)
        elif direction is not None:
            raise ValueError("a Jacobian cannot include a direction")
    except (ValueError, TypeError) as exc:
        return rejected_result(
            contract,
            "sensitivity",
            request,
            str(exc),
            code="invalid_sensitivity",
            execution="invalid",
        )
    coordinates = [claim.domain.space.coordinate(n) for n in inputs]
    if coordinate_space == "normalized" and any(
        c.nominal is None or c.scale is None for c in coordinates
    ):
        return rejected_result(
            contract,
            "sensitivity",
            request,
            "Normalized derivatives require declared nominals and scales.",
        )
    base = evaluate_result(claim, realization)
    if base.execution == "invalid":
        return rejected_result(
            contract,
            "sensitivity",
            request,
            "The requested base is not a valid domain member.",
            code="invalid_base",
            execution="invalid",
        )
    response_contract = {
        "mode": "fixed",
        "recourse_policy": claim.recourse.to_dict(),
        "control_treatment": "held fixed",
    }
    validity = {
        "scope": "global_affine_map",
        "perturbation_scope": "ambient",
        "domain_tangency_asserted": False,
    }

    def unresolved(message):
        return SensitivityResult(
            contract,
            request,
            "unresolved",
            SensitivityPayload(
                "unknown", base.ref, response_contract, validity=validity
            ),
            diagnostics=(Diagnostic("derivative_unresolved", "sensitivity", message),),
            supporting_evaluations=(base,),
        )

    if base.execution != "completed":
        return unresolved(
            "The base response or domain membership could not be verified."
        )
    try:
        scales = tuple(
            c.scale if coordinate_space == "normalized" else 1.0 for c in coordinates
        )
        matrix = []
        for name in outputs:
            coefficients = {t.variable: t.coefficient for t in model.output(name).terms}
            matrix.append(
                tuple(
                    finite(coefficients.get(n, 0) * s, "derivative")
                    for n, s in zip(inputs, scales)
                )
            )
            if any(
                coefficients.get(n, 0) != 0 and v == 0
                for n, v in zip(inputs, matrix[-1])
            ):
                raise ValueError("normalized derivative coefficient underflowed")
        values = (
            tuple(matrix)
            if operator == "jacobian"
            else tuple(dot(row, direction) for row in matrix)
        )
        input_units = tuple(
            c.unit if coordinate_space == "physical" else "1" for c in coordinates
        )
        output_units = tuple(model.output(n).unit for n in outputs)
        derivative = Derivative(
            operator,
            tuple(outputs),
            tuple(inputs),
            coordinate_space,
            values,
            tuple(tuple(f"{out}/{inp}" for inp in input_units) for out in output_units)
            if operator == "jacobian"
            else tuple((out,) for out in output_units),
            input_units,
            output_units,
            tuple(
                c.nominal if coordinate_space == "normalized" else 0
                for c in coordinates
            ),
            scales,
            side,
            direction,
        )
    except (ValueError, OverflowError) as exc:
        return unresolved(str(exc))
    return SensitivityResult(
        contract,
        request,
        "completed",
        SensitivityPayload(
            "available",
            base.ref,
            response_contract,
            derivative,
            {"name": "analytical_affine", "finite_differencing": False},
            validity,
        ),
        (
            Evidence(
                "derivative",
                "derivative",
                "declared_affine_coefficients",
                "verified",
                details={"coordinate_space": coordinate_space},
            ),
        ),
        supporting_evaluations=(base,),
    )


def _failure(claim, evaluation, affected):
    proofs = (
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
            "unique_fixed_response",
            "verified",
            assumptions=("no adjustable controls are permitted by this claim",),
            details={
                "evaluation_ref": evaluation.ref.to_dict(),
                "affected_requirements": affected,
                "recourse_policy": claim.recourse.to_dict(),
            },
        ),
    )
    return Witness(
        "fixed_policy_failure",
        (evaluation.request["realization"],),
        affected,
        ("witness_membership",),
        ("fixed_failure",),
        details={"evaluation_ref": evaluation.ref.to_dict()},
    ), proofs


def _checked_support(claim, coefficients):
    support = claim.domain.maximize_linear(coefficients)
    expected = {name: coefficients.get(name, 0) for name in claim.domain.space.names}
    if (
        not isinstance(support, LinearSupport)
        or support.domain_ref != claim.domain.ref
        or dict(support.coefficients) != expected
    ):
        raise ValueError("support result does not match the domain and objective")
    if support.status not in ("optimal", "bounded"):
        raise ValueError("domain support calculation is unresolved or unsupported")
    return support


def audit_result(claim):
    if type(claim.domain) is FiniteSet:
        return audit_finite(
            claim,
            evaluate_result,
            _failure,
            method="finite_affine_evaluation",
            scope={"arithmetic": "exact_rationals_of_declared_floats"},
        )
    model, contract = claim.adapter, claim.contract
    requirements = _requirements(claim)
    evaluations, evidence, bounds = {}, [], {}

    def evaluate(point):
        result = evaluate_result(claim, point)
        evaluations[result.result_id] = result
        return result

    seed = getattr(claim.domain, "feasible_point", None)
    if seed is not None:
        evaluate(seed)
    fixed = model._fixed_values(claim)
    nominal = claim.domain.space.nominal
    for index, requirement in enumerate(requirements):
        proof_id = f"support:{index}"
        details = {
            "requirement": requirement.name,
            "residual_scale": requirement.residual_scale,
        }
        try:
            output = model.output(requirement.output)
            coefficients = {t.variable: t.coefficient for t in output.terms}
            exact_weights = {
                c.name: requirement.sign
                * Fraction(coefficients.get(c.name, 0))
                * Fraction(c.scale)
                / Fraction(requirement.residual_scale)
                for c in claim.domain.space.coordinates
            }
            w = {
                n: finite(float(v), "normalized residual coefficient")
                for n, v in exact_weights.items()
            }
            if any(coefficients.get(n, 0) != 0 and w[n] == 0 for n in w):
                raise ValueError("normalized residual coefficient underflowed")
            baseline = requirement.sign * (
                output._exact_value({**nominal, **fixed}) - Fraction(requirement.limit)
            )
            support = _checked_support(claim, w)
            details["normalized_support"] = support.to_dict()
            candidate = evaluate(support.point)
            if (
                candidate.execution != "completed"
                or candidate.payload.membership.status != "inside"
            ):
                raise ValueError("support candidate could not be verified")
            score = round_down(normalized_score(claim.domain.space, w, support.point))
            if abs(score - support.lower) > support.tolerance:
                raise ValueError("support lower bound disagrees with its candidate")
            lower = next(
                c.residual.value
                for c in candidate.payload.constraint_checks
                if c.constraint_ref == requirement.name
            )
            # Carry the support operation's numeric tolerance into physical units.
            guard = Fraction(requirement.residual_scale) * Fraction(support.tolerance)
            correction = Fraction(0)
            correction_proofs = {}
            for name, exact_weight in exact_weights.items():
                error = exact_weight - Fraction(w[name])
                if error:
                    bound = _checked_support(claim, {name: 1 if error > 0 else -1})
                    correction += abs(error) * (
                        Fraction(bound.upper) + Fraction(bound.tolerance)
                    )
                    correction_proofs[name] = bound.to_dict()
            upper = round_up(
                baseline
                + Fraction(requirement.residual_scale)
                * (Fraction(support.upper) + correction)
                + guard
            )
            if lower > upper:
                raise ValueError(
                    "support upper bound contradicts the evaluated candidate"
                )
            bounds[requirement.name] = upper
            details.update(
                {
                    "nominal_residual": round_up(baseline),
                    "nominal_residual_exact": str(baseline),
                    "coefficient_rounding_correction": str(correction),
                    "coefficient_rounding_bounds": correction_proofs,
                    "physical_unit": output.unit,
                    "candidate_evaluation_ref": candidate.ref.to_dict(),
                }
            )
            evidence.append(
                Evidence(
                    proof_id,
                    "requirement_bound",
                    "affine_support_transformation",
                    "verified",
                    (
                        Measurement("attained_residual", lower, output.unit),
                        Measurement("residual_upper", upper, output.unit),
                    ),
                    (
                        Measurement("support_guard", round_up(guard), output.unit),
                        Measurement(
                            "requirement_tolerance", requirement.tolerance, output.unit
                        ),
                    ),
                    details=details,
                )
            )
        except (ValueError, TypeError, OverflowError, RuntimeError) as exc:
            details["message"] = str(exc)
            evidence.append(
                Evidence(
                    proof_id,
                    "requirement_bound",
                    "affine_support_transformation",
                    "unresolved",
                    details=details,
                )
            )
    witness = None
    for evaluation in evaluations.values():
        if (
            evaluation.execution != "completed"
            or evaluation.payload.membership.status != "inside"
        ):
            continue
        affected = tuple(
            c.constraint_ref
            for c in evaluation.payload.constraint_checks
            if c.assessment == "violated" and c.constraint_ref in claim.requirements
        )
        if affected:
            witness, proofs = _failure(claim, evaluation, affected)
            evidence.extend(proofs)
            break
    all_bounds = len(bounds) == len(requirements)
    passing_response = any(
        e.payload.feasibility == "feasible"
        and all(c.assessment == "satisfied" for c in e.payload.constraint_checks)
        for e in evaluations.values()
    )
    passed = (
        all_bounds
        and passing_response
        and all(bounds[r.name] <= r.tolerance for r in requirements)
    )
    verdict = "fail" if witness else "pass" if passed else "inconclusive"
    if all_bounds:
        evidence.append(
            Evidence(
                "coverage",
                "domain_coverage",
                "affine_requirement_support",
                "verified",
                details={
                    "requirement_bounds": bounds,
                    "bound_evidence": [
                        f"support:{i}" for i in range(len(requirements))
                    ],
                },
            )
        )
    return RobustnessResult(
        contract,
        {"query": "audit"},
        "completed" if verdict in ("pass", "fail") else "unresolved",
        RobustnessPayload(
            verdict,
            Coverage(
                "analytical_domain" if all_bounds else "partial",
                evaluated_support={
                    "realizations": [
                        e.request["realization"] for e in evaluations.values()
                    ]
                },
                unexplored_support={}
                if verdict != "inconclusive"
                else {
                    "reason": "bounds do not establish all requirements and no verified violation was found"
                },
                evidence_refs=("coverage",) if all_bounds else (),
            ),
            tuple(e.ref for e in evaluations.values()),
            witness,
        ),
        tuple(evidence),
        ()
        if verdict != "inconclusive"
        else (
            Diagnostic(
                "audit_inconclusive",
                "requirements",
                "An upper bound above the limit is not a violating witness; unresolved bounds cannot establish a pass.",
            ),
        ),
        tuple(evaluations.values()),
    )
