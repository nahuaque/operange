"""Portable results for bounded linear dispatch and finite recourse audits."""

from fractions import Fraction

from ._finite_audit import audit_finite
from ._linear_recourse import compile_system, solve_system
from ._numeric import round_up
from .claim import rejected_result
from .contract_types import (
    ConstraintCheck,
    Diagnostic,
    Evidence,
    EvaluationPayload,
    Membership,
    QuantityValue,
    Witness,
    snapshot,
)
from .engineering_results import EvaluationResult
from .primitives import finite


def _included(claim, contract):
    return tuple(
        c
        for c in contract.constraints
        if c.kind != "requirement" or c.constraint_id in claim.requirements
    )


def _physical_response(model, requirements, point, controls):
    variables = {**point, **controls}
    outputs = {o.name: o._exact_value(variables) for o in model.outputs}
    values = {**variables, **{n: finite(float(v), n) for n, v in outputs.items()}}
    residuals = {model.equation_id(o): 0.0 for o in model.outputs}
    for limit in model.operating_limits + model.requirements:
        if limit in model.operating_limits or limit.name in requirements:
            residuals[limit.name] = round_up(
                limit.sign * (outputs[limit.output] - Fraction(limit.limit))
            )
    for control in model.controls:
        value = Fraction(controls[control.name])
        residuals[model.bound_id(control, "lower")] = round_up(
            Fraction(control.lower) - value
        )
        residuals[model.bound_id(control, "upper")] = round_up(
            value - Fraction(control.upper)
        )
    values.update({model.residual_id(n): v for n, v in residuals.items()})
    return values, residuals


def evaluate_result(claim, realization, *, contract=None, diagnose=False, relief=None):
    contract = claim.contract if contract is None else contract
    request = snapshot({"query": "evaluation", "realization": realization})
    if diagnose or relief is not None:
        request = snapshot({**request, "diagnose": diagnose, "relief": relief})
    from ._linear_diagnosis import conflict_evidence, relief_evidence, validate_options

    try:
        validate_options(claim.adapter, diagnose, relief)
    except (ValueError, TypeError) as exc:
        return rejected_result(
            contract,
            "evaluation",
            request,
            str(exc),
            code="invalid_diagnosis_options",
            execution="invalid",
        )
    try:
        check = claim.domain.membership(realization)
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
            "unresolved" if check.status == "unknown" else "verified",
            details={"check": check.to_dict()},
        )
    ]
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
            tuple(evidence),
            (
                Diagnostic(
                    "domain_membership",
                    "realization",
                    f"Domain membership is {check.status}.",
                ),
            ),
        )
    included = _included(claim, contract)
    values = [QuantityValue(n, v) for n, v in point.items()]
    checks = tuple(ConstraintCheck(c.constraint_id, "unknown") for c in included)
    feasibility, message = "unknown", "Linear recourse was not resolved."
    try:
        system = compile_system(claim, point)
        solution = solve_system(system, claim.adapter.solver_tolerance)
        message = solution.message
        if solution.attempts:
            evidence.append(
                Evidence(
                    "solver",
                    "solver_termination",
                    "scipy_highs_candidates",
                    "verified"
                    if all(a["status"] == "optimal" for a in solution.attempts)
                    else "unresolved",
                    details={
                        "attempts": solution.attempts,
                        "scope": "numerical candidate generation only; semantic conclusions require separate exact checks",
                    },
                )
            )
        if solution.feasibility == "infeasible":
            evidence.append(
                Evidence(
                    "recourse",
                    "recourse_infeasibility",
                    "exact_bounded_linear_contradiction",
                    "verified",
                    details=solution.certificate,
                )
            )
            feasibility = "infeasible"
        elif solution.feasibility == "feasible":
            physical, residuals = _physical_response(
                claim.adapter, claim.requirements, point, solution.controls
            )
            if any(residuals[c.constraint_id] > c.tolerance for c in included):
                raise ValueError(
                    "Returned physical controls failed an included constraint."
                )
            evidence.append(
                Evidence(
                    "response",
                    "linear_recourse_feasibility",
                    "exact_physical_residual_check",
                    "verified",
                    details={
                        "arithmetic": "exact_rationals_of_declared_floats; residuals rounded upward for export",
                        "fixed_controls": system.fixed,
                        "scope": "all included constraints jointly satisfied; no dispatch optimum or unique controller asserted",
                    },
                )
            )
            values = [QuantityValue(n, v) for n, v in physical.items()]
            checks = tuple(
                ConstraintCheck(
                    c.constraint_id,
                    "satisfied",
                    QuantityValue(c.residual_ref, residuals[c.constraint_id]),
                    ("response",),
                )
                for c in included
            )
            feasibility = "feasible"
    except (ValueError, OverflowError) as exc:
        message = str(exc)
    # Optional diagnosis does not alter the physical verdict or invent dispatch
    # values for the original infeasible contract.
    if feasibility == "infeasible" and diagnose:
        evidence.append(
            conflict_evidence(system, solution, claim.adapter.solver_tolerance)
        )
    if feasibility in ("feasible", "infeasible") and relief is not None:
        evidence.append(relief_evidence(claim, point, system, relief))
    return EvaluationResult(
        contract,
        request,
        "unresolved" if feasibility == "unknown" else "completed",
        EvaluationPayload(
            feasibility,
            member,
            tuple(values),
            checks,
            included_constraints=tuple(c.constraint_id for c in included),
        ),
        tuple(evidence),
        (Diagnostic("linear_recourse_unresolved", "response", message),)
        if feasibility == "unknown"
        else (),
    )


def _failed_constraints(claim, result):
    if result.payload.feasibility != "infeasible":
        return ()
    # This is a joint infeasibility certificate. It does not assert that every
    # individual requirement is impossible, or identify a minimal conflict.
    return tuple(
        c.constraint_id
        for c in _included(claim, result.contract)
        if c.kind != "equation"
    )


def _failure(claim, result, affected):
    return Witness(
        "fixed_policy_failure"
        if claim.recourse.mode == "fixed"
        else "individual_infeasibility",
        (result.request["realization"],),
        affected,
        ("membership",),
        ("recourse",),
        details={
            "evaluation_ref": result.ref.to_dict(),
            "scope": "joint infeasibility of model and selected requirements at this realization",
            "minimal_conflict": False,
        },
    ), tuple(e for e in result.evidence if e.evidence_id in ("membership", "recourse"))


def audit_result(claim):
    return audit_finite(
        claim,
        evaluate_result,
        _failure,
        method="complete_finite_linear_recourse",
        scope={
            "controls": "fixed as declared or independently adjustable after all inputs are observed",
            "coverage": "declared finite realizations only",
        },
        failure_constraints=_failed_constraints,
        diagnostic="A scenario has neither a verified dispatch nor a verified infeasibility certificate.",
    )
