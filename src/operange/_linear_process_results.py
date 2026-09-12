"""Portable results for bounded linear dispatch and checked recourse audits."""

from fractions import Fraction
from dataclasses import replace
from functools import partial

from ._finite_audit import audit_finite
from ._linear_recourse import RecourseSolution, compile_system, solve_system
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
from .objectives import ControlTrackingObjective


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


def evaluate_result(
    claim,
    realization,
    *,
    contract=None,
    diagnose=False,
    relief=None,
    backend="scipy",
    _controls=None,
):
    contract = claim.contract if contract is None else contract
    request = snapshot({"query": "evaluation", "realization": realization})
    if diagnose or relief is not None:
        request = snapshot({**request, "diagnose": diagnose, "relief": relief})
    if backend != "scipy":
        request = snapshot({**request, "backend": backend})
    if _controls is not None:
        request = snapshot(
            {
                **request,
                "dispatch_source": "shared_relief_candidate",
                "controls": _controls,
            }
        )
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
    if (
        relief is not None
        and relief.get("objective") == "quadratic"
        and backend != "cvxpy"
    ):
        return rejected_result(
            contract,
            "evaluation",
            request,
            "Quadratic joint relief requires backend='cvxpy'.",
            code="relief_backend_unavailable",
            execution="unsupported",
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
    objective = None
    try:
        system = compile_system(claim, point)
        if _controls is not None:
            checked = system.checked_values(_controls)
            solution = RecourseSolution(
                "feasible" if checked is not None else "unknown",
                checked,
                None,
                [],
                "Shared relief commands independently checked against the changed model; dispatch optimality is not asserted.",
            )
            if claim.adapter.objective is not None:
                from ._dispatch_objective import compile_objective

                polynomial = compile_objective(claim.adapter, point, system)
                objective_lower, objective_proof = polynomial.bound(
                    system, [0.0] * len(system.rows)
                )
        elif claim.adapter.objective is None:
            solution = solve_system(
                system,
                claim.adapter.solver_tolerance,
                **({"backend": backend} if backend != "scipy" else {}),
            )
        else:
            from ._dispatch_objective import compile_objective, solve_dispatch

            polynomial = compile_objective(claim.adapter, point, system)
            solution, objective_lower, objective_proof = solve_dispatch(
                system,
                polynomial,
                claim.adapter.solver_tolerance,
                backend=backend,
            )
        message = solution.message
        if solution.attempts:
            evidence.append(
                Evidence(
                    "solver",
                    "solver_termination",
                    "cvxpy_prepared_candidates"
                    if backend == "cvxpy"
                    else "scipy_highs_candidates",
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
            if claim.adapter.objective is not None:
                from ._dispatch_objective import objective_result

                objective, proof = objective_result(
                    claim.adapter,
                    point,
                    system,
                    solution.controls,
                    polynomial,
                    objective_lower,
                    objective_proof,
                )
                evidence.append(proof)
                if isinstance(claim.adapter.objective, ControlTrackingObjective):
                    values.append(
                        QuantityValue(objective.quantity_ref, objective.attained_value)
                    )
    except (ValueError, OverflowError) as exc:
        message = str(exc)
    # Optional diagnosis does not alter the physical verdict or invent dispatch
    # values for the original infeasible contract.
    if feasibility == "infeasible" and diagnose:
        evidence.append(
            conflict_evidence(
                system, solution, claim.adapter.solver_tolerance, backend=backend
            )
        )
    if feasibility in ("feasible", "infeasible") and relief is not None:
        evidence.append(relief_evidence(claim, point, system, relief, backend=backend))
    diagnostics = []
    if feasibility == "unknown":
        diagnostics.append(
            Diagnostic("linear_recourse_unresolved", "response", message)
        )
    elif feasibility == "feasible" and claim.adapter.objective is not None:
        if objective is None or objective.optimality != "verified":
            diagnostics.append(
                Diagnostic(
                    "dispatch_optimality_unresolved",
                    "operating_objective",
                    message
                    if objective is None
                    else "A feasible dispatch and global objective bounds were verified, but the optimality gap remains open.",
                )
            )
    return EvaluationResult(
        contract,
        request,
        "unresolved" if feasibility == "unknown" else "completed",
        EvaluationPayload(
            feasibility,
            member,
            tuple(values),
            checks,
            objective=objective,
            included_constraints=tuple(c.constraint_id for c in included),
        ),
        tuple(evidence),
        tuple(diagnostics),
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


def audit_result(
    claim, *, backend="scipy", max_vertices=256, relief=None, _evaluate=None
):
    from .domains import FiniteSet

    if relief is not None:
        from ._shared_relief import audit_with_relief

        return audit_with_relief(
            claim, relief, backend=backend, max_vertices=max_vertices
        )
    evaluate = evaluate_result if _evaluate is None else _evaluate
    if type(claim.domain) is not FiniteSet:
        from ._linear_vertex_audit import audit_vertices

        return audit_vertices(
            claim, backend=backend, max_vertices=max_vertices, evaluate=evaluate
        )
    result = audit_finite(
        claim,
        partial(evaluate, backend=backend),
        _failure,
        method="complete_finite_linear_recourse",
        scope={
            "controls": "fixed as declared or independently adjustable after all inputs are observed",
            "coverage": "declared finite realizations only",
        },
        failure_constraints=_failed_constraints,
        diagnostic="A scenario has neither a verified dispatch nor a verified infeasibility certificate.",
    )
    return (
        replace(result, request={"query": "audit", "backend": backend})
        if backend != "scipy"
        else result
    )
