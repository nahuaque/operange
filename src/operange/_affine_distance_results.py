"""Nearest affine threshold results over the union of selected requirements."""

from collections.abc import Mapping
from fractions import Fraction

from ._affine_distance import compile_problem, solve_branch
from ._numeric import round_down, round_up
from .affine_results import _failure, _requirements, evaluate_result
from .claim import rejected_result
from .contract_types import (
    Bound,
    Coverage,
    Diagnostic,
    Evidence,
    QuantityValue,
    RobustnessPayload,
    Search,
    snapshot,
)
from .engineering_results import RobustnessResult
from .primitives import finite


def threshold_result(
    claim,
    *,
    boundary=False,
    violation_margins=None,
    distance_tolerance=1e-8,
    requirements=None,
    compile_branch=None,
    evaluate_response=None,
    failure=None,
    failure_constraints=None,
    unit=None,
    residual_id=None,
):
    operation = "boundary" if boundary else "breaking"
    contract = claim.contract
    request = snapshot(
        {
            "query": operation,
            "violation_margins": violation_margins,
            "distance_tolerance": distance_tolerance,
            "distance": claim.distance.to_manifest(),
        }
    )
    if requirements is not None:
        request = snapshot(
            {**request, "constraints": tuple(r.name for r in requirements)}
        )
    requirements = _requirements(claim) if requirements is None else requirements
    compile_branch = compile_problem if compile_branch is None else compile_branch
    evaluate_response = (
        evaluate_result if evaluate_response is None else evaluate_response
    )
    failure = _failure if failure is None else failure
    unit = unit or (lambda r: claim.adapter.output(r.output).unit)
    residual_id = residual_id or claim.adapter.residual_id
    try:
        distance_tolerance = finite(distance_tolerance, "distance_tolerance")
        if distance_tolerance < 0:
            raise ValueError("distance_tolerance must be nonnegative")
        if boundary:
            thresholds = {r.name: 0.0 for r in requirements}
        else:
            if not isinstance(violation_margins, Mapping) or set(violation_margins) != {
                r.name for r in requirements
            }:
                raise ValueError(
                    "violation_margins must give a physical-unit threshold for every selected requirement"
                )
            thresholds = {
                r.name: finite(violation_margins[r.name], "violation margin")
                for r in requirements
            }
            if any(thresholds[r.name] <= r.tolerance for r in requirements):
                raise ValueError(
                    "each violation margin must be strictly greater than that requirement's tolerance"
                )
    except (ValueError, TypeError) as exc:
        return rejected_result(
            contract,
            operation,
            request,
            str(exc),
            code="invalid_distance_search",
            execution="invalid",
        )

    evaluations = {}

    def evaluate(point):
        key = tuple(point[n] for n in claim.domain.space.names)
        if key not in evaluations:
            evaluations[key] = evaluate_response(claim, point, contract=contract)
        return evaluations[key]

    branches = []
    evidence, diagnostics = [], []
    # A branch that cannot even be compiled retains a zero lower bound. It must
    # not disappear from the union or allow another candidate to claim a minimum.
    for index, requirement in enumerate(requirements):
        try:
            problem = compile_branch(claim, requirement, thresholds[requirement.name])
            branch = solve_branch(problem, evaluate)
            # A mathematically finite radius can exceed the JSON float range.
            # Keep that branch unresolved instead of crashing or exporting inf.
            round_down(branch.lower)
            branches.append(branch)
            details = {
                **problem.manifest(),
                "lower_exact": str(branch.lower),
                "lower_certificate": branch.lower_proof,
                "unreachable": branch.unreachable,
                "infeasibility_certificate": branch.infeasibility_proof,
                "candidate_distance_exact": str(branch.distance)
                if branch.distance is not None
                else None,
                "solver_attempts": branch.attempts,
                "diagnostics": branch.diagnostics,
            }
        except (ValueError, OverflowError) as exc:
            branches.append(None)
            details = {
                "requirement": requirement.name,
                "lower_exact": "0",
                "unreachable": False,
                "lower_certificate": {"method": "norm_nonnegativity"},
                "diagnostics": [str(exc)],
            }
        evidence.append(
            Evidence(
                f"branch:{index}",
                "search_branch_bounds",
                "exact_bounded_affine_branch",
                "verified",
                details=details,
            )
        )

    reachable = [b for b in branches if b is None or not b.unreachable]
    unreachable = not reachable
    lower_exact = min(
        (b.lower if b is not None else Fraction(0) for b in reachable),
        default=Fraction(0),
    )
    candidates = [
        b
        for b in branches
        if b is not None and b.distance is not None and not b.unreachable
    ]
    candidate = (
        min(candidates, key=lambda b: (b.distance, b.problem.requirement.name))
        if candidates
        else None
    )
    lower = round_down(lower_exact)
    upper = round_up(candidate.distance) if candidate is not None else None
    evidence.append(
        Evidence(
            "search_lower",
            "search_lower_bound",
            "minimum_over_exhaustive_requirement_branches",
            "verified",
            details={
                "branch_evidence_refs": [f"branch:{i}" for i in range(len(branches))],
                "lower_exact": str(lower_exact),
            },
        )
    )
    minimum = upper is not None and Fraction(upper) - Fraction(lower) <= Fraction(
        distance_tolerance
    )
    if minimum or unreachable:
        evidence.append(
            Evidence(
                "search_resolution",
                "search_resolution",
                "exhaustive_affine_requirement_union",
                "verified",
                details={
                    "resolution": "unreachable" if unreachable else "minimum_verified",
                    "distance_tolerance": distance_tolerance,
                    "branch_evidence_refs": [
                        f"branch:{i}" for i in range(len(branches))
                    ],
                },
            )
        )
    if candidate is not None:
        evidence.append(
            Evidence(
                "search_candidate",
                "search_upper_bound",
                "exact_physical_target_and_distance",
                "verified",
                details={
                    "evaluation_ref": candidate.evaluation.ref.to_dict(),
                    "requirement": candidate.problem.requirement.name,
                    "threshold": candidate.problem.threshold,
                    "distance_exact": str(candidate.distance),
                    "distance_upper": upper,
                    "membership": "exact domain rows and public membership both checked",
                },
            )
        )
    witness = None
    # A boundary target is not itself a failure. Only an evaluated requirement
    # violation beyond its declared tolerance can become a failure witness.
    if candidate is not None:
        affected = (
            failure_constraints(claim, candidate.evaluation)
            if failure_constraints is not None
            else tuple(
                c.constraint_ref
                for c in candidate.evaluation.payload.constraint_checks
                if c.assessment == "violated" and c.constraint_ref in claim.requirements
            )
        )
        if affected:
            witness, proofs = failure(claim, candidate.evaluation, affected)
            evidence.extend(proofs)
    resolution = (
        "unreachable"
        if unreachable
        else "minimum_verified"
        if minimum
        else "bounded"
        if candidate is not None
        else "unresolved"
    )
    resolved = minimum or unreachable
    if not resolved:
        diagnostics.append(
            Diagnostic(
                "distance_search_unresolved",
                "search",
                "The verified bounds do not establish the requested minimum; inspect each branch and any retained candidate.",
            )
        )
    reported = (
        candidate.problem.requirement if candidate is not None else requirements[0]
    )
    # Retain only evaluations actually used as branch candidates. Unsuccessful
    # numerical proposals are described in branch evidence, not operating results.
    supporting = {b.evaluation.result_id: b.evaluation for b in candidates}
    search = Search(
        "nonpositive_margin" if boundary else "breaking_shortfall",
        "ge",
        QuantityValue(residual_id(reported.name), thresholds[reported.name]),
        {
            **claim.distance.to_manifest(),
            "target": "any_selected_requirement_threshold",
            "reported_threshold_requirement": reported.name,
            "thresholds": {
                r.name: {
                    "quantity_ref": residual_id(r.name),
                    "value": thresholds[r.name],
                    "unit": unit(r),
                }
                for r in requirements
            },
        },
        resolution,
        Bound("finite", "1", lower, ("search_lower",)),
        Bound("finite", "1", upper, ("search_candidate",))
        if upper is not None
        else Bound("unknown", "1"),
        distance_tolerance,
        candidate.evaluation.ref if candidate is not None else None,
        upper,
        "attained"
        if candidate is not None
        else "not_attained"
        if unreachable
        else "unknown",
        ("search_resolution",) if resolved else (),
    )
    return RobustnessResult(
        contract,
        request,
        "completed" if resolved else "unresolved",
        RobustnessPayload(
            "fail"
            if witness is not None
            else "not_assessed"
            if resolved
            else "inconclusive",
            Coverage(
                "partial",
                evaluated_support={
                    "realizations": [
                        e.request["realization"] for e in supporting.values()
                    ]
                },
                unexplored_support={"claim_audit": "not performed by threshold search"},
            ),
            tuple(e.ref for e in supporting.values()),
            witness,
            search,
        ),
        tuple(evidence),
        tuple(diagnostics),
        tuple(supporting.values()),
    )
