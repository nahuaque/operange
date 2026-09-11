"""Engineering result adapters for the constant-COP heat model."""

from dataclasses import asdict

from .contract_types import (
    Bound,
    ConstraintCheck,
    ConstraintSpec,
    Coverage,
    Diagnostic,
    Evidence,
    EvaluationPayload,
    Measurement,
    Membership,
    Objective,
    ProcessContract,
    QuantitySpec,
    QuantityValue,
    RobustnessPayload,
    Search,
    Witness,
    snapshot,
)
from .engineering_results import EvaluationResult, RobustnessResult
from .heat_recovery import COORDINATES, MODEL_ID
from .claim import bind_contract

OUTPUTS = (
    "delivered_heat",
    "required_heat",
    "delivery_margin",
    "requirement_residual",
    "electrical_power",
    "source_heat",
)
PHYSICAL = (
    "heat_balance",
    "source_balance",
    "compressor_capacity",
    "source_capacity",
    "power_nonnegative",
    "source_nonnegative",
    "heat_nonnegative",
)


def contract_for(claim):
    quantities = [
        QuantitySpec(name, "fraction", "dimensionless", "input") for name in COORDINATES
    ]
    quantities.extend(
        QuantitySpec(
            name,
            "MW",
            "electrical_power" if name == "electrical_power" else "thermal_power",
            "residual"
            if name == "requirement_residual"
            else "control"
            if name == "electrical_power"
            else "output",
        )
        for name in OUTPUTS
    )
    equations = (
        ("heat_balance", "eq", "q - COP * p"),
        ("source_balance", "eq", "s - (COP - 1) * p"),
        ("compressor_capacity", "le", "p - power_capacity_mw"),
        ("source_capacity", "le", "s - source_capacity_mw * (1 - source_derating)"),
        ("power_nonnegative", "le", "-p"),
        ("source_nonnegative", "le", "-s"),
        ("heat_nonnegative", "le", "-q"),
    )
    constraints = []
    for name, relation, expression in equations:
        quantities.append(
            QuantitySpec(
                f"{name}_residual",
                "MW",
                "electrical_power"
                if name in ("compressor_capacity", "power_nonnegative")
                else "thermal_power",
                "residual",
            )
        )
        constraints.append(
            ConstraintSpec(
                name,
                f"{name}_residual",
                "equation" if relation == "eq" else "operating_limit",
                relation,
                expression,
                claim.tolerance,
            )
        )
    constraints.append(
        ConstraintSpec(
            "heat_demand",
            "requirement_residual",
            "requirement",
            "le",
            "required_heat - delivered_heat",
            claim.tolerance,
        )
    )
    contract = ProcessContract(
        MODEL_ID,
        {
            "cop": claim.cop,
            "output_definitions": {
                "required_heat": "base_heat_mw * (1 + demand_increase) + reserve_mw",
                "delivery_margin": "delivered_heat - required_heat",
            },
            "requirement": asdict(claim.requirement),
        },
        asdict(claim.design),
        {
            "kind": "box",
            "parameters": [asdict(claim.uncertainty.parameter(n)) for n in COORDINATES],
        },
        {
            "recourse": claim.recourse,
            "response": "maximize_delivered_heat",
            "observation": "all disturbances before controls",
            "feasibility_scope": "physical_constraints_only",
        },
        tuple(quantities),
        tuple(constraints),
        {"absolute_tolerance": claim.tolerance},
    )
    return bind_contract(contract, claim.uncertainty, claim.recourse_policy)


def solver_evidence(old, subject="operating_solve", identity="solver"):
    return Evidence(
        identity,
        subject,
        "linear_program",
        "unresolved" if old.status == "unresolved" else "verified",
        tuple(
            Measurement(name, value, "model_units")
            for name, value in (
                ("maximum_primal_residual", old.max_residual),
                ("reference_error", old.reference_error),
            )
            if value is not None
        ),
        (Measurement("absolute_tolerance", old.tolerance, "model_units"),),
        backend=old.solver,
        details={"solver_status": old.status, "message": old.message},
    )


def evaluation_from_operation(claim, operation):
    contract = contract_for(claim)
    membership = claim.uncertainty.membership(operation.realization).evidence[0]
    evidence = [membership, solver_evidence(operation.evidence)]
    values = [QuantityValue(name, operation.realization[name]) for name in COORDINATES]
    values.append(QuantityValue("required_heat", operation.required_heat_mw))
    checks, objective = [], None
    # A verified physical response can exist even when its service verdict is
    # unresolved because the optimum interval straddles the requirement.
    resolved = operation.maximum_heat_mw is not None
    if resolved:
        q, p, s = (
            operation.maximum_heat_mw,
            operation.power_mw,
            operation.source_heat_mw,
        )
        lower = (
            operation.capacity_lower_mw
            if operation.capacity_lower_mw is not None
            else q
        )
        upper = (
            operation.capacity_upper_mw
            if operation.capacity_upper_mw is not None
            else q
        )
        values.extend(
            QuantityValue(name, value)
            for name, value in (
                ("delivered_heat", q),
                ("electrical_power", p),
                ("source_heat", s),
                ("delivery_margin", operation.slack_mw),
                ("requirement_residual", -operation.slack_mw),
            )
        )
        residuals = (
            q - claim.cop * p,
            s - (claim.cop - 1) * p,
            p - claim.design.power_capacity_mw,
            s
            - claim.design.source_capacity_mw
            * (1 - operation.realization["source_derating"]),
            -p,
            -s,
            -q,
            -operation.slack_mw,
        )
        for spec, value in zip(contract.constraints, residuals):
            assessment = (
                "satisfied"
                if (abs(value) if spec.relation == "eq" else value) <= spec.tolerance
                else "violated"
            )
            checks.append(
                ConstraintCheck(
                    spec.constraint_id,
                    assessment,
                    QuantityValue(spec.residual_ref, value),
                    ("solver",),
                )
            )
        evidence.append(
            Evidence(
                "capacity",
                "global_response_bound",
                "analytical_capacity",
                "verified",
                (
                    Measurement("upper_capacity", upper, "MW"),
                    Measurement(
                        "agreement_error", operation.evidence.reference_error, "MW"
                    ),
                ),
                (Measurement("absolute_tolerance", claim.tolerance, "MW"),),
                ("constant COP > 1", "nonnegative source and compressor capacities"),
                details={
                    "formula": "min(COP * power_capacity, COP/(COP-1) * source_capacity * (1-source_derating))"
                },
            )
        )
        objective = Objective(
            "delivered_heat",
            "maximize",
            q,
            Bound("finite", "MW", lower, ("solver",)),
            Bound("finite", "MW", upper, ("capacity",)),
            "attained",
            "verified" if upper - lower <= claim.tolerance else "unknown",
        )
    else:
        checks = [
            ConstraintCheck(c.constraint_id, "unknown") for c in contract.constraints
        ]
    return EvaluationResult(
        contract,
        {"query": "evaluate", "realization": operation.realization},
        "completed" if resolved else "unresolved",
        EvaluationPayload(
            "feasible" if resolved else "unknown",
            Membership("inside", evidence_refs=("membership",)),
            tuple(values),
            tuple(checks),
            objective,
            PHYSICAL,
        ),
        tuple(evidence),
        ()
        if resolved
        else (
            Diagnostic(
                "operating_solve_unresolved", "response", operation.evidence.message
            ),
        ),
    )


def evaluate_result(claim, realization):
    try:
        operation = claim.evaluate(realization)
    except (ValueError, TypeError, KeyError) as exc:
        return EvaluationResult(
            contract_for(claim),
            {"query": "evaluate", "realization": snapshot(realization)},
            "invalid",
            EvaluationPayload("not_evaluated", Membership("not_checked")),
            diagnostics=(Diagnostic("invalid_realization", "request", str(exc)),),
        )
    return evaluation_from_operation(claim, operation)


def _witness(operation):
    return Witness(
        "realization",
        (operation.realization,),
        ("heat_demand",),
        ("membership",),
        ("recourse",),
        details={"optimized_shortfall_mw": -operation.slack_mw},
    )


def _failure_evidence(operation):
    return (
        Evidence(
            "membership",
            "domain_membership",
            "box_bounds",
            "verified",
            details={"realization": operation.realization},
        ),
        Evidence(
            "recourse",
            "recourse_infeasibility",
            "analytical_capacity",
            "verified",
            (
                Measurement("optimized_delivery_margin", operation.slack_mw, "MW"),
                Measurement(
                    "capacity_upper",
                    operation.capacity_upper_mw
                    if operation.capacity_upper_mw is not None
                    else operation.maximum_heat_mw,
                    "MW",
                ),
            ),
            (Measurement("absolute_tolerance", operation.evidence.tolerance, "MW"),),
            details={
                "scope": "no admissible control meets heat_demand at this realization",
                "formula": "maximum delivered heat is min(compressor bound, source bound)",
            },
        ),
    )


def audit_result(claim, audit=None):
    audit = claim.audit() if audit is None else audit
    evaluations = tuple(
        {
            e.result_id: e
            for e in (
                evaluation_from_operation(claim, audit.nominal),
                evaluation_from_operation(claim, audit.worst),
            )
        }.values()
    )
    resolved = audit.status != "inconclusive"
    evidence = [
        Evidence(
            "coverage",
            "domain_coverage",
            "coordinate_monotonicity",
            "verified",
            assumptions=(
                "delivery margin is nonincreasing in both box coordinates",
                "constant COP model and fully observed static recourse",
            ),
            details={"checked_extreme": claim.uncertainty.upper_corner},
        )
    ]
    witness = None
    if audit.status == "fail":
        witness = _witness(audit.worst)
        evidence.extend(_failure_evidence(audit.worst))
    return RobustnessResult(
        contract_for(claim),
        {"query": "audit"},
        "completed" if resolved else "unresolved",
        RobustnessPayload(
            audit.status,
            Coverage(
                "analytical_domain" if resolved else "partial",
                evaluated_support={
                    "realizations": [e.request["realization"] for e in evaluations]
                },
                unexplored_support={}
                if resolved
                else {"reason": "worst response unresolved"},
                evidence_refs=("coverage",) if resolved else (),
            ),
            tuple(e.ref for e in evaluations),
            witness,
        ),
        tuple(evidence),
        ()
        if resolved
        else (
            Diagnostic("audit_unresolved", "coverage", audit.worst.evidence.message),
        ),
        evaluations,
    )


def threshold_result(
    claim, *, boundary=False, shortfall_margin_mw=0.01, threshold=None
):
    request = {
        "query": "boundary" if boundary else "breaking",
        "shortfall_margin_mw": 0.0 if boundary else snapshot(shortfall_margin_mw),
    }
    try:
        old = (
            threshold
            if threshold is not None
            else claim.boundary_distance()
            if boundary
            else claim.breaking_witness(shortfall_margin_mw=shortfall_margin_mw)
        )
    except (ValueError, TypeError) as exc:
        return RobustnessResult(
            contract_for(claim),
            request,
            "invalid",
            RobustnessPayload("not_assessed", Coverage("none")),
            diagnostics=(Diagnostic("invalid_threshold", "request", str(exc)),),
        )
    evidence = [
        solver_evidence(
            b.evidence,
            subject=f"threshold_branch:{b.limit}",
            identity=f"branch:{b.limit}",
        )
        for b in old.branches
    ]
    evaluations = (
        ()
        if old.operation is None
        else (evaluation_from_operation(claim, old.operation),)
    )
    valid_candidate = (
        old.operation is not None
        and old.operation.status != "inconclusive"
        and old.operation.slack_mw <= -old.shortfall_margin_mw + claim.tolerance
    )
    witness = None
    if old.operation is not None and old.operation.status == "fail":
        witness = _witness(old.operation)
        evidence.extend(_failure_evidence(old.operation))
    evidence.append(
        Evidence(
            "nonnegative_distance",
            "search_lower_bound",
            "norm_nonnegativity",
            "verified",
        )
    )
    if valid_candidate:
        evidence.append(
            Evidence(
                "candidate",
                "search_upper_bound",
                "evaluated_candidate",
                "verified",
                (Measurement("distance", old.distance, "1"),),
                details={
                    "evaluation_ref": evaluations[0].ref.to_dict(),
                    "target_satisfied_within_tolerance": claim.tolerance,
                },
            )
        )
    resolved = old.minimum_verified or old.status == "unreachable"
    if resolved:
        evidence.append(
            Evidence(
                "exhaustive_branches",
                "search_resolution",
                "exhaustive_affine_union",
                "verified",
                (Measurement("distance_tolerance", claim.tolerance, "1"),),
                assumptions=(
                    "two affine capacity branches exhaust the constant-COP failure set",
                ),
                details={
                    "branch_evidence_refs": [f"branch:{b.limit}" for b in old.branches]
                },
            )
        )
    lower = Bound(
        "finite",
        "1",
        old.distance if old.minimum_verified else 0,
        ("exhaustive_branches",) if old.minimum_verified else ("nonnegative_distance",),
    )
    upper = (
        Bound("finite", "1", old.distance, ("candidate",))
        if valid_candidate
        else Bound("unknown", "1")
    )
    resolution = (
        "minimum_verified"
        if old.minimum_verified
        else "unreachable"
        if old.status == "unreachable"
        else "bounded"
        if valid_candidate
        else "unresolved"
    )
    search = Search(
        "nonpositive_margin" if boundary else "breaking_shortfall",
        "ge",
        QuantityValue("requirement_residual", old.shortfall_margin_mw),
        {
            "metric": "normalized_linf",
            "coordinates": list(COORDINATES),
            "origins": [claim.uncertainty.parameter(n).nominal for n in COORDINATES],
            "scales": [claim.uncertainty.parameter(n).scale for n in COORDINATES],
            "unit": "1",
        },
        resolution,
        lower,
        upper,
        claim.tolerance,
        evaluations[0].ref if valid_candidate else None,
        old.distance if valid_candidate else None,
        "attained"
        if valid_candidate
        else "not_attained"
        if old.status == "unreachable"
        else "unknown",
        ("exhaustive_branches",) if resolved else (),
    )
    return RobustnessResult(
        contract_for(claim),
        request,
        "completed" if resolved else "unresolved",
        RobustnessPayload(
            "fail" if witness else "not_assessed" if resolved else "inconclusive",
            Coverage(
                "partial",
                evaluated_support={
                    "realizations": [old.operation.realization] if old.operation else []
                },
                unexplored_support={"claim_audit": "not performed by threshold search"},
            ),
            tuple(e.ref for e in evaluations),
            witness,
            search,
        ),
        tuple(evidence),
        ()
        if resolved
        else (
            Diagnostic(
                "minimum_unresolved",
                "search",
                "A candidate does not resolve every outer search branch.",
            ),
        ),
        evaluations,
    )
