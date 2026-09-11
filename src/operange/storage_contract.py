"""Finite-tree engineering results with shared information-node coordinates."""

from dataclasses import asdict

from .contract_types import (
    Axis,
    Bound,
    Cardinality,
    ConstraintCheck,
    ConstraintSpec,
    Coverage,
    Diagnostic,
    Evidence,
    EvaluationPayload,
    Membership,
    Objective,
    ProcessContract,
    QuantitySpec,
    QuantityValue,
    RobustnessPayload,
    SensitivityPayload,
    Witness,
    digest,
    snapshot,
)
from .engineering_results import EvaluationResult, RobustnessResult, SensitivityResult
from .heat_results import solver_evidence
from .storage import STORAGE_MODEL_ID
from .claim import bind_contract


def node_id(label):
    return "preparation:" + digest(label)


def contract_for(claim, mode="causal", fixed=None):
    # Invalid operating requests use the declared causal contract as context;
    # the rejected requested mode is preserved in request, never executed.
    groups = claim._groups(
        mode if mode in ("fixed", "causal", "perfect_foresight") else "causal", fixed
    )
    labels = sorted(set(groups.values()))
    scenarios = sorted(claim.tree.scenarios, key=lambda s: s.name)
    prep = (Axis("preparation_node", tuple(node_id(g) for g in labels)),)
    event = (Axis("event", tuple(s.name for s in scenarios)),)
    quantities = []
    for name, unit, kind, role, axes in (
        ("demand", "MW", "thermal_power", "input", event),
        ("recovered_heat", "MW", "thermal_power", "input", event),
        ("preparation_charge", "MW", "thermal_power", "control", prep),
        ("prepared_energy", "MWh", "thermal_energy", "state", prep),
        ("event_charge", "MW", "thermal_power", "control", event),
        ("event_discharge", "MW", "thermal_power", "control", event),
        ("terminal_energy", "MWh", "thermal_energy", "state", event),
        ("preparation_interval_min", "MWh", "thermal_energy", "output", event),
        ("preparation_interval_max", "MWh", "thermal_energy", "output", event),
        ("total_preparation_charge", "MW", "thermal_power", "output", ()),
    ):
        quantities.append(QuantitySpec(name, unit, kind, role, axes))
    constraints = []

    def constraint(name, expression, unit, axes, kind="operating_limit", relation="le"):
        residual = name + "_residual"
        quantities.append(
            QuantitySpec(
                residual,
                unit,
                "thermal_energy" if unit == "MWh" else "thermal_power",
                "residual",
                axes,
            )
        )
        constraints.append(
            ConstraintSpec(name, residual, kind, relation, expression, claim.tolerance)
        )

    constraint(
        "preparation_balance",
        "E1 - retained_initial - eta_c * preparation_hours * preparation_charge",
        "MWh",
        prep,
        "equation",
        "eq",
    )
    constraint(
        "event_service",
        "event_charge - event_discharge - recovered_heat + demand",
        "MW",
        event,
        "requirement",
        "eq",
    )
    constraint(
        "event_state_balance",
        "E2 - retention * E1 - eta_c * event_hours * event_charge + event_hours / eta_d * event_discharge",
        "MWh",
        event,
        "equation",
        "eq",
    )
    for name, unit, axes, bound in (
        ("preparation_charge", "MW", prep, "charging_power_mw"),
        ("prepared_energy", "MWh", prep, "capacity_mwh"),
        (
            "event_charge",
            "MW",
            event,
            "charging_power_mw if recovered_heat >= demand else 0",
        ),
        (
            "event_discharge",
            "MW",
            event,
            "discharging_power_mw if demand > recovered_heat else 0",
        ),
        ("terminal_energy", "MWh", event, "capacity_mwh"),
    ):
        constraint(name + "_lower", "-" + name, unit, axes)
        constraint(name + "_upper", name + " - " + bound, unit, axes)
    constraint("terminal_min", "terminal_min_mwh - E2", "MWh", event, "requirement")
    if claim.requirement.terminal_max_mwh is not None:
        constraint("terminal_max", "E2 - terminal_max_mwh", "MWh", event, "requirement")
    if mode == "fixed":
        constraint(
            "fixed_preparation",
            "preparation_charge - fixed_preparation_power_mw",
            "MW",
            prep,
            "operating_limit",
            "eq",
        )
    contract = ProcessContract(
        STORAGE_MODEL_ID,
        {
            "preparation_hours": claim.tree.preparation_hours,
            "event_hours": claim.tree.event_hours,
            "requirement": asdict(claim.requirement),
            "heat_dumping": False,
            "simultaneous_charge_discharge": False,
            "retention": "1 - loss_rate_per_hour * period_hours",
            "interval_outputs": "analytical projection of each future's feasible preparation states, including reachability and terminal requirements",
        },
        asdict(claim.design),
        {
            "kind": "finite_tree",
            "futures": [
                {k: v for k, v in asdict(s).items() if k != "preparation_observation"}
                for s in scenarios
            ],
        },
        {
            "mode": mode,
            "fixed_preparation_power_mw": snapshot(fixed),
            "nodes": [{"id": node_id(g), "observation": g} for g in labels],
            "preparation_node_by_event": {
                s.name: node_id(groups[s.name]) for s in scenarios
            },
            "event_observation": "full event identity",
            "objective": "minimize sum of preparation_charge once per information node",
            "feasibility_scope": "model_and_requirements",
        },
        tuple(quantities),
        tuple(constraints),
        {"absolute_tolerance": claim.tolerance},
    )
    return bind_contract(
        contract,
        claim.tree.domain,
        claim.recourse_policy(mode, fixed_preparation_power_mw=fixed),
    )


def evaluation_from_audit(claim, audit):
    contract = contract_for(claim, audit.mode, audit.fixed_preparation_power_mw)
    evidence = [
        solver_evidence(audit.evidence),
        Evidence(
            "membership",
            "domain_membership",
            "declared_finite_tree",
            "verified",
            details={
                "events": [
                    s.name for s in sorted(claim.tree.scenarios, key=lambda s: s.name)
                ]
            },
        ),
    ]
    scenarios = sorted(claim.tree.scenarios, key=lambda s: s.name)
    values = [
        QuantityValue("demand", tuple(s.demand_mw for s in scenarios)),
        QuantityValue("recovered_heat", tuple(s.recovered_heat_mw for s in scenarios)),
        QuantityValue(
            "preparation_interval_min",
            tuple(row.minimum_mwh for row in audit.intervals),
        ),
        QuantityValue(
            "preparation_interval_max",
            tuple(row.maximum_mwh for row in audit.intervals),
        ),
    ]
    evidence.append(
        Evidence(
            "intervals",
            "feasible_preparation_projection",
            "analytical_interval",
            "verified",
            details={
                "intervals": [asdict(row) for row in audit.intervals],
                "scope": "individual future projections; their existence does not imply a shared causal policy",
            },
        )
    )
    objective = None
    residuals = {}
    if audit.status == "pass":
        groups = {row.preparation_information: row for row in audit.dispatch}
        prep_rows = [groups[g] for g in sorted(groups)]
        columns = {
            "preparation_charge": tuple(row.preparation_charge_mw for row in prep_rows),
            "prepared_energy": tuple(row.prepared_energy_mwh for row in prep_rows),
            "event_charge": tuple(row.event_charge_mw for row in audit.dispatch),
            "event_discharge": tuple(row.event_discharge_mw for row in audit.dispatch),
            "terminal_energy": tuple(row.terminal_energy_mwh for row in audit.dispatch),
        }
        values.extend(QuantityValue(name, column) for name, column in columns.items())
        d, tree, req = claim.design, claim.tree, claim.requirement
        residuals["preparation_balance"] = tuple(
            row.prepared_energy_mwh
            - claim._retained_initial()
            - d.charge_efficiency * tree.preparation_hours * row.preparation_charge_mw
            for row in prep_rows
        )
        residuals["event_service"] = tuple(
            row.event_charge_mw
            - row.event_discharge_mw
            - s.recovered_heat_mw
            + s.demand_mw
            for row, s in zip(audit.dispatch, scenarios)
        )
        residuals["event_state_balance"] = tuple(
            row.terminal_energy_mwh
            - (1 - d.loss_rate_per_hour * tree.event_hours) * row.prepared_energy_mwh
            - d.charge_efficiency * tree.event_hours * row.event_charge_mw
            + tree.event_hours / d.discharge_efficiency * row.event_discharge_mw
            for row in audit.dispatch
        )
        for name, column in columns.items():
            upper = (
                tuple(
                    d.charging_power_mw if s.recovered_heat_mw >= s.demand_mw else 0
                    for s in scenarios
                )
                if name == "event_charge"
                else tuple(
                    d.discharging_power_mw if s.demand_mw > s.recovered_heat_mw else 0
                    for s in scenarios
                )
                if name == "event_discharge"
                else (
                    d.charging_power_mw
                    if name == "preparation_charge"
                    else d.capacity_mwh,
                )
                * len(column)
            )
            residuals[name + "_lower"] = tuple(-v for v in column)
            residuals[name + "_upper"] = tuple(v - u for v, u in zip(column, upper))
        residuals["terminal_min"] = tuple(
            req.terminal_min_mwh - row.terminal_energy_mwh for row in audit.dispatch
        )
        if req.terminal_max_mwh is not None:
            residuals["terminal_max"] = tuple(
                row.terminal_energy_mwh - req.terminal_max_mwh for row in audit.dispatch
            )
        if audit.mode == "fixed":
            residuals["fixed_preparation"] = tuple(
                row.preparation_charge_mw - audit.fixed_preparation_power_mw
                for row in prep_rows
            )
        total = sum(row.preparation_charge_mw for row in prep_rows)
        values.append(QuantityValue("total_preparation_charge", total))
        evidence.append(
            Evidence(
                "optimum",
                "global_response_bound",
                "analytical_minimum_preparation_charge",
                "verified",
                details={
                    "formula": "sum over information nodes of (max(required lower energy, retained initial energy) - retained initial energy) / preparation charge gain; fixed control substituted when requested",
                    "agreement_error_mw": audit.evidence.reference_error,
                },
            )
        )
        objective = Objective(
            "total_preparation_charge",
            "minimize",
            total,
            Bound("finite", "MW", total, ("optimum",)),
            Bound("finite", "MW", total, ("solver",)),
            "attained",
            "verified",
        )
    elif audit.status == "fail":
        evidence.append(
            Evidence(
                "recourse",
                "recourse_infeasibility",
                "analytical_preparation_conflict",
                "verified",
                details={
                    "certificate": asdict(audit.witness),
                    "scope": "no operating policy satisfies model AND requirements under the declared information permissions",
                },
            )
        )
    checks = tuple(
        ConstraintCheck(
            c.constraint_id,
            "satisfied",
            QuantityValue(c.residual_ref, residuals[c.constraint_id]),
            ("solver",),
        )
        if residuals
        else ConstraintCheck(c.constraint_id, "unknown")
        for c in contract.constraints
    )
    return EvaluationResult(
        contract,
        {
            "query": "evaluate_tree",
            "mode": audit.mode,
            "fixed_preparation_power_mw": audit.fixed_preparation_power_mw,
        },
        "unresolved" if audit.status == "inconclusive" else "completed",
        EvaluationPayload(
            "feasible"
            if audit.status == "pass"
            else "infeasible"
            if audit.status == "fail"
            else "unknown",
            Membership("inside", evidence_refs=("membership",)),
            tuple(values),
            checks,
            objective,
            tuple(c.constraint_id for c in contract.constraints),
        ),
        tuple(evidence),
        (Diagnostic("operating_solve_unresolved", "response", audit.evidence.message),)
        if audit.status == "inconclusive"
        else (),
    )


def evaluate_result(claim, mode="causal", *, fixed_preparation_power_mw=None):
    try:
        audit = claim.audit(mode, fixed_preparation_power_mw=fixed_preparation_power_mw)
    except (ValueError, TypeError) as exc:
        return EvaluationResult(
            contract_for(claim),
            {
                "query": "evaluate_tree",
                "mode": snapshot(mode),
                "fixed_preparation_power_mw": snapshot(fixed_preparation_power_mw),
            },
            "invalid",
            EvaluationPayload("not_evaluated", Membership("not_checked")),
            diagnostics=(Diagnostic("invalid_operating_request", "request", str(exc)),),
        )
    return evaluation_from_audit(claim, audit)


def audit_result(claim, mode="causal", *, fixed_preparation_power_mw=None, audit=None):
    if audit is None:
        try:
            audit = claim.audit(
                mode, fixed_preparation_power_mw=fixed_preparation_power_mw
            )
        except (ValueError, TypeError) as exc:
            return RobustnessResult(
                contract_for(claim),
                {
                    "query": "audit",
                    "mode": snapshot(mode),
                    "fixed_preparation_power_mw": snapshot(fixed_preparation_power_mw),
                },
                "invalid",
                RobustnessPayload("not_assessed", Coverage("none")),
                diagnostics=(
                    Diagnostic("invalid_operating_request", "request", str(exc)),
                ),
            )
    evaluation = evaluation_from_audit(claim, audit)
    evidence = [
        Evidence(
            "coverage",
            "domain_coverage",
            "complete_finite_tree",
            "verified" if audit.status != "inconclusive" else "unresolved",
            details={
                "events": [
                    s.name for s in sorted(claim.tree.scenarios, key=lambda s: s.name)
                ],
                "shared_policy_evaluation": evaluation.ref.to_dict(),
            },
        )
    ]
    witness = None
    if audit.witness:
        old = audit.witness
        evidence.extend(
            e
            for e in evaluation.evidence
            if e.evidence_id in ("membership", "recourse")
        )
        evidence.append(
            Evidence(
                "cardinality",
                "witness_cardinality",
                "one_dimensional_interval_intersection",
                "verified",
                details={
                    "count": len(old.scenarios),
                    "minimal": old.minimum_cardinality,
                    "reason": "single-future checks precede pair conflicts; intervals in one dimension have a two-member empty-intersection certificate",
                },
            )
        )
        futures = {
            s.name: {
                k: v for k, v in asdict(s).items() if k != "preparation_observation"
            }
            for s in claim.tree.scenarios
        }
        count = Bound("finite", "count", len(old.scenarios), ("cardinality",))
        witness = Witness(
            "incompatible_futures" if old.kind == "conflicting_futures" else old.kind,
            tuple(futures[n] for n in old.scenarios),
            tuple(c.constraint_id for c in evaluation.contract.constraints),
            ("membership",),
            ("recourse",),
            node_id(old.preparation_information),
            {
                "certificate": asdict(old),
                "scope": "model and requirements under declared operating contract",
            },
            Cardinality(count, count),
        )
    return RobustnessResult(
        evaluation.contract,
        {**dict(evaluation.request), "query": "audit"},
        evaluation.execution,
        RobustnessPayload(
            audit.status,
            Coverage(
                "complete_finite" if audit.status != "inconclusive" else "partial",
                evaluated_support={"tree": "all declared futures compiled jointly"},
                unexplored_support={}
                if audit.status != "inconclusive"
                else {"reason": "joint operating solve unresolved"},
                evidence_refs=("coverage",) if audit.status != "inconclusive" else (),
            ),
            (evaluation.ref,),
            witness,
        ),
        tuple(evidence),
        evaluation.diagnostics,
        (evaluation,),
    )


def sensitivity_result(claim, **request):
    return SensitivityResult(
        contract_for(claim),
        snapshot({"query": "sensitivity", **request}),
        "unsupported",
        SensitivityPayload(
            "not_evaluated",
            response_contract=snapshot({"mode": request.get("response", "optimized")}),
        ),
        diagnostics=(
            Diagnostic(
                "unsupported_sensitivity",
                "storage",
                "The finite-tree adapter does not yet provide derivatives. Discrete information changes are comparisons between operating contracts.",
            ),
        ),
    )
