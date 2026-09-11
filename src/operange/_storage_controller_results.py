"""Staged storage replay with carried state and controller-specific failures."""

from dataclasses import asdict
from fractions import Fraction

from ._numeric import round_up
from .contract_types import (
    ConstraintCheck,
    Coverage,
    Diagnostic,
    Evidence,
    EvaluationPayload,
    Membership,
    QuantityValue,
    RobustnessPayload,
    Witness,
)
from .engineering_results import EvaluationResult, RobustnessResult
from .primitives import finite
from .storage_contract import node_id


def _check(step, exact, specs, path):
    values = step.setdefault("residuals", {})
    for name, value in exact.items():
        spec = specs[name]
        # Preserve the sign of equality residuals while bounding their magnitude.
        residual = (
            (-1 if value < 0 else 1) * round_up(abs(value))
            if spec.relation == "eq"
            else round_up(value)
        )
        values[name] = residual
        if (abs(residual) if spec.relation == "eq" else residual) > spec.tolerance:
            path["violated_constraints"].append(name)


def _path(model, controller, scenario, specs):
    d, req, tree = model.design, model.requirement, model.tree
    f = Fraction
    eta_c, eta_d = f(d.charge_efficiency), f(d.discharge_efficiency)
    capacity, charge_limit = f(d.capacity_mwh), f(d.charging_power_mw)
    initial = req.initial_energy_mwh
    path = {
        "scenario": asdict(scenario),
        "steps": [],
        "complete": False,
        "violated_constraints": [],
    }
    try:
        # The preparation API receives only this signal, never the future event.
        step = {
            "stage": "preparation",
            "observations": {
                "preparation_observation": scenario.preparation_observation
            },
            "state_before_mwh": initial,
        }
        path["steps"].append(step)
        charge = controller.preparation_command(scenario.preparation_observation)
        step["commands"] = {"preparation_charge": charge}
        _check(
            step,
            {
                "preparation_charge_lower": -f(charge),
                "preparation_charge_upper": f(charge) - charge_limit,
            },
            specs,
            path,
        )
        retained = (1 - f(d.loss_rate_per_hour) * f(tree.preparation_hours)) * f(
            initial
        )
        gain = eta_c * f(tree.preparation_hours) * f(charge)
        prepared = finite(float(retained + gain), "prepared energy")
        step["state_after_mwh"] = prepared
        exact = {
            "preparation_balance": f(prepared) - retained - gain,
            "prepared_energy_lower": -f(prepared),
            "prepared_energy_upper": f(prepared) - capacity,
        }
        _check(step, exact, specs, path)

        # Only now reveal the event and carry the actual prepared float state.
        step = {
            "stage": "event",
            "observations": {
                "preparation_observation": scenario.preparation_observation,
                "event_identity": scenario.name,
                "demand_mw": scenario.demand_mw,
                "recovered_heat_mw": scenario.recovered_heat_mw,
            },
            "state_before_mwh": prepared,
        }
        path["steps"].append(step)
        commands = controller.event_commands(
            scenario.demand_mw, scenario.recovered_heat_mw
        )
        step["commands"] = commands
        event_charge, discharge = (
            f(commands["event_charge"]),
            f(commands["event_discharge"]),
        )
        surplus = scenario.recovered_heat_mw >= scenario.demand_mw
        _check(
            step,
            {
                "event_service": event_charge
                - discharge
                - f(scenario.recovered_heat_mw)
                + f(scenario.demand_mw),
                "event_charge_lower": -event_charge,
                "event_charge_upper": event_charge - (charge_limit if surplus else 0),
                "event_discharge_lower": -discharge,
                "event_discharge_upper": discharge
                - (0 if surplus else f(d.discharging_power_mw)),
            },
            specs,
            path,
        )
        retained = (1 - f(d.loss_rate_per_hour) * f(tree.event_hours)) * f(prepared)
        gain = eta_c * f(tree.event_hours) * event_charge
        draw = f(tree.event_hours) * discharge / eta_d
        terminal = finite(float(retained + gain - draw), "terminal energy")
        step["state_after_mwh"] = terminal
        exact = {
            "event_state_balance": f(terminal) - retained - gain + draw,
            "terminal_energy_lower": -f(terminal),
            "terminal_energy_upper": f(terminal) - capacity,
            "terminal_min": f(req.terminal_min_mwh) - f(terminal),
        }
        if req.terminal_max_mwh is not None:
            exact["terminal_max"] = f(terminal) - f(req.terminal_max_mwh)
        _check(step, exact, specs, path)
        path["complete"] = True
    except (ValueError, OverflowError) as exc:
        path["error"] = str(exc)
    return path


def _rows(paths, contract, quantity):
    """Follow registry axes; trace order remains the caller's path order."""
    axis = contract.quantity(quantity).axes[0]
    if axis.axis_id == "event":
        rows = {p["scenario"]["name"]: p for p in paths}
        return [
            rows[name]["steps"][1] if len(rows[name]["steps"]) > 1 else {}
            for name in axis.coordinates
        ]
    rows = {
        node_id(p["scenario"]["preparation_observation"]): p["steps"][0] for p in paths
    }
    return [rows[name] for name in axis.coordinates]


def evaluate_result(claim):
    contract, model, controller = (
        claim.contract,
        claim.adapter.model,
        claim.adapter.controller,
    )
    specs = {c.constraint_id: c for c in contract.constraints}
    paths = [_path(model, controller, s, specs) for s in model.tree.scenarios]
    complete = all(p["complete"] for p in paths)
    failure = next((p for p in paths if p["violated_constraints"]), None)
    evidence = [
        Evidence(
            "membership",
            "domain_membership",
            "declared_finite_tree",
            "verified",
            details={"events": [s.name for s in model.tree.scenarios]},
        ),
        Evidence(
            "replay",
            "controller_execution",
            "staged_causal_storage_replay",
            "verified",
            details={
                "controller_ref": controller.ref.to_dict(),
                "paths": paths,
                "scope": "commands and checks recorded for completed stages; unresolved stages remain explicit",
                "initial_state_rule": contract.operating["initial_state_rule"],
                "clipping": False,
                "optimization": False,
            },
        ),
    ]
    if failure:
        evidence.append(
            Evidence(
                "recourse",
                "recourse_infeasibility",
                "fixed_controller_violation",
                "verified",
                details={
                    "controller_ref": controller.ref.to_dict(),
                    "path": failure,
                    "scope": "these prescribed commands violate included constraints; other controllers not assessed",
                },
            )
        )
    scenarios = sorted(model.tree.scenarios, key=lambda s: s.name)
    values = [
        QuantityValue("demand", tuple(s.demand_mw for s in scenarios)),
        QuantityValue("recovered_heat", tuple(s.recovered_heat_mw for s in scenarios)),
    ]
    for name in (
        "preparation_charge",
        "prepared_energy",
        "event_charge",
        "event_discharge",
        "terminal_energy",
    ):
        rows = _rows(paths, contract, name)
        column = [
            r.get("state_after_mwh")
            if name.endswith("energy")
            else r.get("commands", {}).get(name)
            for r in rows
        ]
        if all(v is not None for v in column):
            values.append(QuantityValue(name, tuple(column)))
    checks = []
    for spec in contract.constraints:
        rows = _rows(paths, contract, spec.residual_ref)
        column = [r.get("residuals", {}).get(spec.constraint_id) for r in rows]
        violated = any(spec.constraint_id in p["violated_constraints"] for p in paths)
        available = all(v is not None for v in column)
        checks.append(
            ConstraintCheck(
                spec.constraint_id,
                "violated" if violated else "satisfied" if available else "unknown",
                QuantityValue(spec.residual_ref, tuple(column)) if available else None,
                ("replay",) if available or violated else (),
            )
        )
    return EvaluationResult(
        contract,
        {"query": "evaluate_tree", "response": "frozen_causal_storage_controller"},
        "completed" if complete else "unresolved",
        EvaluationPayload(
            "infeasible" if failure else "feasible" if complete else "unknown",
            Membership("inside", evidence_refs=("membership",)),
            tuple(values),
            tuple(checks),
            included_constraints=tuple(specs),
        ),
        tuple(evidence),
        tuple(
            Diagnostic("storage_path_unresolved", p["scenario"]["name"], p["error"])
            for p in paths
            if not p["complete"]
        ),
    )


def audit_result(claim):
    evaluation = evaluate_result(claim)
    replay = next(e for e in evaluation.evidence if e.evidence_id == "replay")
    paths = replay.details["paths"]
    complete = all(p["complete"] for p in paths)
    failed = next((p for p in paths if p["violated_constraints"]), None)
    evidence = list(evaluation.evidence)
    if complete:
        evidence.append(
            Evidence(
                "coverage",
                "domain_coverage",
                "complete_finite_causal_controller_replay",
                "verified",
                details={
                    "events": [p["scenario"]["name"] for p in paths],
                    "controller_ref": claim.adapter.controller.ref.to_dict(),
                },
            )
        )
    witness = (
        None
        if failed is None
        else Witness(
            "fixed_policy_failure",
            (failed["scenario"],),
            tuple(failed["violated_constraints"]),
            ("membership",),
            ("recourse",),
            node_id(failed["scenario"]["preparation_observation"]),
            details={
                "path": failed,
                "controller_ref": claim.adapter.controller.ref.to_dict(),
                "scope": "failure of this controller on this two-period path; other controllers not assessed",
            },
        )
    )
    return RobustnessResult(
        evaluation.contract,
        {"query": "audit", "response": "frozen_causal_storage_controller"},
        evaluation.execution,
        RobustnessPayload(
            "fail" if witness else "pass" if complete else "inconclusive",
            Coverage(
                "complete_finite" if complete else "partial",
                evaluated_support={
                    "paths": [
                        {"name": p["scenario"]["name"], "complete": p["complete"]}
                        for p in paths
                    ],
                    "scope": "independent two-period paths in caller order; each starts at declared initial energy",
                },
                unexplored_support={
                    "unresolved_paths": [
                        p["scenario"]["name"] for p in paths if not p["complete"]
                    ]
                },
                evidence_refs=("coverage",) if complete else (),
            ),
            (evaluation.ref,),
            witness,
        ),
        tuple(evidence),
        evaluation.diagnostics,
        (evaluation,),
    )
