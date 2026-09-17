"""Finite deterministic equivalent using upstream Pyomo and source-model checks."""

from .._finite_audit import audit_finite
from ..contract_types import Diagnostic
from ..domains import FiniteSet
from . import _idaes_results, _idaes_execution
from ._idaes_model import backend, component, number, uid, unit
from ._idaes_shared_result import IdaesSharedControlResult, fixed_policy
from ._idaes_execution import Candidate, OperatingPointRequest, PointExecution


def optimize_shared_controls(adapter, domain):
    if type(domain) is not FiniteSet:
        raise ValueError("Shared-control optimization requires a FiniteSet")
    adapter.input_space.check_alignment(domain.space)
    if not adapter.controls:
        raise ValueError("Shared-control optimization needs at least one actuator")
    pyo, _ = backend()
    objectives = tuple(
        adapter._template.component_data_objects(
            pyo.Objective, active=True, descend_into=True
        )
    )
    if len(objectives) != 1:
        raise ValueError(
            "Shared-control optimization requires one active source objective"
        )
    objective = objectives[0]
    objective_spec = {
        "component": uid(objective),
        "expression": str(objective.expr),
        "sense": str(objective.sense),
        "unit": unit(objective.expr),
        "aggregation": "worst_case",
    }
    preparation, solver, commands, objective_values = {}, {}, {}, {}
    audit = None
    stage = "scenario_setup"
    try:
        model = pyo.ConcreteModel()
        model.commands = pyo.Block(range(len(adapter.controls)))
        for index, binding in enumerate(adapter.controls):
            source = component(adapter._template, binding.component)
            model.commands[index].value = pyo.Var(
                initialize=source.value, units=pyo.units.get_units(source)
            )
        model.worst = pyo.Var(units=pyo.units.get_units(objective.expr))
        model.links = pyo.ConstraintList()
        model.envelope = pyo.ConstraintList()
        scenarios = []
        for index, scenario in enumerate(domain.scenarios):
            stage = f"scenario_setup:{scenario.name}"
            prepared = PointExecution(
                adapter,
                OperatingPointRequest(
                    scenario.values,
                    {b.name: None for b in adapter.controls},
                    tuple(r.name for r in adapter.requirements),
                ),
            )
            try:
                stage = f"initialization:{scenario.name}"
                child = prepared.prepare()
            finally:
                preparation[scenario.name] = [e.to_dict() for e in prepared.before]
            scenario_objective = component(child, objective_spec["component"])
            scenario_objective.deactivate()
            model.add_component(f"scenario_{index}", child)
            scenarios.append(child)
            for control_index, binding in enumerate(adapter.controls):
                model.links.add(
                    component(child, binding.component)
                    == model.commands[control_index].value
                )
            expr = scenario_objective.expr
            model.envelope.add(
                expr <= model.worst
                if objective.sense == pyo.minimize
                else expr >= model.worst
            )
        model.objective = pyo.Objective(expr=model.worst, sense=objective.sense)
        stage = "shared_solve"
        solved, solver = _idaes_execution.solve(model, adapter)
        if not solved:
            raise ValueError(
                f"Solver termination {solver['termination']}; no checked shared command or infeasibility proof"
            )
        stage = "shared_command_check"
        commands = {
            b.name: number(model.commands[i].value)
            for i, b in enumerate(adapter.controls)
        }
        # The solver enforces equalities numerically. Record their errors, then
        # substitute the SAME exact command into every source scenario check.
        # Never accept slightly different scenario commands as one policy.
        link_errors = {}
        for scenario, child in zip(domain.scenarios, scenarios):
            link_errors[scenario.name] = {}
            for binding in adapter.controls:
                var = component(child, binding.component)
                error = number(var) - commands[binding.name]
                link_errors[scenario.name][binding.name] = error
                if abs(error) > adapter.tolerance:
                    raise ValueError(
                        f"Shared command equality violated in {scenario.name}: {binding.name}"
                    )
                var.set_value(commands[binding.name], skip_validation=True)
        candidate_solver = dict(solver)
        solver = {**solver, "shared_command_residuals": link_errors}
        claim = adapter.as_claim(
            domain,
            recourse=fixed_policy(adapter.base_contract.model["controls"], commands),
        )
        candidates = iter(scenarios)

        def evaluate(claim, realization, *, contract):
            return _idaes_results.evaluate_candidate(
                claim,
                realization,
                Candidate(
                    next(candidates),
                    True,
                    {
                        **candidate_solver,
                        "scope": "Candidate from the joint optimization; checked against a separate source-model clone.",
                    },
                ),
                contract=contract,
            )

        def no_failure(*args):
            raise AssertionError("Numerical candidates cannot certify infeasibility")

        stage = "source_checks"
        audit = audit_finite(
            claim,
            evaluate,
            no_failure,
            method="complete_finite_shared_idaes_numerical_checks",
            scope={
                "coverage": "declared finite realizations only",
                "evidence": "source residual checks; not independently certified",
                "controls": "one shared command vector with no observations",
            },
            diagnostic="Shared candidate lacks a checked source scenario; no infeasibility conclusion follows.",
        )
        evaluations = {e.result_id: e for e in audit.supporting_evaluations}
        for entry in audit.payload.coverage.evaluated_support["scenarios"]:
            evaluation = evaluations[entry["evaluation_ref"]["artifact_id"]]
            if evaluation.payload.feasibility == "feasible":
                objective_values[entry["name"]] = next(
                    e.details["value"]
                    for e in evaluation.evidence
                    if e.evidence_id == "source_objective"
                )
        if audit.payload.verdict != "pass":
            raise ValueError(
                "At least one source scenario did not pass the shared-command check; inspect the retained audit"
            )
    except Exception as exc:
        return IdaesSharedControlResult(
            adapter.base_contract,
            domain,
            objective_spec,
            "unresolved",
            commands=commands,
            scenario_objectives=objective_values,
            audit=audit,
            solver=solver,
            preparation=preparation,
            diagnostics=(
                Diagnostic(
                    "shared_controls_unresolved", stage, f"{type(exc).__name__}: {exc}"
                ),
            ),
        )
    aggregate = max if objective.sense == pyo.minimize else min
    return IdaesSharedControlResult(
        adapter.base_contract,
        domain,
        objective_spec,
        "completed",
        commands=commands,
        objective_value=aggregate(objective_values.values()),
        scenario_objectives=objective_values,
        audit=audit,
        solver=solver,
        preparation=preparation,
    )
