"""Shared decisions must respect hidden scenarios and original source equations."""

# ruff: noqa: E402
from dataclasses import replace
import json

import pytest

pyo = pytest.importorskip("pyomo.environ")
pytest.importorskip("idaes")
pytest.importorskip("highspy")

from operange import (
    Coordinate,
    ParameterSpace,
    FiniteSet,
    Scenario,
    BoxSet,
    Parameter,
    verify_result,
)
from operange.integrations.idaes import (
    IdaesAdapter,
    IdaesVariable,
    IdaesRequirement,
    IdaesSharedControlResult,
)
from operange.integrations import _idaes_execution
from operange.integrations._idaes_model import model_manifest


def boiler(*, maximize=False, nonlinear=False):
    m = pyo.ConcreteModel()
    m.demand = pyo.Param(initialize=8, mutable=True, units=pyo.units.MW)
    m.efficiency = pyo.Param(initialize=0.8, mutable=True)
    m.loss = pyo.Var(initialize=0, bounds=(0, 5), units=pyo.units.MW)
    m.loss.fix()
    m.fuel = pyo.Var(initialize=10, bounds=(0, 20), units=pyo.units.MW)
    m.fuel.fix()
    m.steam = pyo.Var(initialize=8, bounds=(0, 20), units=pyo.units.MW)
    supplied = m.fuel**2 / (10 * pyo.units.MW) if nonlinear else m.fuel
    m.balance = pyo.Constraint(expr=m.steam == m.efficiency * supplied - m.loss)
    m.service = pyo.Constraint(expr=m.steam >= m.demand)
    m.objective = pyo.Objective(
        expr=-m.fuel if maximize else m.fuel,
        sense=pyo.maximize if maximize else pyo.minimize,
    )
    adapter = IdaesAdapter(
        model=m,
        name="Shared boiler",
        input_space=ParameterSpace(
            (
                Coordinate("demand", "MW", "Steam demand", nominal=8),
                Coordinate("efficiency", "1", "Unknown efficiency", nominal=0.8),
                Coordinate("loss", "MW", "Unknown heat loss", nominal=0),
            )
        ),
        inputs=(
            IdaesVariable("demand", "demand", "MW"),
            IdaesVariable("efficiency", "efficiency", "1"),
            IdaesVariable("loss", "loss", "MW"),
        ),
        controls=(IdaesVariable("fuel", "fuel", "MW"),),
        outputs=(IdaesVariable("steam", "steam", "MW"),),
        requirements=(IdaesRequirement("service", "service", "MW"),),
        solver="ipopt" if nonlinear else "highs",
        solver_options={"tol": 1e-10, "bound_relax_factor": 0} if nonlinear else {},
        tolerance=1e-6,
    )
    domain = FiniteSet(
        adapter.input_space,
        (
            Scenario(
                "a", {"demand": 8, "efficiency": 0.8, "loss": 0}, "First whole vector"
            ),
            Scenario(
                "b", {"demand": 8, "efficiency": 1, "loss": 1}, "Second whole vector"
            ),
            Scenario(
                "duplicate",
                {"demand": 8, "efficiency": 0.8, "loss": 0},
                "Repeated vector",
            ),
        ),
    )
    return m, adapter, domain


@pytest.mark.parametrize("maximize", [False, True])
def test_shared_boiler_matches_analytic_solution_and_preserves_source(maximize):
    model, adapter, domain = boiler(maximize=maximize)
    before = model_manifest(adapter._template)
    source = model_manifest(model)
    result = adapter.optimize_shared_controls(domain)
    assert result.execution == "completed", result.diagnostics  # nosec B101
    assert result.commands == pytest.approx({"fuel": 10})  # nosec B101
    assert result.objective_value == pytest.approx(-10 if maximize else 10)  # nosec B101
    assert result.audit.payload.verdict == "pass"  # nosec B101
    assert len(result.audit.payload.coverage.evaluated_support["scenarios"]) == 3  # nosec B101
    assert len(result.scenario_objectives) == 3  # nosec B101
    assert model_manifest(adapter._template) == before  # nosec B101
    assert model_manifest(model) == source  # nosec B101
    for evaluation in result.audit.supporting_evaluations:
        values = {v.quantity_ref: v.value for v in evaluation.payload.values}
        assert values["fuel"] == result.commands["fuel"]  # nosec B101
        assert values["steam"] == pytest.approx(
            values["efficiency"] * 10 - values["loss"]
        )  # nosec B101
    assert verify_result(result.audit).status == "unsupported"  # nosec B101
    assert result.scope["optimality"] == "not_certified"  # nosec B101
    assert IdaesSharedControlResult.from_json(result.to_json()) == result  # nosec B101
    # The ideal per-scenario optimizer uses 9 MW in case b; it is not the
    # shared plan and must not leak into the common commands.
    ideal = adapter.as_claim(domain).evaluate_result(domain.scenario("b").values)
    assert next(
        v.value for v in ideal.payload.values if v.quantity_ref == "fuel"
    ) == pytest.approx(9)  # nosec B101


@pytest.mark.parametrize("maximize", [False, True])
def test_worst_case_aggregation_is_not_a_mean_or_separate_scenario_optima(maximize):
    _, adapter, domain = boiler()
    m = adapter._template.clone()
    # Opposing preferences select u=10 under worst-case cost. Averaging these
    # costs would leave every feasible command between 10 and 20 equivalent.
    cost = (2 * m.efficiency - 1.8) * (m.fuel - 10 * pyo.units.MW)
    m.objective.set_value(-cost if maximize else cost)
    m.objective.sense = pyo.maximize if maximize else pyo.minimize
    adapter = replace(adapter, model=m)
    result = adapter.optimize_shared_controls(domain)
    assert result.execution == "completed", result.diagnostics  # nosec B101
    assert result.commands["fuel"] == pytest.approx(10)  # nosec B101
    assert result.objective_value == pytest.approx(0)  # nosec B101


def test_state_dependent_objective_uses_all_scenarios():
    _, adapter, domain = boiler()
    m = adapter._template.clone()
    m.objective.set_value(m.steam)
    result = replace(adapter, model=m).optimize_shared_controls(domain)
    assert result.execution == "completed", result.diagnostics  # nosec B101
    assert result.scenario_objectives == pytest.approx({"a": 8, "b": 9, "duplicate": 8})  # nosec B101
    assert result.objective_value == pytest.approx(9)  # nosec B101


def test_multiple_actuators_keep_native_units_and_all_scenario_bounds(monkeypatch):
    from pyomo.util.check_units import assert_units_consistent

    _, adapter, domain = boiler()
    m = adapter._template.clone()
    m.flow = pyo.Var(initialize=0, bounds=(0, 1), units=pyo.units.kg / pyo.units.s)
    duty = m.flow * pyo.units.MW * pyo.units.s / pyo.units.kg
    m.balance.set_value(m.steam == m.efficiency * m.fuel + duty - m.loss)
    m.objective.set_value(m.fuel + 0.5 * duty)
    adapter = replace(
        adapter,
        model=m,
        controls=adapter.controls + (IdaesVariable("flow", "flow", "kg/s"),),
    )
    solve = _idaes_execution.solve

    def checked_solve(model, adapter):
        assert_units_consistent(model)
        return solve(model, adapter)

    monkeypatch.setattr(_idaes_execution, "solve", checked_solve)
    result = adapter.optimize_shared_controls(domain)
    assert result.execution == "completed", result.diagnostics  # nosec B101
    assert result.commands == pytest.approx({"fuel": 8.75, "flow": 1})  # nosec B101
    assert result.objective_value == pytest.approx(9.25)  # nosec B101


def test_initialization_failure_keeps_scenario_diagnostics(monkeypatch):
    from operange.contract_types import Evidence
    from operange.integrations import _idaes_execution as shared

    _, adapter, domain = boiler()

    def fail(model, adapter, evidence):
        evidence.append(
            Evidence(
                "initialization",
                "unit",
                "idaes_initialization",
                "unresolved",
                details={"error": "unit failed"},
            )
        )
        raise RuntimeError("unit failed")

    monkeypatch.setattr(shared, "initialize", fail)
    monkeypatch.setattr(
        _idaes_execution, "solve", lambda *args: pytest.fail("must not solve")
    )
    result = adapter.optimize_shared_controls(domain)
    assert result.execution == "unresolved"  # nosec B101
    assert result.preparation["a"][0]["details"]["error"] == "unit failed"  # nosec B101
    assert IdaesSharedControlResult.from_json(result.to_json()) == result  # nosec B101


@pytest.mark.parametrize(
    "failure", ["infeasible", "unavailable", "exception", "no_solution"]
)
def test_search_failures_stay_unresolved_and_never_report_an_optimum(
    monkeypatch, failure
):
    _, adapter, domain = boiler()
    if failure == "infeasible":
        m = adapter._template.clone()
        m.fuel.setub(5)
        adapter = replace(adapter, model=m)
    elif failure == "unavailable":
        adapter = replace(
            adapter, model=adapter._template, solver="definitely_missing_solver"
        )
    elif failure == "exception":

        def fail(*args):
            raise RuntimeError("backend failed")

        monkeypatch.setattr(_idaes_execution, "solve", fail)
    else:
        monkeypatch.setattr(
            _idaes_execution,
            "solve",
            lambda *args: (
                False,
                {"termination": "optimal", "reason": "No fresh solution"},
            ),
        )
    result = adapter.optimize_shared_controls(domain)
    assert result.execution == "unresolved"  # nosec B101
    assert result.objective_value is None and result.audit is None  # nosec B101
    assert len(result.domain.scenarios) == 3  # nosec B101
    assert result.diagnostics  # nosec B101
    assert IdaesSharedControlResult.from_json(result.to_json()) == result  # nosec B101


@pytest.mark.parametrize(
    "damage", ["state", "equation", "fixed_input", "links", "objective"]
)
def test_solver_mutations_cannot_replace_source_checks(monkeypatch, damage):
    _, adapter, domain = boiler()
    solve = _idaes_execution.solve

    def altered(model, adapter):
        solved, details = solve(model, adapter)
        if damage == "state":
            model.scenario_0.steam.set_value(0)
        elif damage == "equation":
            model.scenario_0.balance.deactivate()
            model.scenario_0.steam.set_value(9)
        elif damage == "fixed_input":
            model.scenario_0.loss.set_value(0.5)
        elif damage == "links":
            model.scenario_0.fuel.set_value(11)
        else:
            model.scenario_0.objective.set_value(999)
        return solved, details

    monkeypatch.setattr(_idaes_execution, "solve", altered)
    result = adapter.optimize_shared_controls(domain)
    if damage == "objective":
        assert (
            result.execution == "completed"
            and result.objective_value == pytest.approx(10)
        )  # nosec B101
    else:
        assert result.execution == "unresolved"  # nosec B101
        assert result.objective_value is None  # nosec B101
        if damage != "links":
            assert result.audit.payload.verdict == "inconclusive"  # nosec B101
            assert (
                len(result.audit.payload.coverage.evaluated_support["scenarios"]) == 3
            )  # nosec B101
    assert IdaesSharedControlResult.from_json(result.to_json()) == result  # nosec B101


def test_exact_shared_command_substitution_checks_even_small_link_errors(monkeypatch):
    _, adapter, domain = boiler()
    solve = _idaes_execution.solve

    def altered(model, adapter):
        solved, details = solve(model, adapter)
        model.commands[0].value.set_value(pyo.value(model.commands[0].value) - 5e-7)
        return solved, details

    monkeypatch.setattr(_idaes_execution, "solve", altered)
    result = adapter.optimize_shared_controls(domain)
    # Equations allow 1e-6 but the service row uses 1e-7. The exact command
    # leaves the provided states within equation tolerance, as declared.
    assert result.execution == "completed", result.diagnostics  # nosec B101
    for evaluation in result.audit.supporting_evaluations:
        assert (
            next(v.value for v in evaluation.payload.values if v.quantity_ref == "fuel")
            == result.commands["fuel"]
        )  # nosec B101
    assert result.solver["shared_command_residuals"]["a"]["fuel"] == pytest.approx(5e-7)  # nosec B101


def test_invalid_requests_fail_before_solving(monkeypatch):
    _, adapter, domain = boiler()
    monkeypatch.setattr(
        _idaes_execution, "solve", lambda *args: pytest.fail("must not solve")
    )
    with pytest.raises(ValueError, match="FiniteSet"):
        adapter.optimize_shared_controls(
            BoxSet((Parameter("demand", "MW", 8, 7, 9, 1, "test"),))
        )
    m = adapter._template.clone()
    m.objective.deactivate()
    with pytest.raises(ValueError, match="one active"):
        replace(adapter, model=m).optimize_shared_controls(domain)
    with pytest.raises(ValueError, match="actuator"):
        replace(adapter, model=m, controls=()).optimize_shared_controls(domain)


def test_portable_result_rejects_inconsistent_claims():
    _, adapter, domain = boiler()
    result = adapter.optimize_shared_controls(domain)
    for changes in (
        {"commands": {"fuel": 9}},
        {"objective_value": 0},
        {"scenario_objectives": {}},
        {"execution": "unresolved"},
        {"scope": {"confidence_level": 0.95}},
    ):
        with pytest.raises(ValueError):
            replace(result, **changes)
    data = json.loads(result.to_json())
    data["solver"]["status"] = "tampered"
    with pytest.raises(ValueError, match="integrity"):
        IdaesSharedControlResult.from_json(json.dumps(data))


@pytest.mark.skipif(
    not pyo.SolverFactory("ipopt").available(False), reason="Ipopt required"
)
def test_nonlinear_shared_commands_match_analytic_solution():
    _, adapter, domain = boiler(nonlinear=True)
    result = adapter.optimize_shared_controls(domain)
    assert result.execution == "completed", result.diagnostics  # nosec B101
    assert result.commands["fuel"] == pytest.approx(10, abs=1e-6)  # nosec B101
    assert result.audit.payload.verdict == "pass"  # nosec B101
