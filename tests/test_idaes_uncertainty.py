"""Local propagation preserves correlation, operating semantics and failures."""

# ruff: noqa: E402
from dataclasses import replace
import json

import numpy as np
import pytest

pyo = pytest.importorskip("pyomo.environ")
pytest.importorskip("idaes")
pytest.importorskip("highspy")

from operange import Coordinate, ParameterSpace
from operange.integrations.idaes import (
    IdaesAdapter,
    IdaesVariable,
    IdaesRequirement,
    IdaesUncertaintyResult,
)
from operange.contract_types import Diagnostic
from operange.integrations import _idaes_execution
from operange.integrations._idaes_model import model_manifest


PARAMETERS = (IdaesVariable("eta", "eta", "1"), IdaesVariable("loss", "loss", "MW"))
COVARIANCE = ((0.01, 0.04), (0.04, 0.25))


def boiler(*, param=False, nonlinear=False):
    m = pyo.ConcreteModel()
    m.eta = (
        pyo.Param(initialize=0.8, mutable=True)
        if param
        else pyo.Var(initialize=0.8, bounds=(0.1, 1))
    )
    if not param:
        m.eta.fix()
    m.loss = pyo.Var(initialize=1, bounds=(0, 5), units=pyo.units.MW)
    m.loss.fix()
    m.demand = pyo.Param(initialize=8, mutable=True, units=pyo.units.MW)
    m.fuel = pyo.Var(initialize=15, bounds=(0, 20), units=pyo.units.MW)
    m.fuel.fix()
    m.steam = pyo.Var(initialize=11, bounds=(0, 20), units=pyo.units.MW)
    m.balance = pyo.Constraint(expr=m.steam == m.eta * m.fuel - m.loss)
    m.service = pyo.Constraint(expr=m.steam >= m.demand)
    m.cost = pyo.Expression(expr=m.fuel**2 / pyo.units.MW if nonlinear else 3 * m.fuel)
    m.objective = pyo.Objective(expr=m.cost)
    return m, IdaesAdapter(
        model=m,
        name="Local boiler response",
        input_space=ParameterSpace(
            (Coordinate("demand", "MW", "Steam demand", nominal=8),)
        ),
        inputs=(IdaesVariable("demand", "demand", "MW"),),
        controls=(IdaesVariable("fuel", "fuel", "MW"),),
        outputs=(
            IdaesVariable("steam", "steam", "MW"),
            IdaesVariable("cost", "cost", "MW"),
        ),
        requirements=(IdaesRequirement("service", "service", "MW"),),
        solver="ipopt" if nonlinear else "highs",
        solver_options={"tol": 1e-10, "bound_relax_factor": 0} if nonlinear else {},
        tolerance=1e-8,
    )


def propagate(adapter, **kwargs):
    return adapter.propagate_uncertainty(
        parameters=PARAMETERS,
        covariance=COVARIANCE,
        outputs=("steam", "fuel", "cost"),
        margins=("service", "bound:fuel:upper"),
        **kwargs,
    )


@pytest.mark.parametrize("param", [False, True])
def test_fixed_response_and_correlated_covariance_match_analytic_solution(param):
    original, adapter = boiler(param=param)
    before = (model_manifest(original), model_manifest(adapter._template))
    result = propagate(adapter, fixed_controls={"fuel": 15})
    assert result.execution == "completed", result.diagnostics  # nosec B101
    j = np.asarray([[15, -1], [0, 0], [0, 0], [15, -1], [0, 0]])
    np.testing.assert_allclose(result.jacobian, j)
    np.testing.assert_allclose(result.values, [11, 15, 45, 3, 5])
    np.testing.assert_allclose(result.statistics["covariance"], j @ COVARIANCE @ j.T)
    steam = result.statistics["responses"][0]
    assert steam["variance"] == pytest.approx(1.3)  # nosec B101
    assert steam["parameter_contributions"] == pytest.approx(
        {"eta": 1.65, "loss": -0.35}
    )  # nosec B101
    assert steam["ranking"] == ("eta", "loss")  # nosec B101
    assert result.statistics["responses"][3][
        "margin_in_standard_deviations"
    ] == pytest.approx(3 / np.sqrt(1.3))  # nosec B101
    assert (model_manifest(original), model_manifest(adapter._template)) == before  # nosec B101
    assert IdaesUncertaintyResult.from_json(result.to_json()) == result  # nosec B101


@pytest.mark.parametrize("param", [False, True])
def test_reoptimization_moves_commands_and_preserves_service_margin(param):
    _, adapter = boiler(param=param)
    result = propagate(adapter, reoptimize=True, steps={"eta": 1e-4, "loss": 1e-3})
    assert result.execution == "completed", result.diagnostics  # nosec B101
    du = np.asarray([-9 / 0.8**2, 1 / 0.8])
    j = np.asarray([[0, 0], du, 3 * du, [0, 0], -du])
    np.testing.assert_allclose(result.jacobian, j, atol=1e-6)
    np.testing.assert_allclose(
        result.statistics["covariance"], j @ COVARIANCE @ j.T, atol=1e-6
    )
    assert result.values[1] == pytest.approx(11.25)  # nosec B101
    assert len(result.perturbations) == 8  # nosec B101
    assert result.numerical["active_limits"] == ("service",)  # nosec B101
    assert result.scope["optimality"] == "not_certified"  # nosec B101
    assert IdaesUncertaintyResult.from_json(result.to_json()) == result  # nosec B101


@pytest.mark.parametrize(
    "covariance",
    [
        ((1, 2), (2, 1)),
        ((1, 0.1), (0.2, 1)),
        ((0, 1), (1, 1)),
        ((-1, 0), (0, 1)),
        ((1,),),
        ((1, float("nan")), (0, 1)),
        ((1e-30, 2e-15), (2e-15, 1)),
    ],
)
def test_invalid_covariances_fail_before_solver(covariance, monkeypatch):
    _, adapter = boiler()
    monkeypatch.setattr(
        _idaes_execution, "solve", lambda *a: pytest.fail("must not solve")
    )
    with pytest.raises(ValueError, match="[Cc]ovariance|variance"):
        adapter.propagate_uncertainty(
            parameters=PARAMETERS, covariance=covariance, fixed_controls={"fuel": 15}
        )


@pytest.mark.parametrize(
    "covariance", [((0, 0), (0, 0)), ((1, 2), (2, 4)), ((1e-30, 1e-15), (1e-15, 1))]
)
def test_zero_and_singular_covariances_are_valid(covariance):
    _, adapter = boiler()
    result = adapter.propagate_uncertainty(
        parameters=PARAMETERS, covariance=covariance, fixed_controls={"fuel": 15}
    )
    assert result.execution == "completed"  # nosec B101
    assert result.statistics["responses"][0]["variance"] >= 0  # nosec B101


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"reoptimize": True},
        {"fixed_controls": {}},
        {"fixed_controls": {"fuel": 15}, "steps": {"eta": 1}},
        {"fixed_controls": {"fuel": 15}, "reoptimize": True},
        {"reoptimize": True, "steps": {"eta": 0, "loss": 1}},
        {"reoptimize": True, "steps": {"eta": 1e-30, "loss": 1}},
        {"fixed_controls": {"fuel": 15}, "margins": ("constraint:balance:eq",)},
        {"fixed_controls": {"fuel": 15}, "outputs": ("missing",)},
        {
            "fixed_controls": {"fuel": 15},
            "parameters": (IdaesVariable("eta", "eta", "MW"),),
        },
        {
            "fixed_controls": {"fuel": 15},
            "parameters": (
                IdaesVariable("a", "fuel", "MW"),
                IdaesVariable("b", "loss", "MW"),
            ),
        },
    ],
)
def test_invalid_requests_fail_before_solver(options, monkeypatch):
    _, adapter = boiler()
    monkeypatch.setattr(
        _idaes_execution, "solve", lambda *a: pytest.fail("must not solve")
    )
    with pytest.raises(ValueError):
        adapter.propagate_uncertainty(
            **{"parameters": PARAMETERS, "covariance": COVARIANCE, **options}
        )


def test_underdetermined_and_singular_fixed_state_systems_stay_unresolved():
    _, adapter = boiler()
    for extra in (True, False):
        m = adapter._template.clone()
        if extra:
            m.extra = pyo.Var(initialize=1)
        else:
            m.redundant = pyo.Constraint(
                expr=2 * m.steam == 2 * (m.eta * m.fuel - m.loss)
            )
        result = propagate(replace(adapter, model=m), fixed_controls={"fuel": 15})
        assert result.execution == "unresolved"  # nosec B101
        assert not result.jacobian  # nosec B101
        with pytest.raises(ValueError, match="completed"):
            _ = result.statistics


def test_active_set_change_and_curvature_keep_all_probes_and_no_statistics():
    _, adapter = boiler()
    # A second service floor ties at the baseline, creating a genuine kink.
    m = adapter._template.clone()
    m.floor = pyo.Constraint(expr=m.fuel >= 11.25 * pyo.units.MW)
    result = propagate(
        replace(adapter, model=m), reoptimize=True, steps={"eta": 1e-3, "loss": 1e-2}
    )
    assert result.execution == "unresolved"  # nosec B101
    assert len(result.perturbations) == 8  # nosec B101
    assert any(
        "Active constraints change" in d.message
        for p in result.perturbations
        for d in p.diagnostics
    )  # nosec B101
    result = propagate(adapter, reoptimize=True, steps={"eta": 0.1, "loss": 0.1})
    assert result.execution == "unresolved"  # nosec B101
    assert "derivatives disagree" in result.diagnostics[-1].message  # nosec B101


def test_parameter_bounds_and_solver_failures_remain_visible(monkeypatch):
    _, adapter = boiler()
    result = propagate(adapter, reoptimize=True, steps={"eta": 0.5, "loss": 0.1})
    assert result.execution == "unresolved" and len(result.perturbations) == 8  # nosec B101
    assert any(p.evaluation is None for p in result.perturbations)  # nosec B101
    monkeypatch.setattr(
        _idaes_execution, "solve", lambda *a: (False, {"termination": "infeasible"})
    )
    result = propagate(adapter, fixed_controls={"fuel": 15})
    assert (
        result.execution == "unresolved"
        and result.baseline.payload.feasibility == "unknown"
    )  # nosec B101
    assert IdaesUncertaintyResult.from_json(result.to_json()) == result  # nosec B101


def test_solver_mutation_is_checked_on_original_equations(monkeypatch):
    _, adapter = boiler()
    solve = _idaes_execution.solve

    def mutate(model, adapter):
        solved, details = solve(model, adapter)
        model.balance.deactivate()
        model.steam.set_value(19)
        return solved, details

    monkeypatch.setattr(_idaes_execution, "solve", mutate)
    result = propagate(adapter, fixed_controls={"fuel": 15})
    assert result.execution == "unresolved"  # nosec B101


def test_failed_perturbation_retains_other_checks(monkeypatch):
    _, adapter = boiler()
    solve = _idaes_execution.solve
    calls = 0

    def fail_one(model, adapter):
        nonlocal calls
        calls += 1
        if calls == 3:
            return False, {"termination": "maxIterations"}
        return solve(model, adapter)

    monkeypatch.setattr(_idaes_execution, "solve", fail_one)
    result = propagate(adapter, reoptimize=True, steps={"eta": 1e-4, "loss": 1e-3})
    assert result.execution == "unresolved" and len(result.perturbations) == 8  # nosec B101
    assert sum(p.checked for p in result.perturbations) == 7  # nosec B101
    assert not result.jacobian  # nosec B101
    assert IdaesUncertaintyResult.from_json(result.to_json()) == result  # nosec B101


def test_feasibility_only_termination_is_not_reoptimization(monkeypatch):
    _, adapter = boiler()
    solve = _idaes_execution.solve

    def feasible(model, adapter):
        solved, details = solve(model, adapter)
        return solved, {**details, "termination": "feasible"}

    monkeypatch.setattr(_idaes_execution, "solve", feasible)
    result = propagate(adapter, reoptimize=True, steps={"eta": 1e-4, "loss": 1e-3})
    assert result.execution == "unresolved" and not result.perturbations  # nosec B101
    assert "termination" in result.diagnostics[-1].message  # nosec B101
    fixed = propagate(adapter, fixed_controls={"fuel": 15})
    assert fixed.execution == "completed"  # nosec B101


def test_indexed_parameter_and_parameter_dependent_margin_derivatives():
    _, adapter = boiler(param=True)
    m = adapter._template.clone()
    m.coefficients = pyo.Param(["cold side"], initialize=0.8, mutable=True)
    m.balance.set_value(m.steam == m.coefficients["cold side"] * m.fuel - m.loss)
    m.fuel.setub(25 * m.coefficients["cold side"] * pyo.units.MW)
    adapter = replace(adapter, model=m)
    result = adapter.propagate_uncertainty(
        parameters=(
            IdaesVariable("eta", "coefficients['cold side']", "1"),
            PARAMETERS[1],
        ),
        covariance=COVARIANCE,
        outputs=(),
        margins=("bound:fuel:upper",),
        fixed_controls={"fuel": 15},
    )
    assert result.execution == "completed", result.diagnostics  # nosec B101
    np.testing.assert_allclose(result.values, [5])
    np.testing.assert_allclose(result.jacobian, [[25, 0]])
    assert result.statistics["responses"][0]["variance"] == pytest.approx(6.25)  # nosec B101


def test_fixed_derivatives_report_active_limits_without_imposing_them():
    _, adapter = boiler()
    result = propagate(adapter, fixed_controls={"fuel": 11.25})
    assert result.execution == "completed"  # nosec B101
    assert result.values[3] == 0  # nosec B101
    np.testing.assert_allclose(result.jacobian[3], [11.25, -1])
    assert result.numerical["active_limits"] == ("service",)  # nosec B101
    assert any(d.code == "active_operating_limits" for d in result.diagnostics)  # nosec B101


@pytest.mark.skipif(
    not pyo.SolverFactory("ipopt").available(False), reason="Ipopt required"
)
def test_bootstrap_covariance_mapping_uses_fit_order_and_retains_provenance():
    from examples.idaes_uncertainty import campaign, operating_model, PARAMETER_MAP

    bootstrap = campaign().bootstrap(samples=6, seed=7)
    assert bootstrap.execution == "completed", bootstrap.diagnostics  # nosec B101

    adapter = operating_model()
    adapter = replace(
        adapter, model=adapter._template, solver="highs", solver_options={}
    )
    result = bootstrap.propagate_uncertainty(
        adapter,
        parameter_map=dict(reversed(tuple(PARAMETER_MAP.items()))),
        fixed_controls={"fuel": 15},
    )
    assert result.execution == "completed", result.diagnostics  # nosec B101
    np.testing.assert_allclose(result.covariance, bootstrap.statistics["covariance"])
    assert (
        tuple(p.name for p in result.parameters) == bootstrap.statistics["parameters"]
    )  # nosec B101
    assert tuple(p.value for p in result.parameters) == tuple(
        p.value for p in bootstrap.calibration.estimates
    )  # nosec B101
    assert result.provenance["uncertainty_ref"] == bootstrap.ref.to_dict()  # nosec B101
    incomplete = replace(
        bootstrap,
        execution="unresolved",
        draws=(),
        diagnostics=(Diagnostic("incomplete", "draws", "test"),),
    )
    with pytest.raises(ValueError, match="every requested"):
        incomplete.propagate_uncertainty(
            adapter, parameter_map=PARAMETER_MAP, fixed_controls={"fuel": 15}
        )


def test_record_rejects_inconsistent_completion_and_tampering():
    _, adapter = boiler()
    result = propagate(adapter, fixed_controls={"fuel": 15})
    for changes in (
        {"jacobian": ()},
        {"execution": "unresolved"},
        {"scope": {}},
        {"mode": "reoptimized"},
    ):
        with pytest.raises(ValueError):
            replace(result, **changes)
    data = json.loads(result.to_json())
    data["jacobian"][0][0] += 1
    with pytest.raises(ValueError, match="integrity"):
        IdaesUncertaintyResult.from_json(json.dumps(data))


@pytest.mark.skipif(
    not pyo.SolverFactory("ipopt").available(False), reason="Ipopt required"
)
def test_nonlinear_objective_reoptimization():
    _, adapter = boiler(nonlinear=True)
    result = propagate(adapter, reoptimize=True, steps={"eta": 1e-4, "loss": 1e-3})
    assert result.execution == "completed", result.diagnostics  # nosec B101
    np.testing.assert_allclose(
        result.jacobian[2], 2 * 11.25 * np.asarray([-9 / 0.8**2, 1 / 0.8]), atol=1e-4
    )
