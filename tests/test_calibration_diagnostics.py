"""Parameter information and held-out prediction against analytic examples."""

# ruff: noqa: E402
from dataclasses import replace
import json

import numpy as np
import pytest

pytest.importorskip("idaes")
pyo = pytest.importorskip("pyomo.environ")
pytest.importorskip("pandas")
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.core.base.componentuid import ComponentUID

from operange.integrations.parmest import (
    ParmestCalibration,
    CalibrationDiagnostics,
    CalibrationValidation,
)
from operange.integrations import _calibration_diagnostics as backend
from operange.integrations._idaes_model import model_manifest

OPTIONS = {"tol": 1e-10, "bound_relax_factor": 0}
ipopt = pytest.mark.skipif(
    not pyo.SolverFactory("ipopt").available(False), reason="Ipopt required"
)


def experiment(
    fuel,
    measured=None,
    *,
    confounded=False,
    param=False,
    expression=False,
    free_fuel=False,
):
    m = pyo.ConcreteModel()
    m.eta = (
        pyo.Param(initialize=0.8, mutable=True)
        if param
        else pyo.Var(initialize=0.8, bounds=(0.1, 1))
    )
    m.fuel = pyo.Var(initialize=fuel, units=pyo.units.MW)
    if not free_fuel:
        m.fuel.fix()
    m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.unknown_parameters[m.eta] = ComponentUID(m.eta)
    gain = m.eta
    if confounded:
        m.area = pyo.Var(initialize=1, bounds=(0.5, 2))
        m.unknown_parameters[m.area] = ComponentUID(m.area)
        gain = m.eta * m.area
    if expression:
        m.steam = pyo.Expression(expr=gain * m.fuel)
    else:
        m.steam = pyo.Var(initialize=0.8 * fuel, units=pyo.units.MW)
        m.balance = pyo.Constraint(expr=m.steam == gain * m.fuel)
    m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.experiment_outputs[m.steam] = 0.8 * fuel if measured is None else measured
    m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.measurement_error[m.steam] = 0.1
    # This deliberately misleading source objective must not be used to predict.
    m.objective = pyo.Objective(
        expr=(m.steam - m.experiment_outputs[m.steam] * pyo.units.MW) ** 2
    )
    return Experiment(m)


def campaign(**kwargs):
    return ParmestCalibration(
        name="Calibration quality",
        experiments={
            f"run-{i}": experiment(fuel, **kwargs) for i, fuel in enumerate((8, 10, 12))
        },
        solver_options=OPTIONS,
    )


@ipopt
@pytest.mark.parametrize(
    "param,expression", [(False, False), (True, False), (True, True)]
)
def test_scaled_jacobian_matches_analytic_derivatives_and_preserves_sources(
    param, expression
):
    study = campaign(param=param, expression=expression)
    before = tuple(model_manifest(s.model) for s in study._experiments)
    fit = study.fit()
    report = study.diagnose(fit, parameter_scales={"eta": 0.2})
    assert report.execution == "completed", report.diagnostics  # nosec B101
    assert np.array(
        [e.jacobian for e in report.experiments]
    ).flatten() == pytest.approx([16, 20, 24])  # nosec B101
    assert report.analysis["rank"] == 1  # nosec B101
    assert report.analysis["singular_values"][0] == pytest.approx(
        np.linalg.norm([16, 20, 24])
    )  # nosec B101
    assert report.analysis["weak_directions"] == ()  # nosec B101
    assert report.scope["structural_identifiability"] == "not_assessed"  # nosec B101
    assert tuple(model_manifest(s.model) for s in study._experiments) == before  # nosec B101
    assert CalibrationDiagnostics.from_json(report.to_json()) == report  # nosec B101


@ipopt
def test_confounded_product_reports_null_parameter_combination():
    study = campaign(confounded=True)
    fit = study.fit()
    report = study.diagnose(fit, parameter_scales={"eta": 0.1, "area": 0.1})
    assert report.execution == "completed", report.diagnostics  # nosec B101
    assert report.analysis["rank"] == 1 and report.analysis["parameter_count"] == 2  # nosec B101
    assert report.analysis["condition_number"] is None  # nosec B101
    weak = report.analysis["weak_directions"]
    assert len(weak) == 1 and weak[0]["classification"] == "numerically_null"  # nosec B101
    vector = np.array([weak[0]["coefficients"][p.component] for p in fit.estimates])
    jacobian = np.array([row for e in report.experiments for row in e.jacobian])
    assert jacobian @ vector == pytest.approx(np.zeros(3), abs=1e-10)  # nosec B101
    assert fit.execution == "completed"  # nosec B101


@ipopt
def test_zero_information_and_fewer_measurements_than_parameters_keep_nullspace():
    exp = experiment(0, confounded=True)
    study = ParmestCalibration(
        name="No excitation", experiments={"zero": exp}, solver_options=OPTIONS
    )
    report = study.diagnose(parameter_scales={"eta": 0.1, "area": 0.1})
    assert report.execution == "completed", report.calibration.diagnostics  # nosec B101
    assert report.analysis["rank"] == 0  # nosec B101
    assert len(report.analysis["weak_directions"]) == 2  # nosec B101


@ipopt
def test_relative_weak_threshold_is_separate_from_numerical_rank():
    exp = experiment(10, confounded=True)
    # An extremely noisy independent area measurement weakly separates U and A.
    exp.model.experiment_outputs[exp.model.area] = 1
    exp.model.measurement_error[exp.model.area] = 10000
    study = ParmestCalibration(
        name="Weak area measurement", experiments={"run": exp}, solver_options=OPTIONS
    )
    weak = study.diagnose(parameter_scales={"eta": 0.1, "area": 0.1})
    assert weak.analysis["rank"] == 2  # nosec B101
    assert weak.analysis["weak_directions"][0]["classification"] == "weak"  # nosec B101
    assert weak.analysis["condition_number"] > 1e5  # nosec B101


@ipopt
def test_parameter_at_bound_remains_a_warning_not_measurement_information():
    study = ParmestCalibration(
        name="Bounded fit",
        experiments={"run": experiment(10, 12)},
        solver_options=OPTIONS,
    )
    report = study.diagnose(parameter_scales={"eta": 1})
    assert report.execution == "completed"  # nosec B101
    assert any(d.code == "parameter_at_bound" for d in report.diagnostics)  # nosec B101
    assert "eta:upper" in report.experiments[0].active_limits  # nosec B101


@ipopt
def test_held_out_predictions_do_not_fit_measurements_and_keep_large_residuals():
    study = campaign()
    fit = study.fit()
    held = {"new-normal": experiment(15, 12), "new-biased": experiment(20, 14)}
    before = tuple(model_manifest(e.model) for e in held.values())
    validation = study.validate(fit, experiments=held)
    assert validation.execution == "completed", validation.diagnostics  # nosec B101
    assert [
        p.result.observations[0].fitted for p in validation.predictions
    ] == pytest.approx([12, 16])  # nosec B101
    assert validation.metrics[0]["rmse"] == pytest.approx(np.sqrt(2))  # nosec B101
    assert validation.metrics[0]["bias"] == pytest.approx(1)  # nosec B101
    assert any(
        d.code == "large_validation_residual"
        for d in validation.predictions[1].diagnostics
    )  # nosec B101
    assert tuple(model_manifest(e.model) for e in held.values()) == before  # nosec B101
    assert CalibrationValidation.from_json(validation.to_json()) == validation  # nosec B101
    changed = study.validate(fit, experiments={"new-biased": experiment(20, -100)})
    assert changed.predictions[0].result.observations[0].fitted == pytest.approx(16)  # nosec B101


@ipopt
def test_heldout_reconciliation_freedom_is_unresolved_and_never_dropped():
    study = campaign()
    validation = study.validate(
        study.fit(),
        experiments={
            "good": experiment(15),
            "free": experiment(15, 10, free_fuel=True),
        },
    )
    assert validation.execution == "unresolved"  # nosec B101
    assert [p.execution for p in validation.predictions] == ["completed", "unresolved"]  # nosec B101
    assert "square equality" in validation.predictions[1].diagnostics[0].message  # nosec B101
    with pytest.raises(ValueError, match="all held-out"):
        _ = validation.metrics
    assert CalibrationValidation.from_json(validation.to_json()) == validation  # nosec B101


@ipopt
def test_singular_state_equations_are_not_parameter_identifiability():
    study = campaign()
    fit = study.fit()
    exp = experiment(10)
    exp.model.balance.set_value((exp.model.steam - 8 * pyo.units.MW) ** 2 == 0)
    validation = study.validate(fit, experiments={"singular": exp})
    assert validation.execution == "unresolved"  # nosec B101
    assert "state Jacobian" in validation.predictions[0].diagnostics[0].message  # nosec B101


@ipopt
@pytest.mark.parametrize("damage", ["state", "fixed", "equation", "solver"])
def test_validation_checks_original_source_after_solver_mutation(monkeypatch, damage):
    study = campaign()
    fit = study.fit()
    solve = backend._idaes_execution.solve

    def damaged(model, adapter):
        if damage == "solver":
            return False, {"termination": "infeasible"}
        solved, details = solve(model, adapter)
        if damage == "fixed":
            model.fuel.set_value(99)
        else:
            model.steam.set_value(99)
            if damage == "equation":
                model.balance.deactivate()
        return solved, details

    monkeypatch.setattr(backend._idaes_execution, "solve", damaged)
    result = study.validate(fit, experiments={"held": experiment(15)})
    assert result.execution == "unresolved"  # nosec B101
    assert len(result.predictions) == 1  # nosec B101
    assert CalibrationValidation.from_json(result.to_json()) == result  # nosec B101


@ipopt
def test_invalid_fit_identity_holdout_names_units_and_scales_fail_early(monkeypatch):
    study = campaign()
    fit = study.fit()
    with pytest.raises(ValueError, match="disjoint"):
        study.validate(fit, experiments={"run-0": experiment(15)})
    with pytest.raises(ValueError, match="exact calibration"):
        campaign(param=True).diagnose(fit, parameter_scales={"eta": 0.1})
    with pytest.raises(ValueError, match="paths and native units"):
        study.validate(fit, experiments={"held": experiment(15, confounded=True)})
    monkeypatch.setattr(
        ParmestCalibration, "fit", lambda self: pytest.fail("must not fit")
    )
    for scales in ({}, {"eta": 0}, {"eta": True}, {"eta": float("nan")}, {"extra": 1}):
        with pytest.raises(ValueError):
            study.diagnose(parameter_scales=scales)
    with pytest.raises(ValueError, match="threshold"):
        study.diagnose(
            parameter_scales={"eta": 1}, rank_tolerance=0.1, weak_threshold=0.01
        )


@ipopt
def test_incomplete_sensitivities_and_tampered_reports_are_rejected(monkeypatch):
    study = campaign()
    fit = study.fit()
    report = study.diagnose(fit, parameter_scales={"eta": 0.1})
    with pytest.raises(ValueError, match="every experiment"):
        replace(report, experiments=report.experiments[:-1])
    data = json.loads(report.to_json())
    data["parameter_scales"]["eta"] = 0.2
    with pytest.raises(ValueError, match="integrity"):
        CalibrationDiagnostics.from_json(json.dumps(data))
    original = backend._regular_system
    count = []

    def failing(*args):
        count.append(1)
        if len(count) == 2:
            raise ValueError("bad derivative")
        return original(*args)

    monkeypatch.setattr(backend, "_regular_system", failing)
    failed = study.diagnose(fit, parameter_scales={"eta": 0.1})
    assert failed.execution == "unresolved" and len(failed.experiments) == 3  # nosec B101
    with pytest.raises(ValueError, match="every experiment"):
        _ = failed.analysis
    assert CalibrationDiagnostics.from_json(failed.to_json()) == failed  # nosec B101


@ipopt
def test_unweighted_measurements_without_sigma_remain_explicitly_unresolved():
    exp = experiment(10)
    exp.model.measurement_error.clear()
    study = ParmestCalibration(
        name="Unweighted",
        experiments={"run": exp},
        objective="SSE",
        solver_options=OPTIONS,
    )
    fit = study.fit()
    report = study.diagnose(fit, parameter_scales={"eta": 0.1})
    assert report.execution == "unresolved"  # nosec B101
    assert "measurement_error" in report.experiments[0].diagnostics[0].message  # nosec B101
    held = experiment(20)
    held.model.measurement_error.clear()
    validation = study.validate(fit, experiments={"held": held})
    assert validation.execution == "completed"  # nosec B101
    assert validation.metrics[0]["standardized_rmse"] is None  # nosec B101


@ipopt
def test_unresolved_fit_does_not_produce_local_information(monkeypatch):
    study = campaign()
    fit = replace(study.fit(), execution="unresolved", estimates=())
    monkeypatch.setattr(
        backend, "_sensitivity", lambda *args: pytest.fail("must not analyze")
    )
    result = study.diagnose(fit, parameter_scales={"eta": 1})
    assert result.execution == "unresolved" and result.experiments == ()  # nosec B101
    assert CalibrationDiagnostics.from_json(result.to_json()) == result  # nosec B101


@ipopt
def test_heat_data_example_distinguishes_fit_information_and_prediction():
    from examples.calibration_credibility import run_example

    known, confounded, validation = run_example()
    assert known.analysis["rank"] == known.analysis["parameter_count"] == 1  # nosec B101
    assert (
        confounded.analysis["rank"] == 1 and confounded.analysis["parameter_count"] == 2
    )  # nosec B101
    assert any(d.code == "rank_deficient_parameters" for d in confounded.diagnostics)  # nosec B101
    assert validation.execution == "completed"  # nosec B101
    assert validation.predictions[1].result.observations[0].residual > 700  # nosec B101


@ipopt
def test_local_information_agrees_with_upstream_parmest_covariance():
    from operange.integrations._parmest_backend import estimator

    study = campaign()
    report = study.diagnose(parameter_scales={"eta": 0.2})
    upstream = estimator(study)
    upstream.theta_est()
    covariance = upstream.cov_est()
    # In this regular scalar Gaussian-error case, invert the information only
    # as a comparison with upstream's independent finite-difference method.
    jacobian = np.asarray([row for e in report.experiments for row in e.jacobian])
    implied = 0.2**2 / float((jacobian.T @ jacobian)[0, 0])
    assert implied == pytest.approx(covariance.iloc[0, 0], rel=1e-7)  # nosec B101
