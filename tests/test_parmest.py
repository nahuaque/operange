"""Calibration against analytic fits, upstream models and untrusted candidates."""

# Optional imports must follow importorskip.
# ruff: noqa: E402
from dataclasses import replace
import json

import pytest

pytest.importorskip("idaes")
pyo = pytest.importorskip("pyomo.environ")
pytest.importorskip("pandas")
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.core.base.componentuid import ComponentUID

from operange import Coordinate, ParameterSpace
from operange.integrations.idaes import IdaesAdapter, IdaesRequirement, IdaesVariable
from operange.integrations.parmest import ParmestCalibration, CalibrationResult
from operange.integrations import _parmest_backend as bridge
from operange.integrations._idaes_model import model_manifest

ipopt = pytest.mark.skipif(
    not pyo.SolverFactory("ipopt").available(False), reason="Ipopt executable required"
)
OPTIONS = {"tol": 1e-10, "bound_relax_factor": 0}


def experiment(fuel=10, measured=7.4, sigma=0.1, *, reconciliation=False):
    m = pyo.ConcreteModel()
    m.eta = pyo.Var(initialize=0.8, bounds=(0.1, 1))
    m.eta.fix()
    m.fuel = pyo.Var(initialize=fuel, bounds=(0, 30), units=pyo.units.MW)
    m.steam = pyo.Var(initialize=0.8 * fuel, bounds=(0, 30), units=pyo.units.MW)
    m.balance = pyo.Constraint(expr=m.steam == m.eta * m.fuel)
    m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.experiment_outputs[m.steam] = measured
    m.measurement_error[m.steam] = sigma
    if reconciliation:
        m.experiment_outputs[m.fuel] = fuel
        m.measurement_error[m.fuel] = sigma
    else:
        m.fuel.fix()
        m.unknown_parameters[m.eta] = ComponentUID(m.eta)
    return Experiment(m)


def calibration(exp=None, **kwargs):
    return ParmestCalibration(
        name="Boiler campaign",
        experiments={"run": exp or experiment()},
        solver_options=OPTIONS,
        **kwargs,
    )


def candidate(study):
    return 0, {"eta": 0.74}, [{"fuel": 10, "steam": 7.4, "eta": 0.74}]


def target_adapter(**changes):
    m = experiment().model
    m.demand = pyo.Param(initialize=7, mutable=True, units=pyo.units.MW)
    m.service = pyo.Constraint(expr=m.steam >= m.demand)
    for key, value in changes.items():
        getattr(m, key).set_value(value)
    return IdaesAdapter(
        model=m,
        name="Boiler",
        input_space=ParameterSpace((Coordinate("demand", "MW", "Demand"),)),
        inputs=(IdaesVariable("demand", "demand", "MW"),),
        controls=(IdaesVariable("fuel", "fuel", "MW"),),
        outputs=(IdaesVariable("steam", "steam", "MW"),),
        requirements=(IdaesRequirement("service", "service", "MW"),),
        solver="ipopt",
        solver_options=OPTIONS,
    )


@ipopt
def test_shared_weighted_fit_matches_analytic_solution_and_preserves_sources():
    a, b = experiment(10, 7.5, 0.1), experiment(20, 14.7, 0.2)
    before = model_manifest(a.model)
    study = ParmestCalibration(
        name="Two conditions", experiments={"a": a, "b": b}, solver_options=OPTIONS
    )
    fit = study.fit()
    assert fit.execution == "completed", fit.diagnostics  # nosec B101
    assert fit.estimates[0].value == pytest.approx(0.7425, abs=1e-8)  # nosec B101
    assert fit.objective_value == pytest.approx(0.28125, abs=1e-8)  # nosec B101
    assert [
        e.observations[0].standardized_residual for e in fit.experiments
    ] == pytest.approx([-0.75, 0.75])  # nosec B101
    assert all(c.satisfied for e in fit.experiments for c in e.checks)  # nosec B101
    assert model_manifest(a.model) == before  # nosec B101
    assert fit.solver["termination"] is None  # nosec B101
    assert fit.solver["api_outcome"] == "returned"  # nosec B101
    assert fit.study["identifiability"] == "not_assessed"  # nosec B101
    assert fit.study["parameter_uncertainty"] == "not_computed"  # nosec B101
    assert CalibrationResult.from_json(fit.to_json()) == fit  # nosec B101


@ipopt
def test_reconciliation_keeps_known_efficiency_and_adjusts_both_measurements():
    fit = calibration(experiment(measured=8.4, reconciliation=True)).fit()
    assert fit.execution == "completed", fit.diagnostics  # nosec B101
    assert fit.estimates == ()  # nosec B101
    values = fit.experiments[0].values
    # Orthogonal projection of (fuel=10, steam=8.4) onto steam=0.8*fuel.
    assert values["fuel"] == pytest.approx((10 + 0.8 * 8.4) / 1.64)  # nosec B101
    assert values["steam"] == pytest.approx(0.8 * values["fuel"])  # nosec B101
    assert values["eta"] == 0.8  # nosec B101


@ipopt
@pytest.mark.parametrize("parameter", ["Var", "Param", "IndexedVar"])
def test_indexed_components_and_measured_expressions(parameter):
    m = pyo.ConcreteModel()
    if parameter == "IndexedVar":
        m.eta = pyo.Var([0], initialize=0.8)
    else:
        m.eta = getattr(pyo, parameter)(
            initialize=0.8, **({"mutable": True} if parameter == "Param" else {})
        )
    eta = m.eta[0] if m.eta.is_indexed() else m.eta
    m.fuel = pyo.Var([1, 2], initialize={1: 10, 2: 15}, units=pyo.units.MW)
    m.fuel.fix()
    m.steam = pyo.Var([1, 2], initialize={1: 8, 2: 12}, units=pyo.units.MW)
    m.balance = pyo.Constraint([1, 2], rule=lambda m, i: m.steam[i] == eta * m.fuel[i])
    m.total = pyo.Expression(expr=sum(m.steam.values()))
    m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.experiment_outputs[m.total] = 18.5
    m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.measurement_error[m.total] = 0.1
    m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.unknown_parameters[eta] = ComponentUID(eta)
    fit = calibration(Experiment(m)).fit()
    assert fit.execution == "completed", fit.diagnostics  # nosec B101
    assert fit.estimates[0].value == pytest.approx(0.74)  # nosec B101
    assert fit.experiments[0].values["steam[2]"] == pytest.approx(11.1)  # nosec B101
    assert fit.experiments[0].observations[0].fitted == pytest.approx(18.5)  # nosec B101


def test_snapshot_freezes_builders_data_options_and_identity(monkeypatch):
    exp = experiment()
    calls = []
    exp.get_labeled_model = lambda: calls.append(1) or exp.model
    study = calibration(exp)
    original_ref = study.ref
    exp.model.experiment_outputs[exp.model.steam] = 999
    exp.model.eta.fix(0.9)
    monkeypatch.setattr(bridge, "_estimate", candidate)
    assert study.fit().execution == "completed"  # nosec B101
    assert calls == [1]  # nosec B101
    assert study.ref == original_ref  # nosec B101
    assert calibration(exp).ref != original_ref  # nosec B101
    with pytest.raises(TypeError):
        study.solver_options["tol"] = 1


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda m: m.measurement_error.clear(), "measurement_error"),
        (lambda m: m.measurement_error.__setitem__(m.steam, 0), "positive"),
        (lambda m: m.experiment_outputs.__setitem__(m.steam, float("nan")), "finite"),
        (lambda m: m.experiment_outputs.clear(), "at least one"),
        (
            lambda m: m.unknown_parameters.__setitem__(
                m.fuel.parent_component(), ComponentUID(m.fuel)
            ),
            None,
        ),
    ],
)
def test_invalid_measurement_labels_fail_before_solve(change, message):
    exp = experiment()
    change(exp.model)
    if message is None:
        # Differing parameter declarations across experiments are rejected.
        with pytest.raises(ValueError, match="same shared"):
            ParmestCalibration(
                name="different", experiments={"a": exp, "b": experiment()}
            )
    else:
        with pytest.raises(ValueError, match=message):
            calibration(exp)


def test_unweighted_sse_requires_common_native_units():
    exp = experiment()
    exp.model.experiment_outputs[exp.model.eta] = 0.74
    with pytest.raises(ValueError, match="common native"):
        calibration(exp, objective="SSE")


@ipopt
def test_experiments_with_different_state_layouts_keep_their_rows():
    experiments = {}
    for i, use_state in enumerate((True, False)):
        m = pyo.ConcreteModel()
        m.eta = pyo.Param(initialize=0.8, mutable=True)
        m.fuel = pyo.Param(initialize=10 * (i + 1), mutable=True, units=pyo.units.MW)
        if use_state:
            m.steam = pyo.Var(initialize=8, units=pyo.units.MW)
            m.balance = pyo.Constraint(expr=m.steam == m.eta * m.fuel)
        else:
            m.steam = pyo.Expression(expr=m.eta * m.fuel)
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs[m.steam] = 7.4 * (i + 1)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error[m.steam] = 0.1
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters[m.eta] = ComponentUID(m.eta)
        experiments[str(i)] = Experiment(m)
    fit = ParmestCalibration(
        name="Mixed state layouts", experiments=experiments, solver_options=OPTIONS
    ).fit()
    assert fit.execution == "completed", fit.diagnostics  # nosec B101
    assert fit.estimates[0].value == pytest.approx(0.74)  # nosec B101
    assert fit.experiments[0].values["steam"] == pytest.approx(7.4)  # nosec B101
    assert fit.experiments[1].values == {}  # nosec B101
    assert fit.experiments[1].observations[0].fitted == pytest.approx(14.8)  # nosec B101


def test_indexed_unknown_params_report_upstream_limitation_before_solving():
    exp = experiment()
    exp.model.coefficient = pyo.Param([0], initialize=0.8, mutable=True)
    exp.model.unknown_parameters.clear()
    exp.model.unknown_parameters[exp.model.coefficient[0]] = ComponentUID(
        exp.model.coefficient[0]
    )
    with pytest.raises(ValueError, match="indexed Param"):
        calibration(exp)


@ipopt
def test_unweighted_sse_supported():
    fit = calibration(objective="SSE").fit()
    assert fit.execution == "completed", fit.diagnostics  # nosec B101
    assert fit.estimates[0].value == pytest.approx(0.74)  # nosec B101


@pytest.mark.parametrize(
    "failure",
    [
        "fixed_condition",
        "equation",
        "bound",
        "objective",
        "missing_parameter",
        "missing_variables",
        "missing_experiment",
        "nonfinite",
    ],
)
def test_invalid_solver_candidates_stay_unresolved(monkeypatch, failure):
    objective, theta, rows = candidate(None)
    if failure == "fixed_condition":
        rows[0]["fuel"] = 11
    elif failure == "equation":
        rows[0]["steam"] = 8
    elif failure == "bound":
        theta["eta"] = rows[0]["eta"] = 1.1
        rows[0]["steam"] = 11
    elif failure == "objective":
        objective = 10
    elif failure == "missing_parameter":
        theta = {}
    elif failure == "missing_variables":
        del rows[0]["steam"]
    elif failure == "missing_experiment":
        rows = []
    else:
        rows[0]["steam"] = float("nan")
    monkeypatch.setattr(bridge, "_estimate", lambda study: (objective, theta, rows))
    fit = calibration().fit()
    assert fit.execution == "unresolved"  # nosec B101
    assert fit.estimates == ()  # nosec B101
    assert fit.diagnostics[0].code == "calibration_candidate_check_failed"  # nosec B101
    assert CalibrationResult.from_json(fit.to_json()) == fit  # nosec B101
    with pytest.raises(ValueError, match="Only completed"):
        fit.apply_to(target_adapter(), parameter_map={"eta": "eta"})


@pytest.mark.parametrize(
    "error, code",
    [
        (FileNotFoundError("ipopt"), "calibration_solver_unavailable"),
        (RuntimeError("convergence"), "calibration_solve_failed"),
    ],
)
def test_solver_failures_are_portable(monkeypatch, error, code):
    def fail(study):
        raise error

    monkeypatch.setattr(bridge, "_estimate", fail)
    result = calibration().fit()
    assert result.execution == "unresolved"  # nosec B101
    assert result.solver["api_outcome"] == "raised"  # nosec B101
    assert result.diagnostics[0].code == code  # nosec B101
    assert CalibrationResult.from_json(result.to_json()) == result  # nosec B101


def test_explicit_application_creates_new_model_and_preserves_claim(monkeypatch):
    monkeypatch.setattr(bridge, "_estimate", candidate)
    fit = calibration().fit()
    original = target_adapter()
    before = original.base_contract.ref
    calibrated = fit.apply_to(original, parameter_map={"eta": "eta"})
    assert pyo.value(original._template.eta) == 0.8  # nosec B101
    assert pyo.value(calibrated._template.eta) == 0.74  # nosec B101
    assert original.base_contract.ref == before  # nosec B101
    assert calibrated.base_contract.ref != before  # nosec B101
    assert calibrated.input_space == original.input_space  # nosec B101
    assert calibrated.requirements == original.requirements  # nosec B101
    assert calibrated.provenance["calibration"]["result_ref"] == fit.ref.to_dict()  # nosec B101
    assert (
        calibrated.provenance["calibration"]["parent_contract_ref"] == before.to_dict()
    )  # nosec B101
    with pytest.raises(ValueError, match="exactly"):
        fit.apply_to(original, parameter_map={"another": "eta"})
    with pytest.raises(ValueError, match="explicit"):
        fit.apply_to(original, parameter_map={})
    for target in ("fuel", "demand"):
        with pytest.raises(ValueError, match="scenario inputs or actuator"):
            fit.apply_to(original, parameter_map={"eta": target})
    with pytest.raises(ValueError, match="native units"):
        fit.apply_to(original, parameter_map={"eta": "steam"})
    modified = original._template.clone()
    modified.eta.unfix()
    with pytest.raises(ValueError, match="must be fixed"):
        fit.apply_to(replace(original, model=modified), parameter_map={"eta": "eta"})
    modified.eta.fix()
    modified.eta.setlb(0.75)
    with pytest.raises(ValueError, match="target bounds"):
        fit.apply_to(replace(original, model=modified), parameter_map={"eta": "eta"})


def test_portable_record_rejects_tampering_or_incomplete_completed_result(monkeypatch):
    monkeypatch.setattr(bridge, "_estimate", candidate)
    fit = calibration().fit()
    data = json.loads(fit.to_json())
    data["estimates"][0]["value"] = 0.75
    with pytest.raises(ValueError, match="integrity"):
        CalibrationResult.from_json(json.dumps(data))
    with pytest.raises(ValueError, match="experiment results"):
        replace(fit, experiments=(replace(fit.experiments[0], name="wrong"),))
    with pytest.raises(ValueError, match="physical check records"):
        replace(fit, experiments=(replace(fit.experiments[0], checks=()),))
    with pytest.raises(ValueError, match="applicable estimates"):
        replace(fit, execution="unresolved")


def test_application_supports_mutable_params_and_rejects_aliases(monkeypatch):
    from examples.idaes_flowsheet import build_flowsheet, bind_flowsheet

    monkeypatch.setattr(bridge, "_estimate", candidate)
    fit = calibration().fit()
    original = bind_flowsheet(build_flowsheet())
    calibrated = fit.apply_to(
        original, parameter_map={"eta": "fs.boiler[a].efficiency"}
    )
    assert pyo.value(calibrated._template.fs.boiler["a"].efficiency) == 0.74  # nosec B101
    assert pyo.value(original._template.fs.boiler["a"].efficiency) == 0.8  # nosec B101

    simple = target_adapter()
    model = simple._template.clone()
    model.gain = pyo.Param([0], initialize=0.8, mutable=True)
    model.balance.set_value(model.steam == model.gain[0] * model.fuel)
    indexed = fit.apply_to(
        replace(simple, model=model), parameter_map={"eta": "gain[0]"}
    )
    assert pyo.value(indexed._template.gain[0]) == 0.74  # nosec B101

    exp = experiment()
    exp.model.other = pyo.Var(initialize=0.74)
    exp.model.unknown_parameters[exp.model.other] = ComponentUID(exp.model.other)
    objective, theta, rows = candidate(None)
    theta["other"] = rows[0]["other"] = 0.74
    monkeypatch.setattr(bridge, "_estimate", lambda study: (objective, theta, rows))
    two = calibration(exp).fit()
    assert two.execution == "completed", two.diagnostics  # nosec B101
    with pytest.raises(ValueError, match="alias"):
        two.apply_to(
            original,
            parameter_map={
                "eta": "fs.boiler[a].efficiency",
                "other": "fs.boiler[a].efficiency",
            },
        )


@ipopt
def test_boundary_fit_is_explicitly_diagnosed():
    result = calibration(experiment(measured=11)).fit()
    assert result.execution == "completed", result.diagnostics  # nosec B101
    assert result.estimates[0].value == pytest.approx(1)  # nosec B101
    assert any(d.code == "parameter_at_bound" for d in result.diagnostics)  # nosec B101


def test_missing_executable_is_handled_by_actual_execution_bridge(monkeypatch):
    from types import SimpleNamespace

    study = calibration()
    monkeypatch.setattr(
        pyo,
        "SolverFactory",
        lambda name: SimpleNamespace(available=lambda **kwargs: False),
    )
    result = study.fit()
    assert result.execution == "unresolved"  # nosec B101
    assert result.diagnostics[0].code == "calibration_solver_unavailable"  # nosec B101


@ipopt
def test_heat_exchanger_calibration_preserves_known_area_and_reaudits_same_cases():
    from examples import idaes_heat_exchanger as hx

    fit, before, after = hx.calibrated_study()
    assert fit.execution == "completed", fit.diagnostics  # nosec B101
    assert fit.estimates[0].component == "fs.clean_heat_transfer"  # nosec B101
    assert fit.estimates[0].value == pytest.approx(85.00986, abs=0.001)  # nosec B101
    for result in fit.experiments:
        assert result.values["fs.hx.area"] == 1  # nosec B101
        assert all(abs(o.residual) < 0.1 for o in result.observations)  # nosec B101
        assert all(c.satisfied for c in result.checks)  # nosec B101
    baseline = {s["scenario"]: s for s in hx.summary(before)["scenarios"]}
    calibrated = {s["scenario"]: s for s in hx.summary(after)["scenarios"]}
    assert set(baseline) == set(calibrated)  # nosec B101
    assert baseline["normal"]["feasibility"] == "feasible"  # nosec B101
    assert calibrated["normal"]["feasibility"] == "unknown"  # nosec B101
    assert after.payload.verdict == "inconclusive"  # nosec B101
    assert after.payload.witness is None  # nosec B101
    assert CalibrationResult.from_json(fit.to_json()) == fit  # nosec B101
