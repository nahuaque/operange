"""Joint bootstrap vectors must survive sampling, replay and operating audits."""

# ruff: noqa: E402
from dataclasses import replace
import json

import numpy as np
import pytest

pytest.importorskip("idaes")
pyo = pytest.importorskip("pyomo.environ")
pytest.importorskip("pandas")

from examples import idaes_uncertainty as example
from operange import (
    Coordinate,
    ParameterSpace,
    FiniteSet,
    Scenario,
    result_from_json,
    verify_result,
)
from operange.integrations.parmest import CalibrationBootstrap, ParmestCalibration
from operange.integrations import _parmest_bootstrap as bridge
from operange.integrations._idaes_model import model_manifest

ipopt = pytest.mark.skipif(
    not pyo.SolverFactory("ipopt").available(False), reason="Ipopt executable required"
)


@pytest.fixture(scope="module")
def joint():
    if not pyo.SolverFactory("ipopt").available(False):
        pytest.skip("Ipopt executable required")
    return example.campaign().bootstrap(samples=8, seed=7)


def cases(adapter):
    return FiniteSet(
        adapter.input_space,
        (
            Scenario("normal", {"demand": 9}, "Normal demand"),
            Scenario("near_limit", {"demand": 9.8}, "Near nominal limit"),
        ),
    )


def rows_from(result):
    names = [e["name"] for e in result.calibration.study["experiments"]]
    return [
        {
            **{p.component: p.value for p in d.estimates},
            "samples": [names.index(n) for n in d.experiment_names],
        }
        for d in result.draws
    ]


def frozen_study(monkeypatch, result):
    study = example.campaign()
    monkeypatch.setattr(ParmestCalibration, "fit", lambda self: result.calibration)
    monkeypatch.setattr(bridge, "sample", lambda study, count, seed: rows_from(result))
    monkeypatch.setattr(
        bridge,
        "replay",
        lambda study, fitted, draw, indices, estimates: next(
            d.replay for d in result.draws if d.name == draw
        ),
    )
    return study


def test_joint_draws_match_resampled_analytic_fits_and_keep_covariance(joint):
    assert joint.execution == "completed", joint.diagnostics  # nosec B101
    originals = {e["name"]: e for e in joint.calibration.study["experiments"]}
    for draw in joint.draws:
        x, y = [], []
        for name in draw.experiment_names:
            e = originals[name]
            fuel = next(
                v["value"]
                for v in e["source"]["variables"]
                if v["component"] == "fs.boiler.fuel"
            )
            x.append([fuel, -1])
            y.append(e["observations"][0]["measured"])
        expected = np.linalg.lstsq(x, y, rcond=None)[0]
        assert [p.value for p in draw.estimates] == pytest.approx(expected, abs=1e-7)  # nosec B101
        assert draw.checked  # nosec B101
        assert all(c.satisfied for e in draw.replay.experiments for c in e.checks)  # nosec B101
        assert draw.replay.estimates == ()  # nosec B101
    covariance = np.asarray(joint.statistics["covariance"])
    correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    assert correlation > 0.95  # nosec B101
    assert joint.scope["confidence_level"] is None  # nosec B101
    domain = joint.as_domain()
    assert len(domain.scenarios) == 8  # nosec B101
    assert [dict(s.values) for s in domain.scenarios] == [
        {p.component: p.value for p in d.estimates} for d in joint.draws
    ]  # nosec B101
    assert CalibrationBootstrap.from_json(joint.to_json()) == joint  # nosec B101


@ipopt
def test_seed_reproducibility_and_numpy_state_are_preserved(joint):
    state = np.random.get_state()
    study = example.campaign()
    before = tuple(model_manifest(s.model) for s in study._experiments)
    repeated = study.bootstrap(samples=8, seed=7)
    after = np.random.get_state()
    assert (
        state[0] == after[0]
        and np.array_equal(state[1], after[1])
        and state[2:] == after[2:]
    )  # nosec B101
    assert repeated == joint  # nosec B101
    assert tuple(model_manifest(s.model) for s in study._experiments) == before  # nosec B101


def test_propagation_requires_explicit_dispatch_and_preserves_whole_vectors(joint):
    original = example.operating_model()
    original_ref = original.base_contract.ref
    with pytest.raises(ValueError, match="Supply fixed_controls"):
        joint.as_claim(original, cases(original), parameter_map=example.PARAMETER_MAP)
    claim = joint.as_claim(
        original,
        cases(original),
        parameter_map=example.PARAMETER_MAP,
        fixed_controls={"fuel": 13.5},
    )
    assert len(claim.domain.scenarios) == 16  # nosec B101
    assert claim.recourse.mode == "fixed"  # nosec B101
    assert claim.recourse.rules[0].observes == ()  # nosec B101
    assert claim.recourse.rules[0].fixed_value == 13.5  # nosec B101
    expected = {tuple(p.value for p in d.estimates) for d in joint.draws}
    actual = {
        tuple(s.values[f"parameter:{p.component}"] for p in joint.calibration.estimates)
        for s in claim.domain.scenarios
    }
    assert actual == expected  # nosec B101
    assert original.base_contract.ref == original_ref  # nosec B101
    assert original.input_space.names == ("demand",)  # nosec B101
    assert pyo.value(original._template.fs.boiler.efficiency) == 0.8  # nosec B101
    assert claim.requirements == original.as_claim(cases(original)).requirements  # nosec B101
    evidence = claim.adapter.provenance["joint_calibration"]
    assert evidence["uncertainty_ref"] == joint.ref.to_dict()  # nosec B101
    assert len(evidence["scenario_index"]) == 16  # nosec B101
    ideal = joint.as_claim(
        original,
        cases(original),
        parameter_map=example.PARAMETER_MAP,
        fully_observed=True,
    )
    assert set(ideal.recourse.rules[0].observes) == set(ideal.domain.space.names)  # nosec B101
    assert (
        ideal.adapter.provenance["joint_calibration"]["dispatch"]
        == "ideal_fully_observed_parameters"
    )  # nosec B101
    with pytest.raises(ValueError, match="not both"):
        joint.as_claim(
            original,
            cases(original),
            parameter_map=example.PARAMETER_MAP,
            fixed_controls={"fuel": 13.5},
            fully_observed=True,
        )


def test_explicit_operating_spec_never_expands_latent_parameter_observations(joint):
    from operange import OperatingSpec

    adapter = example.operating_model()
    options = {"parameter_map": example.PARAMETER_MAP}
    fixed = OperatingSpec.fixed(adapter.controls, {"fuel": 13.5})
    claim = joint.as_claim(adapter, cases(adapter), operating=fixed, **options)
    assert claim.recourse == fixed.policy  # nosec B101
    with pytest.raises(ValueError, match="observe"):
        joint.as_claim(
            adapter,
            cases(adapter),
            operating=OperatingSpec.dispatch(
                adapter.controls, adapter.input_space.names
            ),
            **options,
        )
    observed = OperatingSpec.dispatch(adapter.controls, claim.domain.space.names)
    ideal = joint.as_claim(adapter, cases(adapter), operating=observed, **options)
    assert ideal.recourse == observed.policy  # nosec B101
    shared = joint.optimize_shared_controls(
        adapter,
        cases(adapter),
        operating=OperatingSpec.shared(adapter.controls),
        **options,
    )
    assert shared.execution == "completed", shared.to_json()  # nosec B101
    assert all(not r.observes for r in shared.operating_policy.rules)  # nosec B101


@ipopt
def test_joint_operating_audit_retains_unknowns_and_portable_finite_scope(joint):
    adapter = example.operating_model()
    claim = joint.as_claim(
        adapter,
        cases(adapter),
        parameter_map=example.PARAMETER_MAP,
        fixed_controls={"fuel": 13.5},
    )
    audit = claim.audit_result()
    evaluated = {e.result_id: e for e in audit.supporting_evaluations}
    support = audit.payload.coverage.evaluated_support["scenarios"]
    assert len(support) == 16  # nosec B101
    states = [
        evaluated[s["evaluation_ref"]["artifact_id"]].payload.feasibility
        for s in support
    ]
    assert "feasible" in states and "unknown" in states  # nosec B101
    assert audit.payload.verdict == "inconclusive"  # nosec B101
    assert audit.payload.witness is None  # nosec B101
    assert verify_result(audit).status == "unsupported"  # nosec B101
    assert result_from_json(audit.to_json(compact=True)).result_id == audit.result_id  # nosec B101
    for evaluation in evaluated.values():
        if evaluation.payload.feasibility == "feasible":
            values = {v.quantity_ref: v.value for v in evaluation.payload.values}
            assert values["fuel"] == 13.5  # nosec B101
            assert values["steam"] == pytest.approx(
                values["parameter:fs.boiler.efficiency"] * 13.5
                - values["parameter:fs.boiler.heat_loss"]
            )  # nosec B101


@pytest.mark.parametrize(
    "problem",
    [
        "solver",
        "missing_rows",
        "missing_coordinate",
        "nonfinite",
        "bad_indices",
        "bad_bound",
        "replay",
    ],
)
def test_failures_are_not_dropped_or_propagated(monkeypatch, joint, problem):
    study = frozen_study(monkeypatch, joint)
    rows = rows_from(joint)
    if problem == "solver":

        def fail(*args):
            raise RuntimeError("upstream solve failed")

        monkeypatch.setattr(bridge, "sample", fail)
    elif problem == "missing_rows":
        monkeypatch.setattr(bridge, "sample", lambda *args: rows[:-1])
    elif problem == "replay":
        replay = replace(joint.draws[0].replay, execution="unresolved")
        monkeypatch.setattr(bridge, "replay", lambda *args: replay)
    else:
        if problem == "missing_coordinate":
            del rows[0]["fs.boiler.heat_loss"]
        elif problem == "nonfinite":
            rows[0]["fs.boiler.heat_loss"] = float("nan")
        elif problem == "bad_bound":
            rows[0]["fs.boiler.efficiency"] = 1.5
        else:
            rows[0]["samples"] = [99] * 6
        monkeypatch.setattr(bridge, "sample", lambda *args: rows)
    result = study.bootstrap(samples=8, seed=7)
    assert result.execution == "unresolved"  # nosec B101
    assert len(result.draws) == (0 if problem in ("solver", "missing_rows") else 8)  # nosec B101
    assert CalibrationBootstrap.from_json(result.to_json()) == result  # nosec B101
    with pytest.raises(ValueError, match="every requested"):
        result.as_domain()
    with pytest.raises(ValueError, match="every requested"):
        _ = result.statistics
    with pytest.raises(ValueError, match="every requested"):
        result.as_claim(
            example.operating_model(), None, parameter_map=example.PARAMETER_MAP
        )


def test_duplicate_draws_keep_multiplicity_without_new_parameter_combinations(
    monkeypatch, joint
):
    study = frozen_study(monkeypatch, joint)
    row = rows_from(joint)[0]
    monkeypatch.setattr(bridge, "sample", lambda *args: [row] * 8)
    monkeypatch.setattr(bridge, "replay", lambda *args: joint.draws[0].replay)
    result = study.bootstrap(samples=8, seed=7)
    assert result.execution == "completed"  # nosec B101
    assert len(result.as_domain().scenarios) == 8  # nosec B101
    assert len({tuple(s.values.values()) for s in result.as_domain().scenarios}) == 1  # nosec B101
    assert np.asarray(result.statistics["covariance"]) == pytest.approx(
        np.zeros((2, 2)), abs=1e-28
    )  # nosec B101


def test_target_mapping_and_domain_validation_apply_to_every_draw(joint):
    adapter = example.operating_model()
    with pytest.raises(ValueError, match="exactly"):
        joint.as_claim(
            adapter,
            cases(adapter),
            parameter_map={"fs.boiler.efficiency": "fs.boiler.efficiency"},
            fixed_controls={"fuel": 13.5},
        )
    wrong_units = {**example.PARAMETER_MAP, "fs.boiler.efficiency": "fs.boiler.steam"}
    with pytest.raises(ValueError, match="native units"):
        joint.as_claim(
            adapter,
            cases(adapter),
            parameter_map=wrong_units,
            fixed_controls={"fuel": 13.5},
        )
    wrong_target = {**example.PARAMETER_MAP, "fs.boiler.heat_loss": "fs.demand"}
    with pytest.raises(ValueError, match="scenario inputs"):
        joint.as_claim(
            adapter,
            cases(adapter),
            parameter_map=wrong_target,
            fixed_controls={"fuel": 13.5},
        )
    other_cases = FiniteSet(
        ParameterSpace((Coordinate("other", "MW", "Wrong input"),)),
        (Scenario("x", {"other": 9}, "Wrong domain"),),
    )
    with pytest.raises(ValueError):
        joint.as_claim(
            adapter,
            other_cases,
            parameter_map=example.PARAMETER_MAP,
            fixed_controls={"fuel": 13.5},
        )
    model = adapter._template.clone()
    nominal = joint.calibration.estimates[0].value
    extreme = max(d.estimates[0].value for d in joint.draws)
    assert extreme > nominal  # nosec B101
    model.fs.boiler.efficiency.setub((nominal + extreme) / 2)
    narrowed = replace(adapter, model=model)
    with pytest.raises(ValueError, match="target bounds"):
        joint.as_claim(
            narrowed,
            cases(narrowed),
            parameter_map=example.PARAMETER_MAP,
            fixed_controls={"fuel": 13.5},
        )


def test_portable_records_reject_missing_draws_and_tampering(joint):
    with pytest.raises(ValueError, match="every returned draw"):
        replace(joint, draws=joint.draws[:-1])
    with pytest.raises(ValueError, match="execution"):
        replace(joint, execution="unresolved")
    with pytest.raises(ValueError, match="qualifications"):
        replace(joint, scope={"confidence_level": 0.95})
    data = json.loads(joint.to_json())
    data["seed"] = 8
    with pytest.raises(ValueError, match="integrity"):
        CalibrationBootstrap.from_json(json.dumps(data))
    changed = replace(
        joint.draws[0],
        estimates=(
            replace(joint.draws[0].estimates[0], value=0.6),
            joint.draws[0].estimates[1],
        ),
    )
    with pytest.raises(ValueError, match="parameter vector"):
        replace(joint, draws=(changed,) + joint.draws[1:])


@pytest.mark.parametrize(
    "samples, seed", [(0, 7), (True, 7), (2, -1), (2, 2**32), (2, True)]
)
def test_invalid_requests_fail_before_any_solve(monkeypatch, samples, seed):
    study = example.campaign()
    monkeypatch.setattr(
        ParmestCalibration, "fit", lambda self: pytest.fail("solve should not run")
    )
    with pytest.raises(ValueError):
        study.bootstrap(samples=samples, seed=seed)


def test_unresolved_initial_fit_never_starts_sampling(monkeypatch, joint):
    study = example.campaign()
    failed = replace(joint.calibration, execution="unresolved", estimates=())
    monkeypatch.setattr(ParmestCalibration, "fit", lambda self: failed)
    monkeypatch.setattr(
        bridge, "sample", lambda *args: pytest.fail("sampling should not run")
    )
    result = study.bootstrap(samples=8, seed=7)
    assert result.execution == "unresolved" and not result.draws  # nosec B101
    assert result.diagnostics[0].code == "bootstrap_calibration_unresolved"  # nosec B101


def test_reconciliation_and_too_few_experiments_are_not_parameter_uncertainty(
    monkeypatch,
):
    from pyomo.contrib.parmest.experiment import Experiment

    source = example.campaign()
    experiments = {s.name: Experiment(s.model.clone()) for s in source._experiments[:2]}
    sparse = ParmestCalibration(
        name="Insufficient experiments", experiments=experiments
    )
    monkeypatch.setattr(
        ParmestCalibration, "fit", lambda self: pytest.fail("fit should not run")
    )
    with pytest.raises(ValueError, match="more experiments"):
        sparse.bootstrap(samples=4, seed=7)
    for exp in experiments.values():
        exp.model.unknown_parameters.clear()
    recon = ParmestCalibration(name="Reconciliation", experiments=experiments)
    with pytest.raises(ValueError, match="reconciliation"):
        recon.bootstrap(samples=4, seed=7)


def test_rng_restored_even_when_upstream_batch_raises(monkeypatch):
    from types import SimpleNamespace

    def fail(*args, **kwargs):
        np.random.seed(2)
        np.random.random(10)
        raise RuntimeError("failed batch")

    monkeypatch.setattr(
        bridge, "estimator", lambda study: SimpleNamespace(theta_est_bootstrap=fail)
    )
    state = np.random.get_state()
    with pytest.raises(RuntimeError, match="batch"):
        bridge.sample(SimpleNamespace(_experiments=(1, 2, 3)), 4, 7)
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]  # nosec B101


@ipopt
def test_indexed_mutable_parameter_targets_are_operational(joint):
    adapter = example.operating_model()
    m = adapter._template.clone()
    m.fs.coefficient = pyo.Param([0], mutable=True, initialize=0.8)
    m.fs.boiler.balance.set_value(
        m.fs.boiler.steam
        == m.fs.coefficient[0] * m.fs.boiler.fuel - m.fs.boiler.heat_loss
    )
    adapter = replace(adapter, model=m)
    mapping = {**example.PARAMETER_MAP, "fs.boiler.efficiency": "fs.coefficient[0]"}
    claim = joint.as_claim(
        adapter, cases(adapter), parameter_map=mapping, fixed_controls={"fuel": 13.5}
    )
    normal = next(s for s in claim.domain.scenarios if s.values["demand"] == 9)
    evaluation = claim.evaluate_result(normal.values)
    assert evaluation.execution == "completed"  # nosec B101
    assert evaluation.payload.feasibility == "feasible"  # nosec B101
    assert pyo.value(adapter._template.fs.coefficient[0]) == 0.8  # nosec B101


def test_fixed_commands_cannot_hide_unspecified_actuators(joint):
    adapter = example.operating_model()
    with pytest.raises(ValueError):
        joint.as_claim(
            adapter,
            cases(adapter),
            parameter_map=example.PARAMETER_MAP,
            fixed_controls={"fuel": None},
        )


def test_joint_draws_choose_one_minimum_fuel_command_and_can_be_reaudited(joint):
    adapter = example.operating_model()
    before = model_manifest(adapter._template)
    result = joint.optimize_shared_controls(
        adapter, cases(adapter), parameter_map=example.PARAMETER_MAP
    )
    expected = max(
        (9.8 + dict((p.component, p.value) for p in d.estimates)["fs.boiler.heat_loss"])
        / dict((p.component, p.value) for p in d.estimates)["fs.boiler.efficiency"]
        for d in joint.draws
    )
    assert result.execution == "completed", result.diagnostics  # nosec B101
    assert result.commands["fuel"] == pytest.approx(expected, abs=1e-6)  # nosec B101
    assert len(result.scenario_objectives) == 16  # nosec B101
    assert model_manifest(adapter._template) == before  # nosec B101
    provenance = result.contract.model["provenance"]["joint_calibration"]
    assert provenance["uncertainty_ref"] == joint.ref.to_dict()  # nosec B101
    assert provenance["dispatch"] == "optimized_shared_actuator_commands"  # nosec B101
    assert len(provenance["scenario_index"]) == 16  # nosec B101
    replay = joint.as_claim(
        adapter,
        cases(adapter),
        parameter_map=example.PARAMETER_MAP,
        fixed_controls=result.commands,
    ).audit_result()
    assert replay.payload.verdict == "pass", replay.diagnostics  # nosec B101


@ipopt
def test_shared_command_example_compares_nominal_and_joint_decisions():
    uncertainty, nominal, nominal_audit, shared = example.run_shared_example()
    assert uncertainty.execution == "completed"  # nosec B101
    assert shared.commands["fuel"] > nominal.commands["fuel"]  # nosec B101
    assert nominal_audit.payload.verdict == "inconclusive"  # nosec B101
    assert shared.audit.payload.verdict == "pass"  # nosec B101
