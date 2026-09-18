"""Execution and parameter boundaries serve studies without estimator coupling."""

# ruff: noqa: E402
from dataclasses import replace
from pathlib import Path
import ast
import subprocess  # nosec B404
import sys

import numpy as np
import pytest

pytest.importorskip("idaes")
pyo = pytest.importorskip("pyomo.environ")
pytest.importorskip("highspy")

from operange import (
    Coordinate,
    ParameterSpace,
    ParameterPoint,
    ParameterCovariance,
    ParameterSamples,
    Scenario,
    FiniteSet,
    OperatingSpec,
)
from operange.integrations import _idaes_execution as execution
from operange.integrations._idaes_parameter_studies import prepare_joint_samples
from operange.integrations import _idaes_uncertainty as propagation
from operange.integrations import _idaes_results
from operange.integrations._idaes_model import model_manifest
from tests.test_idaes_uncertainty import boiler, PARAMETERS, COVARIANCE, propagate


def parameter_point(**values):
    space = ParameterSpace(
        (Coordinate("eta", "1", "external"), Coordinate("loss", "MW", "external"))
    )
    return ParameterPoint(
        space, {"eta": 0.8, "loss": 1, **values}, {"source": "external estimator"}
    )


def request():
    return execution.OperatingPointRequest({"demand": 8}, {"fuel": 15}, ("service",))


def test_checked_point_owns_original_equations_and_stable_request():
    _, adapter = boiler()
    commands = {"fuel": 15}
    req = execution.OperatingPointRequest({"demand": 8}, commands, ("service",))
    commands["fuel"] = 1
    session = execution.PointExecution(adapter, req)
    session.prepare()
    candidate = session.generate_candidate()
    candidate.model.balance.deactivate()
    checked = session.check_candidate(candidate)
    assert checked.feasible and checked.model.balance.active  # nosec B101
    assert checked.model is not candidate.model  # nosec B101
    assert checked.values["fuel"] == 15  # nosec B101
    assert checked.values["steam"] == pytest.approx(11)  # nosec B101
    candidate.model.steam.set_value(19)
    assert pyo.value(checked.model.steam) == pytest.approx(11)  # nosec B101


def test_external_candidates_require_source_checks_and_solver_return():
    _, adapter = boiler()
    session = execution.PointExecution(adapter, request())
    model = session.prepare()
    with pytest.raises(ValueError, match="returned solver candidate"):
        session.check_candidate(execution.Candidate(model, False, {}))
    candidate = session.generate_candidate()
    candidate.model.fuel.set_value(14)
    with pytest.raises(ValueError, match="fixed input, command or state"):
        session.check_candidate(candidate)
    failed = execution.execute_point(adapter, request(), candidate=candidate)
    assert not failed.feasible and failed.failure.code == "idaes_candidate_error"  # nosec B101
    with pytest.raises(ValueError, match="checked feasible"):
        failed.require_checked()


def test_local_derivatives_use_the_checked_runtime_model(monkeypatch):
    _, adapter = boiler()
    seen = []
    original = _idaes_results.execute_point

    def capture(*args, **kwargs):
        point = original(*args, **kwargs)
        seen.append(point.require_checked().model)
        return point

    derivative = propagation._regular_system

    def differentiate(model, parameters):
        assert model is seen[-1]  # nosec B101
        return derivative(model, parameters)

    monkeypatch.setattr(_idaes_results, "execute_point", capture)
    monkeypatch.setattr(propagation, "_regular_system", differentiate)
    result = propagate(adapter, fixed_controls={"fuel": 15})
    assert result.execution == "completed" and len(seen) == 1  # nosec B101


def test_external_point_covariance_and_dependent_bounds():
    _, adapter = boiler(param=True)
    source = adapter._template.clone()
    source.loss.setub(source.eta * 2 * pyo.units.MW)
    adapter = replace(adapter, model=source)
    before = model_manifest(adapter._template)
    point = parameter_point(eta=0.9, loss=1.75)
    # Loss violates the old bound but satisfies the whole updated vector.
    applied = adapter.with_parameters(
        point, parameter_map={"loss": "loss", "eta": "eta"}
    )
    assert model_manifest(adapter._template) == before  # nosec B101
    assert pyo.value(applied._template.loss) == 1.75  # nosec B101
    assert (
        applied.provenance["parameters"]["point"]["provenance"]["source"]
        == "external estimator"
    )  # nosec B101
    cov = ParameterCovariance(
        ParameterSpace(tuple(reversed(point.space.coordinates))),
        ((0.25, 0.04), (0.04, 0.01)),
    )
    result = applied.propagate_uncertainty(
        parameters=PARAMETERS, covariance=cov, fixed_controls={"fuel": 15}
    )
    assert result.execution == "completed", result.diagnostics  # nosec B101
    np.testing.assert_allclose(result.covariance, COVARIANCE)
    with pytest.raises(ValueError, match="target bounds"):
        adapter.with_parameters(
            parameter_point(eta=0.5, loss=1.75),
            parameter_map={"eta": "eta", "loss": "loss"},
        )


def test_joint_composition_consumes_external_samples_without_losing_duplicates():
    _, adapter = boiler()
    point = parameter_point()
    samples = ParameterSamples(
        point.space,
        (
            Scenario("one", {"eta": 0.8, "loss": 1}, "external"),
            Scenario("two", {"eta": 0.9, "loss": 2}, "external"),
            Scenario("duplicate", {"eta": 0.8, "loss": 1}, "external"),
        ),
    )
    cases = FiniteSet(
        adapter.input_space,
        (
            Scenario("normal", {"demand": 8}, "study"),
            Scenario("high", {"demand": 9}, "study"),
        ),
    )
    study = prepare_joint_samples(
        adapter, point, samples, cases, parameter_map={"loss": "loss", "eta": "eta"}
    )
    assert len(study.domain.scenarios) == 6  # nosec B101
    assert tuple(r["sample"] for r in study.scenario_index) == (
        "one",
        "one",
        "two",
        "two",
        "duplicate",
        "duplicate",
    )  # nosec B101
    mapped = replace(
        adapter, model=study.model, input_space=study.domain.space, inputs=study.inputs
    )
    assert mapped.as_claim(study.domain).audit_result().payload.verdict == "pass"  # nosec B101


def test_parameter_application_and_propagation_do_not_load_parmest_bridge():
    example = Path(__file__).parent.parent.joinpath("examples/local_uncertainty.py")
    script = """
import importlib.abc
import runpy
import sys
class BlockCalibration(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('operange.integrations.parmest','operange.integrations._parmest','operange.integrations._calibration')):
            raise ImportError('calibration integration deliberately blocked')
sys.meta_path.insert(0, BlockCalibration())
example = runpy.run_path(sys.argv[1])
from operange import Coordinate, ParameterSpace, ParameterPoint
point = ParameterPoint(ParameterSpace((Coordinate('eta','1','external'),)), {'eta':0.8})
adapter = example['operating_model']().with_parameters(point, parameter_map={'eta':'fs.boiler.efficiency'})
fixed, optimized = example['run_example']()
if fixed.execution != 'completed' or optimized.execution != 'completed':
    raise RuntimeError('local propagation depends on calibration')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(example)],
        check=True,
        capture_output=True,
        text=True,
    )  # nosec B603


def test_internal_import_boundaries():
    import operange.parameters

    root = Path(operange.parameters.__file__).parent
    prohibited = {
        "parameters.py": ("integrations",),
        "integrations/_idaes_execution.py": (
            "parmest",
            "calibration",
            "_idaes_results",
            "engineering_results",
        ),
        "integrations/_idaes_parameters.py": ("parmest", "calibration"),
        "integrations/_idaes_parameter_studies.py": ("parmest", "calibration"),
        "integrations/_idaes_uncertainty.py": ("parmest", "calibration"),
    }
    for filename, forbidden in prohibited.items():
        tree = ast.parse(root.joinpath(filename).read_text())
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        imports += [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not any(part in name for part in forbidden for name in imports), filename  # nosec B101


def test_operating_spec_preserves_fixed_and_reoptimized_local_semantics():
    _, adapter = boiler()
    fixed = propagate(
        adapter, operating=OperatingSpec.fixed(adapter.controls, {"fuel": 15})
    )
    legacy = propagate(adapter, fixed_controls={"fuel": 15})
    assert fixed.to_json() == legacy.to_json()  # nosec B101
    observed = OperatingSpec.dispatch(adapter.controls, adapter.input_space.names)
    optimized = propagate(
        adapter, operating=observed, steps={"eta": 1e-4, "loss": 1e-3}
    )
    assert optimized.execution == "completed" and optimized.mode == "reoptimized"  # nosec B101
    with pytest.raises(ValueError, match="operating"):
        propagate(adapter, operating=observed, fixed_controls={"fuel": 15})
    with pytest.raises(ValueError, match="observe"):
        propagate(adapter, operating=OperatingSpec.shared(adapter.controls))


def test_shared_operating_spec_preserves_fixed_permissions_and_checks_them(monkeypatch):
    from operange import DecisionRule, RecoursePolicy
    from operange.integrations.idaes import IdaesVariable, IdaesSharedControlResult
    from tests.test_idaes_shared_controls import boiler as shared_boiler

    _, adapter, domain = shared_boiler()
    source = adapter._template.clone()
    source.trim = pyo.Var(initialize=0, bounds=(0, 4), units=pyo.units.MW)
    source.trim.fix()
    source.balance.set_value(
        source.steam == source.efficiency * source.fuel + source.trim - source.loss
    )
    source.objective.set_value(source.fuel + source.trim)
    adapter = replace(
        adapter,
        model=source,
        controls=adapter.controls + (IdaesVariable("trim", "trim", "MW"),),
    )
    spec = OperatingSpec(
        RecoursePolicy(
            "static",
            (
                DecisionRule("fuel", "MW", "operation", fixed_value=9),
                DecisionRule("trim", "MW", "operation"),
            ),
        )
    )
    result = adapter.optimize_shared_controls(domain, operating=spec)
    assert result.execution == "completed", result.to_json()  # nosec B101
    assert result.commands["fuel"] == 9 and result.commands["trim"] == pytest.approx(
        0.8
    )  # nosec B101
    assert result.operating_policy == spec.policy  # nosec B101
    assert IdaesSharedControlResult.from_json(result.to_json()) == result  # nosec B101
    with pytest.raises(ValueError, match="prescribed"):
        replace(
            result,
            operating_policy=RecoursePolicy(
                "fixed",
                (
                    DecisionRule("fuel", "MW", "operation", fixed_value=10),
                    DecisionRule("trim", "MW", "operation", fixed_value=0.8),
                ),
            ),
        )
    with pytest.raises(ValueError, match="observe"):
        adapter.optimize_shared_controls(
            domain,
            operating=OperatingSpec.dispatch(
                adapter.controls, adapter.input_space.names
            ),
        )
    original = execution.solve

    def changed_fixed_command(model, settings):
        solved, details = original(model, settings)
        model.commands[0].value.set_value(9.1)
        return solved, details

    monkeypatch.setattr(execution, "solve", changed_fixed_command)
    failed = adapter.optimize_shared_controls(domain, operating=spec)
    assert failed.execution == "unresolved" and failed.audit is None  # nosec B101
