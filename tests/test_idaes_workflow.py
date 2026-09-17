"""Real unit/property integration and failures that must remain unresolved."""

# Optional backend imports follow importorskip, as in test_idaes.py.
# ruff: noqa: E402

from dataclasses import replace
import json
import subprocess  # nosec B404
import sys

import pytest

pytest.importorskip("idaes")
pyo = pytest.importorskip("pyomo.environ")
pytest.importorskip("highspy")

from idaes.core.initialization import InitializationStatus
from idaes.models.unit_models.skeleton_model import SkeletonUnitModelData

from examples.idaes_flowsheet import build_flowsheet, bind_flowsheet
from examples import idaes_heat_exchanger as hx
from operange import FiniteSet, Scenario, result_from_json, verify_result
from operange.integrations import _idaes_execution
from operange.integrations._idaes_model import model_manifest
from operange.integrations.idaes import IdaesInitialization


ipopt = pytest.mark.skipif(
    not pyo.SolverFactory("ipopt").available(False), reason="Ipopt executable required"
)


def claim_for(adapter):
    return adapter.as_claim(
        FiniteSet(
            adapter.input_space,
            (Scenario("normal", adapter.input_space.nominal, "Normal operation"),),
        )
    )


def boiler_adapter(model=None, **kwargs):
    model = build_flowsheet() if model is None else model
    return replace(bind_flowsheet(model), model=model, **kwargs)


def reports(result):
    return {e.evidence_id: e.details for e in result.evidence}


def values(result):
    return {v.quantity_ref: v.value for v in result.payload.values}


@ipopt
def test_real_heat_exchanger_study_conserves_mass_energy_and_keeps_finite_scope():
    source = hx.build_flowsheet()
    adapter = hx.bind_flowsheet(source)
    before = model_manifest(source)
    result = hx.study(adapter)
    summary = hx.summary(result)
    by_name = {r["scenario"]: r for r in summary["scenarios"]}
    assert result.payload.verdict == "inconclusive"  # nosec B101
    assert result.payload.witness is None  # nosec B101
    for name in ("normal", "warmer_feed", "lower_demand", "higher_throughput"):
        assert by_name[name]["feasibility"] == "feasible", by_name[name]  # nosec B101
    assert by_name["normal"]["outputs"]["utility_flow"] == pytest.approx(
        3.2776892, abs=1e-5
    )  # nosec B101
    assert by_name["normal"]["outputs"]["duty"] == pytest.approx(4.5, abs=1e-7)  # nosec B101
    for name in ("fouled", "peak_demand"):
        row = by_name[name]
        assert row["feasibility"] == "unknown"  # nosec B101
        assert row["diagnostics"][0]["code"] == "idaes_solver_failed"  # nosec B101
        assert row["outputs"] == {}  # nosec B101
        assert row["service_violations_at_diagnostic_point"]  # nosec B101
        assert row["diagnostic_point"] == "prepared_start_without_solver_solution"  # nosec B101
    for evaluation in result.supporting_evaluations:
        if evaluation.payload.feasibility != "feasible":
            continue
        v = values(evaluation)
        duties = []
        for side, inlet_flow in (
            ("hot_side", v["utility_flow"]),
            ("cold_side", v["process_flow"]),
        ):
            prefix = f"state:fs.hx.{side}.properties_"
            flow_in = inlet_flow
            flow_out = v[prefix + "out[0.0].flow_mol"]
            assert flow_out == pytest.approx(flow_in, abs=1e-8)  # nosec B101
            for species in ("benzene", "toluene"):
                assert flow_out * v[
                    prefix + f"out[0.0].mole_frac_comp[{species}]"
                ] == pytest.approx(0.5 * flow_in, abs=1e-8)  # nosec B101
            enthalpy_in = v[prefix + "in[0.0].enth_mol_phase[Liq]"]
            enthalpy_out = v[prefix + "out[0.0].enth_mol_phase[Liq]"]
            duties.append(flow_out * enthalpy_out - flow_in * enthalpy_in)
        assert sum(duties) == pytest.approx(0, abs=1e-5)  # nosec B101
        assert duties[1] == pytest.approx(1000 * v["duty"], abs=1e-5)  # nosec B101
        report = reports(evaluation)
        assert report["initialization:0"]["degrees_of_freedom"] == 0  # nosec B101
        assert report["diagnostics:checked_candidate"]["violated_rows_at_iterate"] == ()  # nosec B101
    assert model_manifest(source) == before  # nosec B101
    assert result_from_json(result.to_json(compact=True)).to_dict() == result.to_dict()  # nosec B101
    assert verify_result(result).status == "unsupported"  # nosec B101
    json.dumps(summary, allow_nan=False)
    # The exported workflow/diagnostic reports remain readable without loading
    # any modelling backend, even when Pyomo and IDAES imports are blocked.
    script = """
import importlib.abc
import sys
class BlockBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'idaes', 'pyomo', 'highspy'}:
            raise ImportError('optional backend unavailable')
sys.meta_path.insert(0, BlockBackend())
from operange import result_from_json
result = result_from_json(sys.stdin.read())
if result.payload.verdict != 'inconclusive':
    raise RuntimeError('wrong audit verdict')
if len(result.supporting_evaluations) != 6:
    raise RuntimeError('missing scenarios')
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        input=result.to_json(compact=True),
        text=True,
        check=True,
        capture_output=True,
    )  # nosec B603
    with pytest.raises(ValueError):
        adapter.export_linear()


def test_missing_equipment_specification_is_diagnosed_before_solving(monkeypatch):
    source = hx.build_flowsheet()
    source.fs.hx.area.unfix()
    adapter = hx.bind_flowsheet(source)

    def unexpected(*args):
        raise AssertionError("final solver must not run after failed initialization")

    monkeypatch.setattr(_idaes_execution, "solve", unexpected)
    result = claim_for(adapter).evaluate_result()
    assert result.payload.feasibility == "unknown"  # nosec B101
    assert result.diagnostics[0].code == "idaes_initialization_specification"  # nosec B101
    assert reports(result)["initialization:0"]["degrees_of_freedom"] == 1  # nosec B101
    structure = reports(result)["diagnostics:before_initialization"]
    assert any(
        "fs.hx.area" in block
        for block in structure["structural_partition"]["underconstrained_variables"]
    )  # nosec B101


@ipopt
def test_initialization_failure_is_portable_and_not_a_counterexample():
    model = hx.build_flowsheet()
    original = hx.bind_flowsheet(model)
    adapter = replace(
        original,
        model=model,
        initialization=(
            IdaesInitialization(
                "fs.hx",
                options={"solver": "ipopt", "optarg": {"max_iter": 0}, "outlvl": 40},
            ),
        ),
    )
    result = claim_for(adapter).audit_result()
    evaluation = result.supporting_evaluations[0]
    assert result.payload.verdict == "inconclusive"  # nosec B101
    assert result.payload.witness is None  # nosec B101
    assert evaluation.diagnostics[0].code == "idaes_initialization_failed"  # nosec B101
    assert "error" in reports(evaluation)["initialization:0"]  # nosec B101
    assert not any(v.quantity_ref == "duty" for v in evaluation.payload.values)  # nosec B101
    assert result_from_json(result.to_json()).result_id == result.result_id  # nosec B101


def test_initialization_cannot_change_source_equations_fixed_values_or_permissions(
    monkeypatch,
):
    calls = []

    def destructive_initializer(block, **options):
        calls.append((block.name, options))
        # Upstream routines temporarily fix variables and deactivate equations.
        block.fuel.fix(0)
        block.fuel.setub(1000)
        block.efficiency.set_value(100)
        block.balance.deactivate()
        block.steam.fix(999)
        block.model().fs.service.deactivate()

    monkeypatch.setattr(SkeletonUnitModelData, "initialize", destructive_initializer)
    model = build_flowsheet()
    adapter = boiler_adapter(
        model,
        initialization=(
            IdaesInitialization("fs.boiler[a]", options={"outlvl": 40}),
            IdaesInitialization("fs.boiler[b]"),
        ),
    )
    result = claim_for(adapter).evaluate_result()
    assert result.payload.feasibility == "feasible", result.to_json()  # nosec B101
    assert values(result)["fuel"] == pytest.approx(23)  # nosec B101
    assert [c[0] for c in calls] == ["fs.boiler[a]", "fs.boiler[b]"]  # nosec B101
    assert pyo.value(model.fs.boiler["a"].efficiency) == 0.8  # nosec B101
    assert model.fs.boiler["a"].balance.active  # nosec B101
    assert "service" in result.payload.included_constraints  # nosec B101


def test_default_initializer_api_and_non_success_status(monkeypatch):
    outcomes = [InitializationStatus.Ok, InitializationStatus.Failed]

    class UpstreamInitializer:
        def __init__(self, **options):
            assert options == {"output_level": 40}  # nosec B101

        def initialize(self, block):
            return outcomes.pop(0)

    monkeypatch.setattr(
        SkeletonUnitModelData, "default_initializer", UpstreamInitializer
    )
    adapter = boiler_adapter(
        initialization=(
            IdaesInitialization(
                "fs.boiler[a]",
                method="default",
                options={"output_level": 40},
            ),
        )
    )
    claim = claim_for(adapter)
    assert claim.evaluate_result().payload.feasibility == "feasible"  # nosec B101
    failed = claim.evaluate_result()
    assert failed.diagnostics[0].code == "idaes_initialization_failed"  # nosec B101
    assert failed.payload.feasibility == "unknown"  # nosec B101


def test_diagnostics_remain_advisory_without_asl_and_show_poor_scaling(monkeypatch):
    from pyomo.contrib.pynumero.asl import AmplInterface
    from idaes.core.util.scaling import set_scaling_factor

    monkeypatch.setattr(AmplInterface, "available", lambda: False)
    model = build_flowsheet()
    set_scaling_factor(model.fs.boiler["a"].fuel, 1e9)
    adapter = boiler_adapter(model, diagnostics=True)
    result = claim_for(adapter).evaluate_result()
    assert result.payload.feasibility == "feasible"  # nosec B101
    info = reports(result)["diagnostics:checked_candidate"]
    assert "ASL" in info["numerical_report_unavailable"]  # nosec B101
    assert any(
        v["component"] == "fs.boiler[a].fuel" for v in info["badly_scaled_variables"]
    )  # nosec B101
    assert info["degrees_of_freedom"] == 2  # nosec B101
    assert info["violated_rows_at_iterate"] == ()  # nosec B101


def test_invalid_solver_candidate_has_specific_residual_diagnostics(monkeypatch):
    adapter = boiler_adapter(diagnostics=True)
    monkeypatch.setattr(
        _idaes_execution, "solve", lambda *args: (True, {"termination": "optimal"})
    )
    result = claim_for(adapter).evaluate_result()
    assert result.diagnostics[0].code == "idaes_residual_violation"  # nosec B101
    info = reports(result)["diagnostics:unresolved_candidate_check"]
    assert info["iterate_origin"] == "returned_solver_candidate"  # nosec B101
    assert info["violated_rows_at_iterate"]  # nosec B101
    assert result.payload.feasibility == "unknown"  # nosec B101


def test_failure_of_optional_diagnostics_does_not_hide_a_valid_candidate(monkeypatch):
    try:
        from idaes.core.util.diagnostics_tools import DiagnosticsToolbox
    except ImportError:
        from idaes.core.util.model_diagnostics import DiagnosticsToolbox

    def failed_report(*args, **kwargs):
        raise RuntimeError("diagnostic dependency unavailable")

    monkeypatch.setattr(DiagnosticsToolbox, "report_structural_issues", failed_report)
    result = claim_for(boiler_adapter(diagnostics=True)).evaluate_result()
    assert result.payload.feasibility == "feasible"  # nosec B101
    assert (
        "diagnostic dependency unavailable"
        in reports(result)["diagnostics:checked_candidate"]["error"]
    )  # nosec B101


def test_immutable_indexed_and_lazy_default_parameters_are_bound_and_fingerprinted():
    model = build_flowsheet()
    model.fs.coefficient = pyo.Param(["a", "b"], initialize={"a": 0.8, "b": 0.5})
    model.fs.reference = pyo.Param(default=1, mutable=True)
    for name in ("a", "b"):
        b = model.fs.boiler[name]
        b.balance.set_value(
            b.steam == model.fs.reference * model.fs.coefficient[name] * b.fuel
        )
    adapter = bind_flowsheet(model)
    params = {
        p["component"]: p for p in adapter.base_contract.model["source"]["parameters"]
    }
    assert params["fs.coefficient"]["values"][0]["value"] == 0.8  # nosec B101
    assert params["fs.reference"]["values"][0]["value"] == 1  # nosec B101
    assert claim_for(adapter).evaluate_result().payload.feasibility == "feasible"  # nosec B101
    model.fs.reference.set_value(0.9)
    assert bind_flowsheet(model).base_contract.ref != adapter.base_contract.ref  # nosec B101


def test_scaling_and_workflow_options_affect_contract_identity():
    from idaes.core.util.scaling import set_scaling_factor

    model = build_flowsheet()
    baseline = bind_flowsheet(model)
    changed = boiler_adapter(model, diagnostics=True)
    assert changed.base_contract.ref != baseline.base_contract.ref  # nosec B101
    set_scaling_factor(model.fs.boiler["a"].fuel, 0.1)
    assert (
        bind_flowsheet(model).base_contract.model["source_fingerprint"]
        != baseline.base_contract.model["source_fingerprint"]
    )  # nosec B101
    initialized = boiler_adapter(
        model, initialization=(IdaesInitialization("fs.boiler[a]"),)
    )
    assert initialized.base_contract.ref != bind_flowsheet(model).base_contract.ref  # nosec B101
    assert (
        initialized.base_contract.operating["initialization"][0]["component"]
        == "fs.boiler[a]"
    )  # nosec B101


@pytest.mark.parametrize(
    "step",
    [
        IdaesInitialization("fs.load"),
        IdaesInitialization("fs"),
        IdaesInitialization("absent"),
    ],
)
def test_bad_initialization_bindings_are_rejected(step):
    with pytest.raises(ValueError):
        boiler_adapter(initialization=(step,))


def test_workflow_declarations_are_immutable_and_validate_json_options():
    options = {"optarg": {"tol": 1e-8}}
    step = IdaesInitialization("fs.hx", options=options)
    options["optarg"]["tol"] = 1
    assert step.options["optarg"]["tol"] == 1e-8  # nosec B101
    with pytest.raises(ValueError):
        IdaesInitialization("fs.hx", method="arbitrary_method")
    with pytest.raises(ValueError):
        IdaesInitialization("fs.hx", options={"callback": lambda m: None})
    with pytest.raises(ValueError):
        boiler_adapter(diagnostics="yes")
