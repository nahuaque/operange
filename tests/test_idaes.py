"""Source-model semantics, evidence limits, affine equivalence and upstream fits."""

# Optional-backend imports follow importorskip so the base installation works.
# ruff: noqa: E402

import json
from dataclasses import replace

import pytest

pyo = pytest.importorskip("pyomo.environ")
pytest.importorskip("idaes")
pytest.importorskip("highspy")

from examples.idaes_flowsheet import (
    bind_flowsheet,
    build_flowsheet,
    reconcile_and_estimate,
    study,
)
from operange import (
    BoxSet,
    DecisionRule,
    FiniteSet,
    Parameter,
    RecoursePolicy,
    Scenario,
    result_from_json,
    verify_result,
)
from operange.integrations.idaes import (
    IdaesLinearExport,
    IdaesRequirement,
    IdaesVariable,
)
from operange.integrations import _idaes_execution


def domain(adapter, loads=(16, 20)):
    return FiniteSet(
        adapter.input_space,
        tuple(
            Scenario(f"case_{i}", {"load": load}, "Test load")
            for i, load in enumerate(loads)
        ),
    )


def values(result):
    return {v.quantity_ref: v.value for v in result.payload.values}


def test_existing_flowsheet_and_exact_export_agree_and_preserve_source():
    model = build_flowsheet()
    adapter = bind_flowsheet(model)
    before = adapter.base_contract.ref
    model.fs.boiler["a"].efficiency.set_value(0.4)
    model.fs.boiler["a"].fuel.set_value(3)
    cases = domain(adapter)
    numerical = adapter.as_claim(cases).audit_result()
    assert numerical.payload.verdict == "pass", numerical.to_json()  # nosec B101
    assert numerical.payload.coverage.method == "complete_finite"  # nosec B101
    assert [
        values(e)["fuel"] for e in numerical.supporting_evaluations
    ] == pytest.approx([23, 31])  # nosec B101
    assert pyo.value(model.fs.boiler["a"].fuel) == 3  # nosec B101
    assert pyo.value(model.fs.load) == 16  # nosec B101
    assert adapter.base_contract.ref == before  # nosec B101
    assert verify_result(numerical).status == "unsupported"  # nosec B101
    restored = result_from_json(numerical.to_json(compact=True))
    assert restored.to_dict() == numerical.to_dict()  # nosec B101
    exported = adapter.export_linear()
    assert {c.name for c in exported.adapter.controls} == {
        "fuel_a",
        "fuel_b",
    }  # nosec B101
    assert {s.name for s in exported.adapter.states} == {
        "state:fs.boiler[a].steam",
        "state:fs.boiler[b].steam",
    }  # nosec B101
    audit = exported.as_claim(cases).audit_result()
    assert audit.payload.verdict == "pass", audit.to_json()  # nosec B101
    assert verify_result(audit).verified  # nosec B101
    assert [values(e)["fuel"] for e in audit.supporting_evaluations] == pytest.approx(
        [23, 31]
    )  # nosec B101
    reloaded = IdaesLinearExport.from_json(exported.to_json())
    assert reloaded.to_dict() == exported.to_dict()  # nosec B101
    assert verify_result(reloaded.as_claim(cases).audit_result()).verified  # nosec B101


def test_unknown_failure_is_not_a_counterexample_but_affine_certificate_can_be():
    adapter = bind_flowsheet(build_flowsheet())
    cases = domain(adapter, (16, 23))
    numerical = adapter.as_claim(cases).audit_result()
    assert numerical.payload.verdict == "inconclusive"  # nosec B101
    assert numerical.payload.witness is None  # nosec B101
    assert numerical.payload.coverage.method == "partial"  # nosec B101
    assert numerical.supporting_evaluations[-1].payload.feasibility == "unknown"  # nosec B101
    certified = adapter.export_linear().as_claim(cases).audit_result()
    assert certified.payload.verdict == "fail"  # nosec B101
    assert verify_result(certified).verified  # nosec B101


def test_numerical_finite_scope_and_affine_continuous_scope_are_distinct():
    adapter = bind_flowsheet(build_flowsheet())
    box = BoxSet((Parameter("load", "MW", 16, 12, 20, 4, "load range"),))
    assert adapter.as_claim(box).audit_result().execution == "unsupported"  # nosec B101
    assert (
        adapter.as_claim(box).evaluate_result({"load": 16}).payload.feasibility
        == "feasible"
    )  # nosec B101
    exported = adapter.export_linear().as_claim(box).audit_result()
    assert exported.payload.verdict == "pass"  # nosec B101
    assert verify_result(exported).verified  # nosec B101


def test_fixed_actuators_retain_algebraic_state_recourse():
    adapter = bind_flowsheet(build_flowsheet())
    policy = RecoursePolicy(
        "fixed",
        (
            DecisionRule("fuel_a", "MW", "operation", fixed_value=10),
            DecisionRule("fuel_b", "MW", "operation", fixed_value=16),
        ),
    )
    cases = domain(adapter, (16,))
    native = adapter.as_claim(cases, recourse=policy).evaluate_result({"load": 16})
    assert native.payload.feasibility == "feasible", native.to_json()  # nosec B101
    assert values(native)["state:fs.boiler[a].steam"] == pytest.approx(8)  # nosec B101
    exported = (
        adapter.export_linear()
        .as_claim(cases, recourse=policy)
        .evaluate_result({"load": 16})
    )
    assert exported.payload.feasibility == "feasible"  # nosec B101
    assert verify_result(exported).verified  # nosec B101
    partial = RecoursePolicy(
        "static",
        (
            DecisionRule("fuel_a", "MW", "operation"),
            DecisionRule("fuel_b", "MW", "operation", ("load",)),
        ),
    )
    assert (
        adapter.as_claim(cases, recourse=partial)
        .evaluate_result({"load": 16})
        .execution
        == "unsupported"
    )  # nosec B101
    with pytest.raises(ValueError, match="observe"):
        adapter.export_linear().as_claim(cases, recourse=partial)


def test_requirement_selection_retains_all_physics_and_bounds():
    model = build_flowsheet()
    model.fs.extra_service = pyo.Constraint(expr=model.fs.steam >= 25 * pyo.units.MW)
    adapter = replace(
        bind_flowsheet(model),
        model=model,
        requirements=(
            IdaesRequirement("service", "fs.service", "MW", 0),
            IdaesRequirement("extra", "fs.extra_service", "MW", 0),
        ),
    )
    cases = domain(adapter, (16,))
    assert adapter.as_claim(cases).audit_result().payload.verdict == "inconclusive"  # nosec B101
    selected = adapter.as_claim(cases, requirements=("service",)).evaluate_result(
        {"load": 16}
    )
    assert selected.payload.feasibility == "feasible", selected.to_json()  # nosec B101
    assert "extra" not in selected.payload.included_constraints  # nosec B101
    assert "constraint:fs.boiler[a].balance:eq" in selected.payload.included_constraints  # nosec B101
    assert "bound:fs.boiler[a].steam:upper" in selected.payload.included_constraints  # nosec B101
    assert (
        adapter.export_linear()
        .as_claim(cases, requirements=("service",))
        .audit_result()
        .payload.verdict
        == "pass"
    )  # nosec B101


@pytest.mark.parametrize(
    "kind",
    (
        "nonlinear",
        "nonlinear_input",
        "nonlinear_output",
        "nonlinear_objective",
        "unbounded",
        "rounding",
    ),
)
def test_affine_export_rejects_unsupported_equations_without_approximation(kind):
    model = build_flowsheet()
    b = model.fs.boiler["a"]
    if kind == "nonlinear":
        b.balance.set_value(b.steam == b.efficiency * b.fuel**2 / pyo.units.MW)
    elif kind == "nonlinear_input":
        b.balance.set_value(
            b.steam == b.efficiency * b.fuel * model.fs.load / (16 * pyo.units.MW)
        )
    elif kind == "nonlinear_output":
        model.fs.fuel.set_value(b.fuel**2 / pyo.units.MW)
    elif kind == "nonlinear_objective":
        model.fs.objective.set_value(b.fuel**2)
    elif kind == "unbounded":
        b.steam.setub(None)
    else:
        b.divisor = pyo.Param(initialize=3, mutable=True)
        b.balance.set_value(b.steam == b.fuel / b.divisor)
    with pytest.raises(ValueError):
        bind_flowsheet(model).export_linear()


@pytest.mark.parametrize(
    "kind",
    (
        "unit",
        "integer",
        "inactive_requirement",
        "equality_requirement",
        "external_variable",
    ),
)
def test_bad_bindings_are_rejected(kind):
    model = build_flowsheet()
    if kind == "unit":
        adapter = bind_flowsheet(model)
        with pytest.raises(ValueError, match="unit"):
            replace(
                adapter,
                model=model,
                outputs=(IdaesVariable("steam", "fs.steam", "kW"),),
            )
        return
    if kind == "integer":
        model.fs.boiler["a"].fuel.domain = pyo.Integers
    elif kind == "inactive_requirement":
        model.fs.service.deactivate()
    elif kind == "equality_requirement":
        model.fs.service.set_value(model.fs.steam == model.fs.load)
    else:
        other = pyo.ConcreteModel()
        other.fuel = pyo.Var(bounds=(0, 20), units=pyo.units.MW)
        model.fs.bad = pyo.Constraint(expr=model.fs.steam <= other.fuel)
    with pytest.raises(ValueError):
        bind_flowsheet(model)


def test_parameter_values_and_source_equations_affect_identity():
    first = build_flowsheet()
    a = bind_flowsheet(first)
    assert a.base_contract.ref == bind_flowsheet(build_flowsheet()).base_contract.ref  # nosec B101
    first.fs.boiler["a"].efficiency.set_value(0.7)
    assert bind_flowsheet(first).base_contract.ref != a.base_contract.ref  # nosec B101
    assert (
        bind_flowsheet(
            build_flowsheet(), provenance={"campaign": "2026"}
        ).base_contract.ref
        != a.base_contract.ref
    )  # nosec B101


def test_solver_failure_and_stale_initial_values_do_not_pass(monkeypatch):
    adapter = bind_flowsheet(build_flowsheet())
    cases = domain(adapter, (16,))
    monkeypatch.setattr(
        _idaes_execution, "solve", lambda *args: (False, {"termination": "infeasible"})
    )
    result = adapter.as_claim(cases).evaluate_result({"load": 16})
    assert result.payload.feasibility == "unknown"  # nosec B101
    assert not any(v.quantity_ref == "fuel" for v in result.payload.values)  # nosec B101
    assert adapter.as_claim(cases).audit_result().payload.verdict == "inconclusive"  # nosec B101


def test_reported_solver_success_still_checks_original_rows(monkeypatch):
    adapter = bind_flowsheet(build_flowsheet())
    monkeypatch.setattr(
        _idaes_execution, "solve", lambda *args: (True, {"termination": "optimal"})
    )
    result = adapter.as_claim(domain(adapter)).evaluate_result({"load": 16})
    assert result.payload.feasibility == "unknown"  # nosec B101
    assert any(c.assessment == "violated" for c in result.payload.constraint_checks)  # nosec B101


def test_missing_solver_and_invalid_requests_are_structured():
    model = build_flowsheet()
    adapter = replace(
        bind_flowsheet(model),
        model=model,
        solver="ipopt",
        solver_options={"max_iter": 1},
    )
    claim = adapter.as_claim(domain(adapter))
    assert claim.evaluate_result({"load": 99}).execution == "invalid"  # nosec B101
    assert claim.evaluate_result({"load": float("nan")}).execution == "invalid"  # nosec B101
    assert (
        claim.evaluate_result({"load": 16}, invent_option=True).execution == "invalid"
    )  # nosec B101
    assert claim.sensitivity_result({"load": 16}).execution == "unsupported"  # nosec B101


def test_variable_inputs_are_not_silently_fixed_at_nominal_in_export():
    model = build_flowsheet()
    model.fs.service.deactivate()
    model.fs.variable_load = pyo.Var(bounds=(0, 18), initialize=16, units=pyo.units.MW)
    model.fs.variable_load.fix(16)
    model.fs.variable_service = pyo.Constraint(
        expr=model.fs.steam >= model.fs.variable_load
    )
    base_model = build_flowsheet()
    adapter = replace(
        bind_flowsheet(base_model),
        model=model,
        inputs=(IdaesVariable("load", "fs.variable_load", "MW"),),
        requirements=(IdaesRequirement("service", "fs.variable_service", "MW", 0),),
    )
    cases = domain(adapter, (20,))
    assert adapter.as_claim(cases).audit_result().payload.verdict == "inconclusive"  # nosec B101
    result = adapter.export_linear().as_claim(cases).audit_result()
    assert result.payload.verdict == "fail"  # nosec B101
    assert verify_result(result).verified  # nosec B101


def test_upstream_parmest_reconciliation_and_calibration_reenter_study():
    if not pyo.SolverFactory("ipopt").available(exception_flag=False):
        pytest.skip("Upstream parmest requires the external Ipopt executable")
    calibration = reconcile_and_estimate()
    assert calibration["efficiency"] == pytest.approx(0.74, abs=1e-7)  # nosec B101
    for row in calibration["reconciled"]:
        assert row["fs.boiler.steam"] == pytest.approx(
            0.8 * row["fs.boiler.fuel"], abs=1e-7
        )  # nosec B101
    adapter = bind_flowsheet(
        build_flowsheet(calibration["efficiency"]),
        provenance={"calibration": calibration},
    )
    report = study(adapter)
    assert report["affine_audit"] == "fail"  # nosec B101
    assert report["affine_certificate"] == "verified"  # nosec B101
    assert report["numerical_audit"] == "inconclusive"  # nosec B101
    json.dumps(calibration, allow_nan=False)


def test_export_matches_existing_handwritten_dispatch_model():
    from examples.linear_dispatch import example
    from operange import LinearObjective

    reference, loads = example()
    reference = replace(
        reference,
        objective=LinearObjective("fuel"),
        operating_limits=tuple(
            replace(r, limit=32) if r.name == "shared_fuel" else r
            for r in reference.operating_limits
        ),
    )
    adapter = bind_flowsheet(build_flowsheet())
    exported = adapter.export_linear().as_claim(domain(adapter))
    for point in ({"dryer": 10, "evaporator": 6}, {"dryer": 12, "evaporator": 8}):
        baseline = reference.as_claim(loads).evaluate_result(point)
        actual = exported.evaluate_result({"load": sum(point.values())})
        assert actual.payload.feasibility == baseline.payload.feasibility == "feasible"  # nosec B101
        assert values(actual)["fuel"] == pytest.approx(
            values(baseline)["fuel"], abs=1e-6
        )  # nosec B101
        assert verify_result(actual).verified and verify_result(baseline).verified  # nosec B101


def test_solver_model_transformations_cannot_change_checked_physics(monkeypatch):
    adapter = bind_flowsheet(build_flowsheet())

    def altered(model, _):
        for b in model.fs.boiler.values():
            b.fuel.set_value(0)
            b.steam.set_value(10)
            b.balance.set_value(b.steam == 10 * pyo.units.MW)
        return True, {"termination": "optimal"}

    monkeypatch.setattr(_idaes_execution, "solve", altered)
    result = adapter.as_claim(domain(adapter)).evaluate_result({"load": 16})
    assert result.payload.feasibility == "unknown"  # nosec B101
    assert any(
        c.constraint_ref == "constraint:fs.boiler[a].balance:eq"
        and c.assessment == "violated"
        for c in result.payload.constraint_checks
    )  # nosec B101


def test_solver_cannot_change_a_fixed_command(monkeypatch):
    adapter = bind_flowsheet(build_flowsheet())
    policy = RecoursePolicy(
        "fixed",
        (
            DecisionRule("fuel_a", "MW", "operation", fixed_value=10),
            DecisionRule("fuel_b", "MW", "operation", fixed_value=16),
        ),
    )

    def changed(model, _):
        model.fs.boiler["a"].fuel.set_value(15)
        return True, {"termination": "optimal"}

    monkeypatch.setattr(_idaes_execution, "solve", changed)
    result = adapter.as_claim(domain(adapter), recourse=policy).evaluate_result(
        {"load": 16}
    )
    assert result.payload.feasibility == "unknown"  # nosec B101
    assert "fixed input, command or state" in result.diagnostics[0].message  # nosec B101


def test_missing_solver_produces_an_unresolved_scenario(monkeypatch):
    adapter = bind_flowsheet(build_flowsheet())

    class Missing:
        def available(self, exception_flag=False):
            return False

    monkeypatch.setattr(pyo, "SolverFactory", lambda name: Missing())
    result = adapter.as_claim(domain(adapter)).audit_result()
    assert result.payload.verdict == "inconclusive"  # nosec B101
    assert all(
        "unavailable" in e.diagnostics[0].message for e in result.supporting_evaluations
    )  # nosec B101


def test_ports_references_and_expanded_arcs_preserve_connection_equations():
    from pyomo.network import Arc, Port

    model = build_flowsheet()
    model.fs.alias = pyo.Reference(model.fs.boiler[:].fuel)
    model.fs.delivered = pyo.Var(bounds=(0, 20), initialize=8, units=pyo.units.MW)
    model.fs.source_port = Port(initialize={"flow": model.fs.boiler["a"].steam})
    model.fs.sink_port = Port(initialize={"flow": model.fs.delivered})
    model.fs.connection = Arc(
        source=model.fs.source_port, destination=model.fs.sink_port
    )
    with pytest.raises(ValueError, match="Unsupported active model component"):
        bind_flowsheet(model)
    pyo.TransformationFactory("network.expand_arcs").apply_to(model)
    adapter = bind_flowsheet(model)
    native = adapter.as_claim(domain(adapter)).evaluate_result({"load": 16})
    assert native.payload.feasibility == "feasible", native.to_json()  # nosec B101
    assert values(native)["state:fs.delivered"] == pytest.approx(
        values(native)["state:fs.boiler[a].steam"]
    )  # nosec B101
    exported = adapter.export_linear().as_claim(domain(adapter)).audit_result()
    assert exported.payload.verdict == "pass"  # nosec B101
    assert verify_result(exported).verified  # nosec B101


def test_component_aliases_and_external_mutable_coefficients_are_rejected():
    model = build_flowsheet()
    adapter = bind_flowsheet(model)
    with pytest.raises(ValueError, match="alias"):
        replace(
            adapter,
            model=model,
            controls=(
                IdaesVariable("fuel_a", "fs.boiler[a].fuel", "MW"),
                IdaesVariable("fuel_b", "fs.boiler['a'].fuel", "MW"),
            ),
        )
    external = pyo.ConcreteModel()
    external.eta = pyo.Param(initialize=0.8, mutable=True)
    b = model.fs.boiler["a"]
    b.balance.set_value(b.steam == external.eta * b.fuel)
    with pytest.raises(ValueError, match="parameters outside"):
        bind_flowsheet(model)


def test_nonlinear_ipopt_evaluation_and_finite_scope():
    if not pyo.SolverFactory("ipopt").available(exception_flag=False):
        pytest.skip("Ipopt is not installed")
    model = build_flowsheet()
    # A genuinely nonlinear duty/fuel curve, with commands/states bounded away
    # from their extrema to avoid accepting Ipopt's relaxed variable bounds.
    b = model.fs.boiler["a"]
    b.balance.set_value(b.steam == 0.04 * b.fuel**2 / pyo.units.MW)
    model.fs.objective.set_value(
        (b.fuel - 16 * pyo.units.MW) ** 2
        + (model.fs.boiler["b"].fuel - 12 * pyo.units.MW) ** 2
    )
    adapter = replace(
        bind_flowsheet(model),
        model=model,
        solver="ipopt",
        tolerance=1e-6,
        solver_options={"tol": 1e-10, "bound_relax_factor": 0},
    )
    result = adapter.as_claim(domain(adapter, (16,))).evaluate_result({"load": 16})
    assert result.payload.feasibility == "feasible", result.to_json()  # nosec B101
    assert values(result)["fuel_a"] == pytest.approx(16, abs=1e-5)  # nosec B101
    assert result.payload.objective is None  # nosec B101
    assert verify_result(result).status == "unsupported"  # nosec B101
    with pytest.raises(ValueError):
        adapter.export_linear()


def test_affine_export_keeps_input_dependent_bounds_symbolic_or_rejects_them():
    model = build_flowsheet()
    model.fs.boiler["a"].steam.setub(model.fs.load)
    adapter = bind_flowsheet(model)
    with pytest.raises(ValueError, match="bounds must be constant"):
        adapter.export_linear()


def test_objective_free_feasibility_and_source_fixed_controls():
    model = build_flowsheet()
    model.fs.objective.deactivate()
    model.fs.boiler["a"].fuel.fix(1)
    model.fs.boiler["b"].fuel.fix(1)
    adapter = bind_flowsheet(model)
    result = adapter.as_claim(domain(adapter)).audit_result()
    assert result.payload.verdict == "pass", result.to_json()  # nosec B101
    assert adapter.export_linear().adapter.objective is None  # nosec B101
    assert (
        adapter.export_linear().as_claim(domain(adapter)).audit_result().payload.verdict
        == "pass"
    )  # nosec B101
