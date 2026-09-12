"""A shared equipment plan covers all scenarios with independently checked evidence."""

from dataclasses import replace
from fractions import Fraction as F
import json
from types import SimpleNamespace

import pytest

from operange import (
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    ConvexHullSet,
    Coordinate,
    DecisionRule,
    FiniteSet,
    LinearControl,
    LinearObjective,
    LinearProcessAdapter,
    Parameter,
    ParameterSpace,
    RecoursePolicy,
    Scenario,
    result_from_json,
)
from operange import linear, _shared_relief
from tests.test_joint_relief import check_bound
from tests.test_linear_process import portable


def example():
    space = ParameterSpace(
        tuple(Coordinate(n, "MW", "Declared load") for n in ("a", "b"))
    )
    model = LinearProcessAdapter(
        "Independent thermal services",
        space,
        tuple(
            output
            for name in space.names
            for output in (
                AffineOutput(
                    f"margin_{name}",
                    "MW",
                    "thermal_power",
                    0,
                    (
                        AffineTerm(f"power_{name}", 1, "MW/MW"),
                        AffineTerm(name, -1, "MW/MW"),
                    ),
                    "Delivery minus load",
                ),
                AffineOutput(
                    f"supply_{name}",
                    "MW",
                    "thermal_power",
                    0,
                    (AffineTerm(f"power_{name}", 1, "MW/MW"),),
                    "Equipment output",
                ),
            )
        ),
        tuple(
            AffineRequirement(f"service_{name}", f"margin_{name}", 0, "ge", 0)
            for name in space.names
        ),
        tuple(LinearControl(f"power_{name}", "MW", 0, 4) for name in space.names),
        tuple(
            AffineRequirement(f"capacity_{name}", f"supply_{name}", 1, tolerance=0)
            for name in space.names
        ),
    )
    domain = FiniteSet(
        space,
        (
            Scenario("a_peak", {"a": 2, "b": 0}, "Mode A"),
            Scenario("b_peak", {"a": 0, "b": 3}, "Mode B"),
        ),
    )
    changes = tuple(
        {"constraint": f"capacity_{name}", "maximum": 3, "scale": 1, "unit": "MW"}
        for name in space.names
    )
    return model, domain, {"changes": changes, "objective": "linear"}


def relief(result):
    return next(
        e.to_dict()["details"] for e in result.evidence if e.evidence_id == "relief"
    )


def changed_audit(data):
    return result_from_json(json.dumps(data["candidate"]["reaudit"]))


@pytest.mark.parametrize("backend", ("scipy", "cvxpy"))
@pytest.mark.parametrize("kind", ("finite", "box", "hull"))
def test_one_plan_covers_every_scenario_and_keeps_the_original_verdict(backend, kind):
    if backend == "cvxpy":
        pytest.importorskip("cvxpy")
    model, domain, options = example()
    if kind == "hull":
        domain = ConvexHullSet(domain.space, domain.scenarios)
    elif kind == "box":
        domain = BoxSet(
            (
                Parameter("a", "MW", 0, 0, 2, 1, "A range"),
                Parameter("b", "MW", 0, 0, 3, 1, "B range"),
            )
        )
    claim = model.as_claim(domain)
    baseline = claim.audit_result(backend=backend)
    result = claim.audit_result(relief=options, backend=backend)
    data = relief(result)
    assert result.payload == baseline.payload and result.contract == baseline.contract  # nosec B101
    assert result.execution == "completed" and result.payload.verdict == "fail"  # nosec B101
    assert data["resolution"] == "minimum_verified"  # nosec B101
    assert data["lower"] == data["upper"] == 3  # nosec B101
    assert [c["physical_relief"] for c in data["candidate"]["changes"]] == [1, 2]  # nosec B101
    reaudit = changed_audit(data)
    assert reaudit.payload.verdict == "pass"  # nosec B101
    assert reaudit.contract.domain == baseline.contract.domain  # nosec B101
    assert reaudit.contract.operating == baseline.contract.operating  # nosec B101
    assert reaudit.payload.coverage.method == (
        "complete_finite" if kind == "finite" else "analytical_domain"
    )  # nosec B101
    assert all(
        e.request["dispatch_source"] == "shared_relief_candidate"
        for e in reaudit.supporting_evaluations
    )  # nosec B101
    check_bound(data)
    portable(result)
    portable(reaudit)


def test_individual_repair_does_not_restore_the_other_operating_mode():
    model, domain, options = example()
    single = model.as_claim(domain).evaluate_result({"a": 2, "b": 0}, relief=options)
    individual = relief(single)["candidate"]["changes"]
    updates = {c["constraint"]: c["changed_limit"] for c in individual}
    changed = replace(
        model,
        operating_limits=tuple(
            replace(r, limit=updates[r.name]) for r in model.operating_limits
        ),
    )
    assert changed.as_claim(domain).audit_result().payload.verdict == "fail"  # nosec B101
    shared = relief(model.as_claim(domain).audit_result(relief=options))
    assert changed_audit(shared).payload.verdict == "pass"  # nosec B101


def test_quadratic_joint_optimization_is_not_componentwise_maximum_of_local_plans():
    pytest.importorskip("cvxpy")
    model, domain, options = example()
    model = replace(
        model,
        outputs=tuple(
            replace(
                o,
                terms=(
                    AffineTerm("power_a", 1 if o.name == "margin_a" else 2, "MW/MW"),
                    AffineTerm("power_b", 1, "MW/MW"),
                    AffineTerm(o.name[-1], -1, "MW/MW"),
                ),
            )
            if o.name.startswith("margin")
            else o
            for o in model.outputs
        ),
        operating_limits=tuple(replace(r, limit=0) for r in model.operating_limits),
    )
    domain = replace(
        domain,
        scenarios=(
            Scenario("a", {"a": 1, "b": 0}, "A"),
            Scenario("b", {"a": 0, "b": 2}, "B"),
        ),
    )
    options = {**options, "objective": "quadratic"}
    claim = model.as_claim(domain)
    locals = [
        relief(claim.evaluate_result(s.values, relief=options, backend="cvxpy"))[
            "candidate"
        ]["changes"]
        for s in domain.scenarios
    ]
    naive = sum(max(c[i]["physical_relief"] for c in locals) ** 2 for i in range(2))
    result = claim.audit_result(relief=options, backend="cvxpy")
    data = relief(result)
    assert data["resolution"] == "minimum_verified"  # nosec B101
    assert data["lower"] <= 0.8 <= data["upper"]  # nosec B101
    assert data["upper"] < naive - 0.08  # nosec B101
    assert [
        c["physical_relief"] for c in data["candidate"]["changes"]
    ] == pytest.approx([0.8, 0.4], abs=1e-7)  # nosec B101
    check_bound(data)
    portable(result)


def test_exact_contradiction_proves_insufficient_shared_relief():
    model, domain, options = example()
    options = {**options, "changes": [dict(c, maximum=0.5) for c in options["changes"]]}
    result = model.as_claim(domain).audit_result(relief=options)
    data = relief(result)
    assert data["resolution"] == "unreachable"  # nosec B101
    proof = data["infeasibility_certificate"]
    weights = list(map(F, proof["multipliers"]))
    rows = [list(map(F, r)) for r in proof["rows_exact"]]
    coefficients = [sum(w * a for w, a in zip(weights, col)) for col in zip(*rows)]
    gap = sum(min(F(0), a) for a in coefficients) - sum(
        w * F(b) for w, b in zip(weights, proof["upper_exact"])
    )
    assert gap > 0 and gap == F(proof["contradiction_gap_exact"])  # nosec B101
    assert "candidate" not in data  # nosec B101
    portable(result)


def test_zero_relief_needs_no_equipment_optimization(monkeypatch):
    model, domain, options = example()
    model = replace(
        model,
        operating_limits=tuple(replace(r, limit=4) for r in model.operating_limits),
    )
    monkeypatch.setattr(
        _shared_relief,
        "solve_dispatch",
        lambda *a, **k: pytest.fail("Zero relief is already optimal"),
    )
    result = model.as_claim(domain).audit_result(relief=options)
    data = relief(result)
    assert result.payload.verdict == "pass"  # nosec B101
    assert (
        data["resolution"] == "minimum_verified" and data["lower"] == data["upper"] == 0
    )  # nosec B101
    assert changed_audit(data).contract == result.contract  # nosec B101
    portable(result)


def test_duplicates_retain_names_without_repeating_objective_or_dispatch_variables():
    model, domain, options = example()
    domain = replace(
        domain,
        scenarios=domain.scenarios + (replace(domain.scenarios[0], name="duplicate"),),
    )
    result = model.as_claim(domain).audit_result(relief=options)
    data = relief(result)
    assert data["scenario_count"] == 3 and len(data["dispatch_blocks"]) == 2  # nosec B101
    assert data["upper"] == 3  # nosec B101
    assert len(changed_audit(data).payload.coverage.evaluated_support["scenarios"]) == 3  # nosec B101


def test_fixed_commands_and_lower_limits_preserve_permissions():
    model, domain, options = example()
    model = replace(
        model, operating_limits=(AffineRequirement("minimum", "margin_a", 1, "ge", 0),)
    )
    domain = replace(
        domain,
        scenarios=tuple(
            replace(s, values={"a": s.values["a"], "b": 0}) for s in domain.scenarios
        ),
    )
    policy = RecoursePolicy(
        "fixed",
        tuple(
            DecisionRule(c.name, c.unit, "operation", fixed_value=2)
            for c in model.controls
        ),
    )
    options = {
        "changes": [{"constraint": "minimum", "maximum": 1, "scale": 1, "unit": "MW"}],
        "objective": "linear",
    }
    result = model.as_claim(domain, recourse=policy).audit_result(relief=options)
    data = relief(result)
    assert data["resolution"] == "minimum_verified" and data["upper"] == 1  # nosec B101
    assert data["candidate"]["changes"][0]["changed_limit"] == 0  # nosec B101
    assert all(
        d["controls"] == {"power_a": 2, "power_b": 2}
        for d in data["candidate"]["dispatches"]
    )  # nosec B101
    assert (
        changed_audit(data).contract.operating["recourse_policy"]
        == result.contract.operating["recourse_policy"]
    )  # nosec B101
    portable(result)


@pytest.mark.parametrize(
    "options",
    [
        True,
        {},
        {"constraint": "capacity_a", "maximum": 1},
        {"changes": [], "objective": "linear"},
        {"changes": []},
    ],
)
def test_invalid_shared_declaration_is_explicit(options):
    model, domain, _ = example()
    result = model.as_claim(domain).audit_result(relief=options)
    assert result.execution == "invalid"  # nosec B101
    portable(result)


@pytest.mark.parametrize("limit", [0, True, -1, 2.0, None])
def test_invalid_scenario_budget(limit):
    model, domain, options = example()
    result = model.as_claim(domain).audit_result(
        relief={**options, "max_scenarios": limit}
    )
    assert result.execution == "invalid"  # nosec B101


def test_budget_rejects_before_any_solve(monkeypatch):
    model, domain, options = example()
    monkeypatch.setattr(
        linear, "solve_lp", lambda *a, **k: pytest.fail("Enumeration rejected")
    )
    result = model.as_claim(domain).audit_result(relief={**options, "max_scenarios": 1})
    assert result.execution == "unsupported" and not result.supporting_evaluations  # nosec B101
    assert result.diagnostics[0].code == "relief_scenario_limit_exceeded"  # nosec B101


def test_solver_failure_cannot_be_mistaken_for_unreachability(monkeypatch):
    model, domain, options = example()
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=False, status=2, message="Injected infeasible", x=None
        ),
    )
    result = model.as_claim(domain).audit_result(relief=options)
    data = relief(result)
    assert data["resolution"] == "unresolved" and data["candidate"] is None  # nosec B101
    assert data["lower"] == 0 and data["upper"] is None  # nosec B101
    assert result.execution == "unresolved"  # nosec B101
    portable(result)


def test_dispatch_objective_does_not_enter_the_equipment_objective():
    model, domain, options = example()
    model = replace(model, objective=LinearObjective("supply_a", "maximize"))
    result = model.as_claim(domain).audit_result(relief=options)
    data = relief(result)
    assert data["upper"] == 3  # nosec B101
    assert (
        changed_audit(data).contract.model["objective"]
        == result.contract.model["objective"]
    )  # nosec B101
    portable(result)


def test_selected_services_unselected_limits_and_hard_bounds_stay_in_force():
    model, domain, options = example()
    only_a = {**options, "changes": options["changes"][:1]}
    blocked = model.as_claim(domain).audit_result(relief=only_a)
    assert relief(blocked)["resolution"] == "unreachable"  # nosec B101
    selected = model.as_claim(domain, requirements=("service_a",)).audit_result(
        relief=only_a
    )
    data = relief(selected)
    assert data["upper"] == 1 and changed_audit(data).payload.verdict == "pass"  # nosec B101
    assert changed_audit(data).contract.operating["requirement_refs"] == ("service_a",)  # nosec B101
    beyond_bounds = replace(
        domain,
        scenarios=(Scenario("excess", {"a": 5, "b": 0}, "Exceeds hard control box"),),
    )
    result = model.as_claim(beyond_bounds).audit_result(
        relief={**only_a, "changes": [dict(only_a["changes"][0], maximum=10)]}
    )
    assert relief(result)["resolution"] == "unreachable"  # nosec B101


def test_mixed_physical_units_and_scales_are_shared_across_scenarios():
    model, domain, options = example()
    model = replace(
        model,
        outputs=tuple(
            replace(
                o,
                unit="kW",
                terms=tuple(
                    replace(t, coefficient=t.coefficient * 1000, unit="kW/MW")
                    for t in o.terms
                ),
            )
            if o.name == "supply_b"
            else o
            for o in model.outputs
        ),
        operating_limits=tuple(
            replace(r, limit=1000) if r.name == "capacity_b" else r
            for r in model.operating_limits
        ),
    )
    options = {
        **options,
        "changes": [
            options["changes"][0],
            dict(options["changes"][1], unit="kW", scale=1000, maximum=3000),
        ],
    }
    result = model.as_claim(domain).audit_result(relief=options)
    data = relief(result)
    assert data["lower"] <= 3 <= data["upper"]  # nosec B101
    assert data["upper"] - data["lower"] <= 1e-8  # nosec B101
    assert [c["physical_relief"] for c in data["candidate"]["changes"]] == [1, 2000]  # nosec B101
    check_bound(data)


@pytest.mark.parametrize(
    "fault",
    ("service", "control_bound", "unit", "duplicate", "weight", "objective", "unknown"),
)
def test_invalid_changes_are_rejected_without_altering_the_model(fault):
    model, domain, options = example()
    options = {**options, "changes": [dict(c) for c in options["changes"]]}
    if fault in ("service", "control_bound"):
        options["changes"][0]["constraint"] = (
            "service_a" if fault == "service" else "control_upper:power_a"
        )
    elif fault == "unit":
        options["changes"][0]["unit"] = "kW"
    elif fault == "duplicate":
        options["changes"][1] = options["changes"][0]
    elif fault == "weight":
        options["changes"][0]["weight"] = 0
    elif fault == "objective":
        options["objective"] = "implicit"
    else:
        options["silent_option"] = 1
    result = model.as_claim(domain).audit_result(relief=options)
    assert result.execution == "invalid"  # nosec B101
    assert result.contract.model == model.base_contract.model  # nosec B101


def test_partial_observations_are_not_upgraded_to_independent_dispatch():
    model, domain, options = example()
    policy = RecoursePolicy(
        "static",
        tuple(
            DecisionRule(c.name, c.unit, "operation", ("a",)) for c in model.controls
        ),
    )
    result = model.as_claim(domain, recourse=policy).audit_result(relief=options)
    assert result.execution == "unsupported"  # nosec B101


def test_quadratic_and_missing_backend_requirements_are_explicit(monkeypatch):
    model, domain, options = example()
    assert (
        model.as_claim(domain)
        .audit_result(relief={**options, "objective": "quadratic"})
        .execution
        == "unsupported"
    )  # nosec B101
    from operange import _cvxpy_backend

    def unavailable(*args):
        raise ImportError("Missing optional backend")

    monkeypatch.setattr(_cvxpy_backend, "import_module", unavailable)
    assert (
        model.as_claim(domain).audit_result(relief=options, backend="cvxpy").execution
        == "unsupported"
    )  # nosec B101


def test_open_gap_keeps_a_verified_plan_without_claiming_a_minimum(monkeypatch):
    model, domain, options = example()
    solve = _shared_relief.solve_dispatch

    def no_bound(*args, **kwargs):
        solution, _, _ = solve(*args, **kwargs)
        return solution, F(0), {"method": "nonnegative_relief_objective"}

    monkeypatch.setattr(_shared_relief, "solve_dispatch", no_bound)
    result = model.as_claim(domain).audit_result(relief=options)
    data = relief(result)
    assert result.execution == "unresolved" and data["resolution"] == "bounded"  # nosec B101
    assert data["lower"] == 0 and data["upper"] == 3  # nosec B101
    assert changed_audit(data).payload.verdict == "pass"  # nosec B101
    portable(result)


def test_corrupt_dispatch_fails_independent_reaudit_and_supplies_no_upper_bound(
    monkeypatch,
):
    model, domain, options = example()
    candidate = _shared_relief._candidate

    def corrupt(claim, blocks, dispatches, *args):
        dispatches = [{**c, "power_a": 0} for c in dispatches]
        return candidate(claim, blocks, dispatches, *args)

    monkeypatch.setattr(_shared_relief, "_candidate", corrupt)
    result = model.as_claim(domain).audit_result(relief=options)
    data = relief(result)
    assert data["resolution"] == "unresolved" and data["candidate"] is None  # nosec B101
    assert data["upper"] is None and data["lower"] == 3  # nosec B101
    portable(result)


def test_unrepresentable_changed_limit_cannot_exceed_the_allowed_relief():
    model, domain, options = example()
    model = replace(
        model,
        outputs=tuple(
            replace(o, offset=1e16) if o.name == "supply_a" else o
            for o in model.outputs
        ),
        operating_limits=tuple(
            replace(r, limit=1e16) if r.name == "capacity_a" else r
            for r in model.operating_limits
        ),
    )
    domain = replace(
        domain, scenarios=(Scenario("a", {"a": 1, "b": 0}, "Needs one MW"),)
    )
    options = {**options, "changes": [dict(options["changes"][0], maximum=1)]}
    result = model.as_claim(domain).audit_result(relief=options)
    data = relief(result)
    assert data["resolution"] == "unresolved" and data["candidate"] is None  # nosec B101
    assert data["lower"] == 1 and data["upper"] is None  # nosec B101
    portable(result)


def test_unexportable_objective_never_produces_an_infinite_bound():
    model, domain, options = example()
    result = model.as_claim(domain).audit_result(
        relief={
            **options,
            "changes": [dict(c, scale=1e-320) for c in options["changes"]],
        }
    )
    data = relief(result)
    assert data["resolution"] == "unresolved" and data["upper"] is None  # nosec B101
    portable(result)


@pytest.mark.parametrize(
    "objective,backend,target", [("linear", "scipy", 3), ("quadratic", "cvxpy", 4.64)]
)
def test_boiler_example_reaudits_both_low_load_and_peak_modes(
    objective, backend, target
):
    if backend == "cvxpy":
        pytest.importorskip("cvxpy")
    from examples.shared_relief import run_example

    for exported in run_example(objective=objective, backend=backend).values():
        result = result_from_json(json.dumps(exported))
        data = relief(result)
        assert result.payload.verdict == "fail"  # nosec B101
        assert data["resolution"] == "minimum_verified"  # nosec B101
        assert data["upper"] == pytest.approx(target, abs=1e-8)  # nosec B101
        assert changed_audit(data).payload.verdict == "pass"  # nosec B101
        assert data["candidate"]["changes"][2]["changed_limit"] == pytest.approx(6)  # nosec B101
        check_bound(data)
        portable(result)
