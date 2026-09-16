"""Local value slopes require exact optimality, feasibility and interval proofs."""

from dataclasses import replace
from fractions import Fraction as F
import json
from pathlib import Path
# CLI checks use a fixed executable and arguments, without a shell.
import subprocess  # nosec B404
import sys

import pytest

from operange import (
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    ControlTarget,
    ControlTrackingObjective,
    DecisionRule,
    DispatchChange,
    DispatchSensitivityResult,
    LinearControl,
    LinearObjective,
    LinearProcessAdapter,
    Parameter,
    RecoursePolicy,
    verify_result,
)
from operange._parametric_dispatch import ParametricDispatch
from examples.dispatch_sensitivity import example, run_example
from examples.equipment_network import example as network_example
from tests.test_network import boiler_controller


def load_change(step=2):
    return DispatchChange("input", "load", "MW", step)


def checked(result):
    for compact in (False, True):
        restored = DispatchSensitivityResult.from_json(result.to_json(compact=compact))
        assert restored == result  # nosec B101
        report = verify_result(
            restored, expected_contract_id=result.contract_ref.artifact_id
        )
        assert report.verified, report.to_json()  # nosec B101


@pytest.mark.parametrize("backend", ("scipy", "cvxpy"))
def test_demand_and_capacity_marginals_agree_with_analytical_fuel_curve(backend):
    if backend == "cvxpy":
        pytest.importorskip("cvxpy")
    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 16},
        changes=(
            load_change(),
            DispatchChange("control_upper", "a", "MW", 2),
            DispatchChange("operating_limit", "fuel_supply", "MW"),
        ),
        backend=backend,
    )
    assert result.execution == "completed"  # nosec B101
    for sensitivity, expected in zip(result.sensitivities, (2, -0.75, 0)):
        assert sensitivity.resolution == "differentiable"  # nosec B101
        for side in (sensitivity.backward, sensitivity.forward):
            assert side.value == expected and side.lower <= expected <= side.upper  # nosec B101
            assert side.base_value == 23 and side.unit == "MW/MW"  # nosec B101
    checked(result)


@pytest.mark.parametrize("backend", ("scipy", "cvxpy"))
def test_kink_has_distinct_exact_left_and_right_derivatives(backend):
    if backend == "cvxpy":
        pytest.importorskip("cvxpy")
    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 12}, changes=(load_change(),), backend=backend
    )
    slope = result.sensitivities[0]
    assert slope.resolution == "kink"  # nosec B101
    assert slope.backward.value == 1.25 and slope.forward.value == 2  # nosec B101
    assert slope.backward.valid_step == slope.forward.valid_step == 2  # nosec B101
    checked(result)


def test_another_binding_limit_blocks_the_capacity_upgrade():
    model, domain = example()
    model = replace(
        model,
        operating_limits=(
            *model.operating_limits,
            AffineRequirement("burner_limit", "a_output", 12, tolerance=0),
        ),
    )
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 16}, changes=(DispatchChange("control_upper", "a", "MW", 2),)
    )
    slope = result.sensitivities[0]
    assert slope.resolution == "kink"  # nosec B101
    assert slope.backward.value == -0.75 and slope.forward.value == 0  # nosec B101
    assert "operating_limit:burner_limit" in slope.forward.supporting_constraints  # nosec B101
    assert "control_upper:a" in slope.backward.supporting_constraints  # nosec B101
    checked(result)


@pytest.mark.parametrize("point,side", ((20.5, "forward"), (0, "backward")))
def test_farkas_ray_excludes_an_entire_infeasible_direction(point, side):
    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": point}, changes=(load_change(),)
    )
    slope = result.sensitivities[0]
    excluded = getattr(slope, side)
    assert slope.resolution == "one_sided"  # nosec B101
    assert excluded.status == "infeasible" and excluded.value is None  # nosec B101
    assert excluded.valid_step == 2  # nosec B101
    checked(result)


def test_fixed_commands_can_make_the_base_an_isolated_feasible_point():
    model, domain = example()
    policy = RecoursePolicy(
        "fixed",
        (
            DecisionRule("a", "MW", "operation", fixed_value=10),
            DecisionRule("b", "MW", "operation", fixed_value=6),
        ),
    )
    result = model.as_claim(domain, recourse=policy).dispatch_sensitivity(
        {"load": 16},
        changes=(load_change(), DispatchChange("control_upper", "a", "MW")),
    )
    assert result.sensitivities[0].resolution == "isolated_feasible_point"  # nosec B101
    assert result.sensitivities[1].resolution == "differentiable"  # nosec B101
    assert result.sensitivities[1].forward.value == 0  # nosec B101
    checked(result)


def test_validity_interval_stops_before_mode_change_or_infeasibility():
    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 16}, changes=(load_change(100),)
    )
    slope = result.sensitivities[0]
    assert slope.backward.valid_step == 4  # nosec B101
    assert slope.forward.valid_step == 4.5  # nosec B101
    assert slope.backward.value == slope.forward.value == 2  # nosec B101
    checked(result)


def test_far_probe_is_refined_to_the_local_regime():
    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 12}, changes=(load_change(100),)
    )
    assert result.sensitivities[0].resolution == "kink"  # nosec B101
    checked(result)


def test_zero_refinement_budget_retains_only_proved_directions():
    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 12}, changes=(load_change(),), max_refinements=0
    )
    assert result.execution == "unresolved"  # nosec B101
    sides = (result.sensitivities[0].backward, result.sensitivities[0].forward)
    assert sorted(s.status for s in sides) == ["unresolved", "verified"]  # nosec B101
    checked(result)  # A partial report asserts only its proved side.


def test_network_balances_retain_exact_float_efficiency_and_selected_services():
    network, domain = network_example()
    model = network.compile()
    result = model.as_claim(domain).dispatch_sensitivity(
        domain.nominal,
        changes=(
            DispatchChange("input", "dryer", "MW"),
            DispatchChange("control_upper", "steam_a", "MW"),
        ),
    )
    slope = result.sensitivities[1].forward
    assert F(slope.slope_exact) == 1 / F(0.8) - 2  # nosec B101
    assert F(slope.lower) <= F(slope.slope_exact) <= F(slope.upper)  # nosec B101
    checked(result)
    selected = model.as_claim(domain, requirements=("service:dryer",))
    result = selected.dispatch_sensitivity(
        domain.nominal, changes=(DispatchChange("input", "evaporator", "MW"),)
    )
    assert result.sensitivities[0].forward.value == 0  # nosec B101
    assert (
        "operating_limit:balance:steam_header:le"
        in result.sensitivities[0].forward.proof["row_refs"]
    )  # nosec B101
    checked(result)


def scalar_model(*, lower=0, upper=10, sense="minimize", floor=2):
    domain = BoxSet((Parameter("load", "MW", 1, 0, 2, 1, "External load"),))
    model = LinearProcessAdapter(
        "Scalar value",
        domain.space,
        outputs=(
            AffineOutput(
                "value",
                "cost/h",
                "operating_cost",
                7,
                (AffineTerm("x", 3, "cost/h/MW"), AffineTerm("load", -2, "cost/h/MW")),
                "Physical cost",
            ),
            AffineOutput(
                "flow", "MW", "thermal_power", 0, (AffineTerm("x", 1, "MW/MW"),), "Flow"
            ),
        ),
        requirements=(AffineRequirement("nonnegative", "flow", 0, "ge", tolerance=0),),
        controls=(LinearControl("x", "MW", lower, upper),),
        operating_limits=(
            AffineRequirement("minimum", "flow", floor, "ge", tolerance=0),
        ),
        objective=LinearObjective("value", sense=sense),
    )
    return model.as_claim(domain)


def test_direct_objective_input_and_ge_limit_signs_and_physical_units():
    claim = scalar_model()
    result = claim.dispatch_sensitivity(
        {"load": 1},
        changes=(
            load_change(0.5),
            DispatchChange("operating_limit", "minimum", "MW", 0.5),
        ),
    )
    assert [s.forward.value for s in result.sensitivities] == [-2, 3]  # nosec B101
    assert all(s.forward.unit == "cost/h/MW" for s in result.sensitivities)  # nosec B101
    assert all(s.forward.base_value == 11 for s in result.sensitivities)  # nosec B101
    checked(result)


def test_maximization_signs_preserve_objective_derivatives():
    claim = scalar_model(sense="maximize")
    result = claim.dispatch_sensitivity(
        {"load": 1},
        changes=(load_change(0.5), DispatchChange("control_upper", "x", "MW", 0.5)),
    )
    assert [s.forward.value for s in result.sensitivities] == [-2, 3]  # nosec B101
    assert all(s.forward.base_value == 35 for s in result.sensitivities)  # nosec B101
    checked(result)


def test_lower_bound_and_zero_width_bounds_are_not_silently_fixed_permissions():
    claim = scalar_model(lower=2, upper=2, floor=0)
    result = claim.dispatch_sensitivity(
        {"load": 1},
        changes=(
            DispatchChange("control_lower", "x", "MW", 0.5),
            DispatchChange("control_upper", "x", "MW", 0.5),
        ),
    )
    assert result.sensitivities[0].backward.value == 3  # nosec B101
    assert result.sensitivities[0].forward.status == "infeasible"  # nosec B101
    assert result.sensitivities[1].backward.status == "infeasible"  # nosec B101
    assert result.sensitivities[1].forward.value == 0  # nosec B101
    checked(result)


def test_nonunique_dispatch_still_has_a_well_defined_value_slope():
    model, domain = example()
    equal = replace(
        model.output("fuel"),
        terms=(AffineTerm("a", 1.25, "MW/MW"), AffineTerm("b", 1.25, "MW/MW")),
    )
    model = replace(
        model, outputs=tuple(equal if o.name == "fuel" else o for o in model.outputs)
    )
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 16},
        changes=(load_change(), DispatchChange("control_upper", "a", "MW")),
    )
    assert [s.forward.value for s in result.sensitivities] == [1.25, 0]  # nosec B101
    checked(result)


def test_reordering_rows_and_controls_does_not_change_the_value_derivative():
    model, domain = example()
    model = replace(
        model,
        controls=tuple(reversed(model.controls)),
        operating_limits=tuple(reversed(model.operating_limits)),
    )
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 12}, changes=(load_change(),)
    )
    assert result.sensitivities[0].backward.value == 1.25  # nosec B101
    assert result.sensitivities[0].forward.value == 2  # nosec B101
    checked(result)


@pytest.mark.parametrize("point", ({"load": 21}, {"load": 50}))
def test_infeasible_or_outside_base_never_gets_a_slope(point):
    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        point, changes=(load_change(),)
    )
    assert result.execution == "unresolved" and not result.sensitivities  # nosec B101
    assert result.base is not None  # nosec B101
    assert verify_result(result).status == "unsupported"  # nosec B101


@pytest.mark.parametrize(
    "change",
    (
        DispatchChange("input", "missing", "MW"),
        DispatchChange("input", "load", "kg/s"),
        DispatchChange("control_upper", "missing", "MW"),
        DispatchChange("operating_limit", "serve", "MW"),
        DispatchChange("operating_limit", "fuel_supply", "kg/s"),
    ),
)
def test_invalid_coordinate_or_unit_is_rejected(change):
    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 16}, changes=(change,)
    )
    assert result.execution == "invalid" and not result.sensitivities  # nosec B101
    assert DispatchSensitivityResult.from_json(result.to_json()) == result  # nosec B101


@pytest.mark.parametrize(
    "options",
    (
        {"changes": ()},
        {"changes": None},
        {"changes": "load"},
        {"changes": (load_change(), load_change())},
        {"max_refinements": -1},
        {"max_refinements": True},
        {"max_refinements": 33},
        {"backend": "unknown"},
    ),
)
def test_invalid_query_options(options):
    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 16}, **{"changes": (load_change(),), **options}
    )
    assert result.execution == "invalid"  # nosec B101


@pytest.mark.parametrize(
    "options",
    (
        {"kind": "coefficient"},
        {"name": ""},
        {"unit": ""},
        {"step": 0},
        {"step": -1},
        {"step": True},
        {"step": float("nan")},
    ),
)
def test_invalid_change_declaration(options):
    with pytest.raises(ValueError):
        DispatchChange(**{"kind": "input", "name": "load", "unit": "MW", **options})


def test_unsupported_bindings_do_not_redispatch_saved_rules_or_tracking():
    model, domain = example()
    for objective in (
        None,
        ControlTrackingObjective((ControlTarget("a", 8, 1, "MW"),)),
    ):
        result = (
            replace(model, objective=objective)
            .as_claim(domain)
            .dispatch_sensitivity({"load": 16}, changes=(load_change(),))
        )
        assert result.execution == "unsupported"  # nosec B101
    network, loads = network_example()
    saved = network.compile().as_claim(loads, controller=boiler_controller())
    assert (
        saved.dispatch_sensitivity(
            loads.nominal, changes=(DispatchChange("input", "dryer", "MW"),)
        ).execution
        == "unsupported"
    )  # nosec B101
    partial = model.as_claim(
        domain,
        recourse=RecoursePolicy(
            "static",
            tuple(
                DecisionRule(c.name, c.unit, "operation", ()) for c in model.controls
            ),
        ),
    )
    assert (
        partial.dispatch_sensitivity({"load": 16}, changes=(load_change(),)).execution
        == "unsupported"
    )  # nosec B101


def test_missing_optional_backend_is_explicit(monkeypatch):
    from operange import _cvxpy_backend

    def missing():
        raise _cvxpy_backend.BackendUnavailable("Injected missing backend")

    monkeypatch.setattr(_cvxpy_backend, "load_cvxpy", missing)
    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 16}, changes=(load_change(),), backend="cvxpy"
    )
    assert result.execution == "unsupported" and result.base is None  # nosec B101


def test_numerical_failure_and_invalid_basis_do_not_assert_derivatives(monkeypatch):
    model, domain = example()
    for basis in (None, [0, 1]):  # Opposite balance rows are linearly dependent.
        monkeypatch.setattr(
            ParametricDispatch, "proposal", lambda *args, **kwargs: (basis, [])
        )
        result = model.as_claim(domain).dispatch_sensitivity(
            {"load": 16}, changes=(load_change(),), max_refinements=0
        )
        assert result.execution == "unresolved"  # nosec B101
        assert all(
            s.value is None
            for s in (result.sensitivities[0].backward, result.sensitivities[0].forward)
        )  # nosec B101
        assert verify_result(result).status == "unsupported"  # nosec B101


def altered_side(result, function, *, forward=True):
    item = result.sensitivities[0]
    side = item.forward if forward else item.backward
    side = function(side)
    item = replace(item, **{"forward" if forward else "backward": side})
    return replace(result, sensitivities=(item, *result.sensitivities[1:]))


@pytest.mark.parametrize(
    "change",
    ("slope", "interval", "commands", "rates", "dual", "missing_row", "base_value"),
)
def test_fresh_identity_does_not_hide_a_forged_local_certificate(change):
    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 16}, changes=(load_change(100),)
    )

    def forge(side):
        if change == "slope":
            return replace(side, slope_exact="3")
        if change == "interval":
            return replace(side, valid_step=50)
        data = json.loads(json.dumps(side.to_dict()["proof"]))
        if change == "commands":
            data["controls_exact"]["a"] = "13"
        elif change == "rates":
            data["control_rates_exact"]["a"] = "1"
        elif change == "dual":
            data["multipliers_exact"][0] = "-1"
        elif change == "missing_row":
            del data["row_refs"][0]
            del data["multipliers_exact"][0]
        else:
            data["base_value_exact"] = "24"
        return replace(side, proof=data)

    forged = altered_side(result, forge)
    assert DispatchSensitivityResult.from_json(forged.to_json()) == forged  # nosec B101
    assert verify_result(forged).status == "failed"  # nosec B101


def test_bad_exclusion_ray_and_changed_request_are_rejected():
    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 20.5}, changes=(load_change(),)
    )
    forged = altered_side(
        result,
        lambda side: replace(
            side,
            proof={
                **side.proof,
                "multipliers_exact": ["0"] * len(side.proof["multipliers_exact"]),
            },
        ),
    )
    assert verify_result(forged).status == "failed"  # nosec B101
    forged = replace(result, request={**result.request, "realization": {"load": 16}})
    assert verify_result(forged).status == "failed"  # nosec B101
    forged = replace(result, request={**result.request, "changes": []})
    assert verify_result(forged).status == "failed"  # nosec B101


def test_offline_verifier_uses_neither_the_parametric_compiler_nor_any_solver(
    monkeypatch,
):
    from operange import _parametric_dispatch, linear

    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 12}, changes=(load_change(),)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Independent verification cannot use the producer or a solver"
        )

    monkeypatch.setattr(_parametric_dispatch, "problem", forbidden)
    monkeypatch.setattr(_parametric_dispatch, "change_rates", forbidden)
    monkeypatch.setattr(ParametricDispatch, "segment", forbidden)
    monkeypatch.setattr(linear, "linprog", forbidden)
    monkeypatch.setattr(linear, "solve_lp", forbidden)
    checked(result)


@pytest.mark.parametrize(
    "mutation",
    ("hash", "schema", "extra", "zero_denominator", "missing", "invalid_array"),
)
def test_loader_and_verification_reject_malformed_documents(mutation):
    model, domain = example()
    result = model.as_claim(domain).dispatch_sensitivity(
        {"load": 16}, changes=(load_change(),)
    )
    data = result.to_dict()
    if mutation == "hash":
        data["request"]["max_refinements"] = 0
    elif mutation == "schema":
        data["schema_version"] = "dispatch_sensitivity/v2"
    elif mutation == "extra":
        data["extra"] = 1
    elif mutation == "zero_denominator":
        data["sensitivities"][0]["forward"]["slope_exact"] = "1/0"
    elif mutation == "invalid_array":
        data["sensitivities"] = None
    else:
        del data["contract"]
    with pytest.raises(ValueError):
        DispatchSensitivityResult.from_json(json.dumps(data))
    assert verify_result(data).status == "invalid"  # nosec B101
    assert (
        verify_result(result, expected_contract_id="another-contract").status
        == "failed"
    )  # nosec B101
    assert verify_result(result, max_checks=1).status == "unsupported"  # nosec B101


def test_public_example_and_verification_cli(tmp_path):
    exported = run_example()
    for study in exported.values():
        result = DispatchSensitivityResult.from_json(json.dumps(study["sensitivity"]))
        assert result.execution == "completed" and study["verification"]["verified"]  # nosec B101
    path = Path(tmp_path) / "sensitivity.json"
    path.write_text(json.dumps(exported["boiler_threshold"]["sensitivity"]))
    process = subprocess.run(
        [sys.executable, "-m", "operange.verify", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )  # nosec B603
    assert process.returncode == 0, process.stdout + process.stderr  # nosec B101
    assert json.loads(process.stdout)["verified"]  # nosec B101
