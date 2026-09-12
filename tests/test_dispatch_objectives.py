"""Operating preferences select dispatch without weakening physical requirements."""

from dataclasses import replace
from fractions import Fraction as F
import json

import pytest

from operange import (
    AffineOutput,
    AffineTerm,
    ControlTarget,
    ControlTrackingObjective,
    DecisionRule,
    FrozenController,
    LinearObjective,
    LinearProcessAdapter,
    RecoursePolicy,
)
from operange import _dispatch_objective as implementation
from operange import _cvxpy_backend as cvxpy_backend
from operange._change_contracts import compare_contracts
from examples.linear_dispatch import example
from examples.frozen_controllers import envelope_example
from tests.test_linear_process import cases, portable, small_model


@pytest.fixture(params=("scipy", "cvxpy"))
def backend(request):
    if request.param == "cvxpy":
        pytest.importorskip("cvxpy")
    cvxpy_backend.clear_cache()
    yield request.param
    cvxpy_backend.clear_cache()


def values(result):
    return {v.quantity_ref: v.value for v in result.payload.values}


def proof(result):
    return next(
        e.details for e in result.evidence if e.evidence_id == "operating_objective"
    )


def tracking(*, a=8, b=8, weight=1):
    return ControlTrackingObjective(
        (
            ControlTarget("boiler_a", a, 1, "MW", weight),
            ControlTarget("boiler_b", b, 1, "MW"),
        )
    )


def check_certificate(result):
    data = proof(result)
    certificate = data["certificate"]
    multipliers = list(map(F, certificate["multipliers"]))
    assert all(v >= 0 for v in multipliers)  # nosec B101
    columns = list(zip(*data["rows_exact"]))
    lower = F(data["constant_exact"]) - sum(
        F(b) * v for b, v in zip(data["rhs_exact"], multipliers)
    )
    for index, (c, d) in enumerate(zip(data["linear_exact"], data["diagonal_exact"])):
        slope = F(c) + sum(F(a) * v for a, v in zip(columns[index], multipliers))
        d = F(d)
        z = min(F(1), max(F(0), -slope / (2 * d))) if d else F(slope < 0)
        lower += d * z * z + slope * z
    assert lower == F(certificate["minimization_lower_exact"])  # nosec B101
    obj = result.payload.objective
    optimum_bound = obj.lower.value if obj.sense == "minimize" else -obj.upper.value
    assert F(optimum_bound) <= lower  # nosec B101
    portable(result)


@pytest.mark.parametrize("sense,expected", (("minimize", 23), ("maximize", 27.5)))
def test_affine_objective_selects_dispatch_and_proves_bounds(backend, sense, expected):
    model, domain = example()
    claim = replace(model, objective=LinearObjective("fuel", sense)).as_claim(domain)
    result = claim.evaluate_result({"dryer": 10, "evaporator": 6}, backend=backend)
    obj = result.payload.objective
    assert result.payload.feasibility == "feasible" and result.execution == "completed"  # nosec B101
    assert obj.optimality == "verified" and obj.attainment == "attained"  # nosec B101
    assert obj.attained_value == pytest.approx(expected)  # nosec B101
    assert obj.lower.unit == obj.upper.unit == "MW"  # nosec B101
    optimum = (
        F(23) - 2 * F(1e-8) if sense == "minimize" else F(27.5) + F(1.25) * F(1e-8)
    )
    assert F(obj.lower.value) <= optimum <= F(obj.upper.value)  # nosec B101
    assert values(result)["boiler_a"] == pytest.approx(12 if sense == "minimize" else 6)  # nosec B101
    check_certificate(result)


def test_preferences_do_not_relax_infeasible_service_or_change_audit_verdict(backend):
    model, domain = example()
    claim = replace(model, objective=LinearObjective("fuel")).as_claim(domain)
    result = claim.audit_result(backend=backend)
    assert (
        result.payload.verdict == "fail"
        and result.payload.coverage.method == "complete_finite"
    )  # nosec B101
    failed = claim.evaluate_result(
        {"dryer": 12, "evaporator": 8},
        backend=backend,
        diagnose=True,
        relief={"constraint": "shared_fuel", "maximum": 2},
    )
    assert (
        failed.payload.feasibility == "infeasible" and failed.payload.objective is None
    )  # nosec B101
    assert any(e.subject == "recourse_infeasibility" for e in failed.evidence)  # nosec B101
    assert (
        next(
            e.details["resolution"]
            for e in failed.evidence
            if e.evidence_id == "relief"
        )
        == "minimum_verified"
    )  # nosec B101
    assert all(
        v.quantity_ref not in ("boiler_a", "boiler_b") for v in failed.payload.values
    )  # nosec B101
    portable(result)
    portable(failed)


def test_offsets_inputs_nonzero_control_origins_and_fixed_commands(backend):
    model = small_model(bounds=(-2, 2))
    model = replace(
        model,
        outputs=model.outputs
        + (
            AffineOutput(
                "operating_score",
                "MW",
                "thermal_power",
                3,
                (AffineTerm("power", 2, "MW/MW"), AffineTerm("load", 4, "MW/MW")),
                "Test score",
            ),
        ),
        objective=LinearObjective("operating_score"),
    )
    domain = cases(model, 0.5)
    result = model.as_claim(domain).evaluate_result({"load": 0.5}, backend=backend)
    assert (
        result.payload.objective.lower.value
        == result.payload.objective.upper.value
        == 6
    )  # nosec B101
    assert values(result)["power"] == 0.5  # nosec B101
    check_certificate(result)
    fixed = RecoursePolicy(
        "fixed", (DecisionRule("power", "MW", "operation", fixed_value=1),)
    )
    result = model.as_claim(domain, recourse=fixed).evaluate_result(
        {"load": 0.5}, backend=backend
    )
    assert (
        result.payload.objective.lower.value
        == result.payload.objective.upper.value
        == 7
    )  # nosec B101
    check_certificate(result)


@pytest.mark.parametrize(
    "fault",
    ("failed_solve", "bad_primal", "bad_dual", "wrong_shape", "false_infeasibility"),
)
def test_failed_optimality_preserves_feasible_dispatch_and_physical_audit(
    backend, monkeypatch, fault
):
    model, domain = example()
    claim = replace(model, objective=LinearObjective("fuel")).as_claim(domain)
    m = len(model.operating_limits) + len(model.requirements)
    proposals = {
        "failed_solve": (None, None),
        "false_infeasibility": (None, None),
        "bad_primal": ([0, 0], [0] * m),
        "bad_dual": ([1, 0.4], [float("nan")] * m),
        "wrong_shape": ([1], [0] * m),
    }
    primal, dual = proposals[fault]
    monkeypatch.setattr(
        implementation,
        "_numerical",
        lambda *a: (
            primal,
            dual,
            [
                {
                    "status": "infeasible"
                    if fault == "false_infeasibility"
                    else "unresolved"
                }
            ],
        ),
    )
    result = claim.evaluate_result({"dryer": 10, "evaporator": 6}, backend=backend)
    assert result.execution == "completed" and result.payload.feasibility == "feasible"  # nosec B101
    assert result.payload.objective.optimality == "unknown"  # nosec B101
    assert (
        result.payload.objective.lower.value
        <= 23
        <= result.payload.objective.upper.value + 1e-7
    )  # nosec B101
    assert any(d.code == "dispatch_optimality_unresolved" for d in result.diagnostics)  # nosec B101
    assert not any(e.subject == "recourse_infeasibility" for e in result.evidence)  # nosec B101
    check_certificate(result)
    # Audits concern hard feasibility, not whether the numerical optimizer closed its gap.
    nominal_only = replace(
        domain, scenarios=tuple(s for s in domain.scenarios if s.name == "nominal")
    )
    audit = replace(claim, domain=nominal_only).audit_result(backend=backend)
    assert audit.payload.verdict == "pass"  # nosec B101
    portable(audit)


def test_default_model_manifest_and_operating_change_classification():
    model, domain = example()
    assert "objective" not in model.to_dict()  # nosec B101
    assert (
        model.as_claim(domain)
        .evaluate_result({"dryer": 10, "evaporator": 6})
        .payload.objective
        is None
    )  # nosec B101
    preferred = replace(model, objective=LinearObjective("fuel"))
    assert (
        LinearProcessAdapter(**json.loads(json.dumps(preferred.to_dict()))) == preferred
    )  # nosec B101
    difference = compare_contracts(
        model.as_claim(domain).contract, preferred.as_claim(domain).contract
    )
    assert difference.model_preserved and difference.requirements_preserved  # nosec B101
    assert "operating" in difference.changed_sections  # nosec B101


@pytest.mark.parametrize(
    "objective",
    (
        {"output": "missing"},
        {"output": "fuel", "sense": "guess"},
        {"output": "fuel", "tolerance": -1},
        {"targets": []},
        {"targets": [{"control": "missing", "target": 0, "scale": 1, "unit": "MW"}]},
        {"targets": [{"control": "boiler_a", "target": 0, "scale": 0, "unit": "MW"}]},
        {"targets": [{"control": "boiler_a", "target": 0, "scale": 1, "unit": "kW"}]},
        {
            "targets": [
                {
                    "control": "boiler_a",
                    "target": 0,
                    "scale": 1,
                    "weight": -1,
                    "unit": "MW",
                }
            ]
        },
        {
            "targets": [
                {
                    "control": "boiler_a",
                    "target": float("nan"),
                    "scale": 1,
                    "unit": "MW",
                }
            ]
        },
        {
            "targets": [{"control": "boiler_a", "target": 0, "scale": 1, "unit": "MW"}],
            "quantity_id": "fuel",
        },
    ),
)
def test_invalid_objectives_rejected_at_declaration(objective):
    model, _ = example()
    with pytest.raises(ValueError):
        replace(model, objective=objective)


def test_tracking_requires_explicit_cvxpy_and_missing_backend_is_clear(monkeypatch):
    model, domain = example()
    claim = replace(model, objective=tracking()).as_claim(domain)
    assert claim.audit_result().execution == "unsupported"  # nosec B101

    def unavailable(*args):
        raise ImportError("Injected missing dependency")

    monkeypatch.setattr(cvxpy_backend, "import_module", unavailable)
    result = claim.audit_result(backend="cvxpy")
    assert result.execution == "unsupported" and "operange[cvxpy]" in result.to_json()  # nosec B101


@pytest.mark.parametrize(
    "weight,expected_a,expected_b,expected_score", ((1, 9, 9, 2), (4, 8.4, 9.6, 3.2))
)
def test_strictly_convex_tracking_and_exact_quadratic_certificate(
    weight, expected_a, expected_b, expected_score
):
    pytest.importorskip("cvxpy")
    model, domain = example()
    result = (
        replace(model, objective=tracking(weight=weight))
        .as_claim(domain)
        .evaluate_result({"dryer": 12, "evaporator": 6}, backend="cvxpy")
    )
    assert result.payload.feasibility == "feasible"  # nosec B101
    assert values(result)["boiler_a"] == pytest.approx(expected_a, abs=1e-4)  # nosec B101
    assert values(result)["boiler_b"] == pytest.approx(expected_b, abs=1e-4)  # nosec B101
    assert result.payload.objective.attained_value == pytest.approx(expected_score)  # nosec B101
    assert result.payload.objective.optimality == "verified"  # nosec B101
    assert proof(result)["strictly_convex_on_adjustable_controls"]  # nosec B101
    optimum = F(weight) / F(weight + 1) * (F(2) - F(1e-8)) ** 2
    assert (
        F(result.payload.objective.lower.value)
        <= optimum
        <= F(result.payload.objective.upper.value)
    )  # nosec B101
    check_certificate(result)


def test_tracking_scales_fixed_commands_partial_targets_and_cache_updates():
    pytest.importorskip("cvxpy")
    model, domain = example()
    cvxpy_backend.clear_cache()
    for a, b in ((8, 8), (10, 6), (8, 8)):
        result = (
            replace(model, objective=tracking(a=a, b=b))
            .as_claim(domain)
            .evaluate_result({"dryer": 10, "evaporator": 6}, backend="cvxpy")
        )
        assert values(result)["boiler_a"] == pytest.approx(a, abs=1e-4)  # nosec B101
        assert result.payload.objective.optimality == "verified"  # nosec B101
    target = ControlTrackingObjective((ControlTarget("boiler_a", 8, 2, "MW", 4),))
    fixed = RecoursePolicy(
        "static",
        (
            DecisionRule("boiler_a", "MW", "operation", fixed_value=10),
            DecisionRule("boiler_b", "MW", "operation", model.input_space.names),
        ),
    )
    result = (
        replace(model, objective=target)
        .as_claim(domain, recourse=fixed)
        .evaluate_result({"dryer": 10, "evaporator": 6}, backend="cvxpy")
    )
    assert (
        result.payload.objective.lower.value
        == result.payload.objective.upper.value
        == 4
    )  # nosec B101
    assert not proof(result)["strictly_convex_on_adjustable_controls"]  # nosec B101
    check_certificate(result)


def test_frozen_controller_retains_preference_but_executes_prescribed_commands():
    claim = envelope_example()
    model = replace(claim.adapter.model, objective=tracking())
    prescribed = model.as_claim(claim.domain, controller=claim.adapter.controller)
    frozen = FrozenController.from_json(prescribed.freeze().to_json())
    result = frozen.evaluate_result(claim.domain.nominal)
    original = claim.evaluate_result(claim.domain.nominal)
    assert result.payload.values == original.payload.values  # nosec B101
    assert result.payload.objective is None  # nosec B101
    assert frozen.model.objective == tracking()  # nosec B101
    portable(result)


@pytest.mark.parametrize(
    "target,expected_power,expected_score", ((2, 1, 1), (-1, 0.5, 2.25))
)
def test_unattainable_targets_respect_control_bounds_and_service(
    target, expected_power, expected_score
):
    pytest.importorskip("cvxpy")
    model = replace(
        small_model(),
        objective=ControlTrackingObjective((ControlTarget("power", target, 1, "MW"),)),
    )
    result = model.as_claim(cases(model, 0.5)).evaluate_result(
        {"load": 0.5}, backend="cvxpy"
    )
    power = values(result)["power"]
    assert 0.5 <= power <= 1  # nosec B101
    assert power == pytest.approx(expected_power)  # nosec B101
    assert result.payload.objective.attained_value == pytest.approx(expected_score)  # nosec B101
    assert result.payload.objective.optimality == "verified"  # nosec B101
    check_certificate(result)


def test_false_optimal_status_and_finite_duals_cannot_certify_suboptimal_dispatch(
    monkeypatch,
):
    model, domain = example()
    claim = replace(model, objective=LinearObjective("fuel")).as_claim(domain)
    monkeypatch.setattr(
        implementation,
        "_numerical",
        lambda *a: ([10 / 12, 0.6], [1, 1, 1], [{"status": "optimal"}]),
    )
    result = claim.evaluate_result({"dryer": 10, "evaporator": 6})
    assert result.payload.feasibility == "feasible"  # nosec B101
    assert result.payload.objective.attained_value == pytest.approx(24.5)  # nosec B101
    assert result.payload.objective.optimality == "unknown"  # nosec B101
    check_certificate(result)


def test_worked_objective_example_exports_and_replays():
    pytest.importorskip("cvxpy")
    from examples.dispatch_objectives import run_example
    from operange import result_from_json

    results = {n: result_from_json(json.dumps(v)) for n, v in run_example().items()}
    assert results["minimum_fuel"].payload.objective.optimality == "verified"  # nosec B101
    assert results["track_nominal_commands"].payload.objective.optimality == "verified"  # nosec B101
    assert results["track_at_peak"].payload.objective.optimality == "verified"  # nosec B101
    assert results["fuel_audit"].payload.verdict == "fail"  # nosec B101
    assert results["combined_peaks_still_infeasible"].payload.objective is None  # nosec B101
    for result in results.values():
        portable(result)
