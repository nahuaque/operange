"""Joint relief objectives preserve permissions, physical units and checked bounds."""

from dataclasses import replace
from fractions import Fraction as F

import pytest

from operange import AffineRequirement, DecisionRule, RecoursePolicy
from operange import _cvxpy_backend as backend
from operange import _joint_relief
from operange._linear_recourse import compile_system
from examples.distance_and_relief import relief_example, run_example
from tests.test_linear_diagnosis import evidence
from tests.test_linear_process import cases, small_model, portable


@pytest.fixture
def cvxpy():
    cp = pytest.importorskip("cvxpy")
    backend.clear_cache()
    yield cp
    backend.clear_cache()


def query(objective="linear", *, backend_name="scipy", changes=None, model=None):
    default, domain, original = relief_example()
    model = default if model is None else model
    claim = model.as_claim(domain)
    return claim, claim.evaluate_result(
        {"dryer": 12, "evaporator": 8},
        backend=backend_name,
        relief={
            "changes": original if changes is None else changes,
            "objective": objective,
        },
    )


@pytest.mark.parametrize("backend_name", ("scipy", "cvxpy"))
def test_linear_joint_relief_selects_cheaper_declared_change_and_replays(backend_name):
    if backend_name == "cvxpy":
        pytest.importorskip("cvxpy")
    claim, result = query(backend_name=backend_name)
    data = evidence(result, "relief")
    assert result.payload.feasibility == "infeasible"  # nosec B101
    assert (
        data["resolution"] == "minimum_verified" and data["lower"] == data["upper"] == 1
    )  # nosec B101
    assert [c["physical_relief"] for c in data["candidate"]["changes"]] == [0, 1]  # nosec B101
    assert data["unit"] == "1"  # nosec B101
    assert not any(v.quantity_ref == "boiler_a" for v in result.payload.values)  # nosec B101
    check_proposal(claim, result, data)
    check_bound(data)
    portable(result)


def check_proposal(claim, result, data):
    updates = {
        c["constraint"]: c["changed_limit"] for c in data["candidate"]["changes"]
    }
    model = replace(
        claim.adapter,
        operating_limits=tuple(
            replace(r, limit=updates.get(r.name, r.limit))
            for r in claim.adapter.operating_limits
        ),
    )
    policy = RecoursePolicy(
        "fixed",
        tuple(
            DecisionRule(
                n,
                next(c.unit for c in model.controls if c.name == n),
                "operation",
                fixed_value=v,
            )
            for n, v in data["candidate"]["controls"].items()
        ),
    )
    checked = model.as_claim(claim.domain, recourse=policy).evaluate_result(
        result.request["realization"]
    )
    assert checked.payload.feasibility == "feasible"  # nosec B101
    assert model.requirements == claim.adapter.requirements  # nosec B101


def check_bound(data):
    proof = data["lower_certificate"]
    weights = list(map(F, proof["multipliers"]))
    rows = [list(map(F, row)) for row in data["rows_exact"]]
    linear = list(map(F, data["objective_linear_exact"]))
    quadratic = list(map(F, data["objective_diagonal_exact"]))
    slopes = [
        c + sum(a * w for a, w in zip(col, weights))
        for c, col in zip(linear, zip(*rows))
    ]
    minimizers = [
        min(F(1), max(F(0), -c / (2 * d))) if d else F(c < 0)
        for c, d in zip(slopes, quadratic)
    ]
    lower = sum(
        d * z * z + c * z for d, c, z in zip(quadratic, slopes, minimizers)
    ) - sum(F(b) * w for b, w in zip(data["upper_exact"], weights))
    assert lower == F(proof["minimization_lower_exact"])  # nosec B101
    candidate = data["candidate"]
    squared = data["declaration"]["objective"] == "quadratic"
    objective = sum(
        F(c["weight"]) * (F(c["relief_exact"]) / F(c["scale"])) ** (2 if squared else 1)
        for c in candidate["changes"]
    )
    assert objective == F(candidate["objective_exact"])  # nosec B101
    assert F(data["lower"]) <= lower <= objective <= F(data["upper"])  # nosec B101


def test_quadratic_joint_relief_spreads_changes_with_exact_global_bound(cvxpy):
    claim, result = query("quadratic", backend_name="cvxpy")
    data = evidence(result, "relief")
    assert data["resolution"] == "minimum_verified"  # nosec B101
    assert data["lower"] <= 0.64 <= data["upper"]  # nosec B101
    assert data["upper"] - data["lower"] <= 1e-8  # nosec B101
    assert [
        c["physical_relief"] for c in data["candidate"]["changes"]
    ] == pytest.approx([0.48, 0.64], abs=1e-7)  # nosec B101
    check_proposal(claim, result, data)
    check_bound(data)
    portable(result)


def test_changing_scales_changes_the_declared_linear_tradeoff():
    _, _, changes = relief_example()
    changes = ({**changes[0], "scale": 2}, changes[1])
    _, result = query(changes=changes)
    data = evidence(result, "relief")
    assert data["resolution"] == "minimum_verified"  # nosec B101
    assert data["upper"] == pytest.approx(2 / 3)  # nosec B101
    assert data["candidate"]["changes"][0]["physical_relief"] == pytest.approx(4 / 3)  # nosec B101
    assert data["candidate"]["changes"][1]["physical_relief"] < 1e-8  # nosec B101
    check_bound(data)


def test_mixed_physical_units_use_declared_scales_in_the_same_objective(cvxpy):
    model, _, changes = relief_example()
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
            if o.name == "fuel"
            else o
            for o in model.outputs
        ),
        operating_limits=tuple(
            replace(r, limit=r.limit * 1000) if r.name == "shared_fuel" else r
            for r in model.operating_limits
        ),
    )
    changes = (changes[0], {**changes[1], "unit": "kW", "scale": 1000, "maximum": 2000})
    claim, result = query(
        "quadratic", backend_name="cvxpy", changes=changes, model=model
    )
    data = evidence(result, "relief")
    assert data["resolution"] == "minimum_verified"  # nosec B101
    assert data["upper"] == pytest.approx(0.64, abs=1e-8)  # nosec B101
    assert data["candidate"]["changes"][1]["physical_relief"] == pytest.approx(
        640, abs=1e-3
    )  # nosec B101
    check_proposal(claim, result, data)
    check_bound(data)


def test_insufficient_combined_relief_has_an_exact_contradiction():
    _, _, changes = relief_example()
    _, result = query(changes=tuple({**c, "maximum": 0.1} for c in changes))
    data = evidence(result, "relief")
    assert data["resolution"] == "unreachable"  # nosec B101
    assert F(data["infeasibility_certificate"]["contradiction_gap_exact"]) > 0  # nosec B101
    portable(result)


def test_zero_relief_and_fixed_commands_with_lower_limits_keep_their_meaning():
    model = small_model(
        limits=(AffineRequirement("minimum_margin", "margin", 0.5, "ge", 0),)
    )
    domain = cases(model, 0.5)
    fixed = RecoursePolicy(
        "fixed", (DecisionRule("power", "MW", "operation", fixed_value=0.5),)
    )
    changes = (
        {"constraint": "minimum_margin", "maximum": 1, "scale": 2, "unit": "MW"},
    )
    result = model.as_claim(domain, recourse=fixed).evaluate_result(
        {"load": 0.5}, relief={"changes": changes, "objective": "linear"}
    )
    data = evidence(result, "relief")
    assert data["lower"] == data["upper"] == 0.25  # nosec B101
    assert data["candidate"]["changes"][0]["changed_limit"] == 0  # nosec B101
    assert data["candidate"]["controls"] == {"power": 0.5}  # nosec B101
    model, domain, changes = relief_example()
    result = model.as_claim(domain).evaluate_result(
        {"dryer": 10, "evaporator": 6},
        relief={"changes": changes, "objective": "linear"},
    )
    assert evidence(result, "relief")["upper"] == 0  # nosec B101


@pytest.mark.parametrize(
    "fault",
    (
        "service",
        "duplicate",
        "unit",
        "scale",
        "weight",
        "maximum",
        "objective",
        "empty",
        "unknown",
    ),
)
def test_invalid_relief_declarations_are_rejected(fault):
    model, domain, changes = relief_example()
    options = {"changes": [dict(c) for c in changes], "objective": "linear"}
    if fault == "service":
        options["changes"][0]["constraint"] = "meet_load"
    elif fault == "duplicate":
        options["changes"][1] = options["changes"][0]
    elif fault == "unit":
        options["changes"][0]["unit"] = "kW"
    elif fault in ("scale", "weight", "maximum"):
        options["changes"][0][fault] = 0
    elif fault == "objective":
        options["objective"] = "implicit"
    elif fault == "empty":
        options["changes"] = []
    else:
        options["unexpected"] = 1
    result = model.as_claim(domain).evaluate_result(
        {"dryer": 12, "evaporator": 8}, relief=options
    )
    assert result.execution == "invalid"  # nosec B101


def test_quadratic_backend_requirement_is_explicit():
    _, result = query("quadratic")
    assert result.execution == "unsupported"  # nosec B101


def test_unexportable_scaled_objective_is_unresolved_instead_of_infinite(cvxpy):
    _, _, changes = relief_example()
    _, result = query(
        "quadratic",
        backend_name="cvxpy",
        changes=tuple({**c, "scale": 1e-300} for c in changes),
    )
    data = evidence(result, "relief")
    assert data["resolution"] == "unresolved" and data["upper"] is None  # nosec B101
    portable(result)


def test_false_solver_status_does_not_prove_relief_or_unreachability(
    cvxpy, monkeypatch
):
    model, domain, changes = relief_example()
    claim = model.as_claim(domain)
    point = {"dryer": 12, "evaporator": 8}
    system = compile_system(claim, point)
    monkeypatch.setattr(
        backend,
        "solve_bounded",
        lambda *a, **k: backend.CandidateSolve(None, None, [{"status": "infeasible"}]),
    )
    from operange import _dispatch_objective

    monkeypatch.setattr(_dispatch_objective, "solve_bounded", backend.solve_bounded)
    result = _joint_relief.relief_evidence(
        claim,
        point,
        system,
        {"changes": changes, "objective": "quadratic"},
        backend="cvxpy",
    )
    assert result.details["resolution"] == "unresolved"  # nosec B101
    assert result.details["candidate"] is None and result.details["lower"] == 0  # nosec B101


def test_worked_example_exports_minima_and_separate_reaudits(cvxpy):
    from operange import result_from_json
    import json

    results = {
        n: result_from_json(json.dumps(data)) for n, data in run_example().items()
    }
    for name in ("linear", "quadratic"):
        assert (
            evidence(results[name + "_relief"], "relief")["resolution"]
            == "minimum_verified"
        )  # nosec B101
        assert results[name + "_reaudit"].payload.verdict == "pass"  # nosec B101
    for result in results.values():
        portable(result)
