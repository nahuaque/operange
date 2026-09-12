"""Conflict minimality and one-limit relief are independently checked."""

from dataclasses import replace
from fractions import Fraction

import pytest

from operange import AffineRequirement, DecisionRule, RecoursePolicy
from operange import linear
from operange._linear_recourse import compile_system, solve_system
from operange._linear_diagnosis import conflict_evidence
from operange.results import SolveEvidence
from examples.linear_dispatch import example
from tests.test_linear_process import cases, portable, small_model


def evidence(result, name):
    return next(e for e in result.evidence if e.evidence_id == name).details


def diagnosed(*, maximum=2, model=None, tolerance=1e-8):
    default, domain = example()
    claim = (model or default).as_claim(domain)
    return claim, claim.evaluate_result(
        {"dryer": 12, "evaporator": 8},
        diagnose=True,
        relief={
            "constraint": "shared_fuel",
            "maximum": maximum,
            "tolerance": tolerance,
        },
    )


def test_irreducible_conflict_records_feasible_removals_and_bound_dependency():
    claim, result = diagnosed()
    assert result.payload.feasibility == "infeasible"  # nosec B101
    data = evidence(result, "conflict")
    assert set(data["constraint_refs"]) == {"meet_load", "shared_fuel"}  # nosec B101
    assert data["irreducible"] and not data["minimum_cardinality"]  # nosec B101
    assert data["certificate_bound_refs"] == ("control_upper:boiler_a",)  # nosec B101
    system = compile_system(claim, result.request["realization"])
    for removed, record in data["removal_feasibility"].items():
        commands = record["controls"]
        z = [
            (Fraction(commands[c.name]) - Fraction(c.lower)) / span
            for c, span in zip(system.controls, system.spans)
        ]
        assert all(0 <= v <= 1 for v in z)  # nosec B101
        for name, row, rhs in zip(system.row_refs, system.rows, system.upper):
            if name in data["constraint_refs"] and name != removed:
                assert sum(a * v for a, v in zip(row, z)) <= rhs  # nosec B101
    assert all(
        v.quantity_ref not in ("boiler_a", "boiler_b") for v in result.payload.values
    )  # nosec B101
    portable(result)


def test_relief_has_exact_lower_certificate_and_feasible_changed_limit():
    claim, result = diagnosed()
    data = evidence(result, "relief")
    assert data["resolution"] == "minimum_verified"  # nosec B101
    expected = Fraction(1) - 3 * Fraction(1e-8)
    assert Fraction(data["lower"]) <= expected <= Fraction(data["upper"])  # nosec B101
    assert data["upper"] - data["lower"] <= data["tolerance"]  # nosec B101
    proof = data["lower_certificate"]
    rows = [[Fraction(v) for v in row] for row in data["rows_exact"]]
    rhs = [Fraction(v) for v in data["upper_exact"]]
    weights = [Fraction(v) for v in proof["multipliers"]]
    objective = [Fraction(v) for v in proof["objective_exact"]]
    lower = sum(
        min(Fraction(0), c + sum(a * w for a, w in zip(column, weights)))
        for c, column in zip(objective, zip(*rows))
    ) - sum(b * w for b, w in zip(rhs, weights))
    assert lower == Fraction(proof["lower_exact"])  # nosec B101
    changed = replace(
        claim.adapter,
        operating_limits=tuple(
            replace(r, limit=data["candidate"]["changed_limit"])
            if r.name == "shared_fuel"
            else r
            for r in claim.adapter.operating_limits
        ),
    )
    commands = data["candidate"]["controls"]
    fixed = RecoursePolicy(
        "fixed",
        tuple(
            DecisionRule(n, "MW", "operation", fixed_value=v)
            for n, v in commands.items()
        ),
    )
    checked = changed.as_claim(claim.domain, recourse=fixed).evaluate_result(
        result.request["realization"]
    )
    assert checked.payload.feasibility == "feasible"  # nosec B101
    portable(result)


def test_insufficient_allowed_relief_is_certified_unreachable():
    _, result = diagnosed(maximum=0.5)
    data = evidence(result, "relief")
    assert data["resolution"] == "unreachable"  # nosec B101
    assert Fraction(data["infeasibility_certificate"]["contradiction_gap_exact"]) > 0  # nosec B101
    portable(result)


def test_zero_relief_on_a_feasible_point_and_opt_in_only():
    model, domain = example()
    claim = model.as_claim(domain)
    plain = claim.evaluate_result({"dryer": 10, "evaporator": 6})
    assert not any(e.evidence_id in ("conflict", "relief") for e in plain.evidence)  # nosec B101
    result = claim.evaluate_result(
        {"dryer": 10, "evaporator": 6},
        relief={"constraint": "shared_fuel", "maximum": 2},
    )
    assert result.payload.feasibility == "feasible"  # nosec B101
    assert evidence(result, "relief")["upper"] == 0  # nosec B101


@pytest.mark.parametrize(
    "options",
    (
        {"diagnose": 1},
        {"relief": {}},
        {"relief": {"constraint": "meet_load", "maximum": 1}},
        {"relief": {"constraint": "shared_fuel", "maximum": 0}},
        {"relief": {"constraint": "shared_fuel", "maximum": 1, "tolerance": -1}},
    ),
)
def test_invalid_queries_do_not_relax_service(options):
    model, domain = example()
    assert (
        model.as_claim(domain)
        .evaluate_result({"dryer": 12, "evaporator": 8}, **options)
        .execution
        == "invalid"
    )  # nosec B101


def test_unresolved_removal_does_not_claim_irreducibility(monkeypatch):
    model, domain = example()
    claim = model.as_claim(domain)
    system = compile_system(claim, {"dryer": 12, "evaporator": 8})
    solution = solve_system(system, model.solver_tolerance)
    monkeypatch.setattr(
        linear,
        "solve_lp",
        lambda *a, **k: (None, SolveEvidence("infeasible", "Injected", "test", 1e-9)),
    )
    result = conflict_evidence(system, solution, model.solver_tolerance)
    assert result.outcome == "unresolved"  # nosec B101
    assert not result.details["irreducible"]  # nosec B101
    assert result.details["unresolved_removals"]  # nosec B101


def test_invalid_solver_candidates_do_not_create_relief_or_minimality(monkeypatch):
    monkeypatch.setattr(
        linear,
        "solve_lp",
        lambda objective, *a, **k: (
            [float("nan")] * len(objective),
            SolveEvidence("optimal", "Injected", "test", 1e-9),
        ),
    )
    _, result = diagnosed()
    assert result.payload.feasibility == "unknown"  # nosec B101
    assert not any(e.evidence_id in ("conflict", "relief") for e in result.evidence)  # nosec B101


def test_lower_operating_limit_relaxes_downward_with_fixed_commands():
    model = small_model(
        limits=(AffineRequirement("minimum_margin", "margin", 0.5, "ge", 0),)
    )
    domain = cases(model, 0.5)
    policy = RecoursePolicy(
        "fixed", (DecisionRule("power", "MW", "operation", fixed_value=0.5),)
    )
    result = model.as_claim(domain, recourse=policy).evaluate_result(
        {"load": 0.5},
        diagnose=True,
        relief={"constraint": "minimum_margin", "maximum": 1},
    )
    data = evidence(result, "relief")
    assert result.payload.feasibility == "infeasible"  # nosec B101
    assert data["resolution"] == "minimum_verified"  # nosec B101
    assert data["lower"] == data["upper"] == 0.5  # nosec B101
    assert data["candidate"]["changed_limit"] == 0  # nosec B101
    assert data["candidate"]["controls"] == {"power": 0.5}  # nosec B101
    portable(result)


def test_relief_solver_failure_preserves_verified_original_infeasibility(monkeypatch):
    import operange._linear_process_results as result_module

    model, domain = example()
    claim = model.as_claim(domain)
    solution = solve_system(
        compile_system(claim, {"dryer": 12, "evaporator": 8}), model.solver_tolerance
    )
    monkeypatch.setattr(result_module, "solve_system", lambda *a, **k: solution)
    monkeypatch.setattr(
        linear,
        "solve_lp",
        lambda *a, **k: (None, SolveEvidence("infeasible", "Injected", "test", 1e-9)),
    )
    result = claim.evaluate_result(
        {"dryer": 12, "evaporator": 8},
        diagnose=True,
        relief={"constraint": "shared_fuel", "maximum": 2},
    )
    assert (
        result.payload.feasibility == "infeasible" and result.execution == "completed"
    )  # nosec B101
    assert evidence(result, "relief")["resolution"] == "unresolved"  # nosec B101
    assert evidence(result, "relief")["upper"] is None  # nosec B101
    assert not evidence(result, "conflict")["irreducible"]  # nosec B101
    portable(result)
