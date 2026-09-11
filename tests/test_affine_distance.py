"""Analytical distance cases, exact certificates and failed numerical searches."""

from dataclasses import replace
from fractions import Fraction
from math import nextafter, inf
from types import SimpleNamespace

import pytest

from operange import (
    AffineOutput,
    AffineProcessAdapter,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    BudgetSet,
    Coordinate,
    DecisionRule,
    FiniteSet,
    LinearConstraint,
    NormalizedLInf,
    Parameter,
    ParameterSpace,
    PolytopeSet,
    QuantitySpec,
    RecoursePolicy,
    Scenario,
    result_from_json,
)
from operange import linear
from operange.results import SolveEvidence
from examples.steam_header import example


def claim_for(model=None, domain=None, *, requirements=None, distance=None):
    default_model, default_domain = example()
    model = default_model if model is None else model
    domain = default_domain if domain is None else domain
    return model.as_claim(
        domain,
        requirements=requirements,
        distance=NormalizedLInf(domain.space) if distance is None else distance,
    )


def portable(result):
    for compact in (False, True):
        assert (
            result_from_json(result.to_json(compact=compact)).to_dict()
            == result.to_dict()
        )  # nosec B101


def candidate(result):
    ref = result.payload.search.candidate_evaluation_ref
    return next(e for e in result.supporting_evaluations if e.ref == ref)


def verify_minimum(result, expected):
    search = result.payload.search
    assert result.execution == "completed" and search.resolution == "minimum_verified"  # nosec B101
    assert search.lower.value <= expected + 1e-14  # nosec B101
    assert search.upper.value >= expected - 1e-14  # nosec B101
    assert search.upper.value - search.lower.value <= search.tolerance  # nosec B101
    portable(result)


def test_box_boundary_and_positive_failure_are_different_queries():
    claim = claim_for()
    boundary = claim.boundary_result()
    verify_minimum(boundary, 0.75)
    assert (
        boundary.payload.verdict == "not_assessed" and boundary.payload.witness is None
    )  # nosec B101
    assert (
        boundary.payload.search.severity["reported_threshold_requirement"]
        == "steam_header"
    )  # nosec B101
    assert candidate(boundary).request["realization"] == {
        "dryer": 11.5,
        "evaporator": 7.5,
    }  # nosec B101
    breaking = claim.breaking_result(
        violation_margins={"steam_header": 0.01, "boiler_fuel": 0.01}
    )
    verify_minimum(breaking, 0.7525)
    assert breaking.payload.verdict == "fail"  # nosec B101
    replay = claim.evaluate_result(breaking.payload.witness.realizations[0])
    values = {v.quantity_ref: v.value for v in replay.payload.values}
    exact_residual = Fraction(values["dryer"]) + Fraction(values["evaporator"]) - 19
    assert exact_residual >= Fraction(0.01)  # nosec B101
    assert any(c.assessment == "violated" for c in replay.payload.constraint_checks)  # nosec B101


def test_subset_selection_changes_the_governing_requirement():
    claim = claim_for(requirements=("boiler_fuel",))
    boundary = claim.boundary_result()
    verify_minimum(boundary, 0.8)
    assert boundary.payload.search.threshold.quantity_ref == "residual:boiler_fuel"  # nosec B101
    verify_minimum(
        claim.breaking_result(violation_margins={"boiler_fuel": 0.01}), 0.802
    )


def test_coupled_polytope_changes_minimum_and_can_make_target_unreachable():
    model, box = example()
    clipped = PolytopeSet(box, (LinearConstraint("dryer_cap", {"dryer": 1}, 0.5),))
    claim = claim_for(model, clipped, requirements=("steam_header",))
    boundary = claim.boundary_result()
    verify_minimum(boundary, 1)
    assert candidate(boundary).request["realization"] == {"dryer": 11, "evaporator": 8}  # nosec B101
    # The fuel branch is unreachable here, but the steam boundary still governs.
    verify_minimum(claim_for(model, clipped).boundary_result(), 1)
    failure = claim.breaking_result(violation_margins={"steam_header": 0.01})
    assert (
        failure.execution == "completed"
        and failure.payload.search.resolution == "unreachable"
    )  # nosec B101
    assert failure.payload.witness is None and failure.payload.verdict == "not_assessed"  # nosec B101
    portable(failure)
    correlated = PolytopeSet(
        box, (LinearConstraint("joint", {"dryer": 1, "evaporator": 1}, 1),)
    )
    result = claim_for(model, correlated).boundary_result()
    assert result.payload.search.resolution == "unreachable"  # nosec B101


def test_polytope_equalities_are_checked_in_both_directions():
    model, box = example()
    domain = PolytopeSet(
        box, (LinearConstraint("ratio", {"dryer": 0.5, "evaporator": -1}, 0, "eq"),)
    )
    claim = claim_for(model, domain, requirements=("steam_header",))
    result = claim.boundary_result()
    verify_minimum(result, 1)
    point = candidate(result).request["realization"]
    assert Fraction(point["dryer"] - 10) / 2 == Fraction(point["evaporator"] - 6)  # nosec B101


def test_distance_uses_its_own_named_origins_and_scales_including_outside_baseline():
    model, domain = example()
    distance = NormalizedLInf(
        ParameterSpace(
            (
                Coordinate("evaporator", "MW", "Distance origin", 3, 1),
                Coordinate("dryer", "MW", "Distance origin", 7, 1),
            )
        )
    )
    result = claim_for(
        model, domain, requirements=("steam_header",), distance=distance
    ).boundary_result()
    verify_minimum(result, 4.5)
    assert result.payload.search.severity["origins"] == (3, 7)  # nosec B101


def scalar_claim(limit, *, relation="le", tolerance=0, constant=None, fixed=None):
    domain = BoxSet((Parameter("x", "MW", 0, -1, 1, 1, "Test envelope"),))
    terms = (AffineTerm("x", 1, "MW/MW"),) if constant is None else ()
    controls = ()
    recourse = None
    if fixed is not None:
        terms += (AffineTerm("control", 1, "MW/MW"),)
        controls = (QuantitySpec("control", "MW", "power", "control"),)
        recourse = RecoursePolicy(
            "fixed", (DecisionRule("control", "MW", "operation", fixed_value=fixed),)
        )
    model = AffineProcessAdapter(
        "Scalar",
        domain.space,
        (
            AffineOutput(
                "response", "MW", "power", constant or 0, terms, "Test equation"
            ),
        ),
        (AffineRequirement("limit", "response", limit, relation, tolerance),),
        controls,
    )
    return model.as_claim(
        domain, recourse=recourse, distance=NormalizedLInf(domain.space)
    )


def test_lower_requirements_and_fixed_controls_keep_their_meaning():
    claim = scalar_claim(-0.5, relation="ge", fixed=0.25)
    boundary = claim.boundary_result()
    verify_minimum(boundary, 0.75)
    assert candidate(boundary).request["realization"] == {"x": -0.75}  # nosec B101
    breaking = claim.breaking_result(violation_margins={"limit": 0.1})
    verify_minimum(breaking, 0.85)
    assert breaking.payload.witness.kind == "fixed_policy_failure"  # nosec B101


@pytest.mark.parametrize(
    "limit,constant,expected,verdict",
    [
        (0, None, "minimum_verified", "not_assessed"),
        (-0.5, None, "minimum_verified", "fail"),
        (1, 0, "unreachable", "not_assessed"),
        (0, 0, "minimum_verified", "not_assessed"),
    ],
)
def test_zero_distance_and_constant_outputs(limit, constant, expected, verdict):
    result = scalar_claim(limit, constant=constant).boundary_result()
    assert (
        result.payload.search.resolution == expected
        and result.payload.verdict == verdict
    )  # nosec B101
    if expected == "minimum_verified":
        assert (
            result.payload.search.lower.value == result.payload.search.upper.value == 0
        )  # nosec B101
    portable(result)


def test_boundary_within_requirement_tolerance_is_not_a_failure():
    result = scalar_claim(-1e-9, tolerance=1e-8).boundary_result()
    verify_minimum(result, 0)
    assert result.payload.verdict == "not_assessed" and result.payload.witness is None  # nosec B101


def test_mixed_units_require_explicit_per_requirement_margins():
    claim = scalar_claim(0.5)
    model = replace(
        claim.adapter,
        outputs=claim.adapter.outputs
        + (
            AffineOutput(
                "temperature",
                "K",
                "temperature",
                300,
                (AffineTerm("x", 10, "K/MW"),),
                "Test temperature",
            ),
        ),
        requirements=claim.adapter.requirements
        + (AffineRequirement("hot", "temperature", 308, tolerance=0),),
    )
    result = claim_for(model, claim.domain).breaking_result(
        violation_margins={"limit": 0.2, "hot": 0.5}
    )
    verify_minimum(result, 0.7)
    assert result.payload.search.severity["thresholds"]["hot"]["unit"] == "K"  # nosec B101


def test_false_solver_infeasibility_cannot_prove_unreachability(monkeypatch):
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=False, status=2, message="False infeasibility", x=None
        ),
    )
    result = claim_for().boundary_result()
    assert (
        result.execution == "unresolved"
        and result.payload.search.resolution == "unresolved"
    )  # nosec B101
    assert (
        result.payload.search.upper.state == "unknown"
        and result.payload.witness is None
    )  # nosec B101
    portable(result)


def test_verified_candidate_survives_unresolved_lower_bound(monkeypatch):
    original = linear.solve_lp

    def no_duals(objective, bounds, tolerance, **kwargs):
        if kwargs.get("equalities"):
            return None, SolveEvidence("unresolved", "No dual", "test", tolerance)
        return original(objective, bounds, tolerance, **kwargs)

    monkeypatch.setattr(linear, "solve_lp", no_duals)
    result = claim_for().breaking_result(
        violation_margins={"steam_header": 0.01, "boiler_fuel": 0.01}
    )
    assert (
        result.execution == "unresolved"
        and result.payload.search.resolution == "bounded"
    )  # nosec B101
    assert result.payload.verdict == "fail" and result.payload.witness is not None  # nosec B101
    assert result.payload.search.lower.value == 0  # nosec B101
    portable(result)


def test_invalid_dual_cannot_inflate_the_lower_bound_or_trust_reported_radius(
    monkeypatch,
):
    def forged(objective, bounds, tolerance, **kwargs):
        point = (
            [0.75, 0.75, 0] if not kwargs.get("equalities") else [0.0] * len(objective)
        )
        return point, SolveEvidence("optimal", "Forged optimum", "test", tolerance)

    monkeypatch.setattr(linear, "solve_lp", forged)
    result = claim_for(requirements=("steam_header",)).boundary_result()
    assert (
        result.payload.search.lower.value == 0
        and result.payload.search.upper.value == 0.75
    )  # nosec B101
    assert result.payload.search.resolution == "bounded"  # nosec B101


def test_physical_target_is_checked_exactly_despite_coefficient_scaling(monkeypatch):
    claim = scalar_claim(0.5)
    model = replace(
        claim.adapter,
        requirements=(replace(claim.adapter.requirements[0], residual_scale=1e15),),
    )

    def forged(objective, bounds, tolerance, **kwargs):
        point = [0.0, 0.0] if not kwargs.get("equalities") else [0.0] * len(objective)
        return point, SolveEvidence(
            "optimal", "Numerical residual tiny", "test", tolerance
        )

    monkeypatch.setattr(linear, "solve_lp", forged)
    result = claim_for(model, claim.domain).breaking_result(
        violation_margins={"limit": 0.1}
    )
    assert result.payload.search.candidate_evaluation_ref is None  # nosec B101
    assert (
        result.payload.witness is None
        and result.payload.search.resolution == "unresolved"
    )  # nosec B101


def test_exact_polytope_check_rejects_float_membership_false_positive(monkeypatch):
    claim = scalar_claim(0.5)
    domain = PolytopeSet(claim.domain, (LinearConstraint("cap", {"x": 1}, 0.25),))
    # Even if membership incorrectly says inside, the search checks original rows.
    original_membership = BoxSet.membership
    monkeypatch.setattr(
        PolytopeSet,
        "membership",
        lambda self, point: replace(
            original_membership(self.envelope, point), domain_ref=self.ref
        ),
    )

    def forged(objective, bounds, tolerance, **kwargs):
        point = [0.75, 0.75] if not kwargs.get("equalities") else [0.0] * len(objective)
        return point, SolveEvidence(
            "optimal", "Outside actual domain", "test", tolerance
        )

    monkeypatch.setattr(linear, "solve_lp", forged)
    result = claim_for(claim.adapter, domain).boundary_result()
    assert (
        result.payload.witness is None
        and result.payload.search.candidate_evaluation_ref is None
    )  # nosec B101


def test_one_unresolved_branch_prevents_a_false_global_minimum(monkeypatch):
    import operange._affine_distance_results as distance_results

    original = distance_results.compile_problem

    def unresolved(claim, requirement, threshold):
        if requirement.name == "boiler_fuel":
            raise ValueError("Injected compilation failure")
        return original(claim, requirement, threshold)

    monkeypatch.setattr(distance_results, "compile_problem", unresolved)
    result = claim_for().boundary_result()
    assert (
        result.payload.search.upper.value == 0.75
        and result.payload.search.lower.value == 0
    )  # nosec B101
    assert result.payload.search.resolution == "bounded"  # nosec B101


def test_exported_lower_and_infeasibility_certificates_recompute_exactly():
    model, box = example()
    domain = PolytopeSet(box, (LinearConstraint("cap", {"dryer": 1}, 0.5),))
    claim = claim_for(model, domain, requirements=("steam_header",))
    for result in (
        claim.boundary_result(),
        claim.breaking_result(violation_margins={"steam_header": 0.01}),
    ):
        details = next(
            e.details for e in result.evidence if e.evidence_id == "branch:0"
        )
        rows = [[Fraction(a) for a in row] for row in details["rows_exact"]]
        rhs = list(map(Fraction, details["rhs_exact"]))
        bounds = [(Fraction(lo), Fraction(hi)) for lo, hi in details["bounds_exact"]]
        infeasible = details["unreachable"]
        proof = (
            details["infeasibility_certificate"]
            if infeasible
            else details["lower_certificate"]
        )
        multipliers = list(map(Fraction, proof["multipliers"]))
        objective = (
            [Fraction(0)] * len(bounds)
            if infeasible
            else list(map(Fraction, details["objective_exact"]))
        )
        residual = [
            c + sum(v * a for v, a in zip(multipliers, col))
            for c, col in zip(objective, zip(*rows))
        ]
        bound = sum(min(v * lo, v * hi) for v, (lo, hi) in zip(residual, bounds)) - sum(
            v * b for v, b in zip(multipliers, rhs)
        )
        assert bound == Fraction(proof["bound_exact"])  # nosec B101
        if infeasible:
            assert bound > 0  # nosec B101


@pytest.mark.parametrize(
    "margins",
    [
        None,
        {},
        {"limit": 0},
        {"limit": -1},
        {"limit": 1e-8},
        {"limit": True},
        {"limit": float("inf")},
        {"other": 0.1},
    ],
)
def test_invalid_or_nonbreaking_thresholds_are_rejected(margins):
    result = scalar_claim(0.5, tolerance=1e-8).breaking_result(
        violation_margins=margins
    )
    assert result.execution == "invalid" and result.payload.verdict == "not_assessed"  # nosec B101
    portable(result)


@pytest.mark.parametrize("tolerance", [-1, True, float("nan")])
def test_invalid_distance_tolerances(tolerance):
    assert (
        scalar_claim(0.5).boundary_result(distance_tolerance=tolerance).execution
        == "invalid"
    )  # nosec B101


def test_unsupported_domains_missing_distance_and_adjustable_controls():
    model, box = example()
    assert model.as_claim(box).boundary_result().execution == "unsupported"  # nosec B101
    for domain in (
        BudgetSet(box, 1),
        FiniteSet(box.space, (Scenario("nominal", box.nominal, "Test"),)),
    ):
        assert claim_for(model, domain).boundary_result().execution == "unsupported"  # nosec B101
    from examples.linear_dispatch import example as adjustable_example

    adjustable, scenarios = adjustable_example()
    assert adjustable.as_claim(scenarios).breaking_result().execution == "unsupported"  # nosec B101


def test_positive_target_at_box_edge_and_one_float_beyond_it():
    claim = scalar_claim(0.5)
    verify_minimum(claim.breaking_result(violation_margins={"limit": 0.5}), 1)
    unreachable = claim.breaking_result(
        violation_margins={"limit": nextafter(0.5, inf)}
    )
    assert unreachable.payload.search.resolution == "unreachable"  # nosec B101
    portable(unreachable)


def test_unexportable_distance_retains_an_unresolved_portable_result():
    claim = scalar_claim(0.5)
    metric = NormalizedLInf(
        ParameterSpace((Coordinate("x", "MW", "Tiny scale", -2, 5e-324),))
    )
    result = replace(claim, distance=metric).boundary_result()
    assert (
        result.execution == "unresolved"
        and result.payload.search.resolution == "unresolved"
    )  # nosec B101
    assert result.payload.witness is None  # nosec B101
    portable(result)


def test_zero_width_box_has_zero_or_unreachable_distance():
    claim = scalar_claim(0)
    box = BoxSet((Parameter("x", "MW", 0, 0, 0, 1, "Fixed input"),))
    fixed = claim_for(claim.adapter, box)
    verify_minimum(fixed.boundary_result(), 0)
    assert (
        fixed.breaking_result(
            violation_margins={"limit": 0.1}
        ).payload.search.resolution
        == "unreachable"
    )  # nosec B101
