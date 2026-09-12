"""Continuous recourse proofs, permission boundaries and incomplete coverage."""

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from operange import (
    AffineOutput,
    AffineTerm,
    AffineRequirement,
    Axis,
    BoxSet,
    ConvexHullSet,
    DecisionRule,
    EllipsoidSet,
    LinearObjective,
    ControlTarget,
    ControlTrackingObjective,
    Parameter,
    PolytopeSet,
    RecoursePolicy,
    LinearControl,
    LinearProcessAdapter,
    VectorParameter,
)
from operange import linear, _cvxpy_backend as backend
from examples.linear_dispatch import example
from tests.test_linear_process import small_model, cases, portable


def box(low=0, high=1):
    return BoxSet((Parameter("load", "MW", (low + high) / 2, low, high, 1, "Test"),))


def test_box_audit_covers_continuum_and_all_endpoints_in_physical_coordinates():
    model = small_model(bounds=(0, 2))
    domain = box()
    result = model.as_claim(domain).audit_result()
    assert result.execution == "completed" and result.payload.verdict == "pass"  # nosec B101
    assert result.payload.coverage.method == "analytical_domain"  # nosec B101
    assert result.contract.domain == model.as_claim(domain).contract.domain  # nosec B101
    evaluations = result.supporting_evaluations
    assert {e.request["realization"]["load"] for e in evaluations} == {0, 1}  # nosec B101
    proof = next(e for e in result.evidence if e.evidence_id == "coverage")
    assert proof.method == "complete_vertex_linear_recourse"  # nosec B101
    assert proof.details["generator_count"] == 2  # nosec B101
    # Independently interpolate the actual exported commands at an interior load.
    weight = Fraction(1, 3)
    responses = {
        e.request["realization"]["load"]: {
            v.quantity_ref: Fraction(v.value) for v in e.payload.values
        }
        for e in evaluations
    }
    power = (1 - weight) * responses[0]["power"] + weight * responses[1]["power"]
    assert 0 <= power <= 2 and power >= weight  # nosec B101
    portable(result)


def test_independent_adjustment_can_pass_when_no_common_fixed_control_exists():
    model = small_model(
        limits=(AffineRequirement("no_excess", "margin", 0, tolerance=0),)
    )
    adjusted = model.as_claim(box()).audit_result()
    fixed = RecoursePolicy(
        "fixed", (DecisionRule("power", "MW", "operation", fixed_value=0.5),)
    )
    failed = model.as_claim(box(), recourse=fixed).audit_result()
    assert adjusted.payload.verdict == "pass"  # nosec B101
    assert failed.payload.verdict == "fail"  # nosec B101
    assert failed.payload.witness.kind == "fixed_policy_failure"  # nosec B101
    portable(failed)


def test_hull_of_permitted_load_modes_passes_while_unrestricted_box_fails():
    model, finite = example()
    domain = ConvexHullSet(
        model.input_space, tuple(s for s in finite.scenarios if s.name != "combined")
    )
    envelope = BoxSet(
        (
            Parameter("dryer", "MW", 10, 10, 12, 2, "Test"),
            Parameter("evaporator", "MW", 6, 6, 8, 2, "Test"),
        )
    )
    passing = model.as_claim(domain).audit_result()
    claim = model.as_claim(envelope)
    failed = claim.audit_result()
    assert passing.payload.verdict == "pass" and failed.payload.verdict == "fail"  # nosec B101
    assert failed.payload.witness.realizations == ({"dryer": 12, "evaporator": 8},)  # nosec B101
    assert (
        claim.evaluate_result(
            failed.payload.witness.realizations[0]
        ).payload.feasibility
        == "infeasible"
    )  # nosec B101
    assert (
        model.as_claim(domain)
        .evaluate_result({"dryer": 11, "evaporator": 7})
        .payload.feasibility
        == "feasible"
    )  # nosec B101
    for result in (passing, failed):
        portable(result)


def test_hull_duplicates_keep_generator_labels_and_finite_coverage_is_unchanged():
    model = small_model()
    finite = cases(model, 0, 1, 1)
    result = model.as_claim(
        ConvexHullSet(model.input_space, finite.scenarios)
    ).audit_result()
    assert len(result.payload.coverage.evaluated_support["scenarios"]) == 3  # nosec B101
    assert len(result.supporting_evaluations) == 2  # nosec B101
    assert result.payload.coverage.method == "analytical_domain"  # nosec B101
    assert (
        model.as_claim(finite).audit_result().payload.coverage.method
        == "complete_finite"
    )  # nosec B101
    assert model.as_claim(finite).audit_result(max_vertices=1).execution == "invalid"  # nosec B101
    portable(result)


def test_limit_is_checked_before_solving_and_does_not_infer_sampled_coverage(
    monkeypatch,
):
    model = small_model()
    monkeypatch.setattr(
        linear, "solve_lp", lambda *a, **k: pytest.fail("No solve allowed")
    )
    result = model.as_claim(box()).audit_result(max_vertices=1)
    assert result.execution == "unsupported"  # nosec B101
    assert (
        result.payload.coverage.method == "none" and not result.supporting_evaluations
    )  # nosec B101
    assert result.diagnostics[0].code == "vertex_limit_exceeded"  # nosec B101
    portable(result)


@pytest.mark.parametrize("limit", [0, -1, True, 2.0, "2", None])
def test_invalid_vertex_limit(limit):
    result = small_model().as_claim(box()).audit_result(max_vertices=limit)
    assert result.execution == "invalid"  # nosec B101
    assert result.diagnostics[0].code == "invalid_vertex_limit"  # nosec B101
    portable(result)


def test_fixed_input_coordinates_do_not_multiply_vertices():
    result = small_model().as_claim(box(0.5, 0.5)).audit_result(max_vertices=1)
    assert result.payload.verdict == "pass" and len(result.supporting_evaluations) == 1  # nosec B101
    constant = small_model(bounds=(0.5, 0.5))
    assert constant.as_claim(box(0, 0.5)).audit_result().payload.verdict == "pass"  # nosec B101


def test_verified_failure_survives_an_unresolved_vertex(monkeypatch):
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=False, status=2, message="Injected failure", x=None
        ),
    )
    result = small_model().as_claim(box(0.5, 2)).audit_result()
    assert result.execution == "unresolved" and result.payload.verdict == "fail"  # nosec B101
    assert result.payload.coverage.method == "partial"  # nosec B101
    assert result.payload.witness.realizations == ({"load": 2},)  # nosec B101
    assert not any(e.evidence_id == "coverage" for e in result.evidence)  # nosec B101
    portable(result)
    unknown = small_model().as_claim(box(0.25, 0.75)).audit_result()
    assert unknown.payload.verdict == "inconclusive" and unknown.payload.witness is None  # nosec B101
    portable(unknown)


def test_invalid_candidate_never_establishes_continuous_coverage(monkeypatch):
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=True, status=0, message="Injected approximate success", x=[0, 0]
        ),
    )
    result = small_model(scale=1e12).as_claim(box(0.25, 0.75)).audit_result()
    assert result.payload.verdict == "inconclusive"  # nosec B101


def test_permissions_and_unsupported_geometries_stay_explicit():
    model = small_model()
    for mode, stage, observes in (
        ("static", "operation", ()),
        ("causal", "operation", ("load",)),
        ("static", "preparation", ("load",)),
    ):
        policy = RecoursePolicy(mode, (DecisionRule("power", "MW", stage, observes),))
        result = model.as_claim(box(), recourse=policy).audit_result()
        assert result.execution == "unsupported"  # nosec B101
    for domain in (
        PolytopeSet(box(), ()),
        EllipsoidSet(box().space, {"load": {"load": 1}}),
    ):
        assert model.as_claim(domain).audit_result().execution == "unsupported"  # nosec B101


def test_selected_requirements_keep_operating_limits_and_tolerances():
    model = small_model(tolerance=0.25, bounds=(0, 0))
    model = replace(
        model,
        requirements=model.requirements
        + (AffineRequirement("reserve", "margin", 1, "ge", 0),),
    )
    assert (
        model.as_claim(box(0, 0.25), requirements=("service",))
        .audit_result()
        .payload.verdict
        == "pass"
    )  # nosec B101

    assert model.as_claim(box(0, 0.25)).audit_result().payload.verdict == "fail"  # nosec B101
    model = replace(
        model,
        operating_limits=(AffineRequirement("equipment", "margin", -0.5, tolerance=0),),
    )
    assert (
        model.as_claim(box(0, 0.25), requirements=("service",))
        .audit_result()
        .payload.verdict
        == "fail"
    )  # nosec B101


def test_vector_box_counts_varying_components_and_preserves_labels():
    vector = VectorParameter(
        "loads",
        "MW",
        Axis("users", ("b", "a", "constant")),
        {"a": 0, "b": 0, "constant": 1},
        {"a": 0, "b": 0, "constant": 1},
        {"a": 1, "b": 1, "constant": 1},
        1,
        "Vector loads",
    )
    domain = BoxSet((vector,))
    model = LinearProcessAdapter(
        "Vector dispatch",
        domain.space,
        (
            AffineOutput(
                "margin",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("power", 1, "MW/MW"),)
                + tuple(AffineTerm(n, -1, "MW/MW") for n in domain.space.names),
                "Balance",
            ),
        ),
        (AffineRequirement("service", "margin", 0, "ge", 0),),
        (LinearControl("power", "MW", 0, 4),),
    )
    assert (
        model.as_claim(domain).audit_result(max_vertices=3).execution == "unsupported"
    )  # nosec B101
    result = model.as_claim(domain).audit_result(max_vertices=4)
    assert result.payload.verdict == "pass" and len(result.supporting_evaluations) == 4  # nosec B101
    assert all(
        e.request["realization"][vector.name_for("constant")] == 1
        for e in result.supporting_evaluations
    )  # nosec B101
    portable(result)


def test_hull_default_budget_counts_declared_generators_before_solving(monkeypatch):
    model = small_model()
    domain = ConvexHullSet(model.input_space, cases(model, *([0] * 257)).scenarios)
    monkeypatch.setattr(
        linear, "solve_lp", lambda *a, **k: pytest.fail("No solve allowed")
    )
    result = model.as_claim(domain).audit_result()
    assert result.diagnostics[0].code == "vertex_limit_exceeded"  # nosec B101
    assert not result.supporting_evaluations  # nosec B101


def test_continuous_mixed_fixed_and_adjustable_permissions():
    model, finite = example()
    domain = ConvexHullSet(
        model.input_space, tuple(s for s in finite.scenarios if s.name != "combined")
    )
    policy = RecoursePolicy(
        "static",
        (
            DecisionRule("boiler_a", "MW", "operation", fixed_value=12),
            DecisionRule("boiler_b", "MW", "operation", model.input_space.names),
        ),
    )
    result = model.as_claim(domain, recourse=policy).audit_result()
    assert result.payload.verdict == "pass"  # nosec B101
    assert all(
        next(v.value for v in e.payload.values if v.quantity_ref == "boiler_a") == 12
        for e in result.supporting_evaluations
    )  # nosec B101
    portable(result)


@pytest.mark.parametrize("selected_backend", ["scipy", "cvxpy"])
def test_continuous_consumer_example(selected_backend):
    if selected_backend == "cvxpy":
        pytest.importorskip("cvxpy")
    import json
    from operange import result_from_json
    from examples.continuous_dispatch import run_example

    results = {
        name: result_from_json(json.dumps(data))
        for name, data in run_example(backend=selected_backend).items()
    }
    assert results["unrestricted_box"].payload.verdict == "fail"  # nosec B101
    assert results["scheduled_hull"].payload.verdict == "pass"  # nosec B101
    assert results["enlarged_fuel_supply"].payload.verdict == "pass"  # nosec B101
    assert results["interior_dispatch"].payload.feasibility == "feasible"  # nosec B101
    assert results["failure_replay"].payload.feasibility == "infeasible"  # nosec B101
    for result in results.values():
        portable(result)


def test_linear_objective_stays_pointwise():
    model = replace(small_model(bounds=(0, 2)), objective=LinearObjective("margin"))
    result = model.as_claim(box()).audit_result()
    assert result.payload.verdict == "pass"  # nosec B101
    assert all(e.payload.objective is not None for e in result.supporting_evaluations)  # nosec B101
    assert (
        "individual generator"
        in result.payload.coverage.evaluated_support["objective_scope"]
    )  # nosec B101
    portable(result)


def test_optional_backend_reuses_prepared_program_for_vertices(monkeypatch):
    pytest.importorskip("cvxpy")
    backend.clear_cache()
    builds = []
    original = backend.PreparedLinearProgram.__init__

    def record(self, *args, **kwargs):
        builds.append(True)
        original(self, *args, **kwargs)

    monkeypatch.setattr(backend.PreparedLinearProgram, "__init__", record)
    claim = small_model().as_claim(box(0.25, 0.75))
    result = claim.audit_result(backend="cvxpy")
    assert result.payload.verdict == "pass" and builds == [True]  # nosec B101
    portable(result)
    backend.clear_cache()


def test_quadratic_objective_preserves_vertex_feasibility_scope():
    pytest.importorskip("cvxpy")
    model = replace(
        small_model(bounds=(0, 2)),
        objective=ControlTrackingObjective((ControlTarget("power", 0.5, 1, "MW"),)),
    )
    claim = model.as_claim(box())
    assert claim.audit_result().execution == "unsupported"  # nosec B101
    result = claim.audit_result(backend="cvxpy")
    assert result.payload.verdict == "pass"  # nosec B101
    assert all(e.payload.objective is not None for e in result.supporting_evaluations)  # nosec B101
    portable(result)
