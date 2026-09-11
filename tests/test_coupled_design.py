"""Independent feasibility and objective evidence for the coupled design example."""

from dataclasses import replace
import json

import numpy as np
import pytest


from examples import coupled_design as design
from operange import result_from_json


def prepared(backend):
    try:
        return design.PreparedDesign(backend)
    except design.BackendUnavailable as exc:
        pytest.skip(str(exc))


def test_nominal_design_breaks_each_requirement_under_declared_uncertainty():
    nominal = design.nominal_design()
    assert not nominal["feasible"]  # nosec B101
    assert {r["verdict"] for r in nominal["requirements"].values()} == {"fail"}  # nosec B101
    for response in design.RESPONSES:
        model = design.fixed_model(response, nominal["normalized_design"])
        assert model.evaluate(model.space.nominal) < 0  # nosec B101


def test_native_design_has_physical_audits_and_replayable_objective_bound():
    reference = design.NativeReference()
    solution = reference.solve(design.WORKLOAD[0])
    result = design.assessment(solution, design.WORKLOAD[0], solution["lower_bound_kg"])
    assert result["status"] == "verified_near_optimal"  # nosec B101
    assert result["objective_gap_kg"] < 0.003  # nosec B101
    assert result["design"]["exchanger_area"]["value"] == pytest.approx(
        18.73646, abs=1e-4
    )  # nosec B101
    evidence = solution["bound_evidence"]
    A = np.array(evidence["scenario_matrix"])
    rhs = np.array(evidence["relaxed_rhs"])
    multipliers = np.array(evidence["multipliers"])
    reduced = design.WORKLOAD[0] + A.T @ multipliers
    independent = (
        np.minimum(
            reduced * evidence["box_lower"], reduced * evidence["box_upper"]
        ).sum()
        - multipliers @ rhs
        - evidence["rounding_guard_kg"]
    )
    assert solution["lower_bound_kg"] == pytest.approx(independent)  # nosec B101
    declared = design.domain()
    for points in evidence["scenario_points"].values():
        for point in points:
            physical = declared.space.denormalize(
                dict(zip(declared.space.names, point))
            )
            assert declared.membership(physical).status == "inside"  # nosec B101
    for bundle in result["engineering_results"].values():
        restored = result_from_json(json.dumps(bundle))
        assert restored.payload.verdict == "pass"  # nosec B101


def test_normalized_design_response_matches_physical_named_model():
    x = np.array([1.5, 2.0, 3.0])
    z = np.array([0.3, -0.2])
    declared = design.domain()
    physical = declared.space.denormalize(dict(zip(declared.space.names, z)))
    for response in design.RESPONSES:
        H, S, s, r = response.arrays()
        residual = 0.5 * z @ H @ z + (S @ x + s) @ z + r @ x + response.offset
        assert design.fixed_model(response, x).evaluate(physical) == pytest.approx(
            response.scale * residual
        )  # nosec B101


def test_dual_bound_covers_toleranced_feasible_set_and_nonoptimal_multipliers():
    # min 2*x0 + x1 + 3*x2, x0 >= 1-tol, box lower = -tol.
    A = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    rhs = np.array([4.8, -1.0 - design.RESERVE])
    weights = np.array([2.0, 1.0, 3.0])
    optimum = 2 - 6 * design.TOLERANCE
    proof = design.NativeReference.lower_bound(weights, A, rhs, [0, -2])
    assert proof["lower_bound_kg"] <= optimum  # nosec B101
    assert optimum - proof["lower_bound_kg"] < 1e-9  # nosec B101
    for marginals in ([0, -0.7], [1, -3], [0, 2]):
        proof = design.NativeReference.lower_bound(weights, A, rhs, marginals)
        assert proof["lower_bound_kg"] <= optimum  # nosec B101
    with pytest.raises(RuntimeError):
        design.NativeReference.lower_bound(weights, A, rhs, [0, float("nan")])


def test_no_optimality_claim_from_a_bad_design_or_invalid_objective_bound():
    solution = design.NativeReference().solve(design.WORKLOAD[0])
    underpowered = dict(solution, design=list(solution["design"]))
    underpowered["design"][2] *= 0.8
    failed = design.assessment(
        underpowered, design.WORKLOAD[0], solution["lower_bound_kg"]
    )
    assert not failed["feasible"] and failed["status"] == "unresolved"  # nosec B101
    assert failed["requirements"]["pressure_drop_excess"]["verdict"] == "fail"  # nosec B101
    false_bound = design.assessment(solution, design.WORKLOAD[0], 1e6)
    assert false_bound["status"] == "unresolved"  # nosec B101
    oversized = design.audit_design([4, 4, 4], design.WORKLOAD[0])
    assert not oversized["base_constraints"]["verified"]  # nosec B101
    assert not oversized["feasible"]  # nosec B101


def test_iteration_limit_does_not_imply_robust_feasibility():
    solution = design.NativeReference().solve(design.WORKLOAD[0], max_rounds=1)
    assert solution["status"] == "iteration_limit"  # nosec B101
    result = design.assessment(solution, design.WORKLOAD[0], solution["lower_bound_kg"])
    assert result["status"] == "unresolved"  # nosec B101


def test_flat_objective_case_closes_with_compatible_lp_and_separation_tolerances():
    solution = design.NativeReference().solve(design.WORKLOAD[3])
    assert solution["status"] == "bounded"  # nosec B101
    assert solution["rounds"] < 20  # nosec B101


@pytest.mark.parametrize("backend", ["direct_sdp", "residopt_sdp", "residopt_hybrid"])
@pytest.mark.parametrize("case", [0, 1, 2, 3])
def test_coupled_backends_close_independent_gap_for_distinct_active_sets(backend, case):
    model = prepared(backend)
    weights = design.WORKLOAD[case]
    reference = design.NativeReference().solve(weights)
    solution = model.solve(weights)
    result = design.assessment(solution, weights, reference["lower_bound_kg"])
    assert result["status"] == "verified_near_optimal"  # nosec B101
    assert result["objective_gap_kg"] < 0.003  # nosec B101
    if backend != "direct_sdp":
        evidence = solution["backend_evidence"]
        assert evidence["solve_report"]["status"] == "verified"  # nosec B101
        assert len(evidence["solve_report"]["checks"]) == 3  # nosec B101
        assert evidence["fully_compiled"] == (backend == "residopt_sdp")  # nosec B101
        if backend == "residopt_hybrid":
            assert evidence["solve_report"]["cuts_added"] > 0  # nosec B101


@pytest.mark.parametrize("backend", ["direct_sdp", "residopt_sdp", "residopt_hybrid"])
def test_reused_design_model_updates_objective_and_can_return_to_first_case(backend):
    model = prepared(backend)
    costs = []
    for index in (0, 2, 0):
        weights = design.WORKLOAD[index]
        reference = design.NativeReference().solve(weights)
        result = design.assessment(
            model.solve(weights), weights, reference["lower_bound_kg"]
        )
        assert result["status"] == "verified_near_optimal"  # nosec B101
        costs.append(result["objective_kg"])
    assert costs[2] == pytest.approx(costs[0], abs=1e-4)  # nosec B101
    assert costs[1] < costs[0] - 100  # nosec B101


def test_unverified_residopt_report_and_hybrid_iteration_limit_are_rejected(
    monkeypatch,
):
    model = prepared("residopt_hybrid")
    with pytest.raises(RuntimeError):
        model.solve(design.WORKLOAD[0], max_rounds=1)
    model.solve(design.WORKLOAD[0])
    bad = replace(
        model.compiled.solve_report, checks=model.compiled.solve_report.checks[:1]
    )
    monkeypatch.setattr(model.compiled, "solve", lambda **kw: None)
    model.compiled.solve_report = bad
    with pytest.raises(RuntimeError, match="not all verified"):
        model.solve(design.WORKLOAD[0])


def test_solver_error_is_unresolved_and_direct_sdp_does_not_need_residopt(monkeypatch):
    def unavailable():
        raise design.BackendUnavailable("injected missing residopt")

    monkeypatch.setattr(design, "load_backend", unavailable)
    model = prepared("direct_sdp")

    def fail(**kwargs):
        raise model.cp.error.SolverError("injected solver failure")

    monkeypatch.setattr(model.problem, "solve", fail)
    with pytest.raises(RuntimeError, match="solver failed"):
        model.solve(design.WORKLOAD[0])
    with pytest.raises(design.BackendUnavailable):
        design.PreparedDesign("residopt_sdp")
    reference = design.NativeReference().solve(design.WORKLOAD[0])
    assert reference["status"] == "bounded"  # nosec B101


def test_invalid_brief_and_nonfinite_design_are_rejected():
    for weights in ((1, 2), (1, 2, float("inf")), (1, -2, 3)):
        with pytest.raises(ValueError):
            design.NativeReference().solve(weights)
    with pytest.raises(ValueError):
        design.audit_design([1, 2, float("nan")], design.WORKLOAD[0])
