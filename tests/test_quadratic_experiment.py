"""Analytical extrema, normalized physics and backend failure obligations."""

from dataclasses import replace
from math import sqrt

import numpy as np
import pytest

from operange import (
    BoxSet,
    Coordinate,
    EllipsoidSet,
    Intersection,
    Parameter,
    ParameterSpace,
    result_from_json,
)
from operange.experimental import quadratic
from operange.experimental._quadratic_bounds import PreparedTrustRegion
from operange.experimental._residopt_quadratic import (
    BackendUnavailable,
    PreparedResidualSDP,
    load_backend,
)
from operange.experimental.quadratic import (
    QuadraticModel,
    QuadraticProcessAdapter,
    ball_problem,
)


CASES = (
    ([[2, 0], [0, 2]], [0, 0], 1),
    ([[-2, 0], [0, -2]], [0.5, 0], 0.0625),
    ([[4, 0], [0, 1]], [0, 1], 13 / 6),
    ([[0, 0], [0, 0]], [1, 2], sqrt(5)),
    ([[0, 0], [0, 0]], [0, 0], 0),
    ([[0, 0], [0, -2]], [0, 0.5], 0.0625),
)


def example(limit=4):
    space = BoxSet(
        (
            Parameter("flow", "kg/s", 10, 8, 12, 2, "Synthetic"),
            Parameter("temperature", "degC", 30, 25, 35, 5, "Synthetic"),
        )
    ).space
    domain = EllipsoidSet(
        space,
        {
            "flow": {"flow": 1, "temperature": 0},
            "temperature": {"flow": 0, "temperature": 1},
        },
    )
    model = QuadraticModel(
        "Quadratic test",
        space,
        "kPa",
        "pressure",
        2,
        {"flow": 0, "temperature": 0},
        {
            "flow": {"flow": 2, "temperature": 0},
            "temperature": {"flow": 0, "temperature": 2},
        },
        limit,
        "Synthetic quadratic response",
    )
    return model, domain


@pytest.mark.parametrize("H,b,maximum", CASES)
def test_independent_extrema_include_hard_and_interior_cases(H, b, maximum):
    result = PreparedTrustRegion(H).maximize(b)
    assert result.lower == pytest.approx(maximum, abs=1e-10)  # nosec B101
    assert result.upper >= maximum - 1e-12  # nosec B101
    assert result.upper - result.lower <= 1e-7  # nosec B101
    assert np.linalg.norm(result.point) <= 1  # nosec B101


@pytest.mark.parametrize("H,b,maximum", CASES)
def test_current_residopt_matches_independent_analytical_extrema(H, b, maximum):
    try:
        load_backend()
    except BackendUnavailable as exc:
        pytest.skip(str(exc))
    prepared = PreparedResidualSDP(H)
    upper, details = prepared.upper_bound(b)
    assert upper >= maximum - 1e-10  # nosec B101
    assert upper - maximum < 1e-6  # nosec B101
    assert details["report"]["status"] == "verified"  # nosec B101
    assert details["certificates"][0]["certified_model"] == "original"  # nosec B101
    assert len(details["backend"]["source_sha256"]) == 64  # nosec B101


def test_prepared_sdp_reuses_coefficients_without_stale_certificates():
    try:
        load_backend()
    except BackendUnavailable as exc:
        pytest.skip(str(exc))
    prepared = PreparedResidualSDP([[2, 0], [0, 2]])
    first, _ = prepared.upper_bound([0, 0])
    second, _ = prepared.upper_bound([1, 0])
    again, _ = prepared.upper_bound([0, 0])
    assert first == pytest.approx(1, abs=1e-6)  # nosec B101
    assert second == pytest.approx(2, abs=1e-6)  # nosec B101
    assert again == pytest.approx(1, abs=1e-6)  # nosec B101


@pytest.mark.parametrize("failure", ["unverified", "deferred", "solver_error"])
def test_sdp_rejects_unverified_or_incomplete_solutions(monkeypatch, failure):
    try:
        load_backend()
    except BackendUnavailable as exc:
        pytest.skip(str(exc))
    prepared = PreparedResidualSDP([[2, 0], [0, 2]])
    prepared.upper_bound([0, 0])

    def injected_solve(**kwargs):
        if failure == "solver_error":
            raise prepared.cp.error.SolverError("injected numerical failure")
        if failure == "unverified":
            prepared.compiled.solve_report = replace(
                prepared.compiled.solve_report, status="master_only"
            )
        else:
            prepared.compiled.oracle_atoms.append(prepared.compiled.atoms[0])

    monkeypatch.setattr(prepared.compiled, "solve", injected_solve)
    with pytest.raises(RuntimeError):
        prepared.upper_bound([0, 0])


def test_sdp_rechecks_lmi_for_requested_coefficients(monkeypatch):
    try:
        load_backend()
    except BackendUnavailable as exc:
        pytest.skip(str(exc))
    prepared = PreparedResidualSDP([[2, 0], [0, 2]])
    prepared.upper_bound([0, 0])
    # A stale solution/report cannot make a new, larger linear term safe.
    monkeypatch.setattr(prepared.compiled, "solve", lambda **kwargs: None)
    upper, details = prepared.upper_bound([1, 0])
    assert upper >= 2  # nosec B101
    assert details["parameter_residual"] == pytest.approx(1)  # nosec B101
    assert details["lmi_correction"] > 0  # nosec B101


def test_root_iteration_limit_keeps_bounds_instead_of_claiming_an_optimum():
    result = PreparedTrustRegion([[6, 1], [1, -2]]).maximize([0.7, 3], max_iterations=1)
    assert result.details["resolution"] == "bounded"  # nosec B101
    assert result.lower <= result.upper  # nosec B101
    assert np.linalg.norm(result.point) <= 1  # nosec B101
    for angle in np.linspace(0, 2 * np.pi, 100):
        point = np.array([np.cos(angle), np.sin(angle)])
        value = (
            0.5 * point @ np.array([[6, 1], [1, -2]]) @ point
            + np.array([0.7, 3]) @ point
        )
        assert value <= result.upper + 1e-10  # nosec B101


def test_physical_normalization_and_named_correlation_mapping():
    model, domain = example()
    model = replace(
        model,
        linear={"flow": 2, "temperature": -0.5},
        hessian={
            "flow": {"flow": 3, "temperature": 0.4},
            "temperature": {"flow": 0.4, "temperature": -1},
        },
    )
    space = ParameterSpace(
        (
            Coordinate("temperature", "degC", "Shifted domain", 31, 2),
            Coordinate("flow", "kg/s", "Shifted domain", 9, 1),
        )
    )
    domain = EllipsoidSet(
        space,
        {
            "temperature": {"temperature": 1, "flow": 0.3},
            "flow": {"temperature": 0.3, "flow": 1},
        },
        radius=1.5,
    )
    A, b, constant, physical = ball_problem(model, domain)
    result = PreparedTrustRegion(A).maximize(b)
    for angle in np.linspace(0, 2 * np.pi, 80):
        point = np.array([np.cos(angle), np.sin(angle)]) * 0.99
        value = (model.evaluate(physical(point)) - model.limit) / model.residual_scale
        normalized = constant + 0.5 * point @ A @ point + b @ point
        assert value == pytest.approx(normalized)  # nosec B101
        assert value <= constant + result.upper + 1e-10  # nosec B101
        assert domain.membership(physical(point)).status == "inside"  # nosec B101


@pytest.mark.parametrize("limit,verdict", [(4, "pass"), (3, "pass"), (2.5, "fail")])
def test_fixed_quadratic_process_contract(limit, verdict, monkeypatch):
    model, domain = example(limit)
    claim = QuadraticProcessAdapter(model, backend="trust_region").as_claim(domain)
    result = claim.audit_result()
    assert result.payload.verdict == verdict  # nosec B101
    assert result.contract.model_id == "quadratic_process_experiment/v1"  # nosec B101
    if verdict == "fail":
        witness = result.payload.witness
        assert witness.kind == "fixed_policy_failure"  # nosec B101
        point = witness.realizations[0]
        assert domain.membership(point).status == "inside"  # nosec B101
        assert model.evaluate(point) > limit + model.tolerance  # nosec B101
    document = result.to_json()
    monkeypatch.setattr(
        QuadraticModel,
        "evaluate",
        lambda *a: pytest.fail("loading must not execute the response"),
    )
    assert result_from_json(document) == result  # nosec B101


def test_sdp_process_result_bundles_current_backend_evidence():
    try:
        load_backend()
    except BackendUnavailable as exc:
        pytest.skip(str(exc))
    model, domain = example()
    result = QuadraticProcessAdapter(model).as_claim(domain).audit_result()
    assert result.payload.verdict == "pass"  # nosec B101
    proof = next(e for e in result.evidence if e.subject == "requirement_bound")
    assert proof.details["residopt"]["report"]["status"] == "verified"  # nosec B101
    assert proof.details["independent_reference"]["resolution"] == "closed"  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101


def test_missing_optional_backend_returns_unsupported(monkeypatch):
    def unavailable():
        raise BackendUnavailable("injected missing optional backend")

    monkeypatch.setattr(quadratic, "load_backend", unavailable)
    model, domain = example()
    result = QuadraticProcessAdapter(model).as_claim(domain).audit_result()
    assert (
        result.execution == "unsupported" and result.payload.verdict == "not_assessed"
    )  # nosec B101
    assert not result.supporting_evaluations  # nosec B101
    assert (
        QuadraticProcessAdapter(model, "trust_region")
        .as_claim(domain)
        .audit_result()
        .payload.verdict
        == "pass"
    )  # nosec B101


@pytest.mark.parametrize(
    "upper,limit,verdict",
    [
        (100, 4, "inconclusive"),
        (-100, 4, "inconclusive"),
        (None, 4, "inconclusive"),
        (None, 2.5, "fail"),
    ],
)
def test_backend_failures_and_inconsistent_or_loose_bounds(
    monkeypatch, upper, limit, verdict
):
    class FakeSDP:
        def __init__(self, *a, **k):
            pass

        def upper_bound(self, *a, **k):
            if upper is None:
                raise RuntimeError("injected inaccurate/failed solve")
            return upper, {"test": "fake bound"}

    monkeypatch.setattr(quadratic, "load_backend", lambda: (None, None))
    monkeypatch.setattr(quadratic, "PreparedResidualSDP", FakeSDP)
    model, domain = example(limit)
    result = QuadraticProcessAdapter(model).as_claim(domain).audit_result()
    assert result.payload.verdict == verdict  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101


def test_domain_and_operation_scope_and_unknown_membership():
    model, domain = example()
    adapter = QuadraticProcessAdapter(model, "trust_region")
    for changed in (replace(domain, radius=0), Intersection((domain, domain))):
        result = adapter.as_claim(changed).audit_result()
        assert result.execution == "unsupported"  # nosec B101
    claim = adapter.as_claim(domain)
    assert claim.sensitivity_result(domain.space.nominal).execution == "unsupported"  # nosec B101
    assert claim.breaking_result().execution == "unsupported"  # nosec B101
    assert claim.audit_result(unknown=True).execution == "invalid"  # nosec B101
    assert (
        claim.evaluate_result({"flow": 100, "temperature": 30}).execution == "invalid"
    )  # nosec B101
    unknown = claim.evaluate_result({"flow": 12 + 1e-12, "temperature": 30})
    assert unknown.execution == "unresolved"  # nosec B101
    assert not unknown.payload.values  # nosec B101


def test_bad_coefficient_domains_and_numerical_overflow():
    model, domain = example()
    for changes in (
        {"linear": {"flow": 1}},
        {"linear": {"flow": True, "temperature": 0}},
        {
            "hessian": {
                "flow": {"flow": 1, "temperature": 2},
                "temperature": {"flow": 0, "temperature": 1},
            }
        },
        {"residual_scale": 0},
    ):
        with pytest.raises(ValueError):
            replace(model, **changes)
    huge = replace(model, linear={"flow": 1e308, "temperature": 0})
    result = (
        QuadraticProcessAdapter(huge, "trust_region")
        .as_claim(domain)
        .evaluate_result({"flow": 11, "temperature": 30})
    )
    assert result.execution == "completed"  # nosec B101
    result = (
        QuadraticProcessAdapter(huge, "trust_region")
        .as_claim(replace(domain, radius=2))
        .evaluate_result({"flow": 14, "temperature": 30})
    )
    assert result.execution == "unresolved"  # nosec B101
