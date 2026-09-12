"""Convex intersections use checked witnesses and support decomposition bounds."""

from dataclasses import replace
from fractions import Fraction as F
import json
from math import sqrt

import pytest

from operange import (
    AffineOutput,
    AffineProcessAdapter,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    BudgetSet,
    EllipsoidSet,
    FrozenController,
    Intersection,
    LinearConstraint,
    ParameterSpace,
    PolytopeSet,
    Product,
    SimplexSet,
    Union,
    domain_from_manifest,
)
from operange import _cvxpy_backend as backend
from operange import _intersection_support as support_backend
from examples.frozen_controllers import envelope_example
from tests.test_geometries import box
from tests.test_linear_process import portable


def circle(envelope):
    return EllipsoidSet(
        envelope.space,
        {
            a: {b: float(a == b) for b in envelope.space.names}
            for a in envelope.space.names
        },
    )


def clipped_circle():
    envelope = box()
    clipped = BoxSet(
        (replace(envelope.parameters[0], upper=0.5), envelope.parameters[1])
    )
    return Intersection((clipped, circle(envelope)), backend="cvxpy")


@pytest.fixture
def cvxpy():
    cp = pytest.importorskip("cvxpy")
    backend.clear_cache()
    yield cp
    backend.clear_cache()


def test_opt_in_manifest_default_and_missing_dependency(monkeypatch):
    original = Intersection((box(), box()))
    assert "backend" not in original.to_manifest()  # nosec B101
    assert not original.capabilities.linear_optimization  # nosec B101
    domain = replace(original, backend="cvxpy")
    assert domain_from_manifest(domain.to_manifest()).ref == domain.ref  # nosec B101
    with pytest.raises(ValueError, match="backend"):
        replace(original, backend="scipy")

    def unavailable(*args):
        raise ImportError("Injected missing optional dependency")

    monkeypatch.setattr(backend, "import_module", unavailable)
    monkeypatch.setattr(support_backend, "find_spec", lambda *a: None)
    assert not domain.capabilities.linear_optimization  # nosec B101
    result = domain.maximize_linear({"a": 1})
    assert result.status == "unsupported"  # nosec B101
    assert "operange[cvxpy]" in json.dumps(result.to_dict())  # nosec B101


def test_joint_optimum_is_tighter_than_either_factor_and_proof_recomputes(cvxpy):
    domain = clipped_circle()
    result = domain.maximize_linear({"a": 1, "b": 1})
    optimum = 0.5 + sqrt(0.75)
    assert result.status in ("optimal", "bounded")  # nosec B101
    assert result.lower <= optimum + 1e-15 <= result.upper + 1e-15  # nosec B101
    assert result.upper - result.lower < 1e-7  # nosec B101
    assert all(
        result.upper < factor.maximize_linear(result.coefficients).upper
        for factor in domain.factors
    )  # nosec B101
    assert support_backend.exact_member(domain.factors, result.point)  # nosec B101
    proof = next(
        e.details for e in result.evidence if e.subject == "support_upper_bound"
    )
    total = F(0)
    residual = {n: F(v) for n, v in result.coefficients.items()}
    for record in proof["factors"]:
        local = record["support"]
        total += F(local["upper"])
        for name, alpha, beta in zip(
            record["coordinates"], record["alpha_exact"], record["beta_exact"]
        ):
            weight = F(local["coefficients"][name])
            total -= weight * F(beta)
            residual[name] -= weight * F(alpha)
    correction = sum(
        max(r * F(b) for b in proof["enclosure_exact"][n]) for n, r in residual.items()
    )
    assert total + correction == F(proof["upper_exact"]) <= F(result.upper)  # nosec B101
    restored = domain_from_manifest(json.loads(json.dumps(domain.to_manifest())))
    assert restored.ref == domain.ref  # nosec B101
    assert restored.maximize_linear({"a": 1, "b": 1}).upper == pytest.approx(
        result.upper
    )  # nosec B101


def test_nonzero_nominals_scales_and_coordinate_order_are_aligned(cvxpy):
    left = box(nominal=10, lower=8, upper=12, scale=2, unit="MW")
    right = box(names=("b", "a"), nominal=11, lower=7, upper=15, scale=4, unit="MW")
    coupled = PolytopeSet(right, (LinearConstraint("cap", {"a": 1, "b": 1}, 0),))
    # The local inequality means a+b <= 22 MW, or z_a+z_b <= 1 jointly.
    domain = Intersection((left, coupled), space=left.space, backend="cvxpy")
    result = domain.maximize_linear({"a": 1, "b": 1})
    assert result.lower <= 1 <= result.upper  # nosec B101
    assert result.upper - result.lower < 1e-7  # nosec B101
    reordered = replace(
        domain, space=ParameterSpace(tuple(reversed(left.space.coordinates)))
    )
    assert reordered.maximize_linear({"b": 1, "a": 1}).upper == pytest.approx(1)  # nosec B101


@pytest.mark.parametrize(
    "kind", ("budget", "simplex", "polytope_eq", "product", "zero_radius")
)
def test_supported_convex_factors_and_degenerate_equalities(cvxpy, kind):
    envelope = box(unit="mole_fraction", nominal=0.5, lower=0, upper=1, scale=1)
    if kind == "simplex":
        factor = SimplexSet(envelope.space, "mole")
    elif kind == "budget":
        factor = BudgetSet(envelope, 0.5)
    elif kind == "polytope_eq":
        factor = PolytopeSet(
            envelope, (LinearConstraint("balance", {"a": 1, "b": 1}, 0, "eq"),)
        )
    elif kind == "product":
        factor = Product(tuple(BoxSet((p,)) for p in envelope.parameters))
    else:
        factor = replace(circle(envelope), radius=0)
    domain = Intersection((envelope, factor), backend="cvxpy")
    result = domain.maximize_linear({"a": 1})
    assert domain.capabilities.linear_optimization  # nosec B101
    assert result.lower is not None and result.upper is not None  # nosec B101
    assert abs(result.upper - (0 if kind == "zero_radius" else 0.5)) < 1e-7  # nosec B101
    assert support_backend.exact_member(support_backend.leaves(domain), result.point)  # nosec B101


def test_unknown_or_nonconvex_factors_are_not_convexified(cvxpy):
    class CustomBox(BoxSet):
        pass

    custom = CustomBox(box().parameters)
    for factor in (
        Union((box(), box())),
        custom,
        BudgetSet(custom, 1),
        PolytopeSet(custom, ()),
    ):
        domain = Intersection((box(), factor), backend="cvxpy")
        assert not domain.capabilities.linear_optimization  # nosec B101
        assert domain.maximize_linear({"a": 1}).status == "unsupported"  # nosec B101


@pytest.mark.parametrize(
    "fault", ("solver_error", "infeasible", "invalid_split", "outside_point")
)
def test_solver_failure_keeps_only_independently_valid_bounds(
    cvxpy, monkeypatch, fault
):
    domain = clipped_circle()
    if fault in ("solver_error", "infeasible"):

        def broken(problem, **options):
            if fault == "solver_error":
                raise cvxpy.error.SolverError("Injected failure")
            problem._status = cvxpy.INFEASIBLE

        monkeypatch.setattr(cvxpy.Problem, "solve", broken)
    else:
        monkeypatch.setattr(
            support_backend.PreparedIntersection,
            "solve",
            lambda *a: (
                [10, 10],
                [[float("nan"), 0], [0, 0]]
                if fault == "invalid_split"
                else [[-10, 3], [2, -5]],
                {"status": "optimal"},
            ),
        )
    result = domain.maximize_linear({"a": 1, "b": 1})
    assert result.status == "bounded"  # nosec B101
    assert result.point == {"a": 0, "b": 0}  # nosec B101
    assert result.lower == 0 and result.upper >= 0.5 + sqrt(0.75)  # nosec B101


def test_rounded_simplex_balance_cannot_establish_exact_nonemptiness(cvxpy):
    # All declared units must align; fsum([1/3]*3) rounds to 1 although the
    # sum of the exact declared binary floats is smaller than 1.
    envelope = box(
        names=("a", "b", "c"), nominal=1 / 3, lower=0, upper=1, unit="mole_fraction"
    )
    domain = Intersection(
        (envelope, SimplexSet(envelope.space, "mole")), backend="cvxpy"
    )
    assert domain.membership(domain.feasible_point).status == "inside"  # nosec B101
    assert domain.maximize_linear({"a": 1}).status == "unresolved"  # nosec B101


def test_support_drives_affine_audits_and_frozen_controller_replay(cvxpy):
    domain = clipped_circle()
    model = AffineProcessAdapter(
        "Joint support",
        domain.space,
        (
            AffineOutput(
                "total",
                "fraction",
                "dimensionless",
                0,
                tuple(
                    AffineTerm(n, 1, "fraction/fraction") for n in domain.space.names
                ),
                "Sum",
            ),
        ),
        (AffineRequirement("cap", "total", 1.38),),
    )
    result = model.as_claim(domain).audit_result()
    assert result.payload.verdict == "pass"  # nosec B101
    portable(result)
    for factor in domain.factors:
        assert model.as_claim(factor).audit_result().payload.verdict == "fail"  # nosec B101

    claim = envelope_example()
    joint = Intersection((claim.domain, BudgetSet(claim.domain, 1)), backend="cvxpy")
    claim = replace(claim, domain=joint)
    assert claim.audit_result().payload.verdict == "pass"  # nosec B101
    frozen = claim.freeze()
    loaded = FrozenController.from_json(frozen.to_json())
    replay = loaded.audit_result()
    assert (
        replay.payload.verdict == "pass"
        and replay.contract.domain == claim.contract.domain
    )  # nosec B101
    portable(replay)


def test_worked_example_exports_checked_results(cvxpy):
    from examples.convex_backends import run_example
    from operange import result_from_json

    exported = run_example()
    results = {n: result_from_json(json.dumps(v)) for n, v in exported.items()}
    assert (
        results["box"].payload.verdict == results["ellipsoid"].payload.verdict == "fail"
    )  # nosec B101
    assert results["intersection"].payload.verdict == "pass"  # nosec B101
    assert results["dispatch"].payload.verdict == "fail"  # nosec B101
    assert results["relief"].payload.feasibility == "infeasible"  # nosec B101
    for result in results.values():
        portable(result)
