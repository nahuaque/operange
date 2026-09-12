"""Explicit continuous hulls never silently convexify finite scenarios."""

from dataclasses import replace
from fractions import Fraction
import json
from types import SimpleNamespace

import pytest

from operange import (
    AffineOutput,
    AffineProcessAdapter,
    AffineRequirement,
    AffineTerm,
    ConvexHullSet,
    Coordinate,
    FiniteSet,
    ParameterSpace,
    Scenario,
    domain_from_json,
    domain_from_manifest,
)
from operange import linear
from tests.test_linear_process import portable


def hull(*points, normalized=False):
    space = ParameterSpace(
        tuple(
            Coordinate(
                n, "MW", "Test", 0 if normalized else None, 1 if normalized else None
            )
            for n in ("x", "y")
        )
    )
    return ConvexHullSet(
        space,
        tuple(
            Scenario(str(i), dict(zip(space.names, point)), "Declared generator")
            for i, point in enumerate(points)
        ),
    )


@pytest.mark.parametrize(
    "point,status",
    [
        ((0, 0), "inside"),
        ((1, 1), "inside"),
        ((1, 0), "inside"),
        ((0.25, 0.75), "inside"),
        ((2, 2), "outside"),
        ((-1, 0), "outside"),
    ],
)
def test_exact_membership_and_portable_barycentric_proofs(point, status):
    domain = hull((0, 0), (3, 0), (0, 3))
    values = dict(zip(domain.space.names, point))
    check = domain.membership(values)
    assert check.status == status  # nosec B101
    details = check.evidence[0].details
    if "weights_exact" in details:
        weights = {n: Fraction(w) for n, w in details["weights_exact"].items()}
        assert sum(weights.values()) == 1 and min(weights.values()) >= 0  # nosec B101
        for name in domain.space.names:
            assert sum(
                weights[s.name] * Fraction(s.values[name]) for s in domain.vertices
            ) == Fraction(values[name])  # nosec B101
    assert check.domain_ref == domain.ref  # nosec B101


def test_hull_does_not_gain_a_tolerance_band():
    domain = hull((0, 0), (1, 0), (0, 1))
    point = {"x": 0.5, "y": 0.5 + 1e-12}
    check = domain.membership(point)
    assert check.status in ("outside", "unknown")  # nosec B101
    assert check.status != "inside"  # nosec B101
    separated = domain.membership({"x": 0.75, "y": 0.75})
    assert separated.status == "outside"  # nosec B101
    assert Fraction(separated.evidence[0].details["separation_gap_exact"]) > 0  # nosec B101


def test_hull_and_finite_set_have_distinct_meaning_and_identity():
    domain = hull((0, 0), (2, 0), (0, 2))
    finite = FiniteSet(domain.space, domain.vertices)
    assert domain.contains({"x": 0.5, "y": 0.5})  # nosec B101
    assert not finite.contains({"x": 0.5, "y": 0.5})  # nosec B101
    assert domain.ref != finite.ref and not domain.capabilities.enumeration  # nosec B101
    assert domain_from_json(json.dumps(domain.to_manifest())) == domain  # nosec B101
    assert domain_from_manifest(domain.to_manifest()).ref == domain.ref  # nosec B101
    assert replace(domain, vertices=tuple(reversed(domain.vertices))).ref == domain.ref  # nosec B101
    with pytest.raises(ValueError):
        domain_from_manifest({**domain.to_manifest(), "extra": 1})


@pytest.mark.parametrize(
    "points,inside,outside",
    [
        (((1, 1),), (1, 1), (1, 2)),
        (((0, 0), (2, 2), (1, 1), (2, 2)), (0.5, 0.5), (0.5, 0.75)),
        (((0, 1), (2, 1)), (1, 1), (1, 0)),
        (((1e16, 0), (1e16 + 4, 0)), (1e16 + 2, 0), (1e16 + 2, 1)),
    ],
)
def test_degenerate_redundant_duplicate_and_large_origin_hulls(points, inside, outside):
    domain = hull(*points)
    assert domain.contains(dict(zip(domain.space.names, inside)))  # nosec B101
    assert domain.membership(dict(zip(domain.space.names, outside))).status == "outside"  # nosec B101


def test_numerical_failure_preserves_unknown_membership(monkeypatch):
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=False,
            status=2,
            message="Injected infeasible",
            x=None,
        ),
    )
    domain = hull((0, 0), (1, 0), (0, 1))
    assert domain.membership({"x": 0.25, "y": 0.25}).status == "unknown"  # nosec B101
    with pytest.raises(ValueError, match="unresolved"):
        domain.contains({"x": 0.25, "y": 0.25})
    assert domain.contains({"x": 1, "y": 0})  # nosec B101


def test_bogus_success_cannot_prove_hull_membership(monkeypatch):
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=True,
            status=0,
            message="Injected success",
            x=[0, 0, 0],
        ),
    )
    domain = hull((0, 0), (1, 0), (0, 1))
    assert domain.membership({"x": 0.75, "y": 0.75}).status == "unknown"  # nosec B101


def test_linear_support_and_affine_audit_use_actual_hull_generators():
    domain = hull((0, 0), (2, 0), (0, 2), normalized=True)
    support = domain.maximize_linear({"x": 1, "y": 1})
    assert support.lower == support.upper == 2  # nosec B101
    assert domain.contains(support.point)  # nosec B101
    model = AffineProcessAdapter(
        "Hull sum",
        domain.space,
        (
            AffineOutput(
                "sum",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("x", 1, "MW/MW"), AffineTerm("y", 1, "MW/MW")),
                "Sum",
            ),
        ),
        (AffineRequirement("cap", "sum", 2),),
    )
    audit = model.as_claim(domain).audit_result()
    assert audit.payload.verdict == "pass"  # nosec B101
    portable(audit)
    unsupported = hull((0, 0), (1, 0)).maximize_linear({"x": 1})
    assert unsupported.status == "unsupported"  # nosec B101


def test_hull_declaration_validation():
    domain = hull((0, 0), (1, 1))
    for fields in (
        {"vertices": ()},
        {"vertices": domain.vertices * 2},
        {"provenance": ""},
        {"tolerance": 0},
    ):
        with pytest.raises(ValueError):
            replace(domain, **fields)
    with pytest.raises(ValueError):
        replace(domain, vertices=(Scenario("bad", {"x": 1}, "Missing y"),))
