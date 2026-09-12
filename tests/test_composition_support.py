"""Composed support identities retain bounds, coordinates and real witnesses."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import product as corners
import json

import pytest

from operange import (
    BoxSet,
    Coordinate,
    FiniteSet,
    FrozenController,
    Intersection,
    ParameterSpace,
    Product,
    Scenario,
    Union,
    domain_from_manifest,
    result_from_json,
)
from operange import linear
from operange._numeric import normalized_score
from examples.frozen_controllers import envelope_example
from tests.test_geometries import SometimesUnknown, box, vector
from tests.test_linear_process import portable


def proof(result):
    return next(
        e.details for e in result.evidence if e.subject == "support_upper_bound"
    )


@pytest.mark.parametrize("weights", ({"a": 2, "b": -3}, {"b": 1}, {}))
def test_product_support_sums_disjoint_factors_without_a_joint_solver(
    monkeypatch, weights
):
    def forbidden(*args, **kwargs):
        pytest.fail("analytical factors must not need a joint solver")

    monkeypatch.setattr(linear, "linprog", forbidden)
    a = box(("a",), nominal=10, lower=8, upper=14, scale=2, unit="MW")
    b = box(("b",), nominal=3, lower=1, upper=9, scale=4, unit="MW")
    domain = Product((a, b))
    result = domain.maximize_linear(weights)
    expected = max(
        sum(
            F(weights.get(n, 0)) * (F(v) - F(c.nominal)) / F(c.scale)
            for n, v, c in zip(domain.space.names, point, domain.space.coordinates)
        )
        for point in corners((8, 14), (1, 9))
    )
    assert result.status == "optimal"  # nosec B101
    assert F(result.lower) <= expected <= F(result.upper)  # nosec B101
    assert domain.membership(result.point).status == "inside"  # nosec B101
    assert proof(result)["formula"] == "sum(factor_supports)"  # nosec B101
    assert proof(result)["witness_factor_indices"] == (0, 1)  # nosec B101


@pytest.mark.parametrize(
    "direction, expected, branch", ((1, 4, 1), (-1, 3, 0), (0, 0, 0))
)
def test_union_uses_all_branches_and_keeps_disconnected_membership(
    direction, expected, branch
):
    left = box(("a",), nominal=-2, lower=-3, upper=-1, scale=2)
    right = box(("a",), nominal=3, lower=2, upper=4, scale=3)
    domain = Union((left, right), space=box(("a",)).space)
    result = domain.maximize_linear({"a": direction})
    assert result.status == "optimal"  # nosec B101
    assert result.lower == result.upper == expected  # nosec B101
    assert domain.membership({"a": 0}).status == "outside"  # nosec B101
    assert domain.factors[branch].membership(result.point).status == "inside"  # nosec B101
    assert proof(result)["witness_factor_indices"] == (branch,)  # nosec B101
    assert len(proof(result)["factors"]) == 2  # nosec B101


@pytest.mark.parametrize("kind", (Product, Union))
@pytest.mark.parametrize("scale, direction", ((3, 1), (10, 1), (3, -1), (10, -1)))
def test_coordinate_rounding_correction_exports_a_recomputable_bound(
    kind, scale, direction
):
    factor = box(("a",), lower=0)
    joint = ParameterSpace(
        (replace(factor.space.coordinates[0], nominal=0.25, scale=scale),)
    )
    domain = kind((factor,), space=joint)
    result = domain.maximize_linear({"a": direction})
    record = proof(result)["factors"][0]
    exact = F(direction) * (F(1 if direction > 0 else 0) - F(1, 4)) / scale
    assert F(result.lower) <= exact <= F(result.upper)  # nosec B101
    assert F(record["coefficients_exact"]["a"]) == F(direction, scale)  # nosec B101
    assert F(record["shift_exact"]) == -F(direction, 4 * scale)  # nosec B101
    correction = record["coefficient_corrections"][0]
    residual = F(direction, scale) - F(record["support"]["coefficients"]["a"])
    assert residual == F(correction["residual_exact"]) != 0  # nosec B101
    corrected = abs(residual) * F(correction["support"]["upper"])
    assert correction["support"]["coefficients"]["a"] == (1 if residual > 0 else -1)  # nosec B101
    assert corrected == F(record["correction_exact"])  # nosec B101
    upper = F(record["shift_exact"]) + F(record["support"]["upper"]) + corrected
    assert upper == F(proof(result)["upper_exact"]) >= exact  # nosec B101
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("kind", (Product, Union))
def test_coefficient_underflow_is_bounded_instead_of_silently_dropped(kind):
    tiny = float.fromhex("0x0.0000000000001p-1022")
    factor = box(("a",), lower=0, upper=tiny, scale=tiny)
    joint = ParameterSpace((replace(factor.space.coordinates[0], scale=2),))
    result = kind((factor,), space=joint).maximize_linear({"a": 1})
    assert F(result.lower) <= F(tiny) / 2 <= F(result.upper)  # nosec B101
    assert result.upper == tiny  # nosec B101
    assert proof(result)["factors"][0]["support"]["coefficients"]["a"] == 0  # nosec B101
    assert F(proof(result)["upper_exact"]) == F(tiny) / 2  # nosec B101


def test_vector_labels_and_different_coordinate_order_are_preserved():
    first, reversed_order = vector(), vector(("B", "A"))
    # Canonical names identify the weights even when the factor axes are reordered.
    domain = Union((BoxSet((first,)), BoxSet((reversed_order,))), space=first.space)
    weights = {first.name_for("A"): 2, first.name_for("B"): -1}
    result = domain.maximize_linear(weights)
    assert result.point == first.flatten({"A": 1, "B": 0})  # nosec B101
    assert result.lower == result.upper == 1.5  # nosec B101
    restored = domain_from_manifest(json.loads(json.dumps(domain.to_manifest())))
    assert restored.ref == domain.ref  # nosec B101
    assert restored.maximize_linear(weights).to_dict() == result.to_dict()  # nosec B101


def test_nested_products_unions_and_finite_scenarios_support_exact_bounds():
    a, b = box(("a",)), box(("b",))
    finite = FiniteSet(a.space, (Scenario("a high", {"a": 0.75}, "Test case"),))
    negative = box(("a",), nominal=-0.75, lower=-1, upper=-0.5)
    alternatives = Union((negative, finite), space=a.space)
    nested = Union((Product((alternatives, b)), Product((a, b))))
    result = nested.maximize_linear({"a": -2, "b": 3})
    assert result.lower == result.upper == 5  # nosec B101
    assert result.point == {"a": -1, "b": 1}  # nosec B101
    assert nested.capabilities.linear_optimization  # nosec B101
    restored = domain_from_manifest(nested.to_manifest())
    assert restored.maximize_linear(result.coefficients).to_dict() == result.to_dict()  # nosec B101


@pytest.mark.parametrize("kind", (Product, Union))
def test_unsupported_factor_does_not_gain_support_from_composition(kind):
    envelope = box(("a",))
    factors = (
        SometimesUnknown(envelope.space),
        Intersection((envelope, envelope)),
        FiniteSet(
            ParameterSpace((Coordinate("a", "fraction", "No normalization"),)),
            (Scenario("nominal", {"a": 0}, "Test case"),),
        ),
    )
    for factor in factors:
        domain = kind((factor,))
        assert not domain.capabilities.linear_optimization  # nosec B101
        result = domain.maximize_linear({"a": 1})
        assert result.status == "unsupported" and result.upper is None  # nosec B101


@pytest.mark.parametrize("kind", (Product, Union))
@pytest.mark.parametrize(
    "fault", ("domain", "objective", "point", "lower", "upper", "unresolved")
)
def test_invalid_factor_support_cannot_be_promoted(monkeypatch, kind, fault):
    factor = box(("a",))
    support = factor.maximize_linear({"a": 1})
    changes = {
        "domain": {"domain_ref": box(("a",), upper=2).ref},
        "objective": {"coefficients": {"a": -1}},
        "point": {"point": {"a": 2}},
        "lower": {"lower": -1, "status": "bounded"},
        "upper": {"lower": 0, "upper": 0},
    }
    bad = (
        replace(support, **changes[fault])
        if fault != "unresolved"
        else replace(support, status="unresolved", point=None, lower=None, upper=None)
    )
    monkeypatch.setattr(BoxSet, "maximize_linear", lambda *a: bad)
    result = kind((factor,)).maximize_linear({"a": 1})
    assert result.status == "unresolved"  # nosec B101
    assert result.point is None and result.lower is None and result.upper is None  # nosec B101


def test_union_preserves_loose_branch_bound_even_when_another_branch_has_best_witness(
    monkeypatch,
):
    first, second = box(("a",)), box(("a",), upper=2)
    original = BoxSet.maximize_linear

    def loose(self, weights):
        result = original(self, weights)
        return replace(result, upper=10, status="bounded") if self is first else result

    monkeypatch.setattr(BoxSet, "maximize_linear", loose)
    result = Union((first, second)).maximize_linear({"a": 1})
    assert result.status == "bounded"  # nosec B101
    assert result.lower == 2 and result.upper == 10  # nosec B101
    assert proof(result)["witness_factor_indices"] == (1,)  # nosec B101


@pytest.mark.parametrize("kind", (Product, Union))
def test_frozen_controller_audits_compositions_and_replays_failure(kind):
    original = envelope_example()
    envelope = original.domain
    if kind is Product:
        domain = Product(tuple(BoxSet((p,)) for p in envelope.parameters))
    else:
        domain = Union((envelope, envelope))
    claim = replace(original, domain=domain)
    frozen = FrozenController.from_json(claim.freeze().to_json())
    result = frozen.audit_result()
    assert result.payload.verdict == "fail"  # nosec B101
    assert result.contract.domain == claim.contract.domain  # nosec B101
    witness = result.payload.witness.realizations[0]
    assert domain.membership(witness).status == "inside"  # nosec B101
    assert frozen.evaluate_result(witness).payload.feasibility == "infeasible"  # nosec B101
    portable(result)


def test_checked_intersections_can_be_nested_in_native_compositions():
    pytest.importorskip("cvxpy")
    from tests.test_cvxpy_intersections import clipped_circle

    clipped = clipped_circle()
    alternatives = Union((clipped, clipped))
    domain = Product((alternatives, box(("c",))))
    result = domain.maximize_linear({"a": 1, "b": 1, "c": -2})
    expected = F(2) + F(0.5 + (0.75**0.5))
    assert result.lower <= float(expected) <= result.upper  # nosec B101
    assert result.upper - result.lower < 1e-7  # nosec B101
    assert domain.membership(result.point).status == "inside"  # nosec B101
    assert normalized_score(domain.space, result.coefficients, result.point) <= F(
        result.upper
    )  # nosec B101
    restored = domain_from_manifest(domain.to_manifest())
    assert restored.ref == domain.ref and restored.capabilities.linear_optimization  # nosec B101


def test_worked_operating_modes_audit_and_failure_replay():
    from examples.uncertainty_compositions import operating_modes_example, run_example

    model, combined, unrestricted = operating_modes_example()
    exports = run_example()
    for name, domain, verdict in (
        ("operating_modes", combined, "pass"),
        ("unrestricted_loads", unrestricted, "fail"),
    ):
        result = result_from_json(json.dumps(exports[name]))
        assert result.payload.verdict == verdict  # nosec B101
        portable(result)
        if verdict == "fail":
            witness = result.payload.witness.realizations[0]
            assert domain.membership(witness).status == "inside"  # nosec B101
            replay = model.as_claim(domain).evaluate_result(witness)
            assert any(
                c.assessment == "violated" for c in replay.payload.constraint_checks
            )  # nosec B101


def test_one_unresolved_union_branch_prevents_a_passing_audit(monkeypatch):
    from examples.uncertainty_compositions import operating_modes_example

    model, domain, _ = operating_modes_example()
    bad_branch = domain.factors[0].factors[1]
    original = BoxSet.maximize_linear

    def unresolved(self, weights):
        result = original(self, weights)
        if self is bad_branch:
            return replace(
                result, status="unresolved", point=None, lower=None, upper=None
            )
        return result

    monkeypatch.setattr(BoxSet, "maximize_linear", unresolved)
    result = model.as_claim(domain).audit_result()
    assert result.execution == "unresolved"  # nosec B101
    assert result.payload.verdict != "pass"  # nosec B101
    portable(result)


@pytest.mark.parametrize("kind", (Product, Union))
def test_nested_missing_cvxpy_remains_unsupported(monkeypatch, kind):
    from operange import _intersection_support

    monkeypatch.setattr(_intersection_support, "find_spec", lambda *a: None)
    joint = Intersection((box(), box()), backend="cvxpy")
    domain = kind((joint,))
    assert not domain.capabilities.linear_optimization  # nosec B101
    assert domain.maximize_linear({"a": 1}).status == "unsupported"  # nosec B101


@pytest.mark.parametrize(
    "name, limit",
    (
        ("ellipsoid_with_bounds", 18.7),
        ("scheduled_loads", 17.9),
        ("specified_feed", 0.7),
    ),
)
def test_convex_workflows_cover_the_domain_and_replay_stricter_limit_failures(
    name, limit
):
    pytest.importorskip("cvxpy")
    from examples.uncertainty_compositions import convex_examples

    model, domain = convex_examples()[name]
    result = model.as_claim(domain).audit_result()
    assert result.payload.verdict == "pass"  # nosec B101
    portable(result)
    stricter = replace(
        model, requirements=(replace(model.requirements[0], limit=limit),)
    )
    claim = stricter.as_claim(domain)
    failure = claim.audit_result()
    assert failure.payload.verdict == "fail"  # nosec B101
    witness = failure.payload.witness.realizations[0]
    assert domain.membership(witness).status == "inside"  # nosec B101
    replay = claim.evaluate_result(witness)
    assert any(c.assessment == "violated" for c in replay.payload.constraint_checks)  # nosec B101
    portable(failure)
