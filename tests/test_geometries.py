"""Analytical geometry, coordinate alignment, and numerical failure evidence."""

from dataclasses import replace
from itertools import product as cartesian_product
import json
from math import sqrt
from types import SimpleNamespace

import pytest

from operange import (
    Axis,
    BoxSet,
    BudgetSet,
    Coordinate,
    DomainCapabilities,
    EllipsoidSet,
    FiniteSet,
    Intersection,
    LinearConstraint,
    LinearSupport,
    MembershipCheck,
    NormalizedLInf,
    Parameter,
    ParameterSpace,
    PolytopeSet,
    Product,
    Scenario,
    SimplexSet,
    Union,
    VectorParameter,
    domain_from_json,
    domain_from_manifest,
    result_from_json,
)
from operange import linear
from operange.contract_types import Evidence, reference
from operange.reference import example_claim


def box(names=("a", "b"), *, lower=-1, upper=1, nominal=0, scale=1, unit="fraction"):
    return BoxSet(
        tuple(
            Parameter(n, unit, nominal, lower, upper, scale, "Analytical test")
            for n in names
        )
    )


def vector(labels=("A", "B"), unit="mole_fraction"):
    return VectorParameter(
        "feed",
        unit,
        Axis("species", labels),
        dict.fromkeys(labels, 1 / len(labels)),
        0,
        1,
        1,
        "Analytical mixture",
    )


def geometries():
    envelope = box()
    v = vector()
    simplex = SimplexSet(v.space, "mole")
    budget = BudgetSet(envelope, 1)
    polytope = PolytopeSet(
        envelope, (LinearConstraint("coupled", {"a": 1, "b": 1}, 1),)
    )
    ellipsoid = EllipsoidSet(
        envelope.space, {"a": {"a": 1, "b": 0.5}, "b": {"a": 0.5, "b": 1}}
    )
    return (
        BoxSet((v,)),
        simplex,
        budget,
        polytope,
        ellipsoid,
        Intersection((BoxSet((v,)), simplex)),
        Union((budget, polytope)),
        Product((BoxSet((v,)), budget)),
    )


def test_vector_labels_round_trip_and_mix_with_scalar_parameters():
    v = vector()
    temperature = Parameter("temperature", "degC", 20, 5, 35, 15, "Design envelope")
    domain = BoxSet((v, temperature))
    nested = {"temperature": 25, "feed": {"B": 0.25, "A": 0.75}}
    flat = domain.space.flatten(nested)
    assert flat == {v.name_for("A"): 0.75, v.name_for("B"): 0.25, "temperature": 25}  # nosec B101
    assert domain.space.unflatten(flat) == nested  # nosec B101
    assert domain.contains(flat)  # nosec B101
    assert domain.parameter(v.name_for("B")).lower == 0  # nosec B101
    assert domain.nominal[v.name_for("A")] == 0.5  # nosec B101
    assert domain.distance(flat) == pytest.approx(1 / 3)  # nosec B101
    assert v.unflatten(v.flatten(nested["feed"])) == nested["feed"]  # nosec B101
    assert domain.space.coordinate(v.name_for("A")).axis == v.axis  # nosec B101
    assert domain_from_json(json.dumps(domain.to_manifest())).ref == domain.ref  # nosec B101
    with pytest.raises(TypeError):
        v.upper["A"] = 5
    with pytest.raises(ValueError, match="labels"):
        domain.space.flatten({"temperature": 25, "feed": [0.75, 0.25]})


def test_vector_order_is_aligned_by_names_and_axes_not_position():
    left = vector(("A", "B"))
    right = vector(("B", "A"))
    left.space.check_alignment(right.space)
    assert left.flatten({"B": 0.75, "A": 0.25}) == right.flatten({"A": 0.25, "B": 0.75})  # nosec B101
    assert NormalizedLInf(left.space).measure(
        left.flatten({"A": 0.25, "B": 0.75})
    ) == NormalizedLInf(right.space).measure(right.flatten({"A": 0.25, "B": 0.75}))  # nosec B101
    metadata_lost = ParameterSpace(
        tuple(
            Coordinate(c.name, c.unit, c.provenance, c.nominal, c.scale)
            for c in left.space.coordinates
        )
    )
    with pytest.raises(ValueError, match="axes"):
        left.space.check_alignment(metadata_lost)
    with pytest.raises(ValueError, match="units"):
        left.space.check_alignment(vector(unit="mass_fraction").space)
    v = VectorParameter(
        "feed/%",
        "fraction",
        Axis("component/=", ("a]b", "a%5Db")),
        {"a]b": 0.5, "a%5Db": 0.5},
        0,
        1,
        1,
        "Escaping test",
    )
    assert len(set(v.space.names)) == 2  # nosec B101
    assert v.unflatten(v.flatten(v.nominal)) == v.nominal  # nosec B101


@pytest.mark.parametrize(
    "changes",
    [
        {"nominal": {"A": 0.5}},
        {"nominal": {"A": True, "B": 0.5}},
        {"lower": {"A": 0, "B": 0, "C": 0}},
        {"upper": 0.1},
        {"scale": 0},
        {"scale": float("nan")},
    ],
)
def test_vector_rejects_ambiguous_or_invalid_components(changes):
    with pytest.raises(ValueError):
        replace(vector(), **changes)


def test_scalar_and_vector_component_names_cannot_collide():
    v = vector()
    with pytest.raises(ValueError, match="duplicate"):
        BoxSet(
            (v, Parameter(v.name_for("A"), "mole_fraction", 0.5, 0, 1, 1, "Collision"))
        )
    with pytest.raises(ValueError, match="unique"):
        BoxSet((v, Parameter("feed", "mole_fraction", 0.5, 0, 1, 1, "Collision")))


def test_simplex_basis_membership_and_vertex_extremum():
    v = vector()
    domain = SimplexSet(v.space, "mole")
    assert domain.membership(v.flatten({"A": 0.25, "B": 0.75})).status == "inside"  # nosec B101
    assert domain.membership(v.flatten({"A": 0.25, "B": 0.25})).status == "outside"  # nosec B101
    assert domain.membership(v.flatten({"A": -0.25, "B": 1.25})).status == "outside"  # nosec B101
    near = v.flatten({"A": 0.5, "B": 0.5 + 1e-12})
    assert domain.membership(near).status == "unknown"  # nosec B101
    with pytest.raises(ValueError, match="unresolved"):
        domain.contains(near)
    result = domain.maximize_linear({v.name_for("A"): 2, v.name_for("B"): -1})
    assert result.status == "optimal"  # nosec B101
    assert result.lower == result.upper == 1.5  # nosec B101
    assert v.unflatten(result.point) == {"A": 1, "B": 0}  # nosec B101
    with pytest.raises(ValueError, match="matching"):
        SimplexSet(v.space, "mass")
    # Parameter bounds belong to an envelope, not to a coordinate space.
    restricted = replace(v, lower=0.25, upper=0.75)
    composed = Intersection(
        (SimplexSet(restricted.space, "mole"), BoxSet((restricted,)))
    )
    assert composed.membership(v.flatten({"A": 1, "B": 0})).status == "outside"  # nosec B101


@pytest.mark.parametrize(
    "budget,expected", [(0, 0), (0.5, 1.5), (1, 3), (1.5, 4), (3, 6), (5, 6)]
)
def test_budget_fractional_and_zero_allocation(budget, expected):
    domain = BudgetSet(box(("a", "b", "c")), budget)
    result = domain.maximize_linear({"a": 3, "b": -2, "c": 1})
    assert result.status == "optimal"  # nosec B101
    assert result.lower == pytest.approx(expected)  # nosec B101
    assert result.upper == pytest.approx(expected)  # nosec B101
    assert domain.membership(result.point).status == "inside"  # nosec B101


def test_budget_respects_asymmetric_envelope_and_unknown_boundary():
    envelope = BoxSet(
        (
            Parameter("a", "fraction", 0, -0.8, 0.2, 1, "test"),
            Parameter("b", "fraction", 0, -1, 1, 1, "test"),
        )
    )
    domain = BudgetSet(envelope, 0.7)
    result = domain.maximize_linear({"a": 4, "b": -3})
    assert result.upper == pytest.approx(2.3)  # nosec B101
    assert result.point == pytest.approx({"a": 0.2, "b": -0.5})  # nosec B101
    assert domain.membership({"a": 0.3, "b": 0}).status == "outside"  # nosec B101
    assert domain.membership({"a": 0, "b": 0.7 + 1e-12}).status == "unknown"  # nosec B101
    with pytest.raises(ValueError, match="nonnegative"):
        BudgetSet(envelope, -1)


def test_polytope_linear_support_coupling_and_dual_evidence():
    domain = PolytopeSet(
        box(lower=0), (LinearConstraint("capacity", {"a": 1, "b": 2}, 1),)
    )
    result = domain.maximize_linear({"a": 1, "b": 1})
    assert result.status == "optimal"  # nosec B101
    assert result.lower == result.upper == pytest.approx(1)  # nosec B101
    assert result.point == pytest.approx({"a": 1, "b": 0})  # nosec B101
    proof = next(e for e in result.evidence if e.subject == "support_upper_bound")
    assert "multipliers" in proof.details  # nosec B101
    assert "residual_support_correction" in proof.details  # nosec B101
    assert domain.membership({"a": 0.5, "b": 0.5}).status == "outside"  # nosec B101
    assert domain.membership({"a": 1 + 1e-12, "b": 0}).status == "outside"  # nosec B101


def test_polytope_equality_off_nominal_seed_and_nonemptiness():
    restriction = LinearConstraint("balance", {"a": 1, "b": 1}, 0.5, "eq")
    with pytest.raises(ValueError, match="feasible_point"):
        PolytopeSet(box(lower=0), (restriction,))
    domain = PolytopeSet(box(lower=0), (restriction,), {"a": 0.5, "b": 0})
    assert domain.membership(domain.space.nominal).status == "outside"  # nosec B101
    result = domain.maximize_linear({"a": 2, "b": 1})
    assert result.status == "optimal"  # nosec B101
    assert result.lower == result.upper == pytest.approx(1)  # nosec B101
    negative = domain.maximize_linear({"a": -1, "b": -1})
    assert negative.upper == pytest.approx(-0.5)  # nosec B101
    with pytest.raises(ValueError, match="feasible_point"):
        PolytopeSet(box(lower=0), (LinearConstraint("impossible", {"a": 1}, -1),))
    with pytest.raises(ValueError, match="declared coordinates"):
        PolytopeSet(box(), (LinearConstraint("wrong", {"unknown": 1}, 1),))


def test_polytope_failure_keeps_valid_bounds_without_false_optimality(monkeypatch):
    domain = PolytopeSet(
        box(lower=0), (LinearConstraint("coupling", {"a": 1, "b": 1}, 0.5),)
    )
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=False, status=1, message="timeout", x=None
        ),
    )
    result = domain.maximize_linear({"a": 1, "b": 1})
    assert result.status == "bounded"  # nosec B101
    assert result.lower == 0 and result.upper == 2  # nosec B101
    assert all(
        e.outcome == "unresolved"
        for e in result.evidence
        if e.subject == "linear_support_solve"
    )  # nosec B101
    assert domain.membership(result.point).status == "inside"  # nosec B101


@pytest.mark.parametrize("status", [2, 3])
def test_false_infeasibility_or_unbounded_termination_cannot_override_geometry(
    monkeypatch, status
):
    domain = PolytopeSet(box(lower=0), ())
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=False, status=status, message="bad backend conclusion", x=None
        ),
    )
    result = domain.maximize_linear({"a": 1, "b": 1})
    assert result.status == "bounded"  # nosec B101
    assert result.lower == 0 and result.upper == 2  # nosec B101


def test_bad_dual_solution_does_not_create_a_false_upper_bound(monkeypatch):
    domain = PolytopeSet(
        box(lower=0), (LinearConstraint("coupling", {"a": 1, "b": 1}, 0.5),)
    )
    original = linear.linprog

    def invalid_dual(objective, **kwargs):
        if len(objective) != 2:
            return SimpleNamespace(
                success=True, status=0, message="wrong optimum", x=[0] * len(objective)
            )
        return original(objective, **kwargs)

    monkeypatch.setattr(linear, "linprog", invalid_dual)
    result = domain.maximize_linear({"a": 1, "b": 1})
    assert result.status == "bounded"  # nosec B101
    assert result.lower == pytest.approx(0.5) and result.upper == 2  # nosec B101
    assert (
        next(e for e in result.evidence if e.evidence_id == "dual_solver").outcome
        == "unresolved"
    )  # nosec B101


def test_ellipsoid_named_correlations_scaling_and_radius():
    space = box(nominal=10, lower=0, upper=20, scale=2, unit="degC").space
    domain = EllipsoidSet(
        space, {"b": {"b": 1, "a": 0.5}, "a": {"b": 0.5, "a": 1}}, radius=2
    )
    result = domain.maximize_linear({"b": 1, "a": 1})
    assert result.status == "optimal"  # nosec B101
    assert result.upper == pytest.approx(2 * sqrt(3))  # nosec B101
    assert result.point == pytest.approx({"a": 10 + 2 * sqrt(3), "b": 10 + 2 * sqrt(3)})  # nosec B101
    assert domain.membership(result.point).status == "inside"  # nosec B101
    reordered = replace(
        domain, space=ParameterSpace(tuple(reversed(space.coordinates)))
    )
    other = reordered.maximize_linear({"a": 1, "b": 1})
    assert other.upper == pytest.approx(result.upper)  # nosec B101
    assert other.point == pytest.approx(result.point)  # nosec B101
    zero = replace(domain, radius=0).maximize_linear({"a": 1})
    assert zero.lower == zero.upper == 0  # nosec B101
    assert zero.point == space.nominal  # nosec B101


@pytest.mark.parametrize(
    "shape",
    [
        {"a": {"a": 1}},
        {"a": {"a": 1, "b": 0.5}, "b": {"a": 0.25, "b": 1}},
        {"a": {"a": 1, "b": 1}, "b": {"a": 1, "b": 1}},
        {"a": {"a": 1, "b": 2}, "b": {"a": 2, "b": 1}},
        {"a": {"a": 1, "b": 0}, "b": {"a": 0, "b": 1e-14}},
        {"a": {"a": float("nan"), "b": 0}, "b": {"a": 0, "b": 1}},
    ],
)
def test_ellipsoid_rejects_bad_or_singular_shapes(shape):
    with pytest.raises(ValueError):
        EllipsoidSet(box().space, shape)


def test_extreme_finite_inputs_do_not_fabricate_zero_support_or_membership():
    # Squaring the weight first would underflow to zero and lose this maximum.
    domain = EllipsoidSet(box(("a",)).space, {"a": {"a": 1}}, radius=1e154)
    result = domain.maximize_linear({"a": 1e-162})
    assert result.status == "optimal"  # nosec B101
    assert result.upper == pytest.approx(1e-8, rel=1e-12, abs=0)  # nosec B101
    assert result.point == {"a": 1e154}  # nosec B101
    assert domain.maximize_linear({"a": 1e308}).status == "unresolved"  # nosec B101
    simplex = SimplexSet(vector().space, "mole")
    point = dict.fromkeys(simplex.space.names, 1e308)
    assert simplex.membership(point).status == "unknown"  # nosec B101
    envelope = box(lower=-1e308, upper=1e308)
    budget = BudgetSet(envelope, 1)
    assert budget.membership({"a": 1e308, "b": 1e308}).status == "unknown"  # nosec B101


def test_analytical_support_bounds_dominate_independent_grid_points():
    domains = geometries()[2:5]
    for domain in domains:
        w = {"a": 1.5, "b": -0.5}
        result = domain.maximize_linear(w)
        assert result.status == "optimal"  # nosec B101
        for a, b in cartesian_product(
            (-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1), repeat=2
        ):
            point = {"a": a, "b": b}
            if domain.membership(point).status == "inside":
                score = sum(w[n] * v for n, v in domain.space.normalize(point).items())
                assert score <= result.upper + 1e-10  # nosec B101


def test_composition_lifts_named_restrictions_and_does_not_convexify_union():
    joint = box(("a", "b"), lower=0)
    a_only = BoxSet((replace(joint.parameter("a"), upper=0.25),))
    intersection = Intersection((joint, a_only))
    assert intersection.membership({"a": 0.5, "b": 0.5}).status == "outside"  # nosec B101
    assert intersection.membership({"a": 0.25, "b": 1}).status == "inside"  # nosec B101
    disjoint = Product((box(("a",)), box(("b",))))
    assert disjoint.contains({"a": 1, "b": -1})  # nosec B101
    with pytest.raises(ValueError, match="disjoint"):
        Product((joint, a_only))
    positive = box(("a",), nominal=0.75, lower=0.5, upper=1)
    negative = box(("a",), nominal=-0.75, lower=-1, upper=-0.5)
    with pytest.raises(ValueError, match="explicit joint space"):
        Union((positive, negative))
    union = Union((positive, negative), space=box(("a",)).space)
    assert union.membership({"a": 0.75}).status == "inside"  # nosec B101
    assert union.membership({"a": -0.75}).status == "inside"  # nosec B101
    assert union.membership({"a": 0}).status == "outside"  # nosec B101
    with pytest.raises(ValueError, match="feasible_point"):
        Intersection((positive, negative), space=box(("a",)).space)
    assert intersection.capabilities == DomainCapabilities()  # nosec B101
    assert intersection.maximize_linear({"a": 1}).status == "unsupported"  # nosec B101


def test_composition_rejects_unit_axis_and_coordinate_mismatches():
    with pytest.raises(ValueError, match="units"):
        Intersection((box(), box(unit="MW")))
    with pytest.raises(ValueError, match="same coordinate"):
        Union((box(("a",)), box(("a", "b"))))
    with pytest.raises(ValueError, match="nonempty"):
        Product(())
    v = vector()
    fake_scalar = box(v.space.names, unit="mole_fraction", nominal=0.5, lower=0)
    with pytest.raises(ValueError, match="axes"):
        Intersection((BoxSet((v,)), fake_scalar))


class SometimesUnknown:
    def __init__(self, space):
        self.space = space

    capabilities = DomainCapabilities()

    @property
    def feasible_point(self):
        return self.space.nominal

    def to_manifest(self):
        return {"kind": "test_oracle", "space": self.space.to_dict()}

    @property
    def ref(self):
        return reference(self.to_manifest(), "uncertainty_set/v1")

    def membership(self, point):
        point = self.space.validate(point)
        status = "inside" if point == self.feasible_point else "unknown"
        return MembershipCheck(
            status,
            self.ref,
            point,
            (
                Evidence(
                    "membership",
                    "domain_membership",
                    "test_oracle",
                    "verified" if status == "inside" else "unresolved",
                ),
            ),
        )


def test_composition_uses_three_valued_membership_logic():
    envelope = box(("a",))
    oracle = SometimesUnknown(envelope.space)
    intersection = Intersection((envelope, oracle))
    union = Union((envelope, oracle))
    assert intersection.membership({"a": 0.5}).status == "unknown"  # nosec B101
    assert intersection.membership({"a": 2}).status == "outside"  # nosec B101
    assert union.membership({"a": 0.5}).status == "inside"  # nosec B101
    assert union.membership({"a": 2}).status == "unknown"  # nosec B101
    assert union.membership({"a": 2}).evidence[0].outcome == "unresolved"  # nosec B101


def test_finite_vector_and_composite_domains_round_trip_without_solver(monkeypatch):
    v = vector()
    finite = FiniteSet(
        v.space, (Scenario("A rich", v.flatten({"A": 0.75, "B": 0.25}), "test"),)
    )
    domains = (*geometries(), finite)
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: pytest.fail("loading a domain must not invoke optimization"),
    )
    for domain in domains:
        encoded = json.dumps(domain.to_manifest(), allow_nan=False)
        restored = domain_from_json(encoded)
        assert restored.ref == domain.ref  # nosec B101
        assert restored.membership(domain.feasible_point).status == "inside"  # nosec B101
        if domain.capabilities.linear_optimization:
            assert restored.capabilities.linear_optimization  # nosec B101
    with pytest.raises(ValueError, match="unsupported"):
        domain_from_manifest(SometimesUnknown(box().space).to_manifest())


def test_domain_decoder_rejects_malformed_declarations():
    with pytest.raises(ValueError, match="duplicate"):
        domain_from_json('{"kind":"simplex","kind":"simplex"}')
    with pytest.raises(ValueError, match="nonfinite"):
        domain_from_json('{"radius":NaN}')
    for domain in geometries():
        manifest = domain.to_manifest()
        manifest["unexpected"] = 1
        with pytest.raises(ValueError):
            domain_from_manifest(manifest)
    manifest = geometries()[0].to_manifest()
    manifest["parameters"][0]["axis"]["coordinates"].append("undeclared")
    with pytest.raises(ValueError):
        domain_from_manifest(manifest)


def test_new_geometry_does_not_inherit_reference_adapter_certification():
    claim = example_claim().as_claim()
    domain = BudgetSet(claim.domain, 1)
    changed = replace(claim, domain=domain)
    assert domain.capabilities.linear_optimization  # nosec B101
    result = changed.audit_result()
    assert result.execution == "unsupported"  # nosec B101
    assert result.payload.verdict == "not_assessed"  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101
    assert result.contract.domain["kind"] == "budget"  # nosec B101
    assert claim.audit_result().payload.verdict == "fail"  # nosec B101


def test_linear_support_records_reject_nonfinite_or_unjustified_bounds():
    result = BudgetSet(box(), 1).maximize_linear({"a": 1})
    assert LinearSupport(**result.to_dict()) == result  # nosec B101
    with pytest.raises(ValueError, match="bounds"):
        replace(result, upper=-1)
    with pytest.raises(ValueError, match="evidence"):
        replace(result, evidence=())
    with pytest.raises(ValueError, match="both bounds"):
        replace(result, status="bounded", point=None, lower=None, upper=None)
    with pytest.raises(ValueError):
        BudgetSet(box(), 1).maximize_linear({"a": float("nan")})
    with pytest.raises(ValueError, match="declared coordinates"):
        BudgetSet(box(), 1).maximize_linear({"unknown": 1})
