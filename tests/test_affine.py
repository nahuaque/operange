"""Analytical acceptance cases for fixed affine claims and support evidence."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from operange import (
    AffineOutput,
    AffineProcessAdapter,
    AffineRequirement,
    AffineTerm,
    Axis,
    BoxSet,
    BudgetSet,
    Coordinate,
    DecisionRule,
    EllipsoidSet,
    FiniteSet,
    Intersection,
    LinearConstraint,
    NormalizedLInf,
    Parameter,
    ParameterSpace,
    PolytopeSet,
    RecoursePolicy,
    Scenario,
    SimplexSet,
    VectorParameter,
    result_from_json,
)
from operange import linear
from operange.contract_types import QuantitySpec


def domain():
    return BoxSet(
        (
            Parameter("a", "MW", 10, 8, 12, 2, "test load"),
            Parameter("b", "MW", 6, 4, 8, 2, "test load"),
        )
    )


def model(limit=19, *, controls=(), terms=None, requirements=None):
    return AffineProcessAdapter(
        "Utility capacity",
        domain().space,
        (
            AffineOutput(
                "demand",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("a", 1, "MW/MW"), AffineTerm("b", 1, "MW/MW"))
                if terms is None
                else terms,
                "Analytical load sum",
            ),
        ),
        (AffineRequirement("capacity", "demand", limit),)
        if requirements is None
        else requirements,
        controls,
    )


def bounds(result):
    return {
        e.details["requirement"]: {m.name: m.value for m in e.measurements}
        for e in result.evidence
        if e.subject == "requirement_bound" and e.outcome == "verified"
    }


def test_box_and_finite_support_names_scaling_and_missing_normalization():
    box = domain()
    result = box.maximize_linear({"b": -2, "a": 1})
    assert result.lower == result.upper == 3  # nosec B101
    assert result.point == {"a": 12, "b": 4}  # nosec B101
    assert box.maximize_linear({}).upper == 0  # nosec B101
    degenerate = BoxSet((Parameter("x", "MW", 3, 3, 3, 2, "fixed"),))
    assert degenerate.maximize_linear({"x": 5}).upper == 0  # nosec B101
    finite = FiniteSet(
        box.space,
        (
            Scenario("first", {"a": 12, "b": 6}, "test"),
            Scenario("duplicate", {"a": 12, "b": 6}, "test"),
            Scenario("second", {"a": 10, "b": 8}, "test"),
        ),
    )
    support = finite.maximize_linear({"a": 3, "b": 1})
    assert support.upper == support.lower == 3  # nosec B101
    assert support.evidence[-1].details["maximizing_scenarios"] == (
        "duplicate",
        "first",
    )  # nosec B101
    assert finite.capabilities.linear_optimization  # nosec B101
    unscaled = replace(
        finite,
        space=ParameterSpace(
            tuple(
                Coordinate(c.name, c.unit, c.provenance) for c in box.space.coordinates
            )
        ),
    )
    assert not unscaled.capabilities.linear_optimization  # nosec B101
    assert unscaled.maximize_linear({"a": 1}).status == "unsupported"  # nosec B101
    assert model().as_claim(unscaled).audit_result().payload.verdict == "pass"  # nosec B101
    assert (
        model()
        .as_claim(unscaled)
        .sensitivity_result(finite.feasible_point)
        .payload.availability
        == "available"
    )  # nosec B101
    assert (
        model()
        .as_claim(unscaled)
        .sensitivity_result(finite.feasible_point, coordinate_space="normalized")
        .execution
        == "unsupported"
    )  # nosec B101


@pytest.mark.parametrize(
    "kind,expected,maximum",
    [
        ("box", "fail", 20),
        ("budget", "pass", 18),
        ("polytope", "pass", 18),
        ("ellipsoid", "pass", 16 + 2 * 2**0.5),
        ("finite", "pass", 18),
    ],
)
def test_same_model_audits_alternative_geometries(kind, expected, maximum):
    box = domain()
    domains = {
        "box": box,
        "budget": BudgetSet(box, 1),
        "polytope": PolytopeSet(box, (LinearConstraint("joint", {"a": 1, "b": 1}, 1),)),
        "ellipsoid": EllipsoidSet(
            box.space, {"a": {"a": 1, "b": 0}, "b": {"a": 0, "b": 1}}
        ),
        "finite": FiniteSet(
            box.space,
            (
                Scenario("a peak", {"a": 12, "b": 6}, "test"),
                Scenario("b peak", {"a": 10, "b": 8}, "test"),
            ),
        ),
    }
    claim = model().as_claim(domains[kind])
    audit = claim.audit_result()
    assert audit.payload.verdict == expected  # nosec B101
    if kind == "finite":
        residual = max(
            e.payload.constraint_checks[-1].residual.value
            for e in audit.supporting_evaluations
        )
        assert residual == pytest.approx(maximum - 19)  # nosec B101
        assert audit.payload.coverage.method == "complete_finite"  # nosec B101
    else:
        assert bounds(audit)["capacity"]["residual_upper"] == pytest.approx(
            maximum - 19
        )  # nosec B101
    assert audit.payload.search is None  # nosec B101
    assert (
        audit.contract.domain == domains[kind].to_manifest()
        or json.loads(audit.to_json())["artifacts"]["contract"]["domain"]
        == domains[kind].to_manifest()
    )  # nosec B101
    assert result_from_json(audit.to_json()) == audit  # nosec B101
    if expected == "fail":
        witness = audit.payload.witness
        assert witness.kind == "fixed_policy_failure"  # nosec B101
        assert witness.realizations == ({"a": 12, "b": 8},)  # nosec B101
        assert (
            claim.evaluate_result(witness.realizations[0])
            .payload.constraint_checks[-1]
            .assessment
            == "violated"
        )  # nosec B101


def test_evaluation_distinguishes_defined_response_from_requirement_failure():
    claim = model().as_claim(domain())
    result = claim.evaluate_result({"b": 8, "a": 12})
    assert result.payload.feasibility == "feasible"  # nosec B101
    assert result.payload.constraint_checks[-1].assessment == "violated"  # nosec B101
    assert result.payload.included_constraints == ("affine_definition:demand",)  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101
    assert (
        claim.sensitivity_result({"a": 12, "b": 8}).payload.availability == "available"
    )  # nosec B101
    outside = claim.evaluate_result({"a": 12.00001, "b": 8})
    assert (
        outside.execution == "invalid"
        and outside.payload.membership.status == "outside"
    )  # nosec B101
    assert not outside.payload.values  # nosec B101
    assert claim.evaluate_result({"a": float("nan"), "b": 6}).execution == "invalid"  # nosec B101
    assert claim.evaluate_result(None).execution == "invalid"  # nosec B101


def test_jacobian_directional_units_and_reordered_normalization():
    adapter = model(terms=(AffineTerm("a", 3, "MW/MW"), AffineTerm("b", -2, "MW/MW")))
    rescaled = BoxSet(
        (
            replace(domain().parameter("b"), nominal=5, scale=4),
            replace(domain().parameter("a"), nominal=11, scale=0.5),
        )
    )
    claim = adapter.as_claim(rescaled)
    result = claim.sensitivity_result(
        {"b": 6, "a": 10}, inputs=("b", "a"), coordinate_space="normalized"
    )
    d = result.payload.derivative
    assert d.values == ((-8, 1.5),)  # nosec B101
    assert d.origins == (5, 11) and d.scales == (4, 0.5)  # nosec B101
    assert d.units == (("MW/1", "MW/1"),)  # nosec B101
    direction = claim.sensitivity_result(
        {"a": 10, "b": 6}, inputs=("b", "a"), operator="directional", direction=(2, -1)
    )
    assert direction.payload.derivative.values == (-7,)  # nosec B101
    assert direction.payload.derivative.units == (("MW",),)  # nosec B101
    assert result.payload.validity["domain_tangency_asserted"] is False  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101
    assert result_from_json(direction.to_json()) == direction  # nosec B101
    # The model's physical coefficients do not change with domain normalization.
    original = adapter.as_claim(domain()).audit_result()
    changed = claim.audit_result()
    assert (
        bounds(changed)["capacity"]["attained_residual"]
        == bounds(original)["capacity"]["attained_residual"]
        == 9
    )  # nosec B101


def test_simplex_vector_blend_and_ambient_derivative():
    mixture = VectorParameter(
        "feed",
        "mole_fraction",
        Axis("species", ("A", "B")),
        {"A": 0.5, "B": 0.5},
        0,
        1,
        0.5,
        "Blend test",
    )
    names = mixture.space.names
    adapter = AffineProcessAdapter(
        "Blend property",
        mixture.space,
        (
            AffineOutput(
                "quality",
                "score",
                "quality_index",
                1,
                (
                    AffineTerm(names[0], 2, "score/mole_fraction"),
                    AffineTerm(names[1], 4, "score/mole_fraction"),
                ),
                "Synthetic affine property",
            ),
        ),
        (AffineRequirement("minimum_quality", "quality", 3, relation="ge"),),
    )
    claim = adapter.as_claim(SimplexSet(mixture.space, "mole"))
    result = claim.audit_result()
    assert result.payload.verdict == "pass"  # nosec B101
    derivative = claim.sensitivity_result(mixture.flatten(mixture.nominal))
    assert derivative.payload.derivative.values == ((2, 4),)  # nosec B101
    assert derivative.payload.validity["perturbation_scope"] == "ambient"  # nosec B101
    assert (
        claim.sensitivity_result(
            mixture.flatten(mixture.nominal), perturbation_scope="domain"
        ).execution
        == "unsupported"
    )  # nosec B101
    broken = (
        replace(adapter, requirements=(replace(adapter.requirements[0], limit=3.25),))
        .as_claim(claim.domain)
        .audit_result()
    )
    assert broken.payload.verdict == "fail"  # nosec B101
    assert broken.payload.witness.realizations[0] == mixture.flatten({"A": 1, "B": 0})  # nosec B101


def test_fixed_control_values_are_consumed_and_adjustability_is_rejected():
    control = QuantitySpec("recovery", "MW", "thermal_power", "control")
    adapter = model(
        controls=(control,),
        terms=(
            AffineTerm("a", 1, "MW/MW"),
            AffineTerm("b", 1, "MW/MW"),
            AffineTerm("recovery", -1, "MW/MW"),
        ),
    )
    fixed = RecoursePolicy(
        "fixed", (DecisionRule("recovery", "MW", "before_disturbance", fixed_value=2),)
    )
    claim = adapter.as_claim(domain(), recourse=fixed)
    assert claim.audit_result().payload.verdict == "pass"  # nosec B101
    zero = replace(fixed, rules=(replace(fixed.rules[0], fixed_value=0),))
    assert (
        adapter.as_claim(domain(), recourse=zero).audit_result().payload.verdict
        == "fail"
    )  # nosec B101
    adjustable = RecoursePolicy(
        "static", (DecisionRule("recovery", "MW", "after", observes=("a",)),)
    )
    unsupported = adapter.as_claim(domain(), recourse=adjustable).audit_result()
    assert (
        unsupported.execution == "unsupported"
        and not unsupported.supporting_evaluations
    )  # nosec B101
    with pytest.raises(ValueError, match="supply fixed"):
        adapter.as_claim(domain())
    assert (
        adapter.as_claim(domain(), recourse=RecoursePolicy("fixed", ()))
        .audit_result()
        .execution
        == "unsupported"
    )  # nosec B101
    assert RecoursePolicy("fixed", ()).rules == ()  # nosec B101
    with pytest.raises(ValueError):
        RecoursePolicy("causal", ())


def test_requirement_subsets_and_lower_limits_do_not_change_physics():
    adapter = model(
        requirements=(
            AffineRequirement("maximum", "demand", 21),
            AffineRequirement("minimum", "demand", 13, relation="ge"),
        )
    )
    full = adapter.as_claim(domain()).audit_result()
    selected = adapter.as_claim(domain(), requirements=("maximum",)).audit_result()
    assert (
        full.payload.verdict == "fail"
        and full.payload.witness.affected_constraints == ("minimum",)
    )  # nosec B101
    assert selected.payload.verdict == "pass"  # nosec B101
    assert set(bounds(selected)) == {"maximum"}  # nosec B101
    assert all(
        "minimum" not in (c.constraint_ref for c in e.payload.constraint_checks)
        for e in selected.supporting_evaluations
    )  # nosec B101


def failed_solver(*args, **kwargs):
    return SimpleNamespace(
        success=False, status=1, message="injected time limit", x=None
    )


def test_loose_upper_bound_does_not_become_a_failure_or_pass(monkeypatch):
    polytope = PolytopeSet(domain(), (LinearConstraint("joint", {"a": 1, "b": 1}, 1),))
    monkeypatch.setattr(linear, "linprog", failed_solver)
    result = model().as_claim(polytope).audit_result()
    assert result.payload.verdict == "inconclusive" and result.execution == "unresolved"  # nosec B101
    assert result.payload.witness is None  # nosec B101
    assert bounds(result)["capacity"]["attained_residual"] == -3  # nosec B101
    assert bounds(result)["capacity"]["residual_upper"] > 1  # nosec B101
    # Independent envelope bounds suffice for a looser requirement despite timeout.
    assert model(21).as_claim(polytope).audit_result().payload.verdict == "pass"  # nosec B101
    # A verified seed violation suffices even when an optimal witness is unavailable.
    assert model(15).as_claim(polytope).audit_result().payload.verdict == "fail"  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101


def test_no_process_conclusion_transfers_from_wrong_support_identity(monkeypatch):
    original = BoxSet.maximize_linear

    def wrong_domain(self, coefficients):
        result = original(self, coefficients)
        return replace(result, domain_ref=BudgetSet(self, 1).ref)

    monkeypatch.setattr(BoxSet, "maximize_linear", wrong_domain)
    result = model(21).as_claim(domain()).audit_result()
    assert result.payload.verdict == "inconclusive"  # nosec B101
    assert not bounds(result)  # nosec B101


def test_unresolved_membership_and_unsupported_operations_do_not_execute_support(
    monkeypatch,
):
    box = domain()
    budget = BudgetSet(box, 1)
    point = {"a": 12, "b": 6 + 1e-12}
    claim = model().as_claim(budget)
    evaluation = claim.evaluate_result(point)
    assert (
        evaluation.execution == "unresolved"
        and evaluation.payload.membership.status == "unknown"
    )  # nosec B101
    assert claim.sensitivity_result(point).payload.availability == "unknown"  # nosec B101
    composed = model().as_claim(Intersection((box, budget)))
    monkeypatch.setattr(
        BoxSet,
        "maximize_linear",
        lambda *a, **k: pytest.fail("unsupported query must not optimize"),
    )
    assert composed.evaluate_result(box.nominal).execution == "completed"  # nosec B101
    assert composed.audit_result().execution == "unsupported"  # nosec B101
    distance = replace(model().as_claim(box), distance=NormalizedLInf(box.space))
    assert distance.boundary_result().execution == "unsupported"  # nosec B101
    assert distance.breaking_result().execution == "unsupported"  # nosec B101


@pytest.mark.parametrize(
    "options,execution",
    [
        ({"inputs": ("missing",)}, "invalid"),
        ({"inputs": ("a", "a")}, "invalid"),
        ({"outputs": ()}, "invalid"),
        ({"direction": (1, 2)}, "invalid"),
        ({"operator": "directional", "direction": (1,)}, "invalid"),
        ({"operator": "hessian"}, "unsupported"),
        ({"response": "optimized"}, "unsupported"),
        ({"method": "finite_difference"}, "unsupported"),
        ({"side": "invalid"}, "invalid"),
    ],
)
def test_sensitivity_options_are_explicit(options, execution):
    result = model().as_claim(domain()).sensitivity_result(domain().nominal, **options)
    assert result.execution == execution and not result.supporting_evaluations  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101


def test_affine_unit_identity_and_numeric_validation():
    for term in (AffineTerm("a", 1, "kW/MW"), AffineTerm("unknown", 1, "MW/MW")):
        with pytest.raises(ValueError):
            model(terms=(term,))
    with pytest.raises(ValueError, match="duplicate"):
        model(terms=(AffineTerm("a", 1, "MW/MW"), AffineTerm("a", 2, "MW/MW")))
    with pytest.raises(ValueError, match="unknown affine output"):
        model(requirements=(AffineRequirement("wrong", "missing", 1),))
    with pytest.raises(ValueError, match="duplicate"):
        replace(
            model(),
            outputs=(replace(model().outputs[0], name="a"),),
            requirements=(AffineRequirement("cap", "a", 1),),
        )
    with pytest.raises(ValueError):
        AffineRequirement("invalid", "demand", 1, residual_scale=0)
    with pytest.raises(ValueError):
        AffineTerm("a", float("inf"), "MW/MW")
    assert (
        model().as_claim(domain()).audit_result(unrecognized=True).execution
        == "invalid"
    )  # nosec B101


def test_overflow_is_unresolved_and_exact_products_avoid_intermediate_underflow():
    huge = model(terms=(AffineTerm("a", 1e308, "MW/MW"),))
    claim = huge.as_claim(domain())
    assert claim.evaluate_result(domain().nominal).execution == "unresolved"  # nosec B101
    assert claim.audit_result().payload.verdict == "inconclusive"  # nosec B101
    tiny = BoxSet((Parameter("x", "MW", 0, 0, 1e108, 1e-200, "Extreme scaling"),))
    adapter = AffineProcessAdapter(
        "Tiny coefficient",
        tiny.space,
        (
            AffineOutput(
                "response",
                "MW",
                "thermal_power",
                0,
                (AffineTerm("x", 1e-200, "MW/MW"),),
                "Extreme test",
            ),
        ),
        (
            AffineRequirement(
                "cap", "response", 1e-100, tolerance=0, residual_scale=1e-300
            ),
        ),
    )
    result = adapter.as_claim(tiny).audit_result()
    assert result.payload.verdict == "fail"  # nosec B101
    replay = adapter.as_claim(tiny).evaluate_result(
        result.payload.witness.realizations[0]
    )
    assert replay.payload.constraint_checks[-1].assessment == "violated"  # nosec B101


def test_serialized_results_load_without_model_execution(monkeypatch):
    claim = model().as_claim(domain())
    results = (
        claim.evaluate_result(domain().nominal),
        claim.sensitivity_result(domain().nominal),
        claim.audit_result(),
    )
    documents = [r.to_json() for r in results]
    monkeypatch.setattr(
        AffineOutput,
        "evaluate",
        lambda *a, **k: pytest.fail("loading a result must not execute equations"),
    )
    for document, result in zip(documents, results):
        assert result_from_json(document) == result  # nosec B101


def test_multiple_output_units_and_residual_scaling_preserve_the_claim():
    original = model(20)
    adapter = replace(
        original,
        outputs=(
            *original.outputs,
            AffineOutput(
                "demand_kw",
                "kW",
                "thermal_power",
                0,
                (AffineTerm("a", 1000, "kW/MW"), AffineTerm("b", 1000, "kW/MW")),
                "Explicit unit conversion",
            ),
        ),
        requirements=(
            *original.requirements,
            AffineRequirement(
                "capacity_kw", "demand_kw", 20000, tolerance=1e-5, residual_scale=1000
            ),
        ),
    )
    claim = adapter.as_claim(domain())
    audit = claim.audit_result()
    assert audit.payload.verdict == "pass"  # nosec B101
    assert bounds(audit)["capacity_kw"]["residual_upper"] == pytest.approx(
        1000 * bounds(audit)["capacity"]["residual_upper"]
    )  # nosec B101
    derivative = claim.sensitivity_result(
        domain().nominal, outputs=("demand_kw", "demand")
    )
    assert derivative.payload.derivative.values == ((1000, 1000), (1, 1))  # nosec B101
    assert derivative.payload.derivative.units == (
        ("kW/MW", "kW/MW"),
        ("MW/MW", "MW/MW"),
    )  # nosec B101
    assert result_from_json(audit.to_json()) == audit  # nosec B101
