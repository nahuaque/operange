"""Shared DSL semantics and adapter boundaries across static and causal models."""

from dataclasses import dataclass, replace

import pytest

from operange import (
    AdapterCapabilities,
    BoxSet,
    Capability,
    Claim,
    Coordinate,
    DecisionRule,
    Distance,
    DomainCapabilities,
    FiniteSet,
    MembershipCheck,
    ModelAdapter,
    NormalizedLInf,
    Parameter,
    ParameterSpace,
    RecoursePolicy,
    Scenario,
    UncertaintySet,
    result_from_json,
)
from operange.reference import HeatRecoveryAdapter, StorageDesign
from operange.claim import bind_contract
from operange.contract_types import (
    ConstraintCheck,
    ConstraintSpec,
    EvaluationPayload,
    Evidence,
    Membership,
    ProcessContract,
    QuantitySpec,
    QuantityValue,
    reference,
)
from operange.engineering_results import EvaluationResult
from operange.reference import example_claim
from operange.reference import example_storage_claim


POINT = {"source_derating": 0.1, "demand_increase": 0.05}


def test_parameter_space_alignment_units_and_normalization():
    box = example_claim().uncertainty
    assert isinstance(box, UncertaintySet)  # nosec B101
    assert box.space.validate(dict(reversed(tuple(POINT.items())))) == POINT  # nosec B101
    assert box.space.normalize(POINT) == {
        "source_derating": 0.4,
        "demand_increase": 0.2,
    }  # nosec B101
    assert box.parameters[0].coordinate.provenance == box.parameters[0].provenance  # nosec B101
    box.space.check_alignment(ParameterSpace(tuple(reversed(box.space.coordinates))))
    with pytest.raises(ValueError, match="units"):
        box.space.check_alignment(
            replace(
                box.space,
                coordinates=(
                    replace(box.space.coordinates[0], unit="percent"),
                    box.space.coordinates[1],
                ),
            )
        )
    with pytest.raises(ValueError, match="names"):
        box.space.check_alignment(ParameterSpace((Coordinate("other", "MW", "test"),)))
    for invalid in (
        {"source_derating": 0.1},
        {**POINT, "extra": 0},
        {**POINT, "source_derating": True},
        {**POINT, "source_derating": float("nan")},
        None,
    ):
        with pytest.raises(ValueError):
            box.space.validate(invalid)


def test_box_membership_and_capabilities_do_not_imply_probability_or_solver_support():
    box = example_claim().uncertainty
    assert box.capabilities == DomainCapabilities(linear_optimization=True)  # nosec B101
    assert box.membership(POINT).status == "inside"  # nosec B101
    outside = box.membership({**POINT, "source_derating": 0.25000000001})
    assert outside.status == "outside"  # nosec B101
    assert outside.evidence[0].details["outside_coordinates"] == ("source_derating",)  # nosec B101
    assert outside.domain_ref == box.ref  # nosec B101
    assert box.ref == replace(box, parameters=tuple(reversed(box.parameters))).ref  # nosec B101
    assert box.membership(box.upper_corner).status == "inside"  # nosec B101


def test_distance_is_first_class_and_independent_of_membership():
    box = example_claim().uncertainty
    distance = NormalizedLInf(box.space)
    assert isinstance(distance, Distance)  # nosec B101
    assert distance.measure(POINT) == box.distance(POINT) == 0.4  # nosec B101
    assert distance.measure(POINT, POINT) == 0  # nosec B101
    assert distance.measure({"source_derating": 0.5, "demand_increase": 0}) == 2  # nosec B101
    reordered = NormalizedLInf(ParameterSpace(tuple(reversed(box.space.coordinates))))
    assert reordered.measure(POINT) == 0.4  # nosec B101
    assert distance.to_manifest()["unit"] == "1"  # nosec B101
    with pytest.raises(ValueError):
        distance.measure(POINT, {"source_derating": 0})
    with pytest.raises(ValueError):
        NormalizedLInf(ParameterSpace((Coordinate("event", "MW", "test"),)))


def test_finite_domain_has_explicit_members_and_no_invented_nominal():
    domain = example_storage_claim().tree.domain
    assert isinstance(domain, UncertaintySet)  # nosec B101
    assert domain.capabilities.enumeration  # nosec B101
    assert not domain.capabilities.sampling  # nosec B101
    assert not domain.capabilities.linear_optimization  # nosec B101
    assert tuple(s.name for s in domain.members()) == ("Demand surge", "Heat surplus")  # nosec B101
    assert domain.membership({"demand": 1, "recovered_heat": 0}).status == "inside"  # nosec B101
    assert domain.membership({"demand": 0.5, "recovered_heat": 0.5}).status == "outside"  # nosec B101
    with pytest.raises(ValueError, match="nominal"):
        _ = domain.space.nominal
    with pytest.raises(ValueError, match="nonempty"):
        FiniteSet(domain.space, ())
    with pytest.raises(ValueError, match="duplicate"):
        FiniteSet(domain.space, (domain.scenarios[0], domain.scenarios[0]))
    with pytest.raises(ValueError, match="coordinates"):
        FiniteSet(domain.space, (Scenario("bad", {"demand": 1}, "test"),))
    with pytest.raises(ValueError, match="unknown scenario"):
        domain.scenario("missing")
    assert domain.ref == FiniteSet(domain.space, tuple(reversed(domain.scenarios))).ref  # nosec B101
    with pytest.raises(TypeError):
        domain.scenarios[0].values["demand"] = 100


def test_equal_numeric_futures_keep_distinct_identities():
    domain = example_storage_claim().tree.domain
    original = domain.scenarios[0]
    duplicate_values = FiniteSet(
        domain.space, (original, replace(original, name="Other observation"))
    )
    check = duplicate_values.membership(original.values)
    assert check.evidence[0].details["matching_scenarios"] == (
        original.name,
        "Other observation",
    )  # nosec B101


def test_unknown_membership_cannot_be_promoted_without_evidence():
    check = MembershipCheck(
        "unknown",
        reference({"kind": "test_oracle"}, "uncertainty_set/v1"),
        {"x": 1},
        (Evidence("membership", "domain_membership", "oracle", "unresolved"),),
    )
    assert check.status == "unknown"  # nosec B101
    with pytest.raises(ValueError, match="verified evidence"):
        replace(check, status="outside")


def test_recourse_information_excludes_future_observations():
    model = example_storage_claim()
    policy = model.recourse_policy()
    prep = policy.rule("preparation_charge")
    a = prep.information(
        {"preparation_observation": "unrevealed", "event_identity": "Demand surge"}
    )
    b = prep.information(
        {"preparation_observation": "unrevealed", "event_identity": "Heat surplus"}
    )
    assert a == b == {"preparation_observation": "unrevealed"}  # nosec B101
    foresight = model.recourse_policy("perfect_foresight").rule("preparation_charge")
    assert foresight.information(
        {"event_identity": "Demand surge"}
    ) != foresight.information({"event_identity": "Heat surplus"})  # nosec B101
    fixed = model.recourse_policy("fixed", fixed_preparation_power_mw=0).rule(
        "preparation_charge"
    )
    assert fixed.fixed_value == 0  # nosec B101
    assert fixed.information({"event_identity": "Heat surplus"}) == {}  # nosec B101
    with pytest.raises(ValueError, match="missing"):
        prep.information({"event_identity": "known too late"})
    with pytest.raises(ValueError, match="fixed controls"):
        replace(prep, fixed_value=0.5)
    with pytest.raises(ValueError, match="duplicate"):
        RecoursePolicy("causal", (prep, prep))


def test_heat_uses_common_claim_and_preserves_numerical_results():
    model = example_claim()
    claim = model.as_claim()
    assert isinstance(claim, Claim) and isinstance(claim.adapter, ModelAdapter)  # nosec B101
    assert claim.requirements == ("heat_demand",)  # nosec B101
    assert (
        claim.capabilities.evaluation.supported and claim.capabilities.audit.supported
    )  # nosec B101
    evaluation = claim.evaluate_result(POINT)
    values = {v.quantity_ref: v.value for v in evaluation.payload.values}
    assert values["delivery_margin"] == pytest.approx(0.9)  # nosec B101
    assert claim.sensitivity_result(POINT).payload.derivative.values == (
        (-8, 0),
        (0, 6),
        (-8, -6),
        (8, 6),
    )  # nosec B101
    assert claim.boundary_result().payload.search.lower.value == pytest.approx(4 / 7)  # nosec B101
    breaking = claim.breaking_result(shortfall_margin_mw=0.01)
    assert breaking.payload.search.lower.value == pytest.approx(201 / 350)  # nosec B101
    assert breaking.payload.verdict == "fail"  # nosec B101
    assert claim.audit_result() == model.audit_result()  # nosec B101
    assert evaluation.contract_ref == claim.contract.ref  # nosec B101
    assert evaluation.contract.operating["recourse_policy"]["mode"] == "static"  # nosec B101
    assert result_from_json(breaking.to_json()) == breaking  # nosec B101


def test_heat_new_box_is_used_in_the_calculation():
    original = example_claim().as_claim()
    smaller = replace(
        original.domain,
        parameters=tuple(replace(p, upper=0.1) for p in original.domain.parameters),
    )
    claim = replace(original, domain=smaller, distance=smaller.default_distance)
    assert original.audit_result().payload.verdict == "fail"  # nosec B101
    assert claim.audit_result().payload.verdict == "pass"  # nosec B101
    assert claim.breaking_result().payload.search.resolution == "unreachable"  # nosec B101
    assert claim.contract.to_dict()["domain"] == smaller.to_manifest()  # nosec B101


def test_membership_does_not_transfer_a_box_proof_to_a_different_domain(monkeypatch):
    claim = example_claim().as_claim()
    domain = FiniteSet(
        claim.domain.space,
        (
            Scenario("nominal", claim.domain.nominal, "test"),
            Scenario("stress", POINT, "test"),
        ),
    )
    assert domain.membership(POINT).status == "inside"  # nosec B101
    claim = replace(claim, domain=domain)
    monkeypatch.setattr(
        HeatRecoveryAdapter,
        "run",
        lambda *a, **k: pytest.fail("unsupported adapter must not execute"),
    )
    for result in (
        claim.evaluate_result(POINT),
        claim.audit_result(),
        claim.sensitivity_result(POINT),
        claim.breaking_result(),
    ):
        assert result.execution == "unsupported"  # nosec B101
        assert not result.supporting_evaluations  # nosec B101
        assert result.contract.domain["kind"] == "finite_set"  # nosec B101
        assert result_from_json(result.to_json()) == result  # nosec B101


def test_severity_change_does_not_change_physics_or_silently_reuse_search():
    claim = example_claim().as_claim()
    distance = NormalizedLInf(
        replace(
            claim.domain.space,
            coordinates=tuple(
                replace(c, scale=0.5) for c in claim.domain.space.coordinates
            ),
        )
    )
    other = replace(claim, distance=distance)
    assert distance.measure(POINT) == 0.2  # nosec B101
    assert other.capabilities.audit.supported  # nosec B101
    assert other.evaluate_result(POINT) == claim.evaluate_result(POINT)  # nosec B101
    assert not other.capabilities.breaking.supported  # nosec B101
    rejected = other.breaking_result()
    assert rejected.execution == "unsupported"  # nosec B101
    assert rejected.request["distance"]["scales"] == (0.5, 0.5)  # nosec B101
    assert rejected.payload.verdict == "not_assessed"  # nosec B101
    assert replace(claim, distance=None).boundary_result().execution == "unsupported"  # nosec B101


def test_storage_modes_and_information_changes_use_one_claim_interface():
    model = example_storage_claim()
    fixed = model.as_claim("fixed", fixed_preparation_power_mw=0.5)
    causal = model.as_claim()
    perfect = model.as_claim("perfect_foresight")
    assert all(isinstance(c, Claim) for c in (fixed, causal, perfect))  # nosec B101
    assert fixed.domain.ref == causal.domain.ref == perfect.domain.ref  # nosec B101
    assert (
        fixed.audit_result().payload.verdict,
        causal.audit_result().payload.verdict,
        perfect.audit_result().payload.verdict,
    ) == ("fail", "fail", "pass")  # nosec B101
    witness = causal.audit_result().payload.witness
    assert witness.cardinality.upper.value == 2  # nosec B101
    assert fixed.audit_result().payload.witness.cardinality.upper.value == 1  # nosec B101
    assert perfect.evaluate_result().payload.feasibility == "feasible"  # nosec B101
    assert causal.sensitivity_result().execution == "unsupported"  # nosec B101
    assert causal.breaking_result().execution == "unsupported"  # nosec B101
    assert (
        causal.evaluate_result({"demand": 1, "recovered_heat": 0}).execution
        == "invalid"
    )  # nosec B101
    larger = replace(model, design=StorageDesign(capacity_mwh=2)).as_claim()
    assert larger.audit_result().payload.verdict == "pass"  # nosec B101
    assert larger.domain == causal.domain  # nosec B101


def test_future_information_is_never_silently_accepted_as_causal_recourse():
    claim = example_storage_claim().as_claim()
    policy = replace(
        claim.recourse,
        rules=(
            replace(claim.recourse.rules[0], observes=("event_identity",)),
            *claim.recourse.rules[1:],
        ),
    )
    changed = replace(claim, recourse=policy)
    assert changed.audit_result().execution == "unsupported"  # nosec B101
    assert changed.audit_result().payload.verdict == "not_assessed"  # nosec B101
    assert changed.contract.operating["recourse_policy"]["rules"][0]["observes"] == (
        "event_identity",
    )  # nosec B101
    assert changed.contract.ref != claim.contract.ref  # nosec B101


def test_removed_requirement_or_changed_storage_domain_does_not_reuse_proof():
    claim = example_storage_claim().as_claim()
    relaxed = replace(claim, requirements=("event_service",))
    assert relaxed.audit_result().execution == "unsupported"  # nosec B101
    assert relaxed.contract.operating["requirement_refs"] == ("event_service",)  # nosec B101
    reduced = replace(
        claim, domain=FiniteSet(claim.domain.space, (claim.domain.scenarios[0],))
    )
    assert reduced.audit_result().execution == "unsupported"  # nosec B101


def test_claim_rejects_invalid_bindings_and_query_options():
    claim = example_claim().as_claim()
    with pytest.raises(ValueError, match="undeclared"):
        replace(claim, requirements=("profit",))
    with pytest.raises(ValueError, match="requirements"):
        replace(claim, requirements=())
    with pytest.raises(ValueError, match="duplicate"):
        replace(claim, requirements=("heat_demand", "heat_demand"))
    with pytest.raises(ValueError, match="units"):
        replace(
            claim,
            domain=replace(
                claim.domain,
                parameters=tuple(
                    replace(p, unit="percent") for p in claim.domain.parameters
                ),
            ),
        )
    with pytest.raises(ValueError, match="units"):
        replace(
            claim,
            recourse=replace(
                claim.recourse, rules=(replace(claim.recourse.rules[0], unit="kW"),)
            ),
        )
    with pytest.raises(ValueError, match="role"):
        replace(
            claim,
            recourse=replace(
                claim.recourse,
                rules=(replace(claim.recourse.rules[0], control="delivered_heat"),),
            ),
        )
    result = claim.evaluate_result(POINT, unexpected=True)
    assert result.execution == "invalid"  # nosec B101
    assert result.payload.feasibility == "not_evaluated"  # nosec B101
    assert claim.audit_result(unexpected=float("nan")).execution == "invalid"  # nosec B101


def test_model_specific_result_methods_dispatch_through_claim(monkeypatch):
    original = Claim._run
    calls = []

    def tracked(self, operation, *args, **kwargs):
        calls.append((type(self.adapter).__name__, operation))
        return original(self, operation, *args, **kwargs)

    monkeypatch.setattr(Claim, "_run", tracked)
    example_claim().audit_result()
    example_storage_claim().audit_result()
    assert calls == [
        ("HeatRecoveryAdapter", "audit"),
        ("ThermalStorageAdapter", "audit"),
    ]  # nosec B101


@dataclass(frozen=True)
class EvaluationOnlyAdapter:
    """A consumer-owned test adapter with unrelated coordinate/control names."""

    @property
    def input_space(self):
        return ParameterSpace((Coordinate("feed", "kg/s", "test", 1, 1),))

    @property
    def base_contract(self):
        return ProcessContract(
            "test_material_balance_v1",
            {},
            {},
            {},
            {},
            (
                QuantitySpec("feed", "kg/s", "mass_flow", "input"),
                QuantitySpec("outlet", "kg/s", "mass_flow", "control"),
                QuantitySpec("balance", "kg/s", "mass_flow", "residual"),
            ),
            (
                ConstraintSpec(
                    "balance_requirement",
                    "balance",
                    "requirement",
                    "eq",
                    "feed - outlet",
                    1e-8,
                ),
            ),
            {"absolute_tolerance": 1e-8},
        )

    def describe(self, claim):
        return bind_contract(
            self.base_contract, claim.domain, claim.recourse, claim.requirements
        )

    def capabilities(self, claim):
        yes = Capability(True, "One evaluated material balance.")
        no = Capability(
            False, "This consumer adapter only evaluates individual points."
        )
        return AdapterCapabilities(yes, no, no, no, no)

    def run(self, claim, operation, realization, options):
        member = claim.domain.membership(realization)
        return EvaluationResult(
            self.describe(claim),
            {"query": "evaluation", "realization": realization},
            "completed",
            EvaluationPayload(
                "feasible",
                Membership(member.status, evidence_refs=("membership",)),
                (
                    QuantityValue("feed", realization["feed"]),
                    QuantityValue("outlet", realization["feed"]),
                ),
                (
                    ConstraintCheck(
                        "balance_requirement",
                        "satisfied",
                        QuantityValue("balance", 0),
                        ("balance",),
                    ),
                ),
                included_constraints=("balance_requirement",),
            ),
            (
                *member.evidence,
                Evidence("balance", "constraint_check", "substitution", "verified"),
            ),
        )


def test_consumer_owned_adapter_uses_the_same_dsl_and_result_contract():
    domain = BoxSet((Parameter("feed", "kg/s", 1, 0, 2, 1, "test"),))
    policy = RecoursePolicy(
        "static", (DecisionRule("outlet", "kg/s", "operation", ("feed",)),)
    )
    claim = Claim(EvaluationOnlyAdapter(), domain, policy)
    result = claim.evaluate_result({"feed": 1.5})
    assert result.payload.feasibility == "feasible"  # nosec B101
    assert result.payload.values[1].value == 1.5  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101
    assert claim.audit_result().execution == "unsupported"  # nosec B101


def test_adapter_cannot_return_a_different_contract(monkeypatch):
    claim = example_claim().as_claim()
    wrong = replace(example_claim(), cop=5).evaluate_result(POINT)
    monkeypatch.setattr(HeatRecoveryAdapter, "run", lambda *a: wrong)
    with pytest.raises(ValueError, match="different contract"):
        claim.evaluate_result(POINT)
