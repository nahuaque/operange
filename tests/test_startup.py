"""Independent transient calculations, finite-domain scope and evidence checks."""

from dataclasses import replace
from math import nextafter, inf

import pytest

from operange import (
    BoxSet,
    DecisionRule,
    FiniteSet,
    MembershipCheck,
    Parameter,
    PiecewiseLinearProfile,
    RecoursePolicy,
    Scenario,
    result_from_json,
)
from operange.reference import StartupEvent, StartupLoadAdapter


def model(*, profile=None, second=True, **kwargs):
    profile = profile or PiecewiseLinearProfile(
        (0, 0.1, 1, 2), (0, 300, 250, 100), "kVA", "Synthetic"
    )
    events = (StartupEvent("a", profile, 0.2, "Test"),)
    if second:
        events += (StartupEvent("b", profile, 0.2, "Test"),)
    return StartupLoadAdapter(
        "Start test",
        events,
        **{
            "horizon_seconds": 8,
            "capacity": 500,
            "background_load": 40,
            **kwargs,
        },
    )


def domain(adapter, *changes):
    return FiniteSet(
        adapter.input_space,
        tuple(
            Scenario(
                f"case_{i}", {**adapter.input_space.nominal, **c}, "Test uncertainty"
            )
            for i, c in enumerate(changes or ({},))
        ),
    )


def values(result):
    return {q.quantity_ref: q.value for q in result.payload.values}


def profile_evidence(result):
    return next(e for e in result.evidence if e.evidence_id == "profile").details


def test_simultaneous_failure_and_staggered_pass_keep_domain_and_capacity():
    original = model()
    stressed = {
        f"{e.name}:{field}": value
        for e in original.events
        for field, value in (
            ("amplitude_scale", 1.1),
            ("duration_scale", 1.25),
            ("timing_jitter_seconds", 0.1),
        )
    }
    uncertainty = domain(original, {}, stressed)
    claim = original.as_claim(uncertainty)
    nominal = claim.evaluate_result(uncertainty.scenarios[0].values)
    assert values(nominal)["peak_load"] == 640  # nosec B101
    audit = claim.audit_result()
    assert audit.payload.verdict == "fail"  # nosec B101
    assert audit.payload.witness.kind == "fixed_policy_failure"  # nosec B101
    replay = claim.evaluate_result(audit.payload.witness.realizations[0])
    assert replay.payload.constraint_checks[-1].assessment == "violated"  # nosec B101
    assert max(values(e)["peak_load"] for e in audit.supporting_evaluations) == 700  # nosec B101
    changed = replace(
        original,
        events=(
            original.events[0],
            replace(original.events[1], scheduled_start_seconds=3.1),
        ),
    )
    revised = changed.as_claim(uncertainty).audit_result()
    assert revised.payload.verdict == "pass"  # nosec B101
    assert max(
        values(e)["peak_load"] for e in revised.supporting_evaluations
    ) == pytest.approx(480)  # nosec B101
    assert revised.contract.domain == audit.contract.domain  # nosec B101
    assert revised.contract.model == audit.contract.model  # nosec B101
    assert revised.contract.design == audit.contract.design  # nosec B101
    assert revised.contract_ref != audit.contract_ref  # nosec B101
    assert revised.payload.coverage.method == "complete_finite"  # nosec B101
    for result in (nominal, audit, revised):
        assert result_from_json(result.to_json()) == result  # nosec B101


def test_narrow_spike_between_coarse_samples_is_found_and_integrated():
    profile = PiecewiseLinearProfile(
        (0, 0.499, 0.5, 0.501, 1), (0, 0, 100, 0, 0), "A", "Narrow triangular pulse"
    )
    adapter = StartupLoadAdapter(
        "Narrow spike", (StartupEvent("a", profile, 0, "Test"),), 1, 50
    )
    uncertainty = domain(adapter)
    result = adapter.as_claim(uncertainty).evaluate_result(uncertainty.feasible_point)
    assert all(profile.value_at(t) == 0 for t in (0, 0.25, 0.75, 1))  # nosec B101
    assert values(result)["peak_load"] == 100  # nosec B101
    assert values(result)["peak_time"] == 0.5  # nosec B101
    assert values(result)["integrated_load"] == pytest.approx(0.1)  # nosec B101
    assert adapter.as_claim(uncertainty).audit_result().payload.verdict == "fail"  # nosec B101
    assert profile_evidence(result)["sampling_grid_used"] is False  # nosec B101


def test_initial_jump_does_not_add_fictitious_pre_start_integral():
    profile = PiecewiseLinearProfile((0, 1), (4, 2), "A", "Initial jump")
    adapter = StartupLoadAdapter(
        "Jump", (StartupEvent("a", profile, 1, "Test"),), 3, 5, 1
    )
    uncertainty = domain(adapter)
    result = adapter.as_claim(uncertainty).evaluate_result(uncertainty.feasible_point)
    assert values(result)["peak_load"] == 5  # nosec B101
    assert values(result)["integrated_load"] == 8  # nosec B101
    assert profile.value_at(-0.1) == 0 and profile.value_at(0) == 4  # nosec B101
    assert profile.value_at(5) == 2  # nosec B101
    sides = [
        b for b in profile_evidence(result)["breakpoints"] if b["time_seconds"] == 1
    ]
    assert {b["side"]: b["total_load"] for b in sides} == {"left": 1, "right": 5}  # nosec B101


def test_duration_changes_integrated_exposure_without_changing_peak():
    profile = PiecewiseLinearProfile(
        (0, 1, 2), (0, 4, 2), "A", "Triangle to running load"
    )
    adapter = StartupLoadAdapter(
        "Exposure",
        (StartupEvent("a", profile, 1, "Test"),),
        6,
        5,
        1,
        integral_limit=17.5,
    )
    uncertainty = domain(adapter, {}, {"a:duration_scale": 2})
    result = adapter.as_claim(uncertainty).audit_result()
    assert [values(e)["peak_load"] for e in result.supporting_evaluations] == [5, 5]  # nosec B101
    assert [values(e)["integrated_load"] for e in result.supporting_evaluations] == [
        17,
        18,
    ]  # nosec B101
    assert result.payload.witness.affected_constraints == ("startup_exposure",)  # nosec B101
    assert result.contract.quantity("integrated_load").unit == "A*s"  # nosec B101
    assert (
        adapter.as_claim(uncertainty, requirements=("shared_capacity",))
        .audit_result()
        .payload.verdict
        == "pass"
    )  # nosec B101


def test_finite_timing_cases_do_not_claim_the_continuous_interval():
    pulse = PiecewiseLinearProfile((0, 1, 2), (0, 10, 0), "A", "Pulse")
    adapter = StartupLoadAdapter(
        "Timing",
        (StartupEvent("a", pulse, 4, "Test"), StartupEvent("b", pulse, 4, "Test")),
        10,
        15,
    )
    uncertainty = domain(
        adapter, {"b:timing_jitter_seconds": -3}, {"b:timing_jitter_seconds": 3}
    )
    claim = adapter.as_claim(uncertainty)
    audit = claim.audit_result()
    assert audit.payload.verdict == "pass"  # nosec B101
    assert audit.evidence[-1].details["continuous_uncertainty_coverage"] is False  # nosec B101
    assert claim.evaluate_result(adapter.input_space.nominal).execution == "invalid"  # nosec B101
    with_middle = domain(
        adapter, {"b:timing_jitter_seconds": -3}, {}, {"b:timing_jitter_seconds": 3}
    )
    assert adapter.as_claim(with_middle).audit_result().payload.verdict == "fail"  # nosec B101


@pytest.mark.parametrize(
    "change",
    [
        {"a:duration_scale": 0},
        {"a:duration_scale": -1},
        {"a:amplitude_scale": 0},
        {"a:amplitude_scale": -1},
        {"a:timing_jitter_seconds": -1},
        {"a:timing_jitter_seconds": 8},
        {"a:duration_scale": 10},
    ],
)
def test_invalid_or_truncated_startup_cannot_pass(change):
    adapter = model(second=False, capacity=1000)
    uncertainty = domain(adapter, {}, change)
    claim = adapter.as_claim(uncertainty)
    bad = claim.evaluate_result(uncertainty.scenarios[1].values)
    assert bad.execution == "invalid" and not bad.payload.values  # nosec B101
    audit = claim.audit_result()
    assert audit.payload.verdict == "inconclusive"  # nosec B101
    assert audit.payload.coverage.method == "partial"  # nosec B101
    assert audit.payload.coverage.unexplored_support["unresolved_scenarios"] == (
        "case_1",
    )  # nosec B101
    assert result_from_json(audit.to_json()) == audit  # nosec B101


def test_verified_failure_survives_another_unresolved_scenario():
    adapter = model()
    uncertainty = domain(adapter, {}, {"a:duration_scale": 0})
    result = adapter.as_claim(uncertainty).audit_result()
    assert result.execution == "unresolved" and result.payload.verdict == "fail"  # nosec B101
    assert result.payload.coverage.method == "partial"  # nosec B101
    assert result_from_json(result.to_json()) == result  # nosec B101


def test_unresolved_membership_never_executes_a_profile(monkeypatch):
    adapter = model()
    uncertainty = domain(adapter)
    monkeypatch.setattr(
        FiniteSet,
        "membership",
        lambda self, point: MembershipCheck("unknown", self.ref, point, ()),
    )
    monkeypatch.setattr(
        PiecewiseLinearProfile,
        "_at",
        lambda *a, **k: pytest.fail("unknown membership must not execute a profile"),
    )
    claim = adapter.as_claim(uncertainty)
    result = claim.evaluate_result(uncertainty.feasible_point)
    assert result.execution == "unresolved" and not result.payload.values  # nosec B101
    assert claim.audit_result().payload.verdict == "inconclusive"  # nosec B101


def test_time_rounding_cannot_collapse_a_narrow_profile():
    pulse = PiecewiseLinearProfile(
        (0, 1e-12, 2e-12), (0, 1, 0), "A", "Small pulse at large absolute time"
    )
    adapter = StartupLoadAdapter(
        "Exact time", (StartupEvent("a", pulse, 1e16, "Test"),), 1e16 + 4, 0.5
    )
    uncertainty = domain(adapter)
    result = adapter.as_claim(uncertainty).evaluate_result(uncertainty.feasible_point)
    assert values(result)["peak_load"] == 1  # nosec B101
    assert values(result)["integrated_load"] == 1e-12  # nosec B101
    points = profile_evidence(result)["breakpoints"]
    assert len({p["exact_time_seconds"] for p in points}) > len(
        {p["time_seconds"] for p in points}
    )  # nosec B101


def test_overflow_remains_unresolved_and_boundaries_use_declared_tolerance():
    adapter = model(second=False, capacity=340, tolerance=0)
    uncertainty = domain(adapter)
    assert adapter.as_claim(uncertainty).audit_result().payload.verdict == "pass"  # nosec B101
    smaller = replace(adapter, capacity=nextafter(340, -inf))
    assert smaller.as_claim(uncertainty).audit_result().payload.verdict == "fail"  # nosec B101
    extreme = domain(adapter, {"a:amplitude_scale": 1e308})
    claim = adapter.as_claim(extreme)
    assert claim.evaluate_result(extreme.feasible_point).execution == "unresolved"  # nosec B101
    assert claim.audit_result().payload.verdict == "inconclusive"  # nosec B101


def test_unsupported_controls_domains_and_options_do_not_claim_coverage():
    adapter = model()
    uncertainty = domain(adapter)
    claim = adapter.as_claim(uncertainty)
    for result in (
        claim.sensitivity_result(uncertainty.feasible_point),
        claim.breaking_result(),
        claim.boundary_result(),
    ):
        assert result.execution == "unsupported"  # nosec B101
    assert claim.audit_result(sample_interval=1).execution == "invalid"  # nosec B101
    box = BoxSet(
        tuple(
            Parameter(c.name, c.unit, c.nominal, c.nominal, c.nominal, 1, "Test")
            for c in adapter.input_space.coordinates
        )
    )
    assert adapter.as_claim(box).audit_result().execution == "unsupported"  # nosec B101
    causal = RecoursePolicy(
        "causal",
        tuple(
            DecisionRule(e.control_name, "s", "event", observes=("load",))
            for e in adapter.events
        ),
    )
    assert (
        adapter.as_claim(uncertainty, recourse=causal).audit_result().execution
        == "unsupported"
    )  # nosec B101
    wrong = replace(
        uncertainty,
        space=replace(
            uncertainty.space,
            coordinates=(
                replace(uncertainty.space.coordinates[0], unit="MW"),
                *uncertainty.space.coordinates[1:],
            ),
        ),
    )
    with pytest.raises(ValueError, match="align"):
        adapter.as_claim(wrong)


@pytest.mark.parametrize(
    "times,loads",
    [
        ((0,), (1,)),
        ((1, 2), (0, 1)),
        ((0, 0), (0, 1)),
        ((0, -1), (0, 1)),
        ((0, 1), (0,)),
        ((0, float("nan")), (0, 1)),
    ],
)
def test_invalid_profiles_are_rejected(times, loads):
    with pytest.raises(ValueError):
        PiecewiseLinearProfile(times, loads, "A", "Test")


def test_profile_and_model_validation_and_loading_without_execution(monkeypatch):
    with pytest.raises(ValueError):
        model(profile=PiecewiseLinearProfile((0, 1), (0, -1), "A", "Test"))
    adapter = model()
    with pytest.raises(ValueError, match="unit"):
        replace(
            adapter,
            events=(
                adapter.events[0],
                replace(
                    adapter.events[1],
                    profile=replace(adapter.events[1].profile, unit="MW"),
                ),
            ),
        )
    with pytest.raises(ValueError, match="duplicate"):
        replace(adapter, events=(adapter.events[0], adapter.events[0]))
    with pytest.raises(ValueError):
        replace(adapter, horizon_seconds=0)
    with pytest.raises(ValueError):
        replace(adapter.events[0], scheduled_start_seconds=-1)
    result = adapter.as_claim(domain(adapter)).audit_result()
    document = result.to_json()
    monkeypatch.setattr(
        PiecewiseLinearProfile,
        "_at",
        lambda *a, **k: pytest.fail("loading must not execute profiles"),
    )
    assert result_from_json(document) == result  # nosec B101
