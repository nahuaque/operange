"""Heat targeting: analytical cases, independent heat-allocation LP and scope."""

from dataclasses import replace
from fractions import Fraction
from math import inf, nextafter
from random import Random

import numpy as np
import pytest
from scipy.optimize import linprog

from operange import (
    BoxSet,
    DecisionRule,
    FiniteSet,
    MembershipCheck,
    Parameter,
    RecoursePolicy,
    Scenario,
    result_from_json,
)
from operange.reference import HeatCascadeAdapter, SensibleHeatStream


def stream(name, kind, supply, target, cp=1):
    return SensibleHeatStream(name, kind, supply, target, cp, "Synthetic test stream")


def model(streams=None, **kwargs):
    return HeatCascadeAdapter(
        "Heat target test",
        streams
        if streams is not None
        else (
            stream("hot", "hot", 180, 40, 2),
            stream("cold", "cold", 20, 160, 1),
        ),
        **{
            "delta_t_min_k": 20,
            "hot_utility_capacity_kw": 200,
            "cold_utility_capacity_kw": 200,
            **kwargs,
        },
    )


def domain(adapter, *changes):
    return FiniteSet(
        adapter.input_space,
        tuple(
            Scenario(
                f"case_{i}", {**adapter.input_space.nominal, **c}, "Finite test case"
            )
            for i, c in enumerate(changes or ({},))
        ),
    )


def evaluate(adapter):
    uncertainty = domain(adapter)
    return adapter.as_claim(uncertainty).evaluate_result(uncertainty.feasible_point)


def values(result):
    return {v.quantity_ref: v.value for v in result.payload.values}


def details(result):
    return next(e.details for e in result.evidence if e.evidence_id == "cascade")


def four_stream_case():
    adapter = model(
        (
            stream("h1", "hot", 180, 60, 2),
            stream("h2", "hot", 150, 30, 1),
            stream("c1", "cold", 20, 140, 2),
            stream("c2", "cold", 80, 170, 1),
        ),
        hot_utility_capacity_kw=50,
        cold_utility_capacity_kw=50,
    )
    hot_change = {"h1:heat_capacity_flow_kw_per_k": 1.5}
    cold_change = {"c1:heat_capacity_flow_kw_per_k": 2.5}
    return adapter, domain(
        adapter, {}, hot_change, cold_change, {**hot_change, **cold_change}
    )


def test_four_stream_hand_cascade_and_combined_failure():
    adapter, uncertainty = four_stream_case()
    claim = adapter.as_claim(uncertainty)
    audit = claim.audit_result()
    expected = [(10, 40, 320), (40, 10, 290), (40, 10, 350), (100, 10, 290)]
    for result, targets in zip(audit.supporting_evaluations, expected):
        assert (
            tuple(
                values(result)[key]
                for key in (  # nosec B101
                    "minimum_hot_utility",
                    "minimum_cold_utility",
                    "maximum_heat_recovery",
                )
            )
            == targets
        )
    nominal = audit.supporting_evaluations[0]
    # Independently worked interval balances, starting without hot utility.
    assert [n["shifted_temperature_c"] for n in details(nominal)["nodes"]] == [
        180,
        170,
        150,
        140,
        90,
        50,
        30,
        20,
    ]  # nosec B101
    assert [n["unadjusted_heat_kw"] for n in details(nominal)["nodes"]] == [
        0,
        -10,
        10,
        0,
        0,
        40,
        20,
        30,
    ]  # nosec B101
    pinch = details(nominal)["pinch_points"]
    assert len(pinch) == 1 and (
        pinch[0]["hot_temperature_c"],
        pinch[0]["cold_temperature_c"],
    ) == (180, 160)  # nosec B101
    assert audit.payload.verdict == "fail"  # nosec B101
    assert audit.payload.witness.realizations == (
        uncertainty.scenario("case_3").values,
    )  # nosec B101
    replay = claim.evaluate_result(audit.payload.witness.realizations[0])
    assert any(
        c.constraint_ref == "hot_utility_target" and c.assessment == "violated"
        for c in replay.payload.constraint_checks
    )  # nosec B101
    changed = (
        replace(adapter, hot_utility_capacity_kw=100)
        .as_claim(uncertainty)
        .audit_result()
    )
    assert changed.payload.verdict == "pass"  # nosec B101
    assert (
        changed.contract.domain == audit.contract.domain
        and changed.contract.model == audit.contract.model
    )  # nosec B101
    assert changed.contract_ref != audit.contract_ref  # nosec B101
    assert (
        changed.payload.coverage.evaluated_support["installed_network_feasibility"]
        == "not_assessed"
    )  # nosec B101
    assert changed.evidence[-1].details["continuous_uncertainty_coverage"] is False  # nosec B101
    for result in (nominal, audit, changed):
        assert result_from_json(result.to_json()) == result  # nosec B101


@pytest.mark.parametrize(
    "streams, expected",
    [
        ((stream("h", "hot", 180, 40, 2), stream("c", "cold", 20, 160)), (0, 140, 140)),
        ((stream("h", "hot", 60, 40), stream("c", "cold", 80, 100)), (20, 20, 0)),
        ((stream("h", "hot", 100, 50, 3),), (0, 150, 0)),
        ((stream("c", "cold", 50, 100, 3),), (150, 0, 0)),
        ((stream("h", "hot", 100, 50, 0), stream("c", "cold", 50, 50, 2)), (0, 0, 0)),
    ],
)
def test_analytical_targets_include_temperature_mismatch_and_zero_duty(
    streams, expected
):
    result = evaluate(model(streams))
    assert (
        tuple(
            values(result)[key]
            for key in (  # nosec B101
                "minimum_hot_utility",
                "minimum_cold_utility",
                "maximum_heat_recovery",
            )
        )
        == expected
    )
    if not any(expected):
        assert details(result)["pinch_points"] == details(result)["nodes"] == ()  # nosec B101


def test_pinches_can_cover_intervals_or_include_terminals():
    # Exactly matching shifted streams: the entire temperature span is pinched.
    result = evaluate(
        model((stream("h", "hot", 120, 60), stream("c", "cold", 40, 100)))
    )
    assert len(details(result)["pinch_intervals"]) == 1  # nosec B101
    interval = details(result)["pinch_intervals"][0]
    assert (
        interval["upper"]["shifted_temperature_c"],
        interval["lower"]["shifted_temperature_c"],
    ) == (110, 50)  # nosec B101
    assert {p["location"] for p in details(result)["pinch_points"]} == {
        "upper_terminal",
        "lower_terminal",
    }  # nosec B101


def test_tied_interior_pinches_are_all_preserved():
    adapter, uncertainty = four_stream_case()
    result = adapter.as_claim(uncertainty).evaluate_result(
        uncertainty.scenario("case_1").values
    )
    assert [p["shifted_temperature_c"] for p in details(result)["pinch_points"]] == [
        90,
        30,
    ]  # nosec B101


def heat_allocation_lp(streams, delta):
    """Independent maximum recovery via heat allocation between temperature bands.

    Each hot band can supply its own or colder cold bands. This solves an
    allocation LP, without lifting cumulative heat deficits or locating a pinch.
    """
    shifted = [
        (
            s.kind,
            min(s.supply_temperature_c, s.target_temperature_c)
            + (-delta / 2 if s.kind == "hot" else delta / 2),
            max(s.supply_temperature_c, s.target_temperature_c)
            + (-delta / 2 if s.kind == "hot" else delta / 2),
            s.heat_capacity_flow_kw_per_k,
        )
        for s in streams
    ]
    levels = sorted({t for _, lo, hi, _ in shifted for t in (lo, hi)}, reverse=True)
    hot, cold = [], []
    for i, (hi, lo) in enumerate(zip(levels, levels[1:])):
        for kind, low, high, cp in shifted:
            if cp and low <= lo and high >= hi:
                (hot if kind == "hot" else cold).append((i, cp * (hi - lo)))
    edges = [
        (i, j)
        for i, (a, _) in enumerate(hot)
        for j, (b, _) in enumerate(cold)
        if a <= b
    ]
    if not edges:
        return 0
    matrix = np.zeros((len(hot) + len(cold), len(edges)))
    for col, (i, j) in enumerate(edges):
        matrix[i, col] = matrix[len(hot) + j, col] = 1
    solution = linprog(
        -np.ones(len(edges)),
        A_ub=matrix,
        b_ub=[q for _, q in hot + cold],
        bounds=(0, None),
        method="highs",
    )
    assert solution.success  # nosec B101
    return -solution.fun


def test_cascades_match_independent_allocation_lp_and_energy_balance():
    # Deterministic numerical test cases, not security-sensitive randomness.
    rng = Random(9081)  # nosec B311
    for _ in range(30):
        streams = []
        for i in range(6):
            lo = rng.randint(-20, 120)
            hi = lo + rng.randint(1, 100)
            kind = "hot" if i < 3 else "cold"
            streams.append(
                stream(
                    str(i),
                    kind,
                    hi if kind == "hot" else lo,
                    lo if kind == "hot" else hi,
                    rng.randint(1, 5),
                )
            )
        adapter = model(tuple(streams), delta_t_min_k=rng.randint(1, 40))
        result = evaluate(adapter)
        v = values(result)
        recovered = heat_allocation_lp(streams, adapter.delta_t_min_k)
        assert v["maximum_heat_recovery"] == pytest.approx(recovered)  # nosec B101
        assert v["minimum_hot_utility"] == pytest.approx(
            v["cold_stream_duty"] - recovered
        )  # nosec B101
        assert v["minimum_cold_utility"] == pytest.approx(
            v["hot_stream_duty"] - recovered
        )  # nosec B101
        assert (
            v["hot_stream_duty"] + v["minimum_hot_utility"]
            == v["cold_stream_duty"] + v["minimum_cold_utility"]
        )  # nosec B101
        assert all(Fraction(n["heat_exact_kw"]) >= 0 for n in details(result)["nodes"])  # nosec B101
        wider = values(
            evaluate(replace(adapter, delta_t_min_k=adapter.delta_t_min_k + 10))
        )
        assert wider["maximum_heat_recovery"] <= v["maximum_heat_recovery"]  # nosec B101


def test_stream_order_temperature_translation_and_inactive_stream_do_not_change_targets():
    adapter = model()
    baseline = values(evaluate(adapter))
    revised = replace(
        adapter,
        streams=tuple(
            replace(
                s,
                supply_temperature_c=s.supply_temperature_c + 20,
                target_temperature_c=s.target_temperature_c + 20,
            )
            for s in reversed(adapter.streams)
        ),
    )
    inactive = replace(
        adapter, streams=(*adapter.streams, stream("absent", "hot", 1000, 0, 0))
    )
    for other in (revised, inactive):
        for name in (
            "minimum_hot_utility",
            "minimum_cold_utility",
            "maximum_heat_recovery",
        ):
            assert values(evaluate(other))[name] == baseline[name]  # nosec B101


def test_temperature_scenarios_rebuild_the_shifted_ordering():
    adapter = model((stream("h", "hot", 120, 60), stream("c", "cold", 40, 100)))
    uncertainty = domain(
        adapter, {}, {"h:supply_temperature_c": 80, "h:target_temperature_c": 20}
    )
    audit = adapter.as_claim(uncertainty).audit_result()
    assert [
        values(r)["maximum_heat_recovery"] for r in audit.supporting_evaluations
    ] == [60, 20]  # nosec B101
    assert [values(r)["minimum_hot_utility"] for r in audit.supporting_evaluations] == [
        0,
        40,
    ]  # nosec B101


def test_cold_target_and_requirement_selection():
    adapter = model(cold_utility_capacity_kw=100)
    uncertainty = domain(adapter)
    audit = adapter.as_claim(uncertainty).audit_result()
    assert audit.payload.witness.affected_constraints == ("cold_utility_target",)  # nosec B101
    hot_only = adapter.as_claim(
        uncertainty, requirements=("hot_utility_target",)
    ).audit_result()
    assert hot_only.payload.verdict == "pass"  # nosec B101
    assert "cold_utility_residual" not in values(hot_only.supporting_evaluations[0])  # nosec B101


@pytest.mark.parametrize(
    "change",
    [
        {"hot:heat_capacity_flow_kw_per_k": -1},
        {"hot:supply_temperature_c": 10},
        {"cold:target_temperature_c": -10},
        {"cold:supply_temperature_c": -274},
    ],
)
def test_invalid_physical_scenario_is_not_a_failure_witness_or_complete_pass(change):
    adapter = model()
    uncertainty = domain(adapter, {}, change)
    audit = adapter.as_claim(uncertainty).audit_result()
    assert audit.execution == "unresolved" and audit.payload.verdict == "inconclusive"  # nosec B101
    assert audit.payload.coverage.method == "partial" and audit.payload.witness is None  # nosec B101
    assert audit.supporting_evaluations[-1].execution == "invalid"  # nosec B101


def test_known_failure_survives_an_unresolved_scenario():
    adapter = model(cold_utility_capacity_kw=100)
    audit = adapter.as_claim(
        domain(adapter, {}, {"hot:heat_capacity_flow_kw_per_k": -1})
    ).audit_result()
    assert audit.payload.verdict == "fail" and audit.execution == "unresolved"  # nosec B101
    assert audit.payload.coverage.method == "partial"  # nosec B101
    assert result_from_json(audit.to_json()) == audit  # nosec B101


def test_outside_malformed_and_unknown_membership_are_not_evaluated(monkeypatch):
    adapter = model()
    uncertainty = domain(adapter)
    claim = adapter.as_claim(uncertainty)
    for point in (
        {},
        None,
        {**uncertainty.feasible_point, "extra": 1},
        {**uncertainty.feasible_point, "hot:heat_capacity_flow_kw_per_k": float("nan")},
        {**uncertainty.feasible_point, "hot:heat_capacity_flow_kw_per_k": 3},
    ):
        result = claim.evaluate_result(point)
        assert result.execution == "invalid" and not result.payload.values  # nosec B101
    monkeypatch.setattr(
        FiniteSet,
        "membership",
        lambda *_: MembershipCheck("unknown", "Test unresolved"),
    )
    audit = claim.audit_result()
    assert audit.payload.verdict == "inconclusive"  # nosec B101
    assert not audit.supporting_evaluations[0].payload.values  # nosec B101


def test_duplicate_scenarios_preserve_names_and_deduplicate_evaluations():
    adapter = model()
    audit = adapter.as_claim(domain(adapter, {}, {})).audit_result()
    assert len(audit.supporting_evaluations) == 1  # nosec B101
    assert len(audit.payload.coverage.evaluated_support["scenarios"]) == 2  # nosec B101
    assert audit.payload.coverage.method == "complete_finite"  # nosec B101


def test_unsupported_domains_controls_searches_and_options():
    adapter = model()
    box = BoxSet(
        tuple(
            Parameter(c.name, c.unit, c.nominal, c.nominal, c.nominal, 1, "Test")
            for c in adapter.input_space.coordinates
        )
    )
    assert adapter.as_claim(box).audit_result().execution == "unsupported"  # nosec B101
    claim = adapter.as_claim(domain(adapter))
    for result in (
        claim.sensitivity_result(),
        claim.boundary_result(),
        claim.breaking_result(),
    ):
        assert result.execution == "unsupported"  # nosec B101
    assert claim.audit_result(extra=True).execution == "invalid"  # nosec B101
    assert (
        claim.evaluate_result(adapter.input_space.nominal, extra=True).execution
        == "invalid"
    )  # nosec B101
    with pytest.raises(ValueError, match="unknown quantity"):
        adapter.as_claim(
            domain(adapter),
            recourse=RecoursePolicy(
                "static", (DecisionRule("bypass", "1", "operation"),)
            ),
        )


def test_exact_values_preserve_nearby_temperature_nodes_and_tolerance_boundary():
    adapter = model(
        (stream("a", "cold", 0, 1), stream("b", "cold", 0, nextafter(1, inf))),
        delta_t_min_k=20,
    )
    result = evaluate(adapter)
    nodes = details(result)["nodes"]
    assert nodes[0]["shifted_temperature_c"] == nodes[1]["shifted_temperature_c"]  # nosec B101
    assert (
        nodes[0]["shifted_temperature_exact_c"]
        != nodes[1]["shifted_temperature_exact_c"]
    )  # nosec B101
    # 1 + nextafter(1, +inf) rounds to 2 in float, but exceeds a tolerance of 2.
    adapter = replace(adapter, hot_utility_capacity_kw=0, tolerance_kw=2)
    audit = adapter.as_claim(domain(adapter)).audit_result()
    assert values(audit.supporting_evaluations[0])["minimum_hot_utility"] == 2  # nosec B101
    assert values(audit.supporting_evaluations[0])["hot_utility_residual"] > 2  # nosec B101
    assert audit.payload.verdict == "fail"  # nosec B101


def test_overflow_is_unresolved():
    adapter = model((stream("cold", "cold", 0, 1e308, 1e308),))
    audit = adapter.as_claim(domain(adapter)).audit_result()
    assert audit.execution == "unresolved" and audit.payload.verdict == "inconclusive"  # nosec B101
    assert (
        audit.supporting_evaluations[0].diagnostics[0].code
        == "heat_cascade_arithmetic_unresolved"
    )  # nosec B101


@pytest.mark.parametrize(
    "changes",
    [
        {"streams": ()},
        {"streams": (stream("x", "hot", 100, 50), stream("x", "cold", 50, 100))},
        {"delta_t_min_k": 0},
        {"delta_t_min_k": -1},
        {"hot_utility_capacity_kw": -1},
        {"cold_utility_capacity_kw": -1},
        {"tolerance_kw": -1},
        {"delta_t_min_k": float("inf")},
    ],
)
def test_invalid_model_declarations(changes):
    with pytest.raises(ValueError):
        model(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "latent"},
        {"supply_temperature_c": 0},
        {"target_temperature_c": -300},
        {"heat_capacity_flow_kw_per_k": -1},
        {"provenance": ""},
        {"name": ""},
        {"heat_capacity_flow_kw_per_k": float("nan")},
    ],
)
def test_invalid_stream_declarations(changes):
    with pytest.raises(ValueError):
        replace(stream("h", "hot", 100, 50), **changes)


def test_loading_does_not_execute_the_adapter(monkeypatch):
    result = model().as_claim(domain(model())).audit_result()
    monkeypatch.setattr(
        HeatCascadeAdapter,
        "run",
        lambda *_: pytest.fail("adapter executed during load"),
    )
    assert result_from_json(result.to_json()) == result  # nosec B101
