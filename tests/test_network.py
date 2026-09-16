"""Connected equipment must preserve physical balances and existing evidence paths."""

from dataclasses import replace
from fractions import Fraction as F
import json

import pytest

from operange import (
    AffineControlRule,
    AffineController,
    AffineMeasurement,
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    Boiler,
    BoxSet,
    DecisionRule,
    FixedRatioSplitter,
    FrozenController,
    LinearObjective,
    LinearProcessAdapter,
    MeasurementModel,
    Mixer,
    Parameter,
    ProcessNetwork,
    ProcessStream,
    RecoursePolicy,
    StreamSink,
    StreamSource,
    StreamTotal,
    UtilityHeader,
    result_from_json,
    verify_result,
)
from examples.equipment_network import example, run_example
from examples.linear_dispatch import example as handwritten


def values(result):
    return {v.quantity_ref: v.value for v in result.payload.values}


def water(name, **options):
    return ProcessStream(name, "kg/s", "mass_flow", "water", **options)


def mixing(weights=(1, 3)):
    domain = BoxSet((Parameter("feed", "kg/s", 4, 2, 6, 2, "Fresh feed"),))
    network = ProcessNetwork(
        "Water mixing and splitting",
        domain.space,
        (
            water("fresh", input="feed"),
            water("makeup", value=2),
            water("mixed", upper=10),
            water("left", upper=10),
            water("right", upper=10),
        ),
        (
            StreamSource("fresh_supply", "fresh"),
            StreamSource("makeup_supply", "makeup"),
            Mixer("mixer", ("fresh", "makeup"), "mixed"),
            FixedRatioSplitter("split", "mixed", ("left", "right"), weights),
            StreamSink("left_user", "left"),
            StreamSink("right_user", "right"),
        ),
        totals=(StreamTotal("delivered", ("left", "right")),),
    )
    return network, domain


@pytest.mark.parametrize("backend", ("scipy", "cvxpy"))
def test_boiler_network_matches_independently_written_dispatch(backend):
    if backend == "cvxpy":
        pytest.importorskip("cvxpy")
    network, domain = example()
    model = network.compile()
    assert type(model) is LinearProcessAdapter  # nosec B101
    baseline, cases = handwritten()
    baseline = replace(baseline, objective=LinearObjective("fuel", tolerance=1e-6))
    for point, expected_fuel in ((domain.nominal, 23), (domain.upper_corner, 31)):
        result = model.as_claim(domain).evaluate_result(point, backend=backend)
        assert result.payload.feasibility == "feasible", result.to_json()  # nosec B101
        v = values(result)
        assert v["fuel"] == pytest.approx(expected_fuel, abs=1e-6)  # nosec B101
        assert v["stream:steam_a"] == pytest.approx(12, abs=1e-6)  # nosec B101
        for boiler, efficiency in (("a", 0.8), ("b", 0.5)):
            assert abs(
                F(v[f"steam_{boiler}"]) - F(efficiency) * F(v[f"fuel_{boiler}"])
            ) <= F(network.balance_tolerance)  # nosec B101
            assert v[f"loss:boiler_{boiler}"] == pytest.approx(
                v[f"fuel_{boiler}"] - v[f"steam_{boiler}"]
            )  # nosec B101
        assert v["stream:to_dryer"] == pytest.approx(point["dryer"], abs=1e-8)  # nosec B101
        assert verify_result(result.to_json(compact=True)).verified  # nosec B101
        if point == domain.nominal:
            reference = baseline.as_claim(cases).evaluate_result(point, backend=backend)
            assert v["fuel"] == pytest.approx(values(reference)["fuel"], abs=1e-6)  # nosec B101
    audit = model.as_claim(domain).audit_result(backend=backend)
    assert audit.payload.verdict == "pass", audit.to_json()  # nosec B101
    assert verify_result(audit).verified  # nosec B101
    failed = (
        replace(
            network, operating_limits=(AffineRequirement("shared_fuel", "fuel", 30),)
        )
        .compile()
        .as_claim(domain)
        .audit_result(backend=backend)
    )
    assert failed.payload.verdict == "fail"  # nosec B101
    assert verify_result(failed).verified  # nosec B101


@pytest.mark.parametrize(
    "weights,expected",
    (((1, 3), (1.5, 4.5)), ((0, 2), (0, 6)), ((0.2, 0.8), (1.2, 4.8))),
)
@pytest.mark.parametrize("backend", ("scipy", "cvxpy"))
def test_mass_mixing_fixed_sources_and_proportional_splitting(
    weights, expected, backend
):
    if backend == "cvxpy":
        pytest.importorskip("cvxpy")
    network, domain = mixing(weights)
    model = network.compile()
    assert "fresh" not in {c.name for c in model.controls}  # nosec B101
    fixed = next(c for c in model.controls if c.name == "makeup")
    assert fixed.lower == fixed.upper == 2  # nosec B101
    result = model.as_claim(domain).evaluate_result(domain.nominal, backend=backend)
    assert result.payload.feasibility == "feasible", result.to_json()  # nosec B101
    v = values(result)
    assert v["mixed"] == pytest.approx(6, abs=1e-7)  # nosec B101
    assert (v["left"], v["right"]) == pytest.approx(expected, abs=1e-7)  # nosec B101
    assert v["delivered"] == pytest.approx(v["mixed"], abs=1e-8)  # nosec B101
    assert (
        model.as_claim(domain).audit_result(backend=backend).payload.verdict == "pass"
    )  # nosec B101
    assert verify_result(result).verified  # nosec B101


def test_service_selection_cannot_remove_conservation_or_stream_bounds():
    network, domain = mixing()
    model = network.compile()
    commands = {"makeup": 2, "mixed": 6, "left": 2, "right": 4}
    policy = RecoursePolicy(
        "fixed",
        tuple(
            DecisionRule(c.name, c.unit, "operation", fixed_value=commands[c.name])
            for c in model.controls
        ),
    )
    result = model.as_claim(
        domain,
        recourse=policy,
        requirements=("service:left_user",),
        controller=AffineController(
            "Fixed test flows",
            tuple(
                AffineControlRule(c.name, c.unit, commands[c.name], ())
                for c in model.controls
            ),
            "Deliberately inconsistent split",
        ),
    ).evaluate_result(domain.nominal)
    assert result.payload.feasibility == "infeasible"  # nosec B101
    assert any(
        c.constraint_ref.startswith("ratio:") and c.assessment == "violated"
        for c in result.payload.constraint_checks
    )  # nosec B101
    assert verify_result(result).verified  # nosec B101
    balance_names = {r.name for r in model.operating_limits}
    assert balance_names.isdisjoint({r.name for r in model.requirements})  # nosec B101
    assert "stream_bound:fresh:lower" in balance_names  # nosec B101


def test_input_bounds_apply_to_true_input_without_an_adjustable_copy():
    network, domain = mixing()
    network = replace(
        network,
        streams=tuple(
            replace(s, upper=3) if s.name == "fresh" else s for s in network.streams
        ),
    )
    model = network.compile()
    result = model.as_claim(domain).evaluate_result(domain.nominal)
    assert result.payload.feasibility == "infeasible"  # nosec B101
    assert verify_result(result).verified  # nosec B101
    negative = BoxSet(
        (Parameter("feed", "kg/s", -1, -1, -1, 1, "Invalid physical feed"),)
    )
    assert (
        model.as_claim(negative).evaluate_result(negative.nominal).payload.feasibility
        == "infeasible"
    )  # nosec B101


def test_linear_recycle_is_supported_without_topological_elimination():
    network, domain = mixing((1, 1))
    recycled = replace(
        network,
        streams=tuple(s for s in network.streams if s.name != "makeup"),
        equipment=(
            StreamSource("supply", "fresh"),
            Mixer("mixer", ("fresh", "right"), "mixed"),
            FixedRatioSplitter("split", "mixed", ("left", "right"), (1, 1)),
            StreamSink("product", "left"),
        ),
        totals=(),
    )
    result = recycled.compile().as_claim(domain).evaluate_result(domain.nominal)
    assert result.payload.feasibility == "feasible"  # nosec B101
    v = values(result)
    assert (v["mixed"], v["left"], v["right"]) == pytest.approx((8, 4, 4), abs=1e-7)  # nosec B101


def boiler_controller():
    def terms(dryer=0, evaporator=0):
        return tuple(
            AffineTerm(n, v, "MW/MW")
            for n, v in (("dryer", dryer), ("evaporator", evaporator))
            if v
        )

    return AffineController(
        "A at capacity",
        (
            AffineControlRule("fuel_a", "MW", 15, ()),
            AffineControlRule("steam_a", "MW", 12, ()),
            AffineControlRule("fuel_b", "MW", -24, terms(2, 2)),
            AffineControlRule("steam_b", "MW", -12, terms(1, 1)),
            AffineControlRule("to_dryer", "MW", 0, terms(1, 0)),
            AffineControlRule("to_evaporator", "MW", 0, terms(0, 1)),
        ),
        "Declared algebraic flow rules",
    )


def test_saved_controller_replays_and_verifies_without_network_compilation(monkeypatch):
    network, domain = example()
    claim = network.compile().as_claim(domain, controller=boiler_controller())
    document = claim.freeze().to_json()

    def forbidden(*args, **kwargs):
        raise AssertionError("replay must use the saved linear equations")

    monkeypatch.setattr(ProcessNetwork, "compile", forbidden)
    frozen = FrozenController.from_json(document)
    for result in (frozen.audit_result(), frozen.evaluate_result(domain.nominal)):
        assert result.execution == "completed"  # nosec B101
        assert verify_result(result).verified  # nosec B101
    assert frozen.audit_result().payload.verdict == "pass"  # nosec B101


def test_network_keeps_measurement_errors_out_of_physical_flow():
    network, domain = mixing()
    domain = BoxSet(
        (
            *domain.parameters,
            Parameter("bias", "kg/s", 0, -0.1, 0.1, 0.1, "Meter error"),
        )
    )
    sensors = MeasurementModel(
        "Feed meter",
        (
            AffineMeasurement(
                "meter",
                "kg/s",
                0,
                (
                    AffineTerm("feed", 1, "kg/s/kg/s"),
                    AffineTerm("bias", 1, "kg/s/kg/s"),
                ),
                "Meter",
            ),
        ),
        ("bias",),
    )
    network = replace(network, input_space=domain.space, measurements=sensors)
    model = network.compile()
    assert model.observation_space.names == ("meter",)  # nosec B101
    assert model.output("stream:fresh").terms[0].variable == "feed"  # nosec B101
    assert (
        model.as_claim(domain).evaluate_result(domain.nominal).execution
        == "unsupported"
    )  # nosec B101
    forged = replace(
        network,
        streams=tuple(
            replace(s, input="bias") if s.name == "fresh" else s
            for s in network.streams
        ),
    )
    with pytest.raises(ValueError, match="error inputs cannot change physical"):
        forged.compile()


def test_cvxpy_synthesizes_all_algebraic_rates_and_replays():
    pytest.importorskip("cvxpy")
    network, domain = example()
    model = network.compile()
    search = model.as_claim(domain).synthesize_controller()
    assert search.verified, search.to_json()  # nosec B101
    assert {r.control for r in search.controller.rules} == {
        c.name for c in model.controls
    }  # nosec B101
    replay = FrozenController.from_json(search.freeze().to_json()).audit_result()
    assert replay.payload.verdict == "pass"  # nosec B101
    assert verify_result(replay).verified  # nosec B101


def test_shared_relief_and_operating_limit_use_the_compiled_equipment():
    network, domain = example()
    model = replace(
        network, operating_limits=(AffineRequirement("shared_fuel", "fuel", 30),)
    ).compile()
    claim = model.as_claim(domain)
    result = claim.audit_result(
        relief={
            "changes": [
                {"constraint": "shared_fuel", "maximum": 2, "scale": 1, "unit": "MW"}
            ],
            "objective": "linear",
            "tolerance": 1e-6,
        }
    )
    relief = next(e.details for e in result.evidence if e.evidence_id == "relief")
    assert relief["resolution"] == "minimum_verified", (
        relief["lower"],
        relief["upper"],
    )  # nosec B101
    assert relief["upper"] == pytest.approx(1, abs=1e-6)  # nosec B101
    assert verify_result(result).verified  # nosec B101
    limit = claim.operating_limit("dryer")
    assert limit.resolution == "limit_bracketed", limit.to_json()  # nosec B101
    assert limit.lower <= 11.5 + 1e-7 <= limit.upper + 1e-7  # nosec B101
    assert verify_result(limit).verified  # nosec B101


def test_manifest_roundtrip_is_declarative_and_compilation_is_deterministic(
    monkeypatch,
):
    from operange import linear

    network, _ = mixing()

    def forbidden(*args, **kwargs):
        raise AssertionError("declarations must not invoke a solver")

    monkeypatch.setattr(linear, "solve_lp", forbidden)
    restored = ProcessNetwork.from_json(network.to_json())
    assert restored == network and restored.ref == network.ref  # nosec B101
    reversed_network = replace(
        network,
        streams=tuple(reversed(network.streams)),
        equipment=tuple(reversed(network.equipment)),
        totals=tuple(reversed(network.totals)),
    )
    assert (
        restored.compile().base_contract.ref
        == reversed_network.compile().base_contract.ref
    )  # nosec B101
    assert type(restored.equipment[2]) is Mixer  # nosec B101


@pytest.mark.parametrize(
    "change", ("identity", "schema", "extra", "kind", "nonfinite", "missing")
)
def test_manifest_rejects_tampering_and_invalid_fields(change):
    network, _ = example()
    data = json.loads(network.to_json())
    if change == "identity":
        data["network"]["name"] = "Altered"
    elif change == "schema":
        data["schema_version"] = "process_network/v999"
    elif change == "extra":
        data["network"]["callback"] = "execute this"
    elif change == "kind":
        data["network"]["equipment"][0]["kind"] = "callback"
    elif change == "nonfinite":
        data["network"]["streams"][0]["upper"] = float("inf")
    else:
        del data["network"]
    with pytest.raises(ValueError):
        ProcessNetwork.from_json(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        ProcessNetwork.from_json('{"network": {}, "network": {}}')


@pytest.mark.parametrize(
    "options",
    (
        {},
        {"upper": -1},
        {"upper": float("inf")},
        {"upper": True},
        {"lower": 2, "upper": 1},
        {"value": -1},
        {"value": 1, "input": "x"},
        {"value": 1, "upper": 1},
        {"value": 1, "lower": 1},
        {"input": ""},
    ),
)
def test_stream_modes_reject_unbounded_or_ambiguous_rates(options):
    with pytest.raises(ValueError):
        water("test", **options)


@pytest.mark.parametrize("efficiency", (0, -0.1, 1.1, True, float("nan")))
def test_invalid_boiler_efficiency(efficiency):
    with pytest.raises(ValueError):
        Boiler("boiler", "fuel", "steam", efficiency)


@pytest.mark.parametrize(
    "weights", ((0, 0), (-1, 2), (1,), (1, float("nan")), (1, True))
)
def test_invalid_splitter_weights(weights):
    with pytest.raises(ValueError):
        FixedRatioSplitter("split", "in", ("a", "b"), weights)


@pytest.mark.parametrize(
    "item",
    (
        lambda: Mixer("m", ("a",), "b"),
        lambda: Mixer("m", ("a", "b"), "a"),
        lambda: UtilityHeader("h", (), ("a",)),
        lambda: FixedRatioSplitter("s", "a", ("b", "b"), (1, 1)),
        lambda: StreamTotal("sum", ("a", "a")),
        lambda: StreamSink("sink", "a", minimum=-1),
    ),
)
def test_invalid_port_declarations(item):
    with pytest.raises(ValueError):
        item()


@pytest.mark.parametrize(
    "change,message",
    (
        ("unknown", "unknown stream"),
        ("duplicate_producer", "more than one producer"),
        ("duplicate_consumer", "more than one consumer"),
        ("orphan", "exactly one producer"),
        ("duplicate_stream", "duplicate stream"),
        ("duplicate_equipment", "duplicate equipment"),
        ("unit", "matching unit"),
        ("commodity", "matching unit"),
        ("physical_kind", "matching unit"),
        ("input_unit", "must have stream unit"),
        ("unknown_input", "unknown network input"),
        ("input_collision", "distinct from input"),
        ("total", "matching unit"),
    ),
)
def test_invalid_connections_fail_before_any_solver(change, message):
    network, _ = mixing()
    streams, equipment, totals = network.streams, network.equipment, network.totals
    if change == "unknown":
        equipment = (replace(equipment[0], outlet="missing"), *equipment[1:])
    elif change == "duplicate_producer":
        equipment += (StreamSource("duplicate", "fresh"),)
    elif change == "duplicate_consumer":
        equipment += (StreamSink("duplicate", "left"),)
    elif change == "orphan":
        equipment = equipment[1:]
    elif change == "duplicate_stream":
        streams += (streams[0],)
    elif change == "duplicate_equipment":
        equipment += (equipment[0],)
    elif change == "unit":
        streams = tuple(
            replace(s, unit="t/h") if s.name == "mixed" else s for s in streams
        )
    elif change == "commodity":
        streams = tuple(
            replace(s, commodity="oil") if s.name == "mixed" else s for s in streams
        )
    elif change == "physical_kind":
        streams = tuple(
            replace(s, physical_kind="thermal_power") if s.name == "mixed" else s
            for s in streams
        )
    elif change == "input_unit":
        streams = (replace(streams[0], unit="t/h"), *streams[1:])
    elif change == "unknown_input":
        streams = (replace(streams[0], input="missing"), *streams[1:])
    elif change == "input_collision":
        streams = (*streams, water("feed", upper=1))
    else:
        streams += (ProcessStream("energy", "MW", "thermal_power", "fuel", upper=1),)
        equipment += (
            StreamSource("supply", "energy"),
            StreamSink("consumer", "energy"),
        )
        totals = (StreamTotal("total", ("energy", "left")),)
    with pytest.raises(ValueError, match=message):
        replace(network, streams=streams, equipment=equipment, totals=totals)


def test_boiler_rejects_mass_flows_and_implicit_unit_conversion():
    for stream in (
        water("steam_a", upper=12),
        ProcessStream("steam_a", "kW", "thermal_power", "steam", upper=12000),
    ):
        network, _ = example()
        with pytest.raises(ValueError, match="boiler ports require"):
            replace(
                network,
                streams=tuple(
                    stream if s.name == stream.name else s for s in network.streams
                ),
            )


def test_shared_input_coefficients_are_combined_exactly_or_rejected():
    network, domain = mixing()
    shared = replace(
        network,
        streams=tuple(
            replace(s, value=None, input="feed") if s.name == "makeup" else s
            for s in network.streams
        ),
    )
    model = shared.compile()
    assert (
        next(
            t.coefficient
            for t in model.output("balance:mixer").terms
            if t.variable == "feed"
        )
        == -2
    )  # nosec B101
    assert (
        model.as_claim(domain).evaluate_result(domain.nominal).payload.feasibility
        == "feasible"
    )  # nosec B101
    thermal = BoxSet((Parameter("power", "MW", 1, 0, 1, 1, "Shared input"),))
    ambiguous = ProcessNetwork(
        "Shared input conversion",
        thermal.space,
        (
            ProcessStream("in", "MW", "thermal_power", "fuel", input="power"),
            ProcessStream("out", "MW", "thermal_power", "steam", input="power"),
            ProcessStream("reserve", "MW", "thermal_power", "steam", value=0),
        ),
        (
            StreamSource("supply", "in"),
            Boiler("b", "in", "out", 0.1),
            StreamSink("consumer", "out"),
            StreamSource("supply2", "reserve"),
            StreamSink("consumer2", "reserve"),
        ),
    )
    with pytest.raises(ValueError, match="cannot be represented exactly"):
        ambiguous.compile()


def test_custom_outputs_requirements_and_named_constraints_share_validation():
    network, domain = mixing()
    output = AffineOutput(
        "custom",
        "kg/s",
        "mass_flow",
        0,
        (AffineTerm("left", 2, "kg/s/kg/s"),),
        "Caller equation",
    )
    model = replace(
        network,
        outputs=(output,),
        requirements=(AffineRequirement("custom_service", "custom", 3, "ge"),),
    ).compile()
    assert (
        model.as_claim(domain).evaluate_result(domain.nominal).payload.feasibility
        == "feasible"
    )  # nosec B101
    with pytest.raises(ValueError, match="duplicate"):
        replace(network, outputs=(replace(output, name="stream:left"),)).compile()
    with pytest.raises(ValueError, match="duplicate"):
        replace(
            network,
            operating_limits=(AffineRequirement("balance:mixer:le", "delivered", 0),),
        ).compile()


def test_equipment_callbacks_are_not_declarative():
    class CustomMixer(Mixer):
        pass

    network, _ = mixing()
    with pytest.raises(ValueError, match="built-in declarative"):
        replace(
            network,
            equipment=tuple(
                CustomMixer("mixer", ("fresh", "makeup"), "mixed")
                if type(e) is Mixer
                else e
                for e in network.equipment
            ),
        )


def test_consumer_example_exports_source_and_checked_results():
    exported = run_example()
    source = ProcessNetwork.from_json(json.dumps(exported["network"]))
    assert source.compile().base_contract.model_id == "linear_process/v1"  # nosec B101
    for name, data in exported["results"].items():
        result = result_from_json(json.dumps(data))
        assert exported["verification"][name]["verified"]  # nosec B101
        assert verify_result(result).verified  # nosec B101
