"""Connected boilers and a steam header compiled to the ordinary linear model."""

from dataclasses import replace
import json

from operange import (
    AffineRequirement,
    Boiler,
    BoxSet,
    FixedRatioSplitter,
    LinearObjective,
    Mixer,
    Parameter,
    ProcessNetwork,
    ProcessStream,
    StreamSink,
    StreamSource,
    StreamTotal,
    UtilityHeader,
    verify_result,
)


def example():
    loads = BoxSet(
        (
            Parameter("dryer", "MW", 10, 8, 12, 2, "Synthetic dryer demand"),
            Parameter("evaporator", "MW", 6, 4, 8, 2, "Synthetic evaporator demand"),
        )
    )
    network = ProcessNetwork(
        "Two boilers sharing a fuel supply",
        loads.space,
        streams=(
            ProcessStream("fuel_a", "MW", "thermal_power", "fuel", upper=40),
            ProcessStream("fuel_b", "MW", "thermal_power", "fuel", upper=40),
            ProcessStream("steam_a", "MW", "thermal_power", "steam", upper=12),
            ProcessStream("steam_b", "MW", "thermal_power", "steam", upper=10),
            ProcessStream("to_dryer", "MW", "thermal_power", "steam", upper=20),
            ProcessStream("to_evaporator", "MW", "thermal_power", "steam", upper=20),
        ),
        equipment=(
            StreamSource("fuel_supply_a", "fuel_a"),
            StreamSource("fuel_supply_b", "fuel_b"),
            Boiler("boiler_a", "fuel_a", "steam_a", efficiency=0.8),
            Boiler("boiler_b", "fuel_b", "steam_b", efficiency=0.5),
            UtilityHeader(
                "steam_header", ("steam_a", "steam_b"), ("to_dryer", "to_evaporator")
            ),
            StreamSink("dryer", "to_dryer", demand="dryer", allow_excess=False),
            StreamSink(
                "evaporator", "to_evaporator", demand="evaporator", allow_excess=False
            ),
        ),
        totals=(StreamTotal("fuel", ("fuel_a", "fuel_b")),),
        operating_limits=(AffineRequirement("shared_fuel", "fuel", 32),),
        objective=LinearObjective("fuel", tolerance=1e-6),
        provenance="Synthetic steady-state utility system; steam is thermal power",
    )
    return network, loads


def material_example():
    feed = BoxSet((Parameter("feed", "kg/s", 4, 2, 6, 2, "Fresh water feed"),))
    network = ProcessNetwork(
        "Water mixing and splitting",
        feed.space,
        streams=(
            ProcessStream("fresh", "kg/s", "mass_flow", "water", input="feed"),
            ProcessStream("makeup", "kg/s", "mass_flow", "water", value=2),
            ProcessStream("mixed", "kg/s", "mass_flow", "water", upper=10),
            ProcessStream("left", "kg/s", "mass_flow", "water", upper=10),
            ProcessStream("right", "kg/s", "mass_flow", "water", upper=10),
        ),
        equipment=(
            StreamSource("fresh_supply", "fresh"),
            StreamSource("makeup_supply", "makeup"),
            Mixer("mixer", ("fresh", "makeup"), "mixed"),
            FixedRatioSplitter("split", "mixed", ("left", "right"), weights=(1, 3)),
            StreamSink("left_user", "left"),
            StreamSink("right_user", "right"),
        ),
        totals=(StreamTotal("delivered", ("left", "right")),),
    )
    return network, feed


def run_example():
    source, loads = example()
    restored = ProcessNetwork.from_json(source.to_json())
    model = restored.compile()
    claim = model.as_claim(loads)
    restricted = replace(
        source, operating_limits=(AffineRequirement("shared_fuel", "fuel", 30),)
    )
    water, feed = material_example()
    material = water.compile().as_claim(feed)
    results = {
        "nominal": claim.evaluate_result(loads.nominal),
        "continuous_audit": claim.audit_result(),
        "restricted_fuel": restricted.compile().as_claim(loads).audit_result(),
        "material_nominal": material.evaluate_result(feed.nominal),
        "material_audit": material.audit_result(),
    }
    return {
        "network": json.loads(source.to_json()),
        "material_network": json.loads(water.to_json()),
        "results": {name: r.to_dict(compact=True) for name, r in results.items()},
        "verification": {
            name: verify_result(r).to_dict() for name, r in results.items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))
