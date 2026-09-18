"""One fuel actuator and an algebraic steam balance under three operating rules."""

import json

from operange import (
    AffineControlRule,
    AffineController,
    AffineTerm,
    Boiler,
    BoxSet,
    FiniteSet,
    LinearObjective,
    OperatingSpec,
    Parameter,
    ProcessNetwork,
    ProcessStream,
    Scenario,
    StreamSink,
    StreamSource,
    StreamTotal,
    verify_result,
)


def example():
    loads = BoxSet((Parameter("load", "MW", 3, 2, 4, 1, "Synthetic demand"),))
    network = ProcessNetwork(
        "Explicit fuel actuator and dependent steam flow",
        loads.space,
        streams=(
            ProcessStream("fuel", "MW", "thermal_power", "fuel", upper=10),
            ProcessStream("steam", "MW", "thermal_power", "steam", upper=5),
        ),
        equipment=(
            StreamSource("supply", "fuel"),
            Boiler("boiler", "fuel", "steam", efficiency=0.5),
            StreamSink("user", "steam", demand="load"),
        ),
        totals=(StreamTotal("fuel_use", ("fuel",)),),
        objective=LinearObjective("fuel_use", tolerance=1e-6),
        actuators=("fuel",),
    )
    cases = FiniteSet(
        loads.space,
        tuple(
            Scenario(str(load), {"load": load}, "Declared replay case")
            for load in (2, 3, 4)
        ),
    )
    return network, loads, cases


def run_example():
    network, loads, cases = example()
    model = network.compile()
    dispatch = model.as_claim(
        loads, operating=OperatingSpec.dispatch(model.controls, loads.space.names)
    )
    fixed = model.as_claim(
        loads, operating=OperatingSpec.fixed(model.controls, {"fuel": 8})
    )
    controller = AffineController(
        "Fuel follows load",
        (AffineControlRule("fuel", "MW", 0, (AffineTerm("load", 2, "MW/MW"),)),),
        "Declared feedforward rule; steam remains a solved state",
    )
    replay = model.as_claim(cases, operating=OperatingSpec.controlled(controller))
    reports = {
        "dispatch": dispatch.audit_result(),
        "fixed": fixed.audit_result(),
        "replay": replay.audit_result(),
    }
    return {
        "network": json.loads(network.to_json()),
        "roles": {
            q.quantity_id: q.role
            for q in model.base_contract.quantities
            if q.role in ("control", "state")
        },
        "results": {
            name: result.to_dict(compact=True) for name, result in reports.items()
        },
        "verified": {
            name: verify_result(result).verified for name, result in reports.items()
        },
        "frozen": json.loads(replay.freeze().to_json()),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2, allow_nan=False))
