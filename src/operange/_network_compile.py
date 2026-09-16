"""Compile named stream connections into ordinary affine physical constraints."""

from fractions import Fraction as F

from .affine import AffineOutput, AffineRequirement, AffineTerm
from .contract_types import unique
from .linear_process import LinearControl, LinearProcessAdapter
from .network import (
    Boiler,
    FixedRatioSplitter,
    Mixer,
    ProcessStream,
    StreamSink,
    StreamSource,
    StreamTotal,
    UtilityHeader,
)
from .primitives import finite


def ports(item):
    if type(item) is StreamSource:
        return (), (item.outlet,)
    if type(item) is StreamSink:
        return (item.inlet,), ()
    if type(item) is Boiler:
        return (item.fuel,), (item.steam,)
    if type(item) is Mixer:
        return item.inlets, (item.outlet,)
    if type(item) is FixedRatioSplitter:
        return (item.inlet,), item.outlets
    if type(item) is UtilityHeader:
        return item.inlets, item.outlets
    raise ValueError("network equipment must use built-in declarative types")


def compatible(streams, names):
    selected = [streams[n] for n in names]
    if len({(s.unit, s.physical_kind, s.commodity) for s in selected}) != 1:
        raise ValueError(
            "conserving ports and totals require matching unit, physical kind and commodity"
        )


def validate_network(network):
    if not network.streams or not network.equipment:
        raise ValueError("network requires streams and equipment")
    if any(type(s) is not ProcessStream for s in network.streams) or any(
        type(t) is not StreamTotal for t in network.totals
    ):
        raise ValueError("network requires built-in declarative streams and totals")
    unique(tuple(s.name for s in network.streams), "stream names")
    unique(tuple(e.name for e in network.equipment), "equipment names")
    unique(tuple(t.name for t in network.totals), "total names")
    streams = {s.name: s for s in network.streams}
    producers, consumers = {}, {}

    def input_unit(name, unit):
        if name not in network.input_space.names:
            raise ValueError(f"unknown network input: {name}")
        if network.input_space.coordinate(name).unit != unit:
            raise ValueError(f"network input {name} must have stream unit {unit}")

    for stream in network.streams:
        if stream.input is not None:
            input_unit(stream.input, stream.unit)
        elif stream.name in network.input_space.names:
            raise ValueError("bounded stream names must be distinct from input names")
    for item in network.equipment:
        incoming, outgoing = ports(item)
        for names, owners, role in (
            (incoming, consumers, "consumer"),
            (outgoing, producers, "producer"),
        ):
            for name in names:
                if name not in streams:
                    raise ValueError(f"{item.name}: unknown stream {name}")
                if name in owners:
                    raise ValueError(f"stream {name} has more than one {role}")
                owners[name] = item.name
        if type(item) is Boiler:
            fuel, steam = streams[item.fuel], streams[item.steam]
            if fuel.unit != steam.unit or any(
                s.physical_kind != "thermal_power" for s in (fuel, steam)
            ):
                raise ValueError(
                    "boiler ports require thermal_power streams with matching units"
                )
        else:
            compatible(streams, (*incoming, *outgoing))
        if type(item) is StreamSink and item.demand is not None:
            input_unit(item.demand, streams[item.inlet].unit)
    for name in streams:
        if name not in producers or name not in consumers:
            raise ValueError(
                f"stream {name} needs exactly one producer and one consumer"
            )
    for total in network.totals:
        if not set(total.streams) <= streams.keys():
            raise ValueError(f"{total.name}: unknown stream in total")
        compatible(streams, total.streams)


def compile_network(network):
    streams = {s.name: s for s in network.streams}
    controls, outputs, limits, services = [], [], [], []
    units = {c.name: c.unit for c in network.input_space.coordinates}
    for stream in sorted(network.streams, key=lambda s: s.name):
        if stream.input is None:
            lower = stream.lower if stream.value is None else stream.value
            upper = stream.upper if stream.value is None else stream.value
            controls.append(
                LinearControl(
                    stream.name, stream.unit, lower, upper, stream.physical_kind
                )
            )
            units[stream.name] = stream.unit

    def expression(name, rates, provenance, *, input_terms=()):
        """Keep declared float coefficients exact, including shared input aliases."""
        stream = streams[rates[0][0]]
        terms = {}
        for key, coefficient in rates:
            variable = streams[key].input or key
            terms[variable] = terms.get(variable, F(0)) + F(coefficient)
        for variable, coefficient in input_terms:
            terms[variable] = terms.get(variable, F(0)) + F(coefficient)
        compiled = []
        for variable, coefficient in sorted(terms.items()):
            if not coefficient:
                continue
            try:
                value = finite(float(coefficient), "combined stream coefficient")
            except OverflowError as exc:
                raise ValueError("combined stream coefficient is not finite") from exc
            if F(value) != coefficient:
                raise ValueError(
                    f"{name}: combined coefficient for {variable} cannot be represented exactly; use distinct bounded stream variables"
                )
            compiled.append(
                AffineTerm(variable, value, f"{stream.unit}/{units[variable]}")
            )
        outputs.append(
            AffineOutput(
                name, stream.unit, stream.physical_kind, 0, tuple(compiled), provenance
            )
        )
        return name

    def balance(name, rates, provenance):
        output = expression(name, rates, provenance)
        for relation in ("le", "ge"):
            limits.append(
                AffineRequirement(
                    f"{name}:{relation}",
                    output,
                    0,
                    relation,
                    network.balance_tolerance,
                    network.balance_scale,
                )
            )

    origin = f"Network {network.name}: {network.provenance}"
    for stream in sorted(network.streams, key=lambda s: s.name):
        name = expression(
            stream.quantity_id,
            [(stream.name, 1)],
            f"{origin}; stream {stream.name} ({stream.commodity}): {stream.provenance}",
        )
        if stream.input is not None:
            for side, value, relation in (
                ("lower", stream.lower, "ge"),
                ("upper", stream.upper, "le"),
            ):
                if value is not None:
                    limits.append(
                        AffineRequirement(
                            f"stream_bound:{stream.name}:{side}",
                            name,
                            value,
                            relation,
                            0,
                        )
                    )
    for item in sorted(network.equipment, key=lambda e: e.name):
        incoming, outgoing = ports(item)
        provenance = f"{origin}; {item.kind} {item.name}: {item.provenance}"
        if type(item) is StreamSource:
            continue
        if type(item) is StreamSink:
            name = expression(
                f"delivery:{item.name}",
                [(item.inlet, 1)],
                provenance,
                input_terms=() if item.demand is None else ((item.demand, -1),),
            )
            services.append(
                AffineRequirement(
                    item.requirement_id,
                    name,
                    item.minimum,
                    "ge",
                    item.tolerance,
                    item.residual_scale,
                )
            )
            if not item.allow_excess:
                limits.append(
                    AffineRequirement(
                        f"no_excess:{item.name}",
                        name,
                        item.minimum,
                        "le",
                        item.tolerance,
                        item.residual_scale,
                    )
                )
        elif type(item) is Boiler:
            balance(
                f"balance:{item.name}",
                [(item.steam, 1), (item.fuel, -item.efficiency)],
                provenance + "; steam = efficiency * fuel",
            )
            expression(
                f"loss:{item.name}",
                [(item.fuel, 1), (item.steam, -1)],
                provenance + "; loss = fuel - steam",
            )
        else:
            balance(
                f"balance:{item.name}",
                [(n, 1) for n in outgoing] + [(n, -1) for n in incoming],
                provenance + "; total outlet rate = total inlet rate",
            )
            if type(item) is FixedRatioSplitter:
                weights = dict(zip(item.outlets, item.weights))
                pivot = min(item.outlets, key=lambda n: (-weights[n], n))
                for name in sorted(set(item.outlets) - {pivot}):
                    balance(
                        f"ratio:{item.name}:{name}",
                        [(name, weights[pivot]), (pivot, -weights[name])],
                        provenance
                        + f"; {weights[pivot]} * {name} = {weights[name]} * {pivot}",
                    )
    for total in sorted(network.totals, key=lambda t: t.name):
        expression(
            total.name,
            [(n, 1) for n in total.streams],
            f"{origin}; total {total.name}: {total.provenance}",
        )
    return LinearProcessAdapter(
        network.name,
        network.input_space,
        tuple(outputs) + network.outputs,
        tuple(services) + network.requirements,
        tuple(controls),
        tuple(limits) + network.operating_limits,
        network.solver_tolerance,
        network.objective,
        network.measurements,
    )
