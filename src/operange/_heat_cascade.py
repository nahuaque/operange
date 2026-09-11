"""Problem-table algorithm using exact rational values of supplied floats."""

from fractions import Fraction
from math import inf, nextafter

from .primitives import finite


def number(value):
    return finite(float(value), "heat-cascade response")


def upper(value):
    """Export a residual without rounding a violation down to its tolerance."""
    rounded = number(value)
    return finite(
        nextafter(rounded, inf) if Fraction(rounded) < value else rounded,
        "residual bound",
    )


def stream_conditions(model, point):
    conditions = []
    for stream in model.streams:
        values = tuple(
            point[stream.input_name(field)]
            for field in (
                "supply_temperature_c",
                "target_temperature_c",
                "heat_capacity_flow_kw_per_k",
            )
        )
        stream.validate_conditions(*values)
        conditions.append((stream, *(Fraction(v) for v in values)))
    return conditions


def cascade(model, conditions):
    shift = Fraction(model.delta_t_min_k) / 2
    active, duties = [], {"hot": Fraction(0), "cold": Fraction(0)}
    for stream, supply, target, cp in conditions:
        duties[stream.kind] += cp * abs(supply - target)
        if cp and supply != target:
            offset = -shift if stream.kind == "hot" else shift
            active.append(
                (stream, min(supply, target) + offset, max(supply, target) + offset, cp)
            )
    temperatures = sorted(
        {t for _, lo, hi, _ in active for t in (lo, hi)}, reverse=True
    )
    unadjusted, intervals = [Fraction(0)], []
    for index, (hi, lo) in enumerate(zip(temperatures, temperatures[1:])):
        members = [(s, cp) for s, low, high, cp in active if low <= lo and high >= hi]
        net_cp = sum((cp if s.kind == "hot" else -cp for s, cp in members), Fraction(0))
        heat = net_cp * (hi - lo)
        unadjusted.append(unadjusted[-1] + heat)
        intervals.append(
            {
                "upper_node": index,
                "lower_node": index + 1,
                "hot_streams": tuple(s.name for s, _ in members if s.kind == "hot"),
                "cold_streams": tuple(s.name for s, _ in members if s.kind == "cold"),
                "net_heat_capacity_flow_kw_per_k": number(net_cp),
                "net_heat_capacity_flow_exact_kw_per_k": str(net_cp),
                "interval_heat_kw": number(heat),
                "interval_heat_exact_kw": str(heat),
            }
        )
    hot = -min(unadjusted)
    adjusted = [v + hot for v in unadjusted]
    cold = adjusted[-1]

    def temperature_pair(t):
        return {
            "shifted_temperature_c": number(t),
            "shifted_temperature_exact_c": str(t),
            "hot_temperature_c": number(t + shift),
            "hot_temperature_exact_c": str(t + shift),
            "cold_temperature_c": number(t - shift),
            "cold_temperature_exact_c": str(t - shift),
        }

    targets = {
        "minimum_hot_utility": hot,
        "minimum_cold_utility": cold,
        "maximum_heat_recovery": duties["cold"] - hot,
        "hot_stream_duty": duties["hot"],
        "cold_stream_duty": duties["cold"],
    }
    details = {
        "scope": "thermodynamic_utility_targets",
        "installed_network_feasibility": "not_assessed",
        "continuous_uncertainty_coverage": False,
        "delta_t_min_k": model.delta_t_min_k,
        "targets_exact_kw": {k: str(v) for k, v in targets.items()},
        "nodes": tuple(
            {
                **temperature_pair(t),
                "unadjusted_heat_kw": number(raw),
                "unadjusted_heat_exact_kw": str(raw),
                "heat_kw": number(heat),
                "heat_exact_kw": str(heat),
            }
            for t, raw, heat in zip(temperatures, unadjusted, adjusted)
        ),
        "intervals": intervals,
        "pinch_points": tuple(
            {
                **temperature_pair(t),
                "location": "upper_terminal"
                if i == 0
                else "lower_terminal"
                if i == len(temperatures) - 1
                else "interior",
            }
            for i, (t, heat) in enumerate(zip(temperatures, adjusted))
            if heat == 0
        ),
        "pinch_intervals": tuple(
            {
                "upper": temperature_pair(hi),
                "lower": temperature_pair(lo),
            }
            for i, (hi, lo) in enumerate(zip(temperatures, temperatures[1:]))
            if adjusted[i] == adjusted[i + 1] == 0
        ),
    }
    return targets, details
