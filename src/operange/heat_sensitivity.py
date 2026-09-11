"""Analytical first derivatives of the optimized constant-COP response."""

from .contract_types import (
    Derivative,
    Diagnostic,
    Evidence,
    SensitivityPayload,
    snapshot,
)
from .engineering_results import SensitivityResult
from .heat_recovery import COORDINATES
from .heat_results import OUTPUTS, contract_for, evaluate_result
from .primitives import finite
from fractions import Fraction


def _sensitivity_result(
    claim,
    realization,
    *,
    inputs=COORDINATES,
    outputs=(
        "delivered_heat",
        "required_heat",
        "delivery_margin",
        "requirement_residual",
    ),
    operator="jacobian",
    coordinate_space="physical",
    response="optimized",
    side="two_sided",
    direction=None,
    method="analytical",
):
    request = snapshot(
        {
            "query": "sensitivity",
            "realization": realization,
            "inputs": inputs,
            "outputs": outputs,
            "operator": operator,
            "coordinate_space": coordinate_space,
            "response": response,
            "side": side,
            "direction": direction,
            "method": method,
        }
    )
    contract = contract_for(claim)
    response_contract = {
        "mode": response,
        "optimization": "maximize_delivered_heat" if response == "optimized" else None,
        "policy_ref": None,
        "control_treatment": "reoptimized at each disturbance"
        if response == "optimized"
        else "unsupported",
    }

    def result(
        execution,
        availability,
        code=None,
        message=None,
        *,
        base=None,
        derivative=None,
        evidence=(),
        validity=None,
    ):
        return SensitivityResult(
            contract,
            request,
            execution,
            SensitivityPayload(
                availability,
                base.ref if base else None,
                snapshot(response_contract),
                derivative,
                {"name": method, "operator": operator},
                validity or {},
            ),
            evidence,
            (Diagnostic(code, "sensitivity", message),) if code else (),
            (base,) if base else (),
        )

    if (
        response not in ("optimized",)
        or method != "analytical"
        or operator not in ("jacobian", "directional")
    ):
        return result(
            "unsupported",
            "not_evaluated",
            "unsupported_sensitivity",
            "This adapter provides analytical Jacobians and directional derivatives of the optimized response only.",
        )
    try:
        if coordinate_space not in ("physical", "normalized") or side not in (
            "two_sided",
            "forward",
            "backward",
        ):
            raise ValueError("unknown coordinate space or derivative side")
        if (
            not isinstance(inputs, (tuple, list))
            or not inputs
            or any(n not in COORDINATES for n in inputs)
            or len(set(inputs)) != len(inputs)
        ):
            raise ValueError("inputs must be distinct declared disturbance coordinates")
        if (
            not isinstance(outputs, (tuple, list))
            or not outputs
            or any(n not in OUTPUTS for n in outputs)
            or len(set(outputs)) != len(outputs)
        ):
            raise ValueError("outputs must be distinct supported output coordinates")
        if operator == "directional":
            if not isinstance(direction, (tuple, list)) or len(direction) != len(
                inputs
            ):
                raise ValueError("direction must have one entry per selected input")
            direction = tuple(finite(v, "direction") for v in direction)
        elif direction is not None:
            raise ValueError("a Jacobian request cannot include a direction")
        point = claim.uncertainty.coordinates(realization)
        if not claim.uncertainty.contains(point):
            raise ValueError("realization is outside the declared domain")
    except (ValueError, TypeError, KeyError) as exc:
        return result("invalid", "not_evaluated", "invalid_sensitivity", str(exc))
    base = evaluate_result(claim, point)
    if base.execution != "completed":
        return result(
            "unresolved",
            "unknown",
            "base_unresolved",
            "The base optimized response could not be verified.",
            base=base,
        )
    params = [claim.uncertainty.parameter(n) for n in inputs]
    factors = [p.scale if coordinate_space == "normalized" else 1.0 for p in params]
    # Each column is a perturbation in physical coordinates. A directional
    # derivative uses t in R (dimensionless), with x(t) = x0 + t * direction.
    vectors = (
        [tuple(_product(direction[i], factors[i]) for i in range(len(inputs)))]
        if operator == "directional"
        else [
            tuple(factors[i] if i == j else 0.0 for i in range(len(inputs)))
            for j in range(len(inputs))
        ]
    )
    validity = {
        "scope": "local",
        "domain": "declared box",
        "active_limits": [
            e.constraint_ref
            for e in base.payload.constraint_checks
            if e.constraint_ref in ("compressor_capacity", "source_capacity")
            and abs(e.residual.value) <= claim.tolerance
        ],
        "branch_tolerance_mw": claim.tolerance,
        "direction_parameter_unit": "1" if operator == "directional" else None,
    }
    for vector in vectors:
        for parameter, delta in zip(params, vector):
            for sign in (
                (-1, 1) if side == "two_sided" else (1,) if side == "forward" else (-1,)
            ):
                if (
                    delta * sign < 0
                    and point[parameter.name] == parameter.lower
                    or delta * sign > 0
                    and point[parameter.name] == parameter.upper
                ):
                    return result(
                        "completed",
                        "undefined",
                        "outside_tangent_domain",
                        "The requested perturbation leaves the declared domain arbitrarily close to the base point.",
                        base=base,
                        validity=validity,
                    )
    source_coefficient = claim.cop / (claim.cop - 1) * claim.design.source_capacity_mw
    source_bound = source_coefficient * (1 - point["source_derating"])
    power_bound = claim.cop * claim.design.power_capacity_mw
    difference = source_bound - power_bound
    columns = []
    for vector in vectors:
        deltas = dict(zip(inputs, vector))
        source_slope = _product(-source_coefficient, deltas.get("source_derating", 0.0))
        demand_slope = _product(
            claim.requirement.base_heat_mw, deltas.get("demand_increase", 0.0)
        )
        affected = source_slope != 0 and any(n != "required_heat" for n in outputs)
        if affected and difference != 0 and abs(difference) <= claim.tolerance:
            return result(
                "unresolved",
                "unknown",
                "active_set_ambiguous",
                "The two capacity bounds differ by less than the declared tolerance; a branch derivative is not asserted.",
                base=base,
                validity=validity,
            )
        if difference == 0:
            if affected and side == "two_sided":
                return result(
                    "completed",
                    "undefined",
                    "nondifferentiable_response",
                    "The requested response crosses the compressor/source capacity kink; request a one-sided derivative.",
                    base=base,
                    validity=validity,
                )
            dq = (
                min(0.0, source_slope) if side != "backward" else max(0.0, source_slope)
            )
        else:
            dq = source_slope if difference < 0 else 0.0
        responses = {
            "delivered_heat": dq,
            "required_heat": demand_slope,
            "delivery_margin": dq - demand_slope,
            "requirement_residual": demand_slope - dq,
            "electrical_power": dq / claim.cop,
            "source_heat": dq * (claim.cop - 1) / claim.cop,
        }
        columns.append(tuple(responses[n] for n in outputs))
    output_units = tuple(contract.quantity(n).unit for n in outputs)
    input_units = tuple(
        contract.quantity(n).unit if coordinate_space == "physical" else "1"
        for n in inputs
    )
    values = (
        tuple(tuple(col[i] for col in columns) for i in range(len(outputs)))
        if operator == "jacobian"
        else columns[0]
    )
    units = (
        tuple(tuple(f"{out}/{inp}" for inp in input_units) for out in output_units)
        if operator == "jacobian"
        else tuple((out,) for out in output_units)
    )
    derivative = Derivative(
        operator,
        tuple(outputs),
        tuple(inputs),
        coordinate_space,
        values,
        units,
        input_units,
        output_units,
        tuple(p.nominal if coordinate_space == "normalized" else 0 for p in params),
        tuple(factors),
        side,
        direction,
    )
    evidence = (
        Evidence(
            "derivative",
            "derivative",
            "analytical_piecewise_affine",
            "verified",
            assumptions=(
                "constant COP",
                "fully observed optimized static response",
                "one-sided derivatives use the stated perturbation side",
            ),
            details={
                "capacity_formula": "min(COP*power_capacity, COP/(COP-1)*source_capacity*(1-source_derating))",
                "physical_source_slope": -source_coefficient,
                "physical_demand_slope": claim.requirement.base_heat_mw,
            },
        ),
    )
    return result(
        "completed",
        "available",
        base=base,
        derivative=derivative,
        evidence=evidence,
        validity=validity,
    )


def sensitivity_result(claim, realization, **options):
    """Keep finite-input arithmetic failures within the portable result contract."""
    try:
        return _sensitivity_result(claim, realization, **options)
    except (ArithmeticError, ValueError) as exc:
        return SensitivityResult(
            contract_for(claim),
            snapshot({"query": "sensitivity", "realization": realization, **options}),
            "unresolved",
            SensitivityPayload("unknown"),
            diagnostics=(
                Diagnostic("derivative_arithmetic_unresolved", "sensitivity", str(exc)),
            ),
        )


def _product(*values):
    exact = Fraction(1)
    for value in values:
        exact *= Fraction(value)
    result = finite(float(exact), "derivative product")
    if exact and result == 0:
        raise ValueError("derivative product underflowed")
    return result
