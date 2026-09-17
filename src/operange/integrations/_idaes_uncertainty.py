"""Local responses using source derivatives or checked upstream solver calls."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from ..contract_types import Diagnostic, unique
from ..domains import Coordinate, ParameterSpace, FiniteSet, Scenario
from ..parameters import ParameterPoint, ParameterCovariance, covariance_matrix
from ..primitives import finite
from ._idaes_derivatives import _regular_system
from ._idaes_model import component, number
from ._idaes_results import evaluate_point
from ._idaes_shared_result import fixed_policy
from ._idaes_uncertainty_result import (
    IdaesUncertainParameter,
    IdaesResponse,
    IdaesPerturbation,
    IdaesUncertaintyResult,
)
from ._idaes_parameters import parameter_model


def _evaluate(adapter, point, commands):
    domain = FiniteSet(
        adapter.input_space, (Scenario("local", point, "Local operating point"),)
    )
    policy = (
        None
        if commands is None
        else fixed_policy(adapter.base_contract.model["controls"], commands)
    )
    return evaluate_point(adapter.as_claim(domain, recourse=policy), point)


def _expressions(model, responses, rows):
    return tuple(
        -rows[r.name.removeprefix("margin:")].expression(model)
        if r.kind == "margin"
        else component(model, r.component)
        for r in responses
    )


def _active(model, rows, tolerance):
    return tuple(
        r.name
        for r in rows.values()
        if r.side != "eq"
        and abs(number(r.expression(model))) <= max(r.tolerance, tolerance)
    )


def propagate_uncertainty(
    adapter,
    *,
    parameters,
    covariance,
    realization,
    outputs,
    margins,
    fixed_controls,
    reoptimize,
    steps,
    relative_tolerance,
    absolute_tolerance,
    provenance=None,
):
    from .idaes import IdaesVariable

    parameters = tuple(parameters)
    if not parameters or not all(isinstance(p, IdaesVariable) for p in parameters):
        raise ValueError("Supply a nonempty ordered tuple of IdaesVariable parameters")
    unique(tuple(p.name for p in parameters), "parameter names")
    space = ParameterSpace(
        tuple(
            Coordinate(p.name, p.unit, "Uncertain model parameter") for p in parameters
        )
    )
    if isinstance(covariance, ParameterCovariance):
        provenance = {
            **(provenance or {}),
            "parameter_covariance": covariance.to_dict(),
        }
        covariance = covariance.in_space(space)
    covariance = covariance_matrix(covariance, len(parameters))
    point = adapter.input_space.validate(
        adapter.input_space.nominal if realization is None else realization
    )
    theta = ParameterPoint(
        space,
        {p.name: number(component(adapter._template, p.component)) for p in parameters},
    )
    mapping = {p.name: p.component for p in parameters}
    _, updates = parameter_model(adapter, theta, mapping)
    parameters = tuple(
        IdaesUncertainParameter(u.parameter, u.component, u.unit, u.value)
        for u in updates
    )
    if type(reoptimize) is not bool:
        raise ValueError("reoptimize must be a bool")
    relative_tolerance = finite(relative_tolerance, "relative tolerance")
    if not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must lie between zero and one")
    absolute_tolerance = finite(absolute_tolerance, "absolute derivative tolerance")
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be nonnegative")
    if reoptimize:
        if fixed_controls is not None:
            raise ValueError("Choose fixed_controls or reoptimize=True, not both")
        if len(adapter.base_contract.model["source"]["objectives"]) != 1:
            raise ValueError("Reoptimization requires one active source objective")
        if not isinstance(steps, Mapping) or set(steps) != {p.name for p in parameters}:
            raise ValueError("Supply explicit native-unit steps for every parameter")
        steps = {k: finite(v, "perturbation step") for k, v in steps.items()}
        if any(v <= 0 for v in steps.values()):
            raise ValueError("Perturbation steps must be positive")
        if any(
            p.value + steps[p.name] / 2 == p.value
            or p.value - steps[p.name] / 2 == p.value
            for p in parameters
        ):
            raise ValueError("Perturbation steps are below floating-point resolution")
        commands = None
    else:
        if steps is not None:
            raise ValueError("Fixed-command implicit derivatives do not use steps")
        steps = {}
        if fixed_controls is None and not adapter.controls:
            fixed_controls = {}
        if not isinstance(fixed_controls, Mapping) or set(fixed_controls) != {
            c.name for c in adapter.controls
        }:
            raise ValueError(
                "Supply fixed_controls for every actuator or explicitly request reoptimize=True"
            )
        commands = {k: finite(v, "fixed command") for k, v in fixed_controls.items()}
    outputs = (
        tuple(v.name for v in adapter.outputs) if outputs is None else tuple(outputs)
    )
    margins = tuple(margins)
    unique(outputs, "selected outputs")
    unique(margins, "selected margins")
    bindings = {v.name: v for v in adapter.outputs + adapter.controls}
    rows = {r.name: r for r in adapter._rows}
    if set(outputs) - set(bindings):
        raise ValueError("Outputs must name declared outputs or actuator controls")
    if any(name not in rows or rows[name].side == "eq" for name in margins):
        raise ValueError("Margins must name source inequalities or variable bounds")
    responses = tuple(
        IdaesResponse(
            name,
            bindings[name].component,
            bindings[name].unit,
            "control" if name in {v.name for v in adapter.controls} else "output",
        )
        for name in outputs
    ) + tuple(
        IdaesResponse(
            f"margin:{name}",
            rows[name].path,
            rows[name].unit,
            "margin",
            rows[name].side,
        )
        for name in margins
    )
    if not responses:
        raise ValueError("Select at least one output or margin")
    unique(tuple(r.name for r in responses), "response names")
    baseline, operating = _evaluate(adapter, point, commands)
    values, jacobian, probes, diagnostics, numerical = (), (), [], [], {}
    try:
        model = operating.require_checked().model
        exprs = _expressions(model, responses, rows)
        values = tuple(number(expr) for expr in exprs)
        active = _active(model, rows, adapter.tolerance)
        numerical["active_limits"] = active
        if active:
            diagnostics.append(
                Diagnostic(
                    "active_operating_limits",
                    "baseline",
                    "Local derivatives meet active bounds or inequalities; extrapolation may leave the feasible region. Inspect numerical.active_limits.",
                )
            )
        if reoptimize:
            termination = next(
                e.details.get("termination")
                for e in baseline.evidence
                if e.evidence_id == "solver"
            )
            if termination not in {"optimal", "locallyOptimal", "globallyOptimal"}:
                raise ValueError(
                    "Reoptimization needs an optimal/local-optimal baseline termination"
                )
            matrix = _finite_differences(
                adapter,
                point,
                parameters,
                theta,
                mapping,
                responses,
                rows,
                values,
                active,
                steps,
                relative_tolerance,
                absolute_tolerance,
                probes,
                numerical,
            )
        else:
            gradient, response, condition = _regular_system(
                model, tuple(p.component for p in parameters)
            )
            nstates = response.shape[0]
            matrix = np.asarray(
                [
                    g[nstates:] + g[:nstates] @ response
                    for g in (gradient(expr) for expr in exprs)
                ]
            )
            numerical["state_condition"] = condition
            numerical["state_regularity"] = (
                "Square equality-state Jacobian, magnitude equilibrated; condition <= 1e10."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Nonfinite response Jacobian")
        if not np.all(np.isfinite(matrix @ covariance @ matrix.T)):
            raise ValueError("Nonfinite propagated covariance")
        jacobian = tuple(tuple(float(x) for x in row) for row in matrix)
    except Exception as exc:
        diagnostics.append(
            Diagnostic(
                "local_propagation_unresolved",
                "responses",
                f"{type(exc).__name__}: {exc}",
            )
        )
    return IdaesUncertaintyResult(
        parameters=parameters,
        covariance=tuple(map(tuple, covariance.tolist())),
        responses=responses,
        mode="reoptimized" if reoptimize else "fixed",
        method="finite_difference" if reoptimize else "implicit",
        execution="completed" if jacobian else "unresolved",
        baseline=baseline,
        values=values,
        jacobian=jacobian,
        steps=steps,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        perturbations=tuple(probes),
        numerical=numerical,
        provenance=provenance or {},
        diagnostics=tuple(diagnostics),
    )


def _finite_differences(
    adapter,
    point,
    parameters,
    theta,
    mapping,
    responses,
    rows,
    values,
    active,
    steps,
    rtol,
    atol,
    probes,
    numerical,
):
    columns, failures, comparisons = [], [], []
    for parameter in parameters:
        samples = []
        h = steps[parameter.name]
        for factor in (-1, 1, -0.5, 0.5):
            offset = factor * h
            evaluation = None
            errors = ()
            try:
                perturbed = replace(
                    theta,
                    values={
                        **dict(theta.values),
                        parameter.name: parameter.value + offset,
                    },
                )
                model, _ = parameter_model(adapter, perturbed, mapping)
                candidate = replace(adapter, model=model)
                evaluation, operating = _evaluate(candidate, point, None)
                source = operating.require_checked().model
                termination = next(
                    e.details.get("termination")
                    for e in evaluation.evidence
                    if e.evidence_id == "solver"
                )
                if termination not in {"optimal", "locallyOptimal", "globallyOptimal"}:
                    raise ValueError(
                        "Reoptimization needs an optimal/local-optimal solver termination, not feasibility alone"
                    )
                if _active(source, rows, adapter.tolerance) != active:
                    raise ValueError(
                        "Active constraints change across perturbations; a single local derivative is unresolved"
                    )
                samples.append(
                    np.asarray(
                        [number(e) for e in _expressions(source, responses, rows)]
                    )
                )
            except Exception as exc:
                errors = (
                    Diagnostic(
                        "perturbation_unresolved",
                        parameter.name,
                        f"{type(exc).__name__}: {exc}",
                    ),
                )
                failures.append(f"{parameter.name} at {offset:g}: {exc}")
                samples.append(None)
            probes.append(IdaesPerturbation(parameter.name, offset, evaluation, errors))
        if any(s is None for s in samples):
            continue
        coarse = (samples[1] - samples[0]) / (2 * h)
        fine = (samples[3] - samples[2]) / h
        left = (np.asarray(values) - samples[2]) / (h / 2)
        right = (samples[3] - np.asarray(values)) / (h / 2)
        error = np.maximum(np.abs(coarse - fine), np.abs(left - right))
        threshold = (
            rtol
            * np.maximum.reduce(
                [np.abs(coarse), np.abs(fine), np.abs(left), np.abs(right)]
            )
            + atol
        )
        comparisons.append(
            {
                "parameter": parameter.name,
                "coarse": coarse.tolist(),
                "fine": fine.tolist(),
                "left": left.tolist(),
                "right": right.tolist(),
                "error": error.tolist(),
                "threshold": threshold.tolist(),
                "absolute_derivative_tolerance": atol,
            }
        )
        if np.any(error > threshold):
            failures.append(
                f"{parameter.name}: step refinement or left/right derivatives disagree"
            )
        columns.append(fine)
    numerical["derivative_checks"] = comparisons
    if failures:
        raise ValueError("; ".join(failures))
    return np.asarray(columns).T
