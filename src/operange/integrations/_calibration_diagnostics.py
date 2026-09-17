"""Local information and held-out predictions using existing source models."""

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from ..contract_types import Diagnostic
from ..primitives import finite
from ._idaes_model import backend, component, number, uid
from . import _idaes_results
from ._parmest_backend import _check_experiment
from ._calibration_quality import (
    CalibrationDiagnostics,
    CalibrationSensitivity,
    CalibrationValidation,
    CalibrationPrediction,
)


def _fit_matches(study, fitted):
    from .parmest import CalibrationResult

    if not isinstance(fitted, CalibrationResult) or fitted.study != study.manifest:
        raise ValueError(
            "The fitted result must belong to this exact calibration study"
        )


def _fixed_parameters(snapshot, estimates):
    pyo, _ = backend()
    model = snapshot.model.clone()
    for name, value in estimates.items():
        target = component(model, name)
        if target.ctype is pyo.Var:
            target.set_value(value, skip_validation=True)
            target.fix()
        else:
            target.set_value(value)
    for name in estimates:
        target = component(model, name)
        if target.ctype is pyo.Var and (
            (target.lower is not None and number(target) < number(target.lower))
            or (target.upper is not None and number(target) > number(target.upper))
        ):
            raise ValueError(f"Fitted parameter violates experiment bounds: {name}")
    return model


def _returned(snapshot, values):
    return {
        group: values[paths[0]] if len(paths) == 1 else [values[p] for p in paths]
        for group, paths in snapshot.variable_groups
    }


def _regular_system(model, parameters):
    """Differentiate equations with Pyomo; do not use measurement fit objectives.

    Row/column equilibration reduces native-unit effects in the state Jacobian.
    A local square nonsingular state system is an explicit supported boundary.
    """
    pyo, _ = backend()
    from pyomo.core.expr.calculus.derivatives import differentiate, Modes
    from pyomo.core.expr.visitor import identify_variables

    states = tuple(
        v for v in model.component_data_objects(pyo.Var, active=True) if not v.fixed
    )
    wrt = list(states) + [component(model, p) for p in parameters]

    def gradient(expr):
        result = np.asarray(
            differentiate(expr, wrt_list=wrt, mode=Modes.reverse_numeric), dtype=float
        )
        if not np.all(np.isfinite(result)):
            raise ValueError("Nonfinite source derivative")
        return result

    equations = []
    for c in model.component_data_objects(pyo.Constraint, active=True):
        if not c.equality:
            continue
        expr = c.body - c.lower
        row = gradient(expr)
        # Constant fixed equalities convey no measured information. Parameter-
        # only restrictions and singular state equations are not discarded.
        if any(
            not v.fixed for v in identify_variables(expr, include_fixed=True)
        ) or np.any(row):
            equations.append(row)
    if len(equations) != len(states):
        raise ValueError(
            "Fixed-parameter prediction needs a square equality-state model; reconciliation or parameter-only constraints require a different analysis"
        )
    if not states:
        return gradient, np.empty((0, len(parameters))), 1.0
    rows = np.asarray(equations)
    a, b = rows[:, : len(states)], rows[:, len(states) :]
    magnitude = np.max(np.abs(a), axis=0)
    if np.any(magnitude == 0):
        raise ValueError("Singular state Jacobian; local prediction is not determined")
    columns = 1 / magnitude
    scaled = a * columns
    magnitude = np.max(np.abs(scaled), axis=1)
    if np.any(magnitude == 0):
        raise ValueError("Singular state Jacobian; local prediction is not determined")
    rowscale = 1 / magnitude
    scaled = rowscale[:, None] * scaled
    condition = float(np.linalg.cond(scaled))
    if not np.isfinite(condition) or condition > 1e10:
        raise ValueError(
            "Singular or ill-conditioned equilibrated state Jacobian (condition > 1e10)"
        )
    response = columns[:, None] * np.linalg.solve(scaled, -rowscale[:, None] * b)
    return gradient, response, condition


def _sensitivity(study, snapshot, fitted, scales, bound_tolerance):
    theta = {p.component: p.value for p in fitted.estimates}
    trained = next(e for e in fitted.experiments if e.name == snapshot.name)
    try:
        checked = _check_experiment(
            snapshot, theta, _returned(snapshot, trained.values)
        )
        if (
            any(not c.satisfied for c in checked.checks)
            or checked.observations != trained.observations
        ):
            raise ValueError(
                "Fit does not replay against original experiment equations and measurements"
            )
        model = _fixed_parameters(snapshot, theta)
        for path, value in checked.values.items():
            component(model, path).set_value(value, skip_validation=True)
        gradient, response, condition = _regular_system(model, tuple(theta))
        matrix = []
        nstates = response.shape[0]
        for observation in snapshot.observations:
            sigma = observation["standard_deviation"]
            if sigma is None:
                raise ValueError(
                    "Local information requires measurement_error for every observation"
                )
            dy = gradient(component(model, observation["component"]))
            jacobian = (
                (dy[nstates:] + dy[:nstates] @ response)
                * np.asarray([scales[p] for p in theta])
                / sigma
            )
            matrix.append(tuple(float(v) for v in jacobian))
        active = tuple(
            f"{c.component}:{c.side}"
            for c in checked.checks
            if c.side != "eq" and abs(c.residual) <= max(c.tolerance, bound_tolerance)
        )
        warnings = (
            ()
            if not active
            else (
                Diagnostic(
                    "active_model_limits",
                    snapshot.name,
                    "Ambient equality-model derivatives may cross active physical bounds or inequalities; these limits are not measurement information.",
                ),
            )
        )
        return CalibrationSensitivity(
            snapshot.name, "completed", tuple(matrix), condition, active, warnings
        )
    except Exception as exc:
        return CalibrationSensitivity(
            snapshot.name,
            "unresolved",
            diagnostics=(
                Diagnostic(
                    "local_sensitivity_unresolved",
                    snapshot.name,
                    f"{type(exc).__name__}: {exc}",
                ),
            ),
        )


def diagnose(
    study, fitted, *, parameter_scales, rank_tolerance, weak_threshold, bound_tolerance
):
    parameters = study._experiments[0].parameters
    if (
        not parameters
        or not isinstance(parameter_scales, Mapping)
        or set(parameter_scales) != {p["component"] for p in parameters}
    ):
        raise ValueError("Supply parameter_scales for exactly the estimated parameters")
    scales = {
        name: finite(value, "parameter scale")
        for name, value in parameter_scales.items()
    }
    if any(value <= 0 for value in scales.values()):
        raise ValueError("Parameter scales must be positive")
    rank_tolerance, weak_threshold, bound_tolerance = (
        finite(v, "diagnostic threshold")
        for v in (rank_tolerance, weak_threshold, bound_tolerance)
    )
    if not 0 < rank_tolerance <= weak_threshold < 1 or bound_tolerance < 0:
        raise ValueError("Invalid diagnostic thresholds")
    if fitted is None:
        fitted = study.fit()
    _fit_matches(study, fitted)
    diagnostics = []
    experiments = ()
    if fitted.execution == "completed":
        for p in fitted.estimates:
            if any(
                bound is not None
                and abs(p.value - bound) / scales[p.component] <= bound_tolerance
                for bound in (p.lower, p.upper)
            ):
                diagnostics.append(
                    Diagnostic(
                        "parameter_at_bound",
                        p.component,
                        "Estimate is near a declared bound in the supplied parameter scale; local information does not establish a two-sided uncertainty region.",
                    )
                )
        experiments = tuple(
            _sensitivity(study, s, fitted, scales, bound_tolerance)
            for s in study._experiments
        )
    complete = bool(experiments) and all(
        e.execution == "completed" for e in experiments
    )
    if not complete:
        diagnostics.append(
            Diagnostic(
                "calibration_diagnostics_incomplete",
                "experiments",
                "A checked fit and every experiment sensitivity are required; no subset is promoted to complete information.",
            )
        )
    result = CalibrationDiagnostics(
        fitted,
        scales,
        rank_tolerance,
        weak_threshold,
        bound_tolerance,
        "completed" if complete else "unresolved",
        experiments,
        tuple(diagnostics),
    )
    if complete and result.analysis["weak_directions"]:
        rank = result.analysis["rank"]
        diagnostics.append(
            Diagnostic(
                "rank_deficient_parameters"
                if rank < len(parameters)
                else "weak_parameter_information",
                "measurement_jacobian",
                f"Local scaled measurement rank is {rank} of {len(parameters)}; inspect the retained weak parameter combinations. This is not a structural identifiability proof.",
            )
        )
        result = replace(result, diagnostics=tuple(diagnostics))
    return result


def _predict(study, snapshot, estimates, residual_threshold):
    pyo, _ = backend()
    solver, result = {}, None
    try:
        theta = {p.component: p.value for p in estimates}
        model = _fixed_parameters(snapshot, theta)
        # A source objective (including one containing observed values) must
        # never choose the prediction. Reject remaining reconciliation freedom.
        for objective in model.component_data_objects(pyo.Objective, active=True):
            objective.deactivate()
        states = [
            v for v in model.component_data_objects(pyo.Var, active=True) if not v.fixed
        ]
        if states:
            # Structural matching, not numerical derivatives at possibly poor
            # initial guesses, guards against fitting held-out measurements.
            from pyomo.contrib.incidence_analysis import IncidenceGraphInterface

            graph = IncidenceGraphInterface(model, include_inequality=False)
            matching = graph.maximum_matching()
            if len(graph.constraints) != len(states) or len(matching) != len(states):
                raise ValueError(
                    "Held-out prediction requires states determined by a square equality model; free reconciliation variables are unsupported"
                )
            solved, solver = _idaes_results._solve(
                model,
                SimpleNamespace(solver="ipopt", solver_options=study.solver_options),
            )
            if not solved:
                raise ValueError(f"Prediction solve unresolved: {solver}")
        else:
            solver = {
                "scope": "Direct substitution; no free state variables or solver call"
            }
        values = {
            uid(v): number(v)
            for v in model.component_data_objects(pyo.Var, active=True)
        }
        result = _check_experiment(snapshot, theta, _returned(snapshot, values))
        if any(not c.satisfied for c in result.checks):
            raise ValueError(
                "Prediction candidate violates an original equation or bound"
            )
        # Check local regularity on a source clone so solver transformations
        # cannot redefine the equations used for acceptance.
        checker = _fixed_parameters(snapshot, theta)
        for path, value in result.values.items():
            component(checker, path).set_value(value, skip_validation=True)
        _, _, state_condition = _regular_system(checker, tuple(theta))
        solver = {**solver, "source_state_condition": state_condition}
        warnings = tuple(
            Diagnostic(
                "large_validation_residual",
                f"{snapshot.name}:{o.component}",
                f"Absolute standardized residual {abs(o.standardized_residual):.6g} exceeds the declared diagnostic threshold {residual_threshold:g}; prediction was not refitted.",
            )
            for o in result.observations
            if o.standardized_residual is not None
            and abs(o.standardized_residual) > residual_threshold
        )
        return CalibrationPrediction(
            snapshot.name, "completed", result, solver, warnings
        )
    except Exception as exc:
        return CalibrationPrediction(
            snapshot.name,
            "unresolved",
            result,
            solver,
            (
                Diagnostic(
                    "validation_prediction_unresolved",
                    snapshot.name,
                    f"{type(exc).__name__}: {exc}",
                ),
            ),
        )


def validate(study, fitted, *, experiments, residual_threshold):
    from .parmest import ParmestCalibration

    _fit_matches(study, fitted)
    if fitted.execution != "completed" or not fitted.estimates:
        raise ValueError("Held-out validation requires a completed parameter fit")
    residual_threshold = finite(residual_threshold, "residual threshold")
    if residual_threshold <= 0:
        raise ValueError("Residual threshold must be positive")
    if (
        not isinstance(experiments, Mapping)
        or not experiments
        or set(experiments) & {e.name for e in fitted.experiments}
    ):
        raise ValueError(
            "Supply nonempty held-out experiments with names disjoint from training"
        )
    heldout = ParmestCalibration(
        name=f"{study.name}: held-out validation",
        experiments=experiments,
        objective=study.objective,
        solver_options=study.solver_options,
        tolerance=study.tolerance,
        provenance={
            "calibration_ref": fitted.ref.to_dict(),
            "role": "held_out_prediction",
        },
    )
    expected = {p.component: p.unit for p in fitted.estimates}
    if any(
        {p["component"]: p["unit"] for p in s.parameters} != expected
        for s in heldout._experiments
    ):
        raise ValueError(
            "Held-out parameter paths and native units must match the fitted parameters"
        )
    predictions = tuple(
        _predict(heldout, s, fitted.estimates, residual_threshold)
        for s in heldout._experiments
    )
    complete = all(p.execution == "completed" for p in predictions)
    diagnostics = (
        ()
        if complete
        else (
            Diagnostic(
                "validation_incomplete",
                "experiments",
                "At least one held-out prediction is unresolved; aggregate accuracy metrics are unavailable.",
            ),
        )
    )
    return CalibrationValidation(
        fitted,
        heldout.manifest,
        residual_threshold,
        "completed" if complete else "unresolved",
        predictions,
        diagnostics,
    )
