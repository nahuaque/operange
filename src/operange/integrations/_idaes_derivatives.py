"""Shared local equality-state differentiation using public Pyomo APIs."""

import numpy as np

from ._idaes_model import backend, component


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
