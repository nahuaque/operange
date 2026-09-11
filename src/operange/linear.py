"""Shared SciPy LP interface with independently recomputed primal residuals.

Solver termination is numerical evidence. Model adapters supply their own
analytical checks before interpreting infeasibility or claiming global bounds.
"""

from __future__ import annotations

from math import isfinite
from typing import Sequence

import scipy
from scipy.optimize import linprog

from .primitives import finite
from .results import SolveEvidence


def solve_lp(
    objective: Sequence[float],
    bounds: Sequence[tuple[float | None, float | None]],
    tolerance: float,
    *,
    inequalities: Sequence[Sequence[float]] = (),
    upper: Sequence[float] = (),
    equalities: Sequence[Sequence[float]] = (),
    rhs: Sequence[float] = (),
) -> tuple[list[float] | None, SolveEvidence]:
    """Recompute primal residuals instead of trusting just the success flag."""
    tolerance = finite(tolerance, "tolerance")
    if not 1e-10 <= tolerance <= 1e-4:
        raise ValueError("tolerance must lie between 1e-10 and 1e-4")
    if not objective or len(bounds) != len(objective):
        raise ValueError("objective and bounds must have the same nonzero length")
    for value in objective:
        finite(value, "objective coefficient")
    for matrix, vector in ((inequalities, upper), (equalities, rhs)):
        if len(matrix) != len(vector):
            raise ValueError("constraint matrix and right-hand side lengths differ")
        for row, value in zip(matrix, vector):
            if len(row) != len(objective):
                raise ValueError("constraint row has the wrong number of variables")
            for coefficient in row:
                finite(coefficient, "constraint coefficient")
            finite(value, "constraint right-hand side")
    for low, high in bounds:
        if low is not None:
            finite(low, "lower bound")
        if high is not None:
            finite(high, "upper bound")
        if low is not None and high is not None and low > high:
            raise ValueError("variable bounds must be ordered")
    solver = f"SciPy {scipy.__version__} / HiGHS (linprog)"
    try:
        result = linprog(
            objective,
            A_ub=inequalities or None,
            b_ub=upper or None,
            A_eq=equalities or None,
            b_eq=rhs or None,
            bounds=bounds,
            method="highs",
            options={
                "primal_feasibility_tolerance": tolerance,
                "dual_feasibility_tolerance": tolerance,
            },
        )
    except (ValueError, RuntimeError) as exc:
        return None, SolveEvidence("unresolved", str(exc), solver, tolerance)
    if result.status != 0 or not result.success:
        status = "infeasible" if result.status == 2 else "unresolved"
        return None, SolveEvidence(status, str(result.message), solver, tolerance)
    x = list(result.x) if result.x is not None else []
    if len(x) != len(objective) or not all(isfinite(value) for value in x):
        return None, SolveEvidence(
            "unresolved", "solver returned invalid coordinates", solver, tolerance
        )
    x = [float(value) for value in x]
    residuals = [0.0]
    for row, limit in zip(inequalities, upper):
        residuals.append(sum(a * b for a, b in zip(row, x)) - limit)
    for row, value in zip(equalities, rhs):
        residuals.append(abs(sum(a * b for a, b in zip(row, x)) - value))
    for value, (low, high) in zip(x, bounds):
        if low is not None:
            residuals.append(max(0.0, low - value))
        if high is not None:
            residuals.append(max(0.0, value - high))
    if not all(isfinite(value) for value in residuals):
        return None, SolveEvidence(
            "unresolved", "non-finite LP residual", solver, tolerance
        )
    residual = max(residuals)
    if not isfinite(residual) or residual > tolerance:
        return None, SolveEvidence(
            "unresolved", "LP residual check failed", solver, tolerance, residual
        )
    return x, SolveEvidence(
        "optimal", "LP solved; primal residuals checked", solver, tolerance, residual
    )
