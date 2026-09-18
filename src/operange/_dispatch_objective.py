"""Preferred dispatch candidates and exact global objective bounds.

Minimize k + c.z + sum(d_i*z_i**2), with d >= 0, A z <= b and 0 <= z <= 1.
For any lambda >= 0, minimizing its separable Lagrangian over the box gives a
global lower bound. Numerical solvers only propose commands and multipliers.
"""

from dataclasses import asdict
from fractions import Fraction as F

from . import linear
from ._linear_problem import Polynomial as Polynomial
from ._linear_compile import objective as physical_objective
from ._cvxpy_backend import numeric_vector, solve_bounded
from ._linear_recourse import RecourseSolution, solve_system
from ._numeric import exact_dot, round_down, round_up
from .contract_types import Bound, Evidence, Measurement, Objective
from .objectives import LinearObjective
from .primitives import finite


def compile_objective(model, point, system):
    return physical_objective(model).normalized(system, point)


def _candidates(system, primal):
    from ._linear_recourse import checked_candidates

    return checked_candidates(system, primal)


def _interpolate(system, primal, seed):
    """Try the checked portion of the line from a feasible seed to the proposal."""
    primal = numeric_vector(primal, len(system.variables), "dispatch candidate")
    if primal is None:
        return []
    start = [
        (F(seed[c.name]) - F(c.lower)) / span
        for c, span in zip(system.variables, system.spans)
    ]
    delta = [F(min(1.0, max(0.0, v))) - s for v, s in zip(primal, start)]
    fraction = F(1)
    for row, upper in zip(system.rows, system.upper):
        change = exact_dot(row, delta)
        if change > 0:
            fraction = min(fraction, (upper - exact_dot(row, start)) / change)
    controls = []
    for shrink in (1.0, 1 - 1e-14, 1 - 1e-12, 1 - 1e-10):
        proposal = [float(s + F(shrink) * fraction * d) for s, d in zip(start, delta)]
        controls.extend(_candidates(system, proposal))
    return controls


def _numerical(system, polynomial, tolerance, backend):
    cost = [finite(float(v), "objective coefficient") for v in polynomial.linear]
    if backend == "cvxpy":
        result = solve_bounded(
            system,
            cost,
            tolerance,
            quadratic=polynomial.diagonal if any(polynomial.diagonal) else None,
        )
        for report in result.attempts:
            report["purpose"] = "operating_objective_candidate"
        return result.point, result.multipliers, result.attempts
    rows = [[finite(float(v), "LP coefficient") for v in row] for row in system.rows]
    rhs = [finite(float(v), "LP rhs") for v in system.upper]
    n, m = len(cost), len(rows)
    primal, report = linear.solve_lp(
        cost, [(0, 1)] * n, tolerance, inequalities=rows, upper=rhs
    )
    attempts = [{"purpose": "operating_objective_candidate", **asdict(report)}]
    dual, report = linear.solve_lp(
        rhs + [-1.0] * n,
        [(0, None)] * m + [(None, 0)] * n,
        tolerance,
        inequalities=[
            [-row[j] for row in rows] + [float(k == j) for k in range(n)]
            for j in range(n)
        ],
        upper=cost,
    )
    attempts.append(
        {"purpose": "operating_objective_bound_candidate", **asdict(report)}
    )
    return primal, None if dual is None else dual[:m], attempts


def solve_dispatch(
    system,
    polynomial,
    tolerance,
    *,
    backend,
    candidate_transform=None,
    candidate_values=None,
):
    lower, proof = polynomial.bound(system, [0.0] * len(system.rows))
    attempts, candidates, primal = [], [], None
    if system.variables and (any(polynomial.linear) or any(polynomial.diagonal)):
        try:
            primal, multipliers, reports = _numerical(
                system, polynomial, tolerance, backend
            )
            attempts.extend(reports)
            if primal is not None and candidate_transform is not None:
                primal = candidate_transform(primal)
            if primal is not None and candidate_values is not None:
                values = candidate_values(primal)
                checked = system.checked_values(values) if values is not None else None
                candidates = [checked] if checked is not None else []
            else:
                candidates = _candidates(system, primal)
            if multipliers is not None:
                proposed, certificate = polynomial.bound(system, multipliers)
                if proposed > lower:
                    lower, proof = proposed, certificate
        except (ValueError, OverflowError) as exc:
            attempts.append(
                {
                    "purpose": "operating_objective_candidate",
                    "status": "unresolved",
                    "message": str(exc),
                }
            )
    if candidates:
        best = min(candidates, key=lambda c: polynomial.value(system, c))
        solution = RecourseSolution(
            "feasible",
            best,
            None,
            attempts,
            "Preferred dispatch passed exact physical feasibility checks.",
        )
    else:
        solution = solve_system(system, tolerance, backend=backend)
        solution.attempts = attempts + solution.attempts
        if solution.feasibility == "feasible" and primal is not None:
            try:
                candidates = _interpolate(system, primal, solution.values)
                solution.values = min(
                    [solution.values, *candidates],
                    key=lambda c: polynomial.value(system, c),
                )
            except (ValueError, OverflowError):
                pass  # Keep the independently verified feasible dispatch.
    return solution, lower, proof


def objective_result(model, point, system, controls, polynomial, lower, proof):
    spec = model.objective
    is_linear = isinstance(spec, LinearObjective)
    sense = spec.sense if is_linear else "minimize"
    sign = 1 if sense == "minimize" else -1
    value = (
        model.output(spec.output)._exact_value({**point, **controls})
        if is_linear
        else sum(
            F(t.weight) * ((F(controls[t.control]) - F(t.target)) / F(t.scale)) ** 2
            for t in spec.targets
        )
    )
    if sign * value != polynomial.value(system, controls) or sign * value < lower:
        raise ValueError("objective certificate disagrees with the physical dispatch")
    low, high = (lower, value) if sign == 1 else (value, -lower)
    lo, hi = round_down(low), round_up(high)
    quantity = spec.output if is_linear else spec.quantity_id
    unit = model.output(spec.output).unit if is_linear else "1"
    closed = F(hi) - F(lo) <= F(spec.tolerance) and hi - lo <= spec.tolerance
    evidence = Evidence(
        "operating_objective",
        "dispatch_objective_bounds",
        "exact_separable_lagrangian_bound",
        "verified",
        (Measurement("attained_value", finite(float(value), "objective value"), unit),),
        (Measurement("absolute_tolerance", spec.tolerance, unit),),
        details={
            "declaration": spec.to_dict(),
            "sense": sense,
            "arithmetic": "exact_rationals_of_declared_floats; bounds rounded outward",
            "scope": "objective optimum at this realization under all included constraints; no dispatch sensitivity or controller asserted",
            "resolution": "minimum_verified"
            if closed and sign == 1
            else "maximum_verified"
            if closed
            else "bounded",
            (
                "strictly_convex_on_adjustable_variables"
                if model.states
                else "strictly_convex_on_adjustable_controls"
            ): bool(system.variables) and all(d > 0 for d in polynomial.diagonal),
            "value_exact": str(value),
            "lower_exact": str(low),
            "upper_exact": str(high),
            "constant_exact": str(polynomial.constant),
            "linear_exact": list(map(str, polynomial.linear)),
            "diagonal_exact": list(map(str, polynomial.diagonal)),
            "rows_exact": [[str(a) for a in row] for row in system.rows],
            "rhs_exact": list(map(str, system.upper)),
            "row_constraint_refs": list(system.row_refs),
            "fixed_controls": system.fixed,
            "control_coordinates": [
                {
                    "name": c.name,
                    "origin_exact": str(F(c.lower)),
                    "scale_exact": str(span),
                }
                for c, span in zip(system.variables, system.spans)
            ],
            "certificate": proof,
        },
    )
    objective = Objective(
        quantity,
        sense,
        float(value),
        Bound("finite", unit, lo, ("operating_objective",)),
        Bound("finite", unit, hi, ("operating_objective",)),
        "attained",
        "verified" if closed else "unknown",
    )
    return objective, evidence
