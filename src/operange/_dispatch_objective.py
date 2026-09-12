"""Preferred dispatch candidates and exact global objective bounds.

Minimize k + c.z + sum(d_i*z_i**2), with d >= 0, A z <= b and 0 <= z <= 1.
For any lambda >= 0, minimizing its separable Lagrangian over the box gives a
global lower bound. Numerical solvers only propose commands and multipliers.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction as F

from . import linear
from ._cvxpy_backend import numeric_vector, solve_bounded
from ._linear_recourse import RecourseSolution, solve_system
from ._numeric import exact_dot, round_down, round_up
from .contract_types import Bound, Evidence, Measurement, Objective
from .objectives import LinearObjective
from .primitives import finite


@dataclass
class Polynomial:
    constant: F
    linear: tuple
    diagonal: tuple

    def value(self, system, controls):
        z = [
            (F(controls[c.name]) - F(c.lower)) / span
            for c, span in zip(system.controls, system.spans)
        ]
        return (
            self.constant
            + exact_dot(self.linear, z)
            + sum(d * v * v for d, v in zip(self.diagonal, z))
        )

    def bound(self, system, multipliers):
        values = numeric_vector(multipliers, len(system.rows), "objective multipliers")
        if values is None:
            raise ValueError("missing objective multipliers")
        multipliers = [max(0.0, v) for v in values]
        slopes = [
            c + exact_dot((row[i] for row in system.rows), multipliers)
            for i, c in enumerate(self.linear)
        ]
        minimizers = [
            min(F(1), max(F(0), -c / (2 * d))) if d else F(c < 0)
            for c, d in zip(slopes, self.diagonal)
        ]
        bound = (
            self.constant
            - exact_dot(system.upper, multipliers)
            + sum(
                d * z * z + c * z for d, c, z in zip(self.diagonal, slopes, minimizers)
            )
        )
        return bound, {
            "formula": "k - lambda.b + sum(min_0<=z_i<=1(d_i*z_i**2 + (c_i + (lambda.A)_i)*z_i))",
            "multipliers": multipliers,
            "lagrangian_slopes_exact": list(map(str, slopes)),
            "box_minimizers_exact": list(map(str, minimizers)),
            "minimization_lower_exact": str(bound),
        }


def compile_objective(model, point, system):
    spec = model.objective
    constants = {**point, **system.fixed, **{c.name: c.lower for c in system.controls}}
    if isinstance(spec, LinearObjective):
        sign = 1 if spec.sense == "minimize" else -1
        output = model.output(spec.output)
        coefficients = {t.variable: F(t.coefficient) for t in output.terms}
        return Polynomial(
            sign * output._exact_value(constants),
            tuple(
                sign * coefficients.get(c.name, 0) * span
                for c, span in zip(system.controls, system.spans)
            ),
            (F(0),) * len(system.controls),
        )
    constant = F(0)
    coefficients, diagonal = (
        dict.fromkeys((c.name for c in system.controls), F(0)),
        dict.fromkeys((c.name for c in system.controls), F(0)),
    )
    for target in spec.targets:
        weight = F(target.weight) / F(target.scale) ** 2
        difference = F(constants[target.control]) - F(target.target)
        constant += weight * difference**2
        if target.control in coefficients:
            control = next(c for c in system.controls if c.name == target.control)
            span = F(control.upper) - F(control.lower)
            coefficients[target.control] = 2 * weight * difference * span
            diagonal[target.control] = weight * span**2
    return Polynomial(constant, tuple(coefficients.values()), tuple(diagonal.values()))


def _candidates(system, primal):
    from ._linear_recourse import checked_candidates

    return checked_candidates(system, primal)


def _interpolate(system, primal, seed):
    """Try the checked portion of the line from a feasible seed to the proposal."""
    primal = numeric_vector(primal, len(system.controls), "dispatch candidate")
    if primal is None:
        return []
    start = [
        (F(seed[c.name]) - F(c.lower)) / span
        for c, span in zip(system.controls, system.spans)
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


def solve_dispatch(system, polynomial, tolerance, *, backend, candidate_transform=None):
    lower, proof = polynomial.bound(system, [0.0] * len(system.rows))
    attempts, candidates, primal = [], [], None
    if system.controls and (any(polynomial.linear) or any(polynomial.diagonal)):
        try:
            primal, multipliers, reports = _numerical(
                system, polynomial, tolerance, backend
            )
            attempts.extend(reports)
            if primal is not None and candidate_transform is not None:
                primal = candidate_transform(primal)
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
                candidates = _interpolate(system, primal, solution.controls)
                solution.controls = min(
                    [solution.controls, *candidates],
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
            "strictly_convex_on_adjustable_controls": bool(system.controls)
            and all(d > 0 for d in polynomial.diagonal),
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
                for c, span in zip(system.controls, system.spans)
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
