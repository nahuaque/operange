"""Physical parametric LPs and bounded proposals for exact affine value segments."""

from dataclasses import dataclass, asdict
from fractions import Fraction as F

from . import linear
from ._numeric import exact_dot as dot
from .primitives import finite


def solve_exact(matrix, rhs):
    """Solve a full-column-rank rational system; reject inconsistent proposals."""
    n = len(matrix[0]) if matrix else 0
    if not n:
        return [] if not any(rhs) else None
    work = [[F(a) for a in row] + [F(b)] for row, b in zip(matrix, rhs)]
    if len(work) < n:
        return None
    for j in range(n):
        pivot = next((i for i in range(j, len(work)) if work[i][j]), None)
        if pivot is None:
            return None
        work[j], work[pivot] = work[pivot], work[j]
        v = work[j][j]
        work[j] = [a / v for a in work[j]]
        for i in range(len(work)):
            if i != j and work[i][j]:
                v = work[i][j]
                work[i] = [a - v * b for a, b in zip(work[i], work[j])]
    if any(row[-1] for row in work[n:]):
        return None
    return [work[i][-1] for i in range(n)]


def independent_rows(rows, order, n):
    reduced, selected = {}, []
    for i in order:
        row = list(rows[i])
        for pivot, previous in sorted(reduced.items()):
            factor = row[pivot]
            row = [a - factor * b for a, b in zip(row, previous)]
        pivot = next((j for j, a in enumerate(row) if a), None)
        if pivot is None:
            continue
        reduced[pivot] = [a / row[pivot] for a in row]
        selected.append(i)
        if len(selected) == n:
            return selected
    return None


@dataclass
class ParametricDispatch:
    names: tuple
    origins: list
    scales: list
    rows: list
    rhs: list
    refs: list
    cost: list
    constant: F
    sign: int

    def proposal(self, rhs, tolerance, backend):
        """Numerical primal/dual values only choose a rational basis to check."""
        n, m = len(self.names), len(self.rows)
        matrix, upper = [], []
        for row, value in zip(self.rows, rhs):
            scaled = [a * s for a, s in zip(row, self.scales)]
            b = value - dot(row, self.origins)
            norm = max([abs(b), *map(abs, scaled)]) or F(1)
            matrix.append([finite(float(a / norm), "LP row") for a in scaled])
            upper.append(finite(float(b / norm), "LP rhs"))
        objective = [a * s for a, s in zip(self.cost, self.scales)]
        norm = max(map(abs, objective)) or F(1)
        objective = [float(a / norm) for a in objective]
        # Use the existing LP wrapper by default. CVXPY remains an explicit
        # optional proposal backend; both paths are checked in original units.
        if backend == "scipy":
            primal, report = linear.solve_lp(
                objective,
                [(None, None)] * n,
                tolerance,
                inequalities=matrix,
                upper=upper,
            )
            dual, dual_report = linear.solve_lp(
                upper,
                [(0, None)] * m,
                tolerance,
                equalities=list(map(list, zip(*matrix))),
                rhs=[-v for v in objective],
            )
            attempts = [asdict(report), asdict(dual_report)]
        else:
            from ._cvxpy_backend import load_cvxpy, numeric_vector

            cp = load_cvxpy()
            x = cp.Variable(n)
            import numpy as np

            constraint = np.asarray(matrix) @ x <= upper
            problem = cp.Problem(cp.Minimize(np.asarray(objective) @ x), [constraint])
            try:
                problem.solve(
                    solver="SCIPY",
                    scipy_options={
                        "method": "highs",
                        "primal_feasibility_tolerance": tolerance,
                        "dual_feasibility_tolerance": tolerance,
                    },
                )
            except cp.error.SolverError as exc:
                return None, [
                    {"solver": "CVXPY", "status": "unresolved", "message": str(exc)}
                ]
            primal = numeric_vector(x.value, n, "primal")
            dual = numeric_vector(constraint.dual_value, m, "dual")
            attempts = [
                {"solver": "CVXPY / SciPy-HiGHS", "status": str(problem.status)}
            ]
        if primal is None:
            return None, attempts
        residuals = [
            abs(sum(a * b for a, b in zip(row, primal)) - b)
            for row, b in zip(matrix, upper)
        ]
        order = sorted(
            range(m),
            key=lambda i: (not (dual is not None and dual[i] > 0), residuals[i], i),
        )
        basis = independent_rows(self.rows, order, n)
        basis = self.repair_basis(basis, rhs)
        return basis, attempts

    def repair_basis(self, basis, rhs):
        """At most eight exact exchanges repair a numerically rounded active set.

        Only dual-feasible bases are repaired. A violated physical row enters
        using the nonnegative multiplier ratio test. Cycling, rank loss or an
        open residual after this bounded attempt leaves the later proof check
        to reject the proposal; this is not an exact LP completion guarantee.
        """
        seen = set()
        for _ in range(8):
            if basis is None or tuple(basis) in seen:
                break
            seen.add(tuple(basis))
            matrix = [self.rows[i] for i in basis]
            transposed = list(map(list, zip(*matrix)))
            x = solve_exact(matrix, [rhs[i] for i in basis])
            dual = solve_exact(transposed, [-c for c in self.cost])
            if x is None or dual is None or any(v < 0 for v in dual):
                break
            violated = next(
                (
                    i
                    for i, (row, b) in enumerate(zip(self.rows, rhs))
                    if dot(row, x) > b
                ),
                None,
            )
            if violated is None:
                break
            weights = solve_exact(transposed, self.rows[violated])
            eligible = [i for i, w in enumerate(weights) if w > 0]
            if not eligible:
                break
            leaving = min(eligible, key=lambda i: (dual[i] / weights[i], basis[i]))
            basis = list(basis)
            basis[leaving] = violated
        return basis

    def segment(self, basis, rates, constant_rate, direction, maximum):
        if basis is None:
            return None
        selected = [self.rows[i] for i in basis]
        x = solve_exact(selected, [self.rhs[i] for i in basis])
        dx = solve_exact(selected, [rates[i] for i in basis])
        multipliers = solve_exact(
            list(map(list, zip(*selected))), [-c for c in self.cost]
        )
        if (
            x is None
            or dx is None
            or multipliers is None
            or any(v < 0 for v in multipliers)
        ):
            return None
        radius = F(maximum)
        for row, bound, rate in zip(self.rows, self.rhs, rates):
            slack = bound - dot(row, x)
            if slack < 0:
                return None
            change = direction * (rate - dot(row, dx))
            if change < 0:
                radius = min(radius, slack / -change)
        if radius <= 0:
            return None
        dual = [F(0)] * len(self.rows)
        for i, value in zip(basis, multipliers):
            dual[i] = value
        value = self.constant + dot(self.cost, x)
        slope = constant_rate + dot(self.cost, dx)
        if value != self.constant - dot(self.rhs, dual) or slope != constant_rate - dot(
            rates, dual
        ):
            return None
        return (
            radius,
            self.sign * slope,
            {
                "row_refs": self.refs,
                "controls_exact": dict(zip(self.names, map(str, x))),
                "control_rates_exact": dict(zip(self.names, map(str, dx))),
                "multipliers_exact": list(map(str, dual)),
                "base_value_exact": str(self.sign * value),
            },
        )

    def infeasible_direction(self, rates, direction, probe, tolerance):
        """Propose a Farkas ray; only a zero base intercept proves local exclusion."""
        m = len(self.rows)
        system = [list(column) for column in zip(*self.rows)] + [[F(1)] * m]
        rhs = [F(0)] * len(self.names) + [F(1)]
        multipliers, _ = linear.solve_lp(
            [
                finite(float(b + probe * h), "Farkas objective")
                for b, h in zip(self.rhs, rates)
            ],
            [(0, None)] * m,
            tolerance,
            equalities=[
                [finite(float(a), "Farkas row") for a in row] for row in system
            ],
            rhs=list(map(float, rhs)),
        )
        if multipliers is None:
            return None
        support = [i for i, value in enumerate(multipliers) if value > 0]
        solved = solve_exact([[row[i] for i in support] for row in system], rhs)
        if solved is None or any(v < 0 for v in solved):
            return None
        dual = [F(0)] * m
        for i, value in zip(support, solved):
            dual[i] = value
        if dot(self.rhs, dual) != 0 or direction * dot(rates, dual) >= 0:
            return None
        return {"row_refs": self.refs, "multipliers_exact": list(map(str, dual))}


def problem(claim, point):
    model = claim.adapter
    fixed = model._permissions(claim)
    names = tuple(c.name for c in model.variables)
    rows, rhs, refs = [], [], []
    limits = model.operating_limits + tuple(
        r for r in model.requirements if r.name in claim.requirements
    )
    for limit in limits:
        output = model.output(limit.output)
        terms = {t.variable: F(t.coefficient) for t in output.terms}
        offset = F(output.offset) + sum(
            (a * F(point[n]) for n, a in terms.items() if n in point), F(0)
        )
        rows.append([limit.sign * terms.get(n, F(0)) for n in names])
        rhs.append(F(limit.tolerance) + limit.sign * (F(limit.limit) - offset))
        refs.append(
            f"{'operating_limit' if limit in model.operating_limits else 'requirement'}:{limit.name}"
        )
    for i, control in enumerate(model.variables):
        for side, sign in (("lower", -1), ("upper", 1)):
            row = [F(sign if j == i else 0) for j in range(len(names))]
            rows.append(row)
            rhs.append(sign * F(getattr(control, side)))
            refs.append(f"{control.quantity.role}_{side}:{control.name}")
            if control.name in fixed:
                rows.append(row)
                rhs.append(sign * F(fixed[control.name]))
                refs.append(f"fixed_{side}:{control.name}")
    objective = model.output(model.objective.output)
    terms = {t.variable: F(t.coefficient) for t in objective.terms}
    sign = 1 if model.objective.sense == "minimize" else -1
    return ParametricDispatch(
        names,
        [F(c.lower) for c in model.variables],
        [
            F(c.upper) - F(c.lower) or max(F(1), abs(F(c.lower)))
            for c in model.variables
        ],
        rows,
        rhs,
        refs,
        [sign * terms.get(n, F(0)) for n in names],
        sign
        * (
            F(objective.offset)
            + sum((v * F(point[n]) for n, v in terms.items() if n in point), F(0))
        ),
        sign,
    )


def change_rates(claim, system, change):
    model = claim.adapter
    rates, constant = [F(0)] * len(system.rows), F(0)
    if change.kind == "input":
        if change.name not in model.input_space.names:
            raise ValueError("unknown sensitivity input")
        unit = model.input_space.coordinate(change.name).unit
        limits = model.operating_limits + tuple(
            r for r in model.requirements if r.name in claim.requirements
        )
        for i, limit in enumerate(limits):
            rates[i] = -limit.sign * sum(
                (
                    F(t.coefficient)
                    for t in model.output(limit.output).terms
                    if t.variable == change.name
                ),
                F(0),
            )
        constant = system.sign * sum(
            (
                F(t.coefficient)
                for t in model.output(model.objective.output).terms
                if t.variable == change.name
            ),
            F(0),
        )
    elif change.kind == "operating_limit":
        limit = next((r for r in model.operating_limits if r.name == change.name), None)
        if limit is None:
            raise ValueError(
                "sensitivity must name an operating limit, not a service requirement"
            )
        unit = model.output(limit.output).unit
        rates[system.refs.index(change.change_id)] = F(limit.sign)
    else:
        role, side = change.kind.split("_")
        control = next(
            (
                c
                for c in model.variables
                if c.name == change.name and c.quantity.role == role
            ),
            None,
        )
        if control is None:
            raise ValueError("unknown sensitivity control/state bound")
        unit = control.unit
        rates[system.refs.index(change.change_id)] = F(1 if side == "upper" else -1)
    if change.unit != unit:
        raise ValueError(
            "sensitivity change unit differs from the physical declaration"
        )
    return rates, constant
