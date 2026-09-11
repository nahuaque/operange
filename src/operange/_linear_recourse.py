"""Numerical candidates and exact, bounded-box linear feasibility certificates.

For rows A z <= b and 0 <= z <= 1, every nonnegative multiplier vector
lambda implies min_box((lambda A) z) <= lambda b. A strictly positive
gap between those two sides proves infeasibility. Exact arithmetic and the
bounded box remove any need to trust solver status or an approximate dual
equality. The LPs only propose controls and multipliers.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction

from . import linear
from ._numeric import exact_dot
from .primitives import finite


@dataclass
class LinearSystem:
    controls: tuple
    fixed: dict
    rows: list
    upper: list
    row_refs: list

    @property
    def spans(self):
        return tuple(Fraction(c.upper) - Fraction(c.lower) for c in self.controls)

    def candidate(self, coordinates):
        if len(coordinates) != len(self.controls):
            raise ValueError("solver returned the wrong number of control coordinates")
        values = dict(self.fixed)
        for control, span, coordinate in zip(self.controls, self.spans, coordinates):
            z = Fraction(min(1.0, max(0.0, finite(coordinate, "control coordinate"))))
            value = finite(float(Fraction(control.lower) + span * z), control.name)
            values[control.name] = min(control.upper, max(control.lower, value))
        # Recheck the physical floats that will actually be returned, including
        # the rounding introduced by denormalizing the proposed coordinates.
        z = [
            (Fraction(values[c.name]) - Fraction(c.lower)) / span
            for c, span in zip(self.controls, self.spans)
        ]
        if any(exact_dot(row, z) > b for row, b in zip(self.rows, self.upper)):
            return None
        return values

    def certificate(self, multipliers):
        if len(multipliers) != len(self.rows):
            raise ValueError("solver returned the wrong number of multipliers")
        multipliers = [max(0.0, finite(v, "multiplier")) for v in multipliers]
        weighted = [exact_dot(column, multipliers) for column in zip(*self.rows)]
        minimum = sum((min(Fraction(0), a) for a in weighted), Fraction(0))
        upper = exact_dot(self.upper, multipliers)
        gap = minimum - upper
        if gap <= 0:
            return None
        return {
            "formula": "min over 0 <= z <= 1 of (lambda A) z > lambda b",
            "arithmetic": "exact_rationals_of_declared_floats",
            "row_constraint_refs": self.row_refs,
            "rows_exact": [[str(a) for a in row] for row in self.rows],
            "upper_exact": [str(b) for b in self.upper],
            "multipliers": multipliers,
            "weighted_coefficients_exact": list(map(str, weighted)),
            "box_minimum_exact": str(minimum),
            "weighted_upper_exact": str(upper),
            "contradiction_gap_exact": str(gap),
            "control_coordinates": [
                {
                    "name": c.name,
                    "origin_exact": str(Fraction(c.lower)),
                    "scale_exact": str(span),
                }
                for c, span in zip(self.controls, self.spans)
            ],
            "fixed_controls": self.fixed,
            "scope": "joint infeasibility; no claim of a minimal conflict or individually impossible requirements",
        }


def compile_system(claim, point):
    model = claim.adapter
    fixed = model._permissions(claim)
    fixed.update(
        {
            c.name: c.lower
            for c in model.controls
            if c.lower == c.upper and c.name not in fixed
        }
    )
    controls = tuple(c for c in model.controls if c.name not in fixed)
    constants = {**point, **fixed, **{c.name: c.lower for c in controls}}
    rows, upper, refs = [], [], []
    limits = model.operating_limits + tuple(
        r for r in model.requirements if r.name in claim.requirements
    )
    for limit in limits:
        output = model.output(limit.output)
        coefficients = {t.variable: Fraction(t.coefficient) for t in output.terms}
        scale = Fraction(limit.residual_scale)
        rows.append(
            tuple(
                limit.sign
                * coefficients.get(c.name, 0)
                * (Fraction(c.upper) - Fraction(c.lower))
                / scale
                for c in controls
            )
        )
        upper.append(
            (
                Fraction(limit.tolerance)
                - limit.sign * (output._exact_value(constants) - Fraction(limit.limit))
            )
            / scale
        )
        refs.append(limit.name)
    # Fixed values remain subject to the declared equipment bounds. Their
    # constant rows also give a certificate for an out-of-bounds fixed command.
    for control in model.controls:
        if control.name in fixed:
            for side, b in (
                ("lower", Fraction(fixed[control.name]) - Fraction(control.lower)),
                ("upper", Fraction(control.upper) - Fraction(fixed[control.name])),
            ):
                rows.append((Fraction(0),) * len(controls))
                upper.append(b)
                refs.append(model.bound_id(control, side))
    return LinearSystem(controls, fixed, rows, upper, refs)


@dataclass
class RecourseSolution:
    feasibility: str
    controls: dict | None
    certificate: dict | None
    attempts: list
    message: str


def solve_system(system, tolerance):
    attempts = []

    def solve(label, *args, **kwargs):
        point, evidence = linear.solve_lp(*args, tolerance=tolerance, **kwargs)
        attempts.append({"purpose": label, **asdict(evidence)})
        return point

    def infeasible(proof):
        return RecourseSolution(
            "infeasible",
            None,
            proof,
            attempts,
            "Exact bounded-control contradiction verified.",
        )

    try:
        m, n = len(system.rows), len(system.controls)
        # A single impossible row can be resolved without invoking a solver.
        for i in range(m):
            proof = system.certificate([float(j == i) for j in range(m)])
            if proof is not None:
                return infeasible(proof)
        if not n:
            values = system.candidate([])
        else:
            rows = [
                [finite(float(a), "LP coefficient") for a in row] for row in system.rows
            ]
            upper = [finite(float(b), "LP right-hand side") for b in system.upper]
            # Phase I chooses a point with the smallest maximum scaled violation.
            # A negative phase-I value is only a proposal, never a certificate.
            primal = solve(
                "dispatch_candidate",
                [0.0] * n + [1.0],
                [(0.0, 1.0)] * n + [(None, None)],
                inequalities=[row + [-1.0] for row in rows],
                upper=upper,
            )
            values = system.candidate(primal[:-1]) if primal is not None else None
        if values is not None:
            return RecourseSolution(
                "feasible",
                values,
                None,
                attempts,
                "Returned physical controls checked with exact rational residuals.",
            )
        # min b.lambda - sum(w), w_j <= (lambda A)_j, w_j <= 0,
        # lambda >= 0, sum(lambda) = 1. The resulting multipliers are verified
        # against the exact original rows, including all declared tolerances.
        dual = solve(
            "infeasibility_candidate",
            upper + [-1.0] * n,
            [(0.0, 1.0)] * m + [(None, 0.0)] * n,
            inequalities=[
                [-row[j] for row in rows] + [float(k == j) for k in range(n)]
                for j in range(n)
            ],
            upper=[0.0] * n,
            equalities=[[1.0] * m + [0.0] * n],
            rhs=[1.0],
        )
        if dual is not None:
            proof = system.certificate(dual[:m])
            if proof is not None:
                return infeasible(proof)
        message = "Neither a feasible physical control vector nor an exact infeasibility certificate was established."
    except (ValueError, OverflowError) as exc:
        message = str(exc)
    return RecourseSolution("unknown", None, None, attempts, message)
