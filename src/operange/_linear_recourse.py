"""Numerical candidates and exact, bounded-box linear feasibility certificates.

For rows A z <= b and 0 <= z <= 1, every nonnegative multiplier vector
lambda implies min_box((lambda A) z) <= lambda b. A strictly positive
gap between those two sides proves infeasibility. Exact arithmetic and the
bounded box remove any need to trust solver status or an approximate dual
equality. The LPs only propose controls and multipliers.
"""

from dataclasses import asdict, dataclass
from math import inf, nextafter
from fractions import Fraction

from . import linear
from ._linear_problem import LinearSystem as LinearSystem
from ._linear_compile import compile_model
from ._numeric import exact_dot, numeric_vector, round_down, round_up
from .primitives import finite


def checked_candidates(system, primal):
    primal = numeric_vector(primal, len(system.variables), "dispatch candidate")
    if primal is None:
        return []
    proposals = [primal]
    # A neighboring representable command can recover a rounded LP boundary.
    for i, value in enumerate(primal):
        for direction in (-inf, inf):
            proposals.append(
                primal[:i] + [nextafter(value, direction)] + primal[i + 1 :]
            )
    controls = []
    for proposal in proposals:
        try:
            candidate = system.candidate(proposal)
            if candidate is not None:
                controls.append(candidate)
        except (ValueError, OverflowError):
            continue
    if not controls:
        # Repair one violated row in physical coordinates. A neighboring float
        # can recover an equality lost in the numerical solve or denormalization.
        # These remain proposals: all rows and fixed commands are rechecked.
        values = system.physical_values(primal)
        z = [
            (Fraction(values[c.name]) - Fraction(c.lower)) / span
            for c, span in zip(system.variables, system.spans)
        ]
        for row, rhs in zip(system.rows, system.upper):
            residual = exact_dot(row, z) - rhs
            if residual <= 0:
                continue
            for c, span, a in zip(system.variables, system.spans, row):
                if not a:
                    continue
                corrected = Fraction(values[c.name]) - residual * span / a
                if abs(corrected - Fraction(values[c.name])) > Fraction(1e-8) * span:
                    continue  # Repair local numerical error, not an arbitrary dispatch.
                for rounded in (round_down(corrected), round_up(corrected)):
                    candidate = system.checked_values({**values, c.name: rounded})
                    if candidate is not None:
                        controls.append(candidate)
    return controls


def compile_system(claim, point):
    return compile_model(claim.adapter, claim.requirements).normalized(
        point, claim.adapter._permissions(claim)
    )


@dataclass
class RecourseSolution:
    feasibility: str
    values: dict | None
    certificate: dict | None
    attempts: list
    message: str


def solve_system(system, tolerance, *, backend="scipy"):
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
        m, n = len(system.rows), len(system.variables)
        if not m:
            return RecourseSolution(
                "feasible",
                system.candidate([0.5] * n),
                None,
                attempts,
                "Empty row set within the declared control box.",
            )
        # A single impossible row can be resolved without invoking a solver.
        for i in range(m):
            proof = system.certificate([float(j == i) for j in range(m)])
            if proof is not None:
                return infeasible(proof)
        # Satisfied constant rows (including equal bounds on fixed streams)
        # must not cap the phase-I interior margin at zero. They remain in the
        # exact physical system and every candidate/certificate check. Only the
        # numerical feasibility proposal omits these already checked rows.
        active = [i for i, row in enumerate(system.rows) if any(row)]
        if not active:
            return RecourseSolution(
                "feasible",
                system.candidate([0.5] * n),
                None,
                attempts,
                "Constant physical rows checked over the declared control box.",
            )
        if backend == "cvxpy" and n:
            from ._cvxpy_backend import solve_bounded

            proposal = system.select_rows(active)
            candidate = solve_bounded(proposal, [0.0] * n, tolerance, phase_one=True)
            attempts.extend(candidate.attempts)
            values = (
                system.candidate(candidate.point)
                if candidate.point is not None
                else None
            )
            if values is None and candidate.point is not None:
                values = next(iter(checked_candidates(system, candidate.point)), None)
            if values is not None:
                return RecourseSolution(
                    "feasible",
                    values,
                    None,
                    attempts,
                    "Returned physical controls checked with exact rational residuals.",
                )
            multipliers = None
            if candidate.multipliers is not None:
                multipliers = [0.0] * m
                for i, value in zip(active, candidate.multipliers):
                    multipliers[i] = value
            proof = system.certificate(multipliers) if multipliers is not None else None
            if proof is not None:
                return infeasible(proof)
            return RecourseSolution(
                "unknown",
                None,
                None,
                attempts,
                "Neither a physical candidate nor an exact contradiction was verified from the prepared program.",
            )
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
                inequalities=[rows[i] + [-1.0] for i in active],
                upper=[upper[i] for i in active],
            )
            values = system.candidate(primal[:-1]) if primal is not None else None
            if values is None and primal is not None:
                values = next(iter(checked_candidates(system, primal[:-1])), None)
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
