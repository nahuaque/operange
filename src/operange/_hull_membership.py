"""Checked barycentric or separating certificates for a declared convex hull."""

from fractions import Fraction

from . import linear
from ._numeric import exact_dot
from .contract_types import Evidence
from .domains import MembershipCheck


def _barycentric(rows, rhs, candidate):
    """Recover exact weights on a numerical LP's proposed basic support.

    Gaussian elimination is certificate reconstruction, not an optimization
    algorithm. Inconsistent or negative reconstructions are discarded.
    """
    if candidate is None:
        return None
    columns = [i for i, value in enumerate(candidate) if value > 0]
    if not columns:
        return None
    matrix = [[row[i] for i in columns] + [b] for row, b in zip(rows, rhs)]
    pivots = []
    for column in range(len(columns)):
        pivot = next(
            (i for i in range(len(pivots), len(matrix)) if matrix[i][column]), None
        )
        if pivot is None:
            continue
        index = len(pivots)
        matrix[index], matrix[pivot] = matrix[pivot], matrix[index]
        divisor = matrix[index][column]
        matrix[index] = [v / divisor for v in matrix[index]]
        for i in range(len(matrix)):
            if i != index:
                factor = matrix[i][column]
                matrix[i] = [a - factor * b for a, b in zip(matrix[i], matrix[index])]
        pivots.append(column)
    if any(not any(row[:-1]) and row[-1] for row in matrix):
        return None
    result = [Fraction(0)] * len(candidate)
    for row, column in zip(matrix, pivots):
        result[columns[column]] = row[-1]
    if any(v < 0 for v in result) or any(
        exact_dot(row, result) != b for row, b in zip(rows, rhs)
    ):
        return None
    return result


def membership(domain, values):
    def record(status, method, details):
        return MembershipCheck(
            status,
            domain.ref,
            values,
            (
                Evidence(
                    "membership",
                    "domain_membership",
                    method,
                    "unresolved" if status == "unknown" else "verified",
                    details={
                        "arithmetic": "exact_rationals_of_declared_floats",
                        **details,
                    },
                ),
            ),
        )

    matches = [s.name for s in domain.vertices if dict(s.values) == values]
    if matches:
        return record(
            "inside", "declared_hull_generator", {"matching_generators": matches}
        )
    outside = [n for n, (lo, hi) in domain._bounds.items() if not lo <= values[n] <= hi]
    if outside:
        return record(
            "outside", "hull_coordinate_bounds", {"outside_coordinates": outside}
        )
    try:
        # Translate and scale by exact hull spans, without requiring the caller
        # to declare nominal/scaling metadata for physical hull membership.
        rows, rhs = [], []
        for name, (low, high) in domain._bounds.items():
            span = Fraction(high) - Fraction(low)
            if span:
                rows.append(
                    [
                        (Fraction(s.values[name]) - Fraction(low)) / span
                        for s in domain.vertices
                    ]
                )
                rhs.append((Fraction(values[name]) - Fraction(low)) / span)
        equalities = rows + [[Fraction(1)] * len(domain.vertices)]
        targets = rhs + [Fraction(1)]
        candidate, _ = linear.solve_lp(
            [0.0] * len(domain.vertices),
            [(0, 1)] * len(domain.vertices),
            domain.tolerance,
            equalities=[list(map(float, row)) for row in equalities],
            rhs=list(map(float, targets)),
        )
        weights = _barycentric(equalities, targets, candidate)
        if weights is not None:
            return record(
                "inside",
                "exact_barycentric_combination",
                {
                    "weights_exact": {
                        s.name: str(w) for s, w in zip(domain.vertices, weights)
                    },
                    "formula": "weights >= 0; sum(weights) = 1; sum(weights * generators) = realization",
                },
            )
        # A separating direction is merely a candidate until it strictly
        # separates the point from EVERY generator in exact arithmetic.
        direction, _ = linear.solve_lp(
            [-float(v) for v in rhs] + [1.0],
            [(-1, 1)] * len(rows) + [(None, None)],
            domain.tolerance,
            inequalities=[list(map(float, column)) + [-1.0] for column in zip(*rows)],
            upper=[0.0] * len(domain.vertices),
        )
        if direction is not None:
            w = direction[:-1]
            upper = max(exact_dot(w, column) for column in zip(*rows))
            score = exact_dot(w, rhs)
            if score > upper:
                return record(
                    "outside",
                    "exact_hull_separation",
                    {
                        "normalized_direction": w,
                        "normalization": "coordinate origins and nonzero spans of generator bounds",
                        "point_score_exact": str(score),
                        "hull_upper_exact": str(upper),
                        "separation_gap_exact": str(score - upper),
                    },
                )
        message = (
            "Neither exact barycentric weights nor a strict separation was verified."
        )
    except (ValueError, OverflowError) as exc:
        message = str(exc)
    return record("unknown", "unresolved_hull_membership", {"message": message})
