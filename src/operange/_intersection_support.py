"""CVXPY proposes a support decomposition; primitive bounds prove it.

For each factor, y_i = alpha_i * z + beta_i. Any local vectors a_i give
    h(w) <= sum(h_i(a_i) - a_i.beta_i) + h_box(w - sum(alpha_i*a_i)).
The final residual is bounded using a finite enclosure, in exact arithmetic.
No conic solver status, dual feasibility tolerance or cone factorization is
itself a certificate. Only explicitly supported built-in convex sets compile.
"""

from fractions import Fraction as F
from importlib.util import find_spec

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from ._cvxpy_backend import (
    BackendUnavailable,
    cached_program,
    load_cvxpy,
    numeric_vector,
)
from ._geometry import LinearSupport, require_normalization
from ._numeric import exact_dot, exact_normalized_bounds, normalized_score, sqrt_upper
from .composition import Intersection, Product
from .geometries import BudgetSet, EllipsoidSet, SimplexSet
from .polytope import PolytopeSet
from .primitives import BoxSet, finite


def leaves(domain):
    if type(domain) in (Intersection, Product):
        return tuple(leaf for factor in domain.factors for leaf in leaves(factor))
    if type(domain) in (BoxSet, PolytopeSet, BudgetSet, SimplexSet, EllipsoidSet):
        if (
            type(domain) in (PolytopeSet, BudgetSet)
            and type(domain.envelope) is not BoxSet
        ):
            raise ValueError(
                "convex intersection support requires built-in box envelopes"
            )
        return (domain,)
    raise ValueError("convex intersection support does not support this factor type")


def supported(domain):
    try:
        require_normalization(domain.space)
        for factor in leaves(domain):
            require_normalization(factor.space)
        return find_spec("cvxpy") is not None
    except (ValueError, ImportError):
        return False


def transform(space, factor):
    alpha, beta = [], []
    for local in factor.space.coordinates:
        joint = space.coordinate(local.name)
        alpha.append(F(joint.scale) / F(local.scale))
        beta.append((F(joint.nominal) - F(local.nominal)) / F(local.scale))
    return alpha, beta


def enclosure(domain, factors):
    bounds = {}
    for factor in factors:
        if type(factor) in (BoxSet, PolytopeSet, BudgetSet):
            box = factor if type(factor) is BoxSet else factor.envelope
            physical = {p.name: (F(p.lower), F(p.upper)) for p in box.scalar_parameters}
        elif type(factor) is SimplexSet:
            physical = dict.fromkeys(factor.space.names, (F(0), F(1)))
        else:
            physical = {}
            for c in factor.space.coordinates:
                radius = F(
                    sqrt_upper(F(factor.radius) ** 2 * F(factor.shape[c.name][c.name]))
                )
                physical[c.name] = (
                    F(c.nominal) - F(c.scale) * radius,
                    F(c.nominal) + F(c.scale) * radius,
                )
        for name, (lo, hi) in physical.items():
            c = domain.space.coordinate(name)
            lo, hi = (lo - F(c.nominal)) / F(c.scale), (hi - F(c.nominal)) / F(c.scale)
            if name in bounds:
                lo, hi = max(lo, bounds[name][0]), min(hi, bounds[name][1])
            if lo > hi:
                raise ValueError("factor enclosures have no common point")
            bounds[name] = lo, hi
    return bounds


def _quadratic_form(matrix, z):
    """Solve S v = z over rationals, then evaluate z.v exactly."""
    rows = [[F(v) for v in row] + [zi] for row, zi in zip(matrix, z)]
    n = len(z)
    for i in range(n):
        pivot = rows[i][i]
        if pivot <= 0:
            raise ValueError("ellipsoid is not exactly positive definite")
        for j in range(i + 1, n):
            ratio = rows[j][i] / pivot
            for k in range(i, n + 1):
                rows[j][k] -= ratio * rows[i][k]
    solution = [F(0)] * n
    for i in reversed(range(n)):
        solution[i] = (
            rows[i][-1] - sum(rows[i][j] * solution[j] for j in range(i + 1, n))
        ) / rows[i][i]
    return exact_dot(z, solution)


def exact_member(factors, point):
    """Check original physical declarations, including equality and cone rows."""
    for factor in factors:
        values = {n: F(point[n]) for n in factor.space.names}
        z = {
            c.name: (values[c.name] - F(c.nominal)) / F(c.scale)
            for c in factor.space.coordinates
        }
        if type(factor) in (BoxSet, PolytopeSet, BudgetSet):
            box = factor if type(factor) is BoxSet else factor.envelope
            if any(
                not F(p.lower) <= values[p.name] <= F(p.upper)
                for p in box.scalar_parameters
            ):
                return False
        if type(factor) is PolytopeSet:
            for row in factor.constraints:
                residual = exact_dot(
                    row.coefficients.values(), (z[n] for n in row.coefficients)
                ) - F(row.rhs)
                violated = residual != 0 if row.relation == "eq" else residual > 0
                if violated:
                    return False
        elif type(factor) is BudgetSet:
            if any(abs(v) > 1 for v in z.values()) or sum(map(abs, z.values())) > F(
                factor.budget
            ):
                return False
        elif type(factor) is SimplexSet:
            if any(v < 0 for v in values.values()) or sum(values.values()) != 1:
                return False
        elif type(factor) is EllipsoidSet:
            if (
                _quadratic_form(factor._matrix(), list(z.values()))
                > F(factor.radius) ** 2
            ):
                return False
    return True


def factor_constraints(cp, factor, y):
    constraints = []
    if type(factor) in (BoxSet, PolytopeSet, BudgetSet):
        box = factor if type(factor) is BoxSet else factor.envelope
        bounds = exact_normalized_bounds(box)
        constraints += [
            y >= [float(lo) for lo, _ in bounds],
            y <= [float(hi) for _, hi in bounds],
        ]
    if type(factor) is PolytopeSet:
        for row in factor.constraints:
            expression = (
                np.array([row.coefficients.get(n, 0) for n in factor.space.names]) @ y
            )
            constraints.append(
                expression == row.rhs if row.relation == "eq" else expression <= row.rhs
            )
    elif type(factor) is BudgetSet:
        constraints += [y >= -1, y <= 1, cp.norm1(y) <= factor.budget]
    elif type(factor) is SimplexSet:
        physical = np.array(
            [c.nominal for c in factor.space.coordinates]
        ) + cp.multiply([c.scale for c in factor.space.coordinates], y)
        constraints += [physical >= 0, cp.sum(physical) == 1]
    elif type(factor) is EllipsoidSet:
        inverse = solve_triangular(
            cholesky(factor._matrix(), lower=True),
            np.eye(len(factor.space.names)),
            lower=True,
        )
        constraints.append(cp.norm(inverse @ y, 2) <= factor.radius)
    return constraints


def domain_constraints(cp, space, z, factors):
    """Shared convex-domain compiler for support and distance programs."""
    links, constraints = [], []
    for factor in factors:
        alpha, beta = transform(space, factor)
        y = cp.Variable(len(factor.space.names))
        indices = [space.names.index(n) for n in factor.space.names]
        link = y == cp.multiply(list(map(float, alpha)), z[indices]) + list(
            map(float, beta)
        )
        links.append(link)
        constraints += [link, *factor_constraints(cp, factor, y)]
    return links, constraints


class PreparedIntersection:
    def __init__(self, domain, factors):
        cp = self.cp = load_cvxpy()
        self.z = cp.Variable(len(domain.space.names), name="joint_coordinates")
        self.objective = cp.Parameter(len(domain.space.names), name="support_direction")
        self.links, constraints = domain_constraints(cp, domain.space, self.z, factors)
        self.problem = cp.Problem(cp.Maximize(self.objective @ self.z), constraints)
        if not self.problem.is_dcp(dpp=True):
            raise ValueError("intersection support program must be DPP-compliant")

    def solve(self, w, tolerance):
        cp = self.cp
        self.objective.value = list(w.values())
        self.z.value = None
        report = {
            "backend": f"CVXPY {cp.__version__} / CLARABEL",
            "status": "unresolved",
            "compilation": "DPP; fixed geometry with parameterized objective",
        }
        try:
            self.problem.solve(
                solver="CLARABEL",
                enforce_dpp=True,
                warm_start=True,
                tol_gap_abs=tolerance,
                tol_gap_rel=tolerance,
                tol_feas=tolerance,
            )
            report["status"] = str(self.problem.status)
            if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return None, None, report
            point = numeric_vector(self.z.value, len(w), "support candidate")
            splits = [
                numeric_vector(link.dual_value, link.size, "support split")
                for link in self.links
            ]
            if any(a is None for a in splits):
                splits = None
            else:
                splits = [[-v for v in a] for a in splits]
            return point, splits, report
        except (cp.error.SolverError, ValueError, OverflowError) as exc:
            report.update(status="unresolved", message=str(exc))
            return None, None, report


def decomposition_bound(domain, factors, w, splits, bounds):
    if len(splits) != len(factors):
        raise ValueError("support split count does not match factors")
    residual, upper, records = {n: F(v) for n, v in w.items()}, F(0), []
    for factor, split in zip(factors, splits):
        a = numeric_vector(split, len(factor.space.names), "support split")
        if a is None:
            raise ValueError("missing support split")
        local_weights = dict(zip(factor.space.names, a))
        support = factor.maximize_linear(local_weights)
        if (
            not isinstance(support, LinearSupport)
            or support.domain_ref != factor.ref
            or support.coefficients != local_weights
            or support.status not in ("optimal", "bounded")
            or support.upper is None
        ):
            raise ValueError("factor did not supply a matching checked support bound")
        alpha, beta = transform(domain.space, factor)
        upper += F(finite(support.upper, "factor bound")) - exact_dot(a, beta)
        for n, v, scale in zip(factor.space.names, a, alpha):
            residual[n] -= scale * F(v)
        records.append(
            {
                "support": support.to_dict(),
                "coordinates": list(factor.space.names),
                "alpha_exact": list(map(str, alpha)),
                "beta_exact": list(map(str, beta)),
            }
        )
    correction = sum(
        max(r * bounds[n][0], r * bounds[n][1]) for n, r in residual.items()
    )
    return upper + correction, {
        "factors": records,
        "residual_exact": {n: str(v) for n, v in residual.items()},
        "correction_exact": str(correction),
    }


def maximize(domain, w):
    try:
        factors = leaves(domain)
        require_normalization(domain.space)
        for factor in factors:
            require_normalization(factor.space)
        load_cvxpy()
    except (BackendUnavailable, ValueError) as exc:
        return domain._unresolved(w, str(exc), status="unsupported")
    try:
        if not exact_member(factors, domain.feasible_point):
            raise ValueError(
                "declared feasible_point could not be checked in exact arithmetic"
            )
        bounds = enclosure(domain, factors)
        upper = sum(max(F(w[n]) * lo, F(w[n]) * hi) for n, (lo, hi) in bounds.items())
        details = {"fallback": "finite enclosure of all factors"}
        # Include coordinate order: manifests can canonicalize box parameter order.
        key = (
            "intersection",
            domain.ref.digest,
            domain.space.names,
            tuple(f.space.names for f in factors),
        )
        program = cached_program(key, lambda: PreparedIntersection(domain, factors))
        z, splits, report = program.solve(w, domain.tolerance)
        point = domain.feasible_point
        if z is not None:
            candidate = {
                c.name: finite(
                    float(F(c.nominal) + F(c.scale) * F(v)), "support witness"
                )
                for c, v in zip(domain.space.coordinates, z)
            }
            for shrink in (1.0, 1 - 1e-14, 1 - 1e-12, 1 - 1e-10):
                trial = {n: point[n] + shrink * (candidate[n] - point[n]) for n in w}
                if (
                    exact_member(factors, trial)
                    and domain.membership(trial).status == "inside"
                ):
                    point = trial
                    break
        if splits is not None:
            try:
                candidate_upper, proof = decomposition_bound(
                    domain, factors, w, splits, bounds
                )
                if (
                    candidate_upper >= normalized_score(domain.space, w, point)
                    and candidate_upper < upper
                ):
                    upper, details = candidate_upper, proof
            except (ValueError, OverflowError) as exc:
                details["split_rejected"] = str(exc)
        details.update(
            solver=report,
            enclosure_exact={n: list(map(str, b)) for n, b in bounds.items()},
            upper_exact=str(upper),
            arithmetic="exact_rationals_of_declared_floats",
            formula="sum(h_i(a_i) - a_i.beta_i) + h_box(w - sum(alpha_i*a_i))",
        )
        return domain._support(
            w, point, upper, "factor_support_decomposition", details=details
        )
    except (ValueError, OverflowError) as exc:
        return domain._unresolved(w, str(exc))
