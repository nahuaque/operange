"""Convex threshold distances using the shared domain compiler and support proofs.

For lambda >= 0 and dual-norm(v) <= 1, a.z >= b implies
    ||M z + d|| >= lambda*b + v.d - h_D(lambda*a - M.T*v).
Any proposed factor split bounds the support on the right independently of
conic dual feasibility. Original domain membership and target replay own the
upper bound; a solver radius or an infeasible termination never does.
"""

from fractions import Fraction as F
from math import inf, nextafter

import numpy as np

from ._affine_distance import BranchResult
from ._cvxpy_backend import cached_program, load_cvxpy, numeric_vector
from ._geometry import require_normalization
from ._intersection_support import (
    decomposition_bound,
    domain_constraints,
    enclosure,
    exact_member,
    leaves,
)
from ._numeric import exact_dot, round_down, round_up, sqrt_upper
from .composition import Intersection
from .distance import NormalizedL2, NormalizedLInf
from .primitives import finite


def supported(domain, metric):
    if type(metric) not in (NormalizedLInf, NormalizedL2):
        return False
    try:
        require_normalization(domain.space)
        for factor in leaves(domain):
            require_normalization(factor.space)
        return True
    except ValueError:
        return False


class PreparedDistance:
    def __init__(self, problem):
        cp = self.cp = load_cvxpy()
        n = len(problem.a)
        self.z = cp.Variable(n, name="domain_coordinates")
        displacement = cp.Variable(n, name="scaled_displacement")
        self.rhs = cp.Parameter(name="physical_target")
        self.links, constraints = domain_constraints(
            cp, problem.claim.domain.space, self.z, problem.factors
        )
        self.distance_link = displacement == cp.multiply(
            list(map(float, problem.scales)), self.z
        ) + list(map(float, problem.offsets))
        self.target = np.array(list(map(float, problem.a))) @ self.z >= self.rhs
        norm = (
            cp.norm2(displacement) if problem.euclidean else cp.norm_inf(displacement)
        )
        self.problem = cp.Problem(
            cp.Minimize(norm), constraints + [self.distance_link, self.target]
        )
        if not self.problem.is_dcp(dpp=True):
            raise ValueError("distance program must be DPP-compliant")

    def solve(self, rhs):
        cp = self.cp
        self.rhs.value = finite(float(rhs), "target right-hand side")
        self.z.value = None
        report = {
            "purpose": "threshold_distance_candidate",
            "backend": f"CVXPY {cp.__version__} / CLARABEL",
            "status": "unresolved",
            "compilation": "DPP; fixed domain and metric with parameterized threshold",
        }
        try:
            self.problem.solve(
                solver="CLARABEL",
                enforce_dpp=True,
                warm_start=True,
                tol_gap_abs=1e-10,
                tol_gap_rel=0.0,
                tol_feas=1e-10,
            )
            report["status"] = str(self.problem.status)
            if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return None, None, report
            point = numeric_vector(self.z.value, self.z.size, "distance candidate")
            v = numeric_vector(self.distance_link.dual_value, self.z.size, "norm dual")
            splits = [
                numeric_vector(link.dual_value, link.size, "domain split")
                for link in self.links
            ]
            dual = None
            if v is not None and all(a is not None for a in splits):
                dual = (
                    finite(self.target.dual_value, "target multiplier"),
                    [-a for a in v],
                    [[-a for a in split] for split in splits],
                )
            return point, dual, report
        except (cp.error.SolverError, ValueError, OverflowError) as exc:
            report.update(status="unresolved", message=str(exc))
            return None, None, report


class ConvexThresholdProblem:
    def __init__(self, claim, requirement, threshold, *, residual_form=None):
        self.claim, self.requirement, self.threshold = claim, requirement, threshold
        self.residual_form = residual_form
        self.factors = leaves(claim.domain)
        self.bounds = enclosure(claim.domain, self.factors)
        self.euclidean = type(claim.distance) is NormalizedL2
        if not exact_member(self.factors, claim.domain.feasible_point):
            raise ValueError("domain feasible point failed exact membership")
        space = claim.domain.space
        if residual_form is None:
            output = claim.adapter.output(requirement.output)
            self.coefficients = {
                t.variable: requirement.sign * F(t.coefficient) for t in output.terms
            }
            self.baseline = requirement.sign * (
                output._exact_value(
                    {**space.nominal, **claim.adapter._fixed_values(claim)}
                )
                - F(requirement.limit)
            )
            self.unit, scale = output.unit, F(requirement.residual_scale)
        else:
            self.coefficients = residual_form.coefficients
            self.baseline = (
                residual_form.offset
                + exact_dot(
                    (self.coefficients.get(n, 0) for n in space.names),
                    space.nominal.values(),
                )
                + residual_form.error
            )
            self.unit, scale = residual_form.unit, F(1)
        self.a = tuple(
            self.coefficients.get(c.name, 0) * F(c.scale) / scale
            for c in space.coordinates
        )
        self.b = (F(threshold) - self.baseline) / scale
        self.scales, self.offsets = [], []
        for c in space.coordinates:
            d = claim.distance.space.coordinate(c.name)
            self.scales.append(F(c.scale) / F(d.scale))
            self.offsets.append((F(c.nominal) - F(d.nominal)) / F(d.scale))

    def accepts(self, point):
        if not exact_member(self.factors, point):
            return False
        if self.residual_form is not None:
            return self.residual_form.actual(self.claim, point) >= F(self.threshold)
        values = {**point, **self.claim.adapter._fixed_values(self.claim)}
        residual = self.requirement.sign * (
            self.claim.adapter.output(self.requirement.output)._exact_value(values)
            - F(self.requirement.limit)
        )
        return residual >= F(self.threshold)

    def distance_details(self, point):
        values = [
            (F(point[c.name]) - F(c.nominal)) / F(c.scale)
            for c in self.claim.distance.space.coordinates
        ]
        if self.euclidean:
            squared = sum(v * v for v in values)
            return F(sqrt_upper(squared)), {
                "distance_squared_exact": str(squared),
                "distance_arithmetic": "exact squared distance; square root rounded upward",
            }
        return max(map(abs, values)), {
            "distance_arithmetic": "exact normalized L-infinity distance"
        }

    def manifest(self):
        return {
            "requirement": self.requirement.name,
            "threshold": self.threshold,
            "unit": self.unit,
            "coordinates": list(self.claim.domain.space.names),
            "metric": self.claim.distance.to_manifest(),
            "target_coefficients_exact": list(map(str, self.a)),
            "target_rhs_exact": str(self.b),
            "distance_scales_exact": list(map(str, self.scales)),
            "distance_offsets_exact": list(map(str, self.offsets)),
            "enclosure_exact": {
                n: list(map(str, bounds)) for n, bounds in self.bounds.items()
            },
            "factor_refs": [f.ref.to_dict() for f in self.factors],
            "controller_rounding_error_exact": str(self.residual_form.error)
            if self.residual_form is not None
            else "0",
            "arithmetic": "exact_rationals_of_declared_floats; support and norm bounds rounded outward",
        }

    def lower_bound(self, dual):
        multiplier, proposed, splits = dual
        multiplier = F(max(0.0, finite(multiplier, "target multiplier")))
        values = numeric_vector(proposed, len(self.a), "norm dual")
        if values is None or splits is None:
            raise ValueError("missing norm dual or support splits")
        v = [F(x) for x in values]
        # Project into the dual unit ball with an exact, conservative divisor.
        norm = (
            F(sqrt_upper(sum(x * x for x in v))) if self.euclidean else sum(map(abs, v))
        )
        v = [x / max(F(1), norm) for x in v]
        w = {
            n: multiplier * a - scale * x
            for n, a, scale, x in zip(
                self.claim.domain.space.names, self.a, self.scales, v
            )
        }
        upper, proof = decomposition_bound(
            self.claim.domain, self.factors, w, splits, self.bounds
        )
        lower = multiplier * self.b + exact_dot(v, self.offsets) - upper
        return lower, {
            "formula": "lambda*b + v.d - h_D(lambda*a - M.T*v)",
            "target_multiplier_exact": str(multiplier),
            "norm_dual_exact": list(map(str, v)),
            "support_direction_exact": {n: str(a) for n, a in w.items()},
            "support_upper_exact": str(upper),
            "support_certificate": proof,
            "lower_exact": str(lower),
        }

    def physical_candidates(self, z):
        if z is None:
            return
        space = self.claim.domain.space
        z = numeric_vector(z, len(space.names), "distance candidate")
        base = {
            c.name: finite(float(F(c.nominal) + F(c.scale) * F(v)), "distance witness")
            for c, v in zip(space.coordinates, z)
        }
        for c in space.coordinates:
            lo, hi = self.bounds[c.name]
            lo = F(c.nominal) + F(c.scale) * lo
            hi = F(c.nominal) + F(c.scale) * hi
            base[c.name] = min(round_down(hi), max(round_up(lo), base[c.name]))
        yield base
        for n in space.names:
            for direction in (-inf, inf):
                yield {
                    **base,
                    n: finite(nextafter(base[n], direction), "nearby witness"),
                }
        seed = self.claim.domain.feasible_point
        for fraction in (1 - 1e-14, 1 - 1e-12, 1 - 1e-10):
            yield {
                n: float(F(seed[n]) + F(fraction) * (F(base[n]) - F(seed[n])))
                for n in space.names
            }

    def solve(self, evaluate):
        result = BranchResult(self, F(0), {"method": "norm_nonnegativity"})

        def consider(point):
            if not self.accepts(point):
                return
            evaluation = evaluate(point)
            if (
                evaluation.execution != "completed"
                or evaluation.payload.membership.status != "inside"
            ):
                return
            distance, _ = self.distance_details(point)
            round_up(distance)
            if result.distance is None or distance < result.distance:
                result.point, result.distance, result.evaluation = (
                    point,
                    distance,
                    evaluation,
                )

        try:
            consider(self.claim.domain.feasible_point)
            consider(self.claim.distance.space.nominal)
            if result.distance == 0:
                return result
            key = (
                "threshold_distance",
                self.claim.domain.ref.digest,
                self.claim.domain.space.names,
                tuple(f.space.names for f in self.factors),
                self.euclidean,
                tuple(self.a),
                tuple(self.scales),
                tuple(self.offsets),
            )
            program = cached_program(key, lambda: PreparedDistance(self))
            z, dual, report = program.solve(self.b)
            result.attempts.append(report)
            for point in self.physical_candidates(z):
                consider(point)
            if dual is not None:
                lower, proof = self.lower_bound(dual)
                if lower > result.lower:
                    result.lower, result.lower_proof = lower, proof
            if result.distance is None:
                # Existing support machinery bounds the physical target and can
                # provide a feasible target witness, even after a failed solve.
                domain = Intersection(
                    (self.claim.domain,),
                    backend="cvxpy",
                    feasible_point=self.claim.domain.feasible_point,
                )
                rounded = {
                    n: finite(float(a), "target coefficient")
                    for n, a in zip(domain.space.names, self.a)
                }
                support = domain.maximize_linear(rounded)
                if support.status in ("optimal", "bounded"):
                    correction = sum(
                        max((a - F(rounded[n])) * lo, (a - F(rounded[n])) * hi)
                        for n, a, (lo, hi) in zip(
                            domain.space.names,
                            self.a,
                            (self.bounds[n] for n in domain.space.names),
                        )
                    )
                    upper = F(support.upper) + correction
                    if upper < self.b:
                        result.unreachable = True
                        result.infeasibility_proof = {
                            "method": "target_support_below_threshold",
                            "support": support.to_dict(),
                            "correction_exact": str(correction),
                            "upper_exact": str(upper),
                            "rhs_exact": str(self.b),
                        }
                    else:
                        consider(support.point)
                        # Convex interpolation moves a near-optimal numerical
                        # proposal toward a checked target point. Recheck floats.
                        for proposed in self.physical_candidates(z):
                            for t in (1e-14, 1e-12, 1e-10, 1e-8):
                                consider(
                                    {
                                        n: float(
                                            (1 - F(t)) * F(proposed[n])
                                            + F(t) * F(support.point[n])
                                        )
                                        for n in domain.space.names
                                    }
                                )
            if result.distance is not None and result.lower > result.distance:
                result.lower, result.lower_proof = (
                    F(0),
                    {"method": "norm_nonnegativity"},
                )
                result.diagnostics.append(
                    "Rejected lower certificate inconsistent with the physical witness"
                )
        except (ValueError, OverflowError) as exc:
            result.diagnostics.append(str(exc))
        return result
