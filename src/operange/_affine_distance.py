"""Affine threshold LPs with exact bounded-box certificates.

The solver proposes points and nonnegative multipliers. All physical targets,
domain rows, distances, lower bounds and infeasibility proofs are checked on
the exact rational values of the declarations. Solver status is not a proof.
"""

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from math import inf, nextafter

from . import linear
from ._numeric import exact_dot, exact_normalized_bounds, round_down, round_up
from .polytope import PolytopeSet
from .primitives import finite


def box_minimum(coefficients, bounds):
    return sum(
        (min(a * lo, a * hi) for a, (lo, hi) in zip(coefficients, bounds)), Fraction(0)
    )


@dataclass
class ThresholdProblem:
    claim: object
    requirement: object
    threshold: float
    parameters: tuple
    bounds: tuple
    domain_rows: list
    domain_upper: list
    rows: list
    upper: list
    row_labels: list
    residual_form: object = None

    @property
    def objective(self):
        return (Fraction(0),) * len(self.parameters) + (Fraction(1),)

    def distance(self, point):
        return max(
            abs(Fraction(point[c.name]) - Fraction(c.nominal)) / Fraction(c.scale)
            for c in self.claim.distance.space.coordinates
        )

    def accepts(self, point):
        if any(not p.lower <= point[p.name] <= p.upper for p in self.parameters):
            return False
        z = [
            (Fraction(point[p.name]) - Fraction(p.nominal)) / Fraction(p.scale)
            for p in self.parameters
        ]
        if any(
            exact_dot(row, z) > b for row, b in zip(self.domain_rows, self.domain_upper)
        ):
            return False
        if self.residual_form is not None:
            return self.residual_form.actual(self.claim, point) >= Fraction(
                self.threshold
            )
        model = self.claim.adapter
        residual = self.requirement.sign * (
            model.output(self.requirement.output)._exact_value(
                {**point, **model._fixed_values(self.claim)}
            )
            - Fraction(self.requirement.limit)
        )
        return residual >= Fraction(self.threshold)

    def physical_candidates(self, coordinates):
        if len(coordinates) != len(self.parameters):
            raise ValueError("wrong number of proposed search coordinates")
        exact = [
            Fraction(p.nominal)
            + Fraction(p.scale) * Fraction(finite(z, "search coordinate"))
            for p, z in zip(self.parameters, coordinates)
        ]
        coefficients = (
            self.residual_form.coefficients
            if self.residual_form is not None
            else {
                t.variable: self.requirement.sign * t.coefficient
                for t in self.claim.adapter.output(self.requirement.output).terms
            }
        )
        bases = []
        for directed in (False, True):
            values = []
            for p, value in zip(self.parameters, exact):
                value = min(Fraction(p.upper), max(Fraction(p.lower), value))
                coefficient = coefficients.get(p.name, 0)
                rounded = (
                    (round_up(value) if coefficient > 0 else round_down(value))
                    if directed and coefficient
                    else float(value)
                )
                values.append(min(p.upper, max(p.lower, rounded)))
            bases.append(dict(zip((p.name for p in self.parameters), values)))
        # Small rounding repairs are candidates only. Every variant must still
        # satisfy the exact polytope and physical target; no tolerance widens them.
        seen = set()
        for point in bases:
            variants = [point]
            for p in self.parameters:
                for direction in (-inf, inf):
                    value = nextafter(point[p.name], direction)
                    if p.lower <= value <= p.upper:
                        variants.append({**point, p.name: value})
            for variant in variants:
                key = tuple(variant[p.name] for p in self.parameters)
                if key not in seen:
                    seen.add(key)
                    yield variant

    def certificate(self, multipliers, *, infeasibility=False):
        if len(multipliers) != len(self.rows):
            raise ValueError("wrong number of proposed dual multipliers")
        multipliers = [max(0.0, finite(v, "dual multiplier")) for v in multipliers]
        weighted = [exact_dot(column, multipliers) for column in zip(*self.rows)]
        objective = (
            (Fraction(0),) * len(self.bounds) if infeasibility else self.objective
        )
        residual = [a + c for a, c in zip(weighted, objective)]
        minimum = box_minimum(residual, self.bounds)
        rhs = exact_dot(self.upper, multipliers)
        bound = minimum - rhs
        return bound, {
            "method": "bounded_box_contradiction"
            if infeasibility
            else "bounded_box_weak_duality",
            "formula": "min_box((lambda A) v) - lambda b > 0"
            if infeasibility
            else "min_box((c + lambda A) v) - lambda b <= minimum_distance",
            "multipliers": multipliers,
            "weighted_residual_exact": list(map(str, residual)),
            "box_minimum_exact": str(minimum),
            "weighted_rhs_exact": str(rhs),
            "bound_exact": str(bound),
        }

    def manifest(self):
        return {
            "requirement": self.requirement.name,
            "threshold": self.threshold,
            "unit": self.residual_form.unit
            if self.residual_form is not None
            else self.claim.adapter.output(self.requirement.output).unit,
            **(
                {
                    "controller_rounding_error_exact": str(self.residual_form.error),
                    "controller_enclosure_evidence_ref": "controller_enclosure",
                }
                if self.residual_form is not None
                else {}
            ),
            "coordinates": [p.name for p in self.parameters] + ["distance_radius"],
            "coordinate_space": "domain_normalized_inputs_and_normalized_linf_radius",
            "origins": [p.nominal for p in self.parameters],
            "scales": [p.scale for p in self.parameters],
            "row_labels": self.row_labels,
            "rows_exact": [[str(a) for a in row] for row in self.rows],
            "rhs_exact": list(map(str, self.upper)),
            "bounds_exact": [[str(lo), str(hi)] for lo, hi in self.bounds],
            "objective_exact": list(map(str, self.objective)),
            "arithmetic": "exact_rationals_of_declared_floats",
        }


def compile_problem(claim, requirement, threshold, *, residual_form=None):
    polytope = claim.domain if type(claim.domain) is PolytopeSet else None
    box = polytope.envelope if polytope is not None else claim.domain
    parameters = box.scalar_parameters
    n = len(parameters)
    domain_rows, domain_upper, labels = [], [], []
    if polytope is not None:
        for constraint in polytope.constraints:
            for sign in (1, -1) if constraint.relation == "eq" else (1,):
                domain_rows.append(
                    tuple(
                        sign * Fraction(constraint.coefficients.get(p.name, 0))
                        for p in parameters
                    )
                )
                domain_upper.append(sign * Fraction(constraint.rhs))
                labels.append(f"domain:{constraint.name}:{sign}")
    rows = [row + (Fraction(0),) for row in domain_rows]
    upper = list(domain_upper)
    radius_min, radius_max = Fraction(0), Fraction(0)
    for i, p in enumerate(parameters):
        distance_coordinate = claim.distance.space.coordinate(p.name)
        origin, scale = (
            Fraction(distance_coordinate.nominal),
            Fraction(distance_coordinate.scale),
        )
        lo, hi = (
            (Fraction(p.lower) - origin) / scale,
            (Fraction(p.upper) - origin) / scale,
        )
        radius_min = max(radius_min, lo, -hi)
        radius_max = max(radius_max, abs(lo), abs(hi))
        weight, offset = (
            Fraction(p.scale) / scale,
            (Fraction(p.nominal) - origin) / scale,
        )
        for sign in (1, -1):
            row = [Fraction(0)] * (n + 1)
            row[i], row[-1] = sign * weight, Fraction(-1)
            rows.append(tuple(row))
            upper.append(-sign * offset)
            labels.append(f"distance:{p.name}:{sign}")
    if residual_form is None:
        model = claim.adapter
        output = model.output(requirement.output)
        coefficients = {
            t.variable: requirement.sign * Fraction(t.coefficient) for t in output.terms
        }
        residual_scale = Fraction(requirement.residual_scale)
        baseline = requirement.sign * (
            output._exact_value({**box.nominal, **model._fixed_values(claim)})
            - Fraction(requirement.limit)
        )
    else:
        coefficients, residual_scale = residual_form.coefficients, Fraction(1)
        # A necessary outer condition supplies lower bounds and unreachability.
        # Candidate acceptance always checks the actual rounded commands.
        baseline = (
            residual_form.offset
            + exact_dot(
                (coefficients.get(p.name, 0) for p in parameters),
                (p.nominal for p in parameters),
            )
            + residual_form.error
        )
    rows.append(
        tuple(
            -coefficients.get(p.name, 0) * Fraction(p.scale) / residual_scale
            for p in parameters
        )
        + (Fraction(0),)
    )
    upper.append((baseline - Fraction(threshold)) / residual_scale)
    labels.append(f"target:{requirement.name}")
    bounds = exact_normalized_bounds(box) + ((radius_min, radius_max),)
    # Explicit bound rows provide the LP dual with all bound multipliers.
    # The exact same box also corrects any residual in its proposed dual equality.
    for i, (lo, hi) in enumerate(bounds):
        row = tuple(Fraction(j == i) for j in range(n + 1))
        rows.extend((row, tuple(-v for v in row)))
        upper.extend((hi, -lo))
        labels.extend((f"bound:{i}:upper", f"bound:{i}:lower"))
    return ThresholdProblem(
        claim,
        requirement,
        threshold,
        parameters,
        bounds,
        domain_rows,
        domain_upper,
        rows,
        upper,
        labels,
        residual_form,
    )


@dataclass
class BranchResult:
    problem: ThresholdProblem
    lower: Fraction
    lower_proof: dict
    unreachable: bool = False
    infeasibility_proof: dict | None = None
    point: dict | None = None
    distance: Fraction | None = None
    evaluation: object = None
    attempts: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)


def solve_branch(problem, evaluate):
    result = BranchResult(
        problem, problem.bounds[-1][0], {"method": "distance_to_enclosing_box"}
    )

    def consider(point):
        if not problem.accepts(point):
            return
        evaluation = evaluate(point)
        if (
            evaluation.execution != "completed"
            or evaluation.payload.membership.status != "inside"
        ):
            return
        distance = problem.distance(point)
        # Ensure the candidate distance can be exported before retaining it.
        round_up(distance)
        if result.distance is None or distance < result.distance:
            result.point, result.distance, result.evaluation = (
                point,
                distance,
                evaluation,
            )

    def solve(purpose, *args, **kwargs):
        point, evidence = linear.solve_lp(*args, tolerance=1e-9, **kwargs)
        result.attempts.append({"purpose": purpose, **asdict(evidence)})
        return point

    try:
        consider(problem.claim.domain.feasible_point)
        consider(
            {
                p.name: min(
                    p.upper,
                    max(
                        p.lower, problem.claim.distance.space.coordinate(p.name).nominal
                    ),
                )
                for p in problem.parameters
            }
        )
        if result.distance == result.lower:
            return result
        for i, (row, rhs) in enumerate(zip(problem.rows, problem.upper)):
            if box_minimum(row, problem.bounds) > rhs:
                gap, proof = problem.certificate(
                    [float(j == i) for j in range(len(problem.rows))],
                    infeasibility=True,
                )
                if gap > 0:
                    result.unreachable, result.infeasibility_proof = True, proof
                    return result
        rows = [
            [finite(float(a), "LP coefficient") for a in row] for row in problem.rows
        ]
        rhs = [finite(float(b), "LP rhs") for b in problem.upper]
        bounds = [(round_down(lo), round_up(hi)) for lo, hi in problem.bounds]
        objective = [float(v) for v in problem.objective]
        primal = solve(
            "threshold_candidate", objective, bounds, inequalities=rows, upper=rhs
        )
        if primal is not None:
            for point in problem.physical_candidates(primal[:-1]):
                consider(point)
        dual = solve(
            "distance_lower_bound",
            rhs,
            [(0, None)] * len(rows),
            equalities=[list(column) for column in zip(*rows)],
            rhs=[-v for v in objective],
        )
        if dual is not None:
            lower, proof = problem.certificate(dual)
            if lower > result.lower:
                result.lower, result.lower_proof = lower, proof
        if result.distance is None:
            multipliers = solve(
                "unreachability_candidate",
                rhs,
                [(0, 1)] * len(rows),
                equalities=[list(column) for column in zip(*rows)]
                + [[1.0] * len(rows)],
                rhs=[0.0] * len(bounds) + [1.0],
            )
            if multipliers is not None:
                gap, proof = problem.certificate(multipliers, infeasibility=True)
                if gap > 0:
                    result.unreachable, result.infeasibility_proof = True, proof
    except (ValueError, OverflowError) as exc:
        result.diagnostics.append(str(exc))
    return result
