"""Bounded coupled engineering design experiment; not a general DSL optimizer.

Run against the sibling source using PYTHONPATH=../residopt/src and
PYTHONDONTWRITEBYTECODE=1. All equipment and response coefficients are synthetic.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import import_module
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter
import warnings

import numpy as np
import scipy
from scipy.optimize import linprog

from operange import Coordinate, EllipsoidSet, ParameterSpace
from operange.experimental._quadratic_bounds import PreparedTrustRegion
from operange.experimental._residopt_quadratic import (
    BackendUnavailable,
    backend_identity,
    load_backend,
)
from operange.experimental.quadratic import (
    QuadraticModel,
    QuadraticProcessAdapter,
)


TOLERANCE = 1e-6
RESERVE = 1e-5
SOLVE_TOLERANCE = 1e-7
INNER_TOLERANCE = 1e-9
OBJECTIVE_GAP_KG = 0.02
LOWER = np.zeros(3)
UPPER = np.full(3, 4.0)
FOOTPRINT = np.array([1.0, 0.4, 0.6])
FOOTPRINT_LIMIT = 4.8
DESIGN_NAMES = ("exchanger_area", "cooler_rating", "pump_rating")
DESIGN_UNITS = ("m2", "kW", "kW")
DESIGN_SCALES = (10.0, 100.0, 10.0)
WORKLOAD = (
    (120.0, 90.0, 60.0),
    (30.0, 200.0, 60.0),
    (240.0, 30.0, 60.0),
    (120.0, 90.0, 240.0),
    (120.0, 300.0, 30.0),
    (60.0, 60.0, 120.0),
)


@dataclass(frozen=True)
class Response:
    name: str
    unit: str
    scale: float
    H: tuple
    S: tuple
    s: tuple
    r: tuple
    offset: float

    def arrays(self):
        return tuple(
            np.asarray(v, dtype=float) for v in (self.H, self.S, self.s, self.r)
        )


RESPONSES = (
    Response(
        "outlet_temperature_excess",
        "K",
        5.0,
        ((0.3, 0.1), (0.1, 0.4)),
        ((-0.2, 0.0, 0.0), (0.0, -0.25, 0.0)),
        (0.7, 0.8),
        (-2.0, -1.2, 0.3),
        4.0,
    ),
    Response(
        "pressure_drop_excess",
        "kPa",
        10.0,
        ((0.5, 0.08), (0.08, 0.1)),
        ((0.12, 0.0, -0.15), (0.0, 0.1, 0.0)),
        (1.0, 0.2),
        (0.5, 0.2, -2.0),
        3.0,
    ),
    Response(
        "cooling_duty_shortfall",
        "kW",
        100.0,
        ((0.2, -0.05), (-0.05, -0.1)),
        ((-0.1, 0.0, 0.0), (0.0, -0.2, 0.0)),
        (0.3, 0.7),
        (-0.5, -1.8, 0.15),
        3.4,
    ),
)


def domain():
    space = ParameterSpace(
        (
            Coordinate("flow", "kg/s", "Synthetic feed flow", 100.0, 10.0),
            Coordinate(
                "feed_temperature", "degC", "Synthetic feed temperature", 50.0, 5.0
            ),
        )
    )
    return EllipsoidSet(
        space,
        {
            "flow": {"flow": 1.0, "feed_temperature": 0.35},
            "feed_temperature": {"flow": 0.35, "feed_temperature": 1.0},
        },
        radius=1.2,
    )


def ball_map():
    declared = domain()
    names = declared.space.names
    shape = np.array([[declared.shape[a][b] for b in names] for a in names])
    return declared.radius * np.linalg.cholesky(shape)


def vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return result.copy()


def material_weights(value):
    result = vector(value, "material weights")
    if np.any(result <= 0):
        raise ValueError("material weights must be positive")
    return result


def fixed_model(response, design):
    x = vector(design, "design")
    H, S, s, r = response.arrays()
    names = domain().space.names
    return QuadraticModel(
        response.name,
        domain().space,
        response.unit,
        "engineering_residual",
        response.scale * float(r @ x + response.offset),
        dict(zip(names, response.scale * (S @ x + s))),
        {
            a: {b: float(response.scale * H[i, j]) for j, b in enumerate(names)}
            for i, a in enumerate(names)
        },
        0.0,
        "Synthetic local design-response equations; no physical validation inferred",
        tolerance=TOLERANCE * response.scale,
        residual_scale=response.scale,
    )


def audit_design(design, weights, *, include_bundles=True):
    """Freeze every response at the returned design and audit the physical domain."""
    x, weights = vector(design, "design"), material_weights(weights)
    results, worst = {}, {}
    for response in RESPONSES:
        result = (
            QuadraticProcessAdapter(fixed_model(response, x), backend="trust_region")
            .as_claim(domain())
            .audit_result()
        )
        proof = next(e for e in result.evidence if e.subject == "requirement_bound")
        upper = next(
            (m.value for m in proof.measurements if m.name == "residual_upper"), None
        )
        worst[response.name] = {
            "verdict": result.payload.verdict,
            "upper": upper,
            "unit": response.unit,
        }
        results[response.name] = result.to_dict() if include_bundles else None
    base_residuals = np.r_[LOWER - x, x - UPPER, FOOTPRINT @ x - FOOTPRINT_LIMIT]
    base_ok = bool(np.max(base_residuals) <= TOLERANCE)
    return {
        "feasible": base_ok and all(v["verdict"] == "pass" for v in worst.values()),
        "objective_kg": float(weights @ x),
        "design": {
            n: {"value": float(v * s), "unit": u}
            for n, v, s, u in zip(DESIGN_NAMES, x, DESIGN_SCALES, DESIGN_UNITS)
        },
        "normalized_design": x.tolist(),
        "base_constraints": {
            "verified": base_ok,
            "maximum_residual": float(max(base_residuals)),
            "tolerance": TOLERANCE,
        },
        "requirements": worst,
        "engineering_results": results if include_bundles else {},
    }


class NativeReference:
    """Independent scenario-cut LP and trust-region separation for this fixture."""

    def __init__(self):
        self.B = ball_map()
        self.prepared = []
        self.points = []
        for response in RESPONSES:
            H, _, _, _ = response.arrays()
            A = self.B.T @ H @ self.B
            self.prepared.append(PreparedTrustRegion(0.5 * (A + A.T)))
            self.points.append([np.zeros(2)])

    def worst(self, x):
        values = []
        for response, prepared in zip(RESPONSES, self.prepared):
            _, S, s, r = response.arrays()
            bound = prepared.maximize(self.B.T @ (S @ x + s), tolerance=INNER_TOLERANCE)
            values.append((bound.upper + float(r @ x + response.offset), bound))
        return values

    def cuts(self):
        rows, rhs = [FOOTPRINT], [FOOTPRINT_LIMIT]
        for response, points in zip(RESPONSES, self.points):
            H, S, s, r = response.arrays()
            for z in points:
                rows.append(z @ S + r)
                rhs.append(-response.offset - s @ z - 0.5 * z @ H @ z - RESERVE)
        return np.asarray(rows), np.asarray(rhs)

    @staticmethod
    def lower_bound(weights, A, rhs, marginals):
        # Any nonnegative multiplier gives a lower bound via the box infimum.
        # Relax the reserve to the actual audit tolerance before bounding.
        relaxed = rhs.copy() + TOLERANCE
        relaxed[1:] += RESERVE
        multipliers = np.maximum(0.0, -np.asarray(marginals))
        reduced = weights + A.T @ multipliers
        terms = np.minimum(reduced * (LOWER - TOLERANCE), reduced * (UPPER + TOLERANCE))
        lower = float(sum(terms) - multipliers @ relaxed)
        guard = (
            256
            * np.finfo(float).eps
            * (1 + np.sum(np.abs(terms)) + np.abs(multipliers) @ np.abs(relaxed))
            * len(rhs)
        )
        if not np.isfinite(lower) or not np.isfinite(guard):
            raise RuntimeError("nonfinite LP dual lower bound")
        return {
            "lower_bound_kg": lower - float(guard),
            "method": "nonnegative_lp_multipliers_and_box_infimum",
            "material_weights": weights.tolist(),
            "scenario_matrix": A.tolist(),
            "relaxed_rhs": relaxed.tolist(),
            "multipliers": multipliers.tolist(),
            "box_lower": (LOWER - TOLERANCE).tolist(),
            "box_upper": (UPPER + TOLERANCE).tolist(),
            "rounding_guard_kg": float(guard),
        }

    def member(self, point):
        # Cuts must be actual declared-domain members, including after mapping
        # through physical floating-point coordinates at an ellipsoid boundary.
        declared = domain()
        physical = declared.space.denormalize(
            dict(zip(declared.space.names, self.B @ point))
        )
        physical = declared._interior_candidate(physical, declared.feasible_point)
        if declared.membership(physical).status != "inside":
            raise RuntimeError("Scenario cut has no verified domain member")
        normalized = declared.space.normalize(physical)
        return np.array([normalized[n] for n in declared.space.names])

    def solve(self, weights, *, max_rounds=100):
        weights = material_weights(weights)
        if type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        lower, x = None, None
        for iteration in range(1, max_rounds + 1):
            A, rhs = self.cuts()
            solved = linprog(
                weights,
                A_ub=A,
                b_ub=rhs,
                bounds=list(zip(LOWER, UPPER)),
                method="highs",
                options={
                    "primal_feasibility_tolerance": INNER_TOLERANCE,
                    "dual_feasibility_tolerance": INNER_TOLERANCE,
                },
            )
            if not solved.success:
                raise RuntimeError(f"Independent LP unresolved: {solved.message}")
            x = vector(solved.x, "LP design")
            proof = self.lower_bound(weights, A, rhs, solved.ineqlin.marginals)
            lower = proof["lower_bound_kg"]
            proof["scenario_points"] = {
                r.name: [p.tolist() for p in points]
                for r, points in zip(RESPONSES, self.points)
            }
            worst = self.worst(x)
            if max(value for value, _ in worst) <= -RESERVE + SOLVE_TOLERANCE:
                return {
                    "design": x.tolist(),
                    "lower_bound_kg": lower,
                    "status": "bounded",
                    "rounds": iteration,
                    "scenario_cuts": sum(map(len, self.points)),
                    "bound_evidence": proof,
                }
            for points, (value, bound) in zip(self.points, worst):
                if value > -RESERVE + SOLVE_TOLERANCE:
                    points.append(self.member(np.asarray(bound.point)))
        return {
            "design": x.tolist(),
            "lower_bound_kg": lower,
            "status": "iteration_limit",
            "rounds": max_rounds,
            "scenario_cuts": sum(map(len, self.points)),
            "bound_evidence": proof,
        }


class PreparedDesign:
    """Example-owned direct SDP, compiled residopt, or mixed compile/oracle solve."""

    def __init__(self, backend, *, solver="CLARABEL"):
        if backend not in ("direct_sdp", "residopt_sdp", "residopt_hybrid"):
            raise ValueError("unknown design backend")
        if backend == "direct_sdp":
            try:
                self.cp, self.residopt = import_module("cvxpy"), None
            except ImportError as exc:
                raise BackendUnavailable("Direct SDP needs optional CVXPY") from exc
        else:
            self.residopt, self.cp = load_backend()
        self.backend, self.solver = backend, solver
        if solver not in self.cp.installed_solvers():
            raise BackendUnavailable(f"Solver {solver} is not installed")
        cp = self.cp
        self.x = cp.Variable(3, name="equipment_sizes")
        self.weights = cp.Parameter(3, nonneg=True, name="material_kg_per_size")
        base = [self.x >= LOWER, self.x <= UPPER, FOOTPRINT @ self.x <= FOOTPRINT_LIMIT]
        C = np.linalg.solve(ball_map(), np.eye(2))
        if backend == "direct_sdp":
            for response in RESPONSES:
                H, S, s, r = response.arrays()
                multiplier = cp.Variable(nonneg=True)
                cross = cp.reshape(-0.5 * (S @ self.x + s), (2, 1), order="C")
                bound = cp.reshape(
                    -r @ self.x - response.offset - RESERVE - multiplier,
                    (1, 1),
                    order="C",
                )
                base.append(
                    cp.bmat(
                        [[multiplier * (C.T @ C) - 0.5 * H, cross], [cross.T, bound]]
                    )
                    >> 0
                )
            self.problem = cp.Problem(cp.Minimize(self.weights @ self.x), base)
            self.compiled = None
        else:
            atoms = []
            for response in RESPONSES:
                H, S, s, r = response.arrays()
                atoms.append(
                    self.residopt.QuadraticEllipsoidAtom(
                        response.name,
                        C,
                        1.0,
                        H,
                        S,
                        s,
                        r,
                        response.offset,
                        np.zeros(3),
                        -RESERVE,
                    )
                )

            def strategy(description, context):
                oracle = (
                    backend == "residopt_hybrid"
                    and description.certificate.atom_id != RESPONSES[0].name
                )
                return self.residopt.StrategyDecision(
                    self.residopt.Strategy.ORACLE
                    if oracle
                    else self.residopt.Strategy.COMPILE,
                    "Explicit mixed strategy experiment"
                    if oracle
                    else "Explicit exact SDP experiment",
                    description.cost.cone_score,
                    None,
                )

            self.compiled = self.residopt.ResidualCompiler(
                acceptable_labels=(self.residopt.CertificateLabel.EXACT,),
                strategy_selector=strategy,
            ).compile_problem(
                objective=self.weights @ self.x,
                x=self.x,
                atoms=atoms,
                base_constraints=base,
            )
            self.problem = self.compiled.problem

    def solve(self, weights, *, max_rounds=100):
        self.weights.value = material_weights(weights)
        if self.compiled is None:
            try:
                self.problem.solve(solver=self.solver)
            except self.cp.error.SolverError as exc:
                raise RuntimeError(f"Direct SDP solver failed: {exc}") from exc
            if self.problem.status != self.cp.OPTIMAL:
                raise RuntimeError(f"Direct SDP unresolved: {self.problem.status}")
            report = {"master_status": self.problem.status}
        else:
            self.compiled.solve(
                solver=self.solver,
                robust_tolerance=SOLVE_TOLERANCE,
                oracle_tolerance=INNER_TOLERANCE,
                max_rounds=max_rounds,
            )
            solved = self.compiled.solve_report
            expected = {r.name for r in RESPONSES}
            if (
                solved is None
                or not solved.robust_feasible
                or len(solved.checks) != len(expected)
                or {c.atom_id for c in solved.checks} != expected
            ):
                raise RuntimeError("Original coupled constraints were not all verified")
            if self.backend == "residopt_sdp" and not self.compiled.exact_certificates:
                raise RuntimeError("Exact SDP formulation was not preserved")
            report = {
                "solve_report": asdict(solved),
                "certificates": [asdict(c) for c in self.compiled.certificates],
                "fully_compiled": self.compiled.is_fully_compiled,
            }
        return {
            "design": vector(self.x.value, "solver design").tolist(),
            "backend_evidence": json.loads(json.dumps(report, allow_nan=False)),
        }


def assessment(solution, weights, lower_bound, *, include_bundles=True):
    result = audit_design(solution["design"], weights, include_bundles=include_bundles)
    gap = result["objective_kg"] - lower_bound
    result.update(
        {
            "lower_bound_kg": lower_bound,
            "objective_gap_kg": gap,
            "status": "verified_near_optimal"
            if result["feasible"] and 0 <= gap <= OBJECTIVE_GAP_KG
            else "unresolved",
            "optimization_evidence": solution,
        }
    )
    return result


@contextmanager
def measured(timings, phase):
    started = perf_counter()
    try:
        yield
    finally:
        timings[phase] += perf_counter() - started


def nominal_design():
    reference = NativeReference()
    A, rhs = reference.cuts()  # initial cuts are nominal realizations only
    solved = linprog(
        WORKLOAD[0], A_ub=A, b_ub=rhs, bounds=list(zip(LOWER, UPPER)), method="highs"
    )
    if not solved.success:
        raise RuntimeError("Nominal design LP failed")
    return audit_design(solved.x, WORKLOAD[0])


def run_experiment(*, repeats=3, solver="CLARABEL"):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be positive")
    references = [NativeReference().solve(w) for w in WORKLOAD]
    trials, examples = [], {}
    backends = ("native_exchange", "direct_sdp", "residopt_sdp", "residopt_hybrid")
    for repeat in range(repeats):
        for backend in backends if repeat % 2 == 0 else reversed(backends):
            for reuse in (False, True):
                prepared = None
                timings = dict.fromkeys(
                    ("build_seconds", "solve_seconds", "audit_seconds"), 0.0
                )
                cases = []
                for index, (weights, reference) in enumerate(zip(WORKLOAD, references)):
                    try:
                        if prepared is None or not reuse:
                            with measured(timings, "build_seconds"):
                                prepared = (
                                    NativeReference()
                                    if backend == "native_exchange"
                                    else PreparedDesign(backend, solver=solver)
                                )
                        with measured(timings, "solve_seconds"):
                            solution = prepared.solve(weights)
                        with measured(timings, "audit_seconds"):
                            checked = assessment(
                                solution, weights, reference["lower_bound_kg"]
                            )
                        if repeat == 0 and not reuse:
                            examples[f"{backend}_{index}"] = checked
                        cases.append(
                            {
                                "status": checked["status"],
                                "objective_kg": checked["objective_kg"],
                                "gap_kg": checked["objective_gap_kg"],
                            }
                        )
                    except (ValueError, RuntimeError) as exc:
                        failure = {"status": "unresolved", "error": str(exc)}
                        if hasattr(exc, "report"):
                            failure["solve_report"] = json.loads(
                                json.dumps(asdict(exc.report))
                            )
                        cases.append(failure)
                        prepared = None
                trials.append(
                    {
                        "backend": backend,
                        "reuse": reuse,
                        "repeat": repeat,
                        **timings,
                        "total_seconds": sum(timings.values()),
                        "cases": cases,
                    }
                )
    summary = []
    for backend in backends:
        for reuse in (False, True):
            rows = [
                r for r in trials if r["backend"] == backend and r["reuse"] == reuse
            ]
            summary.append(
                {
                    "backend": backend,
                    "reuse": reuse,
                    **{
                        f"median_{key}": median(r[key] for r in rows)
                        for key in (
                            "build_seconds",
                            "solve_seconds",
                            "audit_seconds",
                            "total_seconds",
                        )
                    },
                    "verified": sum(
                        c["status"] == "verified_near_optimal"
                        for r in rows
                        for c in r["cases"]
                    ),
                    "calls": sum(len(r["cases"]) for r in rows),
                }
            )
    return {
        "references": references,
        "examples": examples,
        "trials": trials,
        "summary": summary,
        "nominal_design": nominal_design(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--solver", default="CLARABEL")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    residopt, cp = load_backend()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        results = run_experiment(repeats=args.repeats, solver=args.solver)
    counts = Counter(
        (w.category.__name__, str(w.message), w.filename, w.lineno) for w in observed
    )
    report = {
        "experiment": "coupled_process_design/v1",
        "source_revision": args.source_revision,
        "backend_identity": backend_identity(residopt, cp),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "solver": args.solver,
        },
        "model": {
            "responses": [asdict(r) for r in RESPONSES],
            "domain": domain().to_manifest(),
            "lower": LOWER.tolist(),
            "upper": UPPER.tolist(),
            "footprint": FOOTPRINT.tolist(),
            "footprint_limit": FOOTPRINT_LIMIT,
            "reserve": RESERVE,
            "solve_tolerance": SOLVE_TOLERANCE,
            "inner_tolerance": INNER_TOLERANCE,
            "audit_tolerance": TOLERANCE,
            "objective_gap_kg": OBJECTIVE_GAP_KG,
            "material_workload": WORKLOAD,
            "design_names": DESIGN_NAMES,
            "design_units": DESIGN_UNITS,
            "design_scales": DESIGN_SCALES,
        },
        "methodology": {
            "repeats": args.repeats,
            "cases_per_workload": len(WORKLOAD),
            "timed": "construction, optimization, independent physical audits and result bundling",
            "excluded": "imports, common reference lower bounds, nominal comparison and final JSON file writing",
            "reuse": "fixed domains/equations; changed material objective weights; caches and scenario cuts retained",
            "hybrid_policy": "compile temperature; defer pressure and cooling; explicit experiment policy, not learned selection",
        },
        "results": results,
        "observed_warnings": [
            dict(category=c, message=m, file=f, line=line, count=n)
            for (c, m, f, line), n in counts.items()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"Wrote {args.output}; recorded {len(observed)} warnings")
    for row in results["summary"]:
        print(row)


if __name__ == "__main__":
    main()
