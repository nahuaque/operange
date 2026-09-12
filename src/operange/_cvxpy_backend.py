"""Optional prepared convex programs. Returned numbers are candidates, not proof.

The bounded, per-thread cache holds mutable CVXPY workspaces separately from
immutable engineering declarations. Exact rows remain owned by LinearSystem.
"""

from collections import OrderedDict
from dataclasses import dataclass
from importlib import import_module
from threading import local

import numpy as np

from .primitives import finite


class BackendUnavailable(RuntimeError):
    pass


def load_cvxpy():
    try:
        return import_module("cvxpy")
    except ImportError as exc:
        raise BackendUnavailable(
            'This operation requires the optional dependency: pip install "operange[cvxpy]".'
        ) from exc


def validate_backend(backend):
    if backend not in ("scipy", "cvxpy"):
        raise ValueError("backend must be 'scipy' or 'cvxpy'")
    if backend == "cvxpy":
        load_cvxpy()


_workspaces = local()
_CACHE_SIZE = 16


def cached_program(key, build):
    cache = getattr(_workspaces, "cache", None)
    if cache is None:
        _workspaces.cache = cache = OrderedDict()
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    program = build()
    cache[key] = program
    if len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)
    return program


def clear_cache():
    """Release this thread's prepared programs (also useful for cold benchmarks)."""
    _workspaces.cache = OrderedDict()


def numeric_vector(value, size, name):
    if value is None:
        return None
    values = np.asarray(value)
    if values.shape != (size,):
        raise ValueError(f"{name} has an unexpected shape")
    return [finite(v, name) for v in values]


@dataclass
class CandidateSolve:
    point: list | None
    multipliers: list | None
    attempts: list


class PreparedLinearProgram:
    """A fixed row matrix with parameterized RHS and linear objective.

    Phase I always has a finite mathematical optimum on the bounded control
    box, so row duals can propose an infeasibility certificate without trusting
    an infeasible solver termination or requesting a solver-specific ray.
    """

    def __init__(self, rows, *, phase_one):
        cp = self.cp = load_cvxpy()
        self.rows = np.asarray(rows, dtype=float)
        self.m, self.n = self.rows.shape
        self.phase_one = phase_one
        self.z = cp.Variable(self.n, name="normalized_controls")
        self.rhs = cp.Parameter(self.m, name="constraint_upper")
        self.cost = cp.Parameter(self.n, name="linear_objective")
        slack = cp.Variable(name="phase_one_violation") if phase_one else 0
        self.constraint = self.rows @ self.z <= self.rhs + slack
        self.problem = cp.Problem(
            cp.Minimize(slack if phase_one else self.cost @ self.z),
            [self.constraint, self.z >= 0, self.z <= 1],
        )
        if not self.problem.is_dcp(dpp=True):
            raise ValueError("prepared linear program must be DPP-compliant")

    def solve(self, upper, objective, tolerance):
        cp = self.cp
        self.rhs.value = upper
        self.cost.value = objective
        self.z.value = None
        report = {
            "purpose": "dispatch_candidate"
            if self.phase_one
            else "minimum_relief_candidate",
            "solver": f"CVXPY {cp.__version__} / SCIPY-HiGHS",
            "status": "unresolved",
            "tolerance": tolerance,
            "message": "No numerical candidate returned.",
            "compilation": "DPP; fixed rows with parameterized RHS and objective",
        }
        try:
            self.problem.solve(
                solver="SCIPY",
                enforce_dpp=True,
                warm_start=True,
                scipy_options={
                    "method": "highs",
                    "primal_feasibility_tolerance": tolerance,
                    "dual_feasibility_tolerance": tolerance,
                },
            )
            status = self.problem.status
            report.update(
                message=str(status),
                status="optimal" if status == cp.OPTIMAL else "unresolved",
            )
            if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return CandidateSolve(None, None, [report])
            point = numeric_vector(self.z.value, self.n, "control candidate")
            dual = numeric_vector(
                self.constraint.dual_value, self.m, "row dual candidate"
            )
            return CandidateSolve(point, dual, [report])
        except (cp.error.SolverError, ValueError, OverflowError) as exc:
            report.update(status="unresolved", message=str(exc))
            return CandidateSolve(None, None, [report])


def solve_bounded(system, objective, tolerance, *, phase_one=False):
    rows = tuple(
        tuple(finite(float(a), "LP coefficient") for a in row) for row in system.rows
    )
    upper = [finite(float(b), "LP right-hand side") for b in system.upper]
    objective = [finite(v, "LP objective") for v in objective]
    if not rows or not system.controls or len(objective) != len(system.controls):
        raise ValueError(
            "prepared bounded LP needs rows, controls and an aligned objective"
        )
    program = cached_program(
        ("linear", rows, phase_one),
        lambda: PreparedLinearProgram(rows, phase_one=phase_one),
    )
    result = program.solve(upper, objective, tolerance)
    for attempt in result.attempts:
        attempt["row_constraint_refs"] = list(system.row_refs)
    return result
