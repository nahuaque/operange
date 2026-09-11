"""Optional SDP compilation against residopt's verified-solve API."""

from dataclasses import asdict
import hashlib
import json
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.linalg import eigvalsh

from ._quadratic_bounds import array


class BackendUnavailable(RuntimeError):
    pass


def load_backend():
    if sys.version_info < (3, 13):
        raise BackendUnavailable(
            "The current residopt experiment requires Python 3.13+."
        )
    try:
        residopt = import_module("residopt")
        cp = import_module("cvxpy")
    except ImportError as exc:
        raise BackendUnavailable(
            "This experiment needs optional residopt and CVXPY; see the sibling-checkout run instructions."
        ) from exc
    if not hasattr(residopt, "RobustSolveError") or not hasattr(
        getattr(residopt, "CompiledModel", None), "solve_master"
    ):
        raise BackendUnavailable(
            "This experiment requires the current residopt verified-solve API; the older 0.1.0 API is unsupported."
        )
    return residopt, cp


def backend_identity(residopt, cp):
    root = Path(residopt.__file__).resolve().parent
    source = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    try:
        installed_version = version("residopt")
    except PackageNotFoundError:
        installed_version = "uninstalled checkout"
    return {
        "module_path": str(root),
        "source_sha256": source.hexdigest(),
        "distribution_version": installed_version,
        "cvxpy_version": cp.__version__,
    }


class PreparedResidualSDP:
    """Compile fixed H once; each solve binds a new linear coefficient vector.

    The compiler is deliberately forced to emit the exact SDP in this experiment.
    Strategy selection and hybrid optimization are separate future experiments.
    """

    def __init__(self, hessian, *, solver="CLARABEL"):
        started = perf_counter()
        self.residopt, self.cp = load_backend()
        self.H = array(hessian, 2, "hessian")
        d = len(self.H)
        self.solver = solver
        if solver not in self.cp.installed_solvers():
            raise BackendUnavailable(
                f"Requested SDP solver {solver!r} is not installed."
            )
        self.identity = backend_identity(self.residopt, self.cp)
        self.x = self.cp.Variable(d + 1, name="quadratic_bound_and_linear")
        self.linear = self.cp.Parameter(d, name="quadratic_linear")
        atom = self.residopt.QuadraticEllipsoidAtom(
            atom_id="process_quadratic",
            C=np.eye(d),
            rho=1.0,
            H=self.H,
            S=np.column_stack((np.zeros(d), np.eye(d))),
            s0=np.zeros(d),
            r=np.zeros(d + 1),
            r0=0,
            g=np.r_[1.0, np.zeros(d)],
            h=0,
        )

        def compile_exact(description, context):
            return self.residopt.StrategyDecision(
                self.residopt.Strategy.COMPILE,
                "Explicit exact-SDP experiment",
                description.cost.cone_score,
                None,
            )

        self.compiled = self.residopt.ResidualCompiler(
            acceptable_labels=(self.residopt.CertificateLabel.EXACT,),
            strategy_selector=compile_exact,
        ).compile_problem(
            objective=self.x[0],
            x=self.x,
            atoms=(atom,),
            base_constraints=(self.x[1:] == self.linear,),
        )
        if not self.compiled.is_fully_compiled or not self.compiled.exact_certificates:
            raise RuntimeError(
                "The quadratic experiment requires an exact, fully compiled original model."
            )
        self.build_seconds = perf_counter() - started

    def upper_bound(self, linear, *, tolerance=1e-7, solver_options=None):
        b = array(linear, 1, "linear")
        self.linear.value = b
        started = perf_counter()
        try:
            self.compiled.solve(
                solver=self.solver,
                robust_tolerance=tolerance,
                oracle_tolerance=tolerance * 0.1,
                **(solver_options or {}),
            )
        except self.cp.error.SolverError as exc:
            raise RuntimeError(f"SDP solver failed: {exc}") from exc
        elapsed = perf_counter() - started
        report = self.compiled.solve_report
        if (
            report is None
            or not report.robust_feasible
            or len(report.checks) != 1
            or report.checks[0].atom_id != "process_quadratic"
        ):
            raise RuntimeError("residopt did not verify the original quadratic atom")
        if not self.compiled.is_fully_compiled or not self.compiled.exact_certificates:
            raise RuntimeError(
                "Unresolved oracle atoms or inexact reformulations cannot certify this experiment"
            )
        decision = array(self.x.value, 1, "SDP decision")
        multiplier = float(self.compiled.variables["process_quadratic_lambda"].value)
        if not np.isfinite(multiplier):
            raise ValueError("nonfinite SDP multiplier")
        multiplier = max(0.0, multiplier)
        # Rebuild the LMI for the requested b, not the solver's approximate copy.
        matrix = np.block(
            [
                [multiplier * np.eye(len(b)) - 0.5 * self.H, -0.5 * b[:, None]],
                [-0.5 * b[None, :], np.array([[decision[0] - multiplier]])],
            ]
        )
        minimum = float(eigvalsh(matrix)[0])
        rounding = (
            128
            * np.finfo(float).eps
            * max(1.0, float(np.linalg.norm(matrix, 2)))
            * len(matrix)
        )
        correction = 2 * max(0.0, rounding - minimum)
        upper = float(decision[0] + correction + rounding)
        if not np.isfinite(upper):
            raise ValueError("nonfinite SDP upper bound")
        details = {
            "method": "s_lemma_sdp_with_lmi_eigenvalue_correction",
            "backend": self.identity,
            "solver": self.solver,
            "report": asdict(report),
            "certificates": [asdict(c) for c in self.compiled.certificates],
            "objective": float(decision[0]),
            "multiplier": multiplier,
            "minimum_lmi_eigenvalue": minimum,
            "lmi_correction": correction,
            "rounding_guard": rounding,
            "parameter_residual": float(np.linalg.norm(decision[1:] - b)),
            "build_seconds": self.build_seconds,
            "verified_solve_seconds": elapsed,
        }
        return upper, json.loads(json.dumps(details, allow_nan=False))
