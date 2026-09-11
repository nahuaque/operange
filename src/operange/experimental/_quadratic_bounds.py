"""Independent quadratic-ball checks using a scalar KKT root and a dual bound.

This module does not import residopt or CVXPY. Bounds are numerical, not
interval-arithmetic proofs. The completion-of-squares bound also covers a
nonzero stationarity residual and a small negative eigenvalue estimate.
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq


def array(value, ndim, name):
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BallBounds:
    lower: float
    upper: float
    point: tuple[float, ...]
    details: dict

    def __post_init__(self):
        if (
            not np.isfinite(self.lower)
            or not np.isfinite(self.upper)
            or self.lower > self.upper
        ):
            raise ValueError("invalid quadratic bounds")
        if (
            not np.all(np.isfinite(self.point))
            or np.linalg.norm(self.point) > 1 + 1e-12
        ):
            raise ValueError("invalid quadratic candidate")


class PreparedTrustRegion:
    """Maximize .5*y.T@H@y + b.T@y on ||y|| <= 1, caching the spectrum."""

    def __init__(self, hessian):
        self.H = array(hessian, 2, "hessian")
        if (
            not self.H.shape[0]
            or self.H.shape[0] != self.H.shape[1]
            or not np.array_equal(self.H, self.H.T)
        ):
            raise ValueError("hessian must be nonempty, square and symmetric")
        self.eigenvalues, self.basis = eigh(self.H)
        self.scale = max(1.0, float(np.linalg.norm(self.H, 2)))
        self.spectral_error = float(
            np.linalg.norm(self.H @ self.basis - self.basis * self.eigenvalues, 2)
        )
        self.orthogonality_error = float(
            np.linalg.norm(self.basis.T @ self.basis - np.eye(len(self.H)), 2)
        )
        if self.spectral_error > 1e-10 * self.scale or self.orthogonality_error > 1e-10:
            raise ValueError("spectral decomposition could not be verified")

    def maximize(self, linear, *, tolerance=1e-7, max_iterations=200):
        b = array(linear, 1, "linear")
        if b.shape != (len(self.H),) or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("invalid quadratic solve arguments")
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        beta = self.basis.T @ b
        edge = max(0.0, float(self.eigenvalues[-1]))
        rounding = (
            128
            * np.finfo(float).eps
            * max(self.scale, float(np.linalg.norm(b)), 1)
            * len(b)
        )
        gaps = edge - self.eigenvalues
        regular = gaps > rounding
        spectral = np.zeros_like(beta)
        spectral[regular] = beta[regular] / gaps[regular]
        multiplier, iterations = edge, 0
        mode = "interior_or_singular_boundary"

        # The singular hard case needs a component in the top eigenspace.
        # Its upper bound still accounts for any omitted small linear component.
        if np.linalg.norm(spectral) <= 1 and np.linalg.norm(beta[~regular]) <= rounding:
            if edge > 0:
                null = int(np.flatnonzero(~regular)[0])
                spectral[null] = sqrt(max(0.0, 1 - float(spectral @ spectral)))
        else:
            low = edge + max(rounding, np.spacing(edge))
            high = edge + max(1.0, float(np.linalg.norm(b))) + rounding

            def norm_residual(value):
                return float(np.linalg.norm(beta / (value - self.eigenvalues))) - 1

            if norm_residual(low) > 0:
                multiplier, result = brentq(
                    norm_residual,
                    low,
                    high,
                    xtol=1e-14 * self.scale,
                    rtol=4 * np.finfo(float).eps,
                    maxiter=max_iterations,
                    full_output=True,
                    disp=False,
                )
                iterations = result.iterations
                mode = "scalar_kkt_root" if result.converged else "iteration_limit"
            else:
                multiplier, mode = low, "near_singular_boundary"
            spectral = beta / (multiplier - self.eigenvalues)

        v = self.basis @ spectral
        candidate = v / max(1.0, float(np.linalg.norm(v)))
        # A second shrink handles norm rounding without broadening the unit ball.
        if np.linalg.norm(candidate) > 1:
            candidate *= 1 - 8 * np.finfo(float).eps
        lower = float(0.5 * candidate @ self.H @ candidate + b @ candidate)
        M = multiplier * np.eye(len(b)) - self.H
        residual = b - M @ v
        minimum_eigenvalue = multiplier - float(self.eigenvalues[-1])
        eigen_guard = (
            self.spectral_error + self.orthogonality_error * self.scale + rounding
        )
        curvature_correction = (
            0.5
            * max(0.0, eigen_guard - minimum_eigenvalue)
            * (1 + float(np.linalg.norm(v))) ** 2
        )
        upper = float(
            0.5 * multiplier
            + 0.5 * v @ M @ v
            + np.linalg.norm(residual)
            + curvature_correction
            + rounding
        )
        if lower > upper:
            raise ValueError("quadratic dual bound contradicts its candidate")
        return BallBounds(
            lower,
            upper,
            tuple(float(x) for x in candidate),
            {
                "method": "spectral_kkt_with_completed_square_bound",
                "mode": mode,
                "multiplier": multiplier,
                "stationarity_residual": float(np.linalg.norm(residual)),
                "minimum_dual_eigenvalue": minimum_eigenvalue,
                "curvature_correction": curvature_correction,
                "rounding_guard": rounding,
                "gap": upper - lower,
                "tolerance": tolerance,
                "resolution": "closed" if upper - lower <= tolerance else "bounded",
                "iterations": iterations,
            },
        )
