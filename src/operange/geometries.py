"""Continuous uncertainty geometries and analytical linear support functions."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum, hypot
from typing import Literal

from scipy.linalg import cholesky, eigvalsh, solve_triangular

from ._geometry import (
    Geometry,
    check_tolerance,
    dot,
    membership_record,
    normalized_bounds,
    require_normalization,
    weights,
)
from .contract_types import Record, freeze, nonempty
from .domains import ParameterSpace
from .primitives import BoxSet, finite


@dataclass(frozen=True)
class SimplexSet(Record, Geometry):
    """Nonnegative physical mass or mole fractions summing to one."""

    space: ParameterSpace
    basis: Literal["mass", "mole"]
    provenance: str = "Caller-declared composition domain"
    tolerance: float = 1e-10

    def _validate(self):
        nonempty(self.provenance, "provenance")
        check_tolerance(self.tolerance)
        require_normalization(self.space)
        if any(c.unit != f"{self.basis}_fraction" for c in self.space.coordinates):
            raise ValueError(
                "simplex coordinates must declare the matching mass_fraction or mole_fraction unit"
            )

    @property
    def feasible_point(self):
        return {name: 1.0 if i == 0 else 0.0 for i, name in enumerate(self.space.names)}

    def to_manifest(self):
        return {"kind": "simplex", **self.to_dict()}

    def membership(self, realization):
        values = self.space.validate(realization)
        try:
            balance = abs(fsum(values.values()) - 1)
        except OverflowError:
            balance = float("inf")
        residuals = [*(-v for v in values.values()), balance]
        return membership_record(
            self, values, residuals, "simplex_balance", details={"basis": self.basis}
        )

    def maximize_linear(self, coefficients):
        w = weights(self.space, coefficients)
        try:
            scores = {
                c.name: finite(w[c.name] / c.scale, "simplex coefficient")
                for c in self.space.coordinates
            }
            best = max(self.space.names, key=lambda n: scores[n])
            point = {n: float(n == best) for n in self.space.names}
            upper = dot(w.values(), self.space.normalize(point).values())
            return self._support(
                w,
                point,
                upper,
                "simplex_vertex_bound",
                details={"maximizing_vertex": best},
            )
        except (ValueError, OverflowError) as exc:
            return self._unresolved(w, str(exc))


@dataclass(frozen=True)
class BudgetSet(Record, Geometry):
    """Within an envelope, |z_i| <= 1 and sum |z_i| <= budget.

    z uses the envelope's named nominal values and scales. Neither a
    probability model nor a count of outage events is implied.
    """

    envelope: BoxSet
    budget: float
    provenance: str = "Caller-declared simultaneous-deviation budget"
    tolerance: float = 1e-10

    def _validate(self):
        nonempty(self.provenance, "provenance")
        check_tolerance(self.tolerance)
        if self.budget < 0:
            raise ValueError("budget must be nonnegative")

    @property
    def space(self):
        return self.envelope.space

    @property
    def feasible_point(self):
        return self.envelope.nominal

    def to_manifest(self):
        return {
            "kind": "budget",
            "envelope": self.envelope.to_manifest(),
            "budget": self.budget,
            "provenance": self.provenance,
            "tolerance": self.tolerance,
        }

    def membership(self, realization):
        values = self.space.validate(realization)
        if not self.envelope.contains(values):
            check = self.envelope.membership(values)
            from dataclasses import replace

            return replace(check, domain_ref=self.ref)
        z = self.space.normalize(values)
        try:
            total = fsum(abs(v) for v in z.values()) - self.budget
        except OverflowError:
            total = float("inf")
        residuals = [*(abs(v) - 1 for v in z.values()), total]
        return membership_record(self, values, residuals, "normalized_deviation_budget")

    def maximize_linear(self, coefficients):
        w = weights(self.space, coefficients)
        try:
            bounds = dict(zip(self.space.names, normalized_bounds(self.envelope)))
            z = dict.fromkeys(self.space.names, 0.0)
            remaining = self.budget
            for name in sorted(self.space.names, key=lambda n: (-abs(w[n]), n)):
                if w[name] == 0:
                    continue
                low, high = bounds[name]
                amount = min(remaining, 1.0, high if w[name] > 0 else -low)
                z[name] = amount if w[name] > 0 else -amount
                remaining = max(0.0, remaining - amount)
            upper = dot(w.values(), z.values())
            point = self._interior_candidate(
                self.space.denormalize(z), self.feasible_point
            )
            return self._support(
                w,
                point,
                upper,
                "fractional_budget_allocation",
                details={
                    "allocation_order": sorted(
                        self.space.names, key=lambda n: (-abs(w[n]), n)
                    )
                },
            )
        except (ValueError, OverflowError) as exc:
            return self._unresolved(w, str(exc))


@dataclass(frozen=True)
class EllipsoidSet(Record, Geometry):
    """z.T @ inv(shape) @ z <= radius**2 in normalized coordinates.

    The shape matrix is explicitly name-aligned, dimensionless and positive
    definite. Singular or numerically ill-conditioned shapes are rejected.
    """

    space: ParameterSpace
    shape: Mapping[str, Mapping[str, float]]
    radius: float = 1.0
    provenance: str = "Caller-declared ellipsoidal geometry"
    tolerance: float = 1e-10

    def _validate(self):
        nonempty(self.provenance, "provenance")
        check_tolerance(self.tolerance)
        require_normalization(self.space)
        if self.radius < 0:
            raise ValueError("radius must be nonnegative")
        finite(self.radius * self.radius, "squared radius")
        if set(self.shape) != set(self.space.names) or any(
            not isinstance(row, Mapping) or set(row) != set(self.space.names)
            for row in self.shape.values()
        ):
            raise ValueError("shape rows and columns must match all coordinate names")
        shape = {
            a: {
                b: finite(self.shape[a][b], "shape coefficient")
                for b in self.space.names
            }
            for a in self.space.names
        }
        if any(shape[a][b] != shape[b][a] for a in shape for b in shape):
            raise ValueError("ellipsoid shape must be symmetric")
        object.__setattr__(self, "shape", freeze(shape))
        eigenvalues = eigvalsh(self._matrix())
        if eigenvalues[0] <= 0 or eigenvalues[0] / eigenvalues[-1] < 1e-12:
            raise ValueError(
                "ellipsoid shape must be positive definite with condition number <= 1e12; singular shapes are unsupported"
            )
        cholesky(self._matrix(), lower=True)

    def _matrix(self):
        return [[self.shape[a][b] for b in self.space.names] for a in self.space.names]

    @property
    def feasible_point(self):
        return self.space.nominal

    def to_manifest(self):
        return {"kind": "ellipsoid", **self.to_dict()}

    def membership(self, realization):
        values = self.space.validate(realization)
        try:
            z = self.space.normalize(values)
            y = solve_triangular(
                cholesky(self._matrix(), lower=True), list(z.values()), lower=True
            )
            residual = finite(hypot(*y) - self.radius, "ellipsoid residual")
        except (ValueError, OverflowError):
            residual = float("inf")
        return membership_record(
            self,
            values,
            (residual,),
            "ellipsoid_cholesky",
            details={"radius": self.radius, "shape_coordinates": self.space.names},
        )

    def maximize_linear(self, coefficients):
        w = weights(self.space, coefficients)
        try:
            factor = cholesky(self._matrix(), lower=True)
            transformed = [dot(column, w.values()) for column in factor.T]
            norm = finite(hypot(*transformed), "ellipsoid support norm")
            if norm == 0 and any(w.values()):
                raise ValueError("ellipsoid support norm underflowed")
            upper = finite(self.radius * norm, "ellipsoid support")
            direction = [v / norm if norm else 0.0 for v in transformed]
            z = {
                n: self.radius * dot(row, direction)
                for n, row in zip(self.space.names, factor)
            }
            point = self._interior_candidate(
                self.space.denormalize(z), self.feasible_point
            )
            return self._support(
                w,
                point,
                upper,
                "ellipsoid_cauchy_schwarz",
                details={"formula": "radius * sqrt(weights.T @ shape @ weights)"},
            )
        except (ValueError, OverflowError) as exc:
            return self._unresolved(w, str(exc))
