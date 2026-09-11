"""Bounded linear domains with checked primal/dual support bounds."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

from . import linear
from ._geometry import (
    Geometry,
    check_tolerance,
    dot,
    membership_record,
    normalized_bounds,
    weights,
)
from .contract_types import Evidence, Measurement, Record, freeze, nonempty, unique
from .primitives import BoxSet, finite
from ._numeric import exact_dot, exact_normalized_bounds, round_up


@dataclass(frozen=True)
class LinearConstraint(Record):
    """A named restriction on normalized coordinates; omitted coefficients are zero."""

    name: str
    coefficients: Mapping[str, float]
    rhs: float
    relation: Literal["le", "eq"] = "le"

    def _validate(self):
        nonempty(self.name, "constraint name")
        if not self.coefficients:
            raise ValueError("linear constraint needs named coefficients")
        object.__setattr__(
            self,
            "coefficients",
            freeze({n: finite(v, n) for n, v in self.coefficients.items()}),
        )


@dataclass(frozen=True)
class PolytopeSet(Record, Geometry):
    """A finite box intersected with coupled normalized linear restrictions.

    A verified feasible point is mandatory; the nominal is tried by default.
    Construction never treats an empty or unresolved domain as a vacuous pass.
    """

    envelope: BoxSet
    constraints: tuple[LinearConstraint, ...]
    feasible_point: Mapping[str, float] | None = None
    provenance: str = "Caller-declared coupled linear envelope"
    tolerance: float = 1e-10

    def _validate(self):
        nonempty(self.provenance, "provenance")
        check_tolerance(self.tolerance)
        unique(tuple(c.name for c in self.constraints), "linear constraint names")
        for constraint in self.constraints:
            weights(self.space, constraint.coefficients)
        point = (
            self.envelope.nominal
            if self.feasible_point is None
            else self.space.validate(self.feasible_point)
        )
        object.__setattr__(self, "feasible_point", freeze(point))
        if self.membership(point).status != "inside":
            raise ValueError(
                "polytope needs a verified feasible_point; the nominal does not establish nonemptiness"
            )

    @property
    def space(self):
        return self.envelope.space

    def to_manifest(self):
        return {
            "kind": "polytope",
            "envelope": self.envelope.to_manifest(),
            "constraints": [c.to_dict() for c in self.constraints],
            "feasible_point": dict(self.feasible_point),
            "provenance": self.provenance,
            "tolerance": self.tolerance,
        }

    def membership(self, realization):
        values = self.space.validate(realization)
        if not self.envelope.contains(values):
            return replace(self.envelope.membership(values), domain_ref=self.ref)
        z = self.space.normalize(values)
        residuals = []
        for constraint in self.constraints:
            try:
                value = (
                    dot(
                        constraint.coefficients.values(),
                        (z[n] for n in constraint.coefficients),
                    )
                    - constraint.rhs
                )
            except (ValueError, OverflowError):
                value = float("inf")
            residuals.append(abs(value) if constraint.relation == "eq" else value)
        return membership_record(
            self,
            values,
            residuals,
            "normalized_linear_constraints",
            details={"constraints": [c.name for c in self.constraints]},
        )

    def _rows(self, bounds=None):
        rows, rhs = [], []
        for constraint in self.constraints:
            row = [constraint.coefficients.get(n, 0.0) for n in self.space.names]
            rows.append(row)
            rhs.append(constraint.rhs)
            if constraint.relation == "eq":
                rows.append([-v for v in row])
                rhs.append(-constraint.rhs)
        for i, (low, high) in enumerate(
            normalized_bounds(self.envelope) if bounds is None else bounds
        ):
            row = [float(j == i) for j in range(len(self.space.names))]
            rows.extend((row, [-v for v in row]))
            rhs.extend((high, -low))
        return rows, rhs

    def maximize_linear(self, coefficients):
        w = weights(self.space, coefficients)
        try:
            return self._maximize(w)
        except (ValueError, OverflowError) as exc:
            return self._unresolved(w, str(exc))

    def _maximize(self, w):
        names, c = self.space.names, list(w.values())
        rows, rhs = self._rows()
        bounds = normalized_bounds(self.envelope)
        exact_bounds = exact_normalized_bounds(self.envelope)
        point = self.feasible_point
        # Even a failed solver leaves the declared envelope as a valid bound.
        upper = sum(
            max(Fraction(v) * lo, Fraction(v) * hi)
            for v, (lo, hi) in zip(c, exact_bounds)
        )
        primal, primal_evidence = linear.solve_lp(
            [-v for v in c], bounds, self.tolerance, inequalities=rows, upper=rhs
        )
        if primal is not None:
            candidate = self.space.denormalize(dict(zip(names, primal)))
            if self.membership(candidate).status == "inside":
                point = candidate
        # min rhs.T @ lambda, rows.T @ lambda = c, lambda >= 0.
        dual, dual_evidence = linear.solve_lp(
            rhs,
            [(0, None)] * len(rows),
            self.tolerance,
            equalities=[list(column) for column in zip(*rows)],
            rhs=c,
        )
        details = {"fallback": "support of the finite envelope"}
        if dual is not None:
            multipliers = [max(0.0, v) for v in dual]
            exact_rows, exact_rhs = self._rows(exact_bounds)
            residual = [
                Fraction(ci) - exact_dot((row[i] for row in exact_rows), multipliers)
                for i, ci in enumerate(c)
            ]
            correction = sum(
                max(r * lo, r * hi) for r, (lo, hi) in zip(residual, exact_bounds)
            )
            dual_upper = exact_dot(exact_rhs, multipliers) + correction
            upper = min(upper, dual_upper)
            details = {
                "multipliers": multipliers,
                "dual_residual": list(map(round_up, residual)),
                "dual_residual_exact": list(map(str, residual)),
                "residual_support_correction": round_up(correction),
                "residual_support_correction_exact": str(correction),
                "arithmetic": "exact_rationals_of_declared_floats",
                "formula": "rhs.T @ lambda + support_of_envelope(c - rows.T @ lambda)",
                "rows": rows,
                "rhs": rhs,
                "rhs_exact": [str(Fraction(v)) for v in exact_rhs],
            }
        evidence = tuple(
            Evidence(
                name,
                "linear_support_solve",
                "linear_program",
                "verified" if e.status == "optimal" else "unresolved",
                (Measurement("maximum_primal_residual", e.max_residual, "1"),)
                if e.max_residual is not None
                else (),
                backend=e.solver,
                details={"status": e.status, "message": e.message},
            )
            for name, e in (
                ("primal_solver", primal_evidence),
                ("dual_solver", dual_evidence),
            )
        )
        return self._support(
            w,
            point,
            upper,
            "weak_duality_with_box_residual_bound",
            details=details,
            extra=evidence,
        )
