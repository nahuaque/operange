"""Exact production representations, independent of numerical solver backends.

Physical rows retain native-unit tolerances, scaling and source identities.
Normalized systems and diagonal objectives are explicit solver views. Their
candidate checks are production checks, never used by the independent verifier.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from fractions import Fraction as F
from types import MappingProxyType

from ._numeric import exact_dot, numeric_vector
from .primitives import finite


@dataclass(frozen=True)
class Variable:
    name: str
    unit: str
    lower: float
    upper: float
    role: str
    physical_kind: str = "declared_process_control"
    source_name: str | None = None
    scenario: int | None = None

    def __post_init__(self):
        if self.role not in ("control", "state", "auxiliary"):
            raise ValueError("invalid linear variable role")
        object.__setattr__(self, "lower", finite(self.lower, "lower bound"))
        object.__setattr__(self, "upper", finite(self.upper, "upper bound"))
        if self.lower > self.upper:
            raise ValueError("linear variable bounds must be ordered")
        if self.source_name is None:
            object.__setattr__(self, "source_name", self.name)

    def to_dict(self):
        # Historical diagnostic records use the original declaration shape.
        return {
            k: getattr(self, k)
            for k in ("name", "unit", "lower", "upper", "physical_kind")
        }


@dataclass(frozen=True)
class AffineExpression:
    constant: F = F(0)
    terms: tuple[tuple[str, F], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "constant", F(self.constant))
        object.__setattr__(self, "terms", tuple((n, F(a)) for n, a in self.terms))
        if len({n for n, _ in self.terms}) != len(self.terms):
            raise ValueError("duplicate affine expression coordinate")

    @cached_property
    def coefficients(self):
        return MappingProxyType(dict(self.terms))

    def coefficient(self, name):
        return self.coefficients.get(name, F(0))

    def substitute(self, values):
        return AffineExpression(
            self.constant
            + sum((a * F(values[n]) for n, a in self.terms if n in values), F(0)),
            tuple((n, a) for n, a in self.terms if n not in values),
        )

    def scaled(self, factor):
        return AffineExpression(
            F(factor) * self.constant, tuple((n, F(factor) * a) for n, a in self.terms)
        )


@dataclass(frozen=True)
class RowSource:
    kind: str
    name: str
    unit: str
    scale: F = F(1)
    sign: int = 1
    output: str | None = None
    scenario: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "scale", F(self.scale))
        if self.scale <= 0 or self.sign not in (-1, 1):
            raise ValueError("invalid row scale or direction")

    @property
    def ref(self):
        return f"{self.kind}:{self.name}"

    @property
    def legacy_ref(self):
        return (
            self.name if self.kind in ("operating_limit", "requirement") else self.ref
        )


@dataclass(frozen=True)
class PhysicalRow:
    """An exact native-unit residual <= tolerance; scale is purely numerical."""

    expression: AffineExpression
    tolerance: F
    source: RowSource

    def __post_init__(self):
        object.__setattr__(self, "tolerance", F(self.tolerance))
        if self.tolerance < 0:
            raise ValueError("negative physical tolerance")

    def at(self, point, names):
        expression = self.expression.substitute(point)
        if any(n not in names for n, _ in expression.terms):
            raise ValueError("unbound linear coordinate")
        return tuple(
            expression.coefficient(n) for n in names
        ), self.tolerance - expression.constant


@dataclass(frozen=True)
class SquaredTerm:
    variable: str
    target: F
    scale: F
    weight: F

    def __post_init__(self):
        for name in ("target", "scale", "weight"):
            object.__setattr__(self, name, F(getattr(self, name)))
        if self.scale <= 0 or self.weight < 0:
            raise ValueError("invalid squared objective term")


@dataclass(frozen=True)
class PhysicalObjective:
    """Signed affine cost plus scaled squares in physical variables.

    Keep squares centered for numerical synthesis. Expand only when producing
    the exact diagonal polynomial in normalized dispatch coordinates.
    """

    expression: AffineExpression
    squares: tuple[SquaredTerm, ...] = ()
    sign: int = 1
    unit: str = "1"

    def __post_init__(self):
        object.__setattr__(self, "squares", tuple(self.squares))
        if self.sign not in (-1, 1):
            raise ValueError("invalid convex objective")
        if len({t.variable for t in self.squares}) != len(self.squares):
            raise ValueError("duplicate quadratic coordinate")

    def normalized(self, system, point):
        constants = {
            **point,
            **system.fixed,
            **{c.name: c.lower for c in system.variables},
        }
        base = self.expression.substitute(constants)
        if base.terms:
            raise ValueError("unbound objective coordinate")
        squared = {t.variable: t for t in self.squares}
        constant = base.constant + sum(
            (
                t.weight * ((F(constants[t.variable]) - t.target) / t.scale) ** 2
                for t in self.squares
            ),
            F(0),
        )
        linear, diagonal = [], []
        for c, span in zip(system.variables, system.spans):
            target = squared.get(c.name)
            curvature = target.weight / target.scale**2 if target else F(0)
            slope = 2 * curvature * (F(c.lower) - target.target) if target else F(0)
            linear.append((self.expression.coefficient(c.name) + slope) * span)
            diagonal.append(curvature * span**2)
        return Polynomial(constant, tuple(linear), tuple(diagonal))


def bound_rows(variable, fixed=None):
    for side, sign in (("lower", -1), ("upper", 1)):
        yield PhysicalRow(
            AffineExpression(
                -sign * F(getattr(variable, side)), ((variable.name, F(sign)),)
            ),
            F(0),
            RowSource(
                f"{variable.role}_{side}", variable.name, variable.unit, sign=sign
            ),
        )
        if fixed is not None and variable.name in fixed:
            yield PhysicalRow(
                AffineExpression(
                    -sign * F(fixed[variable.name]), ((variable.name, F(sign)),)
                ),
                F(0),
                RowSource(f"fixed_{side}", variable.name, variable.unit, sign=sign),
            )


def coordinates(rows, upper, origins, scales, row_scales):
    """Exact x = origin + scale*z and positive row scaling, before float conversion."""
    rows = tuple(tuple(F(a) for a in row) for row in rows)
    upper, origins, scales, row_scales = (
        tuple(F(v) for v in values) for values in (upper, origins, scales, row_scales)
    )
    if (
        len(rows) != len(upper)
        or len(rows) != len(row_scales)
        or len(origins) != len(scales)
        or any(len(row) != len(scales) for row in rows)
        or any(s <= 0 for s in (*scales, *row_scales))
    ):
        raise ValueError("misaligned or nonpositive coordinate scaling")
    return (
        tuple(
            tuple(a * s / norm for a, s in zip(row, scales))
            for row, norm in zip(rows, row_scales)
        ),
        tuple(
            (b - exact_dot(row, origins)) / norm
            for row, b, norm in zip(rows, upper, row_scales)
        ),
    )


@dataclass(frozen=True)
class LinearProblem:
    variables: tuple[Variable, ...]
    rows: tuple[PhysicalRow, ...]
    objective: PhysicalObjective | None = None
    inputs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "inputs", tuple(tuple(item) for item in self.inputs))
        names = [c.name for c in self.variables] + [n for n, _ in self.inputs]
        if len(set(names)) != len(names):
            raise ValueError("duplicate linear coordinate")
        forms = [row.expression for row in self.rows]
        if self.objective is not None:
            forms.append(self.objective.expression)
            if any(
                term.variable not in {c.name for c in self.variables}
                for term in self.objective.squares
            ):
                raise ValueError("quadratic term must name a mathematical variable")
        if any(n not in names for form in forms for n, _ in form.terms):
            raise ValueError("undeclared linear coordinate")

    def _check_fixed(self, fixed):
        if set(fixed) - {c.name for c in self.variables if c.role == "control"}:
            raise ValueError("only physical actuators may have prescribed commands")

    def explicit_rows(self, fixed):
        """Retain every physical bound and prescribed command for local derivatives."""
        self._check_fixed(fixed)
        return self.rows + tuple(
            row for c in self.variables for row in bound_rows(c, fixed)
        )

    def normalized(self, point, fixed):
        """Eliminate fixed variables; retain their physical bounds as constant rows."""
        self._check_fixed(fixed)
        if set(point) != {n for n, _ in self.inputs}:
            raise ValueError("realization must cover exactly the declared inputs")
        fixed = dict(fixed)
        fixed.update(
            {
                c.name: c.lower
                for c in self.variables
                if c.lower == c.upper and c.name not in fixed
            }
        )
        variables = tuple(c for c in self.variables if c.name not in fixed)
        names = tuple(c.name for c in variables)
        physical = self.rows + tuple(
            row for c in self.variables if c.name in fixed for row in bound_rows(c)
        )
        matrix = [row.at({**point, **fixed}, names) for row in physical]
        rows, upper = coordinates(
            [r for r, _ in matrix],
            [b for _, b in matrix],
            [F(c.lower) for c in variables],
            [F(c.upper) - F(c.lower) for c in variables],
            [row.source.scale for row in physical],
        )
        return LinearSystem(
            variables,
            fixed,
            rows,
            upper,
            tuple(row.source.legacy_ref for row in physical),
            tuple(row.source for row in physical),
        )


@dataclass(frozen=True)
class LinearSystem:
    """Bounded mathematical variables, independent of their actuator/state roles.

    Exported proof keys retain historical names for artifact compatibility.
    Operating permissions are applied only by the model binding at compilation.
    """

    variables: tuple[Variable, ...]
    fixed: Mapping[str, float]
    rows: tuple[tuple[F, ...], ...]
    upper: tuple[F, ...]
    row_refs: tuple[str, ...]
    sources: tuple[RowSource, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "fixed", MappingProxyType(dict(self.fixed)))
        object.__setattr__(
            self, "rows", tuple(tuple(F(a) for a in row) for row in self.rows)
        )
        object.__setattr__(self, "upper", tuple(F(b) for b in self.upper))
        object.__setattr__(self, "row_refs", tuple(self.row_refs))
        object.__setattr__(self, "sources", tuple(self.sources))
        if (
            len(self.rows) != len(self.upper)
            or len(self.rows) != len(self.row_refs)
            or any(len(row) != len(self.variables) for row in self.rows)
            or (self.sources and len(self.sources) != len(self.rows))
            or (
                self.sources
                and tuple(s.legacy_ref for s in self.sources) != self.row_refs
            )
        ):
            raise ValueError("misaligned linear system")
        names = {c.name for c in self.variables}
        if (
            len(names) != len(self.variables)
            or names & set(self.fixed)
            or any(s <= 0 for s in self.spans)
        ):
            raise ValueError("invalid normalized variables")

    def select_rows(self, indices):
        indices = tuple(indices)
        return replace(
            self,
            rows=tuple(self.rows[i] for i in indices),
            upper=tuple(self.upper[i] for i in indices),
            row_refs=tuple(self.row_refs[i] for i in indices),
            sources=tuple(self.sources[i] for i in indices) if self.sources else (),
        )

    @property
    def spans(self):
        return tuple(F(c.upper) - F(c.lower) for c in self.variables)

    def physical_values(self, coordinates):
        if len(coordinates) != len(self.variables):
            raise ValueError("solver returned the wrong number of control coordinates")
        values = dict(self.fixed)
        for control, span, coordinate in zip(self.variables, self.spans, coordinates):
            z = F(min(1.0, max(0.0, finite(coordinate, "control coordinate"))))
            value = finite(float(F(control.lower) + span * z), control.name)
            values[control.name] = min(control.upper, max(control.lower, value))
        return values

    def checked_values(self, values):
        if set(values) != set(self.fixed) | {c.name for c in self.variables}:
            return None
        if any(values[n] != v for n, v in self.fixed.items()) or any(
            not c.lower <= values[c.name] <= c.upper for c in self.variables
        ):
            return None
        # Recheck the physical floats that will actually be returned, including
        # the rounding introduced by denormalizing the proposed coordinates.
        z = [
            (F(values[c.name]) - F(c.lower)) / span
            for c, span in zip(self.variables, self.spans)
        ]
        if any(exact_dot(row, z) > b for row, b in zip(self.rows, self.upper)):
            return None
        return values

    def candidate(self, coordinates):
        return self.checked_values(self.physical_values(coordinates))

    def objective_bound(self, objective, multipliers):
        """Checked weak duality on the normalized control box."""
        if len(objective) != len(self.variables) or len(multipliers) != len(self.rows):
            raise ValueError("invalid objective certificate dimensions")
        multipliers = [max(0.0, finite(v, "multiplier")) for v in multipliers]
        weighted = [
            exact_dot((row[j] for row in self.rows), multipliers)
            for j in range(len(self.variables))
        ]
        residual = [F(c) + a for c, a in zip(objective, weighted)]
        minimum = sum((min(F(0), a) for a in residual), F(0))
        rhs = exact_dot(self.upper, multipliers)
        return minimum - rhs, {
            "formula": "min_box((c + lambda A) z) - lambda b <= minimum relief",
            "objective_exact": list(map(str, objective)),
            "multipliers": multipliers,
            "weighted_residual_exact": list(map(str, residual)),
            "box_minimum_exact": str(minimum),
            "weighted_upper_exact": str(rhs),
            "lower_exact": str(minimum - rhs),
        }

    def certificate(self, multipliers):
        if len(multipliers) != len(self.rows):
            raise ValueError("solver returned the wrong number of multipliers")
        multipliers = [max(0.0, finite(v, "multiplier")) for v in multipliers]
        weighted = [
            exact_dot((row[j] for row in self.rows), multipliers)
            for j in range(len(self.variables))
        ]
        minimum = sum((min(F(0), a) for a in weighted), F(0))
        upper = exact_dot(self.upper, multipliers)
        gap = minimum - upper
        if gap <= 0:
            return None
        return {
            "formula": "min over 0 <= z <= 1 of (lambda A) z > lambda b",
            "arithmetic": "exact_rationals_of_declared_floats",
            "row_constraint_refs": self.row_refs,
            "rows_exact": [[str(a) for a in row] for row in self.rows],
            "upper_exact": [str(b) for b in self.upper],
            "multipliers": multipliers,
            "weighted_coefficients_exact": list(map(str, weighted)),
            "box_minimum_exact": str(minimum),
            "weighted_upper_exact": str(upper),
            "contradiction_gap_exact": str(gap),
            "control_coordinates": [
                {
                    "name": c.name,
                    "origin_exact": str(F(c.lower)),
                    "scale_exact": str(span),
                }
                for c, span in zip(self.variables, self.spans)
            ],
            "fixed_controls": self.fixed,
            "scope": "joint infeasibility; no claim of a minimal conflict or individually impossible requirements",
        }


@dataclass(frozen=True)
class Polynomial:
    constant: F
    linear: tuple
    diagonal: tuple

    def __post_init__(self):
        object.__setattr__(self, "constant", F(self.constant))
        object.__setattr__(self, "linear", tuple(F(v) for v in self.linear))
        object.__setattr__(self, "diagonal", tuple(F(v) for v in self.diagonal))
        if len(self.linear) != len(self.diagonal) or any(v < 0 for v in self.diagonal):
            raise ValueError("invalid separable objective dimensions or curvature")

    def value(self, system, controls):
        if len(self.linear) != len(system.variables):
            raise ValueError("misaligned objective")
        z = [
            (F(controls[c.name]) - F(c.lower)) / span
            for c, span in zip(system.variables, system.spans)
        ]
        return (
            self.constant
            + exact_dot(self.linear, z)
            + sum(d * v * v for d, v in zip(self.diagonal, z))
        )

    def bound(self, system, multipliers):
        if len(self.linear) != len(system.variables):
            raise ValueError("misaligned objective")
        values = numeric_vector(multipliers, len(system.rows), "objective multipliers")
        if values is None:
            raise ValueError("missing objective multipliers")
        multipliers = [max(0.0, v) for v in values]
        slopes = [
            c + exact_dot((row[i] for row in system.rows), multipliers)
            for i, c in enumerate(self.linear)
        ]
        minimizers = [
            min(F(1), max(F(0), -c / (2 * d))) if d else F(c < 0)
            for c, d in zip(slopes, self.diagonal)
        ]
        bound = (
            self.constant
            - exact_dot(system.upper, multipliers)
            + sum(
                d * z * z + c * z for d, c, z in zip(self.diagonal, slopes, minimizers)
            )
        )
        return bound, {
            "formula": "k - lambda.b + sum(min_0<=z_i<=1(d_i*z_i**2 + (c_i + (lambda.A)_i)*z_i))",
            "multipliers": multipliers,
            "lagrangian_slopes_exact": list(map(str, slopes)),
            "box_minimizers_exact": list(map(str, minimizers)),
            "minimization_lower_exact": str(bound),
        }
