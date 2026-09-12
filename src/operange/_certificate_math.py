"""Independent rational checks. No producer compiler, solver or audit is called."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import inf, isfinite, nextafter

from .claim import Claim
from .contract_types import plain
from .controllers import AffineController
from .domain_io import domain_from_manifest
from .domains import FiniteSet
from .convex_hull import ConvexHullSet
from .linear_process import LinearProcessAdapter
from .primitives import BoxSet
from .recourse import RecoursePolicy


class UnsupportedCertificate(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def q(value):
    if isinstance(value, str):
        require(
            len(value) <= 4096 and all(c in "-+0123456789/.eE" for c in value),
            "invalid or oversized exact rational",
        )
        if "e" in value.lower():
            require(
                abs(int(value.lower().split("e")[1])) <= 4096,
                "oversized rational exponent",
            )
    else:
        require(
            type(value) in (int, float, Fraction), "expected a finite rational number"
        )
    return Fraction(value)


def dot(a, b):
    require(len(a) == len(b), "certificate vector dimensions differ")
    return sum((q(x) * q(y) for x, y in zip(a, b)), Fraction(0))


def rounded(value, direction=0):
    value = q(value)
    result = float(value)
    require(isfinite(result), "unexportable physical value")
    if direction > 0 and q(result) < value:
        result = nextafter(result, inf)
    if direction < 0 and q(result) > value:
        result = nextafter(result, -inf)
    require(isfinite(result), "unexportable directed bound")
    return result


def exact_equal(actual, expected, message):
    require(q(actual) == q(expected), message)


def vector_equal(actual, expected, message):
    require(len(actual) == len(expected), message)
    require(all(q(a) == q(b) for a, b in zip(actual, expected)), message)


@dataclass
class System:
    # Coordinates are physical (name, origin, span), normalized to [0, 1].
    coordinates: list
    fixed: dict
    rows: list
    rhs: list
    refs: list

    def match(self, data, *, rhs="upper_exact", exact_coordinates=True):
        require(
            list(data["row_constraint_refs"]) == self.refs,
            "rows omit, reorder or substitute declared constraints",
        )
        require(len(data["rows_exact"]) == len(self.rows), "incorrect row count")
        for a, b in zip(data["rows_exact"], self.rows):
            vector_equal(
                a, b, "exported row differs from the declared physical problem"
            )
        vector_equal(
            data[rhs],
            self.rhs,
            "exported right-hand side differs from the declared problem",
        )
        exported = data["control_coordinates"]
        require(
            len(exported) == len(self.coordinates), "incorrect control coordinate count"
        )
        for item, (name, origin, span) in zip(exported, self.coordinates):
            require(item["name"] == name, "control coordinate identity differs")
            exact_equal(
                item["origin_exact" if exact_coordinates else "origin"],
                origin,
                "control origin differs",
            )
            exact_equal(
                item["scale_exact"]
                if exact_coordinates
                else q(item["upper"]) - q(item["origin"]),
                span,
                "control span differs",
            )
        if exact_coordinates or "fixed_controls" in data:
            require(
                dict(data["fixed_controls"]) == self.fixed,
                "fixed control declaration differs",
            )

    def feasible(self, commands):
        require(
            set(commands) == set(self.fixed) | {n for n, _, _ in self.coordinates},
            "candidate control keys differ",
        )
        require(
            all(commands[n] == v for n, v in self.fixed.items()),
            "candidate changes a fixed command",
        )
        z = [(q(commands[n]) - origin) / span for n, origin, span in self.coordinates]
        require(all(0 <= v <= 1 for v in z), "candidate violates a hard control bound")
        require(
            all(dot(row, z) <= b for row, b in zip(self.rows, self.rhs)),
            "candidate violates a declared row",
        )

    def contradiction(self, data):
        self.match(data)
        multipliers = [q(v) for v in data["multipliers"]]
        require(
            len(multipliers) == len(self.rows) and all(v >= 0 for v in multipliers),
            "invalid contradiction multipliers",
        )
        weighted = [
            dot([r[i] for r in self.rows], multipliers)
            for i in range(len(self.coordinates))
        ]
        lower, upper = (
            sum((min(0, a) for a in weighted), Fraction(0)),
            dot(self.rhs, multipliers),
        )
        vector_equal(
            data["weighted_coefficients_exact"],
            weighted,
            "weighted coefficients differ",
        )
        exact_equal(data["box_minimum_exact"], lower, "box minimum differs")
        exact_equal(data["weighted_upper_exact"], upper, "weighted upper differs")
        exact_equal(
            data["contradiction_gap_exact"], lower - upper, "contradiction gap differs"
        )
        require(lower > upper, "row combination does not prove infeasibility")

    def lower_bound(self, constant, linear, diagonal, data):
        multipliers = [q(v) for v in data["multipliers"]]
        require(
            len(multipliers) == len(self.rows) and all(v >= 0 for v in multipliers),
            "invalid objective multipliers",
        )
        require(
            len(linear) == len(diagonal) == len(self.coordinates)
            and all(d >= 0 for d in diagonal),
            "invalid convex objective dimensions",
        )
        slopes = [
            c + dot([r[i] for r in self.rows], multipliers)
            for i, c in enumerate(linear)
        ]
        minimizers = [
            min(Fraction(1), max(Fraction(0), -c / (2 * d))) if d else Fraction(c < 0)
            for c, d in zip(slopes, diagonal)
        ]
        lower = (
            constant
            - dot(self.rhs, multipliers)
            + sum(
                (d * z * z + c * z for d, c, z in zip(diagonal, slopes, minimizers)),
                Fraction(0),
            )
        )
        vector_equal(
            data["lagrangian_slopes_exact"], slopes, "Lagrangian slopes differ"
        )
        vector_equal(
            data["box_minimizers_exact"], minimizers, "Lagrangian minimizers differ"
        )
        exact_equal(
            data["minimization_lower_exact"], lower, "Lagrangian lower bound differs"
        )
        return lower


class Context:
    def __init__(self, contract):
        if contract.model_id != "linear_process/v1":
            raise UnsupportedCertificate(
                "Only declarative linear-process certificates are supported."
            )
        if contract.domain.get("kind") not in ("box", "finite_set", "convex_hull"):
            raise UnsupportedCertificate(
                "Independent verification supports boxes, finite sets and explicit hull generators."
            )
        self.model = LinearProcessAdapter(**plain(contract.model))
        self.domain = domain_from_manifest(plain(contract.domain))
        self.policy = RecoursePolicy(**plain(contract.operating["recourse_policy"]))
        self.selected = tuple(contract.operating["requirement_refs"])
        self.claim = Claim(self.model, self.domain, self.policy, self.selected)
        self.controller = None
        if contract.operating.get("response") == "frozen_affine_controller":
            import json

            self.controller = AffineController.from_json(
                json.dumps(plain(contract.operating["controller"]))
            )
            self.claim = self.claim.with_controller(self.controller)
        require(
            self.claim.contract == contract,
            "contract registry or numerical policy differs from its model declaration",
        )
        require(
            self.policy.mode in ("static", "fixed"), "unsupported operating permissions"
        )
        require(
            {r.control for r in self.policy.rules}
            == {c.name for c in self.model.controls},
            "missing control permissions",
        )
        for rule in self.policy.rules:
            require(rule.stage == "operation", "unsupported operating stage")
            if self.controller is None and rule.fixed_value is None:
                require(
                    self.policy.mode == "static"
                    and set(rule.observes) == set(self.domain.space.names),
                    "adjustable certificates require full observations",
                )
        self.limits = self.model.operating_limits + tuple(
            r for r in self.model.requirements if r.name in self.selected
        )
        self.included = tuple(
            c
            for c in contract.constraints
            if c.kind != "requirement" or c.constraint_id in self.selected
        )

    def membership(self, point):
        point = self.domain.space.validate(point)
        if type(self.domain) is BoxSet:
            return all(
                q(p.lower) <= q(point[p.name]) <= q(p.upper)
                for p in self.domain.scalar_parameters
            )
        cases = (
            self.domain.scenarios
            if type(self.domain) is FiniteSet
            else self.domain.vertices
        )
        if any(dict(s.values) == point for s in cases):
            return True
        if type(self.domain) is ConvexHullSet:
            raise UnsupportedCertificate(
                "Hull membership away from declared generators needs a separate barycentric certificate."
            )
        return False

    def generators(self, maximum=10000):
        if type(self.domain) is BoxSet:
            parameters = sorted(self.domain.scalar_parameters, key=lambda p: p.name)
            if 2 ** sum(p.lower != p.upper for p in parameters) > maximum:
                raise UnsupportedCertificate("Generator verification limit exceeded.")
            return [
                (f"corner_{i}", dict(zip((p.name for p in parameters), v)))
                for i, v in enumerate(
                    product(
                        *(
                            (p.lower,) if p.lower == p.upper else (p.lower, p.upper)
                            for p in parameters
                        )
                    )
                )
            ]
        cases = (
            self.domain.scenarios
            if type(self.domain) is FiniteSet
            else self.domain.vertices
        )
        require(len(cases) <= maximum, "too many declared generators")
        return [(s.name, dict(s.values)) for s in cases]

    def output(self, name, variables):
        output = self.model.output(name)
        return q(output.offset) + sum(
            (q(t.coefficient) * q(variables[t.variable]) for t in output.terms),
            Fraction(0),
        )

    def system(self, point):
        fixed = {
            r.control: r.fixed_value
            for r in self.policy.rules
            if r.fixed_value is not None
        }
        fixed.update(
            {
                c.name: c.lower
                for c in self.model.controls
                if c.lower == c.upper and c.name not in fixed
            }
        )
        coords = [
            (c.name, q(c.lower), q(c.upper) - q(c.lower))
            for c in self.model.controls
            if c.name not in fixed
        ]
        base = {**point, **fixed, **{n: origin for n, origin, _ in coords}}
        rows, rhs, refs = [], [], []
        for limit in self.limits:
            sign = 1 if limit.relation == "le" else -1
            terms = {
                t.variable: q(t.coefficient)
                for t in self.model.output(limit.output).terms
            }
            rows.append(
                [
                    sign * terms.get(n, 0) * span / q(limit.residual_scale)
                    for n, _, span in coords
                ]
            )
            rhs.append(
                (
                    q(limit.tolerance)
                    - sign * (self.output(limit.output, base) - q(limit.limit))
                )
                / q(limit.residual_scale)
            )
            refs.append(limit.name)
        for c in self.model.controls:
            if c.name in fixed:
                for side, value in (
                    ("lower", q(fixed[c.name]) - q(c.lower)),
                    ("upper", q(c.upper) - q(fixed[c.name])),
                ):
                    rows.append([Fraction(0)] * len(coords))
                    rhs.append(value)
                    refs.append(f"control_{side}:{c.name}")
        return System(coords, fixed, rows, rhs, refs)

    def response(self, point, commands):
        require(
            set(commands) == {c.name for c in self.model.controls},
            "missing or extra commands",
        )
        require(
            all(
                r.fixed_value is None or commands[r.control] == r.fixed_value
                for r in self.policy.rules
            ),
            "command contradicts a fixed permission",
        )
        variables = {**point, **commands}
        outputs = {o.name: self.output(o.name, variables) for o in self.model.outputs}
        residuals = {f"affine_definition:{o.name}": 0.0 for o in self.model.outputs}
        for r in self.limits:
            residuals[r.name] = rounded(
                (1 if r.relation == "le" else -1) * (outputs[r.output] - q(r.limit)), 1
            )
        for c in self.model.controls:
            residuals[f"control_lower:{c.name}"] = rounded(
                q(c.lower) - q(commands[c.name]), 1
            )
            residuals[f"control_upper:{c.name}"] = rounded(
                q(commands[c.name]) - q(c.upper), 1
            )
        values = {
            **variables,
            **{n: rounded(v) for n, v in outputs.items()},
            **{f"residual:{n}": v for n, v in residuals.items()},
        }
        return values, residuals

    def commands(self, point):
        return {
            r.control: rounded(
                q(r.offset)
                + sum(
                    (q(t.coefficient) * q(point[t.variable]) for t in r.terms),
                    Fraction(0),
                )
            )
            for r in self.controller.rules
        }
