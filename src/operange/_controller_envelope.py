"""Continuous bounds for the actual, once-rounded affine controller commands."""

from dataclasses import dataclass
from fractions import Fraction
from math import ulp

from ._numeric import normalized_score, round_down, round_up
from .affine_results import _checked_support
from .contract_types import (
    Coverage,
    Diagnostic,
    Evidence,
    Measurement,
    RobustnessPayload,
)
from .engineering_results import RobustnessResult
from .primitives import BoxSet, finite


@dataclass
class Residual:
    name: str
    unit: str
    tolerance: float
    offset: Fraction
    coefficients: dict
    error: Fraction
    control_bound: tuple | None = None

    def actual(self, claim, point):
        commands = claim.adapter.controller.commands(point)
        if self.control_bound is not None:
            name, side, limit = self.control_bound
            return (
                Fraction(limit) - Fraction(commands[name])
                if side == "lower"
                else Fraction(commands[name]) - Fraction(limit)
            )
        model = claim.adapter.model
        limit = next(
            r
            for r in model.operating_limits + model.requirements
            if r.name == self.name
        )
        return limit.sign * (
            model.output(limit.output)._exact_value({**point, **commands})
            - Fraction(limit.limit)
        )


class ControllerEnvelope:
    def __init__(self, claim):
        self.claim = claim
        self.model = claim.adapter.model
        self.rules = {r.control: r for r in claim.adapter.controller.rules}
        self.commands = {}
        self.points = []
        self.proofs = []
        for name, rule in self.rules.items():
            coefficients = {t.variable: Fraction(t.coefficient) for t in rule.terms}
            lo, hi = self.interval(Fraction(rule.offset), coefficients)
            # Rounding to nearest is monotone, including at equipment bounds.
            rounded_lo, rounded_hi = finite(float(lo), name), finite(float(hi), name)
            identity = (
                rule.offset == 0
                and len(rule.terms) == 1
                and abs(rule.terms[0].coefficient) == 1
            )
            constant = not any(coefficients.values())
            error = (
                Fraction(0)
                if identity or constant
                else Fraction(ulp(round_up(max(abs(lo), abs(hi))))) / 2
            )
            self.commands[name] = (rounded_lo, rounded_hi, error)
        # Every declared output must remain exportable, even when unselected.
        for output in self.model.outputs:
            offset, coefficients, error = self.output_form(output)
            lo, hi = self.interval(offset, coefficients)
            round_down(lo - error)
            round_up(hi + error)

    def upper(self, offset, coefficients):
        space = self.claim.domain.space
        if not any(coefficients.values()):
            return offset
        if type(self.claim.domain) is BoxSet:
            point = {
                p.name: p.upper if coefficients.get(p.name, 0) > 0 else p.lower
                for p in self.claim.domain.scalar_parameters
            }
            upper = offset + sum(
                (a * Fraction(point[n]) for n, a in coefficients.items()), Fraction(0)
            )
            self.points.append(point)
            self.proofs.append(
                {
                    "method": "exact_box_affine_extremum",
                    "offset_exact": str(offset),
                    "coefficients_exact": {n: str(a) for n, a in coefficients.items()},
                    "point": point,
                    "upper_exact": str(upper),
                }
            )
            return upper
        baseline = offset + sum(
            (
                a * Fraction(space.coordinate(n).nominal)
                for n, a in coefficients.items()
            ),
            Fraction(0),
        )
        weights = {
            c.name: coefficients.get(c.name, Fraction(0)) * Fraction(c.scale)
            for c in space.coordinates
        }
        rounded = {n: finite(float(a), n) for n, a in weights.items()}
        support = _checked_support(self.claim, rounded)
        check = self.claim.domain.membership(support.point)
        if (
            check.status != "inside"
            or abs(
                round_down(normalized_score(space, rounded, support.point))
                - support.lower
            )
            > support.tolerance
        ):
            raise ValueError("support candidate or lower bound could not be verified")
        correction, proofs = Fraction(0), {}
        for n, a in weights.items():
            error = a - Fraction(rounded[n])
            if error:
                bound = _checked_support(self.claim, {n: 1 if error > 0 else -1})
                correction += abs(error) * (
                    Fraction(bound.upper) + Fraction(bound.tolerance)
                )
                proofs[n] = bound.to_dict()
        upper = (
            baseline
            + Fraction(support.upper)
            + Fraction(support.tolerance)
            + correction
        )
        self.points.append(dict(support.point))
        self.proofs.append(
            {
                "method": "normalized_support_with_exact_coefficient_correction",
                "offset_exact": str(offset),
                "coefficients_exact": {n: str(a) for n, a in coefficients.items()},
                "support": support.to_dict(),
                "coefficient_corrections": proofs,
                "upper_exact": str(upper),
            }
        )
        return upper

    def interval(self, offset, coefficients):
        return (
            -self.upper(-offset, {n: -a for n, a in coefficients.items()}),
            self.upper(offset, coefficients),
        )

    def output_form(self, output):
        offset, coefficients, error = Fraction(output.offset), {}, Fraction(0)
        for term in output.terms:
            a = Fraction(term.coefficient)
            if term.variable in self.rules:
                rule = self.rules[term.variable]
                offset += a * Fraction(rule.offset)
                error += abs(a) * self.commands[term.variable][2]
                for observed in rule.terms:
                    n = observed.variable
                    coefficients[n] = coefficients.get(n, Fraction(0)) + a * Fraction(
                        observed.coefficient
                    )
            else:
                n = term.variable
                coefficients[n] = coefficients.get(n, Fraction(0)) + a
        return offset, coefficients, error

    def residuals(self):
        rows = []
        for limit in self.model.operating_limits + tuple(
            r for r in self.model.requirements if r.name in self.claim.requirements
        ):
            output = self.model.output(limit.output)
            offset, coefficients, error = self.output_form(output)
            rows.append(
                Residual(
                    limit.name,
                    output.unit,
                    limit.tolerance,
                    limit.sign * (offset - Fraction(limit.limit)),
                    {n: limit.sign * a for n, a in coefficients.items()},
                    error,
                )
            )
        for control in self.model.controls:
            rule = self.rules[control.name]
            for side, sign, limit in (
                ("lower", -1, control.lower),
                ("upper", 1, control.upper),
            ):
                rows.append(
                    Residual(
                        self.model.bound_id(control, side),
                        control.unit,
                        0,
                        sign * (Fraction(rule.offset) - Fraction(limit)),
                        {
                            t.variable: sign * Fraction(t.coefficient)
                            for t in rule.terms
                        },
                        self.commands[control.name][2],
                        (control.name, side, limit),
                    )
                )
        return rows

    def residual_upper(self, row):
        if row.control_bound is not None:
            name, side, limit = row.control_bound
            lo, hi, _ = self.commands[name]
            round_down(
                Fraction(limit) - Fraction(hi)
                if side == "lower"
                else Fraction(lo) - Fraction(limit)
            )
            return (
                Fraction(limit) - Fraction(lo)
                if side == "lower"
                else Fraction(hi) - Fraction(limit)
            )
        lo, hi = self.interval(row.offset, row.coefficients)
        round_down(lo - row.error)
        return hi + row.error


def audit_result(claim):
    from ._controller_results import _failed_constraints, _failure, evaluate_result

    evaluations, evidence, bounds = {}, [], {}

    def evaluate(point):
        key = tuple(point[n] for n in claim.domain.space.names)
        if key not in evaluations:
            evaluations[key] = evaluate_result(claim, point)

    seed = getattr(claim.domain, "feasible_point", None)
    if seed is not None:
        evaluate(seed)
    envelope, complete = None, False
    try:
        envelope = ControllerEnvelope(claim)
        for row in envelope.residuals():
            upper = round_up(envelope.residual_upper(row))
            bounds[row.name] = upper
            evidence.append(
                Evidence(
                    f"bound:{row.name}",
                    "requirement_bound",
                    "rounded_controller_affine_enclosure",
                    "verified",
                    (Measurement("residual_upper", upper, row.unit),),
                    (Measurement("requirement_tolerance", row.tolerance, row.unit),),
                    details={
                        "constraint": row.name,
                        "offset_exact": str(row.offset),
                        "coefficients_exact": {
                            n: str(a) for n, a in row.coefficients.items()
                        },
                        "command_rounding_error_exact": str(row.error),
                        "control_bound": row.control_bound,
                    },
                )
            )
        complete = True
    except (ValueError, TypeError, OverflowError, RuntimeError) as exc:
        evidence.append(
            Evidence(
                "enclosure_unresolved",
                "requirement_bound",
                "rounded_controller_affine_enclosure",
                "unresolved",
                details={"message": str(exc)},
            )
        )
    if envelope is not None:
        for point in envelope.points:
            evaluate(point)
        evidence.append(
            Evidence(
                "enclosure",
                "controller_response",
                "exact_substitution_and_command_rounding_bounds",
                "verified",
                details={
                    "controller_ref": claim.adapter.controller.ref.to_dict(),
                    "support_bounds": envelope.proofs,
                    "command_bounds": {
                        n: {
                            "lower": lo,
                            "upper": hi,
                            "rounding_error_exact": str(error),
                        }
                        for n, (lo, hi, error) in envelope.commands.items()
                    },
                    "scope": "same saved controller throughout the declared domain; no redispatch",
                },
            )
        )
    witness = None
    for result in evaluations.values():
        if (
            result.execution == "completed"
            and result.payload.membership.status == "inside"
        ):
            affected = _failed_constraints(claim, result)
            if affected:
                witness, proofs = _failure(claim, result, affected)
                evidence.extend(proofs)
                break
    passed = (
        complete
        and all(bounds[r.name] <= r.tolerance for r in envelope.residuals())
        and any(e.payload.feasibility == "feasible" for e in evaluations.values())
    )
    verdict = "fail" if witness else "pass" if passed else "inconclusive"
    if complete:
        evidence.append(
            Evidence(
                "coverage",
                "domain_coverage",
                "rounded_controller_constraint_enclosures",
                "verified",
                details={
                    "requirement_bounds": bounds,
                    "enclosure_evidence_ref": "enclosure",
                },
            )
        )
    return RobustnessResult(
        claim.contract,
        {"query": "audit"},
        "unresolved" if verdict == "inconclusive" else "completed",
        RobustnessPayload(
            verdict,
            Coverage(
                "analytical_domain" if complete else "partial",
                evaluated_support={
                    "realizations": [
                        e.request["realization"] for e in evaluations.values()
                    ]
                },
                unexplored_support={
                    "reason": "bounds do not establish a pass and no controller violation was verified"
                }
                if verdict == "inconclusive"
                else {},
                evidence_refs=("coverage",) if complete else (),
            ),
            tuple(e.ref for e in evaluations.values()),
            witness,
        ),
        tuple(evidence),
        (
            Diagnostic(
                "audit_inconclusive",
                "requirements",
                "An upper bound above a limit is not a failure witness.",
            ),
        )
        if verdict == "inconclusive"
        else (),
        tuple(evaluations.values()),
    )
