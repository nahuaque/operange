"""One boundary from native linear declarations to exact production rows."""

from fractions import Fraction as F

from ._linear_problem import (
    AffineExpression,
    LinearProblem,
    PhysicalObjective,
    PhysicalRow,
    RowSource,
    SquaredTerm,
    Variable,
)
from .objectives import LinearObjective


def expression(output):
    return AffineExpression(
        F(output.offset), tuple((t.variable, F(t.coefficient)) for t in output.terms)
    )


def objective(model, spec=None):
    spec = model.objective if spec is None else spec
    if spec is None:
        return None
    if isinstance(spec, LinearObjective):
        sign = 1 if spec.sense == "minimize" else -1
        output = model.output(spec.output)
        return PhysicalObjective(
            expression(output).scaled(sign), sign=sign, unit=output.unit
        )
    return PhysicalObjective(
        AffineExpression(),
        tuple(
            SquaredTerm(t.control, F(t.target), F(t.scale), F(t.weight))
            for t in spec.targets
        ),
    )


def compile_model(model, requirements, *, objective_spec=None):
    rows = []
    for kind, limits in (
        ("operating_limit", model.operating_limits),
        ("requirement", tuple(r for r in model.requirements if r.name in requirements)),
    ):
        for limit in limits:
            output = model.output(limit.output)
            form = expression(output).scaled(limit.sign)
            rows.append(
                PhysicalRow(
                    AffineExpression(
                        form.constant - limit.sign * F(limit.limit), form.terms
                    ),
                    F(limit.tolerance),
                    RowSource(
                        kind,
                        limit.name,
                        output.unit,
                        F(limit.residual_scale),
                        limit.sign,
                        output.name,
                    ),
                )
            )
    return LinearProblem(
        tuple(
            Variable(c.name, c.unit, c.lower, c.upper, c.quantity.role, c.physical_kind)
            for c in model.variables
        ),
        tuple(rows),
        objective(model, objective_spec),
        tuple((c.name, c.unit) for c in model.input_space.coordinates),
    )
