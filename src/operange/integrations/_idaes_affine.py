"""Conservative affine extraction with exact arithmetic and no linearization."""

from dataclasses import dataclass
from fractions import Fraction
from numbers import Real

from .._artifact_json import (
    dumps,
    read_json,
    require_fields,
    require_schema,
)
from ..affine import AffineOutput, AffineRequirement, AffineTerm
from ..contract_types import ProcessContract, Record
from ..linear_process import LinearControl, LinearState, LinearProcessAdapter
from ..objectives import LinearObjective
from ..recourse import DecisionRule, RecoursePolicy
from ._idaes_model import backend, component, number, uid, unit


def _add(left, right):
    offset = left[0] + right[0]
    terms = dict(left[1])
    for name, value in right[1].items():
        terms[name] = terms.get(name, Fraction(0)) + value
    return offset, {name: value for name, value in terms.items() if value}


def _scale(poly, value):
    return poly[0] * value, {
        name: coef * value for name, coef in poly[1].items() if coef * value
    }


def _polynomial(expr, variables):
    """Expand a deliberately small Pyomo arithmetic grammar into exact rationals.

    Evaluating generic coefficient expressions as floats could silently round
    sums/products. This visitor rejects those roundings at the export boundary.
    Functions, conditionals, external calls and nonlinear products are rejected.
    """
    from pyomo.core.base.units_container import _PyomoUnit

    if id(expr) in variables:
        return Fraction(0), {variables[id(expr)]: Fraction(1)}
    if isinstance(expr, Real):
        return Fraction(expr), {}
    if isinstance(expr, _PyomoUnit):
        return Fraction(1), {}
    if expr.is_named_expression_type():
        return _polynomial(expr.expr, variables)
    if expr.is_variable_type():
        if not expr.fixed:
            raise ValueError(f"Unmapped free variable: {expr.name}")
        return Fraction(number(expr)), {}
    if expr.is_parameter_type() or not expr.is_expression_type():
        return Fraction(number(expr)), {}
    kind = type(expr).__name__.removeprefix("NPV_")
    if kind in {"SumExpression", "LinearExpression"}:
        result = Fraction(0), {}
        for arg in expr.args:
            result = _add(result, _polynomial(arg, variables))
        return result
    if kind == "NegationExpression":
        return _scale(_polynomial(expr.args[0], variables), -1)
    if kind in {"ProductExpression", "MonomialTermExpression"}:
        left, right = (_polynomial(arg, variables) for arg in expr.args)
        if left[1] and right[1]:
            raise ValueError(
                "Nonlinear products cannot be exported as affine equations"
            )
        return _scale(left, right[0]) if not right[1] else _scale(right, left[0])
    if kind == "DivisionExpression":
        top, bottom = (_polynomial(arg, variables) for arg in expr.args)
        if bottom[1] or bottom[0] == 0:
            raise ValueError("An affine export needs a nonzero constant denominator")
        return _scale(top, 1 / bottom[0])
    if kind == "PowExpression":
        base, exponent = (_polynomial(arg, variables) for arg in expr.args)
        if not exponent[1] and exponent[0] == 1:
            return base
        if not exponent[1] and exponent[0] == 0:
            return Fraction(1), {}
    raise ValueError(
        f"Unsupported affine expression: {type(expr).__name__}; no approximation was made"
    )


def _exact_float(value):
    try:
        result = float(value)
        exact = Fraction(result) == value
    except (OverflowError, ValueError):
        exact = False
    if not exact:
        raise ValueError(
            "Affine coefficient is not exactly representable as a float; rescale the source equation explicitly"
        )
    return result


@dataclass(frozen=True)
class IdaesLinearExport(Record):
    """A portable coefficient snapshot with its source declaration and role map.

    The existing independent verifier checks results on ``adapter``. It does
    not execute Pyomo or independently establish source-to-export equivalence.
    """

    adapter: LinearProcessAdapter
    source_contract: ProcessContract
    actuator_names: tuple[str, ...]

    def as_claim(self, domain, *, recourse=None, operating=None, requirements=None):
        """Preserve physical actuator permissions independently of algebraic states."""
        if self.adapter.states or operating is not None:
            claim = self.adapter.as_claim(
                domain,
                recourse=recourse,
                operating=operating,
                requirements=requirements,
            )
            if claim.adapter is self.adapter:
                self.adapter._permissions(claim)
            return claim
        # Compatibility for older exports that declared algebraic rates as controls.
        if recourse is not None:
            if recourse.mode not in {"static", "fixed"} or {
                r.control for r in recourse.rules
            } != set(self.actuator_names):
                raise ValueError(
                    "Supply fixed/static permissions for exactly the original actuators"
                )
            rules = list(recourse.rules)
            for rule in rules:
                if rule.fixed_value is None and (
                    recourse.mode != "static"
                    or set(rule.observes) != set(self.adapter.input_space.names)
                ):
                    raise ValueError("Adjustable actuators must observe every input")
            rules.extend(
                DecisionRule(
                    c.name, c.unit, "operation", self.adapter.input_space.names
                )
                for c in self.adapter.controls
                if c.name not in self.actuator_names
            )
            recourse = RecoursePolicy(
                "static" if any(r.fixed_value is None for r in rules) else "fixed",
                tuple(rules),
            )
        return self.adapter.as_claim(
            domain, recourse=recourse, requirements=requirements
        )

    def to_json(self):
        return dumps(
            {"schema_version": "idaes_linear_export/v1", **self.to_dict()},
            indent=2,
        )

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(read_json(text))

    @classmethod
    def from_dict(cls, data):
        require_schema(data, "idaes_linear_export/v1")
        require_fields(
            data, ("schema_version", "adapter", "source_contract", "actuator_names")
        )
        try:
            return cls(**{k: v for k, v in data.items() if k != "schema_version"})
        except (TypeError, KeyError) as exc:
            raise ValueError("invalid IDAES linear export fields") from exc


def export_linear(source):
    pyo, _ = backend()
    model = source._template
    variables = {
        id(component(model, binding.component)): binding.name
        for binding in source.inputs
    }
    control_bindings = {
        id(component(model, binding.component)): binding.name
        for binding in source.controls
    }
    candidates = []
    for var in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        if id(var) in variables:
            continue
        if not var.fixed or id(var) in control_bindings:
            name = control_bindings.get(id(var), f"state:{uid(var)}")
            variables[id(var)] = name
            candidates.append((name, var))
    controls, states = [], []
    for name, var in candidates:
        if var.lower is None or var.upper is None:
            raise ValueError(f"Affine export requires finite bounds on {uid(var)}")
        lower, upper = (
            _polynomial(bound, variables) for bound in (var.lower, var.upper)
        )
        if lower[1] or upper[1]:
            raise ValueError(f"Affine control bounds must be constant: {uid(var)}")
        actuator = id(var) in control_bindings
        (controls if actuator else states).append(
            (LinearControl if actuator else LinearState)(
                name,
                unit(var),
                _exact_float(lower[0]),
                _exact_float(upper[0]),
                "declared_actuator"
                if id(var) in control_bindings
                else "algebraic_state",
            )
        )
    if not controls and not states:
        raise ValueError(
            "This export requires at least one bounded actuator or algebraic state"
        )
    units = {b.name: b.unit for b in source.inputs}
    units.update({c.name: c.unit for c in controls + states})
    fingerprint = source.base_contract.model["source_fingerprint"]

    def output(name, expr, native_unit, origin):
        offset, terms = _polynomial(expr, variables)
        return AffineOutput(
            name,
            native_unit,
            "idaes_affine_quantity",
            _exact_float(offset),
            tuple(
                AffineTerm(key, _exact_float(value), f"{native_unit}/{units[key]}")
                for key, value in sorted(terms.items())
            ),
            f"IDAES/Pyomo source {fingerprint}; {origin}; exact rational arithmetic; no linearization",
        )

    outputs = [
        output(b.name, component(model, b.component), b.unit, b.component)
        for b in source.outputs
    ]
    limits, requirements = [], []
    for row in source._rows:
        name = f"source_row:{row.name}"
        outputs.append(
            output(name, row.expression(model), row.unit, f"{row.path}:{row.side}")
        )
        target = requirements if row.kind == "requirement" else limits
        target.append(AffineRequirement(row.name, name, 0, "le", row.tolerance))
        if row.side == "eq":
            target.append(
                AffineRequirement(f"{row.name}:reverse", name, 0, "ge", row.tolerance)
            )
    objectives = tuple(
        model.component_data_objects(pyo.Objective, active=True, descend_into=True)
    )
    objective = None
    if objectives:
        obj = objectives[0]
        name = "source_objective"
        outputs.append(output(name, obj.expr, unit(obj.expr), uid(obj)))
        objective = LinearObjective(
            name,
            "minimize" if obj.sense == pyo.minimize else "maximize",
            provenance=f"Source objective {uid(obj)}",
        )
    adapter = LinearProcessAdapter(
        source.name,
        source.input_space,
        tuple(outputs),
        tuple(requirements),
        tuple(controls),
        tuple(limits),
        objective=objective,
        states=tuple(states),
    )
    return IdaesLinearExport(
        adapter, source.base_contract, tuple(c.name for c in source.controls)
    )
