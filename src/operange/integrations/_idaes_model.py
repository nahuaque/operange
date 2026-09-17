"""Snapshot, identity, units and residual rows for the optional IDAES bridge."""

from dataclasses import dataclass

from ..contract_types import ConstraintSpec, ProcessContract, QuantitySpec, digest
from ..primitives import finite


def backend():
    try:
        import idaes
        import pyomo
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError(
            "Install operange[idaes] to bind an IDAES/Pyomo model"
        ) from exc
    return pyo, {"idaes": idaes.__version__, "pyomo": pyomo.__version__}


def component(model, path):
    from pyomo.core.base.componentuid import ComponentUID

    value = ComponentUID(path).find_component_on(model)
    if value is None or not hasattr(value, "is_indexed") or value.is_indexed():
        raise ValueError(f"Binding must identify one scalar component: {path}")
    return value


def uid(item):
    from pyomo.core.base.componentuid import ComponentUID

    return str(ComponentUID(item))


def unit(item):
    pyo, _ = backend()
    result = str(pyo.units.get_units(item))
    return "1" if result == "dimensionless" else result


def number(expr):
    pyo, _ = backend()
    value = pyo.value(expr, exception=False)
    if value is None:
        raise ValueError(f"Undefined Pyomo value: {expr}")
    return finite(float(value), "Pyomo value")


@dataclass(frozen=True)
class Row:
    name: str
    path: str
    side: str
    unit: str
    kind: str
    tolerance: float
    variable: bool = False

    def expression(self, model):
        item = component(model, self.path)
        body = item if self.variable else item.body
        return (
            item.lower - body
            if self.side == "lower"
            else body - (item.lower if self.side == "eq" else item.upper)
        )

    @property
    def residual(self):
        return f"residual:{self.name}"


def prepare_template(adapter, model):
    pyo, _ = backend()
    from pyomo.core.base.units_container import _PyomoUnit
    from pyomo.core.base.external import AMPLExternalFunction
    from pyomo.network import Port
    from pyomo.util.check_units import assert_units_consistent

    if not isinstance(model, pyo.ConcreteModel) or model.parent_block() is not None:
        raise ValueError(
            "Supply a constructed root Pyomo ConcreteModel containing the flowsheet"
        )
    template = model.clone()
    # Reject active modelling constructs whose semantics a row check would omit.
    allowed = {
        pyo.Block,
        pyo.Var,
        pyo.Param,
        pyo.Constraint,
        pyo.Objective,
        pyo.Expression,
        pyo.Set,
        pyo.RangeSet,
        pyo.Suffix,
        pyo.ExternalFunction,
        Port,
    }
    for item in template.component_objects(active=True, descend_into=True):
        if item.ctype not in allowed:
            raise ValueError(
                f"Unsupported active model component: {item.name} ({item.ctype})"
            )
        if item.ctype is pyo.ExternalFunction and not isinstance(
            item, AMPLExternalFunction
        ):
            raise ValueError(
                "Python external callbacks are outside this model binding; compiled AMPL property functions are supported numerically"
            )
    assert_units_consistent(template)
    variables = tuple(
        template.component_data_objects(pyo.Var, active=True, descend_into=True)
    )
    if any(not var.is_continuous() for var in variables):
        raise ValueError("The IDAES bridge currently accepts continuous variables only")
    known_variables = {id(v) for v in variables}
    bound_ids = [
        id(component(template, v.component)) for v in adapter.inputs + adapter.controls
    ]
    if len(bound_ids) != len(set(bound_ids)):
        raise ValueError("Input/control bindings cannot alias the same component")
    for role, items in (
        ("input", adapter.inputs),
        ("control", adapter.controls),
        ("output", adapter.outputs),
    ):
        for binding in items:
            item = component(template, binding.component)
            if unit(item) != binding.unit:
                raise ValueError(
                    f"Native unit mismatch for {binding.component}: {unit(item)} != {binding.unit}; convert explicitly in the model"
                )
            if role == "input":
                if item.ctype is pyo.Param:
                    if (
                        not item.parent_component().mutable
                        or item.domain
                        not in (
                            pyo.Reals,
                            pyo.Any,
                        )
                        or item.parent_component()._validate is not None
                    ):
                        raise ValueError(
                            "Parameter inputs must be mutable with domain Reals or Any and no validation callback; encode physical limits as constraints"
                        )
                elif item.ctype is not pyo.Var:
                    raise ValueError("Input bindings require a Var or mutable Param")
            if role == "control" and item.ctype is not pyo.Var:
                raise ValueError("Actuators must bind scalar Vars")
            if role == "output" and item.ctype not in (
                pyo.Var,
                pyo.Param,
                pyo.Expression,
            ):
                raise ValueError(
                    "Outputs must bind numeric Vars, Params or Expressions"
                )
            if item.ctype is pyo.Var and id(item) not in known_variables:
                raise ValueError("Bound variables must belong to active model blocks")
    constraints = tuple(
        template.component_data_objects(pyo.Constraint, active=True, descend_into=True)
    )
    requirements = {
        id(component(template, r.component)): r for r in adapter.requirements
    }
    if len(requirements) != len(adapter.requirements):
        raise ValueError("Service requirements cannot alias the same constraint")
    if any(key not in {id(c) for c in constraints} for key in requirements):
        raise ValueError("Requirements must bind active scalar constraints")
    rows = []
    for con in constraints:
        req = requirements.get(id(con))
        if req and (con.equality or (con.lower is not None and con.upper is not None)):
            raise ValueError("Service requirements must be one-sided inequalities")
        if req and req.unit != unit(con.body):
            raise ValueError("Requirement unit must match its constraint body")
        sides = (
            ("eq",)
            if con.equality
            else tuple(
                side for side in ("lower", "upper") if getattr(con, side) is not None
            )
        )
        for side in sides:
            rows.append(
                Row(
                    req.name if req else f"constraint:{uid(con)}:{side}",
                    uid(con),
                    side,
                    unit(con.body),
                    "requirement"
                    if req
                    else "equation"
                    if con.equality
                    else "operating_limit",
                    req.tolerance if req else adapter.tolerance,
                )
            )
    for var in variables:
        for side in ("lower", "upper"):
            if getattr(var, side) is not None:
                rows.append(
                    Row(
                        f"bound:{uid(var)}:{side}",
                        uid(var),
                        side,
                        unit(var),
                        "operating_limit",
                        0,
                        True,
                    )
                )
    objectives = tuple(
        template.component_data_objects(pyo.Objective, active=True, descend_into=True)
    )
    # Include outputs, objectives and mutable coefficients in ownership checks.
    from pyomo.core.expr.visitor import identify_mutable_parameters, identify_variables

    known_parameters = {
        id(p)
        for p in template.component_data_objects(
            pyo.Param, active=True, descend_into=True
        )
    }
    # A scalar Param with a default can be referenced before its lazy _data[None]
    # entry exists (used by IDAES property reference temperatures).
    known_parameters.update(
        id(p)
        for p in template.component_objects(pyo.Param, active=True)
        if not p.is_indexed()
    )
    expressions = [r.expression(template) for r in rows]
    expressions.extend(
        component(template, v.component) for v in adapter.inputs + adapter.outputs
    )
    expressions.extend(o.expr for o in objectives)
    for expr in expressions:
        if any(
            id(v) not in known_variables
            for v in identify_variables(expr, include_fixed=True)
        ):
            raise ValueError(
                "Expressions cannot reference variables outside active model blocks"
            )
        if any(
            id(p) not in known_parameters and not isinstance(p, _PyomoUnit)
            for p in identify_mutable_parameters(expr)
        ):
            raise ValueError(
                "Expressions cannot reference parameters outside active model blocks"
            )
    bound = {
        id(component(template, v.component)) for v in adapter.inputs + adapter.controls
    }
    states = tuple(
        (f"state:{uid(v)}", uid(v), unit(v)) for v in variables if id(v) not in bound
    )
    if len(objectives) > 1:
        raise ValueError("The model must have at most one active objective")
    for step in adapter.initialization:
        block = component(template, step.component)
        if block.ctype is not pyo.Block or not block.active:
            raise ValueError("Initialization steps must bind active scalar blocks")
        api = "initialize" if step.method == "block" else "default_initializer"
        if not callable(getattr(block, api, None)):
            raise ValueError(f"{step.component} has no callable {api} API")
    return template, tuple(rows), states


def model_manifest(model):
    pyo, _ = backend()
    from pyomo.core.base.componentuid import ComponentUID

    return {
        "variables": [
            {
                "component": uid(v),
                "unit": unit(v),
                "domain": str(v.domain),
                "fixed": v.fixed,
                "value": None if v.value is None else number(v),
                "lower": str(v.lower),
                "upper": str(v.upper),
            }
            for v in model.component_data_objects(
                pyo.Var, active=True, descend_into=True
            )
        ],
        # Immutable indexed Params yield native Python values, not ParamData.
        # Record their index/value pairs under the owning component's CUID.
        "parameters": [
            {
                "component": uid(p),
                "values": [
                    {"index": index, "value": pyo.value(value)}
                    for index, value in (p.items() if p.is_indexed() else ((None, p),))
                ],
                "unit": unit(p),
                "mutable": p.mutable,
            }
            for p in model.component_objects(pyo.Param, active=True, descend_into=True)
        ],
        "suffixes": [
            {
                "component": uid(s),
                "direction": int(s.direction),
                "values": [
                    {
                        "component": None if key is None else uid(key),
                        "value": {"component_uid": str(value)}
                        if isinstance(value, ComponentUID)
                        else value,
                    }
                    for key, value in s.items()
                ],
            }
            for s in model.component_objects(pyo.Suffix, active=True, descend_into=True)
        ],
        "constraints": [
            {"component": uid(c), "expression": str(c.expr)}
            for c in model.component_data_objects(
                pyo.Constraint, active=True, descend_into=True
            )
        ],
        "expressions": [
            {"component": uid(e), "expression": str(e.expr)}
            for e in model.component_data_objects(
                pyo.Expression, active=True, descend_into=True
            )
        ],
        "objectives": [
            {"component": uid(o), "expression": str(o.expr), "sense": str(o.sense)}
            for o in model.component_data_objects(
                pyo.Objective, active=True, descend_into=True
            )
        ],
        "external_functions": [
            {
                "component": uid(f),
                "library": str(f._library),
                "function": str(f._function),
            }
            for f in model.component_objects(
                pyo.ExternalFunction, active=True, descend_into=True
            )
        ],
    }


def build_contract(adapter):
    _, versions = backend()
    manifest = model_manifest(adapter._template)
    quantities = [
        QuantitySpec(v.name, v.unit, "declared_process_quantity", role)
        for role, bindings in (
            ("input", adapter.inputs),
            ("control", adapter.controls),
            ("output", adapter.outputs),
        )
        for v in bindings
    ]
    quantities.extend(
        QuantitySpec(name, units, "algebraic_state", "state")
        for name, _, units in adapter._states
    )
    quantities.extend(
        QuantitySpec(r.residual, r.unit, "model_residual", "residual")
        for r in adapter._rows
    )
    return ProcessContract(
        "idaes_process/v1",
        {
            "name": adapter.name,
            "source_fingerprint": digest(manifest),
            "source": manifest,
            "versions": versions,
            "inputs": [v.to_dict() for v in adapter.inputs],
            "controls": [v.to_dict() for v in adapter.controls],
            "outputs": [v.to_dict() for v in adapter.outputs],
            "requirements": [r.to_dict() for r in adapter.requirements],
            "provenance": adapter.provenance,
        },
        {},
        {"kind": "unbound"},
        {
            "response": "external_numerical_dispatch",
            "stage": "operation",
            "states": "Unbound variables are algebraic states, not independent actuator commands; uniqueness is not established.",
            "solver": adapter.solver,
            "solver_options": adapter.solver_options,
            "initialization": [s.to_dict() for s in adapter.initialization],
            "diagnostics": adapter.diagnostics,
        },
        tuple(quantities),
        tuple(
            ConstraintSpec(
                r.name,
                r.residual,
                r.kind,
                "eq" if r.side == "eq" else "le",
                str(r.expression(adapter._template)),
                r.tolerance,
            )
            for r in adapter._rows
        ),
        {
            "arithmetic": "floating_point_pyomo_residual_checks",
            "absolute_tolerance": adapter.tolerance,
            "variable_bound_tolerance": 0,
            "optimality": "not certified",
            "infeasibility": "solver failure is unresolved",
            "scope": "Numerical feasibility at declared points; no continuous-domain or independent nonlinear certificate.",
        },
    )
