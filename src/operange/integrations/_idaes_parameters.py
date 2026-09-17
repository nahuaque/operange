"""Apply neutral parameter data to Pyomo snapshots; no calibration dependency."""

from collections.abc import Mapping
from dataclasses import dataclass

from ..parameters import ParameterBinding, ParameterPoint
from ._idaes_model import backend, component, number, uid, unit


@dataclass(frozen=True)
class ParameterAssignment:
    parameter: str
    component: str
    unit: str
    previous: float
    value: float


def parameter_bindings(point, parameter_map):
    if not isinstance(point, ParameterPoint):
        raise ValueError("Supply a ParameterPoint")
    if not isinstance(parameter_map, Mapping) or not parameter_map:
        raise ValueError("Supply an explicit parameter-to-component parameter_map")
    if set(parameter_map) != set(point.space.names):
        raise ValueError("parameter_map must cover exactly the declared parameters")
    return tuple(
        ParameterBinding(name, path, point.space.coordinate(name).unit)
        for name, path in parameter_map.items()
    )


def assign_parameters(source, point, bindings, *, reserved=(), require_fixed=True):
    """Clone, assign the whole vector, then check all dependent bounds."""
    bindings = tuple(bindings)
    if len(bindings) != len(point.space.names) or {
        b.parameter for b in bindings
    } != set(point.space.names):
        raise ValueError("Bindings must cover exactly the declared parameters")
    pyo, _ = backend()
    model = source.clone()
    forbidden = {id(component(model, path)) for path in reserved}
    active = {
        id(obj)
        for kind in (pyo.Var, pyo.Param)
        for obj in model.component_data_objects(kind, active=True)
    }
    updates, targets = [], set()
    for binding in bindings:
        target = component(model, binding.component)
        if id(target) not in active:
            raise ValueError("Parameter targets must belong to active model blocks")
        if id(target) in forbidden:
            raise ValueError(
                "Parameter targets cannot be scenario inputs or actuator controls"
            )
        if id(target) in targets:
            raise ValueError("Parameter targets cannot alias each other")
        targets.add(id(target))
        if (
            unit(target) != binding.unit
            or binding.unit != point.space.coordinate(binding.parameter).unit
        ):
            raise ValueError("Parameter target native units must match the coordinate")
        if target.ctype is pyo.Var:
            if require_fixed and not target.fixed:
                raise ValueError("Parameter target Vars must be fixed model parameters")
        elif target.ctype is not pyo.Param or not target.parent_component().mutable:
            raise ValueError("Parameter targets must be fixed Vars or mutable Params")
        elif (
            target.parent_component().domain not in (pyo.Reals, pyo.Any)
            or target.parent_component()._validate is not None
        ):
            raise ValueError(
                "Target Params must use Reals/Any without validation callbacks"
            )
        previous, value = number(target), point.values[binding.parameter]
        target.set_value(value)
        if target.ctype is pyo.Var:
            target.fix()
        updates.append(
            ParameterAssignment(
                binding.parameter, uid(target), binding.unit, previous, value
            )
        )
    for update in updates:
        target = component(model, update.component)
        if target.ctype is pyo.Var and (
            (target.lb is not None and update.value < number(target.lb))
            or (target.ub is not None and update.value > number(target.ub))
        ):
            raise ValueError("Parameter violates target bounds")
    return model, tuple(updates)


def parameter_model(adapter, point, parameter_map):
    from .idaes import IdaesAdapter

    if not isinstance(adapter, IdaesAdapter):
        raise ValueError("Apply parameters to an IdaesAdapter")
    return assign_parameters(
        adapter._template,
        point,
        parameter_bindings(point, parameter_map),
        reserved=tuple(b.component for b in adapter.inputs + adapter.controls),
    )
