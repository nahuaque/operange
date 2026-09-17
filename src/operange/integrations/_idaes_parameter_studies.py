"""Compose whole parameter samples with operating cases, independent of estimator."""

from dataclasses import dataclass

from ..domains import Coordinate, ParameterSpace, FiniteSet, Scenario
from ..parameters import ParameterPoint, ParameterSamples
from ._idaes_parameters import parameter_model


@dataclass(frozen=True)
class JointParameterStudy:
    model: object
    domain: FiniteSet
    inputs: tuple
    scenario_index: tuple


def prepare_joint_samples(
    adapter,
    point,
    samples,
    operating_cases,
    *,
    parameter_map,
    coordinate_provenance="Latent model parameter; joint sample",
):
    from .idaes import IdaesAdapter, IdaesVariable

    if not isinstance(adapter, IdaesAdapter) or type(operating_cases) is not FiniteSet:
        raise ValueError(
            "Joint propagation requires an IdaesAdapter and a FiniteSet of operating cases"
        )
    if not isinstance(point, ParameterPoint) or not isinstance(
        samples, ParameterSamples
    ):
        raise ValueError("Supply a ParameterPoint and ParameterSamples")
    point.space.check_alignment(samples.space)
    adapter.input_space.check_alignment(operating_cases.space)
    model, updates = parameter_model(adapter, point, parameter_map)
    for sample in samples.samples:
        parameter_model(
            adapter, ParameterPoint(samples.space, sample.values), parameter_map
        )
    targets = {u.parameter: u.component for u in updates}
    coordinates = tuple(
        Coordinate(
            f"parameter:{p.name}",
            p.unit,
            coordinate_provenance,
            nominal=point.values[p.name],
        )
        for p in point.space.coordinates
    )
    reserved = {v.name for v in adapter.inputs + adapter.controls + adapter.outputs}
    if any(c.name in reserved for c in coordinates):
        raise ValueError(
            "Generated parameter coordinate names collide with model quantities"
        )
    bindings = tuple(
        IdaesVariable(c.name, targets[p.name], c.unit)
        for c, p in zip(coordinates, point.space.coordinates)
    )
    space = ParameterSpace(adapter.input_space.coordinates + coordinates)
    index, scenarios = [], []
    for i, sample in enumerate(samples.samples):
        values = {f"parameter:{name}": value for name, value in sample.values.items()}
        for j, case in enumerate(operating_cases.scenarios):
            name = f"draw-{i:06d}/case-{j:06d}"
            scenarios.append(
                Scenario(
                    name,
                    {**dict(case.values), **values},
                    f"{sample.name} at operating case {case.name}",
                )
            )
            index.append(
                {"scenario": name, "sample": sample.name, "operating_case": case.name}
            )
    return JointParameterStudy(
        model,
        FiniteSet(space, tuple(scenarios)),
        adapter.inputs + bindings,
        tuple(index),
    )
