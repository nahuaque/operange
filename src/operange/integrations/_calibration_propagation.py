"""Map whole calibration draws into existing finite IDAES operating claims."""

from collections.abc import Mapping
from dataclasses import replace

from ..contract_types import plain
from ..domains import Coordinate, ParameterSpace, FiniteSet, Scenario
from ..recourse import DecisionRule, RecoursePolicy
from ..primitives import finite
from ._parmest_backend import parameter_model
from .idaes import IdaesAdapter, IdaesVariable


def as_claim(
    uncertainty,
    adapter,
    operating_cases,
    *,
    parameter_map,
    fixed_controls,
    fully_observed,
):
    uncertainty._require_complete()
    if not isinstance(adapter, IdaesAdapter) or type(operating_cases) is not FiniteSet:
        raise ValueError(
            "Joint propagation requires an IdaesAdapter and a FiniteSet of operating cases"
        )
    if type(fully_observed) is not bool:
        raise ValueError("fully_observed must be a bool")
    if fully_observed and fixed_controls is not None:
        raise ValueError("Choose fixed_controls or fully_observed dispatch, not both")
    if not fully_observed:
        if fixed_controls is None and not adapter.controls:
            fixed_controls = {}
        if not isinstance(fixed_controls, Mapping) or set(fixed_controls) != {
            c.name for c in adapter.controls
        }:
            raise ValueError(
                "Supply fixed_controls for every actuator, or explicitly opt in to fully_observed=True"
            )
        fixed_controls = {
            name: finite(value, f"fixed control {name}")
            for name, value in fixed_controls.items()
        }

    propagated, domain = prepare_joint(
        uncertainty,
        adapter,
        operating_cases,
        parameter_map=parameter_map,
        dispatch="ideal_fully_observed_parameters"
        if fully_observed
        else "fixed_actuator_commands",
    )
    policy = (
        None
        if fully_observed
        else RecoursePolicy(
            "fixed",
            tuple(
                DecisionRule(
                    v.name, v.unit, "operation", fixed_value=fixed_controls[v.name]
                )
                for v in adapter.controls
            ),
        )
    )
    return propagated.as_claim(domain, recourse=policy)


def prepare_joint(uncertainty, adapter, operating_cases, *, parameter_map, dispatch):
    """Share vector mapping, bounds validation and provenance across studies."""
    uncertainty._require_complete()
    if not isinstance(adapter, IdaesAdapter) or type(operating_cases) is not FiniteSet:
        raise ValueError(
            "Joint propagation requires an IdaesAdapter and a FiniteSet of operating cases"
        )
    adapter.input_space.check_alignment(operating_cases.space)
    model, updates = parameter_model(
        adapter, uncertainty.calibration.estimates, parameter_map
    )
    # Reject an incompatible mapping for any draw before running an audit.
    # Keep vectors intact, including bounds depending on another parameter.
    for draw in uncertainty.draws:
        parameter_model(adapter, draw.estimates, parameter_map)
    target_paths = {u["estimate"]: u["target"] for u in updates}
    coordinates = tuple(
        Coordinate(
            f"parameter:{p.component}",
            p.unit,
            "Latent calibrated parameter; joint bootstrap draw",
            nominal=p.value,
        )
        for p in uncertainty.calibration.estimates
    )
    reserved = {v.name for v in adapter.inputs + adapter.controls + adapter.outputs}
    if any(c.name in reserved for c in coordinates):
        raise ValueError(
            "Generated calibration coordinate names collide with model quantities"
        )
    bindings = tuple(
        IdaesVariable(c.name, target_paths[p.component], c.unit)
        for c, p in zip(coordinates, uncertainty.calibration.estimates)
    )
    space = ParameterSpace(adapter.input_space.coordinates + coordinates)
    case_index, scenarios = [], []
    for i, draw in enumerate(uncertainty.draws):
        values = {f"parameter:{p.component}": p.value for p in draw.estimates}
        for j, case in enumerate(operating_cases.scenarios):
            name = f"draw-{i:06d}/case-{j:06d}"
            scenarios.append(
                Scenario(
                    name,
                    {**dict(case.values), **values},
                    f"{draw.name} at operating case {case.name}",
                )
            )
            case_index.append(
                {"scenario": name, "draw": draw.name, "operating_case": case.name}
            )
    provenance = {
        **plain(adapter.provenance),
        "joint_calibration": {
            "uncertainty_ref": uncertainty.ref.to_dict(),
            "calibration_ref": uncertainty.calibration.ref.to_dict(),
            "parent_contract_ref": adapter.base_contract.ref.to_dict(),
            "operating_cases_ref": operating_cases.ref.to_dict(),
            "parameter_map": dict(parameter_map),
            "scope": dict(uncertainty.scope),
            "combination": "Cartesian stress test of whole parameter vectors and declared operating cases; no case probabilities or statistical independence asserted.",
            "dispatch": dispatch,
            "scenario_index": case_index,
        },
    }
    propagated = replace(
        adapter,
        model=model,
        input_space=space,
        inputs=adapter.inputs + bindings,
        provenance=provenance,
    )
    return propagated, FiniteSet(space, tuple(scenarios))
