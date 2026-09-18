"""Map whole calibration draws into existing finite IDAES operating claims."""

from collections.abc import Mapping
from dataclasses import replace

from ..contract_types import plain
from ..domains import FiniteSet
from ..operating import OperatingSpec, resolve_operating
from ..primitives import finite
from ._idaes_parameters import parameter_model
from ._idaes_parameter_studies import prepare_joint_samples
from .idaes import IdaesAdapter, IdaesVariable


def propagate_local(uncertainty, adapter, *, parameter_map, **options):
    from ._idaes_uncertainty import propagate_uncertainty

    uncertainty._require_complete()
    model, updates = parameter_model(
        adapter, uncertainty.calibration.parameter_point, parameter_map
    )
    targets = {u.parameter: u.component for u in updates}
    # Covariance follows the estimate tuple, never mapping insertion order.
    parameters = tuple(
        IdaesVariable(p.component, targets[p.component], p.unit)
        for p in uncertainty.calibration.estimates
    )
    provenance = {
        "uncertainty_ref": uncertainty.ref.to_dict(),
        "calibration_ref": uncertainty.calibration.ref.to_dict(),
        "parent_contract_ref": adapter.base_contract.ref.to_dict(),
        "parameter_map": dict(parameter_map),
        "linearization_point": "checked_calibration_estimate",
        "covariance": "full_empirical_bootstrap_sample_covariance",
        "samples": uncertainty.requested_samples,
    }
    return propagate_uncertainty(
        replace(adapter, model=model),
        parameters=parameters,
        covariance=uncertainty.statistics["covariance"],
        provenance=provenance,
        **options,
    )


def as_claim(
    uncertainty,
    adapter,
    operating_cases,
    *,
    parameter_map,
    fixed_controls,
    fully_observed,
    operating=None,
):
    uncertainty._require_complete()
    if not isinstance(adapter, IdaesAdapter) or type(operating_cases) is not FiniteSet:
        raise ValueError(
            "Joint propagation requires an IdaesAdapter and a FiniteSet of operating cases"
        )
    if type(fully_observed) is not bool:
        raise ValueError("fully_observed must be a bool")
    if operating is not None:
        if fixed_controls is not None or fully_observed:
            raise ValueError(
                "Supply operating or fixed_controls/fully_observed, not both"
            )
        observations = adapter.input_space.names + tuple(
            f"parameter:{p.component}" for p in uncertainty.calibration.estimates
        )
        spec = resolve_operating(adapter.controls, observations, operating=operating)
        spec.validate(adapter.controls, observations).point_commands(observations)
        propagated, domain = prepare_joint(
            uncertainty,
            adapter,
            operating_cases,
            parameter_map=parameter_map,
            dispatch="explicit_actuator_permissions",
        )
        return propagated.as_claim(domain, operating=spec)
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
    spec = (
        OperatingSpec.dispatch(propagated.controls, propagated.input_space.names)
        if fully_observed
        else OperatingSpec.fixed(adapter.controls, fixed_controls)
    )
    return propagated.as_claim(domain, operating=spec)


def prepare_joint(uncertainty, adapter, operating_cases, *, parameter_map, dispatch):
    """Share vector mapping, bounds validation and provenance across studies."""
    uncertainty._require_complete()
    if not isinstance(adapter, IdaesAdapter) or type(operating_cases) is not FiniteSet:
        raise ValueError(
            "Joint propagation requires an IdaesAdapter and a FiniteSet of operating cases"
        )
    prepared = prepare_joint_samples(
        adapter,
        uncertainty.calibration.parameter_point,
        uncertainty.parameter_samples,
        operating_cases,
        parameter_map=parameter_map,
        coordinate_provenance="Latent calibrated parameter; joint bootstrap draw",
    )
    case_index = [
        {
            "scenario": row["scenario"],
            "draw": row["sample"],
            "operating_case": row["operating_case"],
        }
        for row in prepared.scenario_index
    ]
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
        model=prepared.model,
        input_space=prepared.domain.space,
        inputs=prepared.inputs,
        provenance=provenance,
    )
    return propagated, prepared.domain
