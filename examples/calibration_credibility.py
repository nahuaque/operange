"""Local parameter information and held-out prediction for synthetic heat data.

Run with operange[idaes] and Ipopt on PATH. This deliberately small model uses
duty = heat_transfer * area * temperature_difference. It demonstrates how
estimating both heat-transfer coefficient and area confounds their product.
"""

import argparse
import json
from pathlib import Path

import pyomo.environ as pyo
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.core.base.componentuid import ComponentUID

from operange.contract_types import plain
from operange.integrations.parmest import ParmestCalibration


OPTIONS = {"tol": 1e-10, "bound_relax_factor": 0}


def experiment(delta_temperature, observed_duty, *, estimate_area=False):
    m = pyo.ConcreteModel()
    m.heat_transfer = pyo.Var(
        initialize=85,
        bounds=(10, 200),
        units=pyo.units.W / pyo.units.m**2 / pyo.units.K,
    )
    m.area = pyo.Var(initialize=1, bounds=(0.5, 2), units=pyo.units.m**2)
    m.area.fix()
    m.delta_temperature = pyo.Var(initialize=delta_temperature, units=pyo.units.K)
    m.delta_temperature.fix()
    m.duty = pyo.Var(initialize=85 * delta_temperature, units=pyo.units.W)
    m.balance = pyo.Constraint(
        expr=m.duty == m.heat_transfer * m.area * m.delta_temperature
    )
    m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.experiment_outputs[m.duty] = observed_duty
    m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.measurement_error[m.duty] = 10
    m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.unknown_parameters[m.heat_transfer] = ComponentUID(m.heat_transfer)
    if estimate_area:
        m.unknown_parameters[m.area] = ComponentUID(m.area)
    return Experiment(m)


def campaign(*, estimate_area=False):
    data = ((20, 1702), (40, 3398), (60, 5101))
    return ParmestCalibration(
        name="Synthetic heat-transfer measurements",
        experiments={
            f"training-{i}": experiment(t, q, estimate_area=estimate_area)
            for i, (t, q) in enumerate(data)
        },
        solver_options=OPTIONS,
        provenance={
            "data": "Synthetic duty readings; temperature differences are treated as exact"
        },
    )


def run_example():
    study = campaign()
    fitted = study.fit()
    diagnosed = study.diagnose(fitted, parameter_scales={"heat_transfer": 10})
    confounded = campaign(estimate_area=True).diagnose(
        parameter_scales={"heat_transfer": 10, "area": 0.1}
    )
    validation = study.validate(
        fitted,
        experiments={
            "held-normal": experiment(80, 6800),
            "held-biased": experiment(90, 6900),
        },
    )
    for report in (diagnosed, confounded, validation):
        if report.execution != "completed":
            raise RuntimeError(report.diagnostics)
    return diagnosed, confounded, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-output", type=Path)
    parser.add_argument("--validation-output", type=Path)
    args = parser.parse_args()
    diagnosed, confounded, validation = run_example()
    print(
        json.dumps(
            {
                "known_area_information": plain(diagnosed.analysis),
                "unknown_area_information": plain(confounded.analysis),
                "held_out_metrics": plain(validation.metrics),
                "held_out_predictions": [
                    {
                        "name": p.name,
                        "duty_W": p.result.observations[0].fitted,
                        "residual_W": p.result.observations[0].residual,
                        "diagnostics": [d.to_dict() for d in p.diagnostics],
                    }
                    for p in validation.predictions
                ],
                "scope": dict(diagnosed.scope),
            },
            indent=2,
        )
    )
    if args.diagnostics_output:
        args.diagnostics_output.write_text(confounded.to_json(), encoding="utf-8")
    if args.validation_output:
        args.validation_output.write_text(validation.to_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
