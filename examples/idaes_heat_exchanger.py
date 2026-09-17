"""Finite operating study using IDAES heat-exchanger and liquid BTX properties.

Run ``uv run --extra idaes python examples/idaes_heat_exchanger.py`` with Ipopt
on PATH. Add ``--output study.json`` to save the complete portable audit.
Use ``--calibrate --calibration-output fit.json`` to fit clean U and re-audit.
All equipment and operating data are synthetic. The single-liquid property
assumption is explicit; this example does not establish phase stability.
"""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import pyomo.environ as pyo
from pyomo.common.log import LoggingIntercept
from idaes.core import FlowsheetBlock
from idaes.core.util.scaling import set_scaling_factor
from idaes.models.properties.activity_coeff_models.BTX_activity_coeff_VLE import (
    BTXParameterBlock,
)
from idaes.models.unit_models import HeatExchanger

from operange import Coordinate, FiniteSet, ParameterSpace, Scenario
from operange.integrations.idaes import (
    IdaesAdapter,
    IdaesInitialization,
    IdaesRequirement,
    IdaesVariable,
)


SOLVER_OPTIONS = {"tol": 1e-10, "bound_relax_factor": 0, "max_iter": 300}


def build_flowsheet():
    """Use upstream balances, temperature-dependent enthalpy, and counterflow LMTD."""
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = BTXParameterBlock(valid_phase="Liq", activity_coeff_model="Ideal")
    m.fs.hx = HeatExchanger(
        hot_side={"property_package": m.fs.properties},
        cold_side={"property_package": m.fs.properties},
    )
    hx = m.fs.hx
    for port, flow, temperature in (
        (hx.hot_side_inlet, 5, 365),
        (hx.cold_side_inlet, 1, 300),
    ):
        port.flow_mol.fix(flow)
        port.temperature.fix(temperature)
        port.pressure.fix(200000)
        port.mole_frac_comp[0, "benzene"].fix(0.5)
        port.mole_frac_comp[0, "toluene"].fix(0.5)
    hx.hot_side_inlet.flow_mol[0].setlb(2)
    hx.hot_side_inlet.flow_mol[0].setub(8)
    hx.area.fix(1)  # m2; declared equipment size
    m.fs.clean_heat_transfer = pyo.Var(
        initialize=100,
        bounds=(20, 200),
        units=pyo.units.get_units(hx.overall_heat_transfer_coefficient[0]),
    )
    m.fs.clean_heat_transfer.fix()  # W/(m2 K); a calibratable equipment parameter
    m.fs.fouling = pyo.Param(initialize=1, mutable=True, within=pyo.Reals)
    hx.coefficient_law = pyo.Constraint(
        expr=hx.overall_heat_transfer_coefficient[0]
        == m.fs.clean_heat_transfer * m.fs.fouling
    )
    # This study is restricted to positive counterflow temperature approaches.
    # These are physical operating limits, retained in every residual check.
    hx.delta_temperature_in[0].setlb(0.1)
    hx.delta_temperature_out[0].setlb(0.1)
    m.fs.demand = pyo.Param(
        initialize=4.5, mutable=True, within=pyo.Reals, units=pyo.units.kW
    )
    m.fs.duty = pyo.Expression(
        expr=pyo.units.convert(hx.heat_duty[0], to_units=pyo.units.kW)
    )
    m.fs.heat_service = pyo.Constraint(expr=m.fs.duty >= m.fs.demand)
    m.fs.temperature_service = pyo.Constraint(
        expr=hx.cold_side_outlet.temperature[0] >= 325 * pyo.units.K
    )
    m.fs.objective = pyo.Objective(expr=hx.hot_side_inlet.flow_mol[0])

    # IDAES scaling suffixes guide the solver without transforming the source
    # equations or changing the physical units used by Operange's row checks.
    scales = {
        "K": 1e-2,
        "Pa": 1e-5,
        "J/mol": 1e-4,
        "W": 1e-4,
        "kg*m**2/s**3": 1e-4,
        "kg/K/s**3": 1e-2,
        "kW": 1,
    }
    for kind in (pyo.Var, pyo.Constraint):
        for item in m.component_data_objects(kind, active=True):
            expression = item.body if kind is pyo.Constraint else item
            set_scaling_factor(
                item, scales.get(str(pyo.units.get_units(expression)), 1)
            )
    return m


def bind_flowsheet(model):
    return IdaesAdapter(
        model=model,
        name="IDAES counterflow liquid heat exchanger",
        input_space=ParameterSpace(
            (
                Coordinate(
                    "hot_inlet", "K", "Hot inlet temperature", nominal=365, scale=10
                ),
                Coordinate(
                    "process_flow",
                    "mol/s",
                    "Cold process throughput",
                    nominal=1,
                    scale=0.5,
                ),
                Coordinate(
                    "fouling",
                    "1",
                    "Fraction of clean heat-transfer coefficient",
                    nominal=1,
                    scale=0.4,
                ),
                Coordinate(
                    "demand", "kW", "Required recovered duty", nominal=4.5, scale=1
                ),
            )
        ),
        inputs=(
            IdaesVariable("hot_inlet", "fs.hx.hot_side_inlet.temperature[0]", "K"),
            IdaesVariable("process_flow", "fs.hx.cold_side_inlet.flow_mol[0]", "mol/s"),
            IdaesVariable(
                "fouling",
                "fs.fouling",
                "1",
            ),
            IdaesVariable("demand", "fs.demand", "kW"),
        ),
        controls=(
            IdaesVariable("utility_flow", "fs.hx.hot_side_inlet.flow_mol[0]", "mol/s"),
        ),
        outputs=(
            IdaesVariable("duty", "fs.duty", "kW"),
            IdaesVariable(
                "process_outlet", "fs.hx.cold_side_outlet.temperature[0]", "K"
            ),
            IdaesVariable(
                "utility_outlet", "fs.hx.hot_side_outlet.temperature[0]", "K"
            ),
        ),
        requirements=(
            IdaesRequirement("heat_service", "fs.heat_service", "kW", 1e-7),
            IdaesRequirement(
                "temperature_service", "fs.temperature_service", "K", 1e-6
            ),
        ),
        solver="ipopt",
        solver_options={**SOLVER_OPTIONS, "nlp_scaling_method": "user-scaling"},
        tolerance=1e-5,
        initialization=(
            IdaesInitialization(
                "fs.hx",
                options={"solver": "ipopt", "optarg": SOLVER_OPTIONS, "outlvl": 40},
            ),
        ),
        diagnostics=True,
        provenance={
            "data": "synthetic engineering reference case",
            "property_package": "IDAES BTXParameterBlock, ideal liquid, 50/50 molar benzene/toluene",
            "phase_scope": "Single-liquid assumption; phase stability is not checked.",
            "initialization": "Upstream HeatExchanger.initialize; each scenario starts from the same source snapshot.",
        },
    )


def scenarios(space):
    normal = {"hot_inlet": 365, "process_flow": 1, "fouling": 1, "demand": 4.5}
    return FiniteSet(
        space,
        tuple(
            Scenario(name, {**normal, **changes}, description)
            for name, changes, description in (
                ("normal", {}, "Clean exchanger at normal demand"),
                ("warmer_feed", {"hot_inlet": 375}, "More recoverable inlet heat"),
                ("lower_demand", {"demand": 4}, "Reduced duty commitment"),
                ("higher_throughput", {"process_flow": 1.2}, "Higher process flow"),
                (
                    "fouled",
                    {"fouling": 0.6},
                    "Reduced heat transfer after fouling",
                ),
                ("peak_demand", {"demand": 6}, "Demand beyond the attempted dispatch"),
            )
        ),
    )


def study(adapter=None):
    adapter = bind_flowsheet(build_flowsheet()) if adapter is None else adapter
    return adapter.as_claim(scenarios(adapter.input_space)).audit_result()


def calibration():
    """Fit clean U with known area and fixed, measured experimental conditions.

    Synthetic outlet readings are rounded, perturbed simulations with U=85.
    Measurement sigma is 0.1 K. Neither area nor fouling is estimated.
    """
    from pyomo.contrib.parmest.experiment import Experiment
    from pyomo.core.base.componentuid import ComponentUID
    from operange.integrations.parmest import ParmestCalibration

    # hot flow [mol/s], hot inlet [K], cold flow [mol/s], hot/cold outlets [K]
    readings = (
        (3, 355, 1, 348.04, 322.52),
        (5, 365, 1.2, 359.79, 323.73),
        (7, 375, 0.8, 371.21, 337.02),
        (4, 370, 1.1, 363.24, 327.08),
    )

    class HeatExchangerExperiment(Experiment):
        def __init__(self, reading):
            self.reading = reading

        def get_labeled_model(self):
            hot_flow, hot_inlet, cold_flow, hot_outlet, cold_outlet = self.reading
            model = build_flowsheet()
            hx = model.fs.hx
            hx.hot_side_inlet.flow_mol.fix(hot_flow)
            hx.hot_side_inlet.temperature.fix(hot_inlet)
            hx.cold_side_inlet.flow_mol.fix(cold_flow)
            # Calibration explains observations at recorded conditions. Service
            # commitments and the dispatch objective belong to the later audit.
            model.fs.heat_service.deactivate()
            model.fs.temperature_service.deactivate()
            model.fs.objective.deactivate()
            hx.initialize(solver="ipopt", optarg=SOLVER_OPTIONS, outlvl=40)
            model.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            model.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            for var, measured in (
                (hx.hot_side_outlet.temperature[0], hot_outlet),
                (hx.cold_side_outlet.temperature[0], cold_outlet),
            ):
                model.experiment_outputs[var] = measured
                model.measurement_error[var] = 0.1
            model.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            model.unknown_parameters[model.fs.clean_heat_transfer] = ComponentUID(
                model.fs.clean_heat_transfer
            )
            return model

    return ParmestCalibration(
        name="Heat-exchanger clean coefficient with known area",
        experiments={
            f"run-{i + 1}": HeatExchangerExperiment(row)
            for i, row in enumerate(readings)
        },
        solver_options=SOLVER_OPTIONS,
        provenance={
            "data": "Synthetic perturbed outlet-temperature readings; generating U=85 W/(m2 K).",
            "known_equipment": {"area_m2": 1, "fouling_factor": 1},
            "measurement_standard_deviation_K": 0.1,
            "assumptions": "Inlet flows, inlet temperatures, composition, area and fouling are treated as known.",
        },
    ).fit()


def calibrated_study():
    """Preserve the baseline; apply the checked fit and re-audit the same cases."""
    baseline = bind_flowsheet(build_flowsheet())
    before = study(baseline)
    fit = calibration()
    if fit.execution != "completed":
        return fit, before, None
    calibrated = fit.apply_to(
        baseline,
        parameter_map={"fs.clean_heat_transfer": "fs.clean_heat_transfer"},
    )
    return fit, before, study(calibrated)


def summary(audit):
    evaluations = {e.result_id: e for e in audit.supporting_evaluations}
    rows = []
    for scenario in audit.payload.coverage.evaluated_support["scenarios"]:
        result = evaluations[scenario["evaluation_ref"]["artifact_id"]]
        diagnostic_reports = [
            e.details for e in result.evidence if e.subject == "model_diagnostics"
        ]
        final_diagnostics = diagnostic_reports[-1] if diagnostic_reports else {}
        rows.append(
            {
                "scenario": scenario["name"],
                "feasibility": result.payload.feasibility,
                "outputs": {
                    v.quantity_ref: v.value
                    for v in result.payload.values
                    if v.quantity_ref
                    in {"utility_flow", "duty", "process_outlet", "utility_outlet"}
                },
                "diagnostics": [d.to_dict() for d in result.diagnostics],
                "initialization": [
                    e.to_dict()["details"]
                    for e in result.evidence
                    if e.method == "idaes_initialization"
                ],
                "diagnostic_point": final_diagnostics.get("iterate_origin"),
                "service_violations_at_diagnostic_point": [
                    dict(row)
                    for row in final_diagnostics.get("violated_rows_at_iterate", ())
                    if row["kind"] == "requirement"
                ],
            }
        )
    return {
        "verdict": audit.payload.verdict,
        "scope": "Declared finite scenarios; numerical feasibility only. Unresolved scenarios are not proven infeasible.",
        "scenarios": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Save the complete portable audit JSON"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Fit clean U, then repeat the operating study",
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        help="Save portable calibration JSON (requires --calibrate)",
    )
    args = parser.parse_args()
    if args.calibration_output and not args.calibrate:
        parser.error("--calibration-output requires --calibrate")
    with LoggingIntercept(sys.stderr, "pyomo"), redirect_stdout(sys.stderr):
        if args.calibrate:
            fit, before, audit = calibrated_study()
        else:
            audit = study()
    if args.calibrate:
        report = {
            "calibration": {
                "execution": fit.execution,
                "result_ref": fit.ref.to_dict(),
                "estimates": [p.to_dict() for p in fit.estimates],
                "objective": fit.objective_value,
                "diagnostics": [d.to_dict() for d in fit.diagnostics],
            },
            "baseline": summary(before),
            "calibrated": None if audit is None else summary(audit),
        }
        if args.calibration_output:
            args.calibration_output.write_text(fit.to_json(), encoding="utf-8")
    else:
        report = summary(audit)
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.output and audit is not None:
        args.output.write_text(audit.to_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
