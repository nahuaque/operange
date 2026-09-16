"""Finite operating study using IDAES heat-exchanger and liquid BTX properties.

Run ``uv run --extra idaes python examples/idaes_heat_exchanger.py`` with Ipopt
on PATH. Add ``--output study.json`` to save the complete portable audit.
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
    hx.overall_heat_transfer_coefficient.fix(100)  # W/(m2 K); varies with fouling
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
                    "heat_transfer",
                    "kg/K/s**3",
                    "Effective coefficient including fouling",
                    nominal=100,
                    scale=40,
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
                "heat_transfer",
                "fs.hx.overall_heat_transfer_coefficient[0]",
                "kg/K/s**3",
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
    normal = {"hot_inlet": 365, "process_flow": 1, "heat_transfer": 100, "demand": 4.5}
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
                    {"heat_transfer": 60},
                    "Reduced heat transfer after fouling",
                ),
                ("peak_demand", {"demand": 6}, "Demand beyond the attempted dispatch"),
            )
        ),
    )


def study(adapter=None):
    adapter = bind_flowsheet(build_flowsheet()) if adapter is None else adapter
    return adapter.as_claim(scenarios(adapter.input_space)).audit_result()


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
    args = parser.parse_args()
    with LoggingIntercept(sys.stderr, "pyomo"), redirect_stdout(sys.stderr):
        audit = study()
    print(json.dumps(summary(audit), indent=2, allow_nan=False))
    if args.output:
        args.output.write_text(audit.to_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
