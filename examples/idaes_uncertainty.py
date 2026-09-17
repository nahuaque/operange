"""Joint efficiency/heat-loss calibration and finite operating propagation.

Run with operange[idaes] and Ipopt on PATH. All measurements are synthetic.
The experiments are treated as exchangeable sampling units for this example.
"""

import argparse
import json
from pathlib import Path

import pyomo.environ as pyo
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.core.base.componentuid import ComponentUID
from idaes.core import FlowsheetBlock
from idaes.models.unit_models.skeleton_model import SkeletonUnitModel

from operange import Coordinate, ParameterSpace, FiniteSet, Scenario
from operange.integrations.idaes import IdaesAdapter, IdaesVariable, IdaesRequirement
from operange.integrations.parmest import ParmestCalibration


OPTIONS = {"tol": 1e-10, "bound_relax_factor": 0}
PARAMETER_MAP = {
    "fs.boiler.efficiency": "fs.boiler.efficiency",
    "fs.boiler.heat_loss": "fs.boiler.heat_loss",
}


def build_model():
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.boiler = SkeletonUnitModel(dynamic=False)
    b = m.fs.boiler
    b.efficiency = pyo.Var(initialize=0.8, bounds=(0.5, 0.95))
    b.efficiency.fix()
    b.heat_loss = pyo.Var(initialize=1, bounds=(0, 3), units=pyo.units.MW)
    b.heat_loss.fix()
    b.fuel = pyo.Var(initialize=13.5, bounds=(0, 25), units=pyo.units.MW)
    b.fuel.fix()
    b.steam = pyo.Var(initialize=9.8, bounds=(0, 25), units=pyo.units.MW)
    b.balance = pyo.Constraint(expr=b.steam == b.efficiency * b.fuel - b.heat_loss)
    m.fs.demand = pyo.Param(initialize=9, mutable=True, units=pyo.units.MW)
    m.fs.service = pyo.Constraint(expr=b.steam >= m.fs.demand)
    m.fs.objective = pyo.Objective(expr=b.fuel)
    return m


def campaign():
    # Paired fuel and steam observations; fuel is treated as exactly known.
    data = ((8, 5.45), (10, 6.92), (12, 8.68), (14, 10.14), (16, 11.87), (18, 13.35))
    experiments = {}
    for i, (fuel, steam) in enumerate(data):
        m = build_model()
        b = m.fs.boiler
        b.fuel.fix(fuel)
        m.fs.service.deactivate()
        m.fs.objective.deactivate()
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs[b.steam] = steam
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error[b.steam] = 0.1
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        for unknown in (b.efficiency, b.heat_loss):
            m.unknown_parameters[unknown] = ComponentUID(unknown)
        experiments[f"run-{i}"] = Experiment(m)
    return ParmestCalibration(
        name="Joint boiler efficiency and heat loss",
        experiments=experiments,
        solver_options=OPTIONS,
        provenance={"data": "Synthetic noisy boiler readings; fuel known exactly"},
    )


def operating_model():
    return IdaesAdapter(
        model=build_model(),
        name="Boiler with calibrated heat loss",
        input_space=ParameterSpace(
            (Coordinate("demand", "MW", "Steam commitment", nominal=9),)
        ),
        inputs=(IdaesVariable("demand", "fs.demand", "MW"),),
        controls=(IdaesVariable("fuel", "fs.boiler.fuel", "MW"),),
        outputs=(IdaesVariable("steam", "fs.boiler.steam", "MW"),),
        requirements=(IdaesRequirement("service", "fs.service", "MW"),),
        solver="ipopt",
        solver_options=OPTIONS,
        tolerance=1e-6,
    )


def _study(samples, seed):
    uncertainty = campaign().bootstrap(samples=samples, seed=seed)
    if uncertainty.execution != "completed":
        raise RuntimeError(uncertainty.diagnostics)
    adapter = operating_model()
    cases = FiniteSet(
        adapter.input_space,
        (
            Scenario("normal", {"demand": 9}, "Normal commitment"),
            Scenario("near_limit", {"demand": 9.8}, "Near nominal available steam"),
        ),
    )
    return uncertainty, adapter, cases


def run_example(samples=12, seed=7):
    uncertainty, adapter, cases = _study(samples, seed)
    claim = uncertainty.as_claim(
        adapter, cases, parameter_map=PARAMETER_MAP, fixed_controls={"fuel": 13.5}
    )
    audit = claim.audit_result()
    return uncertainty, audit


def run_shared_example(samples=12, seed=7):
    """Compare nominal commands with one command checked across all joint draws."""
    uncertainty, adapter, cases = _study(samples, seed)
    nominal_adapter = uncertainty.calibration.apply_to(
        adapter, parameter_map=PARAMETER_MAP
    )
    nominal = nominal_adapter.optimize_shared_controls(cases)
    if nominal.execution != "completed":
        raise RuntimeError(nominal.diagnostics)
    nominal_audit = uncertainty.as_claim(
        adapter, cases, parameter_map=PARAMETER_MAP, fixed_controls=nominal.commands
    ).audit_result()
    shared = uncertainty.optimize_shared_controls(
        adapter, cases, parameter_map=PARAMETER_MAP
    )
    if shared.execution != "completed":
        raise RuntimeError(shared.diagnostics)
    return uncertainty, nominal, nominal_audit, shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--uncertainty-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--optimize-shared", action="store_true")
    parser.add_argument("--shared-output", type=Path)
    args = parser.parse_args()
    if args.shared_output and not args.optimize_shared:
        parser.error("--shared-output requires --optimize-shared")
    comparison = {}
    if args.optimize_shared:
        uncertainty, nominal, nominal_audit, shared = run_shared_example(
            args.samples, args.seed
        )
        audit = shared.audit
        comparison = {
            "nominal_fuel_MW": nominal.commands["fuel"],
            "nominal_command_joint_verdict": nominal_audit.payload.verdict,
            "shared_fuel_MW": shared.commands["fuel"],
            "shared_objective_MW": shared.objective_value,
            "optimality": shared.scope["optimality"],
        }
        if args.shared_output:
            args.shared_output.write_text(shared.to_json(), encoding="utf-8")
    else:
        uncertainty, audit = run_example(args.samples, args.seed)
    from operange.contract_types import plain

    # The audit may store one evaluation once for duplicate parameter vectors;
    # coverage keeps every draw/case label and hence empirical multiplicity.
    evaluations = {e.result_id: e for e in audit.supporting_evaluations}
    scenarios = audit.payload.coverage.evaluated_support["scenarios"]
    print(
        json.dumps(
            {
                "uncertainty_ref": uncertainty.ref.to_dict(),
                "scope": dict(uncertainty.scope),
                "statistics": plain(uncertainty.statistics),
                "operating_verdict": audit.payload.verdict,
                "feasible_scenarios": sum(
                    evaluations[s["evaluation_ref"]["artifact_id"]].payload.feasibility
                    == "feasible"
                    for s in scenarios
                ),
                "scenarios": len(scenarios),
                **comparison,
            },
            indent=2,
        )
    )
    if args.uncertainty_output:
        args.uncertainty_output.write_text(uncertainty.to_json(), encoding="utf-8")
    if args.audit_output:
        args.audit_output.write_text(audit.to_json(compact=True), encoding="utf-8")


if __name__ == "__main__":
    main()
