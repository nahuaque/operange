"""An existing IDAES flowsheet, numerical studies, affine certificates, parmest.

Run ``uv run --extra idaes python examples/idaes_flowsheet.py``. Add
``--reconcile`` when the Ipopt executable is installed to run upstream parmest.
"""

import argparse
import json

import pyomo.environ as pyo
from idaes.core import FlowsheetBlock
from idaes.models.unit_models.skeleton_model import SkeletonUnitModel

from operange import Coordinate, FiniteSet, ParameterSpace, Scenario, verify_result
from operange.integrations.idaes import IdaesAdapter, IdaesRequirement, IdaesVariable


def build_flowsheet(efficiency=0.8):
    """A small continuous IDAES model with explicit fixed-efficiency unit laws."""
    model = pyo.ConcreteModel()
    model.fs = FlowsheetBlock(dynamic=False)
    model.fs.boiler = SkeletonUnitModel(["a", "b"], dynamic=False)
    for name, eta, steam_cap, fuel_cap in (
        ("a", efficiency, 12, 24),
        ("b", 0.5, 10, 20),
    ):
        boiler = model.fs.boiler[name]
        boiler.efficiency = pyo.Param(initialize=eta, mutable=True)
        boiler.fuel = pyo.Var(bounds=(0, fuel_cap), initialize=8, units=pyo.units.MW)
        boiler.steam = pyo.Var(bounds=(0, steam_cap), initialize=4, units=pyo.units.MW)
        boiler.balance = pyo.Constraint(
            expr=boiler.steam == boiler.efficiency * boiler.fuel
        )
    model.fs.load = pyo.Param(
        initialize=16, mutable=True, within=pyo.Reals, units=pyo.units.MW
    )
    model.fs.steam = pyo.Expression(expr=sum(b.steam for b in model.fs.boiler.values()))
    model.fs.fuel = pyo.Expression(expr=sum(b.fuel for b in model.fs.boiler.values()))
    model.fs.fuel_supply = pyo.Constraint(expr=model.fs.fuel <= 32 * pyo.units.MW)
    model.fs.service = pyo.Constraint(expr=model.fs.steam >= model.fs.load)
    model.fs.objective = pyo.Objective(expr=model.fs.fuel)
    return model


def bind_flowsheet(model, *, provenance=None, solver="highs"):
    return IdaesAdapter(
        model=model,
        name="IDAES two-boiler utility plant",
        input_space=ParameterSpace(
            (Coordinate("load", "MW", "Declared heat demand", nominal=16, scale=4),)
        ),
        inputs=(IdaesVariable("load", "fs.load", "MW"),),
        controls=(
            IdaesVariable("fuel_a", "fs.boiler[a].fuel", "MW"),
            IdaesVariable("fuel_b", "fs.boiler[b].fuel", "MW"),
        ),
        outputs=(
            IdaesVariable("steam", "fs.steam", "MW"),
            IdaesVariable("fuel", "fs.fuel", "MW"),
        ),
        requirements=(IdaesRequirement("service", "fs.service", "MW", tolerance=0),),
        tolerance=1e-8,
        solver=solver,
        provenance=provenance or {},
    )


def study(adapter):
    domain = FiniteSet(
        adapter.input_space,
        (
            Scenario("normal", {"load": 16}, "Normal load"),
            Scenario("peak", {"load": 20}, "Peak load"),
        ),
    )
    numerical = adapter.as_claim(domain).audit_result()
    exported = adapter.export_linear()
    certified = exported.as_claim(domain).audit_result()
    verification = verify_result(certified)
    return {
        "numerical_audit": numerical.payload.verdict,
        "numerical_scope": "finite scenarios, floating-point residual checks",
        "affine_audit": certified.payload.verdict,
        "affine_certificate": verification.status,
        "source_fingerprint": adapter.base_contract.model["source_fingerprint"],
        "fuel_at_loads": [
            {
                "load": next(
                    v.value for v in e.payload.values if v.quantity_ref == "load"
                ),
                "fuel": next(
                    v.value for v in e.payload.values if v.quantity_ref == "fuel"
                ),
            }
            for e in numerical.supporting_evaluations
            if e.payload.feasibility == "feasible"
        ],
    }


def reconcile_and_estimate():
    """Use current upstream Experiment/Estimator interfaces, without a new fitter.

    The reconciliation stage uses the same boiler model with known efficiency;
    the estimation stage uses a separate calibration campaign. Their statistical
    outputs are recorded as provenance, not turned into hard uncertainty bounds.
    """
    from pyomo.contrib.parmest.experiment import Experiment
    from pyomo.core.base.componentuid import ComponentUID
    from operange.integrations.parmest import ParmestCalibration

    if not pyo.SolverFactory("ipopt").available(exception_flag=False):
        raise RuntimeError(
            "The upstream parmest example requires an installed Ipopt executable"
        )

    class BoilerExperiment(Experiment):
        def __init__(self, fuel, steam, *, estimate=False):
            self.fuel, self.steam, self.estimate = fuel, steam, estimate

        def get_labeled_model(self):
            model = pyo.ConcreteModel()
            model.fs = FlowsheetBlock(dynamic=False)
            model.fs.boiler = SkeletonUnitModel(dynamic=False)
            boiler = model.fs.boiler
            boiler.fuel = pyo.Var(
                bounds=(0, 30), initialize=self.fuel, units=pyo.units.MW
            )
            boiler.steam = pyo.Var(
                bounds=(0, 30), initialize=self.steam, units=pyo.units.MW
            )
            boiler.efficiency = pyo.Var(bounds=(0.1, 1), initialize=0.8)
            boiler.balance = pyo.Constraint(
                expr=boiler.steam == boiler.efficiency * boiler.fuel
            )
            if self.estimate:
                boiler.fuel.fix(self.fuel)
            else:
                boiler.efficiency.fix(0.8)
            model.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            model.experiment_outputs[boiler.fuel] = self.fuel
            model.experiment_outputs[boiler.steam] = self.steam
            model.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            model.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            model.measurement_error[boiler.fuel] = 0.2
            model.measurement_error[boiler.steam] = 0.2
            if self.estimate:
                model.unknown_parameters[boiler.efficiency] = ComponentUID(
                    boiler.efficiency
                )
            return model

    options = {"tol": 1e-10, "bound_relax_factor": 0}
    reconciliation_data = [{"fuel": 10, "steam": 8.4}, {"fuel": 15, "steam": 11.6}]
    calibration_data = [{"fuel": 10, "steam": 7.4}, {"fuel": 15, "steam": 11.1}]
    reconciliation = ParmestCalibration(
        name="Boiler measurement reconciliation",
        experiments={
            f"run-{i}": BoilerExperiment(**row)
            for i, row in enumerate(reconciliation_data)
        },
        solver_options=options,
    ).fit()
    estimation = ParmestCalibration(
        name="Boiler efficiency calibration",
        experiments={
            f"run-{i}": BoilerExperiment(**row, estimate=True)
            for i, row in enumerate(calibration_data)
        },
        solver_options=options,
    ).fit()
    if any(r.execution != "completed" for r in (reconciliation, estimation)):
        raise RuntimeError(
            "Calibration unresolved; inspect solver configuration and experiment models"
        )
    efficiency = estimation.estimates[0].value
    return {
        "method": "pyomo.contrib.parmest.Estimator.theta_est",
        "reconciliation_measurements_MW": reconciliation_data,
        "calibration_measurements_MW": calibration_data,
        "reconciliation_objective": reconciliation.objective_value,
        "reconciled": [
            {name: e.values[name] for name in ("fs.boiler.fuel", "fs.boiler.steam")}
            for e in reconciliation.experiments
        ],
        "estimation_objective": estimation.objective_value,
        "reconciliation_result_ref": reconciliation.ref.to_dict(),
        "calibration_result_ref": estimation.ref.to_dict(),
        "efficiency": efficiency,
        "measurement_standard_deviation_MW": 0.2,
        "scope": "Numerical least-squares estimates; no deterministic error bounds or independent certificate",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    report = {"initial_study": study(bind_flowsheet(build_flowsheet()))}
    if args.reconcile:
        calibration = reconcile_and_estimate()
        report["parmest"] = calibration
        report["calibrated_study"] = study(
            bind_flowsheet(
                build_flowsheet(calibration["efficiency"]),
                provenance={"calibration": calibration},
            )
        )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
