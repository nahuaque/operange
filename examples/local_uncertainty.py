"""Correlated local boiler responses with fixed and reoptimized fuel commands.

Run with operange[idaes]. HiGHS suffices for this affine operating model.
The covariance is illustrative, not a fitted confidence region.
"""

import argparse
import json
from pathlib import Path

import pyomo.environ as pyo
from idaes.core import FlowsheetBlock
from idaes.models.unit_models.skeleton_model import SkeletonUnitModel

from operange import Coordinate, ParameterSpace
from operange.contract_types import plain
from operange.integrations.idaes import IdaesAdapter, IdaesVariable, IdaesRequirement


def operating_model():
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.boiler = SkeletonUnitModel(dynamic=False)
    b = m.fs.boiler
    b.efficiency = pyo.Var(initialize=0.8, bounds=(0.1, 1))
    b.efficiency.fix()
    b.loss = pyo.Param(initialize=1, mutable=True, units=pyo.units.MW)
    b.fuel = pyo.Var(initialize=15, bounds=(0, 20), units=pyo.units.MW)
    b.fuel.fix()
    b.steam = pyo.Var(initialize=11, bounds=(0, 20), units=pyo.units.MW)
    b.balance = pyo.Constraint(expr=b.steam == b.efficiency * b.fuel - b.loss)
    m.fs.demand = pyo.Param(initialize=8, mutable=True, units=pyo.units.MW)
    m.fs.service = pyo.Constraint(expr=b.steam >= m.fs.demand)
    m.fs.objective = pyo.Objective(expr=b.fuel)
    return IdaesAdapter(
        model=m,
        name="Local correlated boiler uncertainty",
        input_space=ParameterSpace(
            (Coordinate("demand", "MW", "Steam commitment", nominal=8),)
        ),
        inputs=(IdaesVariable("demand", "fs.demand", "MW"),),
        controls=(IdaesVariable("fuel", "fs.boiler.fuel", "MW"),),
        outputs=(IdaesVariable("steam", "fs.boiler.steam", "MW"),),
        requirements=(IdaesRequirement("service", "fs.service", "MW"),),
        tolerance=1e-8,
    )


def run_example():
    adapter = operating_model()
    request = dict(
        parameters=(
            IdaesVariable("efficiency", "fs.boiler.efficiency", "1"),
            IdaesVariable("loss", "fs.boiler.loss", "MW"),
        ),
        covariance=((0.01, 0.04), (0.04, 0.25)),
        outputs=("steam", "fuel"),
        margins=("service", "bound:fs.boiler.fuel:upper"),
    )
    fixed = adapter.propagate_uncertainty(**request, fixed_controls={"fuel": 15})
    optimized = adapter.propagate_uncertainty(
        **request, reoptimize=True, steps={"efficiency": 1e-4, "loss": 1e-3}
    )
    for result in (fixed, optimized):
        if result.execution != "completed":
            raise RuntimeError(result.diagnostics)
    return fixed, optimized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-output", type=Path)
    parser.add_argument("--reoptimized-output", type=Path)
    args = parser.parse_args()
    fixed, optimized = run_example()
    print(
        json.dumps({r.mode: plain(r.statistics) for r in (fixed, optimized)}, indent=2)
    )
    for path, result in (
        (args.fixed_output, fixed),
        (args.reoptimized_output, optimized),
    ):
        if path:
            path.write_text(result.to_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
