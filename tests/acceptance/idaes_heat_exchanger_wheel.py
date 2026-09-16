"""Run outside the checkout with an installed [idaes] wheel and Ipopt on PATH."""

import json
from pathlib import Path
import runpy
import sys

import operange
from operange.integrations.idaes import IdaesInitialization


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    require(sys.flags.isolated == 1, "run with isolated Python")
    require(not Path("pyproject.toml").exists(), "run outside the checkout")
    require(
        Path(operange.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),
        "use the installed wheel",
    )
    IdaesInitialization("fs.hx", options={"solver": "ipopt"})
    require(
        "idaes" not in sys.modules and "pyomo" not in sys.modules,
        "imports must stay lazy",
    )
    import pyomo.environ as pyo

    require(
        pyo.SolverFactory("ipopt").available(False), "Ipopt is required for this check"
    )
    example = runpy.run_path(str(Path(__file__).with_name("idaes_heat_exchanger.py")))
    audit = example["study"]()
    summary = example["summary"](audit)
    require(audit.payload.verdict == "inconclusive", "unknown cases must be retained")
    require(audit.payload.witness is None, "numerical failures are not counterexamples")
    require(
        sum(row["feasibility"] == "feasible" for row in summary["scenarios"]) == 4,
        "expected four checked feasible cases",
    )
    normal = next(row for row in summary["scenarios"] if row["scenario"] == "normal")
    require(abs(normal["outputs"]["duty"] - 4.5) < 1e-7, "incorrect heat duty")
    require(
        operange.result_from_json(audit.to_json(compact=True)).result_id
        == audit.result_id,
        "portable diagnostic audit changed on loading",
    )
    print(json.dumps(summary, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
