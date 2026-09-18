"""Run with -I outside the checkout, with idaes_flowsheet.py alongside this file."""

import json
from pathlib import Path
import runpy
import sys

import operange
from operange.integrations.idaes import IdaesLinearExport


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
    require(
        "pyomo" not in sys.modules and "idaes" not in sys.modules,
        "binding imports must remain lazy",
    )
    example = runpy.run_path(str(Path(__file__).with_name("idaes_flowsheet.py")))
    adapter = example["bind_flowsheet"](example["build_flowsheet"]())
    report = example["study"](adapter)
    require(report["numerical_audit"] == "pass", "finite numerical study failed")
    require(
        report["affine_audit"] == "pass" and report["affine_certificate"] == "verified",
        "affine certificate failed",
    )
    require(
        [row["fuel"] for row in report["fuel_at_loads"]] == [23, 31],
        "incorrect dispatch",
    )
    exported = adapter.export_linear()
    restored = IdaesLinearExport.from_json(exported.to_json())
    require(restored.to_dict() == exported.to_dict(), "export changed on loading")
    require(
        {c.name for c in restored.adapter.controls} == {"fuel_a", "fuel_b"},
        "export promotes states to actuators",
    )
    require(len(restored.adapter.states) == 2, "export lost algebraic states")
    domain = operange.FiniteSet(
        adapter.input_space,
        (operange.Scenario("nominal", {"load": 16}, "Fixed command replay"),),
    )
    operating = operange.OperatingSpec.fixed(
        adapter.controls, {"fuel_a": 10, "fuel_b": 16}
    )
    require(
        adapter.as_claim(domain, operating=operating)
        .evaluate_result({"load": 16})
        .payload.feasibility
        == "feasible",
        "IDAES operating specification failed",
    )
    checked = restored.as_claim(domain, operating=operating).audit_result()
    require(
        checked.payload.verdict == "pass" and operange.verify_result(checked).verified,
        "fixed actuators did not retain checked state completion",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
