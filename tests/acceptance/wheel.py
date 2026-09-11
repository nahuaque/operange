"""Run with -I outside the checkout, in a venv containing only the process wheel.

Copy steam_header.py, startup.py, pinch.py and
prototype_result_v1.json here.
There are no pytest, repository, optional solver, or test-environment imports.
"""

from importlib import metadata, util
import json
from math import isclose
from pathlib import Path
import runpy
import sys

import operange as process


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    require(sys.flags.isolated == 1, "acceptance must run with python -I")
    require(
        Path(process.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),
        "process must come from the installed wheel",
    )
    require(not Path("pyproject.toml").exists(), "run outside the checkout")
    for name in (
        "updatesupport",
        "updatesupport_finance",
        "updatesupport_process",
        "residopt",
        "cvxpy",
        "pytest",
        "examples",
    ):
        require(util.find_spec(name) is None, f"unexpected extra dependency: {name}")
    require(
        "operange.heat_recovery" not in sys.modules,
        "shared API must not eagerly import the reference models",
    )
    require(
        not hasattr(process, "RepairAction"),
        "model-specific repairs belong to reference",
    )
    for name in ("demo", "storage_demo"):
        require(
            util.find_spec(f"operange.{name}") is None,
            f"consumer report code leaked into wheel: {name}",
        )
    dependency_names = {
        r.split(">=")[0].strip()
        for r in metadata.requires("operange")
        if "extra ==" not in r
    }
    require(
        dependency_names == {"numpy", "scipy"}, "unexpected runtime dependency metadata"
    )

    directory = Path(__file__).resolve().parent
    consumer = runpy.run_path(str(directory / "steam_header.py"))
    exported = consumer["run_example"]()
    results = {
        n: process.result_from_json(json.dumps(v))
        for n, v in exported.items()
        if n != "restored_result_id"
    }
    require(
        results["evaluation"].payload.feasibility == "feasible",
        "nominal evaluation failed",
    )
    values = {v.quantity_ref: v.value for v in results["evaluation"].payload.values}
    require(
        values["steam"] == 16 and values["fuel"] == 20,
        "shared utility balances changed",
    )
    require(
        results["sensitivity"].payload.derivative.values == ((1.0, 1.0), (1.25, 1.25)),
        "physical sensitivities changed",
    )
    require(
        results["audit"].payload.verdict == "fail",
        "baseline shared capacity should fail",
    )
    witness = results["audit"].payload.witness
    model, loads = consumer["example"]()
    require(
        witness is not None
        and loads.membership(witness.realizations[0]).status == "inside",
        "failure needs a declared-domain member",
    )
    checked = model.as_claim(loads).evaluate_result(witness.realizations[0])
    require(
        any(c.assessment == "violated" for c in checked.payload.constraint_checks),
        "witness must violate an engineering requirement",
    )
    require(
        results["repaired"].payload.verdict == "pass", "enlarged equipment should pass"
    )
    require(results["budgeted"].payload.verdict == "pass", "budgeted claim should pass")
    require(
        results["unsupported_search"].execution == "unsupported",
        "unsupported search was promoted",
    )
    require(
        exported["restored_result_id"] == results["audit"].result_id,
        "export identity changed",
    )

    startup = runpy.run_path(str(directory / "startup.py"))
    startup_results = {
        name: process.result_from_json(json.dumps(data))
        for name, data in startup["run_example"]().items()
    }
    for name, verdict, peak in (
        ("simultaneous", "fail", 700),
        ("staggered", "pass", 480),
    ):
        audit = startup_results[name]
        require(audit.payload.verdict == verdict, f"startup {name} verdict changed")
        require(
            audit.payload.coverage.method == "complete_finite",
            "startup coverage overstated",
        )
        require(len(audit.supporting_evaluations) == 65, "startup scenarios missing")
        maximum = max(
            next(v.value for v in e.payload.values if v.quantity_ref == "peak_load")
            for e in audit.supporting_evaluations
        )
        require(isclose(maximum, peak), f"startup {name} peak changed")
    require(
        startup_results["simultaneous"].contract.domain
        == startup_results["staggered"].contract.domain,
        "staggering changed startup uncertainty",
    )

    pinch = runpy.run_path(str(directory / "pinch.py"))
    pinch_results = {
        name: process.result_from_json(json.dumps(data))
        for name, data in pinch["run_example"]().items()
    }
    target_audit = pinch_results["target_audit"]
    require(
        target_audit.payload.verdict == "fail", "combined thermal stress should fail"
    )
    require(
        target_audit.payload.coverage.method == "complete_finite",
        "heat target coverage changed",
    )
    # Portable bundles order evaluations by identity, not by scenario name.
    hot_by_ref = {
        e.ref.artifact_id: next(
            v.value for v in e.payload.values if v.quantity_ref == "minimum_hot_utility"
        )
        for e in target_audit.supporting_evaluations
    }
    require(
        {
            s["name"]: hot_by_ref[s["evaluation_ref"]["artifact_id"]]
            for s in target_audit.payload.coverage.evaluated_support["scenarios"]
        }
        == {"nominal": 10, "less_hot_flow": 40, "more_cold_flow": 40, "combined": 100},
        "heat cascade targets changed",
    )
    enlarged = pinch_results["enlarged_utility_target_audit"]
    require(enlarged.payload.verdict == "pass", "enlarged utility target should pass")
    require(
        enlarged.contract.domain == target_audit.contract.domain,
        "heat-target uncertainty changed",
    )
    require(
        enlarged.contract.model["installed_network_feasibility"] == "not_assessed",
        "target promoted to network certificate",
    )
    pinch_model, pinch_domain = pinch["example"]()
    replay = pinch_model.as_claim(pinch_domain).evaluate_result(
        target_audit.payload.witness.realizations[0]
    )
    require(
        any(c.assessment == "violated" for c in replay.payload.constraint_checks),
        "heat target witness must replay",
    )
    nominal_cascade = next(
        e for e in pinch_results["nominal"].evidence if e.evidence_id == "cascade"
    )
    require(
        nominal_cascade.details["pinch_points"][0]["hot_temperature_c"] == 180,
        "nominal pinch changed",
    )

    corpus = json.loads((directory / "prototype_result_v1.json").read_text())
    for name in ("evaluation", "sensitivity", "robustness"):
        restored = process.result_from_json(json.dumps(corpus[name]))
        require(
            restored.to_dict() == corpus[name],
            f"pre-refactor v1 bundle changed: {name}",
        )
    from operange.reference import FrozenHeatRecoveryContract

    frozen = FrozenHeatRecoveryContract.from_json(json.dumps(corpus["frozen_heat"]))
    require(
        json.loads(frozen.to_json()) == corpus["frozen_heat"],
        "frozen reference contract changed",
    )
    print(
        "Installed Operange wheel: public consumer, evidence, and compatibility checks passed"
    )


if __name__ == "__main__":
    main()
