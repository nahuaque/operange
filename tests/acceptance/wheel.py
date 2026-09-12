"""Run with -I outside the checkout, in a venv containing only the process wheel.

Copy steam_header.py, linear_dispatch.py, failure_distance.py, engineering_changes.py,
frozen_controllers.py, storage_replay.py, startup.py, pinch.py,
uncertainty_compositions.py and prototype_result_v1.json here.
There are no pytest, repository, optional solver, or test-environment imports.
"""

from importlib import metadata, util
from dataclasses import replace
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
    compositions = runpy.run_path(str(directory / "uncertainty_compositions.py"))
    model, combined, unrestricted = compositions["operating_modes_example"]()
    require(
        combined.capabilities.linear_optimization, "native composed support missing"
    )
    restored = process.domain_from_manifest(combined.to_manifest())
    require(restored.ref == combined.ref, "composition declaration changed")
    passing = model.as_claim(restored).audit_result()
    require(passing.payload.verdict == "pass", "operating modes should pass")
    failure = model.as_claim(unrestricted).audit_result()
    require(failure.payload.verdict == "fail", "unrestricted loads should fail")
    replay = model.as_claim(unrestricted).evaluate_result(
        failure.payload.witness.realizations[0]
    )
    require(
        any(c.assessment == "violated" for c in replay.payload.constraint_checks),
        "composed witness must replay its requirement violation",
    )
    for result in (passing, failure):
        require(
            process.result_from_json(result.to_json(compact=True)) == result,
            "composed audit did not round trip",
        )

    consumer = runpy.run_path(str(directory / "steam_header.py"))
    exported = consumer["run_example"]()
    results = {
        n: process.result_from_json(json.dumps(v))
        for n, v in exported.items()
        if n != "restored_result_id"
    }
    for result in results.values():
        compact = result.to_json(compact=True)
        require(
            process.result_from_json(compact).to_dict() == result.to_dict(),
            "compact transport changed a v1 result or its identity",
        )
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
    metric = process.NormalizedL2(loads.space)
    require(
        metric.measure({"dryer": 12, "evaporator": 8}) > 1.4,
        "Euclidean metric missing from wheel",
    )
    require(
        model.as_claim(loads, distance=metric).boundary_result().execution
        == "unsupported",
        "Euclidean search must require its explicit optional backend",
    )
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

    dispatch = runpy.run_path(str(directory / "linear_dispatch.py"))
    dispatch_results = {
        name: process.result_from_json(json.dumps(data))
        for name, data in dispatch["run_example"]().items()
    }
    dispatch_model, dispatch_cases = dispatch["example"]()
    synthesis = dispatch_model.as_claim(dispatch_cases).synthesize_controller(
        objective=process.LinearObjective("fuel")
    )
    require(
        synthesis.execution == "unsupported",
        "synthesis must report missing optional CVXPY",
    )
    require(
        not synthesis.verified and synthesis.controller is None,
        "missing backend cannot verify a controller",
    )
    joint = dispatch_model.as_claim(dispatch_cases).evaluate_result(
        {"dryer": 12, "evaporator": 8},
        relief={
            "changes": [
                {"constraint": "shared_fuel", "maximum": 2, "scale": 1, "unit": "MW"},
                {
                    "constraint": "no_excess_steam",
                    "maximum": 1,
                    "scale": 1,
                    "unit": "MW",
                },
            ],
            "objective": "linear",
        },
    )
    joint_relief = next(e.details for e in joint.evidence if e.evidence_id == "relief")
    require(
        joint_relief["resolution"] == "minimum_verified"
        and abs(joint_relief["upper"] - 1) < 1e-6,
        "native linear joint relief failed",
    )
    for name, verdict in (
        ("fixed", "fail"),
        ("adjustable", "fail"),
        ("enlarged_fuel_supply", "pass"),
    ):
        result = dispatch_results[name]
        require(result.payload.verdict == verdict, f"linear {name} verdict changed")
        require(
            result.payload.coverage.method == "complete_finite", "linear scope changed"
        )
        require(
            process.result_from_json(result.to_json(compact=True)).to_dict()
            == result.to_dict(),
            "compact linear results changed",
        )
    require(
        dispatch_results["enlarged_fuel_supply"].contract.domain
        == dispatch_results["adjustable"].contract.domain,
        "fuel capacity change altered the scenario domain",
    )
    dispatch_model, dispatch_domain = dispatch["example"]()
    preferred = replace(dispatch_model, objective=process.LinearObjective("fuel"))
    optimized = preferred.as_claim(dispatch_domain).evaluate_result(
        {"dryer": 10, "evaporator": 6}
    )
    require(
        optimized.payload.feasibility == "feasible", "objective lost feasible dispatch"
    )
    require(
        optimized.payload.objective.optimality == "verified",
        "minimum fuel needs a checked gap",
    )
    require(
        isclose(optimized.payload.objective.attained_value, 23, abs_tol=1e-7),
        "minimum fuel value changed",
    )
    require(
        process.result_from_json(optimized.to_json(compact=True)).result_id
        == optimized.result_id,
        "objective result failed portable replay",
    )
    require(
        process.LinearProcessAdapter(**preferred.to_dict()).base_contract.ref
        == preferred.base_contract.ref,
        "objective declaration did not round trip",
    )
    failure = dispatch_results["adjustable"].payload.witness
    require(failure is not None, "linear failure witness missing")
    replay = dispatch_model.as_claim(dispatch_domain).evaluate_result(
        failure.realizations[0]
    )
    require(
        replay.payload.feasibility == "infeasible", "linear witness does not replay"
    )
    require(
        any(
            e.subject == "recourse_infeasibility" and e.outcome == "verified"
            for e in replay.evidence
        ),
        "linear infeasibility certificate missing",
    )

    hull = process.ConvexHullSet(
        dispatch_model.input_space,
        tuple(s for s in dispatch_domain.scenarios if s.name != "combined"),
    )
    restored_hull = process.domain_from_manifest(hull.to_manifest())
    require(restored_hull.ref == hull.ref, "hull declaration did not round trip")
    hull_audit = dispatch_model.as_claim(restored_hull).audit_result()
    require(hull_audit.payload.verdict == "pass", "continuous hull should pass")
    require(
        hull_audit.payload.coverage.method == "analytical_domain",
        "hull coverage must include every convex combination",
    )
    require(
        dispatch_model.as_claim(restored_hull)
        .evaluate_result({"dryer": 11, "evaporator": 7})
        .payload.feasibility
        == "feasible",
        "hull interior membership or dispatch failed",
    )
    require(
        process.result_from_json(hull_audit.to_json(compact=True)) == hull_audit,
        "continuous hull result did not round trip",
    )
    dispatch_box = process.BoxSet(
        (
            process.Parameter("dryer", "MW", 10, 10, 12, 2, "Consumer"),
            process.Parameter("evaporator", "MW", 6, 6, 8, 2, "Consumer"),
        )
    )
    require(
        dispatch_model.as_claim(dispatch_box).audit_result().payload.verdict == "fail",
        "continuous box should retain simultaneous peak failure",
    )
    require(
        dispatch_model.as_claim(dispatch_box).audit_result(max_vertices=3).execution
        == "unsupported",
        "vertex limit must reject incomplete enumeration",
    )

    for domain in (dispatch_domain, dispatch_box):
        shared_result = dispatch_model.as_claim(domain).audit_result(
            relief={
                "changes": [
                    {
                        "constraint": "shared_fuel",
                        "maximum": 2,
                        "scale": 1,
                        "unit": "MW",
                    }
                ],
                "objective": "linear",
            }
        )
        shared = next(
            e.to_dict()["details"]
            for e in shared_result.evidence
            if e.evidence_id == "relief"
        )
        require(
            shared_result.payload.verdict == "fail",
            "shared relief changed the original verdict",
        )
        require(
            shared["resolution"] == "minimum_verified",
            "shared fuel relief minimum missing",
        )
        require(isclose(shared["upper"], 1, abs_tol=1e-7), "wrong shared fuel change")
        repaired = process.result_from_json(json.dumps(shared["candidate"]["reaudit"]))
        require(
            repaired.payload.verdict == "pass", "shared plan failed full-domain replay"
        )
        require(
            repaired.contract.domain == shared_result.contract.domain,
            "relief changed domain coverage",
        )
        require(
            process.result_from_json(shared_result.to_json(compact=True))
            == shared_result,
            "shared plan did not round trip",
        )

    distance_example = runpy.run_path(str(directory / "failure_distance.py"))
    distance_results = {
        name: process.result_from_json(json.dumps(data))
        for name, data in distance_example["run_example"]().items()
    }
    for name, distance in (
        ("boundary", 0.75),
        ("breaking", 0.7525),
        ("restricted_boundary", 1.0),
    ):
        result = distance_results[name]
        search = result.payload.search
        require(search.resolution == "minimum_verified", f"{name} distance unresolved")
        require(
            isclose(search.lower.value, distance, abs_tol=search.tolerance),
            f"{name} lower distance changed",
        )
        require(
            isclose(search.upper.value, distance, abs_tol=search.tolerance),
            f"{name} upper distance changed",
        )
        require(
            process.result_from_json(result.to_json(compact=True)).to_dict()
            == result.to_dict(),
            "distance result round trip changed",
        )
    require(
        distance_results["boundary"].payload.witness is None,
        "boundary promoted to failure",
    )
    require(
        distance_results["breaking"].payload.verdict == "fail",
        "positive violation missing",
    )
    require(
        distance_results["restricted_breaking"].payload.search.resolution
        == "unreachable",
        "restricted target should be unreachable",
    )
    distance_model, distance_domain = distance_example["example"]()
    replay = distance_model.as_claim(distance_domain).evaluate_result(
        distance_results["breaking"].payload.witness.realizations[0]
    )
    require(
        any(c.assessment == "violated" for c in replay.payload.constraint_checks),
        "distance witness does not replay",
    )

    changes_example = runpy.run_path(str(directory / "engineering_changes.py"))
    comparison = changes_example["run_example"]()
    require(comparison.baseline.payload.verdict == "fail", "baseline must fail")
    require(
        [c.transition for c in comparison.candidates]
        == ["restored", "still_failing", "not_comparable", "not_comparable"],
        "change comparison confuses repairs and revised commitments",
    )
    require(
        comparison.candidates[0].delta.changed_sections == ("design",),
        "larger boiler must be an equipment change",
    )
    require(comparison.baseline.payload.witness is not None, "lost baseline witness")
    for compact in (False, True):
        restored = process.ChangeComparison.from_json(
            comparison.to_json(compact=compact)
        )
        require(
            restored.to_dict() == comparison.to_dict(), "comparison round trip changed"
        )
    require(
        comparison.candidates[0].audit.payload.coverage.method == "complete_finite",
        "comparison broadened audit coverage",
    )

    controllers = runpy.run_path(str(directory / "frozen_controllers.py"))
    controller_results = controllers["run_example"]()
    envelope = controller_results["continuous_envelope"]
    envelope_audit = process.result_from_json(json.dumps(envelope["audit"]))
    envelope_replay = process.result_from_json(json.dumps(envelope["replayed_audit"]))
    envelope_distance = process.result_from_json(json.dumps(envelope["breaking"]))
    envelope_diagnosis = process.result_from_json(json.dumps(envelope["diagnosis"]))
    envelope_comparison = process.ChangeComparison.from_json(
        json.dumps(envelope["comparison"])
    )
    require(
        envelope_audit.payload.verdict == "fail",
        "continuous controller failure missing",
    )
    require(
        envelope_replay.payload.verdict == "pass"
        and envelope_replay.payload.coverage.method == "analytical_domain",
        "continuous controller restoration missing",
    )
    require(
        envelope_comparison.candidates[0].transition == "restored",
        "envelope comparison lost restoration",
    )
    require(
        envelope_distance.payload.search.resolution == "minimum_verified",
        "controller distance minimum unresolved",
    )
    conflict = next(
        e.details for e in envelope_diagnosis.evidence if e.evidence_id == "conflict"
    )
    relief = next(
        e.details for e in envelope_diagnosis.evidence if e.evidence_id == "relief"
    )
    require(
        conflict["irreducible"]
        and set(conflict["constraint_refs"]) == {"meet_load", "shared_fuel"},
        "boiler conflict changed",
    )
    require(
        relief["resolution"] == "minimum_verified"
        and isclose(relief["upper"], 1, abs_tol=1e-7),
        "checked fuel relief changed",
    )
    envelope_frozen = process.FrozenController.from_json(
        json.dumps(envelope["frozen_controller"])
    )
    require(
        envelope_frozen.audit_result().result_id == envelope_replay.result_id,
        "continuous frozen audit changed on replay",
    )
    for result in (
        envelope_audit,
        envelope_replay,
        envelope_distance,
        envelope_diagnosis,
    ):
        require(
            process.result_from_json(result.to_json(compact=True)).to_dict()
            == result.to_dict(),
            "continuous result round trip changed",
        )
    controller_comparison = process.ChangeComparison.from_json(
        json.dumps(controller_results["comparison"])
    )
    frozen = process.FrozenController.from_json(
        json.dumps(controller_results["frozen_controller"])
    )
    require(
        controller_comparison.baseline.payload.verdict == "fail",
        "equal allocation must fail",
    )
    require(
        controller_comparison.baseline.payload.witness.kind == "fixed_policy_failure",
        "controller failure scope lost",
    )
    require(
        [c.transition for c in controller_comparison.candidates]
        == ["restored", "restored"],
        "controller comparison changed",
    )
    require(
        frozen.audit_result().payload.verdict == "pass",
        "frozen controller replay failed",
    )
    require(
        frozen.controller.controller_id
        == process.AffineController.from_json(
            frozen.controller.to_json()
        ).controller_id,
        "controller identity changed",
    )
    require(
        frozen.audit_result().result_id
        == process.result_from_json(
            json.dumps(controller_results["replayed_audit"])
        ).result_id,
        "saved controller did not reproduce its audit",
    )
    held_out = process.result_from_json(
        json.dumps(controller_results["held_out_audit"])
    )
    require(held_out.payload.verdict == "pass", "held-out controller cases should pass")
    require(
        held_out.contract.domain != frozen.contract.domain,
        "held-out cases inherited original domain",
    )
    require(
        held_out.contract.operating == frozen.contract.operating,
        "replay changed the controller or permissions",
    )
    require(
        len(held_out.payload.coverage.evaluated_support["scenarios"]) == 2,
        "held-out coverage broadened",
    )

    storage_replay = runpy.run_path(str(directory / "storage_replay.py"))
    storage_results = storage_replay["run_example"]()
    storage_frozen = process.FrozenController.from_json(
        json.dumps(storage_results["frozen_controller"])
    )
    storage_audit = process.result_from_json(
        json.dumps(storage_results["replayed_audit"])
    )
    require(
        storage_frozen.audit_result().result_id == storage_audit.result_id,
        "storage replay changed after loading",
    )
    require(
        storage_audit.payload.verdict == "pass", "original storage paths should pass"
    )
    comparison = process.ChangeComparison.from_json(
        json.dumps(storage_results["comparison"])
    )
    require(
        [c.transition for c in comparison.candidates] == ["restored", "restored"],
        "storage controller comparison changed",
    )
    held_out = process.result_from_json(json.dumps(storage_results["held_out_audit"]))
    require(
        held_out.payload.verdict == "fail", "held-out demand must fail the saved rule"
    )
    require(
        held_out.payload.witness.kind == "fixed_policy_failure",
        "storage failure scope changed",
    )
    require(
        held_out.contract.operating["controller"]
        == storage_frozen.contract.operating["controller"],
        "held-out storage replay retuned the controller",
    )
    traces = next(
        e.details["paths"] for e in held_out.evidence if e.evidence_id == "replay"
    )
    require(
        [p["scenario"]["name"] for p in traces]
        == ["Later demand", "Later recovery", "Demand beyond reserve"],
        "held-out path order changed",
    )
    require(
        all(p["steps"][0]["commands"]["preparation_charge"] == 1 for p in traces),
        "preparation accessed future information",
    )
    require(
        all(
            p["steps"][1]["state_before_mwh"] == p["steps"][0]["state_after_mwh"]
            for p in traces
        ),
        "storage state reset before event",
    )
    require(
        traces[-1]["steps"][1]["state_after_mwh"] == -0.25,
        "failed storage trajectory was clipped",
    )
    unknown = process.result_from_json(
        json.dumps(storage_results["unknown_signal_audit"])
    )
    require(
        unknown.payload.verdict == "inconclusive"
        and unknown.payload.coverage.method == "partial",
        "unknown storage signal received an undeclared fallback",
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
