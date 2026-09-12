"""Run outside the checkout with -I after installing the wheel's cvxpy extra.

Copy linear_dispatch.py and uncertainty_compositions.py alongside this consumer.
It needs no pytest or examples package and checks that the private backend
modules ship in the wheel.
"""

from importlib import util
from dataclasses import replace
from pathlib import Path
import json
import runpy
import sys

import operange as process


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    require(sys.flags.isolated == 1, "run acceptance with python -I")
    require(not Path("pyproject.toml").exists(), "run outside the checkout")
    require(
        Path(process.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),
        "use the installed wheel",
    )
    require(
        util.find_spec("examples") is None and util.find_spec("pytest") is None,
        "consumer environment must be isolated",
    )
    require("cvxpy" not in sys.modules, "package import must leave CVXPY lazy")
    envelope = process.BoxSet(
        (
            process.Parameter("a", "MW", 0, -1, 0.5, 1, "Consumer"),
            process.Parameter("b", "MW", 0, -1, 1, 1, "Consumer"),
        )
    )
    ellipsoid = process.EllipsoidSet(
        envelope.space, {"a": {"a": 1, "b": 0}, "b": {"a": 0, "b": 1}}
    )
    domain = process.Intersection((envelope, ellipsoid), backend="cvxpy")
    loaded = process.domain_from_manifest(domain.to_manifest())
    require("cvxpy" not in sys.modules, "manifest loading must leave CVXPY lazy")
    support = loaded.maximize_linear({"a": 1, "b": 1})
    require(
        support.lower is not None
        and support.upper is not None
        and support.upper - support.lower < 1e-7,
        "joint support failed",
    )
    require(1.36 < support.upper < 1.37, "unexpected intersection upper bound")
    distance_model = process.AffineProcessAdapter(
        "Consumer distance",
        domain.space,
        (
            process.AffineOutput(
                "total",
                "MW",
                "thermal_power",
                0,
                (
                    process.AffineTerm("a", 1, "MW/MW"),
                    process.AffineTerm("b", 1, "MW/MW"),
                ),
                "Sum",
            ),
        ),
        (process.AffineRequirement("total_cap", "total", 1.35),),
    )
    distance_claim = distance_model.as_claim(
        domain, distance=process.NormalizedL2(domain.space)
    )
    distance_result = distance_claim.boundary_result(backend="cvxpy")
    search = distance_result.payload.search
    require(
        search.resolution == "minimum_verified" and 0.986 < search.upper.value < 0.987,
        "Euclidean intersection distance failed",
    )
    require(
        process.result_from_json(distance_result.to_json(compact=True))
        == distance_result,
        "convex distance result changed on round trip",
    )
    auxiliary = process.BoxSet((process.Parameter("c", "MW", 0, -1, 1, 1, "Consumer"),))
    combined = process.Product((process.Union((loaded, loaded)), auxiliary))
    nested = combined.maximize_linear({"a": 1, "b": 1, "c": -2})
    require(
        nested.lower is not None
        and nested.upper is not None
        and 3.36 < nested.lower <= nested.upper < 3.37,
        "nested intersection support failed",
    )
    require(
        combined.membership(nested.point).status == "inside", "nested witness invalid"
    )

    compositions = runpy.run_path(
        str(Path(__file__).parent / "uncertainty_compositions.py")
    )
    for name, (model, joint) in compositions["convex_examples"]().items():
        result = model.as_claim(joint).audit_result()
        require(result.payload.verdict == "pass", f"convex workflow failed: {name}")
        require(
            process.result_from_json(result.to_json(compact=True)) == result,
            f"convex workflow did not round trip: {name}",
        )

    consumer = runpy.run_path(str(Path(__file__).parent / "linear_dispatch.py"))
    model, cases = consumer["example"]()
    hull = process.ConvexHullSet(
        model.input_space, tuple(s for s in cases.scenarios if s.name != "combined")
    )
    hull_audit = model.as_claim(hull).audit_result(backend="cvxpy")
    require(
        hull_audit.payload.verdict == "pass", "prepared continuous hull audit failed"
    )
    require(
        hull_audit.payload.coverage.method == "analytical_domain",
        "continuous coverage missing",
    )
    require(
        process.result_from_json(hull_audit.to_json(compact=True)) == hull_audit,
        "prepared hull evidence did not round trip",
    )
    relief_model = replace(
        model,
        controls=(replace(model.controls[0], upper=14), model.controls[1]),
        outputs=model.outputs
        + (
            process.AffineOutput(
                "a_output",
                "MW",
                "thermal_power",
                0,
                (process.AffineTerm("boiler_a", 1, "MW/MW"),),
                "Boiler A output",
            ),
        ),
        operating_limits=tuple(replace(r, tolerance=0) for r in model.operating_limits)
        + (process.AffineRequirement("a_capacity", "a_output", 12, tolerance=0),),
        requirements=tuple(replace(r, tolerance=0) for r in model.requirements),
    )
    joint_result = relief_model.as_claim(cases).evaluate_result(
        {"dryer": 12, "evaporator": 8},
        backend="cvxpy",
        relief={
            "changes": [
                {"constraint": "a_capacity", "maximum": 2, "scale": 1, "unit": "MW"},
                {"constraint": "shared_fuel", "maximum": 2, "scale": 1, "unit": "MW"},
            ],
            "objective": "quadratic",
        },
    )
    joint_relief = next(
        e.details for e in joint_result.evidence if e.evidence_id == "relief"
    )
    require(
        joint_relief["resolution"] == "minimum_verified"
        and joint_relief["lower"] <= 0.64 <= joint_relief["upper"],
        "quadratic joint relief failed",
    )
    require(
        process.result_from_json(joint_result.to_json(compact=True)) == joint_result,
        "joint relief result changed on round trip",
    )
    shared_result = relief_model.as_claim(cases).audit_result(
        backend="cvxpy",
        relief=joint_result.request["relief"],
    )
    shared = next(
        e.to_dict()["details"]
        for e in shared_result.evidence
        if e.evidence_id == "relief"
    )
    require(
        shared["resolution"] == "minimum_verified",
        "shared quadratic relief did not close its gap",
    )
    require(abs(shared["upper"] - 0.64) <= 1e-8, "wrong shared quadratic objective")
    repaired = process.result_from_json(json.dumps(shared["candidate"]["reaudit"]))
    require(repaired.payload.verdict == "pass", "shared quadratic relief failed replay")
    require(
        process.result_from_json(shared_result.to_json(compact=True)) == shared_result,
        "shared quadratic result did not round trip",
    )
    tracking = replace(
        model,
        objective=process.ControlTrackingObjective(
            (
                process.ControlTarget("boiler_a", 8, 1, "MW"),
                process.ControlTarget("boiler_b", 8, 1, "MW"),
            )
        ),
    )
    tracked = tracking.as_claim(cases).evaluate_result(
        {"dryer": 12, "evaporator": 6}, backend="cvxpy"
    )
    require(
        tracked.payload.feasibility == "feasible"
        and tracked.payload.objective.optimality == "verified",
        "tracking optimum was not verified",
    )
    require(
        abs(tracked.payload.objective.attained_value - 2) < 1e-6,
        "wrong command tracking objective",
    )
    require(
        process.result_from_json(tracked.to_json(compact=True)).result_id
        == tracked.result_id,
        "tracking result did not round trip",
    )
    claim = model.as_claim(cases)
    audit = claim.audit_result(backend="cvxpy")
    require(
        audit.payload.verdict == "fail", "dispatch failed to find the known conflict"
    )
    result = claim.evaluate_result(
        {"dryer": 12, "evaporator": 8},
        backend="cvxpy",
        diagnose=True,
        relief={"constraint": "shared_fuel", "maximum": 2},
    )
    relief = next(e.details for e in result.evidence if e.evidence_id == "relief")
    require(relief["resolution"] == "minimum_verified", "relief did not close bounds")
    for checked in (audit, result):
        require(
            process.result_from_json(checked.to_json(compact=True)).result_id
            == checked.result_id,
            "portable result changed",
        )
    print("Optional CVXPY installed-wheel acceptance passed")


if __name__ == "__main__":
    main()
