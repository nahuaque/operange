"""Run outside the checkout with -I after installing the wheel's cvxpy extra.

Copy linear_dispatch.py alongside this consumer. It needs no pytest or examples
package and checks that the new private backend modules ship in the wheel.
"""

from importlib import util
from pathlib import Path
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

    consumer = runpy.run_path(str(Path(__file__).parent / "linear_dispatch.py"))
    model, cases = consumer["example"]()
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
