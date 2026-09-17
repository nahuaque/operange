"""Run outside the checkout with an installed [idaes] wheel and Ipopt."""

from pathlib import Path
import runpy
import sys

import operange
from operange.integrations.parmest import CalibrationBootstrap
from operange.integrations.idaes import IdaesSharedControlResult


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
        not any(name in sys.modules for name in ("idaes", "pyomo", "pandas")),
        "imports must stay lazy",
    )
    example = runpy.run_path(str(Path(__file__).with_name("idaes_uncertainty.py")))
    uncertainty, audit = example["run_example"]()
    require(uncertainty.execution == "completed", "joint bootstrap did not complete")
    require(
        CalibrationBootstrap.from_json(uncertainty.to_json()) == uncertainty,
        "joint evidence changed on loading",
    )
    require(uncertainty.statistics["covariance"][0][1] > 0, "joint dependence was lost")
    require(len(uncertainty.as_domain().scenarios) == 12, "missing parameter draws")
    require(
        len(audit.payload.coverage.evaluated_support["scenarios"]) == 24,
        "duplicate draws or operating cases were lost",
    )
    require(
        audit.payload.verdict == "inconclusive" and audit.payload.witness is None,
        "unknown solves became infeasibility proofs",
    )
    require(
        operange.result_from_json(audit.to_json(compact=True)).result_id
        == audit.result_id,
        "audit changed on loading",
    )
    _, nominal, nominal_audit, shared = example["run_shared_example"]()
    require(shared.execution == "completed", "shared-command optimization unresolved")
    require(
        shared.audit.payload.verdict == "pass", "shared commands lack finite coverage"
    )
    require(
        shared.commands["fuel"] > nominal.commands["fuel"],
        "joint equipment uncertainty did not affect the command",
    )
    require(
        nominal_audit.payload.verdict == "inconclusive",
        "nominal comparison changed scope",
    )
    require(len(shared.scenario_objectives) == 24, "shared study lost scenario labels")
    require(
        IdaesSharedControlResult.from_json(shared.to_json()) == shared,
        "shared-command evidence changed on loading",
    )
    print(
        "Installed wheel: joint bootstrap, finite propagation and shared-command optimization passed"
    )


if __name__ == "__main__":
    main()
