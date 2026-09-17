"""Consumer check for local uncertainty from an installed wheel."""

from pathlib import Path
import runpy
import sys

import operange
from operange.integrations.idaes import IdaesUncertaintyResult


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    require(sys.flags.isolated == 1, "run isolated Python")
    require(not Path("pyproject.toml").exists(), "run outside checkout")
    require(
        Path(operange.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),
        "use installed wheel",
    )
    require(
        not any(n in sys.modules for n in ("idaes", "pyomo", "pandas")),
        "keep public imports lazy",
    )
    example = runpy.run_path(str(Path(__file__).with_name("local_uncertainty.py")))
    fixed, optimized = example["run_example"]()
    require(
        abs(fixed.statistics["responses"][0]["variance"] - 1.3) < 1e-10,
        "retain correlation",
    )
    require(abs(optimized.jacobian[1][0] + 14.0625) < 1e-6, "reoptimize fuel")
    require(len(optimized.perturbations) == 8, "retain all perturbation checks")
    for result in (fixed, optimized):
        require(
            IdaesUncertaintyResult.from_json(result.to_json()) == result,
            "retain portable evidence",
        )
    print("Installed wheel: fixed and reoptimized correlated uncertainty passed")


if __name__ == "__main__":
    main()
