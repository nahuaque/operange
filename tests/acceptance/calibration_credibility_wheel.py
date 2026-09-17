"""Exercise quality diagnostics outside the checkout with an installed wheel."""

from pathlib import Path
import runpy
import sys

import operange
from operange.integrations.parmest import CalibrationDiagnostics, CalibrationValidation


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
        not any(name in sys.modules for name in ("idaes", "pyomo", "pandas")),
        "imports must stay lazy",
    )
    example = runpy.run_path(
        str(Path(__file__).with_name("calibration_credibility.py"))
    )
    known, confounded, validation = example["run_example"]()
    require(known.analysis["rank"] == 1, "known-area fit information missing")
    require(
        confounded.analysis["rank"] == 1
        and confounded.analysis["parameter_count"] == 2,
        "confounded product lost",
    )
    require(
        validation.execution == "completed"
        and validation.metrics[0]["max_absolute_residual"] > 700,
        "held-out measurement was refitted",
    )
    require(
        CalibrationDiagnostics.from_json(confounded.to_json()) == confounded,
        "diagnostic round trip changed evidence",
    )
    require(
        CalibrationValidation.from_json(validation.to_json()) == validation,
        "validation round trip changed evidence",
    )
    print("Installed wheel: local parameter information and held-out prediction passed")


if __name__ == "__main__":
    main()
