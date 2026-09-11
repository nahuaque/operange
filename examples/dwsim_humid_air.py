"""Prepare and check the DWSIM connection experiment using only Python stdlib.

Usage: python -m examples.dwsim_humid_air prepare RUN_DIRECTORY [--sample PATH]
       python -m examples.dwsim_humid_air check RUN_DIRECTORY [--capacity-kw 2.5]

This consumer experiment does not emit Operange robustness certificates.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import uuid

from examples.dwsim_humid_air_runner import (
    OBJECT_IDS,
    PROPERTY_PACKAGE,
    SCHEMA,
    file_digest,
)


DEFAULT_SAMPLE = Path("/Applications/DWSIM.app/Contents/MacOS/samples/Humid Air.dwxml")


def prepare(directory, sample=DEFAULT_SAMPLE):
    """Create a new run directory without modifying the installed sample."""
    sample = Path(sample).resolve(strict=True)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    flowsheet = directory / "Humid Air.dwxml"
    shutil.copyfile(sample, flowsheet)
    runner = directory / "dwsim_humid_air_runner.py"
    shutil.copyfile(Path(__file__).with_name(runner.name), runner)
    nominal = {
        "temperature_K": 298.15,
        "pressure_Pa": 101325.0,
        "mass_flow_kg_s": 0.280556,
    }
    request = {
        "schema": SCHEMA,
        "run_id": str(uuid.uuid4()),
        "flowsheet_path": str(flowsheet),
        "source_sha256": file_digest(flowsheet),
        "runner_sha256": file_digest(runner),
        "cases": [
            dict(nominal, id="nominal"),
            dict(nominal, id="warmer_feed", temperature_K=303.15),
            dict(nominal, id="higher_flow", mass_flow_kg_s=0.3366672),
            dict(nominal, id="nominal_repeat"),
        ],
    }
    request_path = directory / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    # This short launcher runs in DWSIM; no .NET dependency enters Operange.
    launcher = (
        "import sys\n"
        f"sys.path.insert(0, {str(directory)!r})\n"
        "import dwsim_humid_air_runner\n"
        f"dwsim_humid_air_runner.run(Flowsheet, {str(request_path)!r})\n"
    )
    (directory / "run_in_dwsim.py").write_text(launcher, encoding="utf-8")
    return request


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Missing or nonnumeric observation")
    if not math.isfinite(value):
        raise ValueError("Nonfinite observation")
    return float(value)


def _check_case(case, row, capacity_kw):
    result = {"id": case["id"], "status": "unresolved"}
    try:
        if row["status"] != "solved" or row["errors"]:
            raise ValueError("Calculation unresolved: " + "; ".join(row["errors"]))
        if set(row["calculated"]) != set(OBJECT_IDS) or not all(
            value is True for value in row["calculated"].values()
        ):
            raise ValueError("An object was not calculated")
        if row["object_errors"] != dict.fromkeys(OBJECT_IDS, ""):
            raise ValueError("DWSIM reported an object error")
        feed, outlet = row["feed"], row["outlet"]
        for stream in (feed, outlet):
            for key in ("temperature_K", "pressure_Pa", "mass_flow_kg_s"):
                if _number(stream[key]) <= 0:
                    raise ValueError("Nonpositive stream state")
            _number(stream["enthalpy_kJ_kg"])
            if not 0 <= _number(stream["vapor_fraction"]) <= 1:
                raise ValueError("Invalid vapor fraction")
        for key in ("temperature_K", "pressure_Pa", "mass_flow_kg_s"):
            if not math.isclose(
                feed[key], _number(case[key]), rel_tol=1e-10, abs_tol=1e-10
            ):
                raise ValueError("Calculated feed does not match requested " + key)
        if abs(outlet["vapor_fraction"] - 1.0) > 1e-8:
            raise ValueError("Cooler outlet specification was not met")
        if abs(outlet["pressure_Pa"] - feed["pressure_Pa"]) > 1e-5:
            raise ValueError("Zero pressure-drop specification was not met")
        duty = _number(row["duty_kW"])
        if duty < 0:
            raise ValueError("Expected positive cooling duty")
        mass_residual = feed["mass_flow_kg_s"] - outlet["mass_flow_kg_s"]
        energy_residual = (
            feed["mass_flow_kg_s"] * feed["enthalpy_kJ_kg"]
            - outlet["mass_flow_kg_s"] * outlet["enthalpy_kJ_kg"]
            - duty
        )
        _number(mass_residual)
        _number(energy_residual)
        if abs(mass_residual) > 1e-8 or abs(energy_residual) > 1e-7:
            raise ValueError("Mass or energy balance did not close")
        margin = capacity_kw - duty
        result.update(
            {
                "status": "observed_within_limit"
                if margin >= 0
                else "observed_violation",
                "duty_kW": duty,
                "capacity_margin_kW": margin,
                "outlet_temperature_K": outlet["temperature_K"],
                "mass_residual_kg_s": mass_residual,
                "energy_residual_kW": energy_residual,
                "elapsed_s": _number(row["elapsed_s"]),
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        result["reason"] = str(exc)
    return result


def check(directory, capacity_kw=2.5):
    """Check request binding, finite coverage, physical balances and margins.

    Protocol/provenance failures raise ValueError. Individual solve failures
    remain unresolved, retaining valid observations from the other cases.
    """
    capacity_kw = _number(capacity_kw)
    if capacity_kw <= 0:
        raise ValueError("Capacity must be positive")
    directory = Path(directory)
    request_path = directory / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response = json.loads((directory / "response.json").read_text(encoding="utf-8"))
    for key in ("schema", "run_id", "source_sha256", "runner_sha256"):
        if response.get(key) != request[key]:
            raise ValueError("Response does not match request: " + key)
    if request["schema"] != SCHEMA:
        raise ValueError("Unknown experiment schema")
    if response.get("request_sha256") != file_digest(request_path):
        raise ValueError("Response belongs to a different request")
    if request["source_sha256"] != file_digest(request["flowsheet_path"]):
        raise ValueError("Prepared flowsheet file changed")
    if request["runner_sha256"] != file_digest(directory / "dwsim_humid_air_runner.py"):
        raise ValueError("Prepared runner changed")
    if response.get("errors"):
        raise ValueError("DWSIM batch unresolved: " + "; ".join(response["errors"]))
    if response.get("property_packages") != [PROPERTY_PACKAGE]:
        raise ValueError("Unexpected property package")
    if not response.get("dwsim_version") or not response.get("loaded_state_sha256"):
        raise ValueError("Missing runtime provenance")
    expected = [case["id"] for case in request["cases"]]
    actual = [row["id"] for row in response["cases"]]
    if not expected or len(set(expected)) != len(expected) or actual != expected:
        raise ValueError("Missing, duplicate, reordered or unexpected scenario results")
    rows = [
        _check_case(case, row, capacity_kw)
        for case, row in zip(request["cases"], response["cases"])
    ]
    return {
        "scope": "Observed finite cases under the sample's fixed cooler specification; "
        "no recourse infeasibility or continuous-domain certificate.",
        "dwsim_version": response["dwsim_version"],
        "capacity_kW": capacity_kw,
        "restoration": response.get("restoration", "unresolved"),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("directory", type=Path)
    prep.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    inspect = sub.add_parser("check")
    inspect.add_argument("directory", type=Path)
    inspect.add_argument("--capacity-kw", type=float, default=2.5)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            request = prepare(args.directory, args.sample)
            print("Open in DWSIM: " + request["flowsheet_path"])
            print("In Tools > Script Manager, use IronPython and Run Async:")
            print((args.directory / "run_in_dwsim.py").read_text(encoding="utf-8"))
        else:
            report = check(args.directory, args.capacity_kw)
            print(json.dumps(report, indent=2, allow_nan=False))
            return int(
                any(row["status"] == "unresolved" for row in report["cases"])
                or report["restoration"] != "solved"
            )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, "DWSIM experiment unresolved: " + str(exc) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
