"""Run inside DWSIM's IronPython script manager; no Operange import required.

This is a bounded connection experiment for the bundled Humid Air sample.
The host-side example validates the exported observations separately.
"""

import hashlib
import json
import math
import os
import time


SCHEMA = "operange_dwsim_humid_air/v1"
OBJECT_IDS = {
    "Air": "MAT-4882ac33-f53f-4194-962c-7c9f9d348903",
    "Dew": "MAT-e5226ec1-2e75-4af9-bbde-68ced0a34406",
    "COOL-000": "RESF-ed7e55a8-76f5-4e6b-aad5-a3243894a7be",
    "E": "EN-3e8c2aa8-9aa6-4623-9eb5-02b342109550",
}
PROPERTY_PACKAGE = "DWSIM.Thermodynamics.PropertyPackages.PengRobinsonPropertyPackage"


def file_digest(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def _finite(value):
    if value is None:
        raise ValueError("Missing DWSIM property")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite DWSIM property")
    return number


def _stream_values(obj):
    # DWSIM uses kJ/kg for enthalpy and kW for energy, despite SI naming.
    # Read nullable properties directly: shortcut getters can turn missing data
    # into zero, which would hide an incomplete calculation.
    props = obj.Phases[0].Properties
    return {
        "temperature_K": _finite(props.temperature),
        "pressure_Pa": _finite(props.pressure),
        "mass_flow_kg_s": _finite(props.massflow),
        "enthalpy_kJ_kg": _finite(props.enthalpy),
        "vapor_fraction": _finite(obj.Phases[2].Properties.molarfraction),
    }


def run(flowsheet, request_path):
    """Execute an explicit finite batch, force recalculation, and restore feed."""
    with open(request_path) as stream:
        request = json.load(stream)
    if request["schema"] != SCHEMA:
        raise ValueError("Unknown DWSIM experiment request")
    response = {
        "schema": SCHEMA,
        "run_id": request["run_id"],
        "source_sha256": request["source_sha256"],
        "request_sha256": file_digest(request_path),
        "runner_sha256": file_digest(__file__),
        "cases": [],
        "errors": [],
        "restoration": "not_attempted",
    }
    feed = None
    original = None
    try:
        path = os.path.realpath(flowsheet.GetSimulationFilePath())
        if path != os.path.realpath(request["flowsheet_path"]):
            raise ValueError("Open the prepared flowsheet copy before running")
        if file_digest(path) != request["source_sha256"]:
            raise ValueError("The prepared flowsheet file has changed")
        objects = dict(
            (str(o.GraphicObject.Tag), o) for o in flowsheet.SimulationObjects.Values
        )
        ids = dict((tag, str(o.Name)) for tag, o in objects.items())
        if ids != OBJECT_IDS:
            raise ValueError("This experiment requires the bundled Humid Air sample")
        packages = [
            str(p.GetType().FullName) for p in flowsheet.PropertyPackages.Values
        ]
        if packages != [PROPERTY_PACKAGE]:
            raise ValueError("Expected the sample's Peng-Robinson property package")
        cooler = objects["COOL-000"]
        if (
            str(cooler.CalcMode) != "OutletVaporFraction"
            or float(cooler.OutletVaporFraction) != 1.0
            or float(cooler.Eficiencia) != 100.0
            or float(cooler.DeltaP or 0.0) != 0.0
        ):
            raise ValueError("The sample's cooler operating specification changed")
        feed = objects["Air"]
        original = (feed.GetTemperature(), feed.GetPressure(), feed.GetMassFlow())
        response["dwsim_version"] = str(flowsheet.GetType().Assembly.GetName().Version)
        response["property_packages"] = packages
        # A source-file hash alone does not identify unsaved in-memory changes.
        response["loaded_state_sha256"] = hashlib.sha256(
            str(flowsheet.SaveToXML().ToString()).encode("utf-8")
        ).hexdigest()
        for case in request["cases"]:
            started = time.time()
            row = {"id": case["id"], "status": "unresolved", "errors": []}
            try:
                feed.SetTemperature(float(case["temperature_K"]))
                feed.SetPressure(float(case["pressure_Pa"]))
                feed.SetMassFlow(float(case["mass_flow_kg_s"]))
                flowsheet.ResetCalculationStatus()
                row["errors"] = [str(e) for e in flowsheet.RequestCalculationAndWait()]
                row["calculated"] = dict(
                    (tag, bool(obj.Calculated)) for tag, obj in objects.items()
                )
                row["object_errors"] = dict(
                    (tag, str(obj.ErrorMessage or "")) for tag, obj in objects.items()
                )
                if (
                    row["errors"]
                    or not all(row["calculated"].values())
                    or any(row["object_errors"].values())
                ):
                    raise ValueError("DWSIM did not calculate every object cleanly")
                row["feed"] = _stream_values(feed)
                row["outlet"] = _stream_values(objects["Dew"])
                row["duty_kW"] = _finite(objects["E"].EnergyFlow)
                row["status"] = "solved"
            except Exception as exc:
                row["errors"].append(str(exc))
            row["elapsed_s"] = time.time() - started
            response["cases"].append(row)
    except Exception as exc:
        response["errors"].append(str(exc))
    finally:
        if feed is not None and original is not None:
            try:
                feed.SetTemperature(original[0])
                feed.SetPressure(original[1])
                feed.SetMassFlow(original[2])
                flowsheet.ResetCalculationStatus()
                errors = [str(e) for e in flowsheet.RequestCalculationAndWait()]
                if errors or not all(o.Calculated for o in objects.values()):
                    raise ValueError(
                        "Feed restored but recalculation was unresolved: "
                        + "; ".join(errors)
                    )
                response["restoration"] = "solved"
            except Exception as exc:
                response["restoration"] = "unresolved"
                response["errors"].append(str(exc))
        output_path = os.path.join(os.path.dirname(request_path), "response.json")
        # Write only complete JSON; the reader must never accept a partial run.
        with open(output_path + ".tmp", "w") as stream:
            json.dump(response, stream, indent=2, allow_nan=False)
        os.replace(output_path + ".tmp", output_path)
    print("Operange: wrote " + output_path)
