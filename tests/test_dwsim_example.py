"""Offline checks of the experimental simulator observation boundary.

The synthetic rows here are not DWSIM integration evidence. The live smoke run
is manual and documented separately, so CI needs no simulator installation.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.dwsim_humid_air import check, prepare
from examples.dwsim_humid_air_runner import OBJECT_IDS, PROPERTY_PACKAGE, file_digest
from examples import dwsim_humid_air_runner as runner


@pytest.fixture
def batch(tmp_path):
    sample = tmp_path / "source.dwxml"
    sample.write_text("synthetic source for protocol tests", encoding="utf-8")
    directory = tmp_path / "run"
    request = prepare(directory, sample)
    response = {
        key: request[key]
        for key in ("schema", "run_id", "source_sha256", "runner_sha256")
    }
    response.update(
        {
            "request_sha256": file_digest(directory / "request.json"),
            "loaded_state_sha256": "a" * 64,
            "dwsim_version": "synthetic-test-runtime",
            "property_packages": [PROPERTY_PACKAGE],
            "restoration": "solved",
            "errors": [],
            "cases": [],
        }
    )
    for case in request["cases"]:
        feed = {
            key: case[key] for key in ("temperature_K", "pressure_Pa", "mass_flow_kg_s")
        }
        feed.update(enthalpy_kJ_kg=10.0, vapor_fraction=1.0)
        outlet = dict(feed, temperature_K=290.0, enthalpy_kJ_kg=0.0)
        response["cases"].append(
            {
                "id": case["id"],
                "status": "solved",
                "errors": [],
                "calculated": dict.fromkeys(OBJECT_IDS, True),
                "object_errors": dict.fromkeys(OBJECT_IDS, ""),
                "feed": feed,
                "outlet": outlet,
                "duty_kW": case["mass_flow_kg_s"] * 10.0,
                "elapsed_s": 0.01,
            }
        )
    (directory / "response.json").write_text(json.dumps(response), encoding="utf-8")
    return directory, response


def write_response(batch):
    directory, response = batch
    (directory / "response.json").write_text(json.dumps(response), encoding="utf-8")


def test_preparation_is_non_destructive_and_does_not_copy_simulator_into_repo(batch):
    directory, _ = batch
    source = directory.parent / "source.dwxml"
    before = source.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(directory, source)
    assert source.read_bytes() == before  # nosec B101
    assert (directory / "Humid Air.dwxml").read_bytes() == before  # nosec B101


def test_observed_capacity_change_and_closed_balances(batch):
    directory, _ = batch
    original = check(directory, 3.0)
    enlarged = check(directory, 4.0)
    assert [row["status"] for row in original["cases"]] == [
        "observed_within_limit",
        "observed_within_limit",
        "observed_violation",
        "observed_within_limit",
    ]  # nosec B101
    assert all(row["status"] == "observed_within_limit" for row in enlarged["cases"])  # nosec B101
    assert all(row["energy_residual_kW"] == 0.0 for row in original["cases"])  # nosec B101


@pytest.mark.parametrize(
    "fault",
    [
        "solver_error",
        "uncalculated",
        "object_error",
        "stale_input",
        "mass_balance",
        "energy_balance",
        "nonfinite",
        "missing_value",
        "vapor_spec",
        "pressure_spec",
    ],
)
def test_bad_calculations_are_unresolved_without_erasing_other_observations(
    batch, fault
):
    directory, response = batch
    row = response["cases"][0]
    if fault == "solver_error":
        row.update(status="unresolved", errors=["solver timed out"])
    elif fault == "uncalculated":
        row["calculated"]["Dew"] = False
    elif fault == "object_error":
        row["object_errors"]["Dew"] = "flash failed"
    elif fault == "stale_input":
        row["feed"]["temperature_K"] = 280.0
    elif fault == "mass_balance":
        row["outlet"]["mass_flow_kg_s"] += 0.01
    elif fault == "energy_balance":
        row["duty_kW"] += 1.0
    elif fault == "nonfinite":
        row["duty_kW"] = float("nan")
    elif fault == "missing_value":
        row["outlet"]["enthalpy_kJ_kg"] = None
    elif fault == "vapor_spec":
        row["outlet"]["vapor_fraction"] = 0.5
    else:
        row["outlet"]["pressure_Pa"] = 200000.0
    write_response(batch)
    report = check(directory, 3.0)
    assert report["cases"][0]["status"] == "unresolved"  # nosec B101
    assert "capacity_margin_kW" not in report["cases"][0]  # nosec B101
    assert report["cases"][2]["status"] == "observed_violation"  # nosec B101


@pytest.mark.parametrize(
    "fault",
    [
        "run_id",
        "request",
        "source",
        "runner",
        "missing_case",
        "duplicate",
        "extra_case",
    ],
)
def test_stale_or_incomplete_batches_cannot_be_reported(batch, fault):
    directory, response = batch
    if fault == "run_id":
        response["run_id"] = "old-run"
    elif fault == "request":
        path = directory / "request.json"
        request = json.loads(path.read_text())
        request["cases"][0]["temperature_K"] += 1.0
        path.write_text(json.dumps(request), encoding="utf-8")
    elif fault in ("source", "runner"):
        name = "Humid Air.dwxml" if fault == "source" else "dwsim_humid_air_runner.py"
        path = directory / name
        path.write_text("changed", encoding="utf-8")
    elif fault == "missing_case":
        response["cases"].pop()
    elif fault == "duplicate":
        response["cases"][1] = copy.deepcopy(response["cases"][0])
    else:
        response["cases"].append(copy.deepcopy(response["cases"][0]))
    write_response(batch)
    with pytest.raises(ValueError):
        check(directory)


def test_missing_response_and_invalid_capacity_are_not_successes(batch):
    directory, _ = batch
    for capacity in (True, -1, 0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            check(directory, capacity)
    Path(directory / "response.json").unlink()
    with pytest.raises(FileNotFoundError):
        check(directory)


def test_overflow_in_recomputed_energy_balance_cannot_pass(batch):
    directory, response = batch
    request_path = directory / "request.json"
    request = json.loads(request_path.read_text())
    request["cases"][0]["mass_flow_kg_s"] = 100.0
    request_path.write_text(json.dumps(request), encoding="utf-8")
    response["request_sha256"] = file_digest(request_path)
    for key in ("feed", "outlet"):
        response["cases"][0][key].update(mass_flow_kg_s=100.0, enthalpy_kJ_kg=1e308)
    write_response(batch)
    assert check(directory)["cases"][0]["status"] == "unresolved"  # nosec B101


def test_runner_continues_after_solve_failure_and_restores_original_feed(batch):
    directory, _ = batch
    state = {
        "temperature": 299.0,
        "pressure": 101325.0,
        "massflow": 0.28,
        "enthalpy": 10.0,
    }
    original = dict(state)
    phases = {
        0: SimpleNamespace(Properties=SimpleNamespace(**state)),
        2: SimpleNamespace(Properties=SimpleNamespace(molarfraction=1.0)),
    }
    feed = SimpleNamespace(
        GetTemperature=lambda: state["temperature"],
        GetPressure=lambda: state["pressure"],
        GetMassFlow=lambda: state["massflow"],
        SetTemperature=lambda value: state.update(temperature=value),
        SetPressure=lambda value: state.update(pressure=value),
        SetMassFlow=lambda value: state.update(massflow=value),
        Phases=phases,
    )
    objects = {
        "Air": feed,
        "Dew": SimpleNamespace(Phases=phases),
        "E": SimpleNamespace(EnergyFlow=1.0),
        "COOL-000": SimpleNamespace(
            CalcMode="OutletVaporFraction",
            OutletVaporFraction=1.0,
            Eficiencia=100.0,
            DeltaP=0.0,
        ),
    }
    for tag, obj in objects.items():
        obj.GraphicObject = SimpleNamespace(Tag=tag)
        obj.Name = OBJECT_IDS[tag]
        obj.Calculated = True
        obj.ErrorMessage = ""
    calls = []

    def reset():
        for obj in objects.values():
            obj.Calculated = False

    def solve():
        calls.append(dict(state))
        if len(calls) == 1:
            return [RuntimeError("injected nonconvergence")]
        for obj in objects.values():
            obj.Calculated = True
        return []

    flowsheet = SimpleNamespace(
        GetSimulationFilePath=lambda: str(directory / "Humid Air.dwxml"),
        SimulationObjects=SimpleNamespace(Values=list(objects.values())),
        PropertyPackages=SimpleNamespace(
            Values=[
                SimpleNamespace(
                    GetType=lambda: SimpleNamespace(FullName=PROPERTY_PACKAGE)
                )
            ]
        ),
        GetType=lambda: SimpleNamespace(
            Assembly=SimpleNamespace(
                GetName=lambda: SimpleNamespace(Version="synthetic-test-runtime")
            )
        ),
        SaveToXML=lambda: SimpleNamespace(ToString=lambda: "synthetic state"),
        ResetCalculationStatus=reset,
        RequestCalculationAndWait=solve,
    )
    runner.run(flowsheet, str(directory / "request.json"))
    response = json.loads((directory / "response.json").read_text())
    assert response["cases"][0]["status"] == "unresolved"  # nosec B101
    assert response["cases"][1]["status"] == "solved"  # nosec B101
    assert "feed" not in response["cases"][0]  # nosec B101
    assert len(calls) == 5  # nosec B101
    assert calls[-1] == original == state  # nosec B101
    assert response["restoration"] == "solved"  # nosec B101


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_runner_rejects_missing_and_nonfinite_properties(value):
    with pytest.raises(ValueError):
        runner._finite(value)
