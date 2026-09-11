"""Pytest assertions are test expectations, not production input guards."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import json
from types import SimpleNamespace

import pytest

from operange import BoxSet
from operange.reference import (
    FrozenHeatRecoveryContract,
    HeatDemand,
    HeatRecoveryDesign,
    RepairAction,
)
from operange import linear
from operange.reference import example_claim


def alter_parameter(claim, name, **changes):
    return replace(
        claim,
        uncertainty=BoxSet(
            tuple(
                replace(p, **changes) if p.name == name else p
                for p in claim.uncertainty.parameters
            )
        ),
    )


def test_analytical_baseline_audit_and_minimum_witness():
    claim = example_claim()
    audit = claim.audit()
    assert audit.status == "fail"  # nosec B101
    assert audit.nominal.maximum_heat_mw == pytest.approx(8)  # nosec B101
    assert audit.worst.maximum_heat_mw == pytest.approx(6)  # nosec B101
    assert audit.worst.required_heat_mw == pytest.approx(7.5)  # nosec B101
    assert audit.worst.slack_mw == pytest.approx(-1.5)  # nosec B101
    assert audit.worst.binding_limits == ("source_capacity",)  # nosec B101

    boundary = claim.boundary_distance()
    assert boundary.minimum_verified  # nosec B101
    assert boundary.distance == pytest.approx(float(Fraction(4, 7)))  # nosec B101
    assert boundary.operation.status == "pass"  # nosec B101
    assert boundary.operation.slack_mw == pytest.approx(0)  # nosec B101

    witness = claim.breaking_witness()
    assert witness.status == "found"  # nosec B101
    assert witness.minimum_verified  # nosec B101
    assert witness.distance == pytest.approx(float(Fraction(201, 350)))  # nosec B101
    assert claim.uncertainty.contains(witness.operation.realization)  # nosec B101
    assert witness.operation.realization["source_derating"] == pytest.approx(  # nosec B101
        float(Fraction(201, 1400))
    )
    assert witness.operation.realization["demand_increase"] == pytest.approx(  # nosec B101
        float(Fraction(201, 1400))
    )
    op = witness.operation
    assert op.status == "fail"  # nosec B101
    assert op.slack_mw == pytest.approx(-0.01)  # nosec B101
    assert op.maximum_heat_mw == pytest.approx(float(Fraction(1199, 175)))  # nosec B101
    assert op.required_heat_mw == pytest.approx(float(Fraction(4803, 700)))  # nosec B101
    assert op.source_heat_mw + op.power_mw == pytest.approx(op.maximum_heat_mw)  # nosec B101
    assert op.power_mw < claim.design.power_capacity_mw  # nosec B101
    assert op.evidence.max_residual <= claim.tolerance  # nosec B101
    assert op.evidence.reference_error <= claim.tolerance  # nosec B101


@pytest.mark.parametrize(
    "cop,source,power,base",
    [
        (2, 5, 2, 3),
        (3, 6, 1, 2),
        (4, 8, 2, 6),
        (5, 2, 4, 2),
    ],
)
def test_operating_lp_matches_independent_energy_balance_across_designs(
    cop, source, power, base
):
    claim = replace(
        example_claim(),
        cop=cop,
        design=HeatRecoveryDesign(source, power),
        requirement=HeatDemand(base),
    )
    for f in (0, 0.1, 0.25):
        for delta in (0, 0.15, 0.25):
            op = claim.evaluate({"source_derating": f, "demand_increase": delta})
            # Solve p <= P and (C-1)*p <= S*(1-f) independently in the test.
            p_max = min(power, source * (1 - f) / (cop - 1))
            assert op.maximum_heat_mw == pytest.approx(cop * p_max)  # nosec B101
            assert op.required_heat_mw == pytest.approx(base * (1 + delta))  # nosec B101
            assert op.power_mw == pytest.approx(p_max)  # nosec B101


def test_compressor_can_be_the_nearest_failure_branch():
    claim = replace(example_claim(), design=HeatRecoveryDesign(20, 1.6))
    witness = claim.breaking_witness()
    assert witness.minimum_verified  # nosec B101
    assert witness.distance == pytest.approx(0.41 / 1.5)  # nosec B101
    assert "compressor_capacity" in witness.operation.binding_limits  # nosec B101
    assert witness.operation.maximum_heat_mw == pytest.approx(6.4)  # nosec B101


def test_distance_handles_clipped_axis_and_shifted_nominal():
    claim = alter_parameter(example_claim(), "source_derating", upper=0.1)
    witness = claim.breaking_witness()
    assert witness.minimum_verified  # nosec B101
    assert witness.distance == pytest.approx(float(Fraction(121, 150)))  # nosec B101
    assert witness.operation.realization["source_derating"] == pytest.approx(0.1)  # nosec B101
    assert witness.operation.slack_mw == pytest.approx(-0.01)  # nosec B101

    shifted = alter_parameter(example_claim(), "source_derating", nominal=0.05)
    shifted = alter_parameter(shifted, "demand_increase", nominal=0.1)
    assert shifted.breaking_witness().distance == pytest.approx(1.01 / 3.5)  # nosec B101


def test_engineering_scales_change_distance_and_named_coordinates_ignore_order():
    claim = alter_parameter(example_claim(), "source_derating", scale=0.1)
    witness = claim.breaking_witness()
    assert witness.distance == pytest.approx(2.01 / 2.3)  # nosec B101
    swapped = replace(
        claim, uncertainty=BoxSet(tuple(reversed(claim.uncertainty.parameters)))
    )
    assert swapped.breaking_witness().distance == pytest.approx(witness.distance)  # nosec B101
    assert swapped.audit().worst.slack_mw == pytest.approx(claim.audit().worst.slack_mw)  # nosec B101


def test_margin_unreachable_does_not_imply_claim_passes():
    claim = example_claim()
    assert claim.audit().status == "fail"  # nosec B101
    result = claim.breaking_witness(shortfall_margin_mw=2)
    assert result.status == "unreachable"  # nosec B101
    assert result.distance is None  # nosec B101
    assert result.operation is None  # nosec B101


def test_boundary_at_domain_endpoint_is_distinct_from_strict_failure():
    claim = replace(example_claim(), design=HeatRecoveryDesign(7.5, 2))
    assert claim.audit().status == "pass"  # nosec B101
    boundary = claim.boundary_distance()
    assert boundary.status == "found"  # nosec B101
    assert boundary.distance == pytest.approx(1)  # nosec B101
    assert boundary.operation.status == "pass"  # nosec B101
    assert claim.breaking_witness().status == "unreachable"  # nosec B101


def test_zero_radius_when_nominal_already_meets_breaking_margin():
    claim = replace(example_claim(), design=HeatRecoveryDesign(4, 2))
    result = claim.breaking_witness()
    assert result.status == "already_broken"  # nosec B101
    assert result.distance == 0  # nosec B101
    assert result.operation.status == "fail"  # nosec B101
    assert result.minimum_verified  # nosec B101


def test_nominal_small_failure_does_not_bypass_requested_margin():
    claim = replace(example_claim(), design=HeatRecoveryDesign((6 - 0.005) * 0.75, 2))
    result = claim.breaking_witness()
    assert claim.audit().nominal.status == "fail"  # nosec B101
    assert result.status == "found"  # nosec B101
    assert result.distance > 0  # nosec B101
    assert result.operation.slack_mw == pytest.approx(-0.01)  # nosec B101


def test_degenerate_box_axes_and_zero_capacity_are_supported():
    claim = alter_parameter(example_claim(), "source_derating", upper=0)
    claim = alter_parameter(claim, "demand_increase", upper=0)
    assert claim.audit().status == "pass"  # nosec B101
    assert claim.boundary_distance().status == "unreachable"  # nosec B101
    zero = replace(claim, design=HeatRecoveryDesign(0, 0))
    assert zero.breaking_witness().status == "already_broken"  # nosec B101
    assert zero.evaluate(zero.uncertainty.nominal).maximum_heat_mw == 0  # nosec B101


@pytest.mark.parametrize(
    "changes",
    [
        {"scale": 0},
        {"scale": -1},
        {"lower": 0.1},
        {"upper": -0.1},
        {"nominal": float("nan")},
        {"upper": float("inf")},
        {"scale": True},
        {"name": ""},
        {"unit": ""},
        {"provenance": ""},
    ],
)
def test_invalid_parameter_contracts_rejected(changes):
    with pytest.raises(ValueError):
        replace(example_claim().uncertainty.parameters[0], **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"cop": 1},
        {"cop": float("inf")},
        {"cop": True},
        {"tolerance": 1},
        {"tolerance": 0},
        {"recourse": "perfect_foresight"},
    ],
)
def test_unsupported_claim_inputs_rejected(changes):
    with pytest.raises(ValueError):
        replace(example_claim(), **changes)


def test_units_domain_shape_and_invalid_physical_inputs_are_rejected():
    claim = example_claim()
    for changes in ({"unit": "percent"}, {"upper": 1.01}, {"lower": -0.01}):
        with pytest.raises(ValueError):
            alter_parameter(claim, "source_derating", **changes)
    with pytest.raises(ValueError):
        alter_parameter(claim, "demand_increase", lower=-1.1)
    for point in (
        {"source_derating": 0},
        {"source_derating": 0, "demand_increase": 0, "cop": 4},
        {"source_derating": 0, "demand_increase": float("nan")},
        {"source_derating": 0.26, "demand_increase": 0},
    ):
        with pytest.raises(ValueError):
            claim.evaluate(point)
    for margin in (0, -1, claim.tolerance, float("nan")):
        with pytest.raises(ValueError):
            claim.breaking_witness(shortfall_margin_mw=margin)
    with pytest.raises(ValueError):
        BoxSet(())
    with pytest.raises(ValueError):
        BoxSet((claim.uncertainty.parameters[0],) * 2)
    with pytest.raises(ValueError):
        HeatRecoveryDesign(-1, 2)
    with pytest.raises(ValueError):
        HeatDemand(0)
    with pytest.raises(ValueError):
        RepairAction("invalid provenance", HeatRecoveryDesign(), "")


@pytest.mark.parametrize("status", [1, 2, 3, 4])
def test_solver_failure_never_becomes_physical_failure_or_unbreakability(
    monkeypatch, status
):
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=False,
            status=status,
            message="injected solver failure",
            x=None,
        ),
    )
    claim = example_claim()
    assert claim.audit().status == "inconclusive"  # nosec B101
    result = claim.breaking_witness()
    assert result.status == "inconclusive"  # nosec B101
    assert not result.minimum_verified  # nosec B101
    assert all(
        row.audit.status == "inconclusive"
        for row in claim.compare_repairs(example_repairs()).candidates
    )  # nosec B101


def test_solver_exception_is_an_unresolved_evaluation(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("solver unavailable")

    monkeypatch.setattr(linear, "linprog", fail)
    assert example_claim().audit().status == "inconclusive"  # nosec B101


@pytest.mark.parametrize("coordinates", [[0, 0, 0], [0, 0, 1], [0, 0, float("nan")]])
def test_success_flag_cannot_bypass_residual_or_global_capacity_checks(
    monkeypatch, coordinates
):
    monkeypatch.setattr(
        linear,
        "linprog",
        lambda *a, **k: SimpleNamespace(
            success=True,
            status=0,
            message="optimal",
            x=coordinates,
        ),
    )
    assert example_claim().audit().status == "inconclusive"  # nosec B101


def test_unresolved_outer_branch_prevents_minimum_claim_even_with_verified_failure(
    monkeypatch,
):
    original = linear.linprog

    def fail_compressor(objective, **kwargs):
        if objective == [0, 0, 1] and kwargs["A_ub"][-1][0] == 0:
            return SimpleNamespace(
                success=False, status=1, message="time limit", x=None
            )
        return original(objective, **kwargs)

    monkeypatch.setattr(linear, "linprog", fail_compressor)
    witness = example_claim().breaking_witness()
    assert witness.operation.status == "fail"  # nosec B101
    assert witness.status == "inconclusive"  # nosec B101
    assert witness.distance == pytest.approx(float(Fraction(201, 350)))  # nosec B101
    assert not witness.minimum_verified  # nosec B101


def test_frozen_contract_is_immutable_and_replays_after_roundtrip(monkeypatch):
    claim = example_claim().with_repair(example_repairs()[3])
    frozen = claim.freeze()
    saved = frozen.to_json()
    with pytest.raises(FrozenInstanceError):
        frozen.claim.cop = 5
    with monkeypatch.context() as context:
        context.setattr(
            linear, "linprog", lambda *a, **k: pytest.fail("load must not solve")
        )
        restored = FrozenHeatRecoveryContract.from_json(saved)
    assert restored.fingerprint == frozen.fingerprint  # nosec B101
    assert restored.audit_realization(claim.uncertainty.upper_corner).status == "pass"  # nosec B101
    assert replace(claim, cop=5).freeze().fingerprint != frozen.fingerprint  # nosec B101
    swapped = replace(
        claim, uncertainty=BoxSet(tuple(reversed(claim.uncertainty.parameters)))
    )
    assert swapped.freeze().fingerprint == frozen.fingerprint  # nosec B101
    with pytest.raises(ValueError, match="outside"):
        restored.audit_realization({"source_derating": 0.26, "demand_increase": 0})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(schema_version=2),
        lambda p: p.update(schema_version=True),
        lambda p: p.update(model_id="unknown"),
        lambda p: p.update(extra=True),
        lambda p: p["claim"].update(cop=5),
        lambda p: p["claim"]["design"].update(source_capacity_mw=float("nan")),
        lambda p: p["claim"]["uncertainty"]["parameters"][0].update(scale=0),
    ],
)
def test_invalid_or_tampered_frozen_contracts_rejected(mutation):
    payload = json.loads(example_claim().freeze().to_json())
    mutation(payload)
    with pytest.raises(ValueError):
        FrozenHeatRecoveryContract.from_json(json.dumps(payload))


def test_duplicate_manifest_fields_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        FrozenHeatRecoveryContract.from_json('{"schema_version":1,"schema_version":1}')


def example_repairs():
    return tuple(
        RepairAction(name, HeatRecoveryDesign(source, power), "Test equipment change")
        for name, source, power in (
            ("Baseline", 6, 2),
            ("Compressor uprate", 6, 2.5),
            ("Small source-capture upgrade", 7, 2),
            ("Source-capture upgrade", 7.5, 2),
            ("Larger source-capture upgrade", 8, 2),
        )
    )
