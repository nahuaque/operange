"""Controllers see declared readings; physical checks retain latent truth."""

from dataclasses import replace
from fractions import Fraction as F

import pytest

from operange import (
    AffineControlRule,
    AffineController,
    AffineMeasurement,
    AffineOutput,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    ControlTarget,
    ControlTrackingObjective,
    ConvexHullSet,
    DecisionRule,
    FiniteSet,
    FrozenController,
    LinearControl,
    LinearObjective,
    LinearProcessAdapter,
    MeasurementModel,
    NormalizedLInf,
    Parameter,
    RecoursePolicy,
    Scenario,
    verify_result,
)
from examples.measurement_uncertainty import example
from tests.test_certificate_verification import altered, expect_verified
from tests.test_frozen_controllers import values


def controller(offset=0, terms=None, control="steam"):
    return AffineController(
        "Meter-based command",
        (
            AffineControlRule(
                control,
                "MW",
                offset,
                terms
                if terms is not None
                else (AffineTerm("measured_load", 1, "MW/MW"),),
            ),
        ),
        "Test observation policy",
    )


def detail(result, name):
    return next(e.details for e in result.evidence if e.evidence_id == name)


def test_meter_following_fails_on_true_load_and_exports_actual_observations():
    claim = example().with_controller(controller())
    result = claim.evaluate_result({"load": 9, "meter_bias": -0.25})
    assert result.payload.feasibility == "infeasible"  # nosec B101
    assert values(result)["load"] == 9  # nosec B101
    assert values(result)["steam"] == 8.75  # nosec B101
    assert values(result)["steam_balance"] == -0.25  # nosec B101
    data = detail(result, "controller")
    assert data["measurement_readings"] == {"measured_load": 8.75}  # nosec B101
    assert data["observations_by_control"] == {"steam": {"measured_load": 8.75}}  # nosec B101
    audit = claim.audit_result()
    assert audit.payload.verdict == "fail"  # nosec B101
    assert audit.payload.witness.kind == "fixed_policy_failure"  # nosec B101
    expect_verified(result)
    expect_verified(audit)
    assert claim.evaluate_result({"load": 9}).execution == "invalid"  # nosec B101
    assert claim.evaluate_result({"load": 9, "meter_bias": 1}).execution == "invalid"  # nosec B101


@pytest.mark.parametrize("kind", ("box", "finite", "hull"))
@pytest.mark.parametrize("objective", ("fuel", "tracking"))
def test_synthesis_frozen_replay_and_wider_error_domain(kind, objective):
    pytest.importorskip("cvxpy")
    claim = example()
    if kind != "box":
        scenarios = tuple(
            Scenario(str(i), {"load": x, "meter_bias": b}, "Test")
            for i, (x, b) in enumerate(((8, -0.25), (8, 0.25), (10, -0.25), (10, 0.25)))
        )
        domain = (FiniteSet if kind == "finite" else ConvexHullSet)(
            claim.domain.space, scenarios
        )
        claim = replace(claim, domain=domain)
    if objective == "tracking":
        claim = replace(
            claim,
            adapter=replace(
                claim.adapter,
                objective=ControlTrackingObjective(
                    (ControlTarget("steam", 9, 1, "MW"),)
                ),
            ),
        )
    search = claim.synthesize_controller()
    assert search.verified, search.to_json()  # nosec B101
    assert claim.recourse.rules[0].observes == ("measured_load",)  # nosec B101
    assert search.controller.rules[0].observes == ("measured_load",)  # nosec B101
    if objective == "fuel":
        assert detail(search, "objective")["guaranteed_value"] == pytest.approx(
            13.125, abs=1e-6
        )  # nosec B101
    frozen = FrozenController.from_json(search.freeze().to_json())
    assert frozen.audit_result() == search.audit  # nosec B101
    expect_verified(search.audit)
    point = {"load": 8, "meter_bias": -0.25}
    replay = frozen.evaluate_result(point)
    assert replay.payload.feasibility == "feasible"  # nosec B101
    assert values(replay)["steam"] == pytest.approx(8, abs=1e-7)  # nosec B101
    expect_verified(replay)
    wider = BoxSet(
        (
            example().domain.parameter("load"),
            replace(example().domain.parameter("meter_bias"), lower=-0.5, upper=0.5),
        )
    )
    failed = frozen.as_claim(wider).audit_result()
    assert failed.payload.verdict == "fail"  # nosec B101
    expect_verified(failed)
    assert frozen.audit_result() == search.audit  # nosec B101


def test_free_redispatch_cannot_bypass_measurement_information():
    claim = example()
    assert not claim.capabilities.audit.supported  # nosec B101
    assert claim.evaluate_result(claim.domain.nominal).execution == "unsupported"  # nosec B101
    assert claim.audit_result().execution == "unsupported"  # nosec B101
    fixed = replace(
        claim,
        recourse=RecoursePolicy(
            "fixed", (DecisionRule("steam", "MW", "operation", fixed_value=10),)
        ),
    )
    assert fixed.evaluate_result(claim.domain.nominal).execution == "completed"  # nosec B101


@pytest.mark.parametrize("observation", ("load", "meter_bias", "undeclared"))
def test_true_and_error_inputs_cannot_be_used_as_controller_observations(observation):
    claim = example()
    leaked = controller(terms=(AffineTerm(observation, 1, "MW/MW"),))
    with pytest.raises(ValueError, match="permissions"):
        claim.with_controller(leaked)
    permissions = RecoursePolicy(
        "static", (DecisionRule("steam", "MW", "operation", (observation,)),)
    )
    with pytest.raises(ValueError, match="unknown process observations"):
        replace(claim, recourse=permissions).with_controller(leaked)


@pytest.mark.parametrize(
    "variable,unit", (("steam", "MW/MW"), ("measured_load", "MW/MW"), ("load", "kW/MW"))
)
def test_invalid_measurement_terms_are_rejected(variable, unit):
    model = example().adapter
    reading = replace(
        model.measurements.readings[0], terms=(AffineTerm(variable, 1, unit),)
    )
    with pytest.raises(ValueError, match="measurement"):
        replace(model, measurements=replace(model.measurements, readings=(reading,)))


def test_measurement_declaration_validation_and_controller_units():
    model = example().adapter
    readings = model.measurements.readings
    with pytest.raises(ValueError, match="measurement names"):
        replace(model.measurements, readings=readings + readings)
    with pytest.raises(ValueError, match="distinct from physical"):
        replace(
            model,
            measurements=replace(
                model.measurements, readings=(replace(readings[0], name="load"),)
            ),
        )
    with pytest.raises(ValueError, match="declared uncertainty"):
        replace(
            model, measurements=replace(model.measurements, error_inputs=("hidden",))
        )
    with pytest.raises(ValueError, match="cannot change physical"):
        replace(
            model,
            outputs=(
                replace(
                    model.outputs[0], terms=(AffineTerm("meter_bias", 1, "MW/MW"),)
                ),
                model.outputs[1],
            ),
        )
    with pytest.raises(ValueError, match="coefficient units"):
        example().with_controller(
            controller(terms=(AffineTerm("measured_load", 1, "MW/kW"),))
        )


def shared_bias(separate=False):
    parameters = [
        Parameter(n, "MW", lo, lo, hi, 1, "Test")
        for n, lo, hi in (("x", 2, 3), ("y", 4, 5), ("bias", -1, 1))
    ]
    if separate:
        parameters.append(
            Parameter("bias2", "MW", 0, -1, 1, 1, "Independent meter bias")
        )
    domain = BoxSet(tuple(parameters))
    measurements = MeasurementModel(
        "Two meters",
        tuple(
            AffineMeasurement(
                "m" + n,
                "MW",
                0,
                (AffineTerm(n, 1, "MW/MW"), AffineTerm(b, 1, "MW/MW")),
                "Shared offset",
            )
            for n, b in (("x", "bias"), ("y", "bias2" if separate else "bias"))
        ),
        ("bias", "bias2") if separate else ("bias",),
    )
    model = LinearProcessAdapter(
        "Difference measurement",
        domain.space,
        (
            AffineOutput(
                "balance",
                "MW",
                "thermal_power",
                0,
                (
                    AffineTerm("steam", 1, "MW/MW"),
                    AffineTerm("x", -1, "MW/MW"),
                    AffineTerm("y", 1, "MW/MW"),
                ),
                "Balance against true difference",
            ),
        ),
        (AffineRequirement("service", "balance", 0, "ge"),),
        (LinearControl("steam", "MW", -6, 0),),
        (AffineRequirement("upper", "balance", 0),),
        objective=LinearObjective("balance"),
        measurements=measurements,
    )
    return model.as_claim(domain)


def test_shared_bias_cancels_but_independent_biases_do_not():
    rule = controller(
        terms=(AffineTerm("mx", 1, "MW/MW"), AffineTerm("my", -1, "MW/MW"))
    )
    shared = shared_bias().with_controller(rule).audit_result()
    separate = shared_bias(True).with_controller(rule).audit_result()
    assert shared.payload.verdict == "pass"  # nosec B101
    assert separate.payload.verdict == "fail"  # nosec B101
    expect_verified(shared)
    expect_verified(separate)


def test_synthesis_preserves_shared_bias_cancellation():
    pytest.importorskip("cvxpy")
    search = shared_bias().synthesize_controller()
    assert search.verified, search.to_json()  # nosec B101
    expect_verified(search.audit)
    coefficients = {t.variable: t.coefficient for t in search.controller.rules[0].terms}
    assert coefficients == pytest.approx({"mx": 1, "my": -1}, abs=1e-7)  # nosec B101


def test_zero_error_identity_sensor_keeps_exact_zero_tolerance():
    claim = example()
    domain = BoxSet(
        (
            claim.domain.parameter("load"),
            replace(claim.domain.parameter("meter_bias"), lower=0, upper=0),
        )
    )
    model = replace(
        claim.adapter,
        requirements=(replace(claim.adapter.requirements[0], tolerance=0),),
        operating_limits=(
            replace(claim.adapter.operating_limits[0], limit=0, tolerance=0),
        ),
    )
    result = model.as_claim(domain, controller=controller()).audit_result()
    assert result.payload.verdict == "pass"  # nosec B101
    assert (
        detail(result, "enclosure")["measurement_bounds"]["measured_load"][
            "rounding_error_exact"
        ]
        == "0"
    )  # nosec B101
    expect_verified(result)


def test_sensor_rounding_happens_before_command_and_physical_checks():
    claim = example()
    domain = BoxSet(
        (
            replace(claim.domain.parameter("load"), nominal=1, lower=0, upper=2),
            replace(claim.domain.parameter("meter_bias"), lower=0, upper=0),
        )
    )
    measurement = replace(claim.adapter.measurements.readings[0], offset=2**53)
    model = replace(
        claim.adapter,
        measurements=replace(claim.adapter.measurements, readings=(measurement,)),
        controls=(LinearControl("steam", "MW", 0, 2),),
        requirements=(replace(claim.adapter.requirements[0], limit=-1),),
        operating_limits=(replace(claim.adapter.operating_limits[0], limit=1),),
    )
    measured = model.as_claim(domain, controller=controller(offset=-(2**53)))
    point = measured.evaluate_result({"load": 1.25, "meter_bias": 0})
    assert values(point)["steam"] == 2  # nosec B101
    assert values(point)["steam_balance"] == 0.75  # nosec B101
    audit = measured.audit_result()
    assert audit.payload.verdict == "pass"  # nosec B101
    bounds = detail(audit, "enclosure")["measurement_bounds"]["measured_load"]
    assert F(bounds["rounding_error_exact"]) == 1  # nosec B101
    expect_verified(point)
    expect_verified(audit)


def test_verifier_recomputes_sensors_without_production_execution(monkeypatch):
    claim = example().with_controller(controller(offset=0.25))
    audit = claim.audit_result()
    assert audit.payload.verdict == "pass"  # nosec B101

    def forbidden(*args, **kwargs):
        raise AssertionError("Verifier called production measurement execution")

    monkeypatch.setattr(MeasurementModel, "observe", forbidden)
    monkeypatch.setattr(AffineMeasurement, "read", forbidden)
    monkeypatch.setattr(AffineController, "commands", forbidden)
    expect_verified(audit)


def test_fresh_hashes_cannot_hide_forged_sensor_readings_or_rounding_bounds():
    claim = example().with_controller(controller(offset=0.25))
    result = claim.evaluate_result({"load": 9, "meter_bias": -0.25})
    forged = altered(
        result,
        "controller",
        lambda d: d["measurement_readings"].update(measured_load=9),
    )
    assert not verify_result(forged).verified  # nosec B101
    forged = altered(
        result,
        "controller",
        lambda d: d["observations_by_control"]["steam"].update(measured_load=9),
    )
    assert not verify_result(forged).verified  # nosec B101
    forged = altered(
        claim.audit_result(),
        "enclosure",
        lambda d: d["measurement_bounds"]["measured_load"].update(
            rounding_error_exact="0"
        ),
    )
    assert not verify_result(forged).verified  # nosec B101


def test_sensor_overflow_stays_unresolved():
    claim = example()
    measurements = replace(
        claim.adapter.measurements,
        readings=(
            replace(
                claim.adapter.measurements.readings[0],
                terms=(AffineTerm("load", 1e308, "MW/MW"),),
            ),
        ),
    )
    claim = replace(claim.adapter, measurements=measurements).as_claim(
        claim.domain, controller=controller()
    )
    assert claim.evaluate_result(claim.domain.nominal).execution == "unresolved"  # nosec B101
    assert claim.audit_result().payload.verdict == "inconclusive"  # nosec B101


def test_distance_replays_true_load_and_sensor_error():
    source = example()
    claim = replace(
        source, distance=NormalizedLInf(source.domain.space)
    ).with_controller(controller())
    result = claim.breaking_result(
        violation_margins={"serve_true_load": 0.125}, constraints=("serve_true_load",)
    )
    assert result.execution == "completed", result.to_json()  # nosec B101
    witness = result.payload.witness.realizations[0]
    assert witness["meter_bias"] == pytest.approx(-0.125, abs=1e-6)  # nosec B101
    replay = claim.evaluate_result(witness)
    assert replay.payload.feasibility == "infeasible"  # nosec B101
    expect_verified(replay)


def test_aggregate_meter_synthesis_still_avoids_exponential_corners(monkeypatch):
    pytest.importorskip("cvxpy")
    from operange import _controller_synthesis as synthesis

    def forbidden(*args, **kwargs):
        raise AssertionError("Linear box synthesis enumerated generators")

    monkeypatch.setattr(synthesis, "_generators", forbidden)
    original = example().adapter
    domain = BoxSet(
        tuple(Parameter(f"load{i}", "MW", 0.9, 0.8, 1, 0.1, "Load") for i in range(20))
        + (Parameter("meter_bias", "MW", 0, -0.25, 0.25, 0.25, "Bias"),)
    )
    reading = replace(
        original.measurements.readings[0],
        terms=tuple(AffineTerm(n, 1, "MW/MW") for n in domain.space.names),
    )
    balance = replace(
        original.outputs[0],
        terms=(AffineTerm("steam", 1, "MW/MW"),)
        + tuple(AffineTerm(f"load{i}", -1, "MW/MW") for i in range(20)),
    )
    model = replace(
        original,
        input_space=domain.space,
        outputs=(balance, original.outputs[1]),
        controls=(LinearControl("steam", "MW", 0, 21),),
        measurements=replace(original.measurements, readings=(reading,)),
    )
    search = model.as_claim(domain).synthesize_controller(max_vertices=1)
    assert search.verified, search.to_json()  # nosec B101
    assert detail(search, "objective")["guaranteed_value"] == pytest.approx(
        25.625, abs=1e-6
    )  # nosec B101
    expect_verified(search.audit)


def test_different_units_and_negative_measurement_coefficient():
    source = example()
    reading = AffineMeasurement(
        "measured_load",
        "kW",
        12000,
        (AffineTerm("load", -1000, "kW/MW"), AffineTerm("meter_bias", 1000, "kW/MW")),
        "Inverted transmitter",
    )
    model = replace(
        source.adapter,
        measurements=replace(source.adapter.measurements, readings=(reading,)),
    )
    rule = controller(12.25, (AffineTerm("measured_load", -0.001, "MW/kW"),))
    claim = model.as_claim(source.domain, controller=rule)
    point = claim.evaluate_result({"load": 9, "meter_bias": 0.25})
    assert values(point)["steam"] == pytest.approx(9)  # nosec B101
    assert detail(point, "controller")["measurement_readings"] == {
        "measured_load": 3250
    }  # nosec B101
    audit = claim.audit_result()
    assert audit.payload.verdict == "pass"  # nosec B101
    expect_verified(point)
    expect_verified(audit)


def test_identical_readings_cannot_trigger_different_commands():
    pytest.importorskip("cvxpy")
    source = example()
    domain = FiniteSet(
        source.domain.space,
        (
            Scenario("low", {"load": 8, "meter_bias": 0.25}, "Same reading"),
            Scenario("high", {"load": 8.5, "meter_bias": -0.25}, "Same reading"),
        ),
    )
    model = replace(
        source.adapter,
        operating_limits=(replace(source.adapter.operating_limits[0], limit=0.1),),
    )
    search = model.as_claim(domain).synthesize_controller()
    assert search.execution == "unresolved" and not search.verified  # nosec B101
    # Removing the observation restriction permits separate feasible commands.
    oracle = replace(model, measurements=None).as_claim(domain).audit_result()
    assert oracle.payload.verdict == "pass"  # nosec B101
