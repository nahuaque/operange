"""Guaranteed ranges require checked passing audits and genuine failure evidence."""

from dataclasses import replace
from fractions import Fraction as F
import json
from math import nextafter, inf
# CLI tests use a fixed executable and arguments, without a shell.
import subprocess  # nosec B404
import sys

import pytest

from operange import (
    AffineControlRule,
    AffineController,
    AffineRequirement,
    AffineTerm,
    BoxSet,
    Claim,
    Coverage,
    DecisionRule,
    FiniteSet,
    FrozenController,
    LinearControl,
    OperatingLimitResult,
    OperatingLimitTrial,
    Parameter,
    RecoursePolicy,
    Scenario,
    verify_result,
)
from operange import linear
from operange.claim import rejected_result
from examples.operating_limits import example, run_example
from examples.measurement_uncertainty import example as measured
from tests.test_certificate_verification import altered
from tests.test_linear_process import small_model
from tests.test_frozen_controllers import single_rule


def simple(*, lower=0, upper=2, capacity=1, nominal=1):
    domain = BoxSet(
        (Parameter("load", "MW", nominal, lower, upper, 1, "Search range"),)
    )
    return small_model(bounds=(0, capacity)).as_claim(domain)


def verified(result):
    for compact in (False, True):
        restored = OperatingLimitResult.from_json(result.to_json(compact=compact))
        assert restored == result  # nosec B101
        report = verify_result(
            restored.to_dict(compact=compact),
            expected_contract_id=result.contract_ref.artifact_id,
        )
        assert report.verified, report.to_json()  # nosec B101


@pytest.mark.parametrize(
    "saved,expected", ((False, 9.25), (True, (32 / 1.625 - 2) / 2))
)
def test_production_bracket_retains_true_uncertainty_and_policy(saved, expected):
    source, controller = example()
    claim = source.with_controller(controller) if saved else source
    original = claim.contract
    result = claim.operating_limit("production")
    assert result.execution == "completed", result.to_json()  # nosec B101
    assert result.resolution == "limit_bracketed"  # nosec B101
    assert result.lower <= expected + 2e-8 <= result.upper + 2e-8  # nosec B101
    assert F(result.upper) - F(result.lower) <= F(1e-6)  # nosec B101
    assert result.unit == "t/h"  # nosec B101
    assert result.policy_scope == (
        "saved_controller" if saved else "adjustable_dispatch"
    )  # nosec B101
    assert claim.contract == original  # nosec B101
    assert result.limiting_witness.realizations[0]["auxiliary"] == 2  # nosec B101
    if saved:
        assert result.limiting_constraints == ("fuel_supply",)  # nosec B101
    for trial in result.trials:
        assert trial.audit.contract.model == original.model  # nosec B101
        assert trial.audit.contract.operating == original.operating  # nosec B101
    replay = result.as_claim()
    assert replay.domain.parameter("production").upper == result.lower  # nosec B101
    assert replay.audit_result().payload.verdict == "pass"  # nosec B101
    if saved:
        assert (
            FrozenController.from_json(replay.freeze().to_json())
            .audit_result()
            .payload.verdict
            == "pass"
        )  # nosec B101
    verified(result)


def test_checked_ceiling_is_not_a_global_maximum():
    result = simple(upper=0.75, nominal=0.5).operating_limit("load")
    assert result.resolution == "ceiling_verified"  # nosec B101
    assert result.lower == 0.75 and result.upper is None  # nosec B101
    assert result.limiting_witness is None and result.limiting_constraints == ()  # nosec B101
    verified(result)


def test_infeasible_floor_excludes_all_prefixes_not_isolated_higher_rates():
    source = simple()
    # Fixed power=1 meets an exact balance only at load=1. Higher isolated rates
    # can be feasible even though the range starting at zero never is.
    model = replace(
        source.adapter,
        operating_limits=(AffineRequirement("no_excess", "margin", 0, tolerance=0),),
    )
    permissions = RecoursePolicy(
        "fixed", (DecisionRule("power", "MW", "operation", fixed_value=1),)
    )
    claim = model.as_claim(source.domain, recourse=permissions)
    assert claim.evaluate_result({"load": 1}).payload.feasibility == "feasible"  # nosec B101
    result = claim.operating_limit("load")
    assert (
        result.execution == "completed" and result.resolution == "baseline_infeasible"
    )  # nosec B101
    assert result.lower is None and result.upper == 0  # nosec B101
    assert result.policy_scope == "fixed_commands"  # nosec B101
    with pytest.raises(ValueError, match="no guaranteed"):
        result.as_claim()
    verified(result)


def test_fixed_commands_keep_their_capacity_limit():
    claim = replace(
        simple(),
        recourse=RecoursePolicy(
            "fixed", (DecisionRule("power", "MW", "operation", fixed_value=0.75),)
        ),
    )
    result = claim.operating_limit("load")
    assert result.lower == 0.75 < result.upper < 0.750002  # nosec B101
    assert result.policy_scope == "fixed_commands"  # nosec B101
    verified(result)


def test_saved_measured_controller_keeps_the_bias_range():
    source = measured()
    domain = BoxSet(
        (
            replace(source.domain.parameter("load"), upper=12),
            source.domain.parameter("meter_bias"),
        )
    )
    rule = AffineController(
        "Meter plus reserve",
        (
            AffineControlRule(
                "steam", "MW", 0.25, (AffineTerm("measured_load", 1, "MW/MW"),)
            ),
        ),
        "Allow for underreading",
    )
    claim = source.adapter.as_claim(domain, controller=rule)
    result = claim.operating_limit("load")
    assert result.resolution == "limit_bracketed"  # nosec B101
    assert result.lower <= 10.5 <= result.upper  # nosec B101
    assert "control_upper:steam" in result.limiting_constraints  # nosec B101
    assert result.limiting_witness.realizations[0]["meter_bias"] == 0.25  # nosec B101
    verified(result)
    assert claim.operating_limit("meter_bias").execution == "invalid"  # nosec B101
    assert source.operating_limit("load").execution == "unsupported"  # nosec B101


@pytest.mark.parametrize("where", ("floor", "ceiling", "interior"))
def test_unresolved_audits_never_become_upper_bounds(monkeypatch, where):
    original = Claim.audit_result

    def audit(self, **options):
        top = self.domain.parameter("load").upper
        if top == {"floor": 0, "ceiling": 2, "interior": 1}[where]:
            return rejected_result(
                self.contract,
                "audit",
                {"query": "audit"},
                "Injected numerical uncertainty",
                execution="unresolved",
            )
        return original(self, **options)

    monkeypatch.setattr(Claim, "audit_result", audit)
    result = simple().operating_limit("load")
    assert result.execution == "unresolved" and result.resolution == "unresolved"  # nosec B101
    assert result.lower == (None if where == "floor" else 0)  # nosec B101
    assert result.upper == (2 if where == "interior" else None)  # nosec B101
    assert not any(
        t.outcome == "fail" for t in result.trials if t.audit.execution == "unresolved"
    )  # nosec B101
    if where != "floor":
        verified(result)
    else:
        assert verify_result(result).status == "unsupported"  # nosec B101


def test_iteration_budget_retains_evidence_and_explicit_tolerance():
    result = simple(capacity=0.8).operating_limit("load", max_iterations=1)
    assert len(result.trials) == 3 and result.execution == "unresolved"  # nosec B101
    assert result.lower == 0 and result.upper == 1  # nosec B101
    verified(result)


def test_constant_search_coordinate_and_adjacent_float_bracket():
    result = simple(lower=0.5, upper=0.5, nominal=0.5).operating_limit("load")
    assert result.resolution == "ceiling_verified" and len(result.trials) == 1  # nosec B101
    verified(result)
    result = (
        simple(lower=1, upper=nextafter(1, inf))
        .with_controller(single_rule())
        .operating_limit("load", tolerance=1e-20)
    )
    assert result.execution == "unresolved"  # nosec B101
    assert result.lower == 1 and result.upper == nextafter(1, inf)  # nosec B101
    verified(result)


@pytest.mark.parametrize(
    "parameter,options",
    (
        ("missing", {}),
        (None, {}),
        ("load", {"tolerance": 0}),
        ("load", {"tolerance": True}),
        ("load", {"tolerance": float("nan")}),
        ("load", {"max_iterations": -1}),
        ("load", {"max_iterations": True}),
        ("load", {"max_vertices": 0}),
        ("load", {"backend": "unknown"}),
    ),
)
def test_invalid_requests_are_explicit(parameter, options):
    result = simple().operating_limit(parameter, **options)
    assert result.execution == "invalid" and not result.trials  # nosec B101
    assert OperatingLimitResult.from_json(result.to_json()) == result  # nosec B101


def test_unsupported_domain_partial_observations_and_vertex_budget():
    source = simple()
    finite = replace(
        source,
        domain=FiniteSet(source.domain.space, (Scenario("one", {"load": 1}, "Test"),)),
    )
    assert finite.operating_limit("load").execution == "unsupported"  # nosec B101
    partial = replace(
        source,
        recourse=RecoursePolicy(
            "static", (DecisionRule("power", "MW", "operation", ()),)
        ),
    )
    assert partial.operating_limit("load").execution == "unsupported"  # nosec B101
    rejected = source.operating_limit("load", max_vertices=1)
    assert rejected.execution == "unsupported" and not rejected.trials  # nosec B101
    # A saved controller can use support bounds without enumerating generators.
    assert (
        source.with_controller(single_rule())
        .operating_limit("load", max_vertices=1)
        .resolution
        == "limit_bracketed"
    )  # nosec B101


def test_no_solver_is_used_for_saved_controller_or_certificate_verification(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("No numerical solver or production query may run")

    monkeypatch.setattr(linear, "solve_lp", forbidden)
    monkeypatch.setattr(linear, "linprog", forbidden)
    result = simple().with_controller(single_rule()).operating_limit("load")
    monkeypatch.setattr(Claim, "audit_result", forbidden)
    monkeypatch.setattr(AffineController, "commands", forbidden)
    verified(result)


def test_optional_backend_and_missing_dependency(monkeypatch):
    from operange import _cvxpy_backend

    def unavailable():
        raise _cvxpy_backend.BackendUnavailable("Injected missing dependency")

    monkeypatch.setattr(_cvxpy_backend, "load_cvxpy", unavailable)
    result = simple().operating_limit("load", backend="cvxpy")
    assert result.execution == "unsupported" and not result.trials  # nosec B101


def test_cvxpy_agrees_on_adjustable_operating_limit():
    pytest.importorskip("cvxpy")
    result = simple().operating_limit("load", backend="cvxpy")
    assert result.resolution == "limit_bracketed" and result.lower == 1  # nosec B101
    verified(result)


def test_forged_ranges_and_physical_proofs_fail_even_with_fresh_hashes():
    source = simple().with_controller(single_rule())
    result = source.operating_limit("load", max_iterations=1)
    # A true passing audit for a smaller domain cannot support a larger endpoint.
    trial = result.achievable_trial
    forged = replace(
        result,
        trials=tuple(
            replace(t, endpoint=1.5) if t is trial else t for t in result.trials
        ),
    )
    assert OperatingLimitResult.from_json(forged.to_json()) == forged  # nosec B101
    assert verify_result(forged).status == "failed"  # nosec B101
    # A changed equipment limit cannot silently enter the achievable certificate.
    better = replace(
        source.adapter.model, controls=(LinearControl("power", "MW", 0, 3),)
    ).as_claim(source.domain, controller=single_rule())
    forged = replace(
        result,
        trials=(OperatingLimitTrial(2, better.audit_result()),),
        execution="completed",
    )
    assert verify_result(forged).status == "failed"  # nosec B101
    # Fresh digest plus altered interval arithmetic is still not a proof.
    altered_audit = altered(
        trial.audit, "enclosure", lambda d: d["command_bounds"]["power"].update(upper=0)
    )
    forged = replace(
        result,
        trials=tuple(
            replace(t, audit=altered_audit) if t is trial else t for t in result.trials
        ),
    )
    assert verify_result(forged).status == "failed"  # nosec B101


def test_loader_checks_summary_hash_and_verifier_pins_source_contract():
    result = simple().operating_limit("load", max_iterations=0)
    data = result.to_dict()
    data["upper"] = 0.5
    with pytest.raises(ValueError, match="identity or derived"):
        OperatingLimitResult.from_json(json.dumps(data))
    assert (
        verify_result(result, expected_contract_id="another-contract").status
        == "failed"
    )  # nosec B101
    assert verify_result(result, max_checks=2).status == "unsupported"  # nosec B101
    # Malformed nested declarations must return invalid, never escape the loader.
    for field in ("tolerance", "parameter"):
        malformed = result.to_dict()
        del malformed["request"][field]
        assert verify_result(malformed).status == "invalid"  # nosec B101


def test_consumer_example_exports_checked_brackets():
    exported = run_example()
    for name, data in exported.items():
        result = OperatingLimitResult.from_json(json.dumps(data["search"]))
        assert result.resolution == "limit_bracketed" and result.policy_scope == name  # nosec B101
        assert data["verification"]["verified"]  # nosec B101


def test_partial_audit_with_verified_failure_still_supplies_upper_bound(monkeypatch):
    original = Claim.audit_result

    def partial(self, **options):
        audit = original(self, **options)
        if audit.payload.verdict == "fail":
            audit = replace(
                audit,
                execution="unresolved",
                payload=replace(audit.payload, coverage=Coverage("partial")),
                evidence=tuple(
                    e
                    for e in audit.evidence
                    if e.evidence_id not in ("enclosure", "coverage")
                    and not e.evidence_id.startswith("bound:")
                ),
            )
        return audit

    monkeypatch.setattr(Claim, "audit_result", partial)
    result = simple().with_controller(single_rule()).operating_limit("load")
    assert result.resolution == "limit_bracketed"  # nosec B101
    assert result.excluded_trial.audit.execution == "unresolved"  # nosec B101
    assert result.upper is not None  # nosec B101
    verified(result)


@pytest.mark.parametrize("field,value", (("trials", None), ("diagnostics", 7)))
def test_malformed_report_arrays_are_invalid(field, value):
    data = simple().operating_limit("load", max_iterations=0).to_dict()
    data[field] = value
    with pytest.raises(ValueError):
        OperatingLimitResult.from_json(json.dumps(data))
    assert verify_result(data).status == "invalid"  # nosec B101


def test_verification_cli_accepts_operating_limit_report(tmp_path):
    result = simple().operating_limit("load", max_iterations=0)
    path = tmp_path / "limit.json"
    path.write_text(result.to_json(compact=True))
    report = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "operange.verify",
            str(path),
            "--expected-contract-id",
            result.contract_ref.artifact_id,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert report.returncode == 0, report.stderr + report.stdout  # nosec B101
    assert json.loads(report.stdout)["verified"]  # nosec B101
