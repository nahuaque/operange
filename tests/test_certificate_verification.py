"""Offline verification must reject forged mathematics even with fresh hashes."""

from dataclasses import replace
from fractions import Fraction as F
import json
# CLI tests use a fixed executable without a shell.
import subprocess  # nosec B404
import sys

import pytest

from operange import (
    AffineControlRule,
    AffineController,
    AffineTerm,
    BoxSet,
    ControlTarget,
    ControlTrackingObjective,
    ConvexHullSet,
    DecisionRule,
    Evidence,
    LinearObjective,
    Parameter,
    RecoursePolicy,
    result_from_json,
    verify_result,
)
from examples.controller_synthesis import example as boiler
from examples.linear_dispatch import example as overloaded
from tests.test_shared_relief import example as shared
from tests.test_linear_process import small_model


def altered(result, name, change):
    proofs = []
    for item in result.evidence:
        if item.evidence_id == name:
            data = item.to_dict()["details"]
            change(data)
            item = replace(item, details=data)
        proofs.append(item)
    result = replace(result, evidence=tuple(proofs))
    # The ordinary loader accepts the fresh identities and structural checks.
    return result_from_json(result.to_json())


def expect_verified(result):
    for compact in (False, True):
        report = verify_result(
            result.to_json(compact=compact),
            expected_contract_id=result.contract_ref.artifact_id,
        )
        assert report.verified, report.to_json()  # nosec B101
        assert report.artifact_ref == result.ref  # nosec B101
        assert json.loads(report.to_json())["status"] == "verified"  # nosec B101


def failure():
    model, domain = overloaded()
    return model.as_claim(domain).evaluate_result(
        {"dryer": 12, "evaporator": 8},
        diagnose=True,
        relief={"constraint": "shared_fuel", "maximum": 2},
    )


@pytest.mark.parametrize("backend", ("scipy", "cvxpy"))
@pytest.mark.parametrize("sense", ("minimize", "maximize"))
def test_dispatch_objective_and_audit_round_trips(backend, sense):
    if backend == "cvxpy":
        pytest.importorskip("cvxpy")
    claim = boiler()
    claim = replace(
        claim, adapter=replace(claim.adapter, objective=LinearObjective("fuel", sense))
    )
    expect_verified(claim.evaluate_result(claim.domain.nominal, backend=backend))
    expect_verified(claim.audit_result(backend=backend))


@pytest.mark.parametrize("kind", ("finite", "box", "hull"))
@pytest.mark.parametrize("objective", ("linear", "quadratic"))
def test_shared_relief_proofs_and_changed_audit_are_verified(kind, objective):
    if objective == "quadratic":
        pytest.importorskip("cvxpy")
    model, domain, options = shared()
    if kind == "hull":
        domain = ConvexHullSet(domain.space, domain.scenarios)
    elif kind == "box":
        domain = BoxSet(
            (
                Parameter("a", "MW", 0, 0, 2, 1, "Test"),
                Parameter("b", "MW", 0, 0, 3, 1, "Test"),
            )
        )
    result = model.as_claim(domain).audit_result(
        backend="cvxpy" if objective == "quadratic" else "scipy",
        relief={**options, "objective": objective},
    )
    expect_verified(result)
    assert any("/reaudit/" in c.path for c in verify_result(result).checks)  # nosec B101


def test_single_relief_and_irreducible_conflict_and_unreachable_plan():
    expect_verified(failure())
    model, domain = overloaded()
    result = model.as_claim(domain).evaluate_result(
        {"dryer": 12, "evaporator": 8},
        relief={"constraint": "shared_fuel", "maximum": 0.1},
    )
    expect_verified(result)
    assert (
        next(e.details for e in result.evidence if e.evidence_id == "relief")[
            "resolution"
        ]
        == "unreachable"
    )  # nosec B101


@pytest.mark.parametrize("objective", ("linear", "quadratic"))
def test_point_joint_relief_and_nonnegative_zero_plan(objective):
    if objective == "quadratic":
        pytest.importorskip("cvxpy")
    model, domain, options = shared()
    options = {**options, "objective": objective}
    for point in ({"a": 2, "b": 0}, {"a": 0, "b": 3}):
        expect_verified(
            model.as_claim(domain).evaluate_result(
                point,
                relief=options,
                backend="cvxpy" if objective == "quadratic" else "scipy",
            )
        )
    enlarged = replace(
        model,
        operating_limits=tuple(replace(r, limit=3) for r in model.operating_limits),
    )
    expect_verified(
        enlarged.as_claim(domain).audit_result(
            relief=options, backend="cvxpy" if objective == "quadratic" else "scipy"
        )
    )


def test_control_tracking_bound_and_fixed_control_contradiction():
    pytest.importorskip("cvxpy")
    claim = boiler()
    model = replace(
        claim.adapter,
        objective=ControlTrackingObjective(
            (
                ControlTarget("boiler_a", 8, 1, "MW"),
                ControlTarget("boiler_b", 8, 1, "MW"),
            )
        ),
    )
    expect_verified(
        model.as_claim(claim.domain).evaluate_result(
            claim.domain.nominal, backend="cvxpy"
        )
    )
    fixed = RecoursePolicy(
        "fixed",
        tuple(
            DecisionRule(c.name, c.unit, "operation", fixed_value=100)
            for c in model.controls
        ),
    )
    expect_verified(
        model.as_claim(claim.domain, recourse=fixed).evaluate_result(
            claim.domain.nominal, backend="cvxpy"
        )
    )


@pytest.mark.parametrize("kind", ("box", "hull", "finite"))
def test_controller_replay_and_continuous_rounding_certificates(kind):
    pytest.importorskip("cvxpy")
    claim = boiler()
    if kind != "box":
        from operange import FiniteSet, Scenario

        cases = tuple(
            Scenario(str(i), p, "Test")
            for i, p in enumerate(
                (dict(dryer=8, evaporator=4), dict(dryer=12, evaporator=8))
            )
        )
        domain = (
            FiniteSet(claim.domain.space, cases)
            if kind == "finite"
            else ConvexHullSet(claim.domain.space, cases)
        )
        claim = replace(claim, domain=domain)
    synthesized = claim.synthesize_controller()
    expect_verified(synthesized.audit)
    expect_verified(
        synthesized.claim.evaluate_result(
            synthesized.audit.supporting_evaluations[0].request["realization"]
        )
    )
    bad = replace(
        synthesized.controller,
        rules=tuple(
            replace(r, offset=r.offset + 10) for r in synthesized.controller.rules
        ),
    )
    expect_verified(claim.with_controller(bad).audit_result())


def test_no_optimizer_model_query_or_producer_certificate_code_is_called(monkeypatch):
    pytest.importorskip("cvxpy")
    from operange import (
        Claim,
        linear,
        _cvxpy_backend,
        _linear_recourse,
        _dispatch_objective,
        _controller_envelope,
        _joint_relief,
    )

    claim = boiler()
    model, domain, options = shared()
    results = [
        failure(),
        claim.audit_result(),
        claim.synthesize_controller().audit,
        model.as_claim(domain).audit_result(relief=options),
    ]

    def forbidden(*a, **k):
        raise AssertionError("verification called producer/solver code")

    for owner, names in (
        (Claim, ("evaluate_result", "audit_result")),
        (linear, ("solve_lp",)),
        (_cvxpy_backend, ("load_cvxpy",)),
        (_linear_recourse, ("compile_system", "solve_system")),
        (
            _linear_recourse.LinearSystem,
            ("certificate", "objective_bound", "checked_values"),
        ),
        (_dispatch_objective.Polynomial, ("bound",)),
        (_controller_envelope, ("ControllerEnvelope",)),
        (_joint_relief, ("compile_relief", "candidate_changes")),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    for result in results:
        expect_verified(result)


@pytest.mark.parametrize(
    "field,value",
    (
        ("multipliers", [-1]),
        ("rows_exact", []),
        ("upper_exact", []),
        ("weighted_coefficients_exact", ["0", "0"]),
        ("contradiction_gap_exact", "0"),
        ("fixed_controls", {"boiler_a": 12}),
        ("control_coordinates", []),
        ("box_minimum_exact", "1e999999"),
        ("weighted_upper_exact", "nan"),
    ),
)
def test_bad_infeasibility_arithmetic_fails_despite_fresh_hashes(field, value):
    forged = altered(failure(), "recourse", lambda d: d.update({field: value}))
    report = verify_result(forged)
    assert report.status == "failed" and not report.verified, report.to_json()  # nosec B101


def test_internally_valid_contradiction_for_wrong_rows_is_rejected():
    def wrong_problem(d):
        d["upper_exact"] = [str(F(v) - 100) for v in d["upper_exact"]]
        weighted = [
            sum(F(a) * F(b) for a, b in zip(column, d["multipliers"]))
            for column in zip(*d["rows_exact"])
        ]
        minimum = sum(min(0, a) for a in weighted)
        upper = sum(F(a) * F(b) for a, b in zip(d["upper_exact"], d["multipliers"]))
        d.update(
            weighted_coefficients_exact=list(map(str, weighted)),
            box_minimum_exact=str(minimum),
            weighted_upper_exact=str(upper),
            contradiction_gap_exact=str(minimum - upper),
        )

    report = verify_result(altered(failure(), "recourse", wrong_problem))
    assert report.status == "failed"  # nosec B101
    assert any("declared problem" in c.message for c in report.checks)  # nosec B101


@pytest.mark.parametrize(
    "field,value",
    (
        ("constant_exact", "-100"),
        ("linear_exact", ["0", "0"]),
        ("diagonal_exact", ["1", "1"]),
        ("lower_exact", "100000"),
        ("row_constraint_refs", []),
    ),
)
def test_objective_problem_and_bounds_are_reconstructed(field, value):
    claim = boiler()
    result = claim.evaluate_result(claim.domain.nominal)
    assert (
        verify_result(
            altered(result, "operating_objective", lambda d: d.update({field: value}))
        ).status
        == "failed"
    )  # nosec B101


def test_objective_dual_slopes_and_gap_are_recomputed():
    claim = boiler()
    result = claim.evaluate_result(claim.domain.nominal)
    for field in ("lagrangian_slopes_exact", "box_minimizers_exact", "multipliers"):
        forged = altered(
            result,
            "operating_objective",
            lambda d: d["certificate"].update({field: []}),
        )
        assert verify_result(forged).status == "failed"  # nosec B101


def test_physical_values_and_included_constraints_cannot_be_forged():
    claim = boiler()
    result = claim.evaluate_result(claim.domain.nominal)
    for name in ("fuel", "boiler_a"):
        values = tuple(
            replace(v, value=v.value + 1) if v.quantity_ref == name else v
            for v in result.payload.values
        )
        # Remove the optional objective to keep the altered quantity structurally valid.
        forged = replace(
            result,
            payload=replace(result.payload, values=values, objective=None),
            evidence=tuple(
                e for e in result.evidence if e.evidence_id != "operating_objective"
            ),
        )
        report = verify_result(forged)
        assert report.status == "failed"  # nosec B101
    forged = replace(
        result,
        payload=replace(
            result.payload,
            included_constraints=result.payload.included_constraints[:-1],
        ),
    )
    assert verify_result(forged).status == "failed"  # nosec B101


def test_coverage_cannot_omit_a_generator_or_change_its_identity():
    result = boiler().audit_result()
    for kind in ("omit", "rename", "point"):
        data = result.payload.coverage.to_dict()
        refs = data["evaluated_support"]["scenarios"]
        if kind == "omit":
            refs.pop()
        elif kind == "rename":
            refs[0]["name"] = "substitute"
        else:
            refs[0]["evaluation_ref"] = refs[1]["evaluation_ref"]
        forged = replace(result, payload=replace(result.payload, coverage=data))
        assert verify_result(forged).status == "failed"  # nosec B101


def test_conflict_irreducibility_requires_all_removal_witnesses():
    for change in (
        lambda d: d.update(removal_feasibility={}),
        lambda d: d.update(minimum_cardinality=True),
        lambda d: d["removal_feasibility"][next(iter(d["removal_feasibility"]))][
            "controls"
        ].update(boiler_a=100),
    ):
        assert verify_result(altered(failure(), "conflict", change)).status == "failed"  # nosec B101


@pytest.mark.parametrize(
    "change",
    (
        lambda d: d["rows_exact"].pop(),
        lambda d: d["dispatch_blocks"].pop(),
        lambda d: d["candidate"]["dispatches"].pop(),
        lambda d: d["candidate"]["changes"][0].update(changed_limit=100),
        lambda d: d["candidate"]["dispatches"][0]["controls"].update(power_a=0),
        lambda d: d["candidate"].update(objective_exact="0"),
        lambda d: d.update(lower=100),
        lambda d: d["declaration"]["changes"][0].update(unit="kg"),
    ),
)
def test_shared_relief_cannot_change_problem_plan_or_objective(change):
    model, domain, options = shared()
    result = model.as_claim(domain).audit_result(relief=options)
    assert verify_result(altered(result, "relief", change)).status == "failed"  # nosec B101


def test_nested_changed_audit_is_reverified_even_when_all_hashes_are_updated():
    model, domain, options = shared()
    result = model.as_claim(domain).audit_result(relief=options)

    def tamper(d):
        nested = result_from_json(json.dumps(d["candidate"]["reaudit"]))
        coverage = nested.payload.coverage.to_dict()
        coverage["evaluated_support"]["scenarios"].pop()
        nested = replace(nested, payload=replace(nested.payload, coverage=coverage))
        d["candidate"].update(
            reaudit=nested.to_dict(compact=True), reaudit_ref=nested.ref.to_dict()
        )

    report = verify_result(altered(result, "relief", tamper))
    assert report.status == "failed"  # nosec B101
    assert any("/reaudit/" in c.path and c.status == "failed" for c in report.checks)  # nosec B101


def test_controller_rounding_error_cannot_be_erased_to_claim_pass():
    model = small_model(bounds=(0, 2**54))
    model = replace(model, outputs=(replace(model.outputs[0], offset=-(2**53)),))
    domain = BoxSet((Parameter("load", "MW", 0, 0, 2, 1, "Test"),))
    controller = AffineController(
        "Rounded",
        (AffineControlRule("power", "MW", 2**53, (AffineTerm("load", 1, "MW/MW"),)),),
        "Test",
    )
    result = model.as_claim(domain, controller=controller).audit_result()
    assert result.payload.verdict == "inconclusive"  # nosec B101
    forged = replace(
        result, execution="completed", payload=replace(result.payload, verdict="pass")
    )
    assert verify_result(forged).status == "failed"  # nosec B101
    forged = altered(
        forged, "bound:service", lambda d: d.update(command_rounding_error_exact="0")
    )
    assert verify_result(forged).status == "failed"  # nosec B101


def test_hash_integrity_and_pinned_contract_are_separate_from_math():
    claim = boiler()
    result = claim.evaluate_result(claim.domain.nominal)
    raw = result.to_dict()
    raw["result_id"] = "sha256:" + "0" * 64
    assert verify_result(raw).status == "invalid"  # nosec B101
    changed = replace(
        claim, adapter=replace(claim.adapter, name="Different declaration")
    )
    changed_result = changed.evaluate_result(claim.domain.nominal)
    assert verify_result(changed_result).verified  # nosec B101
    assert (
        verify_result(
            changed_result, expected_contract_id=result.contract_ref.artifact_id
        ).status
        == "failed"
    )  # nosec B101


def test_solver_status_is_ignored_and_unknown_evidence_is_explicit():
    result = failure()
    logged = altered(
        result,
        "solver",
        lambda d: d.update(attempts=[{"status": "infeasible", "message": "Not proof"}]),
    )
    assert verify_result(logged).verified  # nosec B101
    unknown = replace(
        result,
        evidence=result.evidence
        + (Evidence("future", "new_theorem", "unimplemented", "verified"),),
    )
    assert verify_result(unknown).status == "unsupported"  # nosec B101


def test_unsupported_result_family_and_hull_interior_are_explicit():
    from examples.steam_header import example

    model, domain = example()
    assert verify_result(model.as_claim(domain).audit_result()).status == "unsupported"  # nosec B101
    claim = boiler()
    from operange import Scenario

    hull = ConvexHullSet(
        claim.domain.space,
        (
            Scenario("lo", {"dryer": 8, "evaporator": 4}, "Test"),
            Scenario("hi", {"dryer": 12, "evaporator": 8}, "Test"),
        ),
    )
    result = replace(claim, domain=hull).evaluate_result(claim.domain.nominal)
    assert verify_result(result).status == "unsupported"  # nosec B101


@pytest.mark.parametrize("document", ("{}", "{", '{"a":1,"a":2}', b"\xff", [], "NaN"))
def test_malformed_inputs_return_invalid(document):
    assert verify_result(document).status == "invalid"  # nosec B101


def test_resource_limits_do_not_return_partial_success():
    result = boiler().audit_result()
    assert verify_result(result, max_bytes=1).status == "invalid"  # nosec B101
    assert verify_result(result, max_checks=2).status == "unsupported"  # nosec B101
    assert verify_result(result, max_checks=True).status == "invalid"  # nosec B101


def test_membership_proof_is_bound_to_the_same_realization_and_domain():
    result = failure()
    for change in (
        lambda d: d["check"]["realization"].update(dryer=10),
        lambda d: d["check"]["domain_ref"].update(artifact_id="sha256:wrong"),
    ):
        assert verify_result(altered(result, "membership", change)).status == "failed"  # nosec B101


@pytest.mark.parametrize("value", (None, [], "not a proof"))
def test_malformed_nested_proofs_return_diagnostics(value):
    result = boiler().evaluate_result(boiler().domain.nominal)
    report = verify_result(
        altered(result, "operating_objective", lambda d: d.update(certificate=value))
    )
    assert report.status == "failed"  # nosec B101


def test_verification_consumer_example():
    from examples.certificate_verification import run_example

    exported = run_example()
    assert exported["verification"]["verified"]  # nosec B101
    assert exported["tampered_verification"]["status"] == "failed"  # nosec B101


def test_cli_json_and_exit_codes(tmp_path):
    source = tmp_path / "result.json"
    result = failure()
    for data, code, status in (
        (result.to_json(compact=True), 0, "verified"),
        (
            altered(
                result, "recourse", lambda d: d.update(contradiction_gap_exact="0")
            ).to_json(),
            1,
            "failed",
        ),
        ("{}", 2, "invalid"),
    ):
        source.write_text(data)
        command = subprocess.run(
            [sys.executable, "-m", "operange.verify", str(source)],
            capture_output=True,
            text=True,
            check=False,
        )  # nosec B603
        assert command.returncode == code, command.stderr  # nosec B101
        assert json.loads(command.stdout)["status"] == status  # nosec B101
